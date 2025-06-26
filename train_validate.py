import os
import numpy as np
import torch
import random

from tqdm import tqdm

from pi_seg.model.modeling.probabilistic_unet.utils import l2_regularisation
from pi_seg.utils.train_utils import *

from config import (
    device, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, VALIDATION_STEP, SAMPLE_CNT,
    NUM_MAX_POINTS, SAVE_DIR, PRE_TRAINED, PRE_TRAINED_PATH
)


def train_and_validate(model, train_loader, val_loader, optimizer,
                       start_epoch=1, best_iou=0.0, best_epoch=0,
                       train_loss_history=None, val_metrics_history=None):
    """
    Main function to train and validate the model over multiple epochs.
    It handles checkpointing for the best and latest models.
    """
    # Ensure the directory for saving models exists.
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Initialize history lists if they are not provided (i.e., for a fresh training run).
    if train_loss_history is None:
        train_loss_history = []
    if val_metrics_history is None:
        val_metrics_history = []
    
    # Main training loop, starting from 'start_epoch' (which is > 1 if resuming).
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss = []
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']
            
            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            num_points = random.randint(0, NUM_MAX_POINTS)

            # Simulate interactive clicks without gradient calculation to prepare model input.
            with torch.no_grad():
                model.eval()
                for click_idx in range(num_points):
                    net_input = torch.cat((image, prev_output), dim=1) if model.module.with_prev_mask else image
                    outputs = model.forward(image=net_input, mask=gt_mask, points=points, training=False, sample_cnt=SAMPLE_CNT, un_weight=False)

                    ps = outputs['samples']
                    ps = [torch.sigmoid(s) for s in ps]
                    prev_output = torch.stack(ps).mean(dim=0)
                    uncertainty_map = torch.stack(ps).std(dim=0, unbiased=False)

                    points = get_next_points_removeall_realworld(gt_mask, points, click_idx + 1, uncertainty_map=uncertainty_map, top_uncertainty=True)

            # --- Actual Training Step ---
            # Perform a forward and backward pass with gradient tracking.
            model.train()
            net_input = torch.cat((image, prev_output), dim=1) if model.module.with_prev_mask else image
            output = model.forward(image=net_input, mask=gt_mask, points=points, training=True, sample_cnt=SAMPLE_CNT, un_weight=False)

            # Calculate the total loss (ELBO + regularization).
            reg_loss = l2_regularisation(model.module.feature_extractor.posterior) + \
                       l2_regularisation(model.module.feature_extractor.prior) + \
                       l2_regularisation(model.module.feature_extractor.fcomb.layers)
            elbo = output['loss'].mean()
            loss = elbo + 1e-3 * reg_loss

            # Backpropagation.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # --- End of Epoch Processing ---
        # Record the average training loss for this epoch.
        avg_train_loss = np.mean(train_loss)
        train_loss_history.append(avg_train_loss)

        # Run validation at the specified interval.
        if epoch % VALIDATION_STEP == 0:
            val_metrics = validate_model(model, val_loader)
            val_metrics['epoch'] = epoch
            val_metrics_history.append(val_metrics)
            
            val_iou = val_metrics["iou"]
            
            # Save a checkpoint if the current model has the best IoU so far.
            if val_iou > best_iou:
                best_iou = val_iou
                best_epoch = epoch
                best_model_path = os.path.join(SAVE_DIR, "model_best_iou.pth")
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": val_metrics,
                }, best_model_path)
                print(f"Saved best IoU model at {best_model_path} (IoU: {best_iou:.4f})")

        # Save the latest model state at the end of every epoch.
        latest_model_path = os.path.join(SAVE_DIR, "model_latest.pth")
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss_history": train_loss_history,
            "val_metrics_history": val_metrics_history
        }, latest_model_path)
        
        # Print a summary of the epoch's performance.
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Best Model: Epoch {best_epoch} (IoU: {best_iou:.4f})")

    print(f"Training finished. Latest model saved at {latest_model_path}")


def validate_model(model, val_loader):
    """
    Evaluates the model on the validation dataset and returns performance metrics.
    """
    # Initialize lists to store metrics for each sample.
    accuracy_lst, iou_lst, recall_lst, precision_lst, dsc_lst = [], [], [], [], []

    # Set model to evaluation mode and disable gradients.
    with torch.no_grad():
        model.eval()
        
        for batch_data in tqdm(val_loader, desc="Validation"):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']

            # Simulate interactive clicks to get the final prediction.
            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            for click_idx in range(NUM_MAX_POINTS):
                net_input = torch.cat((image, prev_output), dim=1) if model.module.with_prev_mask else image
                outputs = model.forward(image=net_input, mask=gt_mask, points=points, training=False, sample_cnt=SAMPLE_CNT, un_weight=False)

                ps = outputs['samples']
                ps = [torch.sigmoid(s) for s in ps]
                prev_output = torch.stack(ps).mean(dim=0)
                uncertainty_map = torch.stack(ps).std(dim=0, unbiased=False)

                points = get_next_points_removeall_realworld(gt_mask, points, click_idx + 1, uncertainty_map=uncertainty_map, top_uncertainty=True)

            # Get the final model output and calculate validation loss.
            net_input = torch.cat((image, prev_output), dim=1) if model.module.with_prev_mask else image
            output = model(image=net_input, mask=gt_mask, points=points, training=False, sample_cnt=SAMPLE_CNT, un_weight=False)
            
            # Calculate performance metrics for each item in the batch.
            output_mean = torch.stack(ps).mean(dim=0)
            for res, tar in zip(output_mean, gt_mask):
                np_restored = fn_tonumpy(res)
                np_restored = fn_threshold(np_restored)

                np_target = fn_tonumpy(tar)
                np_target = fn_threshold(np_target)

                performance_dict = fn_performance(np_restored, np_target)
                accuracy_lst.append(performance_dict['accuracy'])
                iou_lst.append(performance_dict['iou_f'])
                recall_lst.append(performance_dict['recall'])
                precision_lst.append(performance_dict['precision'])
                dsc_lst.append(performance_dict['DSC'])

    # Aggregate the metrics by taking the mean across all validation samples.
    metrics = {
        "accuracy": np.mean(accuracy_lst),
        "iou": np.mean(iou_lst),
        "recall": np.mean(recall_lst),
        "precision": np.mean(precision_lst),
        "dsc": np.mean(dsc_lst),
    }
    
    # Print a summary of the validation results.
    print(f"IoU: {metrics['iou']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, DSC: {metrics['dsc']:.4f}")

    return metrics


