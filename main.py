import torch
import os
from train_validate import train_and_validate
from dataset import train_dataloader, val_dataloader
from model import initialize_model
from config import device, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, RESUME_TRAINING, SAVE_DIR


# --- 1. Initialization ---
# Initialize the model and optimizer based on the configurations.
model = initialize_model()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# --- 2. Training State Setup ---
# Initialize variables to track the training progress.
# These values will be used for a fresh training run, but will be overwritten
# if training is resumed from a checkpoint.
start_epoch = 1
best_iou = 0.0
best_epoch = 0
train_loss_history = []
val_metrics_history = []


# --- 3. Resume Training Logic ---
# Check if the RESUME_TRAINING flag is set to True in the config file.
if RESUME_TRAINING:
    # Define paths for the latest and best-performing model checkpoints.
    latest_checkpoint_path = os.path.join(SAVE_DIR, "model_latest.pth")
    best_checkpoint_path = os.path.join(SAVE_DIR, "model_best_iou.pth")
    
    # Check if a 'latest' checkpoint exists to resume from.
    if os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from the latest checkpoint: {latest_checkpoint_path}")
        
        # Load the latest checkpoint file.
        # It contains the most recent state of the model, optimizer, and history.
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        
        # Restore the states.
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        
        # Use .get() to safely retrieve history lists. This provides backward compatibility
        # with older checkpoints that might not have these keys.
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_metrics_history = checkpoint.get('val_metrics_history', [])

        print(f"Model, optimizer, and history states loaded. Starting from epoch {start_epoch}.")

        # Separately, load the 'best' checkpoint to restore the record of the best performance.
        # This ensures we don't lose track of the actual best model from the previous run.
        if os.path.exists(best_checkpoint_path):
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
            best_iou = best_checkpoint['metrics']['iou']
            best_epoch = best_checkpoint['epoch']
            print(f"Restored best IoU from previous run: {best_iou:.4f} at epoch {best_epoch}")
        else:
            # If only a 'latest' checkpoint exists but not a 'best' one.
            print("Best checkpoint not found. 'best_iou' will start from 0.")
            
    else:
        # If no checkpoint is found, start a fresh training session.
        print("No checkpoint found. Starting training from scratch.")


# --- 4. Start Training ---
# Call the main training and validation function.
# Pass all the state variables, which are either newly initialized
# or have been loaded from the checkpoint.
train_and_validate(
    model=model, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader, 
    optimizer=optimizer,
    start_epoch=start_epoch,
    best_iou=best_iou,
    best_epoch=best_epoch,
    train_loss_history=train_loss_history,
    val_metrics_history=val_metrics_history
)