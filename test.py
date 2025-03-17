import os
import sys
import torch
import numpy as np
from time import time
from tqdm import tqdm

from pi_seg.inference.predictors import get_predictor
from pi_seg.utils.val_utils import evaluate, compute_noc_metric
from pi_seg.utils.log import Logger

from model import initialize_model
from dataset import valset as dataset

from config import device, NUM_MAX_POINTS, TEST_SAVE_DIR, TEST_MODEL_PATH, DATASET_NAME, DATASET_DIR


def test(model_name="deepPI", max_iou_thr=1):
    """Load the trained model and evaluate its performance on the dataset."""
    # Ensure save directory exists
    os.makedirs(TEST_SAVE_DIR, exist_ok=True)

    # Redirect output to log file
    log_file_path = os.path.join(TEST_SAVE_DIR, "test_results.log")
    sys.stdout = Logger(log_file_path)  # Redirect print() to both console and file

    print(f"Test results will be saved in {log_file_path}\n")
    
    # Load model
    model = initialize_model()
    
    checkpoint = torch.load(TEST_MODEL_PATH, map_location=device)    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {TEST_MODEL_PATH}")
    print(f"Testing on Dataset: {DATASET_NAME}")
    print(f"Dataset Path: {DATASET_DIR}")

    # Initialize predictor
    predictor = get_predictor(model, brs_mode='deepPI', device=device, prob_thresh=0.5)

    # Evaluation
    all_ious, all_ious_b, pred_mask_li, uncertainty_li, clicks_li = [], [], [], [], []
    start_time = time()

    with torch.no_grad():
        for index in tqdm(range(len(dataset)), desc="Testing"):
            sample = dataset.get_sample(index)
            sample = dataset.augment_sample(sample)
            
            eval_result = evaluate(
                predictor, sample, sample_cnt=16, index=index, max_clicks=NUM_MAX_POINTS, 
                max_iou_thr=max_iou_thr, vis=False, uncertainty=True, save_dir=TEST_SAVE_DIR
            )

            sample_ious, sample_iou_b, prob_mask, pred_mask, uncertainty_map, clicks, ps = eval_result
            
            all_ious.append(sample_ious)
            all_ious_b.append(sample_iou_b)
            uncertainty_li.append(uncertainty_map)
            pred_mask_li.append(pred_mask)
            clicks_li.append(clicks)
            
    end_time = time()
    elapsed_time = end_time - start_time

    # Save results
    os.makedirs(TEST_SAVE_DIR, exist_ok=True)

    iou_thrs = np.arange(0.85, min(0.96, max_iou_thr), 0.01)
    noc_list, over_max_list = compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=NUM_MAX_POINTS)

    print(f"IoU Thresholds: {list(iou_thrs)}")
    print(f"NOC List: {noc_list}")
    print(f"Over Max List: {over_max_list}")

    min_num_clicks = min(len(x) for x in all_ious)
    mean_ious = np.array([x[:min_num_clicks] for x in all_ious]).mean(axis=0)
    miou_str = ' '.join([f'mIoU@{click_id}={mean_ious[click_id - 1]:.2%};'
                         for click_id in range(1, 21) if click_id <= min_num_clicks])

    print("Mean IoU List:", list(mean_ious))

    sys.stdout = sys.__stdout__
    print(f"Testing completed. Results saved in {TEST_SAVE_DIR}.")

if __name__ == "__main__":
    test()