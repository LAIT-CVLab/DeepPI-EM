import os
import pickle

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from time import time
from pathlib import Path
from datetime import timedelta

from pi_seg.inference.clicker.clicker_deepPI import Clicker as PI_Clicker
from pi_seg.inference.clicker.clicker import Clicker
from pi_seg.utils.exp_imports.default import *
from pi_seg.utils.vis import draw_probmap, draw_with_blend_and_clicks, add_tag


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int)

        score = scores_arr.mean()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        over_max_list.append(over_max)

    return noc_list, over_max_list

            
def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    def callback(image, gt_mask, pred_probs, sample_id, click_indx, clicks_list):
        sample_path = save_path / f'{sample_id}_{click_indx}.jpg'
        prob_map = draw_probmap(pred_probs)
        image_with_mask = draw_with_blend_and_clicks(image, pred_probs > prob_thresh, clicks_list=clicks_list)
        cv2.imwrite(str(sample_path), np.concatenate((image_with_mask, prob_map), axis=1)[:, :, ::-1])

    return callback


def check_divide_zero(value):
    if value > 1:
        return -1
    
    return value


def fn_performance(label_, output):
    FP = len(np.where(output - label_  == 1)[0])
    FN = len(np.where(output - label_  == -1)[0])
    TP = len(np.where(output + label_ ==2)[0])
    TN = len(np.where(output + label_ == 0)[0])
    
    accuracy = (TP + TN) / (TP + FN + FP + TN + 1e-6)
    DSC = 2*TP / (2*TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    jarcard = TP / (TP + FP + FN + 1e-6)
    jarcard_b = TN / (TN + FP + FN + 1e-6)
    iou_o = (jarcard + jarcard_b)/2
    
    accuracy = check_divide_zero(accuracy)
    DSC = check_divide_zero(DSC)
    precision = check_divide_zero(precision)
    recall = check_divide_zero(recall)
    jarcard = check_divide_zero(jarcard)
    jarcard_b = check_divide_zero(jarcard_b)
    iou_o = check_divide_zero(iou_o)
    
    result = {'accuracy': accuracy,'DSC': DSC, 'precision': precision, 'recall': recall, 'iou_f':jarcard, 'iou_b': jarcard_b, 'iou_o':iou_o}

    return result


def evaluate(predictor, sample, sample_cnt=16, index=0, min_clicks = 1, max_clicks = 30, pred_thr = 0.5, max_iou_thr = 1.0, vis = False, save_dir = './experiments', uncertainty=False):
    image, gt_mask, init_mask = sample.image, sample.gt_mask, sample.init_mask
    
    clicker = PI_Clicker(gt_mask=gt_mask)
    
    if uncertainty is False:
        clicker = Clicker(gt_mask=gt_mask)
    
    pred_mask = torch.zeros_like(torch.tensor(gt_mask)).cuda()
    pred_mask = pred_mask.unsqueeze(0)
    pred_mask = pred_mask.unsqueeze(0)
    prev_mask = pred_mask
    
    uncertainty_map = torch.zeros_like(torch.tensor(gt_mask)).cuda()
    prev_uncertainty_map = uncertainty_map
    out_image = None

    ious_list = []
    ious_b_list = []
    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            vis_pred = prev_mask
            
            if uncertainty:
                clicker.make_next_click(pred_mask, uncertainty_map=prev_uncertainty_map, top_uncertainty=True)
            else:
                clicker.make_next_click(pred_mask)
                
            pred_probs, uncertainty_map, ps = predictor.get_prediction(clicker, sample_cnt=sample_cnt, prev_mask=prev_mask)
            pred_mask = pred_probs > pred_thr

            prev_uncertainty_map = torch.tensor(uncertainty_map).cuda()

            iou = fn_performance(gt_mask, pred_mask)['iou_f']
            ious_list.append(iou)
            
            iou_b = fn_performance(gt_mask, pred_mask)['iou_b']
            ious_b_list.append(iou_b)
            
            prev_mask = torch.tensor(pred_mask).cuda()
            prev_mask = prev_mask.unsqueeze(0)
            prev_mask = prev_mask.unsqueeze(0)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

            if vis:
                clicks_list = clicker.get_clicks()
                last_y, last_x = predictor.last_y, predictor.last_x
                out_image = vis_result_base(save_dir, index, image, pred_mask, gt_mask, init_mask, iou, 
                                            click_indx+1, clicks_list, vis_pred, last_y, last_x, uncertainty_map)
    
    if vis:
        return np.array(ious_list, dtype=np.float32), np.array(ious_b_list, dtype=np.float32), pred_probs, pred_mask, uncertainty_map, out_image, clicker, ps
    
    return np.array(ious_list, dtype=np.float32), np.array(ious_b_list, dtype=np.float32), pred_probs, pred_mask, uncertainty_map, clicker, ps


def get_contour(mask, thresh=20):
    tmp_mask = np.array(mask, dtype=np.uint8)
    contour, hierarchy = cv2.findContours(tmp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = [x for x in contour if len(x)>thresh]
    
    return contour


def get_mito_info(contour):
    x0, y0 = zip(*np.squeeze(contour))
    x, y, w, h = cv2.boundingRect(contour)
    area = np.ceil(cv2.contourArea(contour))
    
    return area, w, h
    
    
def get_mito_info_lst(contours):
    mito_info_lst = []
    for contour in contours:
        tmp_info = {}
        area, w, h = get_mito_info(contour)
        
        tmp_info['Area'] = area
        tmp_info['witdh'] = w
        tmp_info['height'] = h
        
        mito_info_lst.append(tmp_info)
        
    return mito_info_lst


fn_tonumpy = lambda x: x.to('cpu').detach().numpy()