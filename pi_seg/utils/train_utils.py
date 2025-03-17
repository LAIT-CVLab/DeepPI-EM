import numpy as np
import cv2


def get_next_points_removeall_realworld(gt, points, click_indx, points_focus=None, rois=None, 
                              pred_thresh=0.5, remove_prob = 0.0, uncertainty_map=None, top_uncertainty=True):
    assert click_indx > 0

    gt = gt.cpu().detach().numpy() > 0.5
    uncertainty_map = uncertainty_map.cpu().detach().numpy()
    num_points = points.size(1) // 2
    points = points.clone()

    bindx = 0
    for uncertainty, gt_mask in zip(uncertainty_map, gt):            
        max_index = np.where(uncertainty[0] == np.max(uncertainty[0]))
        min_index = np.where(uncertainty[0] == np.min(uncertainty[0]))

        max_x, max_y = max_index[0][0], max_index[1][0]
        min_x, min_y = min_index[0][0], min_index[1][0]

        coords = (max_x, max_y)
        
        if top_uncertainty == None:
            coords = (max_x, max_y)
        elif top_uncertainty == False:
            coords = (min_x, min_y)

        is_positive = gt_mask[0][coords[0], coords[1]] == 1

        if np.random.rand() < remove_prob:
            points[bindx] = points[bindx] * 0.0 - 1.0
        if is_positive:
            points[bindx, num_points - click_indx, 0] = float(coords[0])
            points[bindx, num_points - click_indx, 1] = float(coords[1])
            points[bindx, num_points - click_indx, 2] = float(click_indx)
        else:
            points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
            points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
            points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)
            
        bindx += 1
            
    return points


def get_next_points_removeall(pred, gt, points, click_indx, points_focus=None, rois=None, 
                              pred_thresh=0.5, remove_prob = 0.0):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred >= pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if np.random.rand() < remove_prob:
                points[bindx] = points[bindx] * 0.0 - 1.0
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

    return points


def fn_iou(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union


def check_divide_zero(value):
    if value > 1:
        return -1
    
    return value


def fn_performance(label_, output):
    FP = len(np.where(output - label_  == 1)[0])
    FN = len(np.where(output - label_  == -1)[0])
    TP = len(np.where(output + label_ == 2)[0])
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


def fn_threshold(x):
    x[x >= 0.5] = 1
    x[x < 0.5] = 0
    
    return x


fn_tonumpy = lambda x: x.to('cpu').detach().numpy()