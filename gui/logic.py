# gui/logic.py

import torch
import numpy as np
import cv2
from torchvision import transforms as T
from sklearn.cluster import KMeans

from copy import deepcopy

def run_inference(model, original_img, sel_pix, device='cpu'):
    """
    Runs inference on the input image using the provided model and selected points.
    Returns the mean prediction mask and uncertainty map based on multiple samples.
    """
    
    model.eval()

    positive_points_li = []
    negative_points_li = []

    positive_points_li = [(p[0][1], p[0][0], i) for i, p in enumerate(sel_pix) if p[1] == 1]
    negative_points_li = [(p[0][1], p[0][0], i) for i, p in enumerate(sel_pix) if p[1] == 0]
    points_nd = get_points_nd(positive_points_li, negative_points_li, device)

    input_img = deepcopy(original_img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) #gray scale
    input_img = T.functional.to_tensor(input_img).type(torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model.forward(image=input_img, mask=None, points=points_nd, training=False, sample_cnt=16)

        ps = outputs['samples']
        ps = [torch.sigmoid(s) for s in ps]

        o_mask = torch.stack(ps).mean(dim=0)
        o_uncertainty = torch.stack(ps).std(dim=0, unbiased=False)

    return o_mask, o_uncertainty


def get_points_nd(pos_clicks, neg_clicks, device='cpu'):
    """
    Prepares a fixed-size tensor of positive and negative click coordinates for model input.
    Pads with placeholder values if there are fewer than the expected number of points.
    """
    
    limit = 1000
    max_points = max(len(pos_clicks), len(neg_clicks))
    max_points = min(limit, max_points) if limit else max_points
    max_points = max(1, max_points)

    pos = pos_clicks + [(-1, -1, -1)] * (max_points - len(pos_clicks))
    neg = neg_clicks + [(-1, -1, -1)] * (max_points - len(neg_clicks))
    return torch.tensor([pos + neg], device=device)


def get_uncertainty_points(uncertainty_map):
    """
    Identifies regions of highest uncertainty from the map using top-K selection and K-means clustering.
    Returns 3 representative coordinates.
    """
    
    flat = uncertainty_map.flatten()
    top_indices = np.argsort(flat)[-100:]
    active_coords = np.column_stack(np.unravel_index(top_indices, uncertainty_map.shape))
    kmeans = KMeans(n_clusters=3, random_state=42).fit(active_coords)
    return [(int(c[1]), int(c[0])) for c in kmeans.cluster_centers_]


def get_contour(mask):
    """
    Extracts external contours from a binary segmentation mask for object-level analysis.
    """
    
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    gray = cv2.cvtColor(np.stack([mask]*3, axis=2).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_mito_info(image, mask, contour):
    """
    Computes object-level metrics (area, width, height) for a single segmented mitochondrion.
    Also returns the cropped object patch for further visualization or analysis.
    """
    
    x, y, w, h = cv2.boundingRect(contour)
    area = np.ceil(cv2.contourArea(contour))
    crop = image[y:y+h, x:x+w]

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask.ndim == 2:
        maskgray = mask
    elif mask.ndim == 3:
        maskgray = mask[:, :, 0]
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
        
    maskgray = cv2.cvtColor(np.stack([maskgray]*3, axis=2).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    filled = np.zeros(imgray.shape, dtype=imgray.dtype)
    filled = cv2.drawContours(filled, [contour], -1, (255, 255, 255), -1)
    _, inv_mask = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY_INV)
    subtracted = cv2.subtract(imgray, cv2.bitwise_and(imgray, inv_mask))
    _, cristae = cv2.threshold(subtracted, 127, 255, cv2.THRESH_BINARY)

    unique = np.unique(cristae, return_counts=True)[1]
    back_area = 0 if len(unique) < 2 else unique[1]
    crista_ratio = 1 - (back_area / area)

    return area, w, h, crista_ratio, crop


def get_mito_info_lst(image, mask, contours):
    """
    Computes mitochondrial properties for a list of contours using `get_mito_info`.
    Returns a list of object-wise metric dictionaries.
    """
    
    result = []
    for c in contours:
        area, w, h, cr, crop = get_mito_info(image, mask, c)
        result.append({
            'area': area,
            'width': w,
            'height': h,
            'crista_ratio': cr,
            'crop_object': crop
        })
    return result


def get_magnitute_patch(img, coord, patch_size=(100, 100)):
    """
    Extracts a zoomed-in image patch around a given coordinate with a marker overlay.
    Used for displaying uncertainty-based recommendation visuals.
    """
    
    marker_img = deepcopy(img)
    marker_img = cv2.drawMarker(marker_img, coord, (255, 255, 0), cv2.MARKER_CROSS, 15, 1)
    
    alpha = 0.7
    marker_img = cv2.addWeighted(img, 1 - alpha, marker_img, alpha, 0)

    x = int(coord[1] - patch_size[0]/2)
    x = x if x > 0 else 0
    x = x if x+patch_size[0] < marker_img.shape[0] else marker_img.shape[0]-patch_size[0]

    y = int(coord[0] - patch_size[1]/2)
    y = y if y > 0 else 0
    y = y if y+patch_size[1] < marker_img.shape[1] else marker_img.shape[1]-patch_size[1]
    
    patch = marker_img[x:x+patch_size[0], y:y+patch_size[1]]
    patch = np.clip(patch * 255, 0, 255).astype(np.uint8)
    
    return patch
