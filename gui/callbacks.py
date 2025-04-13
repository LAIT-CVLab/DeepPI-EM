# gui/callbacks.py

import os
import json
import base64
import cv2
import torch
import datetime
import numpy as np
from PIL import Image
from copy import deepcopy
import gradio as gr
from .logic import (
    run_inference,
    get_uncertainty_points,
    get_contour,
    get_mito_info_lst,
    get_magnitute_patch
)


# Visualization
point_colors = [(1, 0, 0), (0, 1, 0), (1, 1, 0)]
canvas_size = (640, 640)


_model = None
_device = None

def register_model(model, device):
    """Registers the segmentation model and device (CPU/GPU) for inference."""
    
    global _model, _device
    _model = model
    _device = device

    
def select_file(evt: gr.SelectData, files):
    """
    Loads the selected image file, prepares it for segmentation, 
    and initializes the result tracking dictionary.
    """
    
    selected_image = np.asarray(Image.open(files[evt.index].name).convert('RGB'))

    preprocessed_image_ = selected_image.copy()
    input_image_new = gr.ImageEditor(value=preprocessed_image_, height=canvas_size[1], label='Input',
                                     type='numpy', show_label=False, interactive=False, transforms=[],
                                     eraser=False, brush=False)

    curr_file_name = os.path.splitext(os.path.basename(files[evt.index].name))[0]

    date = datetime.datetime.now().date().strftime('%Y_%m_%d')
#     save_dir = './gradio_results/'
#     os.makedirs(os.path.join(save_dir, date), exist_ok=True)

#     file_index = 1
#     while os.path.exists(os.path.join(save_dir, date, f"{curr_file_name}_trial_{file_index}")):
#         file_index += 1

#     os.makedirs(os.path.join(save_dir, date, f"{curr_file_name}_trial_{file_index}"))

    tmp_save_results = {'file_name': curr_file_name}
    save_results_str = json.dumps(tmp_save_results)

    return selected_image, preprocessed_image_, input_image_new, [], [], [], None, None, None, None, None, None, False, curr_file_name, 0, save_results_str
    

def select_point(original_img, sel_pix, point_type, evt: gr.SelectData, processed_mask_li, 
                 uncertainty_map_li, output_mask_inter, uncertainty_points_li, 
                 save_results, user_name='unknown', consumption_time=0):
    """
    Handles user-selected positive/negative points and updates the segmentation mask,
    uncertainty map, and visual feedback accordingly.
    """
    
    minute, second = divmod(consumption_time, 60)
    consumption_time_txt = f'{minute:02d}:{second:02d}'

    brush = gr.Brush(
        colors=['#000000'], default_color="#000000", color_mode="fixed", default_size=5
    )

    with torch.no_grad():
        img = original_img.copy()

        # Append point based on selection
        label = 0 if point_type == 'negative_point' else 1
        sel_pix.append((evt.index, label))

        # Run model
        o_mask, o_uncertainty = run_inference(_model, original_img, sel_pix, _device)
        o_mask = (o_mask.detach().cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255
        processed_mask_li.append(o_mask)

        o_uncertainty = o_uncertainty.detach().cpu().numpy()[0, 0]
        uncertainty_map_li.append(o_uncertainty)

        # Visualization
        stack_mask = np.stack([o_mask]*3, axis=2) * (0, 0, 1)
        img = (img / 255 * 0.8 + stack_mask / 255 * 0.2).astype(np.float32)

        uncertainty_points = get_uncertainty_points(o_uncertainty)
        uncertainty_points_li.append(uncertainty_points)

        for point, label in sel_pix:
            cv2.circle(img, point, radius=5, color=point_colors[label], thickness=-1)

        mag_img_li = []
        marker_img = deepcopy(img)
        for i, point in enumerate(uncertainty_points):
            cv2.drawMarker(marker_img, point, point_colors[2], cv2.MARKER_CROSS, 25, 3)
            cv2.putText(marker_img, str(i+1), (point[0]+10, point[1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, point_colors[2], 2)
            mag_img_li.append(get_magnitute_patch(img, point))

        marker_img = np.clip(marker_img * 255, 0, 255).astype(np.uint8)
        uncertainty_map_viewer = (o_uncertainty * 255).astype(np.uint8)

        # Save as base64
        b64_mask = base64.b64encode(cv2.imencode('.png', o_mask)[1]).decode('utf-8')
        b64_uncertainty_map = base64.b64encode(
            cv2.imencode('.png', uncertainty_map_viewer)[1]
        ).decode('utf-8')

        tmp_save_results = {} if save_results is None else json.loads(save_results)
        result_entry = {
            'consumption_time': consumption_time_txt,
            'num_points': len(sel_pix),
            'points_coord': sel_pix,
            'uncertainty_points': uncertainty_points,
            'mask': b64_mask,
            'uncertainty_map': b64_uncertainty_map
        }

        if not output_mask_inter:
            tmp_save_results['user'] = user_name
            tmp_save_results['results'] = [result_entry]
            save_results_str = json.dumps(tmp_save_results)

            output_mask_new = gr.ImageEditor(
                value=o_mask, height=canvas_size[1], label='Mask', type='numpy',
                interactive=True, transforms=[], eraser=False, brush=brush,
                show_label=False, show_download_button=True, elem_classes='output_mask_ie'
            )

            uncertainty_map_new = gr.ImageEditor(
                value=uncertainty_map_viewer, height=canvas_size[1], label='Uncertainty',
                type='numpy', interactive=False, show_label=False, show_download_button=False
            )

            return (marker_img, output_mask_new, uncertainty_map_new, o_mask, True, len(sel_pix),
                    *mag_img_li, save_results_str)

        # If not the first interaction
        tmp_save_results['results'].append(result_entry)
        save_results_str = json.dumps(tmp_save_results)

        return (marker_img, o_mask, uncertainty_map_viewer, o_mask, True, len(sel_pix),
                *mag_img_li, save_results_str)

    
def select_uncertainty_point(original_img, sel_pix, point_type_orange, point_type_blue, point_type_yellow, processed_mask_li,
                 uncertainty_map_li, output_mask_inter, uncertainty_points_li, submit_uncertainty_points_li, save_results, user_name='unknown', consumption_time=0):
    """
    Applies uncertainty-based feedback points (suggested by the system), runs inference again,
    and updates mask, uncertainty map, and visual overlays.
    """
    
    minute, second = divmod(consumption_time, 60)
    consumption_time_txt = '{:02d}:{:02d}'.format(minute, second)

    if len(uncertainty_points_li) < 1:
        gr.Warning("There is no feedbacks")
             
        return {input_image: original_img}
    

    with torch.no_grad():
        img = original_img.copy()

        tmp_uncertainty_points_li = []
        if point_type_orange == 'positive':
            sel_pix.append((uncertainty_points_li[-1][0], 1))
            tmp_uncertainty_points_li.append(uncertainty_points_li[-1][0])
        elif point_type_orange == 'negative':
            sel_pix.append((uncertainty_points_li[-1][0], 0))
            tmp_uncertainty_points_li.append(uncertainty_points_li[-1][0])

        if point_type_blue == 'positive':
            sel_pix.append((uncertainty_points_li[-1][1], 1))
            tmp_uncertainty_points_li.append(uncertainty_points_li[-1][1])
        elif point_type_blue == 'negative':
            sel_pix.append((uncertainty_points_li[-1][1], 0))
            tmp_uncertainty_points_li.append(uncertainty_points_li[-1][1])

        if point_type_yellow == 'positive':
            sel_pix.append((uncertainty_points_li[-1][2], 1))
            tmp_uncertainty_points_li.append(uncertainty_points_li[-1][2])
        elif point_type_yellow == 'negative':
            sel_pix.append((uncertainty_points_li[-1][2], 0))
            tmp_uncertainty_points_li.append(uncertainty_points_li[-1][2])
            
        if len(tmp_uncertainty_points_li) > 0:
            submit_uncertainty_points_li.append(tmp_uncertainty_points_li)

        # run inference
        o_mask, o_uncertainty = (None, None)
        o_mask, o_uncertainty = run_inference(_model, original_img, sel_pix, _device)
        o_mask = np.asarray(o_mask.detach().cpu()[0, 0, :, :]) > 0.5
        o_mask = o_mask * 255
        processed_mask_li.append(o_mask)

        o_uncertainty = np.asarray(o_uncertainty.detach().cpu()[0, 0])
        uncertainty_map = o_uncertainty

        uncertainty_map_li.append(uncertainty_map)

        stack_mask = np.stack([o_mask, o_mask, o_mask], axis=2) * (0, 0, 1)
        img = img/255 * 0.8 + stack_mask/255 * 0.2

        # Uncertainty based feedback
        uncertainty_points = get_uncertainty_points(o_uncertainty)
        uncertainty_points_li.append(uncertainty_points)

        # draw point
        for point, label in sel_pix:
            cv2.circle(img, point, radius=5, color=point_colors[label], thickness=-1)

        # draw point
        mag_img_li = []
        marker_img = deepcopy(img)
        for i, point in enumerate(uncertainty_points):
            cv2.drawMarker(marker_img, point, point_colors[2], cv2.MARKER_CROSS, 25, 3)
            cv2.putText(marker_img, str(i+1), (point[0]+10, point[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, point_colors[2], 2)
            mag_img_li.append(get_magnitute_patch(img, point))

        output_mask_new = None
        uncertainty_map_new = None

        b64_mask = o_mask.tobytes()
        b64_mask = base64.b64encode(b64_mask).decode('utf-8')

        b64_uncertainty_map = uncertainty_map.tobytes()
        b64_uncertainty_map = base64.b64encode(b64_uncertainty_map).decode('utf-8')

        tmp_save_results = {} if save_results is None else json.loads(save_results)
        tmp_save_results['results'].append({
            'consumption_time': consumption_time_txt,
            'num_points': len(sel_pix),
            'points_coord': sel_pix,
            'uncertainty_points': uncertainty_points,
            'mask': b64_mask,
            'uncertainty_map': b64_uncertainty_map
        })
        save_results_str = json.dumps(tmp_save_results)

        return marker_img, o_mask, uncertainty_map, o_mask, True, len(sel_pix), mag_img_li[0], mag_img_li[1], mag_img_li[2], save_results_str

    
def edit_mask(edit_img):
    """Returns the edited mask from the image editor (after manual editing)."""
    
    return edit_img['composite']


def undo_points(original_img, sel_pix, processed_mask_li, uncertainty_map_li, uncertainty_points_li, submit_uncertainty_points_li):
    """
    Removes the most recent user interaction (either manual or uncertainty-guided),
    and updates the mask, uncertainty map, and visualization.
    """
    
    brush = gr.Brush(colors=['#000000'], default_color="#000000", color_mode="fixed", default_size=5)

    if not isinstance(original_img, np.ndarray):
        return gr.Warning("Please upload an image.")

    if len(sel_pix) < 1 or len(processed_mask_li) < 1:
        while len(uncertainty_points_li): uncertainty_points_li.pop()
        return original_img, None, None, None, None, None, None, False, 0, None, None, None

    # Pop last mask & uncertainty map
    processed_mask_li.pop()
    uncertainty_map_li.pop()

    if len(processed_mask_li):
        temp_mask = processed_mask_li[-1]
        temp_uncertainty = uncertainty_map_li[-1]
    else:
        sel_pix.clear()
        uncertainty_points_li.clear()
        submit_uncertainty_points_li.clear()
        return original_img, None, None, None, None, None, None, False, 0, None, None, None

    temp_img = original_img.copy().astype(np.float32) / 255.0
    stack_mask = np.zeros_like(temp_img, dtype=np.float32)
    stack_mask[..., 2] = temp_mask.astype(np.float32) / 255.0

    temp_img = temp_img * 0.8 + stack_mask * 0.2
    
    # Undo clicked points
    if len(submit_uncertainty_points_li) > 0:
        for _ in submit_uncertainty_points_li[-1]:
            sel_pix.pop()
        submit_uncertainty_points_li.pop()
    else:
        sel_pix.pop()
        uncertainty_points_li.pop()

    # Re-draw points
    for point, label in sel_pix:
        cv2.circle(temp_img, point, radius=5, color=point_colors[label], thickness=-1)
       
    mag_img_li = []
    marker_img = deepcopy(temp_img)
    for i, point in enumerate(uncertainty_points_li[-1]):
        cv2.drawMarker(marker_img, point, point_colors[2], cv2.MARKER_CROSS, 25, 3)
        cv2.putText(marker_img, str(i+1), (point[0]+10, point[1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, point_colors[2], 2)
        mag_img_li.append(get_magnitute_patch(temp_img, point))

    temp_img = np.clip(temp_img * 255, 0, 255).astype(np.uint8)
        
    output_mask_new = gr.ImageEditor(value=temp_mask, height=canvas_size[1], label='Mask',
                                     type='numpy', interactive=True, transforms=[], eraser=False,
                                     brush=brush, show_label=False, show_download_button=True)

    uncertainty_map_new = gr.ImageEditor(value=temp_uncertainty, height=canvas_size[1],
                                         label='Uncertainty', type='numpy', interactive=False,
                                         show_label=False)

    return marker_img, output_mask_new, uncertainty_map_new, temp_mask, None, None, None, True, len(sel_pix), mag_img_li[0], mag_img_li[1], mag_img_li[2]


def reset_image(img):
    """
    Resets all states and visualization components to start from a clean image.
    """
    
    if isinstance(img, np.ndarray) is not True:
        gr.Warning("Please upload image.")
        return {input_image: None}
    
    preprocessed_image_ = img.copy()        
    output_mask_new = gr.ImageEditor(value=None, height=canvas_size[1], label='Mask', type='numpy',
                                     interactive=False, show_label=False, show_download_button=False)
    uncertainty_map_new = gr.ImageEditor(value=None, height=canvas_size[1], label='Uncertainty', type='numpy', 
                                   interactive=False, show_label = False)

    return [
        img,                  # original_image
        preprocessed_image_,  # preprocessed_image
        preprocessed_image_,  # input_image
        [],                   # selected_points
        [],                   # processed_mask_li
        [],                   # uncertainty_map_li
        output_mask_new,      # output_mask
        uncertainty_map_new,  # uncertainty_map
        None,                 # edited_mask
        None,                 # crop_object_gallery
        None,                 # num_object
        None,                 # result_table
        False,                # output_mask_inter
        0,                    # num_points
        [],                   # uncertainty_points_li
        None,                 # mag_image_1
        None,                 # mag_image_2
        None,                 # mag_image_3
    ]
    

def clear_image():
    """Clears all UI elements (input image, mask, map) and state values."""
    
    input_image = gr.ImageEditor(value=None, height=canvas_size[1], label='Input', type='numpy', 
                                 interactive=False, show_label=False, transforms=[], eraser=False, brush=False)

    output_mask = gr.ImageEditor(value=None, height=canvas_size[1], label='Mask', type='numpy', 
                                 interactive=False, show_label=False, show_download_button=False)
    
    uncertainty_map = gr.ImageEditor(value=None, height=canvas_size[1], label='Uncertainty', type='numpy', 
                                     interactive=False, show_label=False)
    
    return (None, None, input_image, [], [], [], output_mask, uncertainty_map, None, None, None, None, False, 0, None, None, None, None)


def complete_edit(original_image, edited_mask, sel_pix, curr_file_name, save_results, consumption_time=0):
    """
    Prepares the final edited mask for export and disables further editing.
    (Commented part originally included saving results to file.)
    """
    
#     minute, second = divmod(consumption_time, 60)
#     consumption_time_txt = '{:02d}:{:02d}'.format(minute, second)

#     save_dir = './gradio_results/'
#     date = datetime.datetime.now().date().strftime('%Y_%m_%d')

#     if not os.path.exists(os.path.join(save_dir, date)):
#         os.makedirs(os.path.join(save_dir, date))

#     tmp_file_name_li = [x for x in os.listdir(os.path.join(save_dir, date)) if curr_file_name in x]
#     trial_index = max([int(x.split('_trial_')[1]) for x in tmp_file_name_li], default=0) + 1
#     file_pth = f"{curr_file_name}_trial_{trial_index}"
#     os.makedirs(os.path.join(save_dir, date, file_pth), exist_ok=True)

#     tmp_save_results = {} if save_results is None else json.loads(save_results)

#     b64_image = base64.b64encode(original_image.tobytes()).decode('utf-8')
#     b64_mask = base64.b64encode(edited_mask.tobytes()).decode('utf-8')

#     tmp_save_results['image'] = b64_image
#     tmp_save_results['final'] = {
#         'consumption_time': consumption_time_txt,
#         'num_points': len(sel_pix),
#         'points_coord': sel_pix,
#         'mask': b64_mask
#     }

#     with open(os.path.join(save_dir, date, file_pth, f'experiment.json'), "w") as f:
#         json.dump(tmp_save_results, f)

    output_mask_new = gr.ImageEditor(value=edited_mask, height=canvas_size[1], type='numpy',
                                     interactive=False, show_label=False, show_download_button=True)

    return output_mask_new, False


def quantify_object(image, edited_mask):
    """
    Computes metrics (area, width, height) for segmented mitochondria 
    and returns cropped image patches of the objects.
    """
    
    contours = get_contour(edited_mask)
    mito_info_lst = get_mito_info_lst(image, edited_mask, contours)

    results, crops = [], []
    for i, m in enumerate(mito_info_lst, start=1):
        if m['area'] > 10:
            results.append([i, m['width'], m['height'], m['area']])
            crops.append(m['crop_object'])

    return len(results), results, crops


def stop_timer():
    """Stops the UI usage timer."""
    
    return True


def get_time_consumption(consumption_time, stop_flag):
    """
    Increments the elapsed time counter unless paused, 
    and formats the result into MM:SS format.
    """
    
    if stop_flag:
        consumption_time -= 1
        stop_flag = False

    consumption_time += 1
    minute, second = divmod(consumption_time, 60)
    consumption_time_txt = '{:02d}:{:02d}'.format(minute, second)

    return consumption_time_txt, consumption_time, stop_flag


def clear_time():
    """Resets the elapsed time counter to 00:00 and restarts the timer."""
    
    consumption_time = 0
    minute, second = divmod(consumption_time, 60)
    consumption_time_txt = '{:02d}:{:02d}'.format(minute, second)
    return consumption_time_txt, 0, True