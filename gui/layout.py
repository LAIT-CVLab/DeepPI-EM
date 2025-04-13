# gui/layout.py

import gradio as gr
from .callbacks import (
    select_point, select_uncertainty_point, select_file, reset_image, clear_image, 
    undo_points, edit_mask, quantify_object, complete_edit, get_time_consumption, 
    stop_timer, clear_time
)

canvas_size = (640, 640)

custom_css = f"""
    .output_mask_ie>.image-container>.wrap>.stage-wrap {{
        margin: 0 !important;
    }}
    .output_mask_ie>.image-container>.wrap>.stage-wrap>canvas {{
        margin-top: 56.3px !important;
        max-width: var(--size-full) !important;
        max-height: var(--size-full) !important;
        height: var(--size-full) !important;
        width: var(--size-full) !important;
        border-radius: var(--radius-lg) !important;
    }}
    .file_explorer {{
        height: {canvas_size[1]}px !important;    
"""

def build_gradio_ui(device, canvas_size, point_colors):
    with gr.Blocks(css=custom_css) as demo:
        # --- States ---
        selected_points = gr.State(value=[])
        original_image = gr.State(value=None)
        preprocessed_image = gr.State(value=None)
        processed_mask_li = gr.State(value=[])
        uncertainty_map_li = gr.State(value=[])
        edited_mask = gr.State(value=None)
        output_mask_inter = gr.State(value=False)
        curr_file_name = gr.State(value=None)
        uncertainty_points_li = gr.State(value=[])
        submit_uncertainty_points_li = gr.State(value=[])
        save_results = gr.State(value=None)
        consumption_time = gr.State(value=0)
        time_stop_flag = gr.State(value=True)

        # --- Layout ---
        with gr.Row():
            gr.Markdown("## DEMO: Probabilistic Interactive Segmentation")

        with gr.Row():
            with gr.Column():
                with gr.Tab("Files"):
                    with gr.Row():
                        file_explorer = gr.File(file_count="multiple", file_types=["image"], show_label=False,
                                                height=canvas_size[1], elem_classes='file_explorer')
                    
                    with gr.Row(visible=False):
                        user_name_txt = gr.Textbox(label='User name', placeholder='User name')
                    with gr.Row(visible=False):
                        with gr.Column(scale=6):
                            consumption_time_txt = gr.Textbox(label="Usage Time", lines=1, value='00:00')
                        with gr.Column(scale=5):
                            st_time_button = gr.Button('start')
                            stop_time_button = gr.Button('stop')
                        with gr.Column():
                            clear_time_button = gr.Button('new start')

                with gr.Tab("Input Image"):
                    input_image = gr.ImageEditor(height=canvas_size[1], label='Input', type='numpy', show_label=False,
                                                 interactive=False, show_download_button=False, transforms=[],
                                                 eraser=False, brush=False, layers=False)
                    
                    with gr.Row():
                        with gr.Column(scale=6):
                            point_radio = gr.Radio(
                                ['positive_point', 'negative_point'],
                                info="Select positive or negative point",
                                label='Point label')

                        with gr.Column(scale=5):
                            undo_button = gr.Button('Undo point')
                            clear_button = gr.Button('Clear')

                    with gr.Row():
                        num_points = gr.Number(info='Number of points', interactive=False, show_label=False)

                    gr.Markdown("------")
                    gr.Markdown("### Uncertainty feedback recommendation")
                    with gr.Accordion("Magnitute samples", open=True):
                        with gr.Row():
                            mag_image_1 = gr.Image(show_download_button=False, show_label=False, interactive=False)
                            mag_image_2 = gr.Image(show_download_button=False, show_label=False, interactive=False)
                            mag_image_3 = gr.Image(show_download_button=False, show_label=False, interactive=False)

                    with gr.Row():
                        uncertainty_point_radio_orange = gr.Radio(['positive', 'negative', 'non'], label="Point 1")
                        uncertainty_point_radio_blue = gr.Radio(['positive', 'negative', 'non'], label="Point 2")
                        uncertainty_point_radio_yellow = gr.Radio(['positive', 'negative', 'non'], label="Point 3")
                        uncertainty_point_button = gr.Button("Submit")

            with gr.Column(scale=1):
                with gr.Tab("Mask"):
                    output_mask = gr.ImageEditor(height=canvas_size[1], label='Mask', type='numpy', show_label=False,
                                                 interactive=False, show_download_button=False, transforms=[],
                                                 eraser=False, brush=False, layers=False, elem_classes='output_mask_ie')

                    complete_button = gr.Button("Complete")

                    with gr.Accordion("Uncertainty Map", open=True):
                        uncertainty_map = gr.ImageEditor(height=canvas_size[1], label='Uncertainty', type='numpy',
                                                         show_label=False, interactive=False, show_download_button=False)

                with gr.Tab("Results"):
                    crop_object_gallery = gr.Gallery(object_fit="fit", show_label=False)
                    num_object = gr.Number(info="Number of Mitochondria", interactive=False, show_label=False)
                    result_table = gr.Dataframe(type="numpy", datatype="number", row_count=1, col_count=4,
                                                headers=["index", "width", "height", "area"],
                                                wrap=True, show_label=False, interactive=False)
                    analyze_button = gr.Button("Analyze")

        # --- Events ---
        input_image.select(
            select_point,
            [preprocessed_image, selected_points, point_radio, processed_mask_li, uncertainty_map_li,
             output_mask_inter, uncertainty_points_li, save_results, user_name_txt, consumption_time],
            [input_image, output_mask, uncertainty_map, edited_mask, output_mask_inter, num_points,
             mag_image_1, mag_image_2, mag_image_3, save_results]
        )

        uncertainty_point_button.click(
            select_uncertainty_point,
            [preprocessed_image, selected_points, uncertainty_point_radio_orange,
             uncertainty_point_radio_blue, uncertainty_point_radio_yellow, processed_mask_li,
             uncertainty_map_li, output_mask_inter, uncertainty_points_li, submit_uncertainty_points_li,
             save_results, user_name_txt, consumption_time],
            [input_image, output_mask, uncertainty_map, edited_mask, output_mask_inter, num_points,
             mag_image_1, mag_image_2, mag_image_3, save_results]
        )

        input_image.clear(
            clear_image,
            [],
            [original_image, preprocessed_image, input_image, selected_points, processed_mask_li,
             uncertainty_map_li, output_mask, uncertainty_map, edited_mask, crop_object_gallery,
             num_object, result_table, output_mask_inter, num_points, mag_image_1, mag_image_2,
             mag_image_3, save_results]
        )

        undo_button.click(
            undo_points,
            [preprocessed_image, selected_points, processed_mask_li, uncertainty_map_li,
             uncertainty_points_li, submit_uncertainty_points_li],
            [input_image, output_mask, uncertainty_map, edited_mask, crop_object_gallery, num_object,
             result_table, output_mask_inter, num_points, mag_image_1, mag_image_2, mag_image_3]
        )

        clear_button.click(
            reset_image,
            [original_image],
            [original_image, preprocessed_image, input_image, selected_points, processed_mask_li,
             uncertainty_map_li, output_mask, uncertainty_map, edited_mask, crop_object_gallery,
             num_object, result_table, output_mask_inter, num_points, uncertainty_points_li,
             mag_image_1, mag_image_2, mag_image_3]
        )

        output_mask.apply(edit_mask, [output_mask], [edited_mask])
        analyze_button.click(quantify_object, [preprocessed_image, edited_mask], [num_object, result_table, crop_object_gallery])

        get_time_event = st_time_button.click(
            get_time_consumption,
            [consumption_time, time_stop_flag],
            [consumption_time_txt, consumption_time, time_stop_flag],
            every=1
        )

        stop_time_button.click(stop_timer, [], [time_stop_flag], cancels=[get_time_event])
        clear_time_button.click(clear_time, [], [consumption_time_txt, consumption_time, time_stop_flag], cancels=[get_time_event])

        complete_button.click(
            complete_edit,
            [original_image, edited_mask, selected_points, curr_file_name, save_results, consumption_time],
            [output_mask, output_mask_inter],
            cancels=[get_time_event]
        ).then(stop_timer, [], [time_stop_flag])

        file_explorer.select(
            select_file,
            [file_explorer],
            [original_image, preprocessed_image, input_image, selected_points, processed_mask_li,
             uncertainty_map_li, output_mask, uncertainty_map, edited_mask, crop_object_gallery,
             num_object, result_table, output_mask_inter, curr_file_name, num_points, save_results]
        )

        file_explorer.clear(
            clear_image,
            [],
            [original_image, preprocessed_image, input_image, selected_points, processed_mask_li,
             uncertainty_map_li, output_mask, uncertainty_map, edited_mask, crop_object_gallery,
             num_object, result_table, output_mask_inter, num_points, mag_image_1, mag_image_2,
             mag_image_3, save_results]
        )

    return demo