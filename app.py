import gradio as gr
from experiments.otsu_segmenter import generate_segmented_image
from PIL import Image

with gr.Blocks() as demo:
    gr.Markdown("# Image Segmentation using Classical CV")
    
    with gr.Tabs() as tabs:
        # Tab 1: CNN+LSTM
        with gr.TabItem("Otsu's Method"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Image File")
                    display_btn = gr.Button("Segment this image")
                    threshold_text = gr.Textbox(label="Threshold Comparison", value="", interactive=False)
                
                with gr.Column(scale=2):
                    image_output = gr.Image(label="Original Image", container=False)
                    histogram_output = gr.Image(label="Histogram", container=False)
                    segmented_image_output = gr.Image(label="Our Segmented Image", container=False)
                    opencv_segmented_image_output = gr.Image(label="OpenCV Segmented Image", container=False)
            # Connect buttons to functions
            display_btn.click(
                fn=generate_segmented_image,
                inputs=file_input,
                outputs=[image_output, segmented_image_output, opencv_segmented_image_output, histogram_output, threshold_text]
            )
        # Tab 2: GCN+GRU
        with gr.TabItem("GCN+GRU"):
            with gr.Row():
                with gr.Column(scale=1):
                    gcn_gru_file_input = gr.File(label="Upload PLY File")
                    gcn_gru_display_btn = gr.Button("Display Point Cloud and Generate Caption")
                    gcn_gru_view_only_btn = gr.Button("Display Point Cloud Only")
                    gcn_gru_status_text = gr.Textbox(label="Status", value="Ready to load point cloud", interactive=False)
                
                with gr.Column(scale=2):
                    gcn_gru_plot_output = gr.Plot(label="Point Cloud Visualization", container=False)
                    gcn_gru_caption_output = gr.Textbox(label="Generated Caption", interactive=False)
            
            
        # Tab 3: ViT+Transformer
        with gr.TabItem("ViT+Transformer"):
            with gr.Row():
                with gr.Column(scale=1):
                    vit_transformer_file_input = gr.File(label="Upload PLY File")
                    vit_transformer_display_btn = gr.Button("Display Point Cloud and Generate Caption")
                    vit_transformer_view_only_btn = gr.Button("Display Point Cloud Only")
                    vit_transformer_status_text = gr.Textbox(label="Status", value="Ready to load point cloud", interactive=False)
                
                with gr.Column(scale=2):
                    vit_transformer_plot_output = gr.Plot(label="Point Cloud Visualization", container=False)
                    vit_transformer_caption_output = gr.Textbox(label="Generated Caption", interactive=False)
if __name__ == "__main__":
    demo.launch(server_name="172.31.100.127")

