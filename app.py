import gradio as gr
from experiments.otsu_segmenter import generate_segmented_image
from experiments.kmeans_segmenter import generate_kmeans_segmented_image
from PIL import Image

def generate_kmeans(image_path,k):
    kmeans_image_output, kmeans_segmented_image_output,_,kmeans_threshold_text=generate_kmeans_segmented_image(image_path, k)
    return kmeans_image_output, kmeans_segmented_image_output, kmeans_threshold_text

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
        with gr.TabItem("K-means Segmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    kmeans_file_input = gr.File(label="Upload Image File")
                    kmeans_k_value = gr.Slider(minimum=2, maximum=10, value=3, step=1, label="Number of Clusters (K)")
                    kmeans_display_btn = gr.Button("Segment this image")
                    kmeans_threshold_text = gr.Textbox(label="K-means Info", value="", interactive=False)
                
                with gr.Column(scale=2):
                    kmeans_image_output = gr.Image(label="Original Image", container=False)
                    kmeans_segmented_image_output = gr.Image(label="K-means Segmented Image", container=False)
            
            # Connect buttons to function
            kmeans_display_btn.click(
                fn=generate_kmeans,
                inputs=[kmeans_file_input, kmeans_k_value],
                outputs=[kmeans_image_output, kmeans_segmented_image_output, kmeans_threshold_text]
        )
            
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

