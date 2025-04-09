import gradio as gr
from experiments.otsu_segmenter import generate_segmented_image
from experiments.kmeans_segmenter import generate_kmeans_segmented_image
from experiments.enhanced_kmeans_segmenter import slic_kmeans
from experiments.watershed_segmenter import generate_watershed
from PIL import Image

def generate_kmeans(image_path,k):
    kmeans_image_output, kmeans_segmented_image_output,_,kmeans_threshold_text=generate_kmeans_segmented_image(image_path, k)
    return kmeans_image_output, kmeans_segmented_image_output, kmeans_threshold_text

def generate_slic(image_path,k,m,max_iter):
    image,seg_img, labels, centers = slic_kmeans(image_path, K=k, m=m, max_iter=max_iter)
    return image,seg_img

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
        with gr.TabItem("SLIC Segmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    slic_file_input = gr.File(label="Upload Image File")
                    slic_k_value = gr.Slider(minimum=2, maximum=200, value=3, step=1, label="Number of superpixels")
                    slic_m_value = gr.Slider(minimum=1, maximum=40, value=3, step=1, label="Compactness factor")
                    slic_max_iter_value = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of iterations")
                    slic_display_btn = gr.Button("Segment this image")
                
                with gr.Column(scale=2):
                    slic_image_output = gr.Image(label="Original Image", container=False)
                    slic_segmented_image_output = gr.Image(label="SLIC Segmented Image", container=False)
            
            # Connect buttons to function
            slic_display_btn.click(
                fn=generate_slic,
                inputs=[slic_file_input, slic_k_value,slic_m_value,slic_max_iter_value],
                outputs=[slic_image_output,slic_segmented_image_output]
        )
            
        with gr.TabItem("Watershed Algorithm Segmentation"):
            with gr.Row():
                with gr.Column(scale=1):
                    watershed_file_input = gr.File(label="Upload Image File")
                    watershed_display_btn = gr.Button("Segment this image")
                
                with gr.Column(scale=2):
                    watershed_image_output = gr.Image(label="Original Image", container=False)
                    watershed_segmented_image_output = gr.Image(label="watershed Segmented Image", container=False)
            
            # Connect buttons to function
            watershed_display_btn.click(
                fn=generate_watershed,
                inputs=[watershed_file_input],
                outputs=[watershed_image_output,watershed_segmented_image_output]
        )
if __name__ == "__main__":
    demo.launch(server_name="172.31.100.127")

