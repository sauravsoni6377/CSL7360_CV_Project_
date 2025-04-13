# Image Segmentation Toolkit

## Overview
This project implements a comprehensive image segmentation toolkit that combines classical computer vision techniques with deep learning-based approaches. The application provides an interactive interface to compare different segmentation algorithms on user-provided images.

Our Gradio web Demo: https://huggingface.co/spaces/AJain1234/Image_Segmentation_ 

## Features
- **Classical Segmentation Methods**:
  - Otsu's Thresholding: Optimal global thresholding for binary segmentation
  - K-means Clustering: Color-based segmentation with adjustable clusters
  - SLIC (Simple Linear Iterative Clustering): Superpixel segmentation
  - Watershed Algorithm: Gradient-based segmentation for separating touching objects
  - Felzenszwalb Algorithm: Graph-based segmentation with adaptive thresholding

- **Deep Learning Models**:
  - SegNet with EfficientNet B0 backbone: Pretrained semantic segmentation model
  - SegNet with VGG backbone: Alternative architecture for comparison

- **Ensemble Methods**:
  - Otsu + SegNet: Combining boundary information from Otsu with semantic labels from SegNet
  - Custom ensemble segmentation with adjustable parameters

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (recommended)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/CSL7360_Project.git
cd CSL7360_Project
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download pretrained models:
```bash
python download_models.py
```
   The application will also automatically download models when first launched.

## Usage

### Running the Application
Start the Gradio web interface:
```bash
python app.py
```

The interface will be available at http://127.0.0.1:7860 in your web browser.

### Using the Interface
1. Select a segmentation method from the tabs at the top
2. Upload an image using the file picker
3. Adjust algorithm parameters if available
4. Click the "Segment this image" button
5. View the results in the display area

### Algorithm Parameters

#### Otsu's Method
- No parameters, fully automatic threshold selection

#### K-means Segmentation
- **Number of Clusters (K)**: Controls how many color groups to segment into

#### SLIC Segmentation
- **Number of superpixels**: Controls the granularity of segmentation
- **Compactness factor**: Controls how much superpixels adhere to boundaries
- **Number of iterations**: Controls refinement of superpixel boundaries

#### Felzenszwalb Algorithm
- **Sigma**: Gaussian pre-processing smoothing parameter
- **K value**: Controls segment size preference 
- **Min Size Factor**: Minimum component size

#### Ensemble Segmentation
- **Boundary Refinement Weight**: Controls influence of classical methods on deep learning boundaries

## Project Structure
```
CSL7360_Project/
├── app.py                      # Main application with  pretrained models
├── experiments/                # Implementation of segmentation algorithms
│   ├── ensemble_method.py      # Ensemble segmentation implementation
│   ├── felzenszwalb_segmentation/ # Felzenszwalb algorithm implementation
│   ├── kmeans_segmenter.py     # K-means segmentation implementation
│   ├── enhanced_kmeans_segmenter.py # SLIC implementation
│   ├── otsu_segmenter.py       # Otsu thresholding implementation
│   ├── watershed_segmenter.py  # Watershed algorithm implementation
│   └── SegNet/                 # Deep learning models
│       ├── efficient_b0_backbone/ # EfficientNet backbone for SegNet
│       └── vgg_backbone/       # VGG backbone for SegNet
├── saved_models/              # Directory for pretrained weights
└── requirements.txt           # Package dependencies
```

## Examples
The application works well on a variety of images:
- Natural scenes
- Urban environments
- Medical images
- Aerial/satellite imagery
- Objects with clear boundaries

## Technologies Used
- **PyTorch**: Deep learning framework
- **OpenCV**: Classical computer vision algorithms
- **NumPy**: Numerical computations
- **PIL/Pillow**: Image loading and manipulation
- **Gradio**: Interactive web interface
- **Matplotlib**: Visualization of results

## Credits
- Built as part of CSL7360 course project
- Uses pretrained models based on Pascal VOC and CamVid datasets
- Implements algorithms from classical computer vision literature

## License
This project is available under the MIT License.
