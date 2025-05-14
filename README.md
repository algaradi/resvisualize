# ResNet50 Feature Map Visualization Tool

A comprehensive visualization tool for exploring and understanding how ResNet50 processes images through its internal layers.

## Features

### Multiple Visualization Techniques
- **Feature Maps**: Visualize activation patterns at different layers
- **Activation Maximization**: Generate images that maximally activate specific neurons
- **Layer-wise Relevance Propagation (LRP)**: Understand which input pixels contribute most to specific activations
- **Grad-CAM**: Visualize important regions for classification decisions


### Advanced Customization Options
- **Image Preprocessing**:
  - Resize with multiple strategies (simple, preserve aspect ratio, pad and resize)
  - Custom cropping options (center, random, five-crop, ten-crop)
  - Normalization options (ImageNet standard, simple, custom)
  - Data augmentation (color jitter, rotation, flips, grayscale, blur)

- **Visualization Controls**:
  - Custom grid size for feature map displays
  - Channel selection across all network layers
  - Multiple normalization methods for visualization
  - Various colormaps for different visualization preferences

### Analysis Tools
- Channel statistics and distribution visualizations
- Feature map overlays on original images
- Export options (JSON, CSV, image downloads)
- Detailed processing logs

[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
## Installation

1. Clone the repository:
```bash
git clone https://github.com/algaradi/ResVisualize.git
cd ResVisualize
```

2. Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload an image and experiment with different visualization techniques!

## Visualization Techniques Explained

### Feature Maps
Shows the actual activations of different channels in each layer when processing an input image. This helps understand what patterns each filter responds to.

### Activation Maximization
Generates synthetic images that maximally activate specific neurons, revealing what patterns or features a particular filter is looking for.

### Layer-wise Relevance Propagation (LRP)
Highlights which parts of the input image contribute most to specific activations or classifications, helping explain model decisions.

### Grad-CAM
Creates class activation maps showing which regions of an image are important for classification decisions.

## System Requirements

- Python 3.7+
- PyTorch 2.0+
- Modern web browser with JavaScript enabled
- GPU recommended for faster processing (but not required)

## Technical Details

- Backend: Flask (Python)
- Frontend: HTML, JavaScript, TailwindCSS
- Deep Learning: PyTorch, Captum
- Visualization: Matplotlib, Seaborn 
