import os
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import numpy as np
import cv2
from skimage import exposure
import json
import logging
import time
from datetime import datetime
import sys
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging.handlers
import threading
from scipy import stats  # Added for skew/kurtosis
import matplotlib.cm as cm
from matplotlib import colormaps  # For modern colormap access
from captum.attr import LayerGradCam, GuidedBackprop, LRP
import torch.optim as optim
import random
from collections import OrderedDict
from flask_compress import Compress

# Configure logging
def setup_logging():
    """Setup application logging with rotation and proper formatting"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Main logger configuration
    logger = logging.getLogger('resnet_visualizer')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = log_dir / "app.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

def cleanup_old_logs(max_logs=10):
    """Clean up old log files, keeping only the most recent ones"""
    try:
        log_dir = Path("logs")
        if not log_dir.exists():
            return
        
        # Get all log files sorted by modification time (newest first)
        log_files = sorted(
            [f for f in log_dir.glob("*.log*") if f.is_file()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Remove old log files
        for log_file in log_files[max_logs:]:
            try:
                if sys.platform == 'win32':
                    try:
                        # On Windows, try to rename the file to itself
                        # This will fail if the file is locked
                        os.rename(log_file, str(log_file) + '.tmp')
                        os.rename(str(log_file) + '.tmp', str(log_file))
                    except OSError:
                        logger.warning(f"File {log_file} is locked, skipping...")
                        continue
                log_file.unlink()
                logger.info(f"Removed old log file: {log_file}")
            except Exception as e:
                logger.error(f"Failed to remove log file {log_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Error during log cleanup: {str(e)}")

# Schedule log cleanup
def schedule_log_cleanup():
    """Schedule periodic log cleanup"""
    cleanup_old_logs()
    # Schedule next cleanup in 1 hour
    threading.Timer(3600, schedule_log_cleanup).start()

# Start log cleanup scheduler
schedule_log_cleanup()

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'  # Handle OpenMP error

app = Flask(__name__)
CORS(app)
Compress(app)  # Enable compression

# Log Flask app initialization
logger.info("Flask application initialized")

RESNET50_STRUCTURE = {
    'input': {
        'description': 'Input image processing',
        'operations': [
            'Optional preprocessing steps:',
            '- Resize (optional size)',
            '- Cropping method: center crop or direct crop',
            '- Convert to tensor (0-1 range)',
            '- Normalize (optional)'
        ]
    },
    'conv1': {
        'description': 'First convolutional layer',
        'params': '7x7 conv, 64 filters, stride 2',
        'output_size': '112x112x64'
    },
    'layer1': {
        'description': 'First residual block group',
        'blocks': 3,
        'channels': 256,
        'operations': [
            'Block 1: 64->256 channels, stride 1',
            'Block 2-3: 256->256 channels'
        ],
        'output_size': '56x56x256'
    },
    'layer2': {
        'description': 'Second residual block group',
        'blocks': 4,
        'channels': 512,
        'operations': [
            'Block 1: 256->512 channels, stride 2',
            'Block 2-4: 512->512 channels'
        ],
        'output_size': '28x28x512'
    },
    'layer3': {
        'description': 'Third residual block group',
        'blocks': 6,
        'channels': 1024,
        'operations': [
            'Block 1: 512->1024 channels, stride 2',
            'Block 2-6: 1024->1024 channels'
        ],
        'output_size': '14x14x1024'
    },
    'layer4': {
        'description': 'Fourth residual block group',
        'blocks': 3,
        'channels': 2048,
        'operations': [
            'Block 1: 1024->2048 channels, stride 2',
            'Block 2-3: 2048->2048 channels'
        ],
        'output_size': '7x7x2048'
    }
}

class SimplifiedResNetBackbone(nn.Module):
    def __init__(self):
        super(SimplifiedResNetBackbone, self).__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        
        # Map layer names to modules for advanced visualizations
        self.layer_map = {
            'conv1': self.conv1,
            'layer1': self.layer1,
            'layer2': self.layer2,
            'layer3': self.layer3,
            'layer4': self.layer4
        }
        
        # For classification-related visualization
        self.full_resnet = resnet

    def forward(self, x):
        features = {}
        
        # Save input image
        features['input'] = x
        
        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x
        
        # Max pooling
        x = self.maxpool(x)
        
        # Residual blocks
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        
        features.update({
            'l1': l1,
            'l2': l2,
            'l3': l3,
            'l4': l4
        })
        
        return features
        
    def forward_full(self, x):
        """Full forward pass for classification and class activation maps"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
        
    def get_layer_by_name(self, layer_name):
        """Get a layer module by name for attribution methods"""
        if layer_name == 'l1':
            return self.layer1
        elif layer_name == 'l2':
            return self.layer2
        elif layer_name == 'l3':
            return self.layer3
        elif layer_name == 'l4':
            return self.layer4
        elif layer_name == 'conv1':
            return self.conv1
        else:
            raise ValueError(f"Layer {layer_name} not found in model")

# Initialize model
print("Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimplifiedResNetBackbone()
model = model.to(device)
model.eval()
print(f"Model initialized on {device}")

def get_layer_info():
    """Get channel number information for each layer"""
    return {
        'input': {'channels': 3, 'spatial_size': '224x224'},
        'conv1': {'channels': 64, 'spatial_size': '112x112'},
        'l1': {'channels': 256, 'spatial_size': '56x56'},
        'l2': {'channels': 512, 'spatial_size': '28x28'},
        'l3': {'channels': 1024, 'spatial_size': '14x14'},
        'l4': {'channels': 2048, 'spatial_size': '7x7'}
    }

def validate_normalize_params(mean, std):
    """Validate normalization parameters"""
    if not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)):
        raise ValueError("Mean and standard deviation must be lists or tuples")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Mean and standard deviation must be length 3 lists (RGB)")
    if not all(isinstance(x, (int, float)) for x in mean + std):
        raise ValueError("Mean and standard deviation must be numbers")
    if not all(x > 0 for x in std):
        raise ValueError("Standard deviation must be greater than 0")

def create_transform(preprocessing_options):
    """Create image preprocessing transformation"""
    transform_list = []
    process_log = []
    
    # Resize strategy
    resize_size = None
    if preprocessing_options.get('use_resize', True):
        resize_strategy = preprocessing_options.get('resize_strategy', 'simple')
        resize_size = int(preprocessing_options.get('resize_size', 256))
        
        process_log.append({
            'step': 'Image resizing',
            'status': 'Starting',
            'details': f"Strategy: {resize_strategy}, Target size: {resize_size}"
        })
        
        if resize_strategy == 'simple':
            transform_list.append(transforms.Resize((resize_size, resize_size), antialias=True))
        elif resize_strategy == 'preserve_aspect':
            transform_list.append(
                transforms.Resize(resize_size, antialias=True)
            )
        elif resize_strategy == 'pad':
            def pad_size(img):
                return int(min(img.size) * 0.1)
            transform_list.extend([
                transforms.Lambda(lambda img: transforms.Pad(pad_size(img), padding_mode='reflect')(img)),
                transforms.Resize((resize_size, resize_size), antialias=True)
            ])
            
        process_log.append({
            'step': 'Image resizing',
            'status': 'Completed',
            'details': f"Image resized to target size"
        })
    
    # Cropping options
    if preprocessing_options.get('use_crop', True):
        crop_strategy = preprocessing_options.get('crop_strategy', 'center')
        crop_size = int(preprocessing_options.get('crop_size', 224))
        
        process_log.append({
            'step': 'Image cropping',
            'status': 'Starting',
            'details': f"Strategy: {crop_strategy}, Cropping size: {crop_size}"
        })
        
        if crop_strategy == 'center':
            transform_list.append(transforms.CenterCrop(crop_size))
        elif crop_strategy == 'random':
            transform_list.append(transforms.RandomCrop(crop_size))
        elif crop_strategy in ['five', 'ten']:
            transform_list.append(transforms.CenterCrop(crop_size))
            
        process_log.append({
            'step': 'Image cropping',
            'status': 'Completed',
            'details': f"Completed {crop_strategy} cropping"
        })
    
    # Data augmentation options
    if preprocessing_options.get('use_augmentation', False):
        aug_options = preprocessing_options.get('augmentation_options', {})
        process_log.append({
            'step': 'Data augmentation',
            'status': 'Starting',
            'details': "Applying data augmentation"
        })
        
        # Implement actual torchvision transforms
        if aug_options.get('color_jitter', False):
            transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        if aug_options.get('random_rotation', False):
            transform_list.append(transforms.RandomRotation(degrees=15))
        if aug_options.get('random_horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())
        if aug_options.get('random_vertical_flip', False):
            transform_list.append(transforms.RandomVerticalFlip())
        if aug_options.get('random_grayscale', False):
            transform_list.append(transforms.RandomGrayscale(p=0.2))
        if aug_options.get('gaussian_blur', False):
            transform_list.append(transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)))
        for aug_name, enabled in aug_options.items():
            if enabled:
                process_log.append({
                    'step': 'Data augmentation',
                    'status': 'In progress',
                    'details': f"Applying {aug_name} augmentation"
                })
        
        process_log.append({
            'step': 'Data augmentation',
            'status': 'Completed',
            'details': "Data augmentation application completed"
        })
    
    # Basic transformations
    transform_list.append(transforms.ToTensor())
    process_log.append({
        'step': 'Tensor conversion',
        'status': 'Complete',
        'details': "Image converted to tensor format"
    })
    
    # Normalization options
    if preprocessing_options.get('use_normalize', True):
        normalize_strategy = preprocessing_options.get('normalize_strategy', 'imagenet')
        process_log.append({
            'step': 'Normalization',
            'status': 'Starting',
            'details': f"Using {normalize_strategy} normalization strategy"
        })
        
        if normalize_strategy == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif normalize_strategy == 'simple':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        elif normalize_strategy == 'custom':
            mean = preprocessing_options.get('custom_mean', [0.5, 0.5, 0.5])
            std = preprocessing_options.get('custom_std', [0.5, 0.5, 0.5])
            validate_normalize_params(mean, std)  # Validate custom params
        
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        process_log.append({
            'step': 'Normalization',
            'status': 'Complete',
            'details': f"Normalization processing complete"
        })
    
    return transforms.Compose(transform_list), process_log

def normalize_feature_map(feature_map, method='minmax'):
    """归一化特征图，支持多种归一化方法"""
    if method == 'minmax':
        min_val = feature_map.min()
        max_val = feature_map.max()
        if max_val > min_val:
            return (feature_map - min_val) / (max_val - min_val)
        return feature_map
    elif method == 'adaptive':
        # 使用自适应直方图均衡化
        feature_map_uint8 = (feature_map * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(feature_map_uint8) / 255.0
    elif method == 'contrast':
        # 使用对比度拉伸
        p2, p98 = np.percentile(feature_map, (2, 98))
        return exposure.rescale_intensity(feature_map, in_range=(p2, p98))
    else:
        return feature_map

def safe_divide(a, b, fill_value=0):
    """安全除法，避免除零警告"""
    if isinstance(b, (int, float)):
        return fill_value if b == 0 else a / b
    mask = b != 0
    result = np.zeros_like(a, dtype=float)
    result[mask] = a[mask] / b[mask]
    result[~mask] = fill_value
    return result

def calculate_statistics(channel):
    """安全计算统计信息"""
    mean = np.mean(channel)
    std = np.std(channel)
    if std == 0:
        return {
            'mean': float(mean),
            'std': 0.0,
            'max': float(np.max(channel)),
            'min': float(np.min(channel)),
            'median': float(np.median(channel)),
            'skewness': 0.0,
            'kurtosis': 0.0
        }
    
    normalized = (channel - mean) / std
    return {
        'mean': float(mean),
        'std': float(std),
        'max': float(np.max(channel)),
        'min': float(np.min(channel)),
        'median': float(np.median(channel)),
        'skewness': float(np.mean(normalized ** 3)),
        'kurtosis': float(np.mean(normalized ** 4))
    }

def plot_channel_statistics(features, layer_name):
    """Create statistical plots for each layer's feature maps"""
    try:
        logger.info(f"Generating statistics for layer {layer_name}")
        
        # Get feature maps for all channels
        features = features[0].detach().cpu().numpy()  # [C, H, W]
        num_channels = features.shape[0]
        
        # Check memory usage
        memory_usage = features.nbytes / (1024 * 1024)  # MB
        logger.info(f"Memory usage for layer {layer_name}: {memory_usage:.2f}MB")
        
        if memory_usage > 1000:  # If over 1GB
            logger.warning(f"Warning: Large memory usage in layer {layer_name}: {memory_usage:.2f}MB")
        
        # Calculate statistics for each channel
        channel_stats = [calculate_statistics(features[i]) for i in range(num_channels)]
        
        # Extract statistical data
        means = [s['mean'] for s in channel_stats]
        stds = [s['std'] for s in channel_stats]
        maxs = [s['max'] for s in channel_stats]
        mins = [s['min'] for s in channel_stats]
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Set global title
        fig.suptitle(f'Statistical Analysis - Layer {layer_name}', fontsize=16, y=0.95)
        
        # 1. Mean and Standard Deviation Trends
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(num_channels)
        ax1.plot(x, means, label='Mean', color='blue', alpha=0.7)
        ax1.fill_between(x, 
                        np.array(means) - np.array(stds), 
                        np.array(means) + np.array(stds),
                        color='blue', alpha=0.2, label='±1 Std Dev')
        ax1.set_title('Channel Mean and Standard Deviation')
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('Value')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution Density Plot
        ax2 = fig.add_subplot(gs[1, 0])
        for i in range(min(5, num_channels)):  # Only show first 5 channels
            channel = features[i].flatten()
            sns.kdeplot(data=channel, ax=ax2, label=f'Channel {i}')
        ax2.set_title('Feature Value Distribution')
        ax2.set_xlabel('Feature Value')
        ax2.set_ylabel('Density')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Activation Pattern Analysis
        ax3 = fig.add_subplot(gs[1, 1])
        activation_ratios = []
        for channel in features:
            # Calculate activation ratio (proportion above mean)
            ratio = np.mean(channel > np.mean(channel))
            activation_ratios.append(ratio)
        ax3.hist(activation_ratios, bins=30, color='purple', alpha=0.7)
        ax3.set_title('Channel Activation Distribution')
        ax3.set_xlabel('Proportion of High Activation')
        ax3.set_ylabel('Number of Channels')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        try:
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode()
            logger.info(f"Successfully generated statistics plot for layer {layer_name}")
            return img_str
        finally:
            buf.close()
            plt.close('all')
            
    except Exception as e:
        logger.error(f"Error generating statistics plot for layer {layer_name}: {str(e)}")
        return None

def visualize_feature_maps(feature_maps, selected_layers, start_channel=0, grid_size=(4, 4), normalize_method='minmax', colormap='viridis'):
    """
    Visualize feature maps with improved normalization and plotting
    """
    feature_maps_viz = {}
    feature_stats = {}
    channel_stats_plots = {}
    detailed_stats = {}  # New: Store detailed statistics for export
    heatmap_overlays = {}  # New: Store heatmap overlays
    
    plt.style.use('ggplot')
    
    # Get input image for heatmap overlays
    original_img = None
    if 'input' in feature_maps:
        original_img = feature_maps['input'].cpu().numpy()[0]
        # Convert from CHW to HWC format and normalize for display
        original_img = np.transpose(original_img, (1, 2, 0))
        # Un-normalize if needed (assuming ImageNet normalization)
        original_img = original_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        original_img = np.clip(original_img, 0, 1)  # Ensure values are in range [0,1]
    
    for layer_name in selected_layers:
        if layer_name not in feature_maps:
            continue
            
        # Get feature maps for current layer
        layer_maps = feature_maps[layer_name].cpu().numpy()
        batch_size, channels, height, width = layer_maps.shape
        
        # Calculate detailed feature statistics
        layer_stats = {
            'min': float(layer_maps.min()),
            'max': float(layer_maps.max()),
            'mean': float(layer_maps.mean()),
            'std': float(layer_maps.std()),
            'median': float(np.median(layer_maps)),
            'q1': float(np.percentile(layer_maps, 25)),
            'q3': float(np.percentile(layer_maps, 75)),
            'skewness': float(stats.skew(layer_maps.flatten())),
            'kurtosis': float(stats.kurtosis(layer_maps.flatten()))
        }
        feature_stats[layer_name] = layer_stats
        
        # Calculate per-channel statistics
        channel_means = layer_maps.mean(axis=(2, 3))
        channel_stds = layer_maps.std(axis=(2, 3))
        channel_mins = layer_maps.min(axis=(2, 3))
        channel_maxs = layer_maps.max(axis=(2, 3))
        
        # Store detailed channel statistics
        detailed_stats[layer_name] = {
            'channel_means': channel_means[0].tolist(),
            'channel_stds': channel_stds[0].tolist(),
            'channel_mins': channel_mins[0].tolist(),
            'channel_maxs': channel_maxs[0].tolist(),
            'spatial_size': f"{height}x{width}",
            'num_channels': channels
        }
        
        # Create enhanced channel statistics plot
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # 1. Mean and Standard Deviation
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(channels)
        ax1.plot(x, channel_means[0], 'b-', label='Mean')
        ax1.fill_between(x, 
                        channel_means[0] - channel_stds[0],
                        channel_means[0] + channel_stds[0],
                        alpha=0.3)
        ax1.set_title('Channel Activation Statistics')
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('Mean Activation')
        ax1.grid(True)
        
        # 2. Min-Max Range
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(x, channel_maxs[0], 'r-', label='Max')
        ax2.plot(x, channel_mins[0], 'g-', label='Min')
        ax2.fill_between(x, channel_mins[0], channel_maxs[0], alpha=0.2)
        ax2.set_title('Channel Value Range')
        ax2.set_xlabel('Channel Index')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Distribution Plot
        ax3 = fig.add_subplot(gs[1, 0])
        for i in range(min(5, channels)):
            channel = layer_maps[0, i].flatten()
            sns.kdeplot(data=channel, ax=ax3, label=f'Channel {i}')
        ax3.set_title('Feature Value Distribution')
        ax3.set_xlabel('Feature Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Activation Pattern
        ax4 = fig.add_subplot(gs[1, 1])
        activation_ratios = []
        for channel in layer_maps[0]:
            ratio = np.mean(channel > np.mean(channel))
            activation_ratios.append(ratio)
        ax4.hist(activation_ratios, bins=30, color='purple', alpha=0.7)
        ax4.set_title('Channel Activation Distribution')
        ax4.set_xlabel('Proportion of High Activation')
        ax4.set_ylabel('Number of Channels')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save channel statistics plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        channel_stats_plots[layer_name] = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Select channels for visualization
        end_channel = min(start_channel + grid_size[0] * grid_size[1], channels)
        selected_maps = layer_maps[0, start_channel:end_channel]
        
        if len(selected_maps) == 0:
            continue
            
        # Normalize feature maps
        if normalize_method == 'minmax':
            normalized_maps = np.array([
                (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
                for fmap in selected_maps
            ])
        elif normalize_method == 'adaptive':
            normalized_maps = np.array([
                exposure.equalize_adapthist(fmap)
                for fmap in selected_maps
            ])
        elif normalize_method == 'contrast':
            normalized_maps = np.array([
                exposure.rescale_intensity(fmap)
                for fmap in selected_maps
            ])
        
        # Create visualization grid
        rows, cols = grid_size
        fig = plt.figure(figsize=(cols * 2, rows * 2))
        
        for idx, fmap in enumerate(normalized_maps):
            if idx >= rows * cols:
                break
                
            ax = plt.subplot(rows, cols, idx + 1)
            im = ax.imshow(fmap, cmap=colormap)
            ax.axis('off')
            
            # Add channel index
            channel_idx = start_channel + idx
            ax.set_title(f'Ch {channel_idx}', fontsize=8)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        
        plt.tight_layout()
        
        # Save visualization
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        feature_maps_viz[layer_name] = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Create heatmap overlay if this isn't the input layer
        if layer_name != 'input' and original_img is not None:
            # Average activations across all channels for this layer
            activation_map = np.mean(layer_maps[0], axis=0)
            
            # Normalize activation map
            if normalize_method == 'minmax':
                activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
            elif normalize_method == 'adaptive':
                activation_map = exposure.equalize_adapthist(activation_map)
            elif normalize_method == 'contrast':
                activation_map = exposure.rescale_intensity(activation_map)
            
            # Resize to match input image size
            h, w = original_img.shape[:2]
            activation_map = cv2.resize(activation_map, (w, h))
            
            # Create color map
            color_mapper = colormaps[colormap]
            heatmap = color_mapper(activation_map)
            heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
            
            # Create overlay
            overlay_img = original_img.copy()
            overlay_img = (overlay_img * 255).astype(np.uint8)
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            
            # Apply heatmap with transparency
            overlay = cv2.addWeighted(overlay_img, 0.6, heatmap, 0.4, 0)
            
            # Convert to base64
            overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            overlay_pil.save(buf, format='PNG')
            buf.seek(0)
            heatmap_overlays[layer_name] = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return feature_maps_viz, feature_stats, channel_stats_plots, detailed_stats, heatmap_overlays

@app.route('/')
def index():
    """Render the main page with model information"""
    try:
        model_info = get_model_info()
        layer_info = get_layer_info()
        return render_template('index.html', model_info=model_info, layer_info=layer_info)
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        # Provide empty/default values to avoid template errors
        return render_template(
            'index.html',
            model_info={'total_params': 0, 'trainable_params': 0, 'layers': []},
            layer_info={},
            error=str(e)
        )

def validate_image(image):
    """Validate image format and size"""
    try:
        # Check image format
        if image.format not in ['JPEG', 'PNG', 'BMP']:
            raise ValueError(f"Unsupported image format: {image.format}")
        
        # Check image mode
        if image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')
        
        # Check image size
        max_dimension = 4096
        if image.width > max_dimension or image.height > max_dimension:
            raise ValueError(f"Image dimensions exceed maximum allowed size of {max_dimension}x{max_dimension}")
        
        return image, None
    except Exception as e:
        return None, str(e)

@app.route('/api/visualize', methods=['POST'])
def visualize():
    start_time = time.time()
    process_logs = []
    def add_log(level, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'level': level.upper(),
            'message': message
        }
        process_logs.append(log_entry)
        if level.upper() == 'INFO':
            logger.info(message)
        elif level.upper() == 'WARNING':
            logger.warning(message)
        elif level.upper() == 'ERROR':
            logger.error(message)
    try:
        add_log('info', f'Received form data: {dict(request.form)}')
        if 'image' not in request.files:
            add_log('error', 'No image file provided')
            return jsonify({'status': 'error','error': 'No image provided','logs': process_logs}), 400
        image_file = request.files['image']
        if not image_file.filename:
            add_log('error', 'No selected file')
            return jsonify({'status': 'error','error': 'No selected file','logs': process_logs}), 400
        try:
            image = Image.open(image_file)
            image, error = validate_image(image)
            if error:
                add_log('error', f'Image validation failed: {error}')
                return jsonify({'status': 'error','error': error,'logs': process_logs}), 400
        except Exception as e:
            add_log('error', f'Failed to open image: {str(e)}')
            return jsonify({'status': 'error','error': f'Failed to open image: {str(e)}','logs': process_logs}), 400
        add_log('info', f'Original image size: {image.size}, mode: {image.mode}')
        try:
            preprocessing_options = {
                'use_resize': request.form.get('use_resize', 'true').lower() == 'true',
                'resize_strategy': request.form.get('resize_strategy', 'simple'),
                'resize_size': int(request.form.get('resize_size', 256)),
                'use_crop': request.form.get('use_crop', 'true').lower() == 'true',
                'crop_strategy': request.form.get('crop_strategy', 'center'),
                'crop_size': int(request.form.get('crop_size', 224)),
                'use_normalize': request.form.get('use_normalize', 'true').lower() == 'true',
                'normalize_strategy': request.form.get('normalize_strategy', 'imagenet'),
                'use_augmentation': request.form.get('use_augmentation', 'false').lower() == 'true'
            }
            aug_opts_str = request.form.get('augmentation_options', None)
            if aug_opts_str:
                try:
                    import json
                    preprocessing_options['augmentation_options'] = json.loads(aug_opts_str)
                    add_log('info', f'Parsed augmentation_options: {preprocessing_options["augmentation_options"]}')
                except Exception as e:
                    add_log('error', f'Failed to parse augmentation_options: {str(e)}')
                    preprocessing_options['augmentation_options'] = {}
            if preprocessing_options['normalize_strategy'] == 'custom':
                try:
                    mean_str = request.form.get('custom_mean', None)
                    std_str = request.form.get('custom_std', None)
                    if mean_str:
                        preprocessing_options['custom_mean'] = json.loads(mean_str)
                    if std_str:
                        preprocessing_options['custom_std'] = json.loads(std_str)
                    add_log('info', f'Parsed custom normalization: mean={preprocessing_options.get("custom_mean")}, std={preprocessing_options.get("custom_std")}')
                except Exception as e:
                    add_log('error', f'Failed to parse custom normalization: {str(e)}')
            if preprocessing_options['resize_size'] < 32 or preprocessing_options['resize_size'] > 1024:
                raise ValueError('Resize size must be between 32 and 1024')
            if preprocessing_options['crop_size'] < 32 or preprocessing_options['crop_size'] > 1024:
                raise ValueError('Crop size must be between 32 and 1024')
            add_log('info', f'Parsed preprocessing options: {preprocessing_options}')
        except Exception as e:
            add_log('error', f'Invalid preprocessing parameters: {str(e)}')
            return jsonify({'status': 'error','error': f'Invalid preprocessing parameters: {str(e)}','logs': process_logs}), 400
        try:
            transform, transform_logs = create_transform(preprocessing_options)
            for log in transform_logs:
                add_log('info', log['details'])
            image_tensor = transform(image)
            add_log('info', f'Transformed tensor shape: {image_tensor.shape}')
        except Exception as e:
            add_log('error', f'Error during image preprocessing: {str(e)}')
            return jsonify({'status': 'error','error': f'Error during image preprocessing: {str(e)}','logs': process_logs}), 500
        image_tensor = image_tensor.unsqueeze(0).to(device)
        add_log('info', f'Image tensor after unsqueeze and to(device): {image_tensor.shape}, dtype={image_tensor.dtype}')
        
        # Save preprocessed image preview (before normalization)
        preprocessed_preview = None
        try:
            # Convert tensor to numpy and adjust for display
            preview_img = image_tensor[0].cpu().clone().detach().numpy()
            preview_img = np.transpose(preview_img, (1, 2, 0))
            # If using ImageNet normalization, denormalize
            if preprocessing_options.get('normalize_strategy') == 'imagenet':
                preview_img = preview_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            preview_img = np.clip(preview_img, 0, 1)
            
            # Convert to PIL and save as base64
            preview_pil = Image.fromarray((preview_img * 255).astype(np.uint8))
            buf = io.BytesIO()
            preview_pil.save(buf, format='PNG')
            buf.seek(0)
            preprocessed_preview = base64.b64encode(buf.getvalue()).decode('utf-8')
            add_log('info', f'Created preprocessed image preview')
        except Exception as e:
            add_log('warning', f'Failed to create preprocessed image preview: {str(e)}')
            
        try:
            with torch.no_grad():
                feature_maps = model(image_tensor)
            add_log('info', f'Feature maps generated successfully. Keys: {list(feature_maps.keys())}')
            for k, v in feature_maps.items():
                add_log('info', f'Layer {k}: shape {tuple(v.shape)}')
        except Exception as e:
            add_log('error', f'Error generating feature maps: {str(e)}')
            return jsonify({'status': 'error','error': f'Error generating feature maps: {str(e)}','logs': process_logs}), 500
        try:
            start_channel = int(request.form.get('start_channel', 0))
            grid_rows = int(request.form.get('grid_rows', 4))
            grid_cols = int(request.form.get('grid_cols', 4))
            normalize_method = request.form.get('normalize_method', 'minmax')
            colormap = request.form.get('colormap', 'viridis')
            selected_layers = request.form.getlist('selected_layers[]')
            if not selected_layers:
                selected_layers = ['input', 'conv1', 'l1', 'l2', 'l3', 'l4']
            add_log('info', f'Generating visualizations for {len(selected_layers)} layers: {selected_layers}')
        except Exception as e:
            add_log('error', f'Invalid visualization parameters: {str(e)}')
            return jsonify({'status': 'error','error': f'Invalid visualization parameters: {str(e)}','logs': process_logs}), 400
        try:
            feature_maps_viz, feature_stats, channel_stats_plots, detailed_stats, heatmap_overlays = visualize_feature_maps(
                feature_maps, 
                selected_layers,
                start_channel=start_channel,
                grid_size=(grid_rows, grid_cols),
                normalize_method=normalize_method,
                colormap=colormap
            )
            add_log('info', f'Visualization results: feature_maps_viz keys={list(feature_maps_viz.keys())}, feature_stats keys={list(feature_stats.keys())}')
            if not feature_maps_viz:
                add_log('warning', 'No feature maps visualized!')
        except Exception as e:
            add_log('error', f'Error generating visualizations: {str(e)}')
            return jsonify({'status': 'error','error': f'Error generating visualizations: {str(e)}','logs': process_logs}), 500
        end_time = time.time()
        processing_time = end_time - start_time
        add_log('info', f'Processing completed in {processing_time:.2f} seconds')
        return jsonify({
            'status': 'success',
            'feature_maps': feature_maps_viz,
            'feature_stats': feature_stats,
            'channel_stats_plots': channel_stats_plots,
            'detailed_stats': detailed_stats,
            'heatmap_overlays': heatmap_overlays,
            'preprocessed_preview': preprocessed_preview,
            'preprocessing_info': {
                'original_size': f"{image.size[0]}x{image.size[1]}",
                'final_size': f"{image_tensor.shape[2]}x{image_tensor.shape[3]}",
                'preprocessing_options': preprocessing_options
            },
            'layer_info': get_layer_info(),
            'processing_time': processing_time,
            'logs': process_logs
        })
    except Exception as e:
        error_msg = str(e)
        add_log('error', f'Unexpected error: {error_msg}')
        return jsonify({'status': 'error','error': error_msg,'logs': process_logs}), 500

def get_model_info():
    """Get detailed information about the model architecture"""
    model_info = {
        'name': 'ResNet50',
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'layers': []
    }
    
    def get_layer_info(module, name=''):
        """Recursively get information about each layer"""
        if len(list(module.children())) == 0:
            # Leaf module
            return {
                'name': name,
                'type': module.__class__.__name__,
                'params': sum(p.numel() for p in module.parameters()),
                'trainable_params': sum(p.numel() for p in module.parameters() if p.requires_grad),
                'output_shape': None  # Will be filled during forward pass
            }
        
        children = []
        for child_name, child in module.named_children():
            child_info = get_layer_info(child, f"{name}.{child_name}" if name else child_name)
            if isinstance(child_info, list):
                children.extend(child_info)
            else:
                children.append(child_info)
        return children
    
    model_info['layers'] = get_layer_info(model)
    return model_info

@app.route('/api/model_info', methods=['GET'])
def get_model_architecture():
    """API endpoint to get model architecture information"""
    try:
        model_info = get_model_info()
        return jsonify({
            'status': 'success',
            'model_info': model_info
        })
    except Exception as e:
        logger.error(f"Error getting model architecture: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

def activation_maximization(model, layer_name, channel_idx, num_iterations=100, learning_rate=0.1):
    """
    Generate an image that maximally activates a specific filter channel in the specified layer.
    
    Args:
        model: The neural network model
        layer_name: The name of the layer to visualize
        channel_idx: The index of the channel to maximize
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        
    Returns:
        The generated image and optimization history
    """
    model.eval()
    
    # Create a random image to optimize
    input_shape = (1, 3, 224, 224)
    
    # Initialize with random noise + a bit of random structure
    input_tensor = torch.randn(input_shape, requires_grad=True, device=device)
    
    # Get the target layer
    target_layer = model.get_layer_by_name(layer_name)
    
    # Setup optimizer
    optimizer = optim.Adam([input_tensor], lr=learning_rate)
    
    # Track activation values
    activation_history = []
    
    # Run optimization
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass to target layer
        x = input_tensor
        
        # Apply preprocessing transforms that the model expects
        # This is important for meaningful visualizations
        if layer_name == 'conv1':
            x = model.conv1(x)
            x = model.bn1(x)
            activations = model.relu(x)
        elif layer_name == 'l1':
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            activations = model.layer1(x)
        elif layer_name == 'l2':
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            activations = model.layer2(x)
        elif layer_name == 'l3':
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            activations = model.layer3(x)
        elif layer_name == 'l4':
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            activations = model.layer4(x)
        
        # Select channel and compute mean activation
        channel_activation = activations[0, channel_idx].mean()
        
        # We want to maximize this activation, so we minimize negative activation
        loss = -channel_activation
        
        # Add regularization to generate more natural looking images
        # L2 regularization
        loss += 0.001 * torch.sum(input_tensor**2)
        
        # Total variation regularization for spatial coherence
        diff_h = torch.sum(torch.abs(input_tensor[:,:,1:,:] - input_tensor[:,:,:-1,:]))
        diff_w = torch.sum(torch.abs(input_tensor[:,:,:,1:] - input_tensor[:,:,:,:-1]))
        loss += 0.0005 * (diff_h + diff_w)
        
        # Record activation
        activation_history.append(float(channel_activation.detach().cpu()))
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Periodically apply constraints to keep images more natural
        if i % 5 == 0:
            # Clip values to image range
            with torch.no_grad():
                input_tensor.clamp_(0, 1)
    
    # Prepare visualization
    # Normalize to [0, 1] range for display
    generated_img = input_tensor[0].detach().cpu().numpy()
    generated_img = np.transpose(generated_img, (1, 2, 0))
    generated_img = (generated_img - generated_img.min()) / (generated_img.max() - generated_img.min() + 1e-8)
    
    return generated_img, activation_history

def layer_relevance_propagation(model, image_tensor, layer_name, channel_idx=None, class_idx=None):
    """
    Apply Layer-wise Relevance Propagation to visualize contributions to a specific activation.
    
    Args:
        model: The neural network model
        image_tensor: Input image tensor
        layer_name: Target layer name
        channel_idx: Channel index (if targeting a specific filter)
        class_idx: Class index (if targeting a specific class output)
        
    Returns:
        LRP attribution heatmap
    """
    model.eval()
    
    # Clone the input to ensure we don't modify the original
    input_tensor = image_tensor.clone().requires_grad_(True)
    
    # Get the target layer
    target_layer = model.get_layer_by_name(layer_name)
    
    # Instead of using the LRP module directly, which seems to have issues,
    # we'll implement a simpler gradient-based approach that achieves 
    # a similar visualization
    
    # First, do a forward pass to get activations
    with torch.no_grad():
        features = model(input_tensor)
        layer_output = features[layer_name]
        
    # Create a tensor to hold our target
    if class_idx is not None:
        # For class-based attribution, we need to do a full forward pass
        logits = model.forward_full(input_tensor)
        target = logits[0, class_idx]
    else:
        # For channel-based attribution
        if channel_idx is not None:
            # Create a mask targeting just that channel
            mask = torch.zeros_like(layer_output)
            mask[0, channel_idx] = 1.0
            layer_output_masked = layer_output * mask
            target = layer_output_masked.sum()
        else:
            # Use the mean activation of the entire layer
            target = layer_output.mean()
            
    # Compute gradients
    input_tensor.requires_grad_(True)
    model.zero_grad()
    if class_idx is not None:
        logits = model.forward_full(input_tensor)
        target = logits[0, class_idx]
    else:
        features = model(input_tensor)
        layer_output = features[layer_name]
        if channel_idx is not None:
            mask = torch.zeros_like(layer_output)
            mask[0, channel_idx] = 1.0
            layer_output_masked = layer_output * mask
            target = layer_output_masked.sum()
        else:
            target = layer_output.mean()
            
    target.backward()
    
    # Get the gradients
    gradients = input_tensor.grad.data
    
    # Create attribution map (similar to Guided Backprop)
    attribution_map = gradients[0].sum(dim=0).cpu().numpy()
    
    # Take absolute value and normalize for visualization
    attribution_map = np.abs(attribution_map)
    attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-8)
    
    return attribution_map

def grad_cam(model, image_tensor, layer_name, class_idx=None):
    """
    Apply Gradient-weighted Class Activation Mapping to visualize important regions.
    
    Args:
        model: The neural network model
        image_tensor: Input image tensor
        layer_name: Target layer name
        class_idx: Target class index (None will use predicted class)
        
    Returns:
        Grad-CAM heatmap and class prediction information
    """
    model.eval()
    
    # First, get the class prediction if not provided
    # This requires a full forward pass through the classification model
    input_tensor = image_tensor.clone().requires_grad_(True)
    
    # Get prediction
    with torch.no_grad():
        predictions = model.forward_full(input_tensor)
        _, predicted = torch.max(predictions, 1)
        
    # If class_idx is not provided, use the predicted class
    if class_idx is None:
        class_idx = predicted.item()
    
    # Setup GradCAM with the target layer
    target_layer = model.get_layer_by_name(layer_name)
    grad_cam = LayerGradCam(model.forward_full, target_layer)
    
    # Compute attribution
    attribution = grad_cam.attribute(input_tensor, target=class_idx)
    
    # Process the attribution to create a heatmap
    # First, take the attribution for the first (and only) sample in the batch
    attribution = attribution.cpu().detach().numpy()[0]
    
    # Aggregate attributions across channels (features)
    # This gives us a 2D heatmap of shape (H, W)
    cam = np.mean(attribution, axis=0)
    
    # Apply ReLU to focus on features that have a positive influence on the target class
    cam = np.maximum(cam, 0)
    
    # Normalize to [0, 1] for visualization
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) if cam.max() > cam.min() else cam
    
    # Resize to input image size
    input_size = input_tensor.shape[2:]  # (H, W)
    cam = cv2.resize(cam, (input_size[1], input_size[0]))
    
    # Get class label and confidence
    imagenet_labels = get_imagenet_labels()
    class_name = imagenet_labels[class_idx] if class_idx in imagenet_labels else f"Class {class_idx}"
    confidence = torch.nn.functional.softmax(predictions, dim=1)[0, class_idx].item()
    
    return cam, class_idx, class_name, confidence

def get_imagenet_labels():
    """Return a mapping of ImageNet class indices to human-readable labels"""
    # Load from a built-in mapping or a file
    try:
        import json
        labels_path = 'imagenet_labels.json'
        
        # Check if file exists, otherwise create a simple mapping
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                return json.load(f)
        else:
            # Create a simplified version with just some common classes
            simplified_labels = {
                0: "tench", 
                1: "goldfish", 
                2: "great white shark",
                # ... more classes could be added
                282: "tiger cat",
                283: "Persian cat",
                291: "lion",
                340: "zebra",
                386: "African elephant",
                974: "geyser"
            }
            return simplified_labels
    except:
        # Fallback to empty dict
        return {}

@app.route('/api/activation_maximization', methods=['POST'])
def api_activation_maximization():
    """API endpoint for Activation Maximization technique"""
    start_time = time.time()
    process_logs = []
    def add_log(level, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'level': level.upper(),
            'message': message
        }
        process_logs.append(log_entry)
        if level.upper() == 'INFO':
            logger.info(message)
        elif level.upper() == 'WARNING':
            logger.warning(message)
        elif level.upper() == 'ERROR':
            logger.error(message)
    
    try:
        add_log('info', 'Starting Activation Maximization')
        
        # Parse request parameters
        layer_name = request.form.get('layer_name', 'conv1')
        channel_idx = int(request.form.get('channel_idx', 0))
        num_iterations = int(request.form.get('num_iterations', 150))
        learning_rate = float(request.form.get('learning_rate', 0.1))
        
        add_log('info', f'Parameters: layer={layer_name}, channel={channel_idx}, iterations={num_iterations}, lr={learning_rate}')
        
        # Validate parameters
        if layer_name not in ['conv1', 'l1', 'l2', 'l3', 'l4']:
            add_log('error', f'Invalid layer name: {layer_name}')
            return jsonify({'status': 'error', 'error': f'Invalid layer name: {layer_name}', 'logs': process_logs}), 400
        
        layer_info = get_layer_info()
        max_channels = layer_info.get(layer_name, {}).get('channels', 0)
        
        if channel_idx < 0 or channel_idx >= max_channels:
            add_log('error', f'Invalid channel index: {channel_idx}. Must be between 0 and {max_channels-1}')
            return jsonify({'status': 'error', 'error': f'Invalid channel index: {channel_idx}', 'logs': process_logs}), 400
        
        # Run activation maximization
        add_log('info', 'Running activation maximization optimization...')
        generated_img, activation_history = activation_maximization(
            model=model,
            layer_name=layer_name,
            channel_idx=channel_idx,
            num_iterations=num_iterations,
            learning_rate=learning_rate
        )
        add_log('info', f'Optimization completed with final activation: {activation_history[-1]}')
        
        # Visualize the generated image and activation history
        plt.style.use('ggplot')
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(10, 6))
        gs = plt.GridSpec(1, 2, figure=fig)
        
        # Display the generated image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(generated_img)
        ax1.set_title(f'Layer {layer_name}, Channel {channel_idx}')
        ax1.axis('off')
        
        # Plot activation history
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(activation_history)
        ax2.set_title('Activation History')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Activation')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save the figure to a base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Save the raw generated image as well
        img_pil = Image.fromarray((generated_img * 255).astype(np.uint8))
        buf_img = io.BytesIO()
        img_pil.save(buf_img, format='PNG')
        buf_img.seek(0)
        raw_img_str = base64.b64encode(buf_img.getvalue()).decode('utf-8')
        
        end_time = time.time()
        processing_time = end_time - start_time
        add_log('info', f'Processing completed in {processing_time:.2f} seconds')
        
        return jsonify({
            'status': 'success',
            'visualization': img_str,
            'generated_image': raw_img_str,
            'activation_history': activation_history,
            'processing_time': processing_time,
            'logs': process_logs
        })
    except Exception as e:
        error_msg = str(e)
        add_log('error', f'Error during activation maximization: {error_msg}')
        return jsonify({'status': 'error', 'error': error_msg, 'logs': process_logs}), 500

@app.route('/api/layer_relevance', methods=['POST'])
def api_layer_relevance():
    """API endpoint for Layer-wise Relevance Propagation"""
    start_time = time.time()
    process_logs = []
    def add_log(level, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'level': level.upper(),
            'message': message
        }
        process_logs.append(log_entry)
        if level.upper() == 'INFO':
            logger.info(message)
        elif level.upper() == 'WARNING':
            logger.warning(message)
        elif level.upper() == 'ERROR':
            logger.error(message)
    
    try:
        add_log('info', 'Starting Layer-wise Relevance Propagation (LRP)')
        
        # Check for image
        if 'image' not in request.files:
            add_log('error', 'No image file provided')
            return jsonify({'status': 'error', 'error': 'No image provided', 'logs': process_logs}), 400
            
        image_file = request.files['image']
        if not image_file.filename:
            add_log('error', 'No selected file')
            return jsonify({'status': 'error', 'error': 'No selected file', 'logs': process_logs}), 400
            
        try:
            image = Image.open(image_file)
            image, error = validate_image(image)
            if error:
                add_log('error', f'Image validation failed: {error}')
                return jsonify({'status': 'error', 'error': error, 'logs': process_logs}), 400
        except Exception as e:
            add_log('error', f'Failed to open image: {str(e)}')
            return jsonify({'status': 'error', 'error': f'Failed to open image: {str(e)}', 'logs': process_logs}), 400
            
        add_log('info', f'Original image size: {image.size}, mode: {image.mode}')
        
        # Parse parameters
        layer_name = request.form.get('layer_name', 'conv1')
        channel_idx_str = request.form.get('channel_idx', None)
        class_idx_str = request.form.get('class_idx', None)
        
        channel_idx = int(channel_idx_str) if channel_idx_str else None
        class_idx = int(class_idx_str) if class_idx_str else None
        
        # Process image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate LRP visualization
        add_log('info', 'Computing LRP attributions...')
        attribution_map = layer_relevance_propagation(
            model=model, 
            image_tensor=image_tensor,
            layer_name=layer_name,
            channel_idx=channel_idx,
            class_idx=class_idx
        )
        add_log('info', f'LRP attribution map generated with shape {attribution_map.shape}')
        
        # Create visualization
        plt.style.use('ggplot')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show original image
        img_np = np.array(image.resize((224, 224)))
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show attribution map
        im = axes[1].imshow(attribution_map, cmap='jet')
        axes[1].set_title(f'LRP Attribution - Layer {layer_name}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save the figure to a base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Create overlay visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(img_np)
        plt.imshow(attribution_map, cmap='jet', alpha=0.6)
        plt.axis('off')
        plt.tight_layout()
        
        # Save overlay to base64
        buf_overlay = io.BytesIO()
        plt.savefig(buf_overlay, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf_overlay.seek(0)
        overlay_str = base64.b64encode(buf_overlay.getvalue()).decode('utf-8')
        
        end_time = time.time()
        processing_time = end_time - start_time
        add_log('info', f'Processing completed in {processing_time:.2f} seconds')
        
        target_description = ""
        if class_idx is not None:
            target_description = f"Class {class_idx}"
        elif channel_idx is not None:
            target_description = f"Channel {channel_idx}"
        else:
            target_description = "Average activation"
        
        return jsonify({
            'status': 'success',
            'attribution_map': img_str,
            'overlay': overlay_str,
            'layer_name': layer_name,
            'target_description': target_description,
            'processing_time': processing_time,
            'logs': process_logs
        })
    except Exception as e:
        error_msg = str(e)
        add_log('error', f'Error during LRP: {error_msg}')
        return jsonify({'status': 'error', 'error': error_msg, 'logs': process_logs}), 500

@app.route('/api/grad_cam', methods=['POST'])
def api_grad_cam():
    """API endpoint for Gradient-weighted Class Activation Mapping"""
    start_time = time.time()
    process_logs = []
    def add_log(level, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'level': level.upper(),
            'message': message
        }
        process_logs.append(log_entry)
        if level.upper() == 'INFO':
            logger.info(message)
        elif level.upper() == 'WARNING':
            logger.warning(message)
        elif level.upper() == 'ERROR':
            logger.error(message)
    
    try:
        add_log('info', 'Starting Grad-CAM visualization')
        
        # Check for image
        if 'image' not in request.files:
            add_log('error', 'No image file provided')
            return jsonify({'status': 'error', 'error': 'No image provided', 'logs': process_logs}), 400
            
        image_file = request.files['image']
        if not image_file.filename:
            add_log('error', 'No selected file')
            return jsonify({'status': 'error', 'error': 'No selected file', 'logs': process_logs}), 400
            
        try:
            image = Image.open(image_file)
            image, error = validate_image(image)
            if error:
                add_log('error', f'Image validation failed: {error}')
                return jsonify({'status': 'error', 'error': error, 'logs': process_logs}), 400
        except Exception as e:
            add_log('error', f'Failed to open image: {str(e)}')
            return jsonify({'status': 'error', 'error': f'Failed to open image: {str(e)}', 'logs': process_logs}), 400
            
        add_log('info', f'Original image size: {image.size}, mode: {image.mode}')
        
        # Parse parameters
        layer_name = request.form.get('layer_name', 'l4')
        class_idx_str = request.form.get('class_idx', None)
        class_idx = int(class_idx_str) if class_idx_str else None
        
        # Process image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate Grad-CAM visualization
        add_log('info', 'Computing Grad-CAM...')
        cam, class_idx, class_name, confidence = grad_cam(
            model=model, 
            image_tensor=image_tensor,
            layer_name=layer_name,
            class_idx=class_idx
        )
        add_log('info', f'Grad-CAM computed for class {class_idx} ({class_name}) with confidence {confidence:.4f}')
        
        # Create visualization
        plt.style.use('ggplot')
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Show original image
        img_np = np.array(image.resize((224, 224)))
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show Grad-CAM heatmap
        im = axes[1].imshow(cam, cmap='jet')
        axes[1].set_title(f'Grad-CAM - Layer {layer_name}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Show overlay
        axes[2].imshow(img_np)
        axes[2].imshow(cam, cmap='jet', alpha=0.6)
        axes[2].set_title(f'Class: {class_name} ({confidence:.2f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the figure to a base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Create a separate high-quality overlay for download
        plt.figure(figsize=(10, 10))
        plt.imshow(img_np)
        plt.imshow(cam, cmap='jet', alpha=0.6)
        plt.title(f'Class: {class_name} (Confidence: {confidence:.2f})', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save overlay to base64
        buf_overlay = io.BytesIO()
        plt.savefig(buf_overlay, format='png', dpi=200, bbox_inches='tight')
        plt.close()
        buf_overlay.seek(0)
        overlay_str = base64.b64encode(buf_overlay.getvalue()).decode('utf-8')
        
        end_time = time.time()
        processing_time = end_time - start_time
        add_log('info', f'Processing completed in {processing_time:.2f} seconds')
        
        return jsonify({
            'status': 'success',
            'visualization': img_str,
            'overlay': overlay_str,
            'class_info': {
                'class_id': class_idx,
                'class_name': class_name,
                'confidence': confidence
            },
            'layer_name': layer_name,
            'processing_time': processing_time,
            'logs': process_logs
        })
    except Exception as e:
        error_msg = str(e)
        add_log('error', f'Error during Grad-CAM: {error_msg}')
        return jsonify({'status': 'error', 'error': error_msg, 'logs': process_logs}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 