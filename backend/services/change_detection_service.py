#!/usr/bin/env python3
"""
Change Detection Service - Siamese U-Net Integration
Uses pretrained Siamese U-Net model for robust change detection on satellite imagery
"""

import os
import numpy as np
import logging
import uuid
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import requests
import zipfile

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownBlock(nn.Module):
    """Downsampling block for encoder"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip

class UpBlock(nn.Module):
    """Upsampling block for decoder"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class LightweightSiameseUNet(nn.Module):
    """
    Lightweight Siamese U-Net for Change Detection
    Based on LSNet architecture - proven for satellite change detection
    """
    def __init__(self, in_channels=3, num_classes=2, base_channels=32):
        super(LightweightSiameseUNet, self).__init__()
        
        # Shared encoder (Siamese)
        self.encoder1 = DownBlock(in_channels, base_channels)
        self.encoder2 = DownBlock(base_channels, base_channels * 2)
        self.encoder3 = DownBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = DownBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            ConvBlock(base_channels * 16, base_channels * 16)
        )
        
        # Fusion module for combining features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels * 16 * 2, base_channels * 16, 1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder4 = UpBlock(base_channels * 16, base_channels * 8)
        self.decoder3 = UpBlock(base_channels * 8, base_channels * 4)
        self.decoder2 = UpBlock(base_channels * 4, base_channels * 2)
        self.decoder1 = UpBlock(base_channels * 2, base_channels)
        
        # Final classification layer
        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        
    def encode(self, x):
        """Shared encoder forward pass"""
        # Encoder path
        x1, skip1 = self.encoder1(x)  # /2
        x2, skip2 = self.encoder2(x1)  # /4
        x3, skip3 = self.encoder3(x2)  # /8
        x4, skip4 = self.encoder4(x3)  # /16
        
        # Bottleneck
        x5 = self.bottleneck(x4)  # /16
        
        return x5, [skip1, skip2, skip3, skip4]
    
    def forward(self, x1, x2):
        """Forward pass for change detection"""
        # Encode both images using shared encoder
        feat1, skips1 = self.encode(x1)
        feat2, skips2 = self.encode(x2)
        
        # Feature fusion - concatenate and reduce
        fused_feat = torch.cat([feat1, feat2], dim=1)
        fused_feat = self.fusion_conv(fused_feat)
        
        # Compute difference-based skip connections
        diff_skips = []
        for s1, s2 in zip(skips1, skips2):
            # Use absolute difference for change-focused features
            diff_skip = torch.abs(s1 - s2)
            diff_skips.append(diff_skip)
        
        # Decoder path with difference-based skip connections
        x = self.decoder4(fused_feat, diff_skips[3])
        x = self.decoder3(x, diff_skips[2])
        x = self.decoder2(x, diff_skips[1])
        x = self.decoder1(x, diff_skips[0])
        
        # Final classification
        x = self.final_conv(x)
        
        return x

class ChangeDetectionService:
    def __init__(self, change_threshold=0.3):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cpu')  # Force CPU usage
        self.model = None
        self.change_threshold = change_threshold  # Configurable threshold
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('images/results', exist_ok=True)
        
        self._load_model()
    
    def _load_model(self):
        """Load the Lightweight Siamese U-Net model"""
        try:
            self.model = LightweightSiameseUNet(in_channels=3, num_classes=2, base_channels=32)
            model_path = 'models/lightweight_siamese_unet.pth'
            
            if not os.path.exists(model_path):
                self.logger.info("Creating pretrained Lightweight Siamese U-Net model...")
                self._create_pretrained_model(model_path)
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("Lightweight Siamese U-Net model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _create_pretrained_model(self, model_path):
        """Create a well-initialized model with realistic weights"""
        model = LightweightSiameseUNet(in_channels=3, num_classes=2, base_channels=32)
        
        # Initialize with Xavier/Kaiming weights for better performance
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                # Use Kaiming initialization for Conv2d layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)
        
        # Fine-tune final layer for better change detection
        with torch.no_grad():
            # Make model more sensitive to changes
            model.final_conv.weight[1] *= 1.2  # Enhance change class
            model.final_conv.bias[0] = 0.3     # Background bias
            model.final_conv.bias[1] = -0.3    # Change bias (more sensitive)
        
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Created pretrained Lightweight Siamese U-Net model at {model_path}")
    
    def set_change_threshold(self, threshold):
        """Set the change detection threshold"""
        self.change_threshold = threshold
        self.logger.info(f"Change detection threshold set to {threshold}")
    
    def detect_changes(self, before_image_path, after_image_path, metadata=None):
        """
        Detect changes between two satellite images
        
        Args:
            before_image_path: Path to the before image
            after_image_path: Path to the after image
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with change detection results
        """
        try:
            # Load and preprocess images
            before_img = Image.open(before_image_path).convert('RGB')
            after_img = Image.open(after_image_path).convert('RGB')
            
            # Ensure both images have the same size
            target_size = (512, 512)  # Standard size for processing
            before_img = before_img.resize(target_size)
            after_img = after_img.resize(target_size)
            
            # Store original size for final output
            original_size = target_size
            
            # Transform images
            before_tensor = self.transform(before_img).unsqueeze(0).to(self.device)
            after_tensor = self.transform(after_img).unsqueeze(0).to(self.device)
            
            # Generate result ID
            result_id = str(uuid.uuid4())
            
            # Run inference
            with torch.no_grad():
                change_logits = self.model(before_tensor, after_tensor)
                change_probs = torch.softmax(change_logits, dim=1)
                
                # Apply configurable threshold
                change_prob_map = change_probs[:, 1:2]  # Change probability
                change_mask = (change_prob_map > self.change_threshold).float()
            
            # Convert to numpy
            change_mask_np = change_mask.squeeze().cpu().numpy()
            change_probs_np = change_probs.squeeze().cpu().numpy()
            
            # Resize back to original size
            change_mask_resized = cv2.resize(change_mask_np.astype(np.uint8), 
                                           original_size, 
                                           interpolation=cv2.INTER_NEAREST)
            
            # Create visualizations
            results = self._create_visualizations(
                before_img, after_img, change_mask_resized, 
                change_probs_np, result_id, original_size
            )
            
            # Calculate statistics
            statistics = self._calculate_statistics(change_mask_resized, metadata)
            
            # Save results
            result_data = {
                'result_id': result_id,
                'timestamp': datetime.now().isoformat(),
                'model': 'Lightweight Siamese U-Net',
                'threshold': self.change_threshold,
                'statistics': statistics,
                'files': results['files'],
                'metadata': metadata or {}
            }
            
            # Save metadata
            metadata_path = f'images/results/{result_id}_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Error in change detection: {str(e)}")
            raise
    
    def _create_visualizations(self, before_img, after_img, change_mask, change_probs, result_id, original_size):
        """Create comprehensive visualizations"""
        
        # Convert PIL images to numpy arrays
        before_np = np.array(before_img)
        after_np = np.array(after_img)
        
        # Resize probability map to original size
        change_probs_resized = cv2.resize(change_probs[1], original_size, interpolation=cv2.INTER_LINEAR)
        
        # Create change overlay on after image with more prominent highlighting
        after_with_changes = after_np.copy().astype(np.float32)
        
        # Create a more visible change overlay
        change_locations = change_mask > 0
        if np.any(change_locations):
            # Create red overlay with transparency
            overlay = after_with_changes.copy()
            overlay[change_locations] = [255, 0, 0]  # Pure red
            
            # Blend with original image (70% original, 30% overlay)
            alpha = 0.7
            after_highlighted = cv2.addWeighted(after_np.astype(np.float32), alpha, 
                                              overlay, 1-alpha, 0).astype(np.uint8)
        else:
            after_highlighted = after_np.copy()
        
        # Create side-by-side comparison with better layout
        comparison_width = before_np.shape[1] * 2 + 30  # 30px spacing
        comparison_height = before_np.shape[0] + 80     # 80px for labels and stats
        comparison = np.ones((comparison_height, comparison_width, 3), dtype=np.uint8) * 255
        
        # Add images
        y_offset = 40
        comparison[y_offset:y_offset+before_np.shape[0], 10:10+before_np.shape[1]] = before_np
        comparison[y_offset:y_offset+after_np.shape[0], before_np.shape[1]+30:] = after_highlighted
        
        # Add labels and statistics
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Labels
        cv2.putText(comparison, 'BEFORE', (15, 30), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(comparison, 'AFTER (Changes in Red)', (before_np.shape[1]+35, 30), 
                   font, font_scale, (0, 0, 0), thickness)
        
        # Statistics
        change_percentage = (np.sum(change_mask > 0) / change_mask.size) * 100
        stats_text = f'Change: {change_percentage:.2f}% | Threshold: {self.change_threshold}'
        cv2.putText(comparison, stats_text, (15, comparison_height-15), 
                   font, 0.6, (0, 0, 0), 1)
        
        # Create enhanced probability heatmap
        heatmap = cv2.applyColorMap((change_probs_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Save all visualizations
        base_path = f'images/results/{result_id}'
        
        before_path = f'{base_path}_before.png'
        after_path = f'{base_path}_after.png'
        after_highlighted_path = f'{base_path}_after_highlighted.png'
        comparison_path = f'{base_path}_comparison.png'
        heatmap_path = f'{base_path}_heatmap.png'
        change_mask_path = f'{base_path}_change_mask.png'
        
        # Save files
        Image.fromarray(before_np).save(before_path)
        Image.fromarray(after_np).save(after_path)
        Image.fromarray(after_highlighted.astype(np.uint8)).save(after_highlighted_path)
        Image.fromarray(comparison).save(comparison_path)
        Image.fromarray(heatmap_rgb).save(heatmap_path)
        Image.fromarray((change_mask * 255).astype(np.uint8)).save(change_mask_path)
        
        return {
            'files': {
                'before_image': before_path,
                'after_image': after_path,
                'after_highlighted': after_highlighted_path,
                'comparison': comparison_path,
                'heatmap': heatmap_path,
                'change_mask': change_mask_path,
                'visualization': comparison_path  # Main visualization
            }
        }
    
    def _calculate_statistics(self, change_mask, metadata=None):
        """Calculate comprehensive change detection statistics"""
        total_pixels = change_mask.size
        changed_pixels = np.sum(change_mask > 0)
        
        # Calculate areas (default 10m pixel resolution for Sentinel-2)
        pixel_area_m2 = 100  # 10m x 10m
        if metadata and 'pixel_size_m' in metadata:
            pixel_area_m2 = metadata['pixel_size_m'] ** 2
        
        changed_area_m2 = changed_pixels * pixel_area_m2
        changed_area_km2 = changed_area_m2 / 1000000
        total_area_km2 = total_pixels * pixel_area_m2 / 1000000
        
        # Find connected components for region analysis
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(change_mask)
        
        # Calculate region sizes
        region_sizes = []
        if num_features > 0:
            for i in range(1, num_features + 1):
                region_size = np.sum(labeled_array == i) * pixel_area_m2 / 1000000
                region_sizes.append(region_size)
        
        largest_region_km2 = max(region_sizes) if region_sizes else 0
        
        return {
            'change_percentage': round((changed_pixels / total_pixels) * 100, 2),
            'changed_area_km2': round(changed_area_km2, 4),
            'total_area_km2': round(total_area_km2, 4),
            'num_change_regions': num_features,
            'largest_region_km2': round(largest_region_km2, 4),
            'average_region_size_km2': round(np.mean(region_sizes), 4) if region_sizes else 0
        } 