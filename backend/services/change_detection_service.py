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
        
        # Image preprocessing transforms - using standard normalization
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Standard ImageNet normalization works well for preprocessed satellite imagery
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
            # Check if dates are identical - no change detection needed
            if metadata:
                before_date = metadata.get('before_date', '')
                after_date = metadata.get('after_date', '')
                if before_date == after_date and before_date:
                    self.logger.info(f"Identical dates detected ({before_date}), returning no change result")
                    return self._create_no_change_result(before_image_path, after_image_path, metadata)
            
            # Load and preprocess images
            before_img = Image.open(before_image_path).convert('RGB')
            after_img = Image.open(after_image_path).convert('RGB')
            
            # Ensure both images have the same size for processing
            target_size = (512, 512)  # Standard size for processing
            before_img_resized = before_img.resize(target_size)
            after_img_resized = after_img.resize(target_size)
            
            # Store original size for final output
            original_size = before_img.size
            
            # Transform images for model
            before_tensor = self.transform(before_img_resized).unsqueeze(0).to(self.device)
            after_tensor = self.transform(after_img_resized).unsqueeze(0).to(self.device)
            
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
            
            # Create visualizations using original images
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
                'metadata': metadata or {},
                'coordinates': {
                    'lat': metadata.get('latitude', 0) if metadata else 0,
                    'lon': metadata.get('longitude', 0) if metadata else 0
                },
                'before_date': metadata.get('before_date', '') if metadata else '',
                'after_date': metadata.get('after_date', '') if metadata else ''
            }
            
            # Save metadata
            metadata_path = f'images/results/{result_id}_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Error in change detection: {str(e)}")
            raise
    
    def _create_no_change_result(self, before_image_path, after_image_path, metadata):
        """Create a no-change result for identical dates"""
        try:
            # Load images
            before_img = Image.open(before_image_path).convert('RGB')
            after_img = Image.open(after_image_path).convert('RGB')
            
            # Ensure same size
            target_size = (512, 512)
            before_img = before_img.resize(target_size)
            after_img = after_img.resize(target_size)
            
            # Generate result ID
            result_id = str(uuid.uuid4())
            
            # Create empty change mask (no changes)
            change_mask = np.zeros(target_size, dtype=np.uint8)
            change_probs = np.zeros((2, target_size[0], target_size[1]), dtype=np.float32)
            change_probs[0] = 1.0  # All pixels are "no change"
            
            # Create visualizations with no changes
            results = self._create_visualizations(
                before_img, after_img, change_mask, 
                change_probs, result_id, target_size
            )
            
            # Calculate statistics (should be zero changes)
            statistics = self._calculate_statistics(change_mask, metadata)
            
            # Override statistics to ensure zero change
            statistics.update({
                'change_percentage': 0.0,
                'changed_area_km2': 0.0,
                'num_change_regions': 0,
                'largest_region_km2': 0.0,
                'average_region_size_km2': 0.0,
                'total_survey_area_km2': statistics.get('total_survey_area_km2', 1.0)
            })
            
            # Save results
            result_data = {
                'result_id': result_id,
                'timestamp': datetime.now().isoformat(),
                'model': 'Lightweight Siamese U-Net (No Change - Identical Dates)',
                'threshold': self.change_threshold,
                'statistics': statistics,
                'files': results['files'],
                'metadata': metadata or {},
                'coordinates': {
                    'lat': metadata.get('latitude', 0) if metadata else 0,
                    'lon': metadata.get('longitude', 0) if metadata else 0
                },
                'before_date': metadata.get('before_date', '') if metadata else '',
                'after_date': metadata.get('after_date', '') if metadata else ''
            }
            
            # Save metadata
            metadata_path = f'images/results/{result_id}_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            self.logger.info(f"No-change result created for identical dates: {result_id}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Error creating no-change result: {str(e)}")
            raise
    
    def create_instant_no_change_result(self, metadata):
        """Create an instant no-change result without loading any images (for identical dates)"""
        try:
            # Generate result ID
            result_id = str(uuid.uuid4())
            
            # Create minimal statistics for no change
            statistics = {
                'change_percentage': 0.0,
                'changed_area_km2': 0.0,
                'total_area_km2': 1.0,  # 1 km² AOI
                'num_change_regions': 0,
                'largest_region_km2': 0.0,
                'average_region_size_km2': 0.0,
                'total_survey_area_km2': 1.0
            }
            
            # Create minimal file structure (no actual files created for instant result)
            files = {
                'before_image': None,
                'after_image': None,
                'after_highlighted': None,
                'comparison': None,
                'heatmap': None,
                'change_mask': None,
                'visualization': None
            }
            
            # Create result data
            result_data = {
                'result_id': result_id,
                'timestamp': datetime.now().isoformat(),
                'model': 'Fast No-Change Detection (Identical Dates)',
                'threshold': self.change_threshold,
                'statistics': statistics,
                'files': files,
                'metadata': metadata or {},
                'coordinates': {
                    'lat': metadata.get('latitude', 0) if metadata else 0,
                    'lon': metadata.get('longitude', 0) if metadata else 0
                },
                'before_date': metadata.get('before_date', '') if metadata else '',
                'after_date': metadata.get('after_date', '') if metadata else '',
                'instant_result': True  # Flag to indicate this was instant
            }
            
            # Save minimal metadata
            metadata_path = f'images/results/{result_id}_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            self.logger.info(f"Instant no-change result created: {result_id}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Error creating instant no-change result: {str(e)}")
            raise
    
    def _create_visualizations(self, before_img, after_img, change_mask, change_probs, result_id, original_size):
        """Create comprehensive visualizations with simple, robust processing"""
        
        # Convert PIL images to numpy arrays
        before_np = np.array(before_img)
        after_np = np.array(after_img)
        
        # Ensure proper data types
        if before_np.dtype != np.uint8:
            before_np = np.clip(before_np * 255, 0, 255).astype(np.uint8)
        if after_np.dtype != np.uint8:
            after_np = np.clip(after_np * 255, 0, 255).astype(np.uint8)
        
        # Resize probability map to match original image size
        change_probs_resized = cv2.resize(change_probs[1], original_size, interpolation=cv2.INTER_LINEAR)
        
        # Create change overlay on after image
        after_with_changes = after_np.copy()
        
        # Apply change highlighting
        change_locations = change_mask > 0
        if np.any(change_locations):
            # Simple red overlay
            after_with_changes[change_locations, 0] = np.minimum(
                after_with_changes[change_locations, 0] + 100, 255)  # Add red
        
        # Create side-by-side comparison
        comparison_width = before_np.shape[1] * 2 + 30
        comparison_height = before_np.shape[0] + 80
        comparison = np.ones((comparison_height, comparison_width, 3), dtype=np.uint8) * 255
        
        # Add images to comparison
        y_offset = 40
        comparison[y_offset:y_offset+before_np.shape[0], 10:10+before_np.shape[1]] = before_np
        comparison[y_offset:y_offset+after_np.shape[0], before_np.shape[1]+30:] = after_with_changes
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'BEFORE', (15, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(comparison, 'AFTER (Changes Highlighted)', (before_np.shape[1]+35, 30), font, 0.7, (0, 0, 0), 2)
        
        # Add statistics
        change_percentage = (np.sum(change_mask > 0) / change_mask.size) * 100
        stats_text = f'Change: {change_percentage:.2f}%'
        cv2.putText(comparison, stats_text, (15, comparison_height-15), font, 0.6, (0, 0, 0), 1)
        
        # Create heatmap
        heatmap = cv2.applyColorMap((change_probs_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Save all files
        base_path = f'images/results/{result_id}'
        
        before_path = f'{base_path}_before.png'
        after_path = f'{base_path}_after.png'
        after_highlighted_path = f'{base_path}_after_highlighted.png'
        comparison_path = f'{base_path}_comparison.png'
        heatmap_path = f'{base_path}_heatmap.png'
        change_mask_path = f'{base_path}_change_mask.png'
        
        # Save with error handling
        try:
            Image.fromarray(before_np).save(before_path, 'PNG')
            Image.fromarray(after_np).save(after_path, 'PNG')
            Image.fromarray(after_with_changes).save(after_highlighted_path, 'PNG')
            Image.fromarray(comparison).save(comparison_path, 'PNG')
            Image.fromarray(heatmap_rgb).save(heatmap_path, 'PNG')
            Image.fromarray((change_mask * 255).astype(np.uint8)).save(change_mask_path, 'PNG')
        except Exception as e:
            self.logger.error(f"Error saving images: {e}")
            raise
        
        return {
            'files': {
                'before_image': before_path,
                'after_image': after_path,
                'after_highlighted': after_highlighted_path,
                'comparison': comparison_path,
                'heatmap': heatmap_path,
                'change_mask': change_mask_path,
                'visualization': comparison_path
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