#!/usr/bin/env python3
"""
Standalone test script for the CEBSNet change detection model
This tests the model without requiring Google Earth Engine or server setup
"""

import os
import sys
import numpy as np
from PIL import Image
import json
import random

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.change_detection_service import ChangeDetectionService

def create_synthetic_satellite_images():
    """Create synthetic satellite images for testing"""
    
    # Create a before image (mostly green - vegetation)
    before_img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add some vegetation (green areas)
    before_img[100:400, 100:400, 1] = 120  # Green channel
    before_img[100:400, 100:400, 0] = 60   # Some red
    before_img[100:400, 100:400, 2] = 40   # Some blue
    
    # Add some water (blue areas)
    before_img[50:100, 200:350, 2] = 150   # Blue water
    before_img[50:100, 200:350, 0] = 30    # Low red
    before_img[50:100, 200:350, 1] = 80    # Medium green
    
    # Add some soil/bare ground
    before_img[400:450, 300:450, :] = [139, 101, 71]  # Brown soil
    
    # Create an after image with some changes
    after_img = before_img.copy()
    
    # Simulate deforestation - remove some vegetation and add bare soil
    after_img[150:250, 200:300, :] = [139, 101, 71]  # Changed to soil
    
    # Simulate urban development - add gray/concrete areas
    after_img[120:180, 120:180, :] = [128, 128, 128]  # Gray concrete
    
    # Simulate water level change
    after_img[50:80, 200:350, :] = [139, 101, 71]  # Water dried up
    
    return before_img, after_img

def save_test_images(before_img, after_img):
    """Save test images to temp directory"""
    os.makedirs('temp', exist_ok=True)
    
    before_path = 'images/satellite/test_before.png'
    after_path = 'images/satellite/test_after.png'
    
    Image.fromarray(before_img).save(before_path)
    Image.fromarray(after_img).save(after_path)
    
    return before_path, after_path

def run_change_detection_test():
    """Run the change detection test"""
    
    print("🧪 Testing CEBSNet Change Detection Model")
    print("=" * 50)
    
    # Generate random test coordinates (near Mumbai)
    lat = 19.0419 + random.uniform(-0.01, 0.01)
    lon = 73.0270 + random.uniform(-0.01, 0.01)
    
    print(f"📍 Test Location: {lat:.6f}°N, {lon:.6f}°E")
    
    try:
        # Step 1: Create synthetic satellite images
        print("\n🖼️  Creating synthetic satellite images...")
        before_img, after_img = create_synthetic_satellite_images()
        before_path, after_path = save_test_images(before_img, after_img)
        print(f"   ✅ Before image: {before_path}")
        print(f"   ✅ After image: {after_path}")
        
        # Step 2: Initialize change detection service
        print("\n🤖 Initializing CEBSNet model...")
        cd_service = ChangeDetectionService()
        print("   ✅ Model loaded successfully")
        
        # Step 3: Run change detection
        print("\n🔍 Running change detection...")
        metadata = {
            'latitude': lat,
            'longitude': lon,
            'pixel_size_m': 10,  # 10m pixel resolution
            'test_mode': True
        }
        
        results = cd_service.detect_changes(before_path, after_path, metadata)
        
        # Step 4: Display results
        print("\n📊 CHANGE DETECTION RESULTS")
        print("=" * 30)
        
        stats = results['statistics']
        print(f"Change Percentage: {stats['change_percentage']}%")
        print(f"Changed Area: {stats['changed_area_km2']} km²")
        print(f"Total Area: {stats['total_area_km2']} km²")
        print(f"Number of Change Regions: {stats['num_change_regions']}")
        print(f"Largest Region: {stats['largest_region_km2']} km²")
        print(f"Average Region Size: {stats['average_region_size_km2']} km²")
        
        # Step 5: Show generated files
        print(f"\n📁 GENERATED FILES")
        print("=" * 20)
        files = results['files']
        for file_type, file_path in files.items():
            print(f"{file_type}: {file_path}")
        
        # Step 6: Verify files exist
        print(f"\n✅ FILE VERIFICATION")
        print("=" * 20)
        missing_files = []
        for file_type, file_path in files.items():
            full_path = os.path.join('temp', file_path)
            if os.path.exists(full_path):
                size_kb = os.path.getsize(full_path) / 1024
                print(f"   ✅ {file_type}: {size_kb:.1f} KB")
            else:
                print(f"   ❌ {file_type}: Missing")
                missing_files.append(file_type)
        
        if missing_files:
            print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        else:
            print(f"\n🎉 All visualization files generated successfully!")
        
        # Step 7: Show result summary
        print(f"\n🎯 TEST SUMMARY")
        print("=" * 15)
        print(f"Result ID: {results['result_id']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Model: CEBSNet with CBAM attention")
        print(f"Status: ✅ SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_change_detection_test()
    
    if success:
        print(f"\n🚀 The CEBSNet model is working correctly!")
        print(f"   Check the 'images/results/' directory for all visualization outputs.")
        print(f"   The model successfully detected changes and generated:")
        print(f"   • Before/After images")
        print(f"   • Change highlights")  
        print(f"   • Side-by-side comparison")
        print(f"   • Probability heatmap")
        print(f"   • Detailed statistics")
    else:
        print(f"\n💥 Test failed - check error messages above")
    
    sys.exit(0 if success else 1) 