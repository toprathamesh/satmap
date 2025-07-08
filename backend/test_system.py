#!/usr/bin/env python3
"""
Comprehensive system test for Change Detection Platform
Tests all backend components and API endpoints
"""

import sys
import os
import requests
import json
import time
import traceback
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        # Core imports
        import flask
        import numpy as np
        import PIL
        import cv2
        
        # Try GEE import (might fail if not authenticated)
        try:
            import ee
            print("  ✅ Google Earth Engine imported")
        except Exception as e:
            print(f"  ⚠️  Google Earth Engine import warning: {e}")
        
        # Try torch import (might fail if not installed)
        try:
            import torch
            print("  ✅ PyTorch imported")
        except Exception as e:
            print(f"  ❌ PyTorch import failed: {e}")
            return False
        
        # Try geospatial imports
        try:
            import rasterio
            import geopandas
            print("  ✅ Geospatial libraries imported")
        except Exception as e:
            print(f"  ❌ Geospatial libraries failed: {e}")
            return False
            
        print("  ✅ All core imports successful")
        return True
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False

def test_services():
    """Test if services can be instantiated"""
    print("\n🔍 Testing service instantiation...")
    
    try:
        from services.gee_service import GEEService
        from services.change_detection_service import ChangeDetectionService
        from services.export_service import ExportService
        
        # Test GEE Service (might fail without auth)
        try:
            gee_service = GEEService()
            print("  ✅ GEE Service instantiated")
        except Exception as e:
            print(f"  ⚠️  GEE Service warning (authentication needed): {e}")
        
        # Test Change Detection Service
        try:
            cd_service = ChangeDetectionService()
            print("  ✅ Change Detection Service instantiated")
        except Exception as e:
            print(f"  ❌ Change Detection Service failed: {e}")
            return False
        
        # Test Export Service
        try:
            export_service = ExportService()
            print("  ✅ Export Service instantiated")
        except Exception as e:
            print(f"  ❌ Export Service failed: {e}")
            return False
            
        print("  ✅ All services instantiated successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Service test failed: {e}")
        traceback.print_exc()
        return False

def test_utils():
    """Test utility functions"""
    print("\n🔍 Testing utility functions...")
    
    try:
        from utils.validators import validate_coordinates, validate_date_range
        from utils.gmaps_parser import parse_gmaps_url
        
        # Test coordinate validation
        assert validate_coordinates(19.0419252, 73.0270304) == True
        assert validate_coordinates(91, 0) == False  # Invalid lat
        assert validate_coordinates(0, 181) == False  # Invalid lon
        print("  ✅ Coordinate validation working")
        
        # Test date validation
        assert validate_date_range("2023-01-01", "2023-06-01") == True
        assert validate_date_range("2023-06-01", "2023-01-01") == False  # Wrong order
        print("  ✅ Date validation working")
        
        # Test Google Maps URL parsing
        test_url = "https://www.google.com/maps/@19.0419252,73.0270304,17z"
        coords = parse_gmaps_url(test_url)
        assert coords is not None
        assert abs(coords['lat'] - 19.0419252) < 0.001
        print("  ✅ Google Maps URL parsing working")
        
        print("  ✅ All utility functions working")
        return True
        
    except Exception as e:
        print(f"  ❌ Utility test failed: {e}")
        traceback.print_exc()
        return False

def test_flask_app():
    """Test if Flask app can start"""
    print("\n🔍 Testing Flask application...")
    
    try:
        from app import create_app
        app = create_app()
        
        # Test app configuration
        assert app is not None
        print("  ✅ Flask app created successfully")
        
        # Test app can be configured for testing
        app.config['TESTING'] = True
        client = app.test_client()
        
        # Test health endpoint
        response = client.get('/api/health')
        print(f"  📊 Health endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = json.loads(response.data)
            print(f"  📊 Health response: {data.get('status', 'unknown')}")
        
        print("  ✅ Flask app test completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Flask app test failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test if ML model can be loaded"""
    print("\n🔍 Testing ML model loading...")
    
    try:
        from services.change_detection_service import ChangeDetectionService
        
        # Create service (this should create/load model)
        cd_service = ChangeDetectionService()
        
        if cd_service.model is not None:
            print("  ✅ Model loaded successfully")
            
            # Test model can make predictions (dummy data)
            import torch
            dummy_input1 = torch.randn(1, 3, 256, 256)
            dummy_input2 = torch.randn(1, 3, 256, 256)
            
            with torch.no_grad():
                output = cd_service.model(dummy_input1, dummy_input2)
                
            # Lightweight Siamese U-Net outputs (batch_size, num_classes, height, width)
            assert output.shape == (1, 2, 256, 256)
            print("  ✅ Model inference test passed")
            
            # Test configurable threshold
            original_threshold = cd_service.change_threshold
            cd_service.set_change_threshold(0.5)
            assert cd_service.change_threshold == 0.5
            cd_service.set_change_threshold(original_threshold)
            print("  ✅ Threshold configuration test passed")
            
            return True
        else:
            print("  ❌ Model not loaded")
            return False
            
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        traceback.print_exc()
        return False

def test_directories():
    """Test if required directories exist or can be created"""
    print("\n🔍 Testing directory structure...")
    
    required_dirs = ['images', 'models', 'images/satellite', 'images/demo', 'images/results']
    
    try:
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
            if os.path.exists(dir_path):
                print(f"  ✅ Directory {dir_path} ready")
            else:
                print(f"  ❌ Directory {dir_path} failed")
                return False
        
        print("  ✅ All directories ready")
        return True
        
    except Exception as e:
        print(f"  ❌ Directory test failed: {e}")
        return False

def run_live_api_test():
    """Test live API endpoints if server is running"""
    print("\n🔍 Testing live API endpoints...")
    
    base_url = "http://localhost:5000/api"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("  ✅ Health endpoint working")
        else:
            print(f"  ❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test Google Maps parsing
        test_data = {
            "url": "https://www.google.com/maps/@19.0419252,73.0270304,17z"
        }
        response = requests.post(f"{base_url}/parse-gmaps-link", 
                               json=test_data, timeout=10)
        if response.status_code == 200:
            print("  ✅ Google Maps parsing endpoint working")
        else:
            print(f"  ⚠️  Google Maps parsing status: {response.status_code}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("  ℹ️  Server not running - skipping live API tests")
        return True
    except Exception as e:
        print(f"  ❌ Live API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Change Detection Platform - System Tests")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Service Instantiation", test_services),
        ("Utility Functions", test_utils),
        ("Flask Application", test_flask_app),
        ("ML Model Loading", test_model_loading),
        ("Directory Structure", test_directories),
        ("Live API Endpoints", run_live_api_test),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🧪 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 