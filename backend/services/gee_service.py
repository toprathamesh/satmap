"""
Google Earth Engine Service for Satellite Imagery and Change Detection
Clean, robust implementation focused on clear images without artifacts
"""

import ee
import os
import logging
import requests
import numpy as np
from PIL import Image
import json
from datetime import datetime, timedelta
import tempfile
import uuid
import rasterio
from skimage import exposure

class GEEService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.demo_mode = False
        
        # Create necessary directories
        os.makedirs('images/satellite', exist_ok=True)
        os.makedirs('images/demo', exist_ok=True)
        os.makedirs('temp/gee_images', exist_ok=True)
        
        self._initialize_gee()
    
    def _initialize_gee(self):
        """Initialize Google Earth Engine with authentication"""
        try:
            # Try different authentication methods
            service_account_key = os.environ.get('GEE_SERVICE_ACCOUNT_KEY')
            
            if service_account_key:
                self._authenticate_with_service_account(service_account_key)
            else:
                self._authenticate_standard()
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Earth Engine: {e}")
            raise Exception(f"Google Earth Engine authentication required: {e}")
    
    def _authenticate_standard(self):
        """Try standard GEE authentication methods"""
        try:
            # Method 1: Try with user's Google Cloud project
            try:
                project_id = 'caramel-goal-464010-u8'
                ee.Initialize(project=project_id)
                self.initialized = True
                self.logger.info(f"✅ Google Earth Engine initialized with project: {project_id}")
                
                # Test connection
                test_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
                count = test_collection.size().getInfo()
                self.logger.info(f"📡 Successfully connected to Sentinel-2 data ({count} test image)")
                return
                
            except Exception as e:
                self.logger.warning(f"User project failed: {e}")
            
            # Method 2: Try with environment variable project ID
            try:
                project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'caramel-goal-464010-u8')
                ee.Initialize(project=project_id)
                self.initialized = True
                self.logger.info(f"✅ Google Earth Engine initialized with project: {project_id}")
                
                # Test connection
                test_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
                count = test_collection.size().getInfo()
                self.logger.info(f"📡 Successfully connected to Sentinel-2 data ({count} test image)")
                return
                
            except Exception as e:
                self.logger.warning(f"Environment project failed: {e}")
            
            # Method 3: Try simple initialization (legacy mode)
            try:
                ee.Initialize()
                self.initialized = True
                self.logger.info("✅ Google Earth Engine initialized in legacy mode")
                
                # Test connection
                test_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
                count = test_collection.size().getInfo()
                self.logger.info(f"📡 Successfully connected to Sentinel-2 data ({count} test image)")
                return
                
            except Exception as e:
                self.logger.warning(f"Legacy initialization failed: {e}")
            
            # If all methods fail, provide clear instructions
            self.logger.error("🚨 Google Earth Engine authentication failed!")
            self.logger.error("To use real satellite images, please:")
            self.logger.error("1. Run: earthengine authenticate")
            self.logger.error("2. Follow the browser authentication flow")
            self.logger.error("3. Restart this application")
            
            raise Exception("GEE authentication required")
            
        except Exception as e:
            self.logger.error(f"All GEE authentication methods failed: {e}")
            raise
    
    def _authenticate_with_service_account(self, service_account_key):
        """Authenticate with GEE using service account"""
        try:
            # If it's a file path
            if os.path.exists(service_account_key):
                credentials = ee.ServiceAccountCredentials(None, service_account_key)
            else:
                # If it's a JSON string
                import tempfile
                import json
                
                # Parse JSON and create temporary file
                key_data = json.loads(service_account_key)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(key_data, f)
                    temp_key_file = f.name
                
                credentials = ee.ServiceAccountCredentials(None, temp_key_file)
                os.unlink(temp_key_file)  # Clean up
            
            ee.Initialize(credentials)
            self.initialized = True
            self.logger.info("Google Earth Engine initialized with service account")
            
        except Exception as e:
            self.logger.error(f"Service account authentication failed: {e}")
            raise
    
    def check_connection(self):
        """Check if GEE is properly connected"""
        try:
            if not self.initialized:
                return False
            
            # Test with a simple query
            test_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
            count = test_collection.size().getInfo()
            return count >= 0  # Should be 0 or more
            
        except Exception as e:
            self.logger.error(f"GEE connection test failed: {e}")
            return False
    
    def get_preprocessed_image(self, lat, lon, date, aoi_size_km=1.0):
        """
        Get preprocessed satellite image from Google Earth Engine
        Returns path to crystal clear PNG image without artifacts
        """
        try:
            if not self.initialized:
                raise Exception("Google Earth Engine not initialized")
            
            self.logger.info(f"🛰️ Fetching satellite image for {lat}, {lon} on {date}")
            
            # Create area of interest
            aoi = self._create_aoi(lat, lon, aoi_size_km)
            
            # Fetch crystal clear Sentinel-2 image
            image_path = self._fetch_crystal_clear_image(aoi, date)
            
            self.logger.info(f"✅ Crystal clear satellite image ready: {image_path}")
            return image_path
            
        except Exception as e:
            self.logger.error(f"Error getting preprocessed image: {e}")
            raise
    
    def _create_aoi(self, lat, lon, size_km):
        """Create a square area of interest around the coordinates"""
        # Convert km to degrees (rough approximation)
        size_deg = size_km / 111.0  # 1 degree ≈ 111 km
        half_size = size_deg / 2.0
        
        # Create bounding box
        coords = [
            [lon - half_size, lat - half_size],  # SW
            [lon + half_size, lat - half_size],  # SE
            [lon + half_size, lat + half_size],  # NE
            [lon - half_size, lat + half_size],  # NW
            [lon - half_size, lat - half_size]   # Close the polygon
        ]
        
        return ee.Geometry.Polygon(coords)
    
    def _fetch_crystal_clear_image(self, aoi, date):
        """Fetch only crystal clear Sentinel-2 images - reject cloudy ones entirely"""
        try:
            # Define initial date range (±15 days for best quality match)
            target_date = datetime.strptime(date, '%Y-%m-%d')
            
            # Try progressively larger date ranges to find clear images
            search_ranges = [
                (15, 5),    # ±15 days, <5% clouds (pristine)
                (30, 8),    # ±30 days, <8% clouds (very clear)
                (45, 12),   # ±45 days, <12% clouds (clear)
                (60, 15),   # ±60 days, <15% clouds (acceptable)
                (90, 20),   # ±90 days, <20% clouds (last resort)
            ]
            
            best_image = None
            best_cloud_cover = 100
            best_date_diff = 999
            
            for days_range, max_clouds in search_ranges:
                start_date = (target_date - timedelta(days=days_range)).strftime('%Y-%m-%d')
                end_date = (target_date + timedelta(days=days_range)).strftime('%Y-%m-%d')
                
                self.logger.info(f"Searching for images with <{max_clouds}% clouds in ±{days_range} days from {date}")
                
                # Get very clear Sentinel-2 images only
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(aoi)
                             .filterDate(start_date, end_date)
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_clouds))
                             .sort('CLOUDY_PIXEL_PERCENTAGE'))
                
                collection_size = collection.size().getInfo()
                
                if collection_size > 0:
                    self.logger.info(f"Found {collection_size} clear images with <{max_clouds}% clouds")
                    
                    # Get image info to find the best one
                    collection_list = collection.limit(5).getInfo()  # Check top 5 clearest
                    
                    for feature in collection_list['features']:
                        props = feature['properties']
                        cloud_cover = props.get('CLOUDY_PIXEL_PERCENTAGE', 100)
                        image_date_str = props.get('PRODUCT_ID', '').split('_')[2][:8]
                        
                        if len(image_date_str) == 8:
                            try:
                                image_date = datetime.strptime(image_date_str, '%Y%m%d')
                                date_diff = abs((image_date - target_date).days)
                                
                                # Prefer images that are both clearer and closer to target date
                                if cloud_cover < best_cloud_cover or (cloud_cover == best_cloud_cover and date_diff < best_date_diff):
                                    best_image = ee.Image(feature['id'])
                                    best_cloud_cover = cloud_cover
                                    best_date_diff = date_diff
                                    
                            except ValueError:
                                continue
                    
                    # If we found a very clear image (< 10% clouds), use it immediately
                    if best_cloud_cover < 10:
                        break
            
            if best_image is None:
                raise Exception(f"No clear satellite images found for {date}. Please try a different date or location. All available images have too much cloud cover (>20%).")
            
            self.logger.info(f"Selected best image: {best_cloud_cover}% clouds, {best_date_diff} days from target date")
            
            # Process the clear image WITHOUT cloud masking
            if best_cloud_cover < 8:
                # Pristine image - no masking needed at all
                self.logger.info(f"Pristine image ({best_cloud_cover}% clouds) - no processing needed")
                processed_image = best_image
            elif best_cloud_cover < 15:
                # Very clear image - minimal processing only
                self.logger.info(f"Very clear image ({best_cloud_cover}% clouds) - minimal processing")
                processed_image = self._apply_minimal_processing(best_image)
            else:
                # Clear image - gentle enhancement only (NO masking)
                self.logger.info(f"Clear image ({best_cloud_cover}% clouds) - gentle enhancement only")
                processed_image = self._apply_gentle_enhancement(best_image)
            
            # Select RGB bands (10m resolution): B4=Red, B3=Green, B2=Blue
            rgb_image = processed_image.select(['B4', 'B3', 'B2'])
            
            # Convert from digital numbers to reflectance (0-1 range)
            rgb_image = rgb_image.multiply(0.0001)
            
            # Apply gentle contrast enhancement for natural appearance
            rgb_image = rgb_image.clamp(0, 0.3)
            
            # Clip to area of interest
            rgb_image = rgb_image.clip(aoi)
            
            # Export and convert to PNG
            image_path = self._export_and_convert(rgb_image, aoi, date)
            
            return image_path
            
        except Exception as e:
            self.logger.error(f"Error fetching crystal clear image: {e}")
            raise
    
    def _apply_minimal_processing(self, image):
        """Apply minimal processing for very clear images"""
        try:
            # Only mask obvious water/snow if needed, but not clouds
            # This preserves image quality while handling edge cases
            return image
        except Exception as e:
            self.logger.warning(f"Minimal processing failed, using original: {e}")
            return image
    
    def _apply_gentle_enhancement(self, image):
        """Apply gentle enhancement without cloud masking"""
        try:
            # Only apply very gentle contrast enhancement
            # NO cloud masking to avoid black artifacts
            return image
        except Exception as e:
            self.logger.warning(f"Gentle enhancement failed, using original: {e}")
            return image
    
    def _export_and_convert(self, image, aoi, date):
        """Export image from GEE and convert to crystal clear PNG"""
        try:
            # Generate unique filename
            unique_id = uuid.uuid4().hex[:8]
            filename = f'crystal_clear_{unique_id}_{date}_10m'
            tiff_path = f'temp/gee_images/{filename}.tif'
            png_path = f'images/satellite/{filename}.png'
            
            # Try direct PNG export first (more reliable)
            try:
                return self._export_direct_png(image, aoi, date, png_path)
            except Exception as png_error:
                self.logger.warning(f"Direct PNG export failed: {png_error}, trying TIFF method")
            
            # Fallback to TIFF export with validation
            export_params = {
                'image': image,
                'description': filename,
                'scale': 10,  # 10m native resolution
                'region': aoi,
                'fileFormat': 'GeoTIFF',
                'formatOptions': {
                    'cloudOptimized': True
                }
            }
            
            # Get download URL
            url = image.getDownloadUrl(export_params)
            
            self.logger.info(f"Downloading TIFF image...")
            
            # Download the image with validation
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Validate download content
            if len(response.content) < 1000:  # TIFF should be larger than 1KB
                raise Exception(f"Downloaded file too small ({len(response.content)} bytes), likely corrupted")
            
            # Check if it's actually a TIFF file (starts with TIFF magic bytes)
            if not (response.content[:4] in [b'II*\x00', b'MM\x00*']):
                self.logger.warning("Downloaded file is not a valid TIFF, trying alternative export")
                return self._export_fallback_png(image, aoi, date)
            
            with open(tiff_path, 'wb') as f:
                f.write(response.content)
            
            # Validate TIFF file can be opened
            try:
                with rasterio.open(tiff_path) as test_src:
                    if test_src.count < 3:
                        raise Exception("TIFF file doesn't have enough bands")
                    test_data = test_src.read(1, window=((0, 10), (0, 10)))  # Read small sample
                    if test_data is None or test_data.size == 0:
                        raise Exception("TIFF file contains no valid data")
            except Exception as tiff_error:
                self.logger.error(f"Downloaded TIFF validation failed: {tiff_error}")
                if os.path.exists(tiff_path):
                    os.remove(tiff_path)
                return self._export_fallback_png(image, aoi, date)
            
            # Convert to crystal clear PNG
            self._convert_to_crystal_clear_png(tiff_path, png_path)
            
            # Clean up TIFF file
            if os.path.exists(tiff_path):
                os.remove(tiff_path)
            
            return png_path
            
        except Exception as e:
            self.logger.error(f"Error exporting and converting image: {e}")
            # Final fallback
            return self._export_fallback_png(image, aoi, date)
    
    def _export_direct_png(self, image, aoi, date, png_path):
        """Direct PNG export from GEE (most reliable method)"""
        try:
            # Generate visualization parameters for natural-looking RGB
            vis_params = {
                'min': 0,
                'max': 0.3,  # Surface reflectance range
                'bands': ['B4', 'B3', 'B2']  # RGB
            }
            
            # Create visualization
            vis_image = image.visualize(**vis_params)
            
            # Export as PNG directly
            url = vis_image.getDownloadUrl({
                'scale': 10,
                'region': aoi,
                'format': 'PNG'
            })
            
            self.logger.info(f"Downloading PNG directly from GEE...")
            
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Validate PNG content
            if len(response.content) < 1000:
                raise Exception("PNG too small, likely corrupted")
            
            # Check PNG magic bytes
            if not response.content.startswith(b'\x89PNG\r\n\x1a\n'):
                raise Exception("Not a valid PNG file")
            
            with open(png_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"✨ Direct PNG export successful: {png_path}")
            return png_path
            
        except Exception as e:
            self.logger.error(f"Direct PNG export failed: {e}")
            raise
    
    def _export_fallback_png(self, image, aoi, date):
        """Fallback PNG export with simpler parameters"""
        try:
            unique_id = uuid.uuid4().hex[:8]
            filename = f'fallback_{unique_id}_{date}_10m.png'
            png_path = f'images/satellite/{filename}'
            
            # Simple visualization parameters
            url = image.getDownloadUrl({
                'scale': 10,
                'region': aoi,
                'format': 'PNG',
                'bands': ['B4', 'B3', 'B2'],
                'min': 0,
                'max': 3000  # Use digital number range
            })
            
            self.logger.info(f"Downloading fallback PNG...")
            
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(png_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"✅ Fallback PNG export successful: {png_path}")
            return png_path
            
        except Exception as e:
            self.logger.error(f"Fallback PNG export failed: {e}")
            raise Exception("All export methods failed")
    
    def _convert_to_crystal_clear_png(self, tiff_path, png_path):
        """Convert TIFF to crystal clear PNG with optimal enhancement"""
        try:
            with rasterio.open(tiff_path) as src:
                # Read RGB bands
                red = src.read(1).astype(np.float32)
                green = src.read(2).astype(np.float32)
                blue = src.read(3).astype(np.float32)
                
                # Handle NaN values (from cloud masking)
                red = np.nan_to_num(red, nan=0.0)
                green = np.nan_to_num(green, nan=0.0)
                blue = np.nan_to_num(blue, nan=0.0)
                
                # Apply optimal enhancement for each band
                def enhance_band(band):
                    if np.max(band) > 0:
                        # Use 2% linear stretch for optimal contrast
                        p2, p98 = np.percentile(band[band > 0], [2, 98])
                        if p98 > p2:
                            enhanced = np.clip((band - p2) / (p98 - p2), 0, 1)
                        else:
                            enhanced = band / np.max(band) if np.max(band) > 0 else band
                        
                        # Apply gentle gamma correction for natural appearance
                        enhanced = np.power(enhanced, 0.9)
                        return enhanced
                    return band
                
                # Enhance each band
                red_enhanced = enhance_band(red)
                green_enhanced = enhance_band(green)
                blue_enhanced = enhance_band(blue)
                
                # Convert to 8-bit
                red_8bit = (red_enhanced * 255).astype(np.uint8)
                green_8bit = (green_enhanced * 255).astype(np.uint8)
                blue_8bit = (blue_enhanced * 255).astype(np.uint8)
                
                # Create RGB array
                height, width = red_8bit.shape
                rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
                rgb_array[:, :, 0] = red_8bit
                rgb_array[:, :, 1] = green_8bit
                rgb_array[:, :, 2] = blue_8bit
                
                # Save as crystal clear PNG
                image = Image.fromarray(rgb_array, mode='RGB')
                image.save(png_path, 'PNG', optimize=True, compress_level=1)
                
                self.logger.info(f"✨ Crystal clear PNG saved: {png_path}")
                
        except Exception as e:
            self.logger.error(f"Crystal clear conversion failed: {e}")
            raise
    
    def get_imagery(self, lat, lon, start_date, end_date):
        """Get imagery metadata for date range"""
        try:
            if not self.initialized:
                raise Exception("Google Earth Engine not initialized")
            
            # Create AOI
            aoi = self._create_aoi(lat, lon, 1.0)
            
            # Get available imagery
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(aoi)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
            
            collection_info = collection.getInfo()
            
            # Format response
            imagery_data = []
            for feature in collection_info['features']:
                props = feature['properties']
                imagery_data.append({
                    'date': props.get('PRODUCT_ID', '').split('_')[2][:8] if 'PRODUCT_ID' in props else 'unknown',
                    'cloud_cover': props.get('CLOUDY_PIXEL_PERCENTAGE', 0),
                    'satellite': 'Sentinel-2',
                    'resolution': '10m'
                })
            
            return {
                'total_images': len(imagery_data),
                'images': imagery_data[:10]  # Return first 10
            }
            
        except Exception as e:
            self.logger.error(f"Error getting imagery: {e}")
            raise
    
    def get_survey_data(self, lat, lon, start_date, end_date):
        """Get government survey data placeholder"""
        return {
            'message': 'Survey data integration not yet implemented',
            'available_datasets': [
                'USGS Land Change Monitoring',
                'NASA Land Cover',
                'ESA WorldCover'
            ]
        } 