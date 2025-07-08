"""
Google Earth Engine Service for Satellite Imagery and Change Detection
Handles real satellite data fetching, preprocessing, and authentication
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

class GEEService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.demo_mode = False
        
        # Create necessary directories
        os.makedirs('images/satellite', exist_ok=True)
        os.makedirs('images/demo', exist_ok=True)
        
        self._initialize_gee()
    
    def _initialize_gee(self):
        """Initialize Google Earth Engine with authentication"""
        try:
            # Try different authentication methods
            service_account_key = os.environ.get('GEE_SERVICE_ACCOUNT_KEY')
            
            if service_account_key:
                # Use service account authentication
                self._authenticate_with_service_account(service_account_key)
            else:
                # Try standard authentication methods
                self._authenticate_standard()
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Earth Engine: {e}")
            # No demo mode fallback - require real GEE authentication
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
                
                # Test connection with a simple query
                test_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
                count = test_collection.size().getInfo()
                self.logger.info(f"📡 Successfully connected to Sentinel-2 data ({count} test image)")
                return
                
            except Exception as e:
                self.logger.warning(f"User project failed: {e}")
            
            # Method 1b: Try with environment variable project ID
            try:
                import os
                project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'caramel-goal-464010-u8')
                ee.Initialize(project=project_id)
                self.initialized = True
                self.logger.info(f"✅ Google Earth Engine initialized with project: {project_id}")
                
                # Test connection with a simple query
                test_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
                count = test_collection.size().getInfo()
                self.logger.info(f"📡 Successfully connected to Sentinel-2 data ({count} test image)")
                return
                
            except Exception as e:
                self.logger.warning(f"Environment project failed: {e}")
            
            # Method 1b: Try simple initialization (legacy mode)
            try:
                ee.Initialize()
                self.initialized = True
                self.logger.info("✅ Google Earth Engine initialized in legacy mode")
                
                # Test connection with a simple query
                test_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
                count = test_collection.size().getInfo()
                self.logger.info(f"📡 Successfully connected to Sentinel-2 data ({count} test image)")
                return
                
            except Exception as e:
                self.logger.warning(f"Legacy initialization failed: {e}")
            
            # Method 2: Try with explicit authentication
            try:
                # Force re-authentication if needed
                ee.Authenticate(force=True)
                ee.Initialize()
                self.initialized = True
                self.logger.info("✅ Google Earth Engine re-authenticated and initialized")
                return
                
            except Exception as e:
                self.logger.warning(f"Re-authentication failed: {e}")
            
            # Method 3: Try with Cloud Project (if user has one)
            try:
                # This will work if user has access to a Cloud Project with EE enabled
                import os
                project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
                if project_id:
                    ee.Initialize(project=project_id)
                    self.initialized = True
                    self.logger.info(f"✅ Google Earth Engine initialized with project: {project_id}")
                    return
            except Exception as e:
                self.logger.warning(f"Cloud project initialization failed: {e}")
                
            # If all methods fail, provide clear instructions
            self.logger.error("🚨 Google Earth Engine authentication failed!")
            self.logger.error("To use real satellite images, please:")
            self.logger.error("1. Run: earthengine authenticate")
            self.logger.error("2. Follow the browser authentication flow")
            self.logger.error("3. Restart this application")
            self.logger.error("OR set GOOGLE_CLOUD_PROJECT environment variable")
            
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
    
    def _setup_demo_mode(self):
        """Setup demo mode with sample satellite images"""
        self.demo_mode = True
        self.logger.info("Setting up demo mode - using sample satellite images")
        
        # Create sample satellite images if they don't exist
        self._create_demo_images()
    
    def _create_demo_images(self):
        """Create realistic demo satellite images for testing"""
        demo_dir = 'images/demo'
        
        # Check if demo images already exist
        demo_files = [
            'mumbai_2015_sample.png',
            'mumbai_2025_sample.png'
        ]
        
        if all(os.path.exists(os.path.join(demo_dir, f)) for f in demo_files):
            self.logger.info("Demo images already exist")
            return
        
        # Create synthetic but realistic satellite images
        self.logger.info("Creating demo satellite images...")
        
        # Create a 1km x 1km sample area (assume 10m pixels = 100x100 pixels)
        img_size = (400, 400)  # Higher resolution for better visualization
        
        # Before image (2015) - more vegetation
        before_img = self._create_realistic_satellite_image(img_size, year=2015)
        before_path = os.path.join(demo_dir, 'mumbai_2015_sample.png')
        Image.fromarray(before_img).save(before_path)
        
        # After image (2025) - more urban development
        after_img = self._create_realistic_satellite_image(img_size, year=2025)
        after_path = os.path.join(demo_dir, 'mumbai_2025_sample.png')
        Image.fromarray(after_img).save(after_path)
        
        self.logger.info(f"Demo images created at {demo_dir}/")
    
    def _create_realistic_satellite_image(self, size, year):
        """Create a realistic satellite image with proper RGB values"""
        height, width = size
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Base vegetation (greenish-brown)
        base_color = [85, 107, 47] if year == 2015 else [139, 101, 71]  # More urban in 2025
        img[:, :] = base_color
        
        # Add vegetation patches (more in 2015)
        vegetation_coverage = 0.6 if year == 2015 else 0.3
        for _ in range(int(20 * vegetation_coverage)):
            x = np.random.randint(50, width-50)
            y = np.random.randint(50, height-50)
            radius = np.random.randint(15, 40)
            
            # Create vegetation patch
            y_coords, x_coords = np.ogrid[:height, :width]
            mask = (x_coords - x)**2 + (y_coords - y)**2 <= radius**2
            img[mask] = [34, 139, 34]  # Green vegetation
        
        # Add water bodies (blue)
        for _ in range(2):
            x = np.random.randint(100, width-100)
            y = np.random.randint(100, height-100)
            w, h = np.random.randint(30, 80), np.random.randint(20, 60)
            img[y:y+h, x:x+w] = [65, 105, 225]  # Water blue
        
        # Add urban areas (more in 2025)
        urban_coverage = 0.2 if year == 2015 else 0.5
        for _ in range(int(15 * urban_coverage)):
            x = np.random.randint(20, width-50)
            y = np.random.randint(20, height-50)
            w, h = np.random.randint(20, 60), np.random.randint(20, 60)
            
            # Urban/concrete gray
            urban_color = [128, 128, 128] if year == 2025 else [160, 160, 160]
            img[y:y+h, x:x+w] = urban_color
        
        # Add roads (more in 2025)
        road_count = 3 if year == 2015 else 6
        for _ in range(road_count):
            if np.random.random() > 0.5:  # Horizontal road
                y = np.random.randint(50, height-50)
                img[y-2:y+3, :] = [64, 64, 64]  # Dark gray road
            else:  # Vertical road
                x = np.random.randint(50, width-50)
                img[:, x-2:x+3] = [64, 64, 64]  # Dark gray road
        
        # Add some noise for realism
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def check_connection(self):
        """Check GEE connection status"""
        try:
            if self.initialized:
                # Test with a simple operation
                collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                count = collection.limit(1).size().getInfo()
                return "connected"
            else:
                return "not_initialized"
        except Exception as e:
            self.logger.error(f"GEE connection check failed: {e}")
            return "error"
    
    def get_preprocessed_image(self, lat, lon, date, aoi_size_km=1.0):
        """
        Get preprocessed satellite image for given location and date
        
        Args:
            lat: Latitude
            lon: Longitude  
            date: Date string (YYYY-MM-DD)
            aoi_size_km: Size of area of interest in kilometers
            
        Returns:
            Path to preprocessed image file
        """
        # Only use real satellite data - no demo mode fallback
        try:
            # Define AOI
            aoi = self._create_aoi(lat, lon, aoi_size_km)
            
            # Get best available image for the date
            image_path = self._fetch_sentinel2_image(aoi, date)
            
            return image_path
            
        except Exception as e:
            self.logger.error(f"Error getting satellite image: {e}")
            # No fallback - only real satellite data
            raise Exception(f"Failed to get real satellite image for {lat}, {lon} on {date}: {e}")
    
    def _get_demo_image(self, lat, lon, date):
        """Get demo image based on date"""
        # Determine which demo image to use based on date
        demo_dir = 'demo_images'
        
        # Parse date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        if date_obj.year < 2020:
            demo_file = 'mumbai_2015_sample.png'
        else:
            demo_file = 'mumbai_2025_sample.png'
        
        demo_path = os.path.join(demo_dir, demo_file)
        
        # Copy to images directory with unique name
        import shutil
        temp_name = f"demo_{uuid.uuid4().hex[:8]}_{date}.png"
        temp_path = os.path.join('images/satellite', temp_name)
        shutil.copy(demo_path, temp_path)
        
        self.logger.info(f"Using demo image: {demo_file} for date {date}")
        return temp_path
    
    def _create_aoi(self, lat, lon, size_km):
        """Create Area of Interest geometry"""
        # Convert km to degrees (approximate)
        size_deg = size_km / 111.32
        half_size = size_deg / 2
        
        coords = [
            [lon - half_size, lat - half_size],
            [lon + half_size, lat - half_size], 
            [lon + half_size, lat + half_size],
            [lon - half_size, lat + half_size],
            [lon - half_size, lat - half_size]
        ]
        
        return ee.Geometry.Polygon(coords)
    
    def _fetch_sentinel2_image(self, aoi, date):
        """Fetch ultra-high quality Sentinel-2 image using only 10m bands"""
        try:
            # Define date range (±30 days for optimal quality images)
            target_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Get Sentinel-2 collection with ULTRA-STRICT quality filters for crystal clear images
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(aoi)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))  # Ultra-strict cloud filter
                         .filter(ee.Filter.lt('NODATA_PIXEL_PERCENTAGE', 5))  # Minimal missing data
                         .filter(ee.Filter.gt('SUN_ELEVATION', 30))  # Good illumination
                         .sort('CLOUDY_PIXEL_PERCENTAGE')
                         .sort('system:time_start', False))  # Prefer newer images
            
            # Check if any ultra-high-quality images are available
            collection_size = collection.size()
            size_info = collection_size.getInfo()
            
            if size_info == 0:
                # Relax filters slightly but maintain high quality
                self.logger.info("No ultra-high-quality images found, relaxing filters slightly...")
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(aoi)
                             .filterDate(start_date, end_date)
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
                             .filter(ee.Filter.lt('NODATA_PIXEL_PERCENTAGE', 10))
                             .sort('CLOUDY_PIXEL_PERCENTAGE'))
                
                size_info = collection.size().getInfo()
                
                if size_info == 0:
                    # Extend date range if needed
                    start_date = (target_date - timedelta(days=90)).strftime('%Y-%m-%d')
                    end_date = (target_date + timedelta(days=90)).strftime('%Y-%m-%d')
                    
                    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                 .filterBounds(aoi)
                                 .filterDate(start_date, end_date)
                                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                                 .sort('CLOUDY_PIXEL_PERCENTAGE'))
                    
                    size_info = collection.size().getInfo()
                    
                    if size_info == 0:
                        raise Exception("No suitable Sentinel-2 images found for this location and date range")
            
            self.logger.info(f"Found {size_info} high-quality Sentinel-2 images for {date}")
            
            # Get the best quality image
            image = collection.first()
            
            # Apply advanced cloud masking for crystal clear results
            image = self._apply_advanced_cloud_mask(image)
            
            # Select ONLY 10m bands as requested: B2=Blue, B3=Green, B4=Red
            rgb_image = image.select(['B4', 'B3', 'B2'])  # Red, Green, Blue order
            
            # Apply ultra-high quality enhancement to match target image
            rgb_image = self._enhance_to_target_quality(rgb_image)
            
            # Clip to AOI
            rgb_image = rgb_image.clip(aoi)
            
            # Export image with maximum quality settings
            temp_path = self._export_crystal_clear_image(rgb_image, aoi, date)
            
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Error fetching ultra-high quality Sentinel-2 image: {e}")
            raise
    
    def _apply_advanced_cloud_mask(self, image):
        """Apply advanced cloud masking for crystal clear satellite images"""
        # Use QA60 band for cloud masking
        qa = image.select('QA60')
        
        # Bits 10 and 11 are clouds and cirrus
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        
        # Create comprehensive mask
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        
        # Additional cloud shadow masking using SCL band if available
        try:
            scl = image.select('SCL')
            # Mask out clouds (9), cloud shadows (3), and saturated/defective pixels (1)
            scl_mask = scl.neq(1).And(scl.neq(3)).And(scl.neq(9)).And(scl.neq(8)).And(scl.neq(10))
            mask = mask.And(scl_mask)
        except:
            pass  # SCL band might not be available in all collections
        
        return image.updateMask(mask)
    
    def _enhance_to_target_quality(self, image):
        """Enhance image to match target ultra-high quality with only 10m bands"""
        # Convert surface reflectance values (scale from 0-10000 to 0-1)
        image = image.multiply(0.0001)
        
        # Apply atmospheric correction enhancement
        # Use 2% linear stretch for each band to maximize contrast
        def enhance_band(band_image, band_name):
            # Calculate percentiles for contrast stretching
            percentiles = band_image.reduceRegion(**{
                'reducer': ee.Reducer.percentile([1, 99]),
                'scale': 10,
                'maxPixels': 1e9,
                'bestEffort': True
            })
            
            # Get percentile values
            p1 = ee.Number(percentiles.get(f'{band_name}_p1')).max(0.001)
            p99 = ee.Number(percentiles.get(f'{band_name}_p99')).min(0.999)
            
            # Apply linear stretch
            stretched = band_image.subtract(p1).divide(p99.subtract(p1)).clamp(0, 1)
            
            return stretched
        
        # Enhance each 10m band individually
        b4_enhanced = enhance_band(image.select('B4'), 'B4')  # Red (10m)
        b3_enhanced = enhance_band(image.select('B3'), 'B3')  # Green (10m)
        b2_enhanced = enhance_band(image.select('B2'), 'B2')  # Blue (10m)
        
        # Combine enhanced bands
        enhanced = ee.Image.cat([b4_enhanced, b3_enhanced, b2_enhanced]).rename(['B4', 'B3', 'B2'])
        
        # Apply subtle gamma correction for natural appearance
        gamma = 1.2
        enhanced = enhanced.pow(1.0/gamma)
        
        # Final contrast enhancement
        enhanced = enhanced.multiply(1.1).clamp(0, 1)
        
        return enhanced
    
    def _export_crystal_clear_image(self, image, aoi, date):
        """Export crystal clear image with maximum resolution within GEE limits"""
        
        # Test ultra-high resolutions focusing on 10m bands
        # Start with highest quality and work down to stay under 50MB
        resolutions = [2, 3, 5, 8, 10]  # Ultra-high to high resolution
        
        for scale in resolutions:
            try:
                self.logger.info(f"Testing {scale}m resolution for crystal clear quality (10m bands only)...")
                
                # Method 1: Try with region and scale (no dimensions)
                try:
                    url = image.getDownloadURL({
                        'scale': scale,
                        'crs': 'EPSG:4326', 
                        'region': aoi,
                        'format': 'GEO_TIFF',
                        'filePerBand': False
                    })
                    
                    # Check file size first
                    head_response = requests.head(url, timeout=30)
                    file_size = int(head_response.headers.get('content-length', 0))
                    file_size_mb = file_size / (1024 * 1024)
                    
                    self.logger.info(f"Crystal clear image at {scale}m resolution: {file_size_mb:.1f} MB")
                    
                    if file_size_mb <= 45:  # Keep under 50MB limit
                        # Download the crystal clear image
                        self.logger.info(f"✅ Using {scale}m resolution ({file_size_mb:.1f} MB) for crystal clear quality")
                        response = requests.get(url, timeout=180)
                        response.raise_for_status()
                        
                        # Save to images file with clear naming
                        temp_name = f"crystal_clear_{uuid.uuid4().hex[:8]}_{date}_{scale}m_10mbands.tif"
                        temp_path = os.path.join('images/satellite', temp_name)
                        
                        with open(temp_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Convert to crystal clear PNG
                        png_path = temp_path.replace('.tif', '.png')
                        self._convert_to_crystal_clear_png(temp_path, png_path, scale)
                        
                        # Clean up TIFF file
                        os.remove(temp_path)
                        
                        return png_path
                        
                    else:
                        self.logger.info(f"❌ {scale}m resolution too large ({file_size_mb:.1f} MB), trying lower resolution...")
                        continue
                
                except Exception as region_error:
                    self.logger.info(f"Region method failed for {scale}m: {region_error}")
                    
                    # Method 2: Try with dimensions only (no region)
                    try:
                        # Calculate optimal dimensions based on scale and 1km AOI
                        pixels_per_km = 1000 / scale  # pixels per kilometer
                        max_dimension = min(int(pixels_per_km), 512)  # Cap at 512 for reasonable file size
                        
                        url = image.getDownloadURL({
                            'scale': scale,
                            'crs': 'EPSG:4326',
                            'format': 'GEO_TIFF',
                            'filePerBand': False,
                            'dimensions': f'{max_dimension}x{max_dimension}'
                        })
                        
                        # Check file size
                        head_response = requests.head(url, timeout=30)
                        file_size = int(head_response.headers.get('content-length', 0))
                        file_size_mb = file_size / (1024 * 1024)
                        
                        self.logger.info(f"Crystal clear image at {scale}m resolution (dimensions method): {file_size_mb:.1f} MB")
                        
                        if file_size_mb <= 45:
                            self.logger.info(f"✅ Using {scale}m resolution ({file_size_mb:.1f} MB) with dimensions method")
                            response = requests.get(url, timeout=180)
                            response.raise_for_status()
                            
                            # Save to images file
                            temp_name = f"crystal_clear_{uuid.uuid4().hex[:8]}_{date}_{scale}m_10mbands.tif"
                            temp_path = os.path.join('images/satellite', temp_name)
                            
                            with open(temp_path, 'wb') as f:
                                f.write(response.content)
                            
                            # Convert to crystal clear PNG
                            png_path = temp_path.replace('.tif', '.png')
                            self._convert_to_crystal_clear_png(temp_path, png_path, scale)
                            
                            # Clean up TIFF file
                            os.remove(temp_path)
                            
                            return png_path
                        else:
                            self.logger.info(f"❌ Dimensions method also too large, trying next resolution...")
                            continue
                            
                    except Exception as dim_error:
                        self.logger.warning(f"Both region and dimensions methods failed for {scale}m: {dim_error}")
                        continue
                     
            except Exception as e:
                self.logger.warning(f"Failed to export crystal clear image at {scale}m resolution: {e}")
                continue
        
        # If all ultra-high resolutions fail, use enhanced standard export
        self.logger.warning("Ultra-high quality export failed, using enhanced standard quality...")
        return self._export_enhanced_standard_image(image, aoi, date)
    
    def _export_enhanced_standard_image(self, image, aoi, date):
        """Enhanced standard export with good quality"""
        try:
            url = image.getDownloadURL({
                'scale': 10,  # Use native 10m resolution 
                'crs': 'EPSG:4326',
                'region': aoi,
                'format': 'GEO_TIFF'
            })
            
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            temp_name = f"enhanced_std_{uuid.uuid4().hex[:8]}_{date}_10m.tif"
            temp_path = os.path.join('images/satellite', temp_name)
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            png_path = temp_path.replace('.tif', '.png')
            self._convert_to_crystal_clear_png(temp_path, png_path, 10)
            
            os.remove(temp_path)
            return png_path
            
        except Exception as e:
            self.logger.error(f"Enhanced standard export failed: {e}")
            raise
    
    def _convert_tiff_to_png(self, tiff_path, png_path):
        """Convert GeoTIFF to PNG with proper scaling"""
        import rasterio
        from PIL import Image as PILImage
        
        try:
            with rasterio.open(tiff_path) as src:
                # Read RGB bands
                red = src.read(1)
                green = src.read(2) 
                blue = src.read(3)
                
                # Stack and normalize
                rgb = np.stack([red, green, blue], axis=-1)
                
                # Clip and scale to 0-255
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                
                # Save as PNG
                PILImage.fromarray(rgb).save(png_path)
                
        except Exception as e:
            self.logger.error(f"Error converting TIFF to PNG: {e}")
            # Fallback: create a simple conversion
            try:
                img = PILImage.open(tiff_path)
                img.save(png_path)
            except Exception as fallback_e:
                self.logger.error(f"Fallback conversion also failed: {fallback_e}")
                raise Exception("Failed to convert image format")
    
    def _export_ee_image(self, image, aoi, date):
        """Export Earth Engine image to local file"""
        try:
            # Get download URL
            url = image.getDownloadURL({
                'scale': 10,  # 10m resolution
                'crs': 'EPSG:4326',
                'region': aoi,
                'format': 'GEO_TIFF'
            })
            
            # Download image
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Save to images file
            temp_name = f"gee_image_{uuid.uuid4().hex[:8]}_{date}.tif"
            temp_path = os.path.join('images/satellite', temp_name)
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Convert to PNG for easier handling
            png_path = temp_path.replace('.tif', '.png')
            self._convert_tiff_to_png(temp_path, png_path)
            
            # Clean up TIFF file
            os.remove(temp_path)
            
            return png_path
            
        except Exception as e:
            self.logger.error(f"Error exporting EE image: {e}")
            raise
    
    def _convert_tiff_to_png_hq(self, tiff_path, png_path, resolution):
        """Convert GeoTIFF to high-quality PNG with advanced satellite image processing"""
        import rasterio
        from PIL import Image as PILImage
        import numpy as np
        from skimage import exposure, filters
        
        try:
            with rasterio.open(tiff_path) as src:
                self.logger.info(f"Processing {resolution}m resolution image: {src.width}x{src.height} pixels")
                
                # Read RGB bands
                red = src.read(1).astype(np.float32)
                green = src.read(2).astype(np.float32) 
                blue = src.read(3).astype(np.float32)
                
                # Handle NaN/NoData values
                red = np.nan_to_num(red, nan=0.0, posinf=1.0, neginf=0.0)
                green = np.nan_to_num(green, nan=0.0, posinf=1.0, neginf=0.0)
                blue = np.nan_to_num(blue, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Apply advanced enhancement techniques
                
                # 1. Adaptive histogram equalization for each band
                red_eq = exposure.equalize_adapthist(red, clip_limit=0.03)
                green_eq = exposure.equalize_adapthist(green, clip_limit=0.03)
                blue_eq = exposure.equalize_adapthist(blue, clip_limit=0.03)
                
                # 2. Apply gentle gaussian filter to reduce noise
                red_smooth = filters.gaussian(red_eq, sigma=0.5)
                green_smooth = filters.gaussian(green_eq, sigma=0.5)
                blue_smooth = filters.gaussian(blue_eq, sigma=0.5)
                
                # 3. Enhance contrast with gamma correction
                gamma = 0.8  # Brighten the image
                red_gamma = np.power(red_smooth, gamma)
                green_gamma = np.power(green_smooth, gamma)
                blue_gamma = np.power(blue_smooth, gamma)
                
                # 4. Final scaling to 0-255 with robust percentile stretching
                def robust_scale(band, lower=1, upper=99):
                    p_low, p_high = np.percentile(band[band > 0], [lower, upper])
                    band_scaled = np.clip((band - p_low) / (p_high - p_low), 0, 1)
                    return (band_scaled * 255).astype(np.uint8)
                
                red_final = robust_scale(red_gamma)
                green_final = robust_scale(green_gamma)
                blue_final = robust_scale(blue_gamma)
                
                # Stack channels
                rgb_enhanced = np.stack([red_final, green_final, blue_final], axis=-1)
                
                # Save as high-quality PNG
                pil_image = PILImage.fromarray(rgb_enhanced)
                pil_image.save(png_path, 'PNG', optimize=True, compress_level=6)
                
                self.logger.info(f"✅ High-quality PNG saved: {png_path}")
                
        except Exception as e:
            self.logger.error(f"Error in high-quality conversion: {e}")
            # Fallback to basic conversion
            try:
                img = PILImage.open(tiff_path)
                img.save(png_path)
                self.logger.info("Used fallback PNG conversion")
            except Exception as fallback_e:
                self.logger.error(f"Fallback conversion failed: {fallback_e}")
                raise Exception("Failed to convert image format")
    
    def _export_ee_image_fallback(self, image, aoi, date):
        """Fallback export method if high-quality export fails"""
        try:
            url = image.getDownloadURL({
                'scale': 20,  # Conservative resolution
                'crs': 'EPSG:4326',
                'region': aoi,
                'format': 'GEO_TIFF'
            })
            
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            
            temp_name = f"gee_fallback_{uuid.uuid4().hex[:8]}_{date}.tif"
            temp_path = os.path.join('images/satellite', temp_name)
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            png_path = temp_path.replace('.tif', '.png')
            self._convert_tiff_to_png_hq(temp_path, png_path, 20)
            
            os.remove(temp_path)
            return png_path
            
        except Exception as e:
            self.logger.error(f"Fallback export failed: {e}")
            raise
    
    def get_imagery(self, lat, lon, start_date, end_date):
        """Get imagery data for a location and date range"""
        try:
            # Create AOI
            aoi = self._create_aoi(lat, lon, 1.0)
            
            # Get available images
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(aoi)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
            
            # Get image info
            image_list = collection.getInfo()
            
            available_dates = []
            if image_list and 'features' in image_list:
                for feature in image_list['features']:
                    date = feature['properties'].get('PRODUCT_ID', '').split('_')[2][:8]
                    if date:
                        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                        available_dates.append(formatted_date)
            
            return {
                'status': 'success',
                'available_dates': sorted(list(set(available_dates))),
                'total_images': len(available_dates)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting imagery data: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_survey_data(self, lat, lon, start_date, end_date):
        """Get available survey data (mock implementation)"""
        # Mock survey data for demonstration
        survey_data = []
        
        # Generate some mock survey dates
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        while current <= end:
            if current.day == 1 or current.day == 15:  # Bi-monthly surveys
                survey_data.append({
                    'date': current.strftime('%Y-%m-%d'),
                    'satellite': 'Sentinel-2',
                    'cloud_cover': np.random.randint(5, 25),
                    'resolution': '10m',
                    'bands': ['RGB', 'NIR']
                })
            current += timedelta(days=14)
        
        return survey_data[:10]  # Limit to 10 entries 

    def _convert_to_crystal_clear_png(self, tiff_path, png_path, resolution):
        """Convert GeoTIFF to crystal clear PNG matching target image quality"""
        import rasterio
        from PIL import Image as PILImage, ImageEnhance
        import numpy as np
        from skimage import exposure, restoration
        
        try:
            with rasterio.open(tiff_path) as src:
                self.logger.info(f"Processing crystal clear {resolution}m resolution image: {src.width}x{src.height} pixels (10m bands only)")
                
                # Read RGB bands (already in correct order from GEE processing)
                red = src.read(1).astype(np.float64)
                green = src.read(2).astype(np.float64) 
                blue = src.read(3).astype(np.float64)
                
                # Handle NaN/NoData values and invalid pixels
                def clean_band(band):
                    band = np.nan_to_num(band, nan=0.0, posinf=1.0, neginf=0.0)
                    # Remove extreme outliers
                    band = np.clip(band, 0, 1)
                    return band
                
                red = clean_band(red)
                green = clean_band(green)
                blue = clean_band(blue)
                
                # Apply ULTRA-HIGH QUALITY processing to match target image
                
                # 1. Advanced histogram equalization per band (like professional satellite software)
                def ultra_enhance_band(band):
                    if np.max(band) > 0:
                        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                        enhanced = exposure.equalize_adapthist(band, 
                                                             kernel_size=band.shape[0]//8,
                                                             clip_limit=0.01,
                                                             nbins=256)
                        # Additional contrast stretching
                        p2, p98 = np.percentile(enhanced[enhanced > 0], [0.5, 99.5])
                        enhanced = exposure.rescale_intensity(enhanced, in_range=(p2, p98))
                        return enhanced
                    return band
                
                red_ultra = ultra_enhance_band(red)
                green_ultra = ultra_enhance_band(green)
                blue_ultra = ultra_enhance_band(blue)
                
                # 2. Apply denoising for crystal clear appearance
                def denoise_band(band):
                    # Use gaussian filter for noise reduction while preserving edges
                    if np.max(band) > 0:
                        # Apply gentle gaussian denoising instead of wiener
                        from scipy import ndimage
                        denoised = ndimage.gaussian_filter(band, sigma=0.5)
                        return np.clip(denoised, 0, 1)
                    return band
                
                red_clean = denoise_band(red_ultra)
                green_clean = denoise_band(green_ultra)
                blue_clean = denoise_band(blue_ultra)
                
                # 3. Final sharpening for crystal clear details
                def sharpen_band(band):
                    # Create unsharp mask for sharpening
                    from scipy import ndimage
                    blurred = ndimage.gaussian_filter(band, sigma=1.0)
                    sharpened = band + 0.5 * (band - blurred)
                    return np.clip(sharpened, 0, 1)
                
                red_sharp = sharpen_band(red_clean)
                green_sharp = sharpen_band(green_clean)
                blue_sharp = sharpen_band(blue_clean)
                
                # 4. Advanced color balancing for natural appearance
                def color_balance(r, g, b):
                    # Calculate gray world assumption correction
                    r_mean = np.mean(r[r > 0]) if np.any(r > 0) else 0.5
                    g_mean = np.mean(g[g > 0]) if np.any(g > 0) else 0.5
                    b_mean = np.mean(b[b > 0]) if np.any(b > 0) else 0.5
                    
                    # Target gray level
                    target_gray = (r_mean + g_mean + b_mean) / 3
                    
                    # Apply correction factors
                    if r_mean > 0: r = r * (target_gray / r_mean)
                    if g_mean > 0: g = g * (target_gray / g_mean)
                    if b_mean > 0: b = b * (target_gray / b_mean)
                    
                    return np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)
                
                red_balanced, green_balanced, blue_balanced = color_balance(red_sharp, green_sharp, blue_sharp)
                
                # 5. Final gamma correction for optimal brightness (matching target image)
                gamma = 0.9  # Slightly brighten
                red_final = np.power(red_balanced, gamma)
                green_final = np.power(green_balanced, gamma)
                blue_final = np.power(blue_balanced, gamma)
                
                # 6. Convert to 8-bit with optimal scaling
                def optimal_8bit_conversion(band):
                    # Use robust percentile-based scaling for best results
                    if np.max(band) > 0:
                        p1, p99 = np.percentile(band[band > 0], [0.1, 99.9])
                        scaled = (band - p1) / (p99 - p1)
                        scaled = np.clip(scaled, 0, 1)
                        return (scaled * 255).astype(np.uint8)
                    else:
                        return (band * 255).astype(np.uint8)
                
                red_8bit = optimal_8bit_conversion(red_final)
                green_8bit = optimal_8bit_conversion(green_final)
                blue_8bit = optimal_8bit_conversion(blue_final)
                
                # 7. Stack and create final crystal clear image
                rgb_crystal_clear = np.stack([red_8bit, green_8bit, blue_8bit], axis=-1)
                
                # 8. Final PIL enhancement for crystal clear finish
                pil_image = PILImage.fromarray(rgb_crystal_clear, 'RGB')
                
                # Apply subtle sharpening filter
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.1)
                
                # Apply subtle contrast enhancement
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.05)
                
                # Save as maximum quality PNG
                pil_image.save(png_path, 'PNG', optimize=True, compress_level=1)
                
                self.logger.info(f"✅ CRYSTAL CLEAR PNG saved: {png_path} ({resolution}m resolution, 10m bands)")
                
        except Exception as e:
            self.logger.error(f"Error in crystal clear conversion: {e}")
            # Fallback to basic high-quality conversion
            try:
                with rasterio.open(tiff_path) as src:
                    # Basic but enhanced conversion
                    red = src.read(1)
                    green = src.read(2)
                    blue = src.read(3)
                    
                    # Simple enhancement
                    rgb = np.stack([red, green, blue], axis=-1)
                    rgb = exposure.equalize_adapthist(rgb.astype(np.float64))
                    rgb = (rgb * 255).astype(np.uint8)
                    
                    PILImage.fromarray(rgb).save(png_path)
                    self.logger.info("Used enhanced fallback PNG conversion")
            except Exception as fallback_e:
                self.logger.error(f"Enhanced fallback conversion failed: {fallback_e}")
                raise Exception("Failed to convert to crystal clear format") 