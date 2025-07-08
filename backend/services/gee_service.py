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
        """Fetch RAW Sentinel-2 image using only 10m bands with minimal processing"""
        try:
            # Define date range (±15 days for best temporal match)
            target_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = (target_date - timedelta(days=15)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=15)).strftime('%Y-%m-%d')
            
            # Get Sentinel-2 collection with basic quality filters only
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(aoi)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Basic cloud filter
                         .sort('system:time_start', False))  # Get most recent
            
            # Check if any images are available
            collection_size = collection.size().getInfo()
            
            if collection_size == 0:
                # Extend date range if needed
                start_date = (target_date - timedelta(days=60)).strftime('%Y-%m-%d')
                end_date = (target_date + timedelta(days=60)).strftime('%Y-%m-%d')
                
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(aoi)
                             .filterDate(start_date, end_date)
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))  # Relaxed for availability
                             .sort('system:time_start', False))
                
                collection_size = collection.size().getInfo()
                
                if collection_size == 0:
                    raise Exception("No Sentinel-2 images found for this location and date range")
            
            self.logger.info(f"Found {collection_size} Sentinel-2 images for {date}")
            
            # Get the most recent image
            image = collection.first()
            
            # MINIMAL processing - only basic cloud mask and band selection
            image = self._apply_basic_cloud_mask(image)
            
            # Select ONLY 10m bands: B2=Blue, B3=Green, B4=Red (RAW)
            rgb_image = image.select(['B4', 'B3', 'B2'])  # Red, Green, Blue order
            
            # Convert surface reflectance to 0-1 range (minimal processing)
            rgb_image = rgb_image.multiply(0.0001).clamp(0, 1)
            
            # Clip to AOI
            rgb_image = rgb_image.clip(aoi)
            
            # Export RAW image with maximum resolution
            temp_path = self._export_raw_image(rgb_image, aoi, date)
            
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Error fetching RAW Sentinel-2 image: {e}")
            raise
    
    def _apply_basic_cloud_mask(self, image):
        """Apply minimal cloud masking to preserve raw data quality"""
        # Use only QA60 band for basic cloud masking
        qa = image.select('QA60')
        
        # Bits 10 and 11 are clouds and cirrus (basic masking only)
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        
        # Create basic mask - only remove obvious clouds
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        
        return image.updateMask(mask)
    
    def _export_raw_image(self, image, aoi, date):
        """Export RAW satellite image at maximum resolution within GEE limits"""
        try:
            # Generate unique filename for raw image
            unique_id = uuid.uuid4().hex[:8]
            filename = f'raw_sentinel2_{unique_id}_{date}_10m_bands'
            tiff_path = f'temp/gee_images/{filename}.tif'
            png_path = f'images/satellite/{filename}.png'
            
            # Ensure directories exist
            os.makedirs('temp/gee_images', exist_ok=True)
            os.makedirs('images/satellite', exist_ok=True)
            
            # Get AOI bounds for maximum resolution calculation
            bounds = aoi.bounds().getInfo()['coordinates'][0]
            min_lon, max_lon = bounds[0][0], bounds[2][0] 
            min_lat, max_lat = bounds[0][1], bounds[2][1]
            
            # Calculate optimal scale for maximum resolution within GEE limits
            # Sentinel-2 10m bands native resolution is 10m
            # GEE export limit is typically 100MB for getDownloadURL
            optimal_scale = 10  # Use native 10m resolution for 10m bands
            
            # Calculate dimensions to check GEE limits
            aoi_width_deg = max_lon - min_lon
            aoi_height_deg = max_lat - min_lat
            
            # Convert to meters (approximate)
            width_m = aoi_width_deg * 111320 * np.cos(np.radians((min_lat + max_lat) / 2))
            height_m = aoi_height_deg * 111320
            
            # Calculate image dimensions at 10m resolution
            width_pixels = int(width_m / optimal_scale)
            height_pixels = int(height_m / optimal_scale)
            total_pixels = width_pixels * height_pixels
            
            self.logger.info(f"Raw image dimensions: {width_pixels}x{height_pixels} pixels at {optimal_scale}m resolution")
            
            # If too large for GEE, use slightly lower resolution but maintain quality
            max_pixels = 50000000  # Conservative limit for GEE
            if total_pixels > max_pixels:
                scale_factor = np.sqrt(total_pixels / max_pixels)
                optimal_scale = optimal_scale * scale_factor
                self.logger.info(f"Adjusted to {optimal_scale:.1f}m resolution to fit GEE limits")
            
            # Export parameters for RAW image
            export_params = {
                'image': image,
                'description': filename,
                'scale': optimal_scale,
                'region': aoi,
                'fileFormat': 'GeoTIFF',
                'formatOptions': {
                    'cloudOptimized': False  # Raw format
                },
                'maxPixels': max_pixels
            }
            
            # Export using getDownloadURL for immediate processing
            url = image.getDownloadUrl({
                'scale': optimal_scale,
                'crs': 'EPSG:4326',
                'region': aoi,
                'format': 'GEO_TIFF'
            })
            
            # Download the raw image
            self.logger.info(f"Downloading RAW Sentinel-2 image...")
            response = requests.get(url)
            response.raise_for_status()
            
            # Save raw TIFF
            with open(tiff_path, 'wb') as f:
                f.write(response.content)
            
            # Convert to PNG with minimal processing (preserve raw characteristics)
            self._convert_raw_tiff_to_png(tiff_path, png_path)
            
            self.logger.info(f"RAW Sentinel-2 image saved: {png_path}")
            return png_path
            
        except Exception as e:
            self.logger.error(f"Error exporting RAW image: {e}")
            # Fallback to simpler export
            return self._export_raw_image_fallback(image, aoi, date)
    
    def _convert_raw_tiff_to_png(self, tiff_path, png_path):
        """Convert RAW TIFF to PNG with minimal processing to preserve raw characteristics"""
        try:
            import rasterio
            from PIL import Image
            
            with rasterio.open(tiff_path) as src:
                # Read all bands (B4=Red, B3=Green, B2=Blue)
                red = src.read(1).astype(np.float32)
                green = src.read(2).astype(np.float32) 
                blue = src.read(3).astype(np.float32)
                
                # Minimal processing - only basic scaling for visibility
                def raw_scale(band):
                    # Remove any zero/negative values
                    band = np.maximum(band, 0)
                    
                    # Use 2nd and 98th percentiles for minimal contrast adjustment
                    if np.max(band) > 0:
                        p2, p98 = np.percentile(band[band > 0], [2, 98])
                        # Gentle scaling to maintain raw characteristics
                        band = np.clip((band - p2) / (p98 - p2), 0, 1)
                    
                    return band
                
                # Apply minimal scaling to each band
                red_scaled = raw_scale(red)
                green_scaled = raw_scale(green)
                blue_scaled = raw_scale(blue)
                
                # Convert to 8-bit for PNG (minimal processing)
                red_8bit = (red_scaled * 255).astype(np.uint8)
                green_8bit = (green_scaled * 255).astype(np.uint8)
                blue_8bit = (blue_scaled * 255).astype(np.uint8)
                
                # Create RGB array
                height, width = red_8bit.shape
                rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
                rgb_array[:, :, 0] = red_8bit
                rgb_array[:, :, 1] = green_8bit
                rgb_array[:, :, 2] = blue_8bit
                
                # Save as PNG (preserving raw characteristics)
                image = Image.fromarray(rgb_array, mode='RGB')
                image.save(png_path, 'PNG', optimize=False)  # No optimization to preserve raw data
                
                self.logger.info(f"RAW TIFF converted to PNG: {png_path}")
                
        except Exception as e:
            self.logger.error(f"Error converting RAW TIFF to PNG: {e}")
            raise
    
    def _export_raw_image_fallback(self, image, aoi, date):
        """Fallback method for RAW image export if main method fails"""
        try:
            unique_id = uuid.uuid4().hex[:8]
            filename = f'raw_fallback_{unique_id}_{date}_10m.png'
            png_path = f'images/satellite/{filename}'
            
            # Simple export with basic parameters
            url = image.getDownloadUrl({
                'scale': 10,  # Native 10m resolution
                'crs': 'EPSG:4326',
                'region': aoi,
                'format': 'PNG'
            })
            
            response = requests.get(url)
            response.raise_for_status()
            
            with open(png_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"RAW fallback image saved: {png_path}")
            return png_path
            
        except Exception as e:
            self.logger.error(f"RAW fallback export failed: {e}")
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