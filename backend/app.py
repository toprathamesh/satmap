#!/usr/bin/env python3
"""
Change Detection Platform - Main Flask Application
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from services.gee_service import GEEService
from services.change_detection_service import ChangeDetectionService
from services.export_service import ExportService
from utils.validators import validate_coordinates, validate_date_range

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Configure Flask app
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Configure CORS
    cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://localhost:8080')
    allowed_origins = [origin.strip() for origin in cors_origins.split(',')]
    CORS(app, origins=allowed_origins)
    
    # Initialize services
    gee_service = GEEService()
    change_detection_service = ChangeDetectionService()
    export_service = ExportService()
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        try:
            gee_status = gee_service.check_connection()
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'gee': gee_status,
                    'change_detection': True
                }
            })
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
    
    @app.route('/api/aoi/imagery', methods=['POST'])
    def get_aoi_imagery():
        """Get satellite imagery for a specific AOI and date range"""
        try:
            data = request.get_json()
            
            # Validate input
            lat = data.get('latitude')
            lon = data.get('longitude')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            
            if not validate_coordinates(lat, lon):
                return jsonify({'error': 'Invalid coordinates'}), 400
            
            if not validate_date_range(start_date, end_date):
                return jsonify({'error': 'Invalid date range'}), 400
            
            # Get imagery from GEE
            imagery_data = gee_service.get_imagery(lat, lon, start_date, end_date)
            
            return jsonify({
                'success': True,
                'data': imagery_data
            })
            
        except Exception as e:
            logger.error(f"Error getting AOI imagery: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/change-detection', methods=['POST'])
    def detect_changes():
        """Perform change detection between two time periods using real satellite images"""
        try:
            data = request.get_json()
            
            # Validate input
            lat = data.get('latitude')
            lon = data.get('longitude')
            before_date = data.get('before_date')
            after_date = data.get('after_date')
            
            if not validate_coordinates(lat, lon):
                return jsonify({'error': 'Invalid coordinates'}), 400
            
            if not validate_date_range(before_date, after_date):
                return jsonify({'error': 'Invalid date range'}), 400
            
            logger.info(f"Starting change detection for {lat}, {lon} between {before_date} and {after_date}")
            
            # Get images for both time periods from GEE
            before_image_path = gee_service.get_preprocessed_image(lat, lon, before_date)
            after_image_path = gee_service.get_preprocessed_image(lat, lon, after_date)
            
            logger.info(f"Retrieved images: {before_image_path}, {after_image_path}")
            
            # Prepare metadata for change detection
            metadata = {
                'latitude': lat,
                'longitude': lon,
                'before_date': before_date,
                'after_date': after_date,
                'pixel_size_m': 10,  # Sentinel-2 resolution
                'gee_demo_mode': gee_service.demo_mode
            }
            
            # Perform change detection using Siamese U-Net
            change_result = change_detection_service.detect_changes(
                before_image_path, after_image_path, metadata
            )
            
            logger.info(f"Change detection completed: {change_result['statistics']['change_percentage']}% change detected")
            
            return jsonify({
                'success': True,
                'data': change_result
            })
            
        except Exception as e:
            logger.error(f"Error in change detection: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/survey-data', methods=['POST'])
    def get_survey_data():
        """Get available government survey data for AOI and date range"""
        try:
            data = request.get_json()
            
            lat = data.get('latitude')
            lon = data.get('longitude')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            
            if not validate_coordinates(lat, lon):
                return jsonify({'error': 'Invalid coordinates'}), 400
            
            # Get available survey data
            survey_data = gee_service.get_survey_data(lat, lon, start_date, end_date)
            
            return jsonify({
                'success': True,
                'data': survey_data
            })
            
        except Exception as e:
            logger.error(f"Error getting survey data: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/export', methods=['POST'])
    def export_results():
        """Export change detection results in GIS-compatible formats"""
        try:
            data = request.get_json()
            
            result_id = data.get('result_id')
            export_format = data.get('format', 'geotiff')  # geotiff, geojson, shapefile
            
            if not result_id:
                return jsonify({'error': 'Result ID required'}), 400
            
            # Export the results
            export_path = export_service.export_change_results(result_id, export_format)
            
            return send_file(export_path, as_attachment=True)
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/parse-gmaps-link', methods=['POST'])
    def parse_gmaps_link():
        """Parse Google Maps link to extract coordinates"""
        try:
            data = request.get_json()
            gmaps_url = data.get('url')
            
            if not gmaps_url:
                return jsonify({'error': 'URL required'}), 400
            
            # Extract coordinates from Google Maps URL
            from utils.gmaps_parser import parse_gmaps_url
            coords = parse_gmaps_url(gmaps_url)
            
            if not coords:
                return jsonify({'error': 'Could not extract coordinates from URL'}), 400
            
            return jsonify({
                'success': True,
                'latitude': coords['lat'],
                'longitude': coords['lon']
            })
            
        except Exception as e:
            logger.error(f"Error parsing Google Maps link: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    # Create necessary directories
    os.makedirs(os.environ.get('TEMP_DIR', 'temp'), exist_ok=True)
    os.makedirs(os.environ.get('EXPORTS_DIR', 'exports'), exist_ok=True)
    os.makedirs(os.environ.get('MODELS_DIR', 'models'), exist_ok=True)
    
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port) 