#!/usr/bin/env python3
"""
Change Detection Platform - Main Flask Application
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from services.gee_service import GEEService
from services.change_detection_service import ChangeDetectionService
from services.export_service import ExportService
from utils.validators import validate_coordinates, validate_date_range, validate_threshold
from utils.gmaps_parser import parse_gmaps_url

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS with explicit settings for frontend port
CORS(app, 
     origins=[
         "http://localhost:3000", 
         "http://localhost:8080", 
         "http://localhost:8081",  # Added support for port 8081
         "http://localhost:8082"   # Added support for port 8082
     ],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

# Initialize services
logger.info("Initializing services...")
gee_service = GEEService()
change_detection_service = ChangeDetectionService()
export_service = ExportService()

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serve images from both backend/images and images directories"""
    try:
        # Try multiple possible locations for images
        possible_paths = [
            ('images', filename),                    # Current working directory images/
            ('backend/images', filename),            # Explicit backend/images/
            ('../images', filename),                 # Parent directory images/
        ]
        
        for directory, file in possible_paths:
            full_path = os.path.join(directory, file)
            if os.path.exists(full_path):
                logger.info(f"Serving image from: {full_path}")
                return send_from_directory(directory, file)
        
        # If no file found, log attempted paths
        attempted_paths = [os.path.join(d, f) for d, f in possible_paths]
        logger.error(f"Image not found: {filename}. Checked paths: {attempted_paths}")
        return jsonify({'error': f'Image not found: {filename}'}), 404
        
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({'error': f'Error serving image: {str(e)}'}), 500

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
        
        # Quick check for identical dates - skip expensive image processing
        if before_date == after_date:
            logger.info(f"Identical dates detected ({before_date}), using fast no-change logic")
            
            # Create a minimal result without expensive processing
            metadata = {
                'latitude': lat,
                'longitude': lon,
                'before_date': before_date,
                'after_date': after_date,
                'pixel_size_m': 10,
                'gee_demo_mode': True  # Skip actual GEE for identical dates
            }
            
            # Create immediate no-change result
            change_result = change_detection_service.create_instant_no_change_result(metadata)
            
            logger.info("Instant no-change result completed in <1 second")
            
            return jsonify({
                'success': True,
                'data': change_result
            })
        
        # Get images for both time periods from GEE (only for different dates)
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

@app.route('/api/export', methods=['POST', 'OPTIONS'])
def export_results():
    """Export change detection results"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        result_id = data.get('result_id')
        export_format = data.get('format', 'geojson')
        
        if not result_id:
            return jsonify({'error': 'Missing result_id'}), 400
        
        # Export the results
        export_path = export_service.export_change_results(result_id, export_format)
        
        # Send the file
        return send_file(export_path, as_attachment=True, download_name=f'{result_id}_changes.geojson')
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/pdf', methods=['POST', 'OPTIONS'])
def export_pdf():
    """Export change detection results as PDF report"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        result_id = data.get('result_id')
        
        if not result_id:
            return jsonify({'error': 'Missing result_id'}), 400
        
        # Generate PDF report
        pdf_path = export_service.export_pdf_report(result_id)
        
        # Check multiple possible locations for the generated PDF
        possible_paths = [
            pdf_path,  # Original path
            os.path.join('exports', f'{result_id}_report.pdf'),  # Root exports
            os.path.join('../exports', f'{result_id}_report.pdf'),  # Parent exports
            os.path.abspath(os.path.join('exports', f'{result_id}_report.pdf')),  # Absolute root exports
            os.path.abspath(pdf_path)  # Absolute version
        ]
        
        final_path = None
        for path in possible_paths:
            if os.path.exists(path):
                final_path = os.path.abspath(path)
                break
        
        if not final_path:
            return jsonify({'error': f'Generated PDF not found. Checked paths: {possible_paths}'}), 500
        
        # Send the PDF file
        return send_file(final_path, as_attachment=True, download_name=f'{result_id}_report.pdf')
        
    except Exception as e:
        logger.error(f"Error exporting PDF: {e}")
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

if __name__ == '__main__':
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Create necessary directories
    os.makedirs(os.environ.get('TEMP_DIR', 'temp'), exist_ok=True)
    os.makedirs(os.environ.get('EXPORTS_DIR', 'exports'), exist_ok=True)
    os.makedirs(os.environ.get('MODELS_DIR', 'models'), exist_ok=True)
    
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port) 