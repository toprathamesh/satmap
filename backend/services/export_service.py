#!/usr/bin/env python3
"""
Export Service - GeoJSON and PDF export for change detection results
"""

import os
import json
import logging
import numpy as np
from typing import Dict
from PIL import Image
import cv2
from geojson import FeatureCollection, Feature
from shapely.geometry import Polygon
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

logger = logging.getLogger(__name__)

class ExportService:
    """Service for exporting change detection results as GeoJSON and PDF"""
    
    def __init__(self):
        """Initialize export service"""
        # Always use exports directory relative to current working directory
        # Flask app will handle the correct path
        self.exports_dir = 'exports'
        os.makedirs(self.exports_dir, exist_ok=True)
    
    def export_change_results(self, result_id: str, export_format: str) -> str:
        """Export change detection results as GeoJSON"""
        if export_format.lower() != 'geojson':
            raise ValueError(f"Only GeoJSON export is supported, got: {export_format}")
            
        try:
            # Load result metadata
            metadata_path = f'images/results/{result_id}_metadata.json'
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Result metadata not found: {result_id}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if this is an instant result (no actual files created)
            if metadata.get('instant_result', False):
                # For instant results, create empty change map
                change_map = np.zeros((512, 512), dtype=np.uint8)
                logger.info("Using empty change map for instant result")
            else:
                # Load change map (check both 'change_map' and 'change_mask' fields)
                change_map_path = metadata['files'].get('change_map') or metadata['files'].get('change_mask')
                if not change_map_path or not os.path.exists(change_map_path):
                    raise FileNotFoundError(f"Change map not found. Available files: {list(metadata['files'].keys())}")
                    
                change_map = np.array(Image.open(change_map_path))
            
            # Get coordinates with proper type checking and conversion
            lat, lon = self._extract_coordinates(metadata)
            
            return self.export_geojson(change_map, lat, lon, result_id, metadata)
                
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise
    
    def _extract_coordinates(self, metadata):
        """Safely extract and convert coordinates to float"""
        try:
            # Try different coordinate storage methods
            if 'coordinates' in metadata:
                lat_raw = metadata['coordinates']['lat']
                lon_raw = metadata['coordinates']['lon']
            elif 'metadata' in metadata and 'latitude' in metadata['metadata']:
                lat_raw = metadata['metadata']['latitude']
                lon_raw = metadata['metadata']['longitude']
            else:
                # Fallback to root level
                lat_raw = metadata.get('latitude', 0)
                lon_raw = metadata.get('longitude', 0)
            
            # Convert to float, handling various data types
            if isinstance(lat_raw, (list, tuple)):
                lat = float(lat_raw[0])  # Take first element if it's a list
            else:
                lat = float(lat_raw)
                
            if isinstance(lon_raw, (list, tuple)):
                lon = float(lon_raw[0])  # Take first element if it's a list
            else:
                lon = float(lon_raw)
            
            logger.info(f"Extracted coordinates: lat={lat}, lon={lon}")
            return lat, lon
            
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error extracting coordinates from metadata: {e}")
            # Return default coordinates (Mumbai center)
            return 19.0760, 72.8777
    
    def export_geojson(self, change_map: np.ndarray, lat: float, lon: float, 
                      result_id: str, metadata: Dict) -> str:
        """Export change regions as GeoJSON"""
        try:
            output_path = os.path.join(self.exports_dir, f'{result_id}_changes.geojson')
            
            # Ensure change_map is binary
            if change_map.dtype != np.uint8:
                change_map = (change_map * 255).astype(np.uint8)
            
            # If change_map has multiple channels, use first channel
            if len(change_map.shape) == 3:
                change_map = change_map[:, :, 0]
            
            # Apply threshold to ensure binary image
            _, change_map = cv2.threshold(change_map, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours of change regions
            contours, _ = cv2.findContours(change_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Define geographic bounds (1km x 1km AOI)
            size_degrees = 1.0 / 111.32  # 1 km in degrees (approximate)
            half_size = size_degrees / 2
            
            bounds = {
                'left': lon - half_size,
                'bottom': lat - half_size,
                'right': lon + half_size,
                'top': lat + half_size
            }
            
            height, width = change_map.shape
            
            # Convert pixel coordinates to geographic coordinates
            def pixel_to_geo(x, y):
                geo_x = bounds['left'] + (x / width) * (bounds['right'] - bounds['left'])
                geo_y = bounds['top'] - (y / height) * (bounds['top'] - bounds['bottom'])
                return geo_x, geo_y
            
            features = []
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 10:  # Filter small regions (>10 pixels)
                    # Convert contour to polygon coordinates
                    coords = []
                    for point in contour:
                        x, y = point[0]
                        geo_x, geo_y = pixel_to_geo(x, y)
                        coords.append([geo_x, geo_y])
                    
                    # Close polygon if it has enough points
                    if len(coords) >= 3:
                        coords.append(coords[0])  # Close the polygon
                        
                        # Calculate area
                        area_pixels = cv2.contourArea(contour)
                        area_km2 = (area_pixels / (width * height)) * 1.0  # 1 km² AOI
                        
                        feature = Feature(
                            geometry=Polygon([coords]),
                            properties={
                                'id': i + 1,
                                'area_pixels': int(area_pixels),
                                'area_km2': round(area_km2, 6),
                                'change_type': 'detected_change',
                                'detection_date': metadata.get('timestamp', ''),
                                'before_date': metadata.get('before_date', ''),
                                'after_date': metadata.get('after_date', ''),
                                'confidence': 'high'
                            }
                        )
                        features.append(feature)
            
            # If no contours found, create a point feature at the center
            if not features:
                logger.warning(f"No change regions found for {result_id}, creating center point")
                feature = Feature(
                    geometry={
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    properties={
                        'id': 1,
                        'area_pixels': 0,
                        'area_km2': 0.0,
                        'change_type': 'no_change_detected',
                        'detection_date': metadata.get('timestamp', ''),
                        'before_date': metadata.get('before_date', ''),
                        'after_date': metadata.get('after_date', ''),
                        'confidence': 'high'
                    }
                )
                features.append(feature)
            
            # Create FeatureCollection
            feature_collection = FeatureCollection(
                features,
                properties={
                    'title': 'Change Detection Results',
                    'description': f'Change detection between {metadata.get("before_date", "N/A")} and {metadata.get("after_date", "N/A")}',
                    'center_coordinates': {'lat': lat, 'lon': lon},
                    'total_statistics': metadata.get('statistics', {}),
                    'creation_date': metadata.get('timestamp', ''),
                    'aoi_bounds': bounds,
                    'pixel_resolution': f'{width}x{height}',
                    'coordinate_system': 'WGS84'
                }
            )
            
            # Write GeoJSON
            with open(output_path, 'w') as f:
                json.dump(feature_collection, f, indent=2)
            
            logger.info(f"GeoJSON exported successfully: {output_path}")
            logger.info(f"Exported {len(features)} change features")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting GeoJSON: {e}")
            raise 
    
    def export_pdf_report(self, result_id: str) -> str:
        """Export change detection results as a comprehensive PDF report"""
        try:
            # Load result metadata - check multiple possible locations
            possible_paths = [
                f'images/results/{result_id}_metadata.json',
                f'backend/images/results/{result_id}_metadata.json',
                f'../images/results/{result_id}_metadata.json'
            ]
            
            metadata_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    metadata_path = path
                    break
            
            if not metadata_path:
                raise FileNotFoundError(f"Result metadata not found: {result_id}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create PDF document - ensure exports directory exists
            os.makedirs(self.exports_dir, exist_ok=True)
            output_path = os.path.join(self.exports_dir, f'{result_id}_report.pdf')
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.HexColor('#2E86AB'),
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            
            # Title
            story.append(Paragraph("Satellite Change Detection Report", title_style))
            story.append(Spacer(1, 20))
            
            # Summary Information
            lat, lon = self._extract_coordinates(metadata)
            stats = metadata.get('statistics', {})
            
            summary_data = [
                ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Analysis ID', result_id],
                ['Location (Lat, Lon)', f"{lat:.6f}, {lon:.6f}"],
                ['Before Date', metadata.get('before_date', 'N/A')],
                ['After Date', metadata.get('after_date', 'N/A')],
                ['Model Used', metadata.get('model', 'Siamese U-Net')],
                ['Change Threshold', f"{metadata.get('threshold', 0.3):.1f}"],
                ['Change Percentage', f"{stats.get('change_percentage', 0):.1f}%"],
                ['Changed Area', f"{stats.get('changed_area_km2', 0):.3f} km²"],
                ['Total Area Analyzed', f"{stats.get('total_area_km2', 0):.3f} km²"],
                ['Number of Change Regions', str(stats.get('num_change_regions', 0))],
            ]
            
            summary_table = Table(summary_data, colWidths=[2.5*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4FD')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(Paragraph("Analysis Summary", styles['Heading2']))
            story.append(Spacer(1, 10))
            story.append(summary_table)
            story.append(Spacer(1, 30))
            
            # Add images if available
            files = metadata.get('files', {})
            
            # Comparison image
            comparison_path = files.get('comparison')
            if comparison_path and os.path.exists(comparison_path):
                story.append(Paragraph("Change Detection Results", styles['Heading2']))
                story.append(Spacer(1, 10))
                
                # Resize image to fit in PDF
                img = RLImage(comparison_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 10))
                story.append(Paragraph("Figure 1: Before/After comparison with change detection overlay", 
                                     styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Heatmap image
            heatmap_path = files.get('heatmap')
            if heatmap_path and os.path.exists(heatmap_path):
                story.append(Paragraph("Change Probability Heatmap", styles['Heading2']))
                story.append(Spacer(1, 10))
                
                img = RLImage(heatmap_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 10))
                story.append(Paragraph("Figure 2: Change probability heatmap showing likelihood of change in each area", 
                                     styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Analysis Details
            story.append(Paragraph("Technical Details", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            technical_text = f"""
            This analysis was performed using a Lightweight Siamese U-Net neural network model 
            specifically designed for satellite change detection. The model analyzes differences 
            between two time periods using Sentinel-2 satellite imagery at 10-meter resolution.
            
            <b>Analysis Parameters:</b><br/>
            • Satellite Data: Sentinel-2 Surface Reflectance (10m bands)<br/>
            • Processing: Raw satellite data with minimal atmospheric correction<br/>
            • Change Threshold: {metadata.get('threshold', 0.3):.1f} (0.0 = no change, 1.0 = maximum change)<br/>
            • Area of Interest: 1km × 1km centered on the specified coordinates<br/>
            
            <b>Results Interpretation:</b><br/>
            • Areas highlighted in red show detected changes between the two dates<br/>
            • Change percentage indicates the proportion of the analyzed area that changed<br/>
            • The heatmap shows probability of change, with warmer colors indicating higher confidence<br/>
            """
            
            story.append(Paragraph(technical_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Footer
            footer_text = f"""
            <b>Disclaimer:</b> This analysis is based on satellite imagery and automated detection algorithms. 
            Results should be verified with ground truth data for critical applications. The analysis covers 
            environmental and land use changes visible at 10-meter resolution.
            
            Generated by Satellite Change Detection Platform | Report ID: {result_id}
            """
            
            story.append(Paragraph(footer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            # Return absolute path to ensure Flask can find the file
            abs_output_path = os.path.abspath(output_path)
            logger.info(f"PDF report exported successfully: {abs_output_path}")
            return abs_output_path
            
        except Exception as e:
            logger.error(f"Error exporting PDF report: {e}")
            raise 