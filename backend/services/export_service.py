#!/usr/bin/env python3
"""
Export Service - GIS-compatible format exports
Handles export of change detection results to GeoTIFF, GeoJSON, and shapefile formats
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import Polygon, Point
import cv2
from geojson import FeatureCollection, Feature

logger = logging.getLogger(__name__)

class ExportService:
    """Service for exporting change detection results in GIS-compatible formats"""
    
    def __init__(self):
        """Initialize export service"""
        self.exports_dir = 'exports'
        os.makedirs(self.exports_dir, exist_ok=True)
    
    def export_change_results(self, result_id: str, export_format: str) -> str:
        """Export change detection results in specified format"""
        try:
            # Load result metadata
            metadata_path = f'images/results/{result_id}_metadata.json'
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Result metadata not found: {result_id}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load change map
            change_map_path = metadata['files']['change_map']
            change_map = np.array(Image.open(change_map_path))
            
            # Get coordinates
            lat = metadata['coordinates']['lat']
            lon = metadata['coordinates']['lon']
            
            # Export based on format
            if export_format.lower() == 'geotiff':
                return self.export_geotiff(change_map, lat, lon, result_id, metadata)
            elif export_format.lower() == 'geojson':
                return self.export_geojson(change_map, lat, lon, result_id, metadata)
            elif export_format.lower() == 'shapefile':
                return self.export_shapefile(change_map, lat, lon, result_id, metadata)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise
    
    def export_geotiff(self, change_map: np.ndarray, lat: float, lon: float, 
                      result_id: str, metadata: Dict) -> str:
        """Export change map as GeoTIFF"""
        try:
            output_path = os.path.join(self.exports_dir, f'{result_id}_change_map.tif')
            
            # Define geographic bounds (1km x 1km AOI)
            size_degrees = 1.0 / 111.32  # 1 km in degrees
            half_size = size_degrees / 2
            
            bounds = {
                'left': lon - half_size,
                'bottom': lat - half_size,
                'right': lon + half_size,
                'top': lat + half_size
            }
            
            # Create affine transform
            height, width = change_map.shape
            transform = from_bounds(bounds['left'], bounds['bottom'], 
                                  bounds['right'], bounds['top'], 
                                  width, height)
            
            # Write GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=change_map.dtype,
                crs=CRS.from_epsg(4326),  # WGS84
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(change_map, 1)
                
                # Add metadata
                dst.update_tags(
                    title='Change Detection Results',
                    description=f'Change detection between {metadata["before_date"]} and {metadata["after_date"]}',
                    source='Siamese U-Net Change Detection',
                    creation_date=metadata['timestamp'],
                    coordinates=f'{lat}, {lon}',
                    **metadata['statistics']
                )
            
            logger.info(f"GeoTIFF exported: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting GeoTIFF: {e}")
            raise
    
    def export_geojson(self, change_map: np.ndarray, lat: float, lon: float, 
                      result_id: str, metadata: Dict) -> str:
        """Export change regions as GeoJSON"""
        try:
            output_path = os.path.join(self.exports_dir, f'{result_id}_changes.geojson')
            
            # Find contours of change regions
            contours, _ = cv2.findContours(change_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Define geographic bounds
            size_degrees = 1.0 / 111.32  # 1 km in degrees
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
                if cv2.contourArea(contour) > 10:  # Filter small regions
                    # Convert contour to polygon
                    coords = []
                    for point in contour:
                        x, y = point[0]
                        geo_x, geo_y = pixel_to_geo(x, y)
                        coords.append([geo_x, geo_y])
                    
                    # Close polygon
                    if len(coords) > 2:
                        coords.append(coords[0])
                        
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
                                'detection_date': metadata['timestamp'],
                                'before_date': metadata['before_date'],
                                'after_date': metadata['after_date']
                            }
                        )
                        features.append(feature)
            
            # Create FeatureCollection
            feature_collection = FeatureCollection(
                features,
                properties={
                    'title': 'Change Detection Results',
                    'description': f'Change detection between {metadata["before_date"]} and {metadata["after_date"]}',
                    'center_coordinates': {'lat': lat, 'lon': lon},
                    'total_statistics': metadata['statistics'],
                    'creation_date': metadata['timestamp']
                }
            )
            
            # Write GeoJSON
            with open(output_path, 'w') as f:
                json.dump(feature_collection, f, indent=2)
            
            logger.info(f"GeoJSON exported: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting GeoJSON: {e}")
            raise
    
    def export_shapefile(self, change_map: np.ndarray, lat: float, lon: float, 
                        result_id: str, metadata: Dict) -> str:
        """Export change regions as shapefile"""
        try:
            output_dir = os.path.join(self.exports_dir, f'{result_id}_shapefile')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'changes.shp')
            
            # Find contours of change regions
            contours, _ = cv2.findContours(change_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Define geographic bounds
            size_degrees = 1.0 / 111.32  # 1 km in degrees
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
            
            # Create GeoDataFrame
            geometries = []
            properties = []
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 10:  # Filter small regions
                    # Convert contour to polygon
                    coords = []
                    for point in contour:
                        x, y = point[0]
                        geo_x, geo_y = pixel_to_geo(x, y)
                        coords.append((geo_x, geo_y))
                    
                    if len(coords) > 2:
                        polygon = Polygon(coords)
                        geometries.append(polygon)
                        
                        # Calculate area
                        area_pixels = cv2.contourArea(contour)
                        area_km2 = (area_pixels / (width * height)) * 1.0  # 1 km² AOI
                        
                        properties.append({
                            'id': i + 1,
                            'area_px': int(area_pixels),
                            'area_km2': round(area_km2, 6),
                            'type': 'change',
                            'before_dt': metadata['before_date'],
                            'after_dt': metadata['after_date'],
                            'detect_dt': metadata['timestamp'][:10]  # Date only
                        })
            
            if geometries:
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')
                
                # Write shapefile
                gdf.to_file(output_path)
                
                # Create metadata file
                metadata_file = os.path.join(output_dir, 'metadata.txt')
                with open(metadata_file, 'w') as f:
                    f.write(f"Change Detection Results\n")
                    f.write(f"========================\n\n")
                    f.write(f"Location: {lat}, {lon}\n")
                    f.write(f"Before Date: {metadata['before_date']}\n")
                    f.write(f"After Date: {metadata['after_date']}\n")
                    f.write(f"Detection Date: {metadata['timestamp']}\n\n")
                    f.write(f"Statistics:\n")
                    for key, value in metadata['statistics'].items():
                        f.write(f"  {key}: {value}\n")
                
                logger.info(f"Shapefile exported: {output_path}")
                return output_dir
            else:
                # No changes detected - create empty shapefile
                empty_gdf = gpd.GeoDataFrame(columns=['id', 'area_px', 'area_km2', 'type'], 
                                           crs='EPSG:4326')
                empty_gdf.to_file(output_path)
                logger.info(f"Empty shapefile exported (no changes): {output_path}")
                return output_dir
            
        except Exception as e:
            logger.error(f"Error exporting shapefile: {e}")
            raise
    
    def create_export_package(self, result_id: str) -> str:
        """Create a complete export package with all formats"""
        try:
            import zipfile
            
            package_path = os.path.join(self.exports_dir, f'{result_id}_complete_package.zip')
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Export all formats
                geotiff_path = self.export_change_results(result_id, 'geotiff')
                geojson_path = self.export_change_results(result_id, 'geojson')
                shapefile_dir = self.export_change_results(result_id, 'shapefile')
                
                # Add files to zip
                zipf.write(geotiff_path, os.path.basename(geotiff_path))
                zipf.write(geojson_path, os.path.basename(geojson_path))
                
                # Add all shapefile components
                for root, dirs, files in os.walk(shapefile_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join('shapefile', file)
                        zipf.write(file_path, arcname)
                
                # Add metadata and visualization
                metadata_path = f'images/results/{result_id}_metadata.json'
                visualization_path = f'images/results/{result_id}_visualization.png'
                
                if os.path.exists(metadata_path):
                    zipf.write(metadata_path, 'metadata.json')
                if os.path.exists(visualization_path):
                    zipf.write(visualization_path, 'visualization.png')
            
            logger.info(f"Complete export package created: {package_path}")
            return package_path
            
        except Exception as e:
            logger.error(f"Error creating export package: {e}")
            raise 