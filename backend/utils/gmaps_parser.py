#!/usr/bin/env python3
"""
Google Maps URL parser utility
Extracts latitude and longitude coordinates from various Google Maps URL formats
"""

import re
from typing import Optional, Dict
from urllib.parse import urlparse, parse_qs

def parse_gmaps_url(url: str) -> Optional[Dict[str, float]]:
    """
    Parse Google Maps URL to extract coordinates
    
    Supports various Google Maps URL formats:
    - https://www.google.com/maps/@19.0419252,73.0270304,17z
    - https://maps.google.com/maps?q=19.0419252,73.0270304
    - https://www.google.com/maps/place/Mumbai/@19.0419252,73.0270304,17z
    - https://goo.gl/maps/... (if expanded)
    """
    try:
        if not url or not isinstance(url, str):
            return None
        
        # Normalize URL
        url = url.strip()
        
        # Pattern 1: @lat,lon format
        # https://www.google.com/maps/@19.0419252,73.0270304,17z
        pattern1 = r'@(-?\d+\.?\d*),(-?\d+\.?\d*)(?:,\d+\.?\d*)?'
        match1 = re.search(pattern1, url)
        if match1:
            lat, lon = float(match1.group(1)), float(match1.group(2))
            if _validate_coords(lat, lon):
                return {'lat': lat, 'lon': lon}
        
        # Pattern 2: q=lat,lon format
        # https://maps.google.com/maps?q=19.0419252,73.0270304
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        if 'q' in query_params:
            q_value = query_params['q'][0]
            pattern2 = r'^(-?\d+\.?\d*),(-?\d+\.?\d*)$'
            match2 = re.search(pattern2, q_value)
            if match2:
                lat, lon = float(match2.group(1)), float(match2.group(2))
                if _validate_coords(lat, lon):
                    return {'lat': lat, 'lon': lon}
        
        # Pattern 3: ll parameter
        if 'll' in query_params:
            ll_value = query_params['ll'][0]
            pattern3 = r'^(-?\d+\.?\d*),(-?\d+\.?\d*)$'
            match3 = re.search(pattern3, ll_value)
            if match3:
                lat, lon = float(match3.group(1)), float(match3.group(2))
                if _validate_coords(lat, lon):
                    return {'lat': lat, 'lon': lon}
        
        # Pattern 4: center parameter
        if 'center' in query_params:
            center_value = query_params['center'][0]
            pattern4 = r'^(-?\d+\.?\d*),(-?\d+\.?\d*)$'
            match4 = re.search(pattern4, center_value)
            if match4:
                lat, lon = float(match4.group(1)), float(match4.group(2))
                if _validate_coords(lat, lon):
                    return {'lat': lat, 'lon': lon}
        
        # Pattern 5: Place URL with coordinates
        # https://www.google.com/maps/place/Mumbai/@19.0419252,73.0270304,17z
        if '/place/' in url:
            pattern5 = r'/place/[^/]+/@(-?\d+\.?\d*),(-?\d+\.?\d*)(?:,\d+\.?\d*)?'
            match5 = re.search(pattern5, url)
            if match5:
                lat, lon = float(match5.group(1)), float(match5.group(2))
                if _validate_coords(lat, lon):
                    return {'lat': lat, 'lon': lon}
        
        # Pattern 6: Search for any lat,lon pattern in the URL
        pattern6 = r'(-?\d{1,3}\.\d+),(-?\d{1,3}\.\d+)'
        matches = re.findall(pattern6, url)
        for match in matches:
            lat, lon = float(match[0]), float(match[1])
            if _validate_coords(lat, lon):
                return {'lat': lat, 'lon': lon}
        
        # Pattern 7: decimal format without comma
        pattern7 = r'(-?\d+\.?\d+)%2C(-?\d+\.?\d+)'  # URL encoded comma
        match7 = re.search(pattern7, url)
        if match7:
            lat, lon = float(match7.group(1)), float(match7.group(2))
            if _validate_coords(lat, lon):
                return {'lat': lat, 'lon': lon}
        
        return None
        
    except Exception:
        return None

def _validate_coords(lat: float, lon: float) -> bool:
    """Validate that coordinates are within valid ranges"""
    return -90 <= lat <= 90 and -180 <= lon <= 180

def extract_coords_from_text(text: str) -> Optional[Dict[str, float]]:
    """
    Extract coordinates from plain text (e.g., "19.0419, 73.0270")
    """
    try:
        # Pattern for decimal coordinates
        pattern = r'(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)'
        match = re.search(pattern, text)
        
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            if _validate_coords(lat, lon):
                return {'lat': lat, 'lon': lon}
        
        return None
        
    except Exception:
        return None

def is_google_maps_url(url: str) -> bool:
    """Check if URL is a Google Maps URL"""
    if not url:
        return False
    
    google_domains = [
        'maps.google.com',
        'www.google.com/maps',
        'google.com/maps',
        'maps.app.goo.gl',
        'goo.gl/maps'
    ]
    
    return any(domain in url.lower() for domain in google_domains)

# Example usage and test cases
if __name__ == "__main__":
    test_urls = [
        "https://www.google.com/maps/@19.0419252,73.0270304,17z",
        "https://maps.google.com/maps?q=19.0419252,73.0270304",
        "https://www.google.com/maps/place/Mumbai/@19.0419252,73.0270304,17z",
        "https://www.google.com/maps?ll=19.0419252,73.0270304&z=15",
        "19.0419252, 73.0270304"
    ]
    
    for url in test_urls:
        result = parse_gmaps_url(url) if is_google_maps_url(url) else extract_coords_from_text(url)
        print(f"URL: {url}")
        print(f"Result: {result}")
        print("---") 