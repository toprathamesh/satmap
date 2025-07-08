#!/usr/bin/env python3
"""
Validation utilities for API inputs
"""

import re
from datetime import datetime
from typing import Optional

def validate_coordinates(lat: Optional[float], lon: Optional[float]) -> bool:
    """Validate latitude and longitude coordinates"""
    try:
        if lat is None or lon is None:
            return False
        
        # Convert to float if string
        lat = float(lat)
        lon = float(lon)
        
        # Check valid ranges
        if not (-90 <= lat <= 90):
            return False
        
        if not (-180 <= lon <= 180):
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False

def validate_date_range(start_date: Optional[str], end_date: Optional[str]) -> bool:
    """Validate date range format and logic"""
    try:
        if not start_date or not end_date:
            return False
        
        # Validate date format (YYYY-MM-DD)
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(date_pattern, start_date) or not re.match(date_pattern, end_date):
            return False
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Check logical order - allow identical dates for testing
        if start_dt > end_dt:
            return False
        
        # Allow reasonable date range (for demo and testing)
        # Sentinel-2 started in 2015, allow up to 2030 for testing
        min_date = datetime(2015, 1, 1)
        max_date = datetime(2030, 12, 31)
        
        if start_dt < min_date or end_dt > max_date:
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False

def validate_threshold(threshold: Optional[float]) -> bool:
    """Validate change detection threshold"""
    try:
        if threshold is None:
            return True  # Optional parameter
        
        threshold = float(threshold)
        return 0.0 <= threshold <= 1.0
        
    except (ValueError, TypeError):
        return False 