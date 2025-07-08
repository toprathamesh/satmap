# 📚 Usage Examples

This document provides step-by-step examples for using the Change Detection Platform.

## 🚀 Quick Start Example

### Example 1: Detect Urban Development in Mumbai

```bash
# 1. Start the backend
cd backend && python app.py

# 2. Start the frontend  
cd frontend && npm run serve

# 3. Open browser to http://localhost:8080
```

**In the UI:**
1. Paste Google Maps URL: `https://www.google.com/maps/@19.0419252,73.0270304,17z`
2. Click "Parse URL" - coordinates auto-fill
3. Set Before Date: `2022-01-01`
4. Set After Date: `2023-01-01`
5. Click "Detect Changes"
6. Export results as GeoTIFF

## 🏭 Industrial Development Detection

### Example 2: Track Factory Construction

**Location:** Industrial area coordinates
- Latitude: `28.7041`
- Longitude: `77.1025`

**Time Period:** 6 months
- Before: `2023-01-01`
- After: `2023-07-01`

**Expected Results:**
- New building footprints
- Road construction
- Vegetation loss

## 🌾 Agricultural Change Monitoring

### Example 3: Crop Pattern Analysis

**Location:** Agricultural region
- Latitude: `30.3753`
- Longitude: `76.7821`

**Seasonal Comparison:**
- Before: `2023-04-01` (Pre-harvest)
- After: `2023-06-01` (Post-harvest)

**Analysis Features:**
- Field boundary changes
- Crop rotation patterns
- Irrigation infrastructure

## 🌊 Water Body Monitoring

### Example 4: Reservoir Level Changes

**Location:** Water reservoir
- Latitude: `23.2599`
- Longitude: `77.4126`

**Drought Analysis:**
- Before: `2023-01-01` (Monsoon season)
- After: `2023-05-01` (Dry season)

**Export Formats:**
1. GeoTIFF for QGIS analysis
2. Shapefile for ArcGIS
3. GeoJSON for web mapping

## 🏗️ Construction Monitoring

### Example 5: Highway Development

**Location:** Highway corridor
- Google Maps: `https://maps.google.com/maps?q=28.4595,77.0266`

**Timeline:** 
- Before: `2022-06-01`
- After: `2023-06-01`

**Workflow:**
```bash
# 1. Parse Google Maps URL
curl -X POST http://localhost:5000/api/parse-gmaps-link \
  -H "Content-Type: application/json" \
  -d '{"url": "https://maps.google.com/maps?q=28.4595,77.0266"}'

# 2. Get survey data
curl -X POST http://localhost:5000/api/survey-data \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 28.4595,
    "longitude": 77.0266,
    "start_date": "2022-01-01",
    "end_date": "2023-12-31"
  }'

# 3. Detect changes
curl -X POST http://localhost:5000/api/change-detection \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 28.4595,
    "longitude": 77.0266,
    "before_date": "2022-06-01",
    "after_date": "2023-06-01"
  }'
```

## 🌳 Deforestation Analysis

### Example 6: Forest Cover Loss

**Location:** Forest area
- Latitude: `11.2588`
- Longitude: `75.7804`

**Analysis Period:** 2 years
- Before: `2021-01-01`
- After: `2023-01-01`

**Statistical Output:**
- Total area changed: 15.6%
- Changed area: 0.156 km²
- Number of change regions: 8
- Largest region: 0.045 km²

## 📊 API Integration Examples

### Python Client Example

```python
import requests
import json

class ChangeDetectionClient:
    def __init__(self, base_url="http://localhost:5000/api"):
        self.base_url = base_url
    
    def detect_changes(self, lat, lon, before_date, after_date):
        response = requests.post(f"{self.base_url}/change-detection", 
            json={
                "latitude": lat,
                "longitude": lon,
                "before_date": before_date,
                "after_date": after_date
            })
        return response.json()
    
    def export_results(self, result_id, format="geotiff"):
        response = requests.post(f"{self.base_url}/export",
            json={"result_id": result_id, "format": format})
        return response.content

# Usage
client = ChangeDetectionClient()
result = client.detect_changes(19.0419, 73.0270, "2023-01-01", "2023-06-01")
print(f"Change detected: {result['data']['statistics']['change_percentage']}%")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

class ChangeDetectionAPI {
    constructor(baseURL = 'http://localhost:5000/api') {
        this.api = axios.create({ baseURL });
    }

    async parseGoogleMapsURL(url) {
        const response = await this.api.post('/parse-gmaps-link', { url });
        return response.data;
    }

    async detectChanges(lat, lon, beforeDate, afterDate) {
        const response = await this.api.post('/change-detection', {
            latitude: lat,
            longitude: lon,
            before_date: beforeDate,
            after_date: afterDate
        });
        return response.data;
    }

    async getSurveyData(lat, lon, startDate, endDate) {
        const response = await this.api.post('/survey-data', {
            latitude: lat,
            longitude: lon,
            start_date: startDate,
            end_date: endDate
        });
        return response.data;
    }
}

// Usage
const api = new ChangeDetectionAPI();

async function analyzeLocation() {
    try {
        // Parse Google Maps URL
        const coords = await api.parseGoogleMapsURL(
            'https://www.google.com/maps/@19.0419252,73.0270304,17z'
        );
        
        // Get survey data
        const surveyData = await api.getSurveyData(
            coords.latitude, coords.longitude,
            '2023-01-01', '2023-12-31'
        );
        
        // Detect changes
        const changes = await api.detectChanges(
            coords.latitude, coords.longitude,
            '2023-01-01', '2023-06-01'
        );
        
        console.log('Change Analysis Results:', changes);
    } catch (error) {
        console.error('Analysis failed:', error.message);
    }
}

analyzeLocation();
```

## 🗺️ GIS Integration Examples

### QGIS Integration

1. **Export GeoTIFF**
   - Use export API to download GeoTIFF
   - Import in QGIS: `Layer > Add Layer > Add Raster Layer`
   - Style with binary classification

2. **Vector Analysis**
   - Export as GeoJSON or Shapefile
   - Import: `Layer > Add Layer > Add Vector Layer`
   - Perform spatial analysis

### ArcGIS Integration

```python
# ArcPy script for batch processing
import arcpy
import requests

def process_change_detection_results(result_ids):
    for result_id in result_ids:
        # Download shapefile
        response = requests.post('http://localhost:5000/api/export', 
            json={"result_id": result_id, "format": "shapefile"})
        
        # Save and import to ArcGIS
        shapefile_path = f"changes_{result_id}.zip"
        with open(shapefile_path, 'wb') as f:
            f.write(response.content)
        
        # Extract and add to map
        arcpy.management.ExtractPackage(shapefile_path, f"extracted_{result_id}")
        arcpy.management.MakeFeatureLayer(f"extracted_{result_id}/changes.shp", 
                                        f"changes_layer_{result_id}")
```

## 📈 Batch Processing Example

### Monitor Multiple Locations

```python
import pandas as pd
from datetime import datetime, timedelta

# Define monitoring locations
locations = pd.DataFrame({
    'name': ['Mumbai Port', 'Delhi Airport', 'Bangalore Tech Park'],
    'lat': [19.0419, 28.5562, 12.9716],
    'lon': [73.0270, 77.1000, 77.5946]
})

# Batch analysis function
def batch_change_detection(locations_df, before_date, after_date):
    results = []
    
    for idx, location in locations_df.iterrows():
        try:
            result = client.detect_changes(
                location['lat'], location['lon'], 
                before_date, after_date
            )
            
            results.append({
                'location': location['name'],
                'change_percentage': result['data']['statistics']['change_percentage'],
                'changed_area_km2': result['data']['statistics']['changed_area_km2'],
                'result_id': result['data']['result_id']
            })
            
        except Exception as e:
            print(f"Error processing {location['name']}: {e}")
    
    return pd.DataFrame(results)

# Execute batch analysis
results_df = batch_change_detection(
    locations, 
    '2023-01-01', 
    '2023-06-01'
)

print(results_df)
```

## 🔍 Quality Assurance Examples

### Validation Workflow

```python
def validate_change_detection(lat, lon, before_date, after_date):
    """Validate change detection results with multiple checks"""
    
    # 1. Check coordinate validity
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Invalid coordinates")
    
    # 2. Check date validity
    before_dt = datetime.strptime(before_date, '%Y-%m-%d')
    after_dt = datetime.strptime(after_date, '%Y-%m-%d')
    
    if before_dt >= after_dt:
        raise ValueError("Before date must be earlier than after date")
    
    # 3. Get survey data first
    survey_data = client.get_survey_data(lat, lon, before_date, after_date)
    
    if not survey_data['data']:
        print("Warning: No survey data available for this period")
    
    # 4. Run change detection
    result = client.detect_changes(lat, lon, before_date, after_date)
    
    # 5. Validate results
    stats = result['data']['statistics']
    
    if stats['change_percentage'] > 50:
        print("Warning: High change percentage detected - verify results")
    
    if stats['num_change_regions'] > 20:
        print("Warning: Many change regions - possible noise")
    
    return result

# Usage
validated_result = validate_change_detection(
    19.0419, 73.0270, '2023-01-01', '2023-06-01'
)
```

## 🚨 Error Handling Examples

### Robust API Client

```python
import time
from typing import Optional

class RobustChangeDetectionClient:
    def __init__(self, base_url="http://localhost:5000/api", max_retries=3):
        self.base_url = base_url
        self.max_retries = max_retries
    
    def _make_request(self, endpoint, data, retry_count=0):
        """Make request with retry logic"""
        try:
            response = requests.post(f"{self.base_url}/{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                print(f"Request failed, retrying... ({retry_count + 1}/{self.max_retries})")
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self._make_request(endpoint, data, retry_count + 1)
            else:
                raise Exception(f"Request failed after {self.max_retries} retries: {e}")
    
    def detect_changes_with_fallback(self, lat, lon, before_date, after_date):
        """Detect changes with automatic fallback dates"""
        try:
            return self._make_request('change-detection', {
                "latitude": lat,
                "longitude": lon,
                "before_date": before_date,
                "after_date": after_date
            })
            
        except Exception as e:
            print(f"Primary analysis failed: {e}")
            
            # Try with extended date range
            before_dt = datetime.strptime(before_date, '%Y-%m-%d')
            after_dt = datetime.strptime(after_date, '%Y-%m-%d')
            
            # Extend range by 30 days on each side
            fallback_before = (before_dt - timedelta(days=30)).strftime('%Y-%m-%d')
            fallback_after = (after_dt + timedelta(days=30)).strftime('%Y-%m-%d')
            
            print(f"Trying fallback dates: {fallback_before} to {fallback_after}")
            
            return self._make_request('change-detection', {
                "latitude": lat,
                "longitude": lon,
                "before_date": fallback_before,
                "after_date": fallback_after
            })
```

## 🎯 Use Case Scenarios

### Scenario 1: Environmental Impact Assessment

```python
def environmental_impact_analysis(project_coordinates, construction_start_date):
    """Analyze environmental impact of construction project"""
    
    # Pre-construction baseline
    baseline_result = client.detect_changes(
        project_coordinates['lat'], project_coordinates['lon'],
        '2020-01-01', construction_start_date
    )
    
    # Post-construction impact
    impact_result = client.detect_changes(
        project_coordinates['lat'], project_coordinates['lon'],
        construction_start_date, '2023-12-31'
    )
    
    return {
        'baseline_change': baseline_result['data']['statistics']['change_percentage'],
        'construction_impact': impact_result['data']['statistics']['change_percentage'],
        'total_area_affected': impact_result['data']['statistics']['changed_area_km2']
    }
```

### Scenario 2: Disaster Response

```python
def disaster_damage_assessment(disaster_location, disaster_date):
    """Assess damage from natural disaster"""
    
    # Before disaster
    before_date = (datetime.strptime(disaster_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # After disaster  
    after_date = (datetime.strptime(disaster_date, '%Y-%m-%d') + timedelta(days=7)).strftime('%Y-%m-%d')
    
    damage_result = client.detect_changes(
        disaster_location['lat'], disaster_location['lon'],
        before_date, after_date
    )
    
    # Export for emergency response teams
    shapefile_data = client.export_results(
        damage_result['data']['result_id'], 'shapefile'
    )
    
    return {
        'damage_percentage': damage_result['data']['statistics']['change_percentage'],
        'affected_area_km2': damage_result['data']['statistics']['changed_area_km2'],
        'gis_data': shapefile_data
    }
```

This documentation provides comprehensive examples for all major use cases of the Change Detection Platform. Each example includes practical code and real-world scenarios. 