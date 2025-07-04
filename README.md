# Google Earth Engine Satellite Analysis

This project provides tools for satellite data analysis using Google Earth Engine with project `caramel-goal-464010-u8`.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Authenticate Google Earth Engine

**First time setup:**
```bash
python gee_setup.py
```

This will:
- Open a browser window for Google OAuth authentication
- Initialize GEE with your project `caramel-goal-464010-u8`
- Test basic functionality

### 3. Run Satellite Analysis

```bash
python satellite_example.py
```

## 📋 Authentication Process

When you run `gee_setup.py`, you'll see:

1. **Browser opens** - Sign in with your Google account that has access to the GEE project
2. **Authorization** - Grant permissions to Earth Engine
3. **Verification code** - Copy the code back to your terminal
4. **Success** - GEE will be authenticated and ready to use

## 🛰️ What's Included

### `gee_setup.py`
- Handles authentication with your project
- Tests basic GEE functionality
- Troubleshooting guidance

### `satellite_example.py` 
- Complete satellite data analysis examples
- Landsat 8 image processing
- NDVI and NDWI calculations
- Time series analysis
- Download URL generation

## 📊 Example Analyses

### Regional Analysis
```python
# Analyze any region using bounding box coordinates
coordinates = [-122.5, 37.4, -121.8, 38.0]  # [west, south, east, north]
results = analyze_region(coordinates)
```

### Time Series Analysis
```python
# Get NDVI time series for a specific point
point = [-122.2, 37.7]  # [longitude, latitude]
timeseries = get_time_series(point)
```

## 🔧 Customization

### Change Region of Interest
Modify coordinates in `satellite_example.py`:
```python
# Format: [west_longitude, south_latitude, east_longitude, north_latitude]
my_coordinates = [-74.1, 40.6, -73.9, 40.8]  # New York City
```

### Change Date Range
```python
analyze_region(coordinates, start_date='2022-01-01', end_date='2022-12-31')
```

### Use Different Satellites
```python
# Sentinel-2
sentinel = ee.ImageCollection('COPERNICUS/S2_SR')

# MODIS
modis = ee.ImageCollection('MODIS/006/MOD13Q1')
```

## 🗺️ Output

The analysis provides:
- **Vegetation statistics** (NDVI): Mean, min, max values
- **Water statistics** (NDWI): Water body identification
- **Image counts**: Number of available satellite images
- **Download URLs**: Direct links to processed images
- **Time series data**: NDVI trends over time

## 🔐 Project Configuration

This setup uses Google Earth Engine project: `caramel-goal-464010-u8`

Make sure you have:
- Google account with GEE access
- Access to the specific project
- Earth Engine API enabled

## 🆘 Troubleshooting

### Authentication Issues
```bash
# Force re-authentication
python -c "import ee; ee.Authenticate(force=True)"
```

### Project Access Issues
- Verify you have access to project `caramel-goal-464010-u8`
- Check with your GEE administrator
- Ensure the project is active

### Common Errors
- **"Project not found"**: Check project ID spelling
- **"Quota exceeded"**: Wait or request quota increase
- **"Authentication failed"**: Re-run authentication

## 📚 Next Steps

- Explore different satellite collections
- Implement custom indices and algorithms
- Export large datasets using GEE Tasks
- Create interactive visualizations
- Set up automated monitoring workflows

## 🌍 Use Cases

- **Environmental monitoring**: Track deforestation, urbanization
- **Agriculture**: Crop health assessment, yield prediction
- **Water resources**: Monitor lakes, rivers, irrigation
- **Climate research**: Land surface temperature, precipitation
- **Disaster response**: Flood mapping, fire detection 