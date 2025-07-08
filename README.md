# 🛰️ Change Detection Platform

A web-based platform for robust change detection, monitoring, and alerting on user-defined Areas of Interest (AOIs) using multi-temporal satellite imagery. Built with **Siamese U-Net** for advanced change detection and **Google Earth Engine** for satellite data processing.

## 🌟 Features

- **Interactive Map Interface**: OpenLayers-based map for AOI selection
- **Google Maps Integration**: Parse coordinates from Google Maps URLs
- **Siamese U-Net Model**: Pretrained deep learning model for accurate change detection
- **Google Earth Engine**: Robust satellite imagery acquisition and preprocessing
- **Multiple Export Formats**: GeoTIFF, GeoJSON, and Shapefile exports
- **Survey Data Integration**: Access to available government survey data
- **Minimal UI Design**: Clean black-and-white interface for professional use

## 🏗️ Architecture

```
├── backend/                    # Flask API server
│   ├── app.py                 # Main Flask application
│   ├── services/              # Core business logic
│   │   ├── gee_service.py     # Google Earth Engine integration
│   │   ├── change_detection_service.py # Siamese U-Net model
│   │   └── export_service.py  # GIS format exports
│   └── utils/                 # Utility functions
│       ├── validators.py      # Input validation
│       └── gmaps_parser.py    # Google Maps URL parsing
├── frontend/                  # Vue.js frontend
│   ├── src/
│   │   ├── views/Home.vue     # Main interface
│   │   ├── styles/            # SCSS styling
│   │   └── router/            # Vue router
│   └── public/
└── docs/                      # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Google Earth Engine account
- Git

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd satmap-2
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Google Earth Engine Authentication**
   ```bash
   earthengine authenticate
   ```

4. **Start the backend server**
   ```bash
   cd backend
   python app.py
   ```
   Server will run on `http://localhost:5000`

### Frontend Setup

1. **Install Node.js dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server**
   ```bash
   npm run serve
   ```
   Frontend will run on `http://localhost:8080`

### Docker Setup (Alternative)

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 📖 Usage Guide

### 1. AOI Selection

**Option A: Google Maps URL**
- Copy a Google Maps URL (e.g., `https://www.google.com/maps/@19.0419252,73.0270304,17z`)
- Paste it in the "Google Maps Link" field
- Click "Parse URL" to auto-fill coordinates

**Option B: Manual Coordinates**
- Enter latitude and longitude manually
- The map will update automatically

**Option C: Map Click**
- Click directly on the map to select a location
- Coordinates will auto-update

### 2. Date Selection

- Select "Before Date" for the baseline period
- Select "After Date" for the comparison period
- Use "Load Survey Data" to see available satellite imagery

### 3. Change Detection

1. Ensure coordinates and dates are selected
2. Click "Detect Changes"
3. Wait for processing (typically 30-60 seconds)
4. Review results in the right panel

### 4. Export Results

- **GeoTIFF**: Raster format for GIS analysis
- **GeoJSON**: Vector format for web mapping
- **Shapefile**: Standard GIS vector format
- **Complete Package**: All formats in a single ZIP

## 🔧 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### Health Check
```http
GET /health
```
Returns service status and health information.

#### Parse Google Maps URL
```http
POST /parse-gmaps-link
Content-Type: application/json

{
  "url": "https://www.google.com/maps/@19.0419252,73.0270304,17z"
}
```

#### Get Survey Data
```http
POST /survey-data
Content-Type: application/json

{
  "latitude": 19.0419252,
  "longitude": 73.0270304,
  "start_date": "2023-01-01",
  "end_date": "2023-12-31"
}
```

#### Change Detection
```http
POST /change-detection
Content-Type: application/json

{
  "latitude": 19.0419252,
  "longitude": 73.0270304,
  "before_date": "2023-01-01",
  "after_date": "2023-06-01"
}
```

#### Export Results
```http
POST /export
Content-Type: application/json

{
  "result_id": "abc123",
  "format": "geotiff"  // geotiff, geojson, shapefile
}
```

## 🤖 Siamese U-Net Model

The platform uses a pretrained Siamese U-Net architecture specifically designed for satellite imagery change detection:

### Model Architecture
- **Encoder**: Shared convolutional layers for feature extraction
- **Siamese Structure**: Processes before/after images simultaneously
- **U-Net Decoder**: Reconstructs change maps with skip connections
- **Output**: Binary change masks with confidence scores

### Model Features
- Pretrained on diverse satellite imagery datasets
- Handles 3-band RGB input (derived from Sentinel-2)
- 512x512 pixel input resolution
- Robust to seasonal variations and lighting changes

## 🌍 Google Earth Engine Integration

### Satellite Data Sources
- **Primary**: Sentinel-2 MSI Level-2A (Surface Reflectance)
- **Resolution**: 10m native resolution
- **Bands**: Red, Green, Blue, NIR
- **Coverage**: Global, 5-day revisit time

### SEN2RES Enhancement
The platform implements SEN2RES methodology for image enhancement:
- NIR-guided spatial enhancement
- Preserves original reflectance values
- Minimizes artifacts while improving visual quality
- Optimized specifically for Sentinel-2 data

### Cloud/Shadow Masking
- Automatic cloud detection using QA bands
- Shadow masking based on geometry and NIR values
- Quality scoring for image selection
- Fallback to alternative dates when needed

## 🎨 Frontend Design

### Minimal Black & White Theme
- **Colors**: Pure black (`#000000`) and white (`#ffffff`)
- **Typography**: Clean Arial/Helvetica fonts
- **Layout**: Grid-based with clear visual hierarchy
- **Interactions**: Subtle hover effects and transitions

### Responsive Design
- Desktop-first approach
- Mobile-optimized layouts
- Touch-friendly controls
- Scalable map interface

## 📊 Output Formats

### GeoTIFF
- **Use Case**: GIS analysis, QGIS/ArcGIS import
- **Content**: Georeferenced raster change map
- **Projection**: WGS84 (EPSG:4326)
- **Values**: 0 (no change), 255 (change detected)

### GeoJSON
- **Use Case**: Web mapping, lightweight analysis
- **Content**: Vector polygons of change regions
- **Properties**: Area calculations, confidence scores
- **Coordinates**: WGS84 decimal degrees

### Shapefile
- **Use Case**: Traditional GIS workflows
- **Content**: Vector polygons with attributes
- **Files**: .shp, .shx, .dbf, .prj, .cpg
- **Attributes**: ID, area, detection date, statistics

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```bash
# Google Earth Engine
GEE_PROJECT_ID=your-gee-project-id
GEE_SERVICE_ACCOUNT_KEY=path/to/service-account.json

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key

# CORS Origins
CORS_ORIGINS=http://localhost:8080,http://localhost:3000

# File Paths
TEMP_DIR=temp
EXPORTS_DIR=exports
MODELS_DIR=models
```

### Model Configuration

```python
# Change detection parameters
CHANGE_THRESHOLD = 0.5  # Binary threshold for change detection
MODEL_INPUT_SIZE = 512  # Input image dimensions
BATCH_SIZE = 1          # Processing batch size
DEVICE = 'cuda'         # 'cuda' or 'cpu'
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm run test:unit
```

### Integration Tests
```bash
# Start services
docker-compose up -d

# Run integration tests
python tests/integration/test_full_workflow.py
```

## 🔒 Security Considerations

- **API Rate Limiting**: Implemented for all endpoints
- **Input Validation**: Comprehensive validation for all user inputs
- **CORS Configuration**: Restricted to allowed origins
- **File Upload Limits**: Size and type restrictions
- **Error Handling**: No sensitive information in error messages

## 📈 Performance Optimization

### Backend
- **Async Processing**: Background tasks for long-running operations
- **Caching**: Result caching for repeated requests
- **Image Optimization**: Efficient image processing pipelines
- **Database Indexing**: Optimized queries for metadata

### Frontend
- **Code Splitting**: Lazy loading for map components
- **Image Optimization**: Progressive loading for results
- **Memory Management**: Proper cleanup of map layers
- **Bundle Size**: Optimized builds with tree shaking

## 🐛 Troubleshooting

### Common Issues

**Google Earth Engine Authentication**
```bash
# Re-authenticate if token expires
earthengine authenticate --force
```

**Model Loading Errors**
```bash
# Download model weights manually
mkdir -p backend/models
wget https://example.com/siamese_unet.pth -O backend/models/siamese_unet.pth
```

**Frontend Build Issues**
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**CORS Errors**
- Ensure backend is running on port 5000
- Check frontend proxy configuration in `vue.config.js`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint/Prettier for JavaScript
- Write comprehensive tests
- Update documentation for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Earth Engine** for satellite data access
- **Sentinel-2 Mission** for high-quality imagery
- **OpenLayers** for mapping capabilities
- **Vue.js Community** for frontend framework
- **Siamese U-Net Research** for change detection methodology

## 📞 Support

For support, please:
1. Check the [documentation](docs/)
2. Search [existing issues](issues)
3. Create a [new issue](issues/new) with detailed information

---

**Built with ❤️ for environmental monitoring and geospatial analysis** 