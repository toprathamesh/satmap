<template>
  <div class="home">
    <div class="container">
      <!-- Main Content - Centered -->
      <div class="main-content">
        <!-- Area Selection Card - Centered -->
        <div class="selection-card">
          <div class="card-header">
            <h2>Area of Interest Selection</h2>
          </div>
          
          <div class="card-body">
            <!-- Manual Coordinates First -->
            <div class="coordinates-row">
              <div class="coordinate-input">
                <label for="latitude">Latitude</label>
                <input
                  id="latitude"
                  v-model.number="coordinates.lat"
                  type="number"
                  step="0.000001"
                  placeholder="19.0419252"
                  @change="updateMapCenter"
                />
              </div>
              <div class="coordinate-input">
                <label for="longitude">Longitude</label>
                <input
                  id="longitude"
                  v-model.number="coordinates.lon"
                  type="number"
                  step="0.000001"
                  placeholder="73.0270304"
                  @change="updateMapCenter"
                />
              </div>
            </div>
            
            <!-- Google Maps URL Input - Moved After Coordinates -->
            <div class="input-section">
              <label for="gmaps-url">Google Maps Link (Optional)</label>
              <div class="input-with-button">
                <input
                  id="gmaps-url"
                  v-model="gmapsUrl"
                  type="text"
                  placeholder="Paste Google Maps URL here..."
                  @blur="parseGoogleMapsUrl"
                />
                <button @click="parseGoogleMapsUrl" class="btn btn-secondary">Parse</button>
              </div>
            </div>
            
            <!-- Map Container -->
            <div class="map-section">
              <div id="map" class="map-container"></div>
            </div>
            
            <!-- Date Selection -->
            <div class="dates-row">
              <div class="date-input">
                <label for="before-date">Before Date</label>
                <input
                  id="before-date"
                  v-model="dates.before"
                  type="date"
                  :max="dates.after"
                />
              </div>
              <div class="date-input">
                <label for="after-date">After Date</label>
                <input
                  id="after-date"
                  v-model="dates.after"
                  type="date"
                  :min="dates.before"
                  :max="maxDate"
                />
              </div>
            </div>
            
            <!-- Action Button - Centered -->
            <div class="action-buttons">
              <button
                @click="detectChanges"
                :disabled="!canDetectChanges || loading"
                class="btn btn-primary"
              >
                <span v-if="loading" class="loading"></span>
                <span v-if="!loading">Detect Changes</span>
                <span v-if="loading">Processing...</span>
              </button>
            </div>
          </div>
        </div>

        <!-- Results Section - Below Selection -->
        <div v-if="results && !loading" class="results-section">
          <div class="card-header">
            <h2>Change Detection Results</h2>
          </div>
          
          <!-- Statistics Summary -->
          <div class="stats-summary">
            <div class="stat-card primary">
              <div class="stat-label">Change Detected</div>
              <div class="stat-value">{{ results.statistics?.change_percentage?.toFixed(1) || '0' }}%</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Model Used</div>
              <div class="stat-value">{{ results.model || 'Lightweight Siamese U-Net' }}</div>
            </div>
          </div>
          
          <!-- Image Results Grid -->
          <div class="image-results">
            <div class="image-grid">
              <div class="image-item">
                <h3>Before Image</h3>
                <img v-if="results.files?.before_image" :src="getImageUrl(results.files.before_image)" alt="Before" class="result-image">
                <div v-else class="no-image">No before image available</div>
              </div>
              
              <div class="image-item">
                <h3>After Image</h3>
                <img v-if="results.files?.after_image" :src="getImageUrl(results.files.after_image)" alt="After" class="result-image">
                <div v-else class="no-image">No after image available</div>
              </div>
              
              <div class="image-item">
                <h3>Change Comparison</h3>
                <img v-if="results.files?.comparison" :src="getImageUrl(results.files.comparison)" alt="Comparison" class="result-image">
                <div v-else class="no-image">No comparison image available</div>
              </div>
              
              <div class="image-item">
                <h3>Change Heatmap</h3>
                <img v-if="results.files?.heatmap" :src="getImageUrl(results.files.heatmap)" alt="Heatmap" class="result-image">
                <div v-else class="no-image">No heatmap available</div>
              </div>
            </div>
          </div>

          <!-- Export Options (PDF Only) -->
          <div class="export-section">
            <h2>Export Results</h2>
            <button @click="exportPDF" :disabled="exporting" class="btn btn-export">
              <span class="icon">📄</span>
              <span v-if="exporting" class="loading"></span>
              <span v-if="!exporting">Download PDF Report</span>
              <span v-if="exporting">Generating PDF...</span>
            </button>
            <div v-if="showExportMessage" class="alert alert-success">
              {{ exportMessage }}
            </div>
          </div>
        </div>

        <!-- Error Display -->
        <div class="alert alert-error" v-if="error">
          {{ error }}
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Map, View } from 'ol'
import TileLayer from 'ol/layer/Tile'
import OSM from 'ol/source/OSM'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import Feature from 'ol/Feature'
import Point from 'ol/geom/Point'
import Polygon from 'ol/geom/Polygon'
import { Style, Fill, Stroke, Circle } from 'ol/style'
import { fromLonLat, toLonLat } from 'ol/proj'
import { format } from 'date-fns'
import axios from 'axios'

export default {
  name: 'Home',
  data() {
    return {
      apiBaseUrl: process.env.VUE_APP_API_URL || 'http://localhost:5000',
      map: null,
      coordinates: {
        lat: 19.0419252,
        lon: 73.0270304
      },
      dates: {
        before: '',
        after: ''
      },
      gmapsUrl: '',
      changeResults: null,
      loading: false,
      error: null,
      aoiLayer: null,
      centerLayer: null,
      exporting: false,
      showExportMessage: false,
      exportMessage: '',
      results: null
    }
  },
  computed: {
    maxDate() {
      return new Date().toISOString().split('T')[0]
    },
    canDetectChanges() {
      return (
        this.coordinates.lat &&
        this.coordinates.lon &&
        this.dates.before &&
        this.dates.after
      )
    }
  },
  mounted() {
    this.initMap()
    this.setDefaultDates()
  },
  methods: {
    initMap() {
      // Create AOI layer
      this.aoiLayer = new VectorLayer({
        source: new VectorSource(),
        style: new Style({
          fill: new Fill({
            color: 'rgba(0, 0, 0, 0.1)'
          }),
          stroke: new Stroke({
            color: '#000000',
            width: 2
          })
        })
      })
      
      // Create center point layer
      this.centerLayer = new VectorLayer({
        source: new VectorSource(),
        style: new Style({
          image: new Circle({
            radius: 6,
            fill: new Fill({ color: '#000000' }),
            stroke: new Stroke({
              color: '#ffffff',
              width: 2
            })
          })
        })
      })
      
      // Create map
      this.map = new Map({
        target: 'map',
        layers: [
          new TileLayer({
            source: new OSM()
          }),
          this.aoiLayer,
          this.centerLayer
        ],
        view: new View({
          center: fromLonLat([this.coordinates.lon, this.coordinates.lat]),
          zoom: 15
        })
      })
      
      // Add click handler
      this.map.on('click', this.onMapClick)
      
      // Initial AOI display
      this.updateAOI()
    },
    
    onMapClick(event) {
      const coords = toLonLat(event.coordinate)
      this.coordinates.lon = coords[0]
      this.coordinates.lat = coords[1]
      this.updateAOI()
    },
    
    updateMapCenter() {
      if (this.coordinates.lat && this.coordinates.lon) {
        this.map.getView().setCenter(fromLonLat([this.coordinates.lon, this.coordinates.lat]))
        this.updateAOI()
      }
    },
    
    updateAOI() {
      // Clear existing features
      this.aoiLayer.getSource().clear()
      this.centerLayer.getSource().clear()
      
      if (!this.coordinates.lat || !this.coordinates.lon) return
      
      // Add center point
      const centerFeature = new Feature({
        geometry: new Point(fromLonLat([this.coordinates.lon, this.coordinates.lat]))
      })
      this.centerLayer.getSource().addFeature(centerFeature)
      
      // Add 1km x 1km AOI square
      const sizeKm = 1.0
      const sizeDegrees = sizeKm / 111.32 // Approximate conversion
      const halfSize = sizeDegrees / 2
      
      const aoiCoords = [
        [this.coordinates.lon - halfSize, this.coordinates.lat - halfSize],
        [this.coordinates.lon + halfSize, this.coordinates.lat - halfSize],
        [this.coordinates.lon + halfSize, this.coordinates.lat + halfSize],
        [this.coordinates.lon - halfSize, this.coordinates.lat + halfSize],
        [this.coordinates.lon - halfSize, this.coordinates.lat - halfSize]
      ].map(coord => fromLonLat(coord))
      
      const aoiFeature = new Feature({
        geometry: new Polygon([aoiCoords])
      })
      this.aoiLayer.getSource().addFeature(aoiFeature)
    },
    
    async parseGoogleMapsUrl() {
      if (!this.gmapsUrl.trim()) return
      
      try {
        const response = await axios.post(`${this.apiBaseUrl}/api/parse-gmaps-link`, {
          url: this.gmapsUrl
        })
        
        if (response.data.success) {
          this.coordinates.lat = response.data.latitude
          this.coordinates.lon = response.data.longitude
          this.updateMapCenter()
          this.error = null
        }
      } catch (error) {
        this.error = 'Could not parse Google Maps URL'
        console.error('Error parsing URL:', error)
      }
    },
    

    
    async detectChanges() {
      this.loading = true
      this.error = null
      this.changeResults = null
      
      try {
        const response = await axios.post(`${this.apiBaseUrl}/api/change-detection`, {
          latitude: this.coordinates.lat,
          longitude: this.coordinates.lon,
          before_date: this.dates.before,
          after_date: this.dates.after
        })
        
        if (response.data.success) {
          this.changeResults = response.data.data
          this.results = this.changeResults // Assign results to the new state
        }
      } catch (error) {
        this.error = 'Change detection failed: ' + (error.response?.data?.error || error.message)
        console.error('Error in change detection:', error)
      } finally {
        this.loading = false
      }
    },
    

    

    
    async exportPDF() {
      if (!this.results) return;
      
      this.exporting = true;
      try {
        const response = await axios.post('http://localhost:5000/api/export/pdf', {
          result_id: this.results.result_id
        }, {
          responseType: 'blob'
        });
        
        // Check if we received a PDF or an error
        if (response.headers['content-type'] === 'application/json') {
          // This means we got an error response as JSON
          const errorText = await response.data.text();
          const errorData = JSON.parse(errorText);
          throw new Error(errorData.error || 'PDF generation failed');
        }
        
        // Create download link
        const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `${this.results.result_id}_report.pdf`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
        
        this.exportMessage = 'PDF report downloaded successfully!';
        this.showExportMessage = true;
        setTimeout(() => { this.showExportMessage = false; }, 3000);
        
      } catch (error) {
        console.error('Error exporting PDF:', error);
        let errorMessage = 'Failed to generate PDF report';
        
        if (error.response) {
          // Handle different error types
          if (error.response.status === 500) {
            errorMessage = 'Server error generating PDF. Please try again.';
          } else if (error.response.data && error.response.data.error) {
            errorMessage = error.response.data.error;
          }
        } else if (error.message) {
          errorMessage = error.message;
        }
        
        this.exportMessage = errorMessage;
        this.showExportMessage = true;
        setTimeout(() => { this.showExportMessage = false; }, 5000);
      } finally {
        this.exporting = false;
      }
    },
    
    setDefaultDates() {
      const today = new Date()
      const oneYearAgo = new Date()
      oneYearAgo.setFullYear(today.getFullYear() - 1)
      
      this.dates.after = today.toISOString().split('T')[0]
      this.dates.before = oneYearAgo.toISOString().split('T')[0]
    },
    
    formatDate(dateString) {
      return format(new Date(dateString), 'MMM dd, yyyy')
    },
    
    getVisualizationUrl(path) {
      return `${this.apiBaseUrl}/${path.replace(/^\//, '')}`
    },
    
    getImageUrl(imagePath) {
      // Fix image URL to use proper backend serving
      if (!imagePath) return '';
      
      // Remove 'images/' prefix if it exists since we're serving from root
      const cleanPath = imagePath.replace(/^(backend\/)?images\//, '');
      return `http://localhost:5000/images/${cleanPath}`;
    }
  }
}
</script>

<style lang="scss" scoped>
.home {
  min-height: 100vh;
  padding: 1rem 0;
  background-color: #ffffff;
}



/* Main Content Container */
.main-content {
  max-width: 900px;
  margin: 1rem auto;
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

/* Card Styles */
.selection-card,
.results-section {
  background-color: #ffffff;
  border: 3px solid #000000;
  padding: 2rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.card-header {
  text-align: center;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 2px solid #e0e0e0;
}

.card-header h2 {
  font-size: 1.8rem;
  font-weight: 600;
  color: #000000;
  margin: 0;
}

/* Input Sections */
.input-section {
  margin-bottom: 2rem;
}

.input-section label {
  display: block;
  font-size: 1rem;
  font-weight: 600;
  color: #000000;
  margin-bottom: 0.5rem;
}

.input-with-button {
  display: flex;
  gap: 0.5rem;
}

.input-with-button input {
  flex: 1;
  padding: 1rem;
  border: 2px solid #e0e0e0;
  font-size: 1rem;
  transition: border-color 0.2s ease;
}

.input-with-button input:focus {
  outline: none;
  border-color: #000000;
}

/* Coordinate and Date Inputs */
.coordinates-row,
.dates-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 2rem;
}

.coordinate-input,
.date-input {
  display: flex;
  flex-direction: column;
}

.coordinate-input label,
.date-input label {
  font-size: 1rem;
  font-weight: 600;
  color: #000000;
  margin-bottom: 0.5rem;
}

.coordinate-input input,
.date-input input {
  padding: 1rem;
  border: 2px solid #e0e0e0;
  font-size: 1rem;
  transition: border-color 0.2s ease;
}

.coordinate-input input:focus,
.date-input input:focus {
  outline: none;
  border-color: #000000;
}

/* Map Section */
.map-section {
  margin-bottom: 2rem;
}

.map-container {
  height: 400px;
  border: 3px solid #000000;
  background-color: #f8f8f8;
}

/* Action Buttons */
.action-buttons {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 2rem;
}

.btn {
  padding: 1rem 2rem;
  font-size: 1rem;
  font-weight: 600;
  border: 2px solid #000000;
  background-color: #ffffff;
  color: #000000;
  cursor: pointer;
  transition: all 0.2s ease;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.btn:hover:not(:disabled) {
  background-color: #000000;
  color: #ffffff;
}

.btn-primary {
  background-color: #000000;
  color: #ffffff;
}

.btn-primary:hover:not(:disabled) {
  background-color: #333333;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}



/* Results Section */
.stats-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.stat-card {
  text-align: center;
  padding: 1.5rem;
  border: 2px solid #e0e0e0;
  background-color: #f8f8f8;
}

.stat-card.primary {
  border-color: #000000;
  background-color: #000000;
  color: #ffffff;
}

.stat-label {
  font-size: 0.9rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 0.5rem;
}

.stat-value {
  font-size: 1.8rem;
  font-weight: 700;
}

/* Image Results */
.image-results {
  margin-bottom: 2rem;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

.image-item {
  text-align: center;
  border: 2px solid #e0e0e0;
  padding: 1rem;
  background-color: #ffffff;
}

.image-item h3 {
  font-size: 1.1rem;
  font-weight: 600;
  color: #000000;
  margin-bottom: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.result-image {
  width: 100%;
  height: auto;
  max-height: 300px;
  object-fit: contain;
  border: 2px solid #e0e0e0;
}

.no-image {
  padding: 2rem;
  color: #999999;
  font-style: italic;
  background-color: #f8f8f8;
  border: 2px dashed #e0e0e0;
}

/* Export Section */
.export-section {
  text-align: center;
  padding-top: 1.5rem;
  border-top: 2px solid #e0e0e0;
}

.export-section h2 {
  font-size: 1.4rem;
  font-weight: 600;
  color: #000000;
  margin-bottom: 1rem;
}

.btn-export {
  background-color: #000000;
  color: #ffffff;
  padding: 1.2rem 2.5rem;
  font-size: 1.1rem;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.btn-export:hover:not(:disabled) {
  background-color: #333333;
}

.btn-export .icon {
  font-size: 1.2rem;
}

/* Alert Styles */
.alert {
  padding: 1rem;
  margin-top: 1rem;
  border: 2px solid;
  text-align: center;
  font-weight: 500;
}

.alert-success {
  background-color: #f8f8f8;
  border-color: #000000;
  color: #000000;
}

.alert-error {
  background-color: #ffffff;
  border-color: #000000;
  color: #000000;
}

/* Loading Spinner */
.loading {
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 3px solid #e0e0e0;
  border-radius: 50%;
  border-top-color: #000000;
  animation: spin 1s ease-in-out infinite;
  margin-right: 0.5rem;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* Responsive Design */
@media (max-width: 768px) {
  .app-header h1 {
    font-size: 2.2rem;
  }
  
  .subtitle {
    font-size: 1rem;
  }
  
  .main-content {
    max-width: 100%;
    padding: 0 1rem;
  }
  
     .selection-card,
   .results-section {
     padding: 1.5rem;
   }
  
  .coordinates-row,
  .dates-row {
    grid-template-columns: 1fr;
  }
  
  .action-buttons {
    flex-direction: column;
  }
  
  .stats-summary {
    grid-template-columns: 1fr;
  }
  
  .image-grid {
    grid-template-columns: 1fr;
  }
}
</style> 