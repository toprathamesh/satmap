<template>
  <div class="home">
    <div class="container">
      <div class="row">
        <!-- Left Panel - Map and Controls -->
        <div class="col col-half">
          <div class="card">
            <div class="card-header">
              <h3>Area of Interest Selection</h3>
            </div>
            <div class="card-body">
              <!-- Google Maps URL Input -->
              <div class="input-group">
                <label for="gmaps-url">Google Maps Link (Optional)</label>
                <input
                  id="gmaps-url"
                  v-model="gmapsUrl"
                  type="text"
                  placeholder="Paste Google Maps URL here..."
                  @blur="parseGoogleMapsUrl"
                />
                <button @click="parseGoogleMapsUrl" class="btn btn-small">Parse URL</button>
              </div>
              
              <!-- Manual Coordinates -->
              <div class="row">
                <div class="col">
                  <div class="input-group">
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
                </div>
                <div class="col">
                  <div class="input-group">
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
              </div>
              
              <!-- Map Container -->
              <div id="map" class="map-container"></div>
              
              <!-- Date Selection -->
              <div class="row">
                <div class="col">
                  <div class="input-group">
                    <label for="before-date">Before Date</label>
                    <input
                      id="before-date"
                      v-model="dates.before"
                      type="date"
                      :max="dates.after"
                    />
                  </div>
                </div>
                <div class="col">
                  <div class="input-group">
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
              </div>
              
              <!-- Action Buttons -->
              <div class="card-footer">
                <button
                  @click="detectChanges"
                  :disabled="!canDetectChanges || loading"
                  class="btn btn-primary"
                >
                  <span v-if="loading" class="loading"></span>
                  Detect Changes
                </button>
                <button
                  @click="loadSurveyData"
                  :disabled="!coordinates.lat || !coordinates.lon || loading"
                  class="btn"
                >
                  Load Survey Data
                </button>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Right Panel - Survey Data and Results -->
        <div class="col col-half">
          <!-- Survey Data Panel -->
          <div class="card" v-if="surveyData.length > 0">
            <div class="card-header">
              <h3>Available Survey Data</h3>
            </div>
            <div class="card-body">
              <div class="survey-list">
                <div
                  v-for="item in surveyData"
                  :key="item.date"
                  class="survey-item"
                  @click="selectSurveyDate(item.date)"
                  :class="{ selected: selectedSurveyDate === item.date }"
                >
                  <div class="survey-date">{{ formatDate(item.date) }}</div>
                  <div class="survey-info">
                    <span>{{ item.satellite }}</span>
                    <span class="cloud-cover">{{ item.cloud_cover }}% clouds</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Change Detection Results -->
          <div class="card" v-if="changeResults">
            <div class="card-header">
              <h3>Change Detection Results</h3>
            </div>
            <div class="card-body">
              <!-- Statistics -->
              <div class="stats-grid">
                <div class="stat-item">
                  <div class="stat-value">{{ changeResults.statistics.change_percentage }}%</div>
                  <div class="stat-label">Area Changed</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ changeResults.statistics.changed_area_km2 }}</div>
                  <div class="stat-label">km² Changed</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ changeResults.statistics.num_change_regions }}</div>
                  <div class="stat-label">Change Regions</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ changeResults.statistics.largest_region_km2 }}</div>
                  <div class="stat-label">Largest Region (km²)</div>
                </div>
              </div>
              
              <!-- Image Comparison View -->
              <div class="image-comparison-section" v-if="changeResults.files">
                <!-- Before/After Images Side by Side -->
                <div class="before-after-grid" v-if="changeResults.files.before_image && changeResults.files.after_highlighted">
                  <div class="image-container">
                    <h4>Before</h4>
                    <img
                      :src="getVisualizationUrl(changeResults.files.before_image)"
                      alt="Before Image"
                      class="comparison-image"
                    />
                    <p class="image-date">{{ formatDate(dates.before) }}</p>
                  </div>
                  <div class="image-container">
                    <h4>After (Changes Highlighted)</h4>
                    <img
                      :src="getVisualizationUrl(changeResults.files.after_highlighted)"
                      alt="After Image with Changes"
                      class="comparison-image"
                    />
                    <p class="image-date">{{ formatDate(dates.after) }}</p>
                  </div>
                </div>
                
                <!-- Full Comparison View -->
                <div class="full-comparison" v-if="changeResults.files.comparison">
                  <h4>Side-by-Side Comparison</h4>
                  <img
                    :src="getVisualizationUrl(changeResults.files.comparison)"
                    alt="Before/After Comparison"
                    class="full-comparison-image"
                  />
                </div>
                
                <!-- Change Probability Heatmap -->
                <div class="heatmap-section" v-if="changeResults.files.heatmap">
                  <h4>Change Probability Heatmap</h4>
                  <img
                    :src="getVisualizationUrl(changeResults.files.heatmap)"
                    alt="Change Probability Heatmap"
                    class="heatmap-image"
                  />
                  <p class="heatmap-legend">
                    <span class="legend-item blue">Low Probability</span>
                    <span class="legend-item green">Medium Probability</span>
                    <span class="legend-item red">High Probability</span>
                  </p>
                </div>
                
                <!-- Additional Stats -->
                <div class="additional-stats" v-if="changeResults.statistics.average_region_size_km2">
                  <div class="stat-row">
                    <span class="stat-name">Average Region Size:</span>
                    <span class="stat-value">{{ changeResults.statistics.average_region_size_km2 }} km²</span>
                  </div>
                  <div class="stat-row">
                    <span class="stat-name">Total Survey Area:</span>
                    <span class="stat-value">{{ changeResults.statistics.total_area_km2 }} km²</span>
                  </div>
                </div>
              </div>
              
              <!-- Export Options -->
              <div class="card-footer">
                <button @click="exportResults('geotiff')" class="btn btn-small">Export GeoTIFF</button>
                <button @click="exportResults('geojson')" class="btn btn-small">Export GeoJSON</button>
                <button @click="exportResults('shapefile')" class="btn btn-small">Export Shapefile</button>
                <button @click="exportResults('complete')" class="btn">Complete Package</button>
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
      surveyData: [],
      selectedSurveyDate: null,
      changeResults: null,
      loading: false,
      error: null,
      aoiLayer: null,
      centerLayer: null
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
    
    async loadSurveyData() {
      this.loading = true
      this.error = null
      
      try {
        // Use a date range around the selected dates
        const startDate = this.dates.before || '2020-01-01'
        const endDate = this.dates.after || new Date().toISOString().split('T')[0]
        
        const response = await axios.post(`${this.apiBaseUrl}/api/survey-data`, {
          latitude: this.coordinates.lat,
          longitude: this.coordinates.lon,
          start_date: startDate,
          end_date: endDate
        })
        
        if (response.data.success) {
          this.surveyData = response.data.data
        }
      } catch (error) {
        this.error = 'Failed to load survey data'
        console.error('Error loading survey data:', error)
      } finally {
        this.loading = false
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
        }
      } catch (error) {
        this.error = 'Change detection failed: ' + (error.response?.data?.error || error.message)
        console.error('Error in change detection:', error)
      } finally {
        this.loading = false
      }
    },
    
    async exportResults(format) {
      if (!this.changeResults) return
      
      try {
        const exportFormat = format === 'complete' ? 'complete_package' : format
        const response = await axios.post(`${this.apiBaseUrl}/api/export`, {
          result_id: this.changeResults.result_id,
          format: exportFormat
        }, {
          responseType: 'blob'
        })
        
        // Create download link
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        
        const extension = format === 'geotiff' ? 'tif' : 
                         format === 'geojson' ? 'geojson' :
                         format === 'shapefile' ? 'zip' : 'zip'
        
        link.setAttribute('download', `change_detection_${this.changeResults.result_id}.${extension}`)
        document.body.appendChild(link)
        link.click()
        link.remove()
        window.URL.revokeObjectURL(url)
        
      } catch (error) {
        this.error = 'Export failed: ' + (error.response?.data?.error || error.message)
        console.error('Error exporting:', error)
      }
    },
    
    selectSurveyDate(date) {
      this.selectedSurveyDate = date
      // Auto-fill date if appropriate
      if (!this.dates.before) {
        this.dates.before = date
      } else if (!this.dates.after && date > this.dates.before) {
        this.dates.after = date
      }
    },
    
    setDefaultDates() {
      const today = new Date()
      const sixMonthsAgo = new Date()
      sixMonthsAgo.setMonth(today.getMonth() - 6)
      
      this.dates.after = today.toISOString().split('T')[0]
      this.dates.before = sixMonthsAgo.toISOString().split('T')[0]
    },
    
    formatDate(dateString) {
      return format(new Date(dateString), 'MMM dd, yyyy')
    },
    
    getVisualizationUrl(path) {
      return `${this.apiBaseUrl}/${path.replace(/^\//, '')}`
    }
  }
}
</script>

<style lang="scss" scoped>
.survey-list {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid #e0e0e0;
}

.survey-item {
  padding: 0.75rem;
  border-bottom: 1px solid #f0f0f0;
  cursor: pointer;
  transition: background-color 0.2s ease;
  
  &:hover {
    background-color: #f8f8f8;
  }
  
  &.selected {
    background-color: #000000;
    color: #ffffff;
  }
  
  .survey-date {
    font-weight: 500;
    margin-bottom: 0.25rem;
  }
  
  .survey-info {
    font-size: 0.8rem;
    opacity: 0.8;
    
    span {
      margin-right: 1rem;
    }
  }
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1rem;
}

.stat-item {
  text-align: center;
  padding: 1rem;
  border: 1px solid #e0e0e0;
  
  .stat-value {
    font-size: 1.5rem;
    font-weight: 500;
    color: #000000;
  }
  
  .stat-label {
    font-size: 0.8rem;
    color: #666666;
    margin-top: 0.25rem;
  }
}

.change-visualization {
  width: 100%;
  height: auto;
  border: 2px solid #e0e0e0;
  margin-bottom: 1rem;
}

.visualization {
  margin-bottom: 1rem;
}

/* New styles for enhanced image comparison */
.image-comparison-section {
  margin-bottom: 1.5rem;
}

.before-after-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.image-container {
  text-align: center;
  
  h4 {
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
    color: #000000;
  }
}

.comparison-image {
  width: 100%;
  height: auto;
  border: 2px solid #e0e0e0;
  border-radius: 4px;
  margin-bottom: 0.5rem;
}

.image-date {
  font-size: 0.8rem;
  color: #666666;
  margin: 0;
}

.full-comparison {
  text-align: center;
  margin-bottom: 1.5rem;
  
  h4 {
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
    color: #000000;
  }
}

.full-comparison-image {
  width: 100%;
  height: auto;
  border: 2px solid #e0e0e0;
  border-radius: 4px;
}

.heatmap-section {
  text-align: center;
  margin-bottom: 1.5rem;
  
  h4 {
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
    color: #000000;
  }
}

.heatmap-image {
  width: 100%;
  height: auto;
  border: 2px solid #e0e0e0;
  border-radius: 4px;
  margin-bottom: 0.5rem;
}

.heatmap-legend {
  display: flex;
  justify-content: center;
  gap: 1rem;
  font-size: 0.8rem;
  margin: 0;
}

.legend-item {
  padding: 0.25rem 0.5rem;
  border-radius: 3px;
  font-weight: 500;
  
  &.blue {
    background-color: #0066cc;
    color: white;
  }
  
  &.green {
    background-color: #00cc66;
    color: white;
  }
  
  &.red {
    background-color: #cc0000;
    color: white;
  }
}

.additional-stats {
  border-top: 1px solid #e0e0e0;
  padding-top: 1rem;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
  border-bottom: 1px solid #f0f0f0;
  
  &:last-child {
    border-bottom: none;
  }
}

.stat-name {
  font-weight: 500;
  color: #333333;
}

.stat-value {
  font-weight: 500;
  color: #000000;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .before-after-grid {
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }
  
  .heatmap-legend {
    flex-direction: column;
    gap: 0.5rem;
  }
}
</style> 