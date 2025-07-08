<template>
  <div id="app">
    <header class="app-header">
      <div class="container">
        <h1 class="app-title">Change Detection Platform</h1>
        <p class="app-subtitle">Satellite imagery analysis and monitoring</p>
      </div>
    </header>
    
    <main class="app-main">
      <router-view />
    </main>
    
    <footer class="app-footer">
      <div class="container">
        <p>&copy; 2024 Change Detection Platform. Powered by Siamese U-Net & Google Earth Engine.</p>
      </div>
    </footer>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style lang="scss">
#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #ffffff;
  color: #000000;
}

.app-header {
  background-color: #000000;
  color: #ffffff;
  padding: 1rem 0;
  border-bottom: 2px solid #000000;
  
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
  }
  
  .app-title {
    font-size: 2rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: 1px;
  }
  
  .app-subtitle {
    margin: 0.5rem 0 0 0;
    opacity: 0.8;
    font-size: 0.9rem;
  }
}

.app-main {
  flex: 1;
  background-color: #ffffff;
}

.app-footer {
  background-color: #f8f8f8;
  border-top: 1px solid #e0e0e0;
  padding: 1rem 0;
  margin-top: auto;
  
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
  }
  
  p {
    margin: 0;
    font-size: 0.8rem;
    color: #666666;
    text-align: center;
  }
}
</style> 