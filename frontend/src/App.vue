<template>
  <div id="app">
    <main class="app-main">
      <router-view />
    </main>
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

.app-main {
  flex: 1;
  background-color: #ffffff;
}
</style> 