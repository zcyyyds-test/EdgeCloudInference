import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/detect': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/analyze': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
})
