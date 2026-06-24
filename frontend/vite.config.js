import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    proxy: {
      '/transcribe': { target: 'http://127.0.0.1:8000', timeout: 600000 },
      '/reason': { target: 'http://127.0.0.1:8000', timeout: 600000 },
      '/speak': { target: 'http://127.0.0.1:8000', timeout: 120000 },
      '/consult': { target: 'http://127.0.0.1:8000', timeout: 600000 },
      '/health': { target: 'http://127.0.0.1:8000', timeout: 30000 },
    },
  },
})
