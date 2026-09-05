import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The FastAPI backend serves the API on the same origin; in dev it runs on 5173
// and we proxy /api to the uvicorn backend on 8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    allowedHosts: ['.monkeycode-ai.live'],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 4000,
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ['plotly.js-dist-min'],
        },
      },
    },
  },
});
