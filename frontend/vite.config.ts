import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        // Field inference on CPU can take 2-3 minutes — raise proxy timeout to 6 minutes
        proxyTimeout: 360000,
        timeout: 360000,
        // Additional proxy configuration for large file uploads
        ws: true,
        configure: (proxy) => {
          proxy.on('error', (err, req, res) => {
            console.log('Proxy error:', err);
          });
          proxy.on('proxyReq', (proxyReq, req, res) => {
            // Increase timeout for requests
            proxyReq.setTimeout(360000);
          });
        }
      },
    },
    // Increase dev server body size limit for file uploads
    hmr: { port: 5173 },
  },
})
