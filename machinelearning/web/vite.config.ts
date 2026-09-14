import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8081',
      '/health': 'http://127.0.0.1:8081',
      '/docs': 'http://127.0.0.1:8081',
      '/openapi.json': 'http://127.0.0.1:8081',
    },
  },
  build: {
    outDir: '../dots_cordon_ml/audit/web/static',
    emptyOutDir: true,
  },
  test: { include: ['src/**/*.test.ts'] },
})
