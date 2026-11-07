/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_')
  // Production builds are served from https://<user>.github.io/WriteRHand/ unless overridden.
  const base = env.VITE_BASE ?? (command === 'build' ? '/WriteRHand/' : '/')
  return {
    base,
    plugins: [react()],
    build: {
      target: 'es2022',
      chunkSizeWarningLimit: 700,
    },
    test: {
      environment: 'node',
      include: ['tests/**/*.test.ts'],
      testTimeout: 60_000,
    },
  }
})
