import { defineConfig, devices } from '@playwright/test'
import { existsSync } from 'node:fs'

/**
 * End-to-end tests run against a production build of the site that reads the model bundle from
 * `public/bundle` (fetched with `npm run bundle:fetch` from the revision pinned in model-lock.json).
 * CI installs Chromium with `npx playwright install chromium`; locally set PW_CHROMIUM to an
 * existing browser binary to avoid a download.
 */
if (!existsSync('public/bundle/manifest.json')) {
  throw new Error('public/bundle/manifest.json is missing: run `npm run bundle:fetch` first (the E2E tests never skip)')
}
const port = 4173
export default defineConfig({
  testDir: 'e2e',
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: 'retain-on-failure',
    launchOptions: process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {},
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } } }],
  webServer: {
    command: `node scripts/docs.mjs prepare && npx vite build --outDir dist-e2e --emptyOutDir && npx vite preview --outDir dist-e2e --host 127.0.0.1 --port ${port} --strictPort`,
    env: { VITE_BASE: '/', VITE_MODEL_BASE_URL: '/bundle' },
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: !process.env.CI,
    timeout: 240_000,
  },
})
