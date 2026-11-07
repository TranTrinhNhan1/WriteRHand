// Export slides/index.html to a 16:9 PDF (and optional PNG previews) with Playwright + Chromium.
//   node slides/export_pdf.mjs [--png <dir>]
// Needs `playwright-core` and a Chromium executable (CHROME env var, or the Playwright cache).
import { chromium } from 'playwright-core'
import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'
import fs from 'node:fs'

const here = path.dirname(fileURLToPath(import.meta.url))
const exe = process.env.CHROME || process.env.HOME + '/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome'
const pngIdx = process.argv.indexOf('--png')
const pngDir = pngIdx > 0 ? process.argv[pngIdx + 1] : null

const browser = await chromium.launch({ executablePath: exe, args: ['--no-sandbox'] })
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } })
await page.goto(pathToFileURL(path.join(here, 'index.html')).href, { waitUntil: 'networkidle' })
await page.evaluate(() => document.fonts.ready)

if (pngDir) {
  fs.mkdirSync(pngDir, { recursive: true })
  const n = await page.locator('.slide').count()
  for (let i = 1; i <= n; i++) {
    await page.goto(pathToFileURL(path.join(here, 'index.html')).href + '#' + i)
    await page.waitForTimeout(150)
    await page.screenshot({ path: path.join(pngDir, `slide-${String(i).padStart(2, '0')}.png`) })
  }
  await page.goto(pathToFileURL(path.join(here, 'index.html')).href)
}
await page.pdf({ path: path.join(here, 'WriteRHand-slides.pdf'), width: '1920px', height: '1080px', printBackground: true, preferCSSPageSize: true })
await browser.close()
console.log('wrote slides/WriteRHand-slides.pdf')
