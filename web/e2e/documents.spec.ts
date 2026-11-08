import { readFileSync } from 'node:fs'
import { expect, test, type Page } from '@playwright/test'
import { card, predicted, watchProblems } from './helpers'

/**
 * The Documents page and the header links, against the production build: the PDFs are the files published by
 * `scripts/docs.mjs prepare`, and their metadata comes from docs/docs.json. Nothing is mocked.
 */

const REPO_URL = 'https://github.com/TranTrinhNhan1/WriteRHand'
/** single source of truth: src/lib/config.ts (read as text, so the Node-side type check does not pull in Vite code) */
const REPO_PUBLIC = /export const REPO_PUBLIC = true\b/.test(readFileSync('src/lib/config.ts', 'utf8'))

const DOCS = [
  { id: 'report', section: 'Báo cáo khoa học', file: 'report.pdf' },
  { id: 'slides', section: 'Slide thuyết trình', file: 'slides.pdf' },
] as const

interface DocsJson {
  report: { file: string; pages: number; bytes: number; sha256: string }
  slides: { file: string; pages: number; bytes: number; sha256: string }
  generated: string
}

const githubLink = (page: Page) => page.getByRole('banner').getByRole('link', { name: 'Mã nguồn trên GitHub' })
const docsNav = (page: Page) => page.getByRole('navigation', { name: 'Điều hướng chính' }).getByRole('link', { name: 'Tài liệu' })

async function openDocuments(page: Page) {
  await page.goto('/')
  await docsNav(page).click()
  await expect(page).toHaveURL(/#\/documents$/)
  await expect(page.getByRole('heading', { level: 1, name: 'Báo cáo, slide và mã nguồn' })).toBeVisible()
}

test('the header links to the GitHub repository in a new tab', async ({ page }) => {
  await page.goto('/')
  const gh = githubLink(page)
  await expect(gh).toBeVisible()
  await expect(gh).toHaveAttribute('href', REPO_URL)
  await expect(gh).toHaveAttribute('target', '_blank')
  await expect(gh).toHaveAttribute('rel', 'noopener noreferrer')
})

test('the Documents page lists the report and the slides with their page counts, and the PDFs are served', async ({ page, request }) => {
  await openDocuments(page)
  await expect(docsNav(page)).toHaveAttribute('aria-current', 'page')
  const meta = (await (await request.get('docs/docs.json')).json()) as DocsJson

  for (const d of DOCS) {
    const section = page.getByRole('region', { name: d.section })
    await expect(section).toBeVisible()
    const pages = section.locator('[data-pages]')
    await expect(pages).toHaveText(`${meta[d.id].pages} trang`)
    expect(Number(await pages.getAttribute('data-pages'))).toBeGreaterThan(0)
    expect(meta[d.id].file).toBe(d.file)

    const view = section.getByRole('link', { name: /^Xem PDF/ })
    await expect(view).toHaveAttribute('href', `/docs/${d.file}`)
    await expect(view).toHaveAttribute('target', '_blank')
    const download = section.getByRole('link', { name: /^Tải xuống/ })
    await expect(download).toHaveAttribute('href', `/docs/${d.file}`)
    await expect(download).toHaveAttribute('download', /\.pdf$/)

    const res = await request.get(`docs/${d.file}`)
    expect(res.status(), d.file).toBe(200)
    expect(res.headers()['content-type'], d.file).toMatch(/^application\/pdf/)
    const body = await res.body()
    expect(body.subarray(0, 5).toString('latin1')).toBe('%PDF-')
    expect(body.length).toBe(meta[d.id].bytes)
  }

  // source code and model artifacts
  const code = page.getByRole('region', { name: 'Mã nguồn' })
  await expect(code.getByRole('link', { name: /Kho mã trên GitHub/ })).toHaveAttribute('href', REPO_URL)
  await expect(code.getByRole('link', { name: /README/ })).toHaveAttribute('href', `${REPO_URL}/blob/main/README.md`)
  await expect(code.getByRole('link', { name: /Notebook tái lập kết quả/ })).toHaveAttribute(
    'href',
    `${REPO_URL}/blob/main/notebooks/reproduce_results.ipynb`,
  )
  const exp = page.getByRole('region', { name: 'Thí nghiệm và mô hình' })
  await expect(exp.getByRole('link', { name: /Kho mô hình trên Hugging Face/ })).toHaveAttribute('href', /^https:\/\/huggingface\.co\//)
  await expect(exp.getByRole('link', { name: /Thư mục results/ })).toHaveAttribute('href', `${REPO_URL}/tree/main/results`)
  await expect(exp).toContainText('Weights & Biases')
  await expect(exp.getByRole('link', { name: /wandb|W&B/i })).toHaveCount(0)
})

test('the private-repository notice follows REPO_PUBLIC', async ({ page }) => {
  await openDocuments(page)
  const notice = page.getByRole('note').filter({ hasText: 'Kho mã nguồn hiện ở chế độ riêng tư' })
  if (REPO_PUBLIC) await expect(notice).toHaveCount(0)
  else await expect(notice).toBeVisible()
})

test('reloading the Documents route keeps the Documents page', async ({ page }) => {
  await page.goto('/#/documents')
  await expect(page.getByRole('heading', { level: 1, name: 'Báo cáo, slide và mã nguồn' })).toBeVisible()
  await page.reload()
  await expect(page.getByRole('heading', { level: 1, name: 'Báo cáo, slide và mã nguồn' })).toBeVisible()
  await expect(docsNav(page)).toHaveAttribute('aria-current', 'page')
  await expect(page.getByRole('region', { name: 'Báo cáo khoa học' }).locator('[data-pages]')).toBeVisible()
})

test.describe('on a phone', () => {
  test.use({ viewport: { width: 390, height: 844 } })

  test('the GitHub link and the Documents item are reachable, and nothing scrolls sideways', async ({ page }) => {
    await page.goto('/')
    const gh = githubLink(page)
    await expect(gh).toBeInViewport()
    const box = (await gh.boundingBox())!
    expect(box.width).toBeGreaterThanOrEqual(44)
    expect(box.height).toBeGreaterThanOrEqual(44)
    await expect(docsNav(page)).toBeInViewport()
    const noSideScroll = () => page.evaluate<boolean>('document.documentElement.scrollWidth <= document.documentElement.clientWidth')
    expect(await noSideScroll()).toBe(true)
    await docsNav(page).click()
    await expect(page.getByRole('region', { name: 'Slide thuyết trình' }).locator('[data-pages]')).toBeVisible()
    expect(await noSideScroll()).toBe(true)
  })
})

test('no errors or failed requests on the demo and the Documents page, and the demo still answers', async ({ page }) => {
  const problems = watchProblems(page)
  await page.goto('/')
  await page.getByRole('region', { name: 'Đầu vào của bạn' }).waitFor()
  await page.getByRole('button', { name: 'Vẽ số 7' }).click()
  for (const t of ['Softmax regression', 'CNN']) {
    await expect(card(page, t).getByRole('list', { name: /Đầu ra cho từng lớp/ })).not.toContainText('không có', { timeout: 60_000 })
    expect(await predicted(page, t), `${t} reads the 7`).toBe(7)
  }
  await docsNav(page).click()
  await expect(page.getByRole('region', { name: 'Báo cáo khoa học' }).locator('[data-pages]')).toBeVisible()
  await expect(page.getByRole('region', { name: 'Slide thuyết trình' }).locator('[data-pages]')).toBeVisible()
  await page.goto('/#/documents')
  await page.reload()
  await expect(page.getByRole('region', { name: 'Báo cáo khoa học' }).locator('[data-pages]')).toBeVisible()
  expect(problems, 'no runtime errors or failed requests').toEqual([])
})
