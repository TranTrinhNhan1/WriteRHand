import { expect, test } from '@playwright/test'
import { watchProblems } from './helpers'

/** The About page introduces the three datasets with their real images (web/public/data/datasets.json). */

const DATASETS = [
  { title: 'MNIST', first: '0', last: '9' },
  { title: 'Fashion-MNIST', first: 'T-shirt/top', last: 'Ankle boot' },
  { title: 'EMNIST Digits', first: '0', last: '9' },
]

test('About introduces each dataset: facts, ten class figures with a mean image and 8 samples, and the class balance', async ({ page }) => {
  const problems = watchProblems(page)
  await page.goto('/#/about')
  const section = page.getByRole('region', { name: 'Giới thiệu bộ dữ liệu' })
  await expect(section).toBeVisible()
  await expect(section.getByRole('radio', { name: 'MNIST', exact: true })).toBeChecked()

  for (const ds of DATASETS) {
    await section.getByRole('radio', { name: ds.title, exact: true }).check({ force: true })
    await expect(section.getByRole('heading', { level: 3, name: ds.title, exact: true })).toBeVisible()
    await expect(section.getByRole('term').filter({ hasText: 'Pixel khác 0' })).toBeVisible()
    const gallery = section.getByRole('region', { name: 'Mười lớp: ảnh trung bình và ảnh mẫu' })
    const figures = gallery.getByRole('figure')
    await expect(figures).toHaveCount(10)
    await expect(figures.first()).toContainText(ds.first)
    await expect(figures.last()).toContainText(ds.last)
    await expect(gallery.getByRole('img')).toHaveCount(90)
    await expect(gallery.getByRole('img', { name: `${ds.title}, lớp ${ds.last}, ảnh trung bình` })).toHaveCount(1)
    await expect(gallery.getByRole('img', { name: `${ds.title}, lớp ${ds.first}, ảnh mẫu 8` })).toHaveCount(1)
    // the images are actually drawn: the mean image has both paper and ink pixels
    const spread = await gallery
      .getByRole('img', { name: `${ds.title}, lớp ${ds.first}, ảnh trung bình` })
      .evaluate((c) => {
        type Canvas = { getContext(kind: '2d'): { getImageData(x: number, y: number, w: number, h: number): { data: Uint8ClampedArray } } }
        const d = (c as unknown as Canvas).getContext('2d').getImageData(0, 0, 28, 28).data
        let lo = 255
        let hi = 0
        for (let i = 0; i < d.length; i += 4) {
          lo = Math.min(lo, d[i])
          hi = Math.max(hi, d[i])
        }
        return hi - lo
      })
    expect(spread).toBeGreaterThan(120)
    await expect(section.getByRole('img', { name: `Số ảnh huấn luyện của từng lớp trong ${ds.title}, trên trục bắt đầu từ 0` })).toBeVisible()
  }

  // no horizontal page scroll at phone width
  await page.setViewportSize({ width: 390, height: 844 })
  await expect.poll(() => page.evaluate(() => (globalThis as unknown as { document: { documentElement: { scrollWidth: number } } }).document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  expect(problems).toEqual([])
})
