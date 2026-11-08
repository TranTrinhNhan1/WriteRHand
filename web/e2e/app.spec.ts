import { expect, test, type Page } from '@playwright/test'

/**
 * Browser end-to-end tests against the production build and the pinned model bundle.
 * They exercise what a visitor does: draw, pick a model, look inside, step through the computation,
 * and switch the input. Nothing here is mocked; the models are the deployed ONNX/binary files.
 */

const MODEL_TITLES = ['Naive Bayes', 'Softmax regression', 'MLP', 'CNN', 'AdaBoost']

test.beforeEach(async ({ page }) => {
  const problems: string[] = []
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`))
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(`console: ${m.text()}`)
  })
  ;(page as Page & { problems: string[] }).problems = problems
  await page.goto('/')
  await page.getByRole('region', { name: 'Đầu vào của bạn' }).waitFor()
})
test.afterEach(async ({ page }) => {
  expect((page as Page & { problems: string[] }).problems, 'no runtime errors').toEqual([])
})

const answers = (page: Page) => page.getByRole('region', { name: 'Câu trả lời' })
const card = (page: Page, title: string) =>
  answers(page).getByRole('listitem').filter({ has: page.getByRole('heading', { level: 3, name: title, exact: true }) })

/** the class a model card currently reports (first paragraph of the card after the size) */
async function predicted(page: Page, title: string) {
  const bars = card(page, title).getByRole('list', { name: /Đầu ra cho từng lớp/ }).getByRole('listitem')
  const texts = await bars.allInnerTexts()
  let best = -1
  let bestVal = -1
  texts.forEach((t, i) => {
    const m = /(\d+):\s*([\d.,]+)%/.exec(t)
    if (m && Number(m[2].replace(',', '.')) > bestVal) {
      bestVal = Number(m[2].replace(',', '.'))
      best = Number(m[1])
    }
    void i
  })
  return best
}

/** AdaBoost's 9 MB ensemble is loaded on request (not on every drawing): click its load button. */
async function loadAdaBoost(page: Page) {
  const load = card(page, 'AdaBoost').getByRole('button', { name: /^Tải/ })
  if (await load.waitFor({ timeout: 10_000 }).then(() => true, () => false)) await load.click()
}

async function waitAllAnswered(page: Page) {
  await loadAdaBoost(page)
  for (const t of MODEL_TITLES) {
    await expect(card(page, t).getByRole('list', { name: /Đầu ra cho từng lớp/ })).not.toContainText('không có', { timeout: 60_000 })
  }
}

test('the page lists the dataset and all five models, none of them preselected as answered', async ({ page }) => {
  await expect(page.getByRole('radio', { name: /^MNIST/ })).toBeChecked()
  const headings = await answers(page).getByRole('heading', { level: 3 }).allInnerTexts()
  expect(headings).toEqual(MODEL_TITLES)
  await expect(page.getByText('Bạn vẽ một chữ số trước nhé')).toBeVisible()
})

test('a sample digit is read by every model, with per-class outputs', async ({ page }) => {
  await page.getByRole('button', { name: 'Vẽ số 7' }).click()
  await waitAllAnswered(page)
  for (const t of MODEL_TITLES) expect(await predicted(page, t), `${t} reads the 7`).toBe(7)
})

test('drawing with the mouse produces answers, and clearing removes them', async ({ page }) => {
  const pad = page.getByRole('img', { name: /Vùng vẽ/ })
  await pad.scrollIntoViewIfNeeded()
  const box = (await pad.boundingBox())!
  expect(box.y).toBeGreaterThanOrEqual(0)
  expect(box.y + box.height).toBeLessThanOrEqual(page.viewportSize()!.height + 1)
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.18)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5, { steps: 12 })
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.84, { steps: 12 })
  await page.mouse.up()
  await waitAllAnswered(page)
  expect(await predicted(page, 'Softmax regression')).toBe(1)
  await page.getByRole('button', { name: 'Xoá hết' }).click()
  await expect(page.getByText('Bạn vẽ một chữ số trước nhé')).toBeVisible()
})

test('looking inside Softmax: the recomputed logits match the graph, and the view follows a new input', async ({ page }) => {
  await page.getByRole('button', { name: 'Vẽ số 7' }).click()
  await waitAllAnswered(page)
  await card(page, 'Softmax regression').getByRole('button', { name: 'Xem bên trong' }).click()
  const panel = page.getByRole('tabpanel')
  // the page states which input the displayed execution belongs to (fingerprint) and the largest
  // difference between values recomputed here from the learned weights and the deployed graph
  const read = async () => {
    const t = (await panel.innerText()).replace(/\s+/g, ' ')
    const fp = /đầu vào ([0-9a-f]{8})/.exec(t)?.[1]
    const d = /Chênh lệch tuyệt đối lớn nhất trên \d+ lớp: ([\d.]+)e[−-](\d+)/.exec(t)
    return fp && d ? { fp, diff: Number(d[1]) * 10 ** -Number(d[2]) } : null
  }
  await expect.poll(read, { timeout: 30_000 }).not.toBeNull()
  const first = (await read())!
  expect(first.diff).toBeLessThan(1e-3)

  // a new input: the explorer must show the new execution, never the previous one
  await page.getByRole('button', { name: 'Vẽ số 3' }).click()
  await expect.poll(async () => (await read())?.fp, { timeout: 30_000 }).not.toBe(first.fp)
  expect((await read())!.diff).toBeLessThan(1e-3)
})

test('stepping through the MLP: playback buttons and scrubber change the displayed step', async ({ page }) => {
  await page.getByRole('button', { name: 'Vẽ số 4' }).click()
  await waitAllAnswered(page)
  await card(page, 'MLP').getByRole('button', { name: 'Xem bên trong' }).click()
  const count = page.locator('.ex-playback__count').first()
  await expect(count).toBeVisible({ timeout: 30_000 })
  const read = async () => {
    const m = /(\d+)\/(\d+)/.exec(await count.innerText())
    return { step: Number(m![1]), steps: Number(m![2]) }
  }
  const start = await read()
  expect(start.steps).toBeGreaterThan(1)
  const scrub = page.locator('.ex-playback__scrub').first()
  await scrub.fill('0')
  await expect.poll(async () => (await read()).step).toBe(0)
  await page.locator('.ex-pbtn').nth(2).click() // step forward
  await expect.poll(async () => (await read()).step).toBe(1)
  await page.locator('.ex-pbtn').nth(2).click()
  await expect.poll(async () => (await read()).step).toBe(2)
  await page.locator('.ex-pbtn').nth(0).click() // step back
  await expect.poll(async () => (await read()).step).toBe(1)
})

test('switching the explorer model replaces the view and shows that model\'s own computation', async ({ page }) => {
  await page.getByRole('button', { name: 'Vẽ số 2' }).click()
  await waitAllAnswered(page)
  await card(page, 'Naive Bayes').getByRole('button', { name: 'Xem bên trong' }).click()
  await expect(page.getByRole('tabpanel')).toContainText(/log/i, { timeout: 30_000 })
  await card(page, 'AdaBoost').getByRole('button', { name: 'Xem bên trong' }).click()
  await expect(page.getByRole('tabpanel')).toContainText(/phiếu/i, { timeout: 60_000 })
  await expect(page.getByRole('tabpanel')).toContainText(/không phải xác suất/i)
})

test('looking inside the CNN shows its convolution stages taken from the exported graph', async ({ page }) => {
  await page.getByRole('button', { name: 'Vẽ số 9' }).click()
  await waitAllAnswered(page)
  // the CNN (best model) is the default one being viewed; make the choice explicit anyway
  const view = card(page, 'CNN').getByRole('button', { name: 'Xem bên trong' })
  if (await view.isVisible()) await view.click()
  const panel = page.getByRole('tabpanel')
  // the architecture diagram is generated from the trace: layer ids and shapes come from the model file
  await expect(panel.getByRole('button', { name: /conv0, .*16×26×26/ })).toBeVisible({ timeout: 30_000 })
  await expect(panel.getByRole('button', { name: /conv1, .*32×11×11/ })).toBeVisible()
})

test('switching to Fashion-MNIST changes the classes, the models and the answers; datasets are never mixed', async ({ page }) => {
  await page.getByRole('radio', { name: /Fashion-MNIST/ }).check({ force: true })
  await expect(page.getByRole('list', { name: /Các lớp của Fashion-MNIST/ })).toContainText('Ankle boot')
  // a real test image of this dataset (label: Trouser)
  await page.getByRole('button', { name: 'Ảnh test 137, nhãn: Trouser' }).click()
  await waitAllAnswered(page)
  await expect(answers(page)).toContainText('Học trên Fashion-MNIST')
  for (const t of ['MLP', 'CNN']) await expect(card(page, t)).toContainText('Trouser')
  // back to MNIST: the class list is the digits again and no clothing label remains
  await page.getByRole('radio', { name: /^MNIST/ }).check({ force: true })
  await expect(page.getByRole('list', { name: /Các lớp của MNIST/ })).toBeVisible()
  await expect(page.getByText('Ankle boot')).toHaveCount(0)
})

test('results page shows the five models in the measured table', async ({ page }) => {
  await page.goto('/#/results')
  for (const t of ['Naive Bayes', 'MLP', 'CNN', 'AdaBoost']) await expect(page.getByText(t).first()).toBeVisible()
})
