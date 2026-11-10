import { expect, test, type Page } from '@playwright/test'
import { MNIST_TITLES, answers, card, predicted, waitAllAnswered } from './helpers'

/**
 * Browser end-to-end tests against the production build and the pinned model bundle.
 * They exercise what a visitor does: draw, pick a model, look inside, step through the computation,
 * and switch the input. Nothing here is mocked; the models are the deployed ONNX/binary files.
 */

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

test('the page lists the dataset and all six models with their variants, none of them preselected as answered', async ({ page }) => {
  await expect(page.getByRole('radio', { name: /^MNIST/ })).toBeChecked()
  // MNIST: one row per model, the augmented variant on the left and the plain model on the right
  expect(await answers(page).locator('.mpair__name').evaluateAll((els) => els.map((e) => e.firstChild?.textContent?.trim()))).toEqual(MNIST_TITLES)
  for (const t of MNIST_TITLES) await expect(card(page, t)).toHaveCount(1)
  for (const t of MNIST_TITLES) await expect(card(page, t, 'augmented')).toHaveCount(1)
  await expect(page.getByText('Bạn vẽ một chữ số trước nhé')).toBeVisible()
})

test('a sample digit is read by every model, with per-class outputs', async ({ page }) => {
  await page.getByRole('button', { name: 'Vẽ số 7' }).click()
  await waitAllAnswered(page)
  for (const t of MNIST_TITLES) expect(await predicted(page, t), `${t} reads the 7`).toBe(7)
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
    // the difference is printed as `m.me−n`, or as a plain `0` when the two computations agree exactly
    const d = /Chênh lệch tuyệt đối lớn nhất trên \d+ lớp: (?:([\d.]+)e[−-](\d+)|(0)(?![\d.e]))/.exec(t)
    return fp && d ? { fp, diff: d[3] !== undefined ? 0 : Number(d[1]) * 10 ** -Number(d[2]) } : null
  }
  await expect.poll(read, { timeout: 30_000 }).not.toBeNull()
  const first = (await read())!
  expect(first.diff).toBeLessThan(1e-3)

  // a new input: the explorer must show the new execution, never the previous one
  await page.getByRole('button', { name: 'Vẽ số 3' }).click()
  // the example is drawn stroke by stroke and the view follows every intermediate input (while an
  // execution is computed the panel shows no fingerprint at all); wait for a different fingerprint
  // that then stays put, i.e. the drawing is finished
  await expect
    .poll(async () => {
      const a = await read()
      if (a === null || a.fp === first.fp) return false
      await page.waitForTimeout(600)
      const b = await read()
      return b !== null && b.fp === a.fp
    }, { timeout: 30_000 })
    .toBe(true)
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

test('results page shows the five browser models and the two measured-only ones', async ({ page }) => {
  await page.goto('/#/results')
  for (const t of ['Naive Bayes', 'MLP', 'CNN', 'AdaBoost', 'Linear SVM', 'Stacking']) await expect(page.getByText(t).first()).toBeVisible()
  const table = page.getByRole('region', { name: 'Các chỉ số trên tập Test' })
  // the linear SVM runs in the browser now: no tag; the stacking ensemble still does not
  await expect(table.getByRole('row', { name: /^Linear SVM 91[.,]95%/ })).toBeVisible()
  await expect(table.getByRole('row', { name: /Stacking.*không chạy trên trình duyệt.*99[.,]05%/ })).toBeVisible()
})

test('the MLP network is drawn in 3D from the trace, and the 2D drawing is one click away', async ({ page }) => {
  await page.getByRole('button', { name: 'Vẽ số 4' }).click()
  await waitAllAnswered(page)
  await card(page, 'MLP').getByRole('button', { name: 'Xem bên trong' }).click()
  const panel = page.getByRole('tabpanel')
  const stage = panel.getByRole('application', { name: /Các lớp theo thứ tự tính: x · 784 → h1 · \d+/ })
  await expect(stage).toBeVisible({ timeout: 30_000 })
  await expect(stage.locator('canvas')).toHaveCount(1)
  await expect(stage).toHaveAttribute('aria-busy', 'false')
  // the connection counts come from the trace (784 × hidden width), not from the drawing
  await expect(panel.getByText(/^x→h1: \d+\/[\d\s.,]+ kết nối$/)).toBeVisible()
  // keyboard: rotate and reset without errors
  await stage.focus()
  await page.keyboard.press('ArrowLeft')
  await page.keyboard.press('0')
  await panel.getByRole('button', { name: 'Góc nhìn ban đầu' }).click()
  // 2D form: the canvas graph of the same layers; back to 3D
  await panel.getByRole('button', { name: '2D', exact: true }).click()
  await expect(stage).toHaveCount(0)
  await expect(panel.locator('.ex-graph canvas')).toHaveCount(1)
  await panel.getByRole('button', { name: '3D', exact: true }).click()
  await expect(panel.getByRole('application', { name: /Các lớp theo thứ tự tính/ }).locator('canvas')).toHaveCount(1)
})

test('the CNN feature maps are drawn as 3D stacks with the shapes of the trace', async ({ page }) => {
  await page.getByRole('button', { name: 'Vẽ số 9' }).click()
  await waitAllAnswered(page)
  const view = card(page, 'CNN').getByRole('button', { name: 'Xem bên trong' })
  if (await view.isVisible()) await view.click()
  const panel = page.getByRole('tabpanel')
  const stage = panel.getByRole('application', { name: /conv0\+\w+ · 16×26×26 → pool0 · 16×13×13/ })
  await expect(stage).toBeVisible({ timeout: 30_000 })
  await expect(stage.locator('canvas')).toHaveCount(1)
})

test('without WebGL the network views fall back to their 2D drawing', async ({ page }) => {
  await page.addInitScript(() => {
    type Ctx = (this: unknown, type: string, ...rest: unknown[]) => unknown
    const proto = (globalThis as unknown as { HTMLCanvasElement: { prototype: { getContext: Ctx } } }).HTMLCanvasElement.prototype
    const orig = proto.getContext
    proto.getContext = function (type, ...rest) {
      if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') return null
      return orig.call(this, type, ...rest)
    }
  })
  await page.goto('/')
  await page.getByRole('button', { name: 'Vẽ số 4' }).click()
  await waitAllAnswered(page)
  await card(page, 'MLP').getByRole('button', { name: 'Xem bên trong' }).click()
  const panel = page.getByRole('tabpanel')
  await expect(panel.locator('.ex-graph canvas')).toHaveCount(1, { timeout: 30_000 })
  await expect(panel.getByRole('application', { name: /Các lớp theo thứ tự tính/ })).toHaveCount(0)
  await expect(panel.getByRole('button', { name: '3D', exact: true })).toHaveCount(0)
})

test('MNIST pairs each model with its augmented variant; variants load on request and both answer', async ({ page }) => {
  const row = answers(page).locator('.mpair').filter({ hasText: 'MLP' }).first()
  // augmented on the left, plain on the right
  await expect(row.locator('.mcard').nth(0)).toHaveAttribute('data-variant', 'augmented')
  await expect(row.locator('.mcard').nth(1)).toHaveAttribute('data-variant', 'plain')
  await page.getByRole('button', { name: 'Vẽ số 7' }).click()
  await waitAllAnswered(page)
  // nothing augmented is downloaded until asked
  await expect(card(page, 'MLP', 'augmented').getByRole('button', { name: /^Tải/ })).toBeVisible()
  await page.getByRole('button', { name: /^Tải 6 bản tăng cường · / }).click()
  for (const t of MNIST_TITLES) {
    await expect(card(page, t, 'augmented').getByRole('list', { name: /Đầu ra cho từng lớp/ })).not.toContainText('không có', { timeout: 60_000 })
  }
  expect(await predicted(page, 'CNN', 'augmented')).toBe(7)
  expect(await predicted(page, 'MLP', 'augmented')).toBe(7)
})

test('the explorer switches between the plain model and its augmented variant with the same view', async ({ page }) => {
  await page.getByRole('button', { name: 'Vẽ số 4' }).click()
  await waitAllAnswered(page)
  await card(page, 'MLP').getByRole('button', { name: 'Xem bên trong' }).click()
  const variant = page.getByRole('group', { name: 'Biến thể' })
  await expect(variant.getByRole('button', { name: 'thường', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByRole('tab', { name: /Xem lại quá trình huấn luyện/ })).toBeVisible()
  await variant.getByRole('button', { name: 'tăng cường', exact: true }).click()
  await expect(page.getByRole('region', { name: 'MLP · tăng cường' })).toBeVisible({ timeout: 30_000 })
  // no recorded training run for the variant: no replay tab
  await expect(page.getByRole('tab', { name: /Xem lại quá trình huấn luyện/ })).toHaveCount(0)
  const panel = page.getByRole('tabpanel')
  await expect(panel.getByRole('application', { name: /x · 784 → h1 · \d+/ }).locator('canvas')).toHaveCount(1, { timeout: 30_000 })
  await expect(card(page, 'MLP', 'augmented').getByRole('button', { name: 'Đang xem' })).toBeVisible()
  await variant.getByRole('button', { name: 'thường', exact: true }).click()
  await expect(page.getByRole('tab', { name: /Xem lại quá trình huấn luyện/ })).toBeVisible({ timeout: 30_000 })
})

test('results page compares every plain model with its augmented variant, numbers from the results file', async ({ page }) => {
  await page.goto('/#/results')
  const table = page.getByRole('region', { name: 'Accuracy của bản thường và bản tăng cường' })
  await expect(table.getByRole('row', { name: /Naive Bayes 81[.,]32% 70[.,]64% −10[.,]68/ })).toBeVisible()
  await expect(table.getByRole('row', { name: /CNN 98[.,]80% 99[.,]21% \+0[.,]41/ })).toBeVisible()
  await expect(page.getByText(/xoay ±30°, dịch ±2 px, co giãn 0[.,]9–1[.,]1/)).toBeVisible()
})

test('results page shows the 5-fold cross-validation of the six models after the plain / augmented comparison', async ({ page }) => {
  await page.goto('/#/results')
  const heading = page.getByRole('heading', { level: 2, name: 'Kiểm tra chéo 5 phần' })
  await expect(heading).toBeVisible()
  // placed right after the plain-vs-augmented section
  const order = await page.locator('.rsection h2').allInnerTexts()
  expect(order.indexOf('Kiểm tra chéo 5 phần')).toBe(order.indexOf('Thường và tăng cường') + 1)
  const table = page.getByRole('region', { name: 'Accuracy qua 5 fold' })
  await expect(table.locator('tbody tr')).toHaveCount(6)
  await expect(table.getByRole('row', { name: /^CNN 98[.,]66% ± 0[.,]12 98[.,]72% ± 0[.,]09 98[.,]45–98[.,]76%$/ })).toBeVisible()
  const plot = page.getByRole('img', { name: /Accuracy trên fold giữ lại của từng fold trong 5 fold/ })
  await expect(plot).toBeVisible()
  for (const m of ['naive_bayes', 'softmax', 'svm', 'mlp', 'cnn', 'adaboost']) {
    await expect(plot.locator(`.kfdots__row[data-model="${m}"] circle.kfdots__dot`)).toHaveCount(5)
    await expect(plot.locator(`.kfdots__row[data-model="${m}"] .kfdots__mean`)).toHaveCount(1)
  }
})

test('at 1280 × 800 the answer of every model and variant (12 cells) is on screen together with the majority', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 })
  await page.getByRole('button', { name: 'Vẽ số 7' }).click()
  await waitAllAnswered(page)
  await page.getByRole('button', { name: /^Tải 6 bản tăng cường · / }).click()
  await expect(card(page, 'AdaBoost', 'augmented').getByRole('list', { name: /Đầu ra cho từng lớp/ })).not.toContainText('không có', { timeout: 60_000 })
  // scroll to the two panels, as a visitor does after the introduction
  await page.locator('.deck').evaluate((e) => {
    const w = globalThis as unknown as { scrollTo(x: number, y: number): void; scrollY: number }
    w.scrollTo(0, e.getBoundingClientRect().top + w.scrollY - 70)
  })
  const cells = answers(page).locator('.mcard')
  await expect(cells).toHaveCount(12)
  const boxes = await cells.evaluateAll((els) => els.map((e) => e.getBoundingClientRect()).map((r) => ({ top: r.top, bottom: r.bottom })))
  for (const b of boxes) {
    expect(b.top).toBeGreaterThanOrEqual(0)
    expect(b.bottom).toBeLessThanOrEqual(800)
  }
  const verdict = await answers(page).locator('.verdict__digit').boundingBox()
  expect(verdict!.y).toBeGreaterThanOrEqual(0)
  // the readout stays in view while the input column scrolls
  await page.mouse.wheel(0, 500)
  await expect.poll(async () => (await cells.last().boundingBox())!.y + (await cells.last().boundingBox())!.height).toBeLessThanOrEqual(800)
})

test('the linear SVM answers with normalised scores, never called probabilities, and opens in the explorer', async ({ page }) => {
  await page.getByRole('button', { name: 'Vẽ số 3' }).click()
  await waitAllAnswered(page)
  await expect(card(page, 'Linear SVM').getByRole('list', { name: 'Đầu ra cho từng lớp (điểm số chuẩn hóa · không phải xác suất)' })).toBeVisible()
  await expect(answers(page).locator('.mpair').filter({ hasText: 'Linear SVM' })).toContainText('điểm số chuẩn hóa · không phải xác suất')
  await card(page, 'Linear SVM').getByRole('button', { name: 'Xem bên trong' }).click()
  await expect(page.getByRole('region', { name: 'Linear SVM' })).toBeVisible({ timeout: 30_000 })
  const panel = page.getByRole('tabpanel')
  // the softmax-regression view, with scores instead of logits, checked against the deployed graph
  await expect(panel).toContainText(/Điểm số/, { timeout: 30_000 })
  await expect(panel).toContainText(/Chênh lệch tuyệt đối lớn nhất trên \d+ lớp/)
  await page.getByRole('group', { name: 'Biến thể' }).getByRole('button', { name: 'tăng cường', exact: true }).click()
  await expect(page.getByRole('region', { name: 'Linear SVM · tăng cường' })).toBeVisible({ timeout: 30_000 })
})

test('on the Models page the algorithm equations are collapsed by default and expand and collapse on click', async ({ page }) => {
  await page.goto('/#/models')
  const boxes = page.locator('details.equations-box')
  await expect(boxes.first()).toBeVisible()
  expect(await boxes.count()).toBeGreaterThanOrEqual(6)
  expect(await boxes.evaluateAll((els) => els.filter((e) => (e as unknown as { open: boolean }).open).length)).toBe(0)
  const first = boxes.first()
  await expect(first.locator('.katex').first()).toBeHidden()
  await first.locator('summary').click()
  await expect(first).toHaveJSProperty('open', true)
  await expect(first.locator('.katex').first()).toBeVisible()
  await first.locator('summary').press('Enter')
  await expect(first).toHaveJSProperty('open', false)
  await expect(first.locator('.katex').first()).toBeHidden()
})
