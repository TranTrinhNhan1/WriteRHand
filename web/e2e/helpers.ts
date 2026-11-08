import { expect, type Page } from '@playwright/test'

/** Shared helpers for the end-to-end tests (Vietnamese UI, the default language). */

/** the models of every dataset, in display order */
export const MODEL_TITLES = ['Naive Bayes', 'Softmax regression', 'MLP', 'CNN', 'AdaBoost']
/** MNIST also has the linear SVM (after softmax regression) */
export const MNIST_TITLES = ['Naive Bayes', 'Softmax regression', 'Linear SVM', 'MLP', 'CNN', 'AdaBoost']

export const answers = (page: Page) => page.getByRole('region', { name: 'Câu trả lời' })
const escapeRe = (s: string) => s.replace(/[.*+?^$()|[\]\\{}]/g, '\\$&')

/** A model's answer card: the plain model, or its augmented variant (MNIST shows both side by side). */
export const card = (page: Page, title: string, variant: 'plain' | 'augmented' = 'plain') =>
  answers(page)
    .locator(`.mcard[data-variant="${variant}"]`)
    .filter({ has: page.getByRole('heading', { level: 3, name: new RegExp(`^${escapeRe(title)}( (thường|tăng cường))?$`) }) })

/** the class a model card currently reports (first paragraph of the card after the size) */
export async function predicted(page: Page, title: string, variant: 'plain' | 'augmented' = 'plain') {
  const bars = card(page, title, variant).getByRole('list', { name: /Đầu ra cho từng lớp/ }).getByRole('listitem')
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
export async function loadAdaBoost(page: Page) {
  const load = card(page, 'AdaBoost').getByRole('button', { name: /^Tải/ })
  if (await load.waitFor({ timeout: 10_000 }).then(() => true, () => false)) await load.click()
}

export async function waitAllAnswered(page: Page) {
  await loadAdaBoost(page)
  const titles = (await answers(page).locator('.mpair').count()) ? MNIST_TITLES : MODEL_TITLES
  for (const t of titles) {
    await expect(card(page, t).getByRole('list', { name: /Đầu ra cho từng lớp/ })).not.toContainText('không có', { timeout: 60_000 })
  }
}

/**
 * Collects everything that went wrong while a page was open: uncaught errors, console errors, requests that failed
 * and responses with an HTTP error status. Nothing is filtered out.
 */
export function watchProblems(page: Page): string[] {
  const problems: string[] = []
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`))
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(`console: ${m.text()}`)
  })
  page.on('requestfailed', (r) => problems.push(`requestfailed: ${r.url()} ${r.failure()?.errorText ?? ''}`))
  page.on('response', (r) => {
    if (r.status() >= 400) problems.push(`http ${r.status()}: ${r.url()}`)
  })
  return problems
}
