import { expect, test } from '@playwright/test'

/**
 * The pinned model lock is compiled into the script, so when a cached script meets a newer manifest the
 * retry on the pin-mismatch screen must reload the page (fetching the new script) instead of re-running the
 * same check in place.
 */
test('a pin mismatch is retried with a full page reload, which recovers once the files match', async ({ page }) => {
  // serve a manifest whose bytes differ from the pinned one (its SHA-256 no longer matches the lock)
  await page.route('**/data/manifest.json', async (route) => {
    const res = await route.fetch()
    await route.fulfill({ response: res, body: (await res.text()) + '\n' })
  })
  await page.goto('/')
  const alert = page.getByRole('alert')
  await expect(alert).toContainText('manifest.json')
  await page.unroute('**/data/manifest.json')

  await page.evaluate(() => ((globalThis as { beforeRetry?: boolean }).beforeRetry = true))
  await Promise.all([page.waitForEvent('load'), alert.getByRole('button', { name: 'Thử lại' }).click()])
  await page.getByRole('region', { name: 'Đầu vào của bạn' }).waitFor()
  // a new document: the marker set before the retry is gone
  expect(await page.evaluate(() => (globalThis as { beforeRetry?: boolean }).beforeRetry)).toBeUndefined()
})
