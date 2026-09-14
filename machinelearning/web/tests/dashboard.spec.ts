import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'
import { fixture, page as envelope } from './fixtures'

async function mockApi(page: Page) {
  const { data, experiment } = fixture()
  const paths: string[] = []
  await page.route('**/api/v1/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    paths.push(path)
    const id = path.split('/').at(-1)
    const body =
      path === '/api/v1/experiments'
        ? envelope([experiment])
        : path.endsWith('/lineage')
          ? data.lineage
          : path.endsWith('/metrics')
            ? data.metrics[path.split('/').at(-2)!]
            : path.includes('/attempts/')
              ? data.attempts.find((a) => a.id === id)
              : path.includes('/checkpoints/')
                ? data.checkpoints.find((c) => c.id === id)
                : data.evaluations.find((e) => e.id === id)
    await route.fulfill({ json: body })
  })
  return paths
}

test('renders all sections, real parent branches, partial scores, and explicit weight downloads', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (err) => errors.push(err.message))
  const paths = await mockApi(page)
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Weight lineage', exact: true })).toBeVisible()
  await expect(page.locator('.attempt-card')).toHaveCount(2)
  await expect(page.locator('.checkpoint-table tbody tr')).toHaveCount(5)
  await expect(page.locator('.evaluation-card')).toHaveCount(1)
  await expect(page.locator('.evaluation-card .score').first()).toContainText('0.0%')
  await expect(page.locator('.evaluation-card .score').first()).toContainText('partial')
  await expect(page.locator('.suite-table tbody tr').nth(1)).toContainText('—')
  await page.getByRole('button', { name: 'Champion', exact: true }).click()
  await expect(page.locator('.checkpoint-inspector')).toContainText('Episode 13,250')
  await expect(page).toHaveURL(/checkpoint=champion/)
  const selected = page.locator('[data-checkpoint-id="champion"]')
  await expect(selected).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('button', { name: 'Zoom in', exact: true }).click()
  await expect(page.locator('.button-group')).toContainText('120%')
  await page.getByRole('button', { name: 'Refresh data' }).click()
  await expect(selected).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('.button-group')).toContainText('120%')
  expect(paths.some((p) => p.endsWith('/download'))).toBe(false)
  await expect(page.locator('#checkpoint-root .download-link')).toHaveAttribute(
    'href',
    '/api/v1/checkpoints/root/download',
  )
  await page.getByText('Evaluation configuration & suite definitions', { exact: true }).click()
  await expect(page.locator('.evaluation-details pre')).toContainText('9223372036854775815')
  expect(errors).toEqual([])
  await page.screenshot({ path: 'test-results/dashboard-desktop.png', fullPage: true })
})

test('retains displayed history on a failed refresh and lets the user retry', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')
  await expect(page.locator('.attempt-card')).toHaveCount(2)
  await page.getByRole('button', { name: 'Pause live updates' }).click()
  await page.route('**/lineage?**', (route) =>
    route.fulfill({
      status: 503,
      json: { error: { message: 'The audit database is unavailable' } },
    }),
  )
  await page.getByRole('button', { name: 'Refresh data' }).click()
  await expect(page.getByRole('alert')).toContainText('Showing the last successful read.')
  await expect(page.locator('.attempt-card')).toHaveCount(2)
  await page.unroute('**/lineage?**')
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('fits a mobile viewport while the lineage and evidence tables scroll independently', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await mockApi(page)
  await page.goto('/')
  await expect(page.locator('.attempt-card')).toHaveCount(2)
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)
  expect(overflow).toBeLessThanOrEqual(1)
  await expect(page.getByRole('heading', { name: 'Training workspace.' })).toBeVisible()
  await page.screenshot({ path: 'test-results/dashboard-mobile.png', fullPage: true })
  await page.screenshot({ path: 'test-results/dashboard-mobile-top.png' })
})

test('handles empty and unavailable databases with actionable states', async ({ page }) => {
  await page.route('**/api/v1/experiments?**', (route) => route.fulfill({ json: envelope([]) }))
  await page.goto('/')
  await expect(page.getByText('No experiments yet.', { exact: false })).toBeVisible()
  await page.unroute('**/api/v1/experiments?**')
  await page.route('**/api/v1/experiments?**', (route) =>
    route.fulfill({
      status: 503,
      json: {
        error: {
          message: 'Apply audit migrations before using the API: dots-cordon-audit db upgrade',
        },
      },
    }),
  )
  await page.getByRole('button', { name: 'Refresh data' }).click()
  await expect(page.getByRole('alert')).toContainText('dots-cordon-audit db upgrade')
  await expect(page.getByRole('button', { name: 'Retry', exact: true })).toBeVisible()
})

test('reads the live audit database without requesting checkpoint bytes', async ({ page }) => {
  test.skip(!process.env.AUDIT_LIVE_URL, 'Set AUDIT_LIVE_URL to smoke-test a running audit server.')
  const errors: string[] = []
  const downloads: string[] = []
  page.on('pageerror', (err) => errors.push(err.message))
  page.on('request', (request) => {
    if (request.url().endsWith('/download')) downloads.push(request.url())
  })
  await page.goto(process.env.AUDIT_LIVE_URL!)
  await expect(page.getByRole('heading', { name: 'Weight lineage', exact: true })).toBeVisible({
    timeout: 30_000,
  })
  await expect(page.locator('.checkpoint-node').first()).toBeVisible()
  await page.getByRole('button', { name: 'Pause live updates' }).click()
  await page.screenshot({ path: 'test-results/dashboard-live.png', fullPage: true })
  await page.screenshot({ path: 'test-results/dashboard-live-top.png' })
  expect(errors).toEqual([])
  expect(downloads).toEqual([])
})
