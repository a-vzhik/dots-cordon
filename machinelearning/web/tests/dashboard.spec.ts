import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'
import { fixture, page as envelope } from './fixtures'

async function mockApi(page: Page, { data, experiment } = fixture()) {
  const paths: string[] = []
  await page.route('**/api/v1/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    paths.push(path)
    const id = path.split('/').at(-1)
    const body =
      path === '/api/v1/experiments'
        ? envelope([experiment])
        : path.endsWith('/overview')
          ? { ...experiment, current_champion: data.lineage.current_champion }
          : path === '/api/v1/attempts'
            ? envelope(data.lineage.attempts.items)
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

test('opens independent pages and follows record links while keeping experiment context', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (err) => errors.push(err.message))
  const paths = await mockApi(page)
  await page.goto('/')
  await expect(page.locator('.overview-grid')).toBeVisible()
  await expect(
    page.locator('.attempt-card, .graph-panel, .checkpoint-table, .evaluation-card'),
  ).toHaveCount(0)
  expect(paths.every((path) => path === '/api/v1/experiments' || path.endsWith('/overview'))).toBe(
    true,
  )
  await page.getByRole('button', { name: 'Pause live updates' }).click()
  await page.getByRole('button', { name: 'Refresh data' }).click()
  await expect(page.getByRole('button', { name: 'Refresh data' })).toBeEnabled()
  expect(paths.every((path) => path === '/api/v1/experiments' || path.endsWith('/overview'))).toBe(
    true,
  )

  await page.getByRole('link', { name: 'Weight lineage', exact: true }).click()
  await expect(page).toHaveURL(/\/lineage\?experiment=/)
  await expect(page.getByRole('link', { name: 'Weight lineage', exact: true })).toHaveAttribute(
    'aria-current',
    'page',
  )
  await expect(page.getByRole('heading', { name: 'Weight lineage', exact: true })).toBeVisible()
  await expect(
    page.locator('.overview-grid, .attempt-card, .checkpoint-table, .evaluation-card'),
  ).toHaveCount(0)
  expect(paths.some((path) => path.includes('/checkpoints/'))).toBe(false)
  await page.getByRole('button', { name: 'Champion', exact: true }).click()
  await expect(page.locator('.checkpoint-inspector')).toContainText('Episode 13,250')
  await expect(page).toHaveURL(/checkpoint=champion/)
  const selected = page.locator('[data-checkpoint-id="champion"]')
  await expect(selected).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('button', { name: 'Zoom in', exact: true }).click()
  await page.getByRole('button', { name: 'Refresh data' }).click()
  await expect(selected).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('.button-group')).toContainText('120%')

  await page.getByRole('link', { name: 'See evaluations' }).click()
  await expect(page).toHaveURL(/\/evaluations\?experiment=experiment&checkpoint=champion$/)
  await expect(page.getByRole('combobox', { name: 'Checkpoint', exact: true })).toHaveValue(
    'champion',
  )
  await expect(page.getByText('No evaluations found for this checkpoint.')).toBeVisible()
  await page.goBack()
  await expect(page.getByRole('heading', { name: 'Weight lineage', exact: true })).toBeVisible()
  await page.goForward()
  await expect(page.getByRole('combobox', { name: 'Checkpoint', exact: true })).toHaveValue(
    'champion',
  )

  await page.getByRole('link', { name: 'Checkpoints', exact: true }).click()
  await expect(page.locator('.checkpoint-table tbody tr')).toHaveCount(5)
  await expect(page.locator('#checkpoint-champion')).toHaveClass('selected-row')
  await expect(page.locator('#checkpoint-root .download-link')).toHaveAttribute(
    'href',
    '/api/v1/checkpoints/root/download',
  )

  await page.getByRole('link', { name: 'Attempts', exact: true }).click()
  await expect(page.locator('.attempt-card')).toHaveCount(2)
  await expect(
    page.locator('.overview-grid, .graph-panel, .checkpoint-table, .evaluation-card'),
  ).toHaveCount(0)
  await page.getByRole('link', { name: 'Evaluations', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Checkpoint', exact: true })).toHaveValue(
    'champion',
  )
  await page.getByRole('combobox', { name: 'Checkpoint', exact: true }).selectOption('')
  await expect(page.locator('.evaluation-card')).toHaveCount(1)
  await expect(page.locator('.evaluation-card .score').first()).toContainText('0.0%')
  await expect(page.locator('.evaluation-card .score').first()).toContainText('partial')
  await expect(page.locator('.suite-table tbody tr').nth(1)).toContainText('—')
  await page.getByText('Evaluation configuration & suite definitions', { exact: true }).click()
  await expect(page.locator('.evaluation-details pre')).toContainText('9223372036854775815')
  expect(paths.some((path) => path.endsWith('/download'))).toBe(false)
  expect(errors).toEqual([])
})

test('filters checkpoint evaluations from either side and preserves every batch', async ({
  page,
}) => {
  const sample = fixture()
  const template = sample.data.evaluations[0]
  sample.data.evaluations = [
    { ...template, id: 'screening', checkpoint_id: 'champion' },
    ...[
      ['challenge', 'champion', 'root'],
      ['extended', 'champion', 'root'],
      ['defense-1', 'tail', 'champion'],
      ['defense-2', 'child', 'champion'],
      ['unrelated', 'child', 'root'],
    ].map(([id, checkpoint_id, opponent_checkpoint_id]) => ({
      ...template,
      id,
      checkpoint_id,
      opponent_checkpoint_id,
      purpose: 'head_to_head' as const,
      opponent_kind: 'checkpoint' as const,
    })),
  ]
  for (const checkpoint of sample.data.checkpoints) {
    checkpoint.evaluations = envelope(
      sample.data.evaluations.filter((batch) => batch.checkpoint_id === checkpoint.id),
    )
  }
  await mockApi(page, sample)
  await page.goto('/evaluations?experiment=experiment&checkpoint=champion')
  const filter = page.getByRole('combobox', { name: 'Checkpoint', exact: true })
  await expect(filter).toHaveValue('champion')
  await expect(page.locator('.evaluation-card')).toHaveCount(5)
  for (const id of ['screening', 'challenge', 'extended', 'defense-1', 'defense-2']) {
    await expect(page.locator(`#evaluation-${id}`)).toBeVisible()
  }
  await expect(page.locator('#evaluation-unrelated')).toHaveCount(0)
  await expect(page.locator('.section-count')).toHaveText('5 batches')
  await page.getByRole('button', { name: 'Refresh data' }).click()
  await expect(page.getByRole('button', { name: 'Refresh data' })).toBeEnabled()
  await expect(filter).toHaveValue('champion')
  await expect(page.locator('.evaluation-card')).toHaveCount(5)

  // The same episode from different attempts must be distinguishable before filtering.
  await expect(filter.locator('option[value="tail"]')).toHaveText(
    'Attempt 01 · Episode 13,500 · tail',
  )
  await expect(filter.locator('option[value="child"]')).toHaveText(
    'Attempt 02 · Episode 13,500 · child',
  )
  const attemptFilter = page.getByRole('combobox', { name: 'Attempt', exact: true })
  await attemptFilter.selectOption('attempt-1')
  await expect(filter).toHaveValue('')
  await expect(filter.locator('option')).toHaveCount(4)
  await expect(filter.locator('option[value="child"]')).toHaveCount(0)
  await filter.selectOption('champion')
  await expect(page.locator('.evaluation-card')).toHaveCount(5)
  // A later attempt's challenge against this champion still belongs in its history.
  await expect(page.locator('#evaluation-defense-2')).toBeVisible()
  await page.reload()
  await expect(attemptFilter).toHaveValue('attempt-1')
  await expect(filter).toHaveValue('champion')
  await expect(page.locator('.evaluation-card')).toHaveCount(5)
  await attemptFilter.selectOption('attempt-2')
  await expect(page).toHaveURL(/attempt=attempt-2/)
  await expect(page).not.toHaveURL(/checkpoint=/)
  await expect(filter.locator('option')).toHaveCount(2)
  await expect(page.locator('.evaluation-card')).toHaveCount(2)
  await attemptFilter.selectOption('imported')
  await expect(filter.locator('option')).toHaveCount(2)
  await expect(filter.locator('option[value="root"]')).toHaveText(
    'Imported · Episode 12,500 · root',
  )
  await expect(page.locator('.evaluation-card')).toHaveCount(3)
  await attemptFilter.selectOption('')
  await expect(page).not.toHaveURL(/attempt=/)
  await expect(filter.locator('option')).toHaveCount(6)
  await expect(page.locator('.evaluation-card')).toHaveCount(6)

  await filter.selectOption('root')
  await expect(page).toHaveURL(/checkpoint=root/)
  await expect(page.locator('.evaluation-card')).toHaveCount(3)
  await page.reload()
  await expect(filter).toHaveValue('root')
  await expect(page.locator('.evaluation-card')).toHaveCount(3)
  await filter.selectOption('before')
  await expect(page.getByText('No evaluations found for this checkpoint.')).toBeVisible()
  await expect(page.locator('.evaluation-card')).toHaveCount(0)
  await filter.selectOption('')
  await expect(page).not.toHaveURL(/checkpoint=/)
  await expect(page.locator('.evaluation-card')).toHaveCount(6)

  await page.goto('/checkpoints?experiment=experiment')
  await page.locator('#checkpoint-champion').getByRole('link', { name: 'View evaluations' }).click()
  await expect(filter).toHaveValue('champion')
  await expect(page.locator('.evaluation-card')).toHaveCount(5)
})

test('retains displayed history on a failed refresh and lets the user retry', async ({ page }) => {
  await mockApi(page)
  await page.goto('/attempts')
  await expect(page.locator('.attempt-card')).toHaveCount(2)
  await page.getByRole('button', { name: 'Pause live updates' }).click()
  await page.route('**/api/v1/attempts?**', (route) =>
    route.fulfill({
      status: 503,
      json: { error: { message: 'The audit database is unavailable' } },
    }),
  )
  await page.getByRole('button', { name: 'Refresh data' }).click()
  await expect(page.getByRole('alert')).toContainText('Showing the last successful read.')
  await expect(page.locator('.attempt-card')).toHaveCount(2)
  await page.unroute('**/api/v1/attempts?**')
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('fits a mobile viewport while the lineage and evidence tables scroll independently', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await mockApi(page)
  await page.goto('/')
  await expect(page.locator('.overview-grid')).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)
  expect(overflow).toBeLessThanOrEqual(1)
  await expect(page.getByRole('heading', { name: 'Overview.' })).toBeVisible()
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
  await page.goto(new URL('/lineage', process.env.AUDIT_LIVE_URL!).href)
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
