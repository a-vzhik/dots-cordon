import { afterEach, describe, expect, it, vi } from 'vitest'
import { fixture, page } from '../../tests/fixtures'
import { loadPage, ReadCache, readLineage, readPages, request } from './client'
import { percent, scoreLabel } from '../presentation'

afterEach(() => vi.unstubAllGlobals())
describe('audit reads', () => {
  it('finishes checkpoint pagination before advancing the attempt window and replaces boundary nodes', async () => {
    const { data } = fixture()
    const base = data.lineage
    const full = base.checkpoints.items
    const responses = [
      {
        ...base,
        attempts: page([base.attempts.items[0]], 'a2'),
        checkpoints: page(full.slice(0, 2), 'c2'),
        boundary_checkpoints: [full[2]],
      },
      {
        ...base,
        attempts: page([base.attempts.items[0]], 'a2'),
        checkpoints: page(full.slice(2, 4)),
      },
      {
        ...base,
        attempts: page([base.attempts.items[1]]),
        checkpoints: page(full.slice(4)),
        boundary_checkpoints: [full[2]],
      },
    ]
    const urls: URL[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (path: string) => {
        urls.push(new URL(path, 'http://local'))
        return Response.json(responses.shift())
      }),
    )
    const result = await readLineage('experiment', new AbortController().signal)
    expect(
      urls.map((u) => [
        u.searchParams.get('attempt_cursor'),
        u.searchParams.get('checkpoint_cursor'),
      ]),
    ).toEqual([
      [null, null],
      [null, 'c2'],
      ['a2', null],
    ])
    expect(result.checkpoints.items).toHaveLength(5)
    expect(result.attempts.items).toHaveLength(2)
    expect(result.boundary_checkpoints).toHaveLength(0)
    expect(result.edges).toHaveLength(4)
  })
  it('rejects repeated cursors instead of looping indefinitely', async () => {
    await expect(readPages(async () => page([1], 'again'))).rejects.toThrow('repeated page cursor')
  })
  it('preserves server errors and passes cancellation to fetch', async () => {
    const fetcher = vi.fn(async () =>
      Response.json({ error: { message: 'The audit database is unavailable' } }, { status: 503 }),
    )
    vi.stubGlobal('fetch', fetcher)
    const signal = new AbortController().signal
    await expect(request('/test', signal)).rejects.toThrow('The audit database is unavailable')
    expect(fetcher).toHaveBeenCalledWith('/test', expect.objectContaining({ signal }))
  })
  it('caches completed evidence but reads active or changed records again', async () => {
    const cache = new ReadCache()
    const loader = vi.fn(async () => ({ counter: 1 }))
    await cache.get('batch', 'v1', false, loader)
    await cache.get('batch', 'v1', false, loader)
    expect(loader).toHaveBeenCalledTimes(1)
    await cache.get('batch', 'v1', true, loader)
    await cache.get('batch', 'v2', false, loader)
    expect(loader).toHaveBeenCalledTimes(3)
  })
  it('distinguishes measured zero, absent results, and partial aggregates', () => {
    const { data } = fixture()
    expect(percent(0)).toBe('0.0%')
    expect(percent(null)).toBe('—')
    expect(scoreLabel(data.evaluations[0])).toBe('0.0% · partial')
  })
})

describe('page data isolation', () => {
  function mockReads() {
    const { data, experiment } = fixture()
    const paths: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (path: string) => {
        const url = new URL(path, 'http://local')
        paths.push(url.pathname)
        const id = url.pathname.split('/').at(-1)
        const body =
          id === 'overview'
            ? { ...experiment, current_champion: data.lineage.current_champion }
            : id === 'lineage'
              ? data.lineage
              : url.pathname === '/api/v1/attempts'
                ? page(data.lineage.attempts.items)
                : id === 'metrics'
                  ? data.metrics[url.pathname.split('/').at(-2)!]
                  : url.pathname.includes('/attempts/')
                    ? data.attempts.find((a) => a.id === id)
                    : url.pathname.includes('/checkpoints/')
                      ? data.checkpoints.find((c) => c.id === id)
                      : data.evaluations.find((e) => e.id === id)
        if (!body) throw new Error(`Unexpected request: ${path}`)
        return Response.json(body)
      }),
    )
    return { paths, experiment }
  }

  it('loads and refreshes overview without fetching any history', async () => {
    const { paths, experiment } = mockReads()
    const cache = new ReadCache()
    for (let i = 0; i < 2; i++) {
      const result = await loadPage(experiment.id, 'overview', new AbortController().signal, cache)
      expect(result.page).toBe('overview')
    }
    expect(paths).toEqual(Array(2).fill(`/api/v1/experiments/${experiment.id}/overview`))
  })

  it('loads only the graph for weight lineage', async () => {
    const { paths, experiment } = mockReads()
    await loadPage(experiment.id, 'lineage', new AbortController().signal, new ReadCache())
    expect(paths).toEqual([`/api/v1/experiments/${experiment.id}/lineage`])
  })

  it('loads attempts and their metrics without lineage or evaluation details', async () => {
    const { paths, experiment } = mockReads()
    const result = await loadPage(
      experiment.id,
      'attempts',
      new AbortController().signal,
      new ReadCache(),
    )
    expect(result.page === 'attempts' && result.data.attempts).toHaveLength(2)
    expect(paths.every((path) => path.startsWith('/api/v1/attempts'))).toBe(true)
    expect(paths.filter((path) => path.endsWith('/metrics'))).toHaveLength(2)
  })

  it.each(['checkpoints', 'evaluations'] as const)(
    'loads %s without attempt details or metrics',
    async (page) => {
      const { paths, experiment } = mockReads()
      await loadPage(experiment.id, page, new AbortController().signal, new ReadCache())
      expect(paths.some((path) => path.includes('/attempts/'))).toBe(false)
      expect(paths.some((path) => path.includes('/evaluations/'))).toBe(page === 'evaluations')
      expect(paths.some((path) => path.endsWith('/download'))).toBe(false)
    },
  )
})
