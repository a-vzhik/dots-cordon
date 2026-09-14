import { afterEach, describe, expect, it, vi } from 'vitest'
import { fixture, page } from '../../tests/fixtures'
import { ReadCache, readLineage, readPages, request } from './client'
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
