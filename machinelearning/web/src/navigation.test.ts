import { describe, expect, it } from 'vitest'
import { currentPage, pageHref, pages } from './navigation'

describe('page URLs', () => {
  it('opens Overview by default and each section directly', () => {
    expect(currentPage('/')).toBe('overview')
    for (const page of pages) expect(currentPage(`/${page}`)).toBe(page)
  })
  it('keeps experiment context and selects the linked checkpoint', () => {
    expect(
      pageHref('checkpoints', '?experiment=run+one&checkpoint=old', {
        kind: 'checkpoint',
        id: 'new',
      }),
    ).toBe('/checkpoints?experiment=run+one&checkpoint=new#checkpoint-new')
    expect(pageHref('evaluations', '?experiment=run', { kind: 'evaluation', id: 'batch' })).toBe(
      '/evaluations?experiment=run#evaluation-batch',
    )
  })
})
