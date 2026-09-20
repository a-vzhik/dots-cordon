export const pages = ['overview', 'lineage', 'attempts', 'checkpoints', 'evaluations'] as const
export type DashboardPage = (typeof pages)[number]

export function currentPage(pathname: string): DashboardPage {
  const page = pathname.replace(/^\//, '').replace(/\/$/, '')
  return pages.find((candidate) => candidate === page) ?? 'overview'
}

export function pageHref(
  page: DashboardPage,
  search: string,
  target?: { kind: 'checkpoint' | 'attempt' | 'evaluation'; id: string },
) {
  const params = new URLSearchParams(search)
  if (target?.kind === 'checkpoint') {
    params.set('checkpoint', target.id)
    params.delete('attempt')
  }
  // A direct batch link must remain visible regardless of the current filter.
  if (target?.kind === 'evaluation') {
    params.delete('checkpoint')
    params.delete('attempt')
  }
  const query = params.toString()
  const hash =
    target && !(page === 'evaluations' && target.kind === 'checkpoint')
      ? `#${target.kind}-${encodeURIComponent(target.id)}`
      : ''
  return `/${page}${query ? `?${query}` : ''}${hash}`
}
