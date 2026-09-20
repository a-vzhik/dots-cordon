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
  if (target?.kind === 'checkpoint') params.set('checkpoint', target.id)
  const query = params.toString()
  const hash = target ? `#${target.kind}-${encodeURIComponent(target.id)}` : ''
  return `/${page}${query ? `?${query}` : ''}${hash}`
}
