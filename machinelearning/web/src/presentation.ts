import type { Evaluation, Stats } from './api/types'

export const number = (value: number) => value.toLocaleString('en-GB')
export const shortId = (id: string) => id.slice(0, 8)
export const label = (value: string) => value.replaceAll('_', ' ')
export const percent = (value: number | null | undefined) =>
  value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(1)}%`
export const signed = (value: number | null | undefined) =>
  value == null || !Number.isFinite(value) ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(2)}`
export const bytes = (value: number | null | undefined) =>
  value == null ? 'Unavailable' : `${(value / 1024 / 1024).toFixed(1)} MB`
export const date = (value: string | null | undefined) =>
  value
    ? new Date(value).toLocaleString('en-GB', {
        day: 'numeric',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
      })
    : '—'
export const scoreLabel = (evaluation: Evaluation) =>
  `${percent(evaluation.aggregate?.overall.match_score)}${evaluation.aggregate_is_partial ? ' · partial' : ''}`
export const record = (stats: Stats | null | undefined) =>
  stats ? `${number(stats.wins)} / ${number(stats.draws)} / ${number(stats.losses)}` : '—'
export const metricValue = (metrics: Record<string, unknown>, key: string): number | null => {
  const value = metrics[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}
