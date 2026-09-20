import type { Evaluation, EvaluationDetail, Stats } from './api/types'

export const number = (value: number) => value.toLocaleString('en-GB')
export const shortId = (id: string) => id.slice(0, 8)
export const label = (value: string) => value.replaceAll('_', ' ')
export const percent = (value: number | null | undefined) =>
  value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(1)}%`
export const precisePercent = (value: number) =>
  `${Number((value * 100).toFixed(6)).toLocaleString('en-GB', { maximumFractionDigits: 6 })}%`

export function challengeEvidence(evaluation: EvaluationDetail) {
  // Only explain a gate using its own completed candidate batch, never a
  // decision that merely references this batch as the champion's baseline.
  if (
    evaluation.purpose !== 'head_to_head' ||
    !evaluation.is_complete ||
    evaluation.aggregate_is_partial ||
    !evaluation.aggregate ||
    evaluation.suites.items.length !== evaluation.planned_suite_count ||
    evaluation.suites.items.some((suite) => suite.status !== 'completed' || !suite.result)
  )
    return null
  const decision = evaluation.decisions.items
    .filter(
      (d) =>
        d.stage === 'challenge' &&
        d.checkpoint_id === evaluation.checkpoint_id &&
        d.candidate_evaluation_id === evaluation.id,
    )
    .sort((a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id))
    .at(-1)
  if (!decision) return null
  const minimumScore = metricValue(decision.policy, 'promotion_min_match_score')
  const minimumSuiteWins = metricValue(decision.policy, 'promotion_min_suite_wins')
  if (minimumScore == null || minimumSuiteWins == null) return null
  const suiteWins = evaluation.suites.items.filter(
    (suite) => (suite.result?.overall.match_score ?? 0) > 0.5,
  ).length
  return {
    decision,
    minimumScore,
    minimumSuiteWins,
    suiteWins,
    scoreMet: (evaluation.aggregate.overall.match_score ?? 0) > 0.5,
    suitesMet: suiteWins >= minimumSuiteWins,
  }
}
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
