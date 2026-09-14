import type { components } from './schema'

type Schema = components['schemas']
export type Experiment = Schema['Experiment']
export type Lineage = Schema['Lineage']
export type Checkpoint = Schema['CheckpointSummary']
export type CheckpointNode = Schema['CheckpointNode']
export type CheckpointDetail = Schema['CheckpointDetail']
export type Attempt = Schema['AttemptSummary']
export type AttemptDetail = Schema['AttemptDetail']
export type Evaluation = Schema['EvaluationSummary']
export type EvaluationDetail = Schema['EvaluationDetail']
export type Suite = Schema['EvaluationSuite']
export type Decision = Schema['Decision']
export type Metrics = Schema['Metrics']
export type Stats = Schema['MatchStats']

export interface Page<T> {
  items: T[]
  total: number
  remaining: number
  limit: number
  next_cursor: string | null
  snapshot_at: string
}

export interface DashboardData {
  lineage: Lineage
  attempts: AttemptDetail[]
  checkpoints: CheckpointDetail[]
  evaluations: EvaluationDetail[]
  metrics: Record<string, Metrics>
}
