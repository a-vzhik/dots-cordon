import type {
  AttemptDetail,
  CheckpointDetail,
  DashboardData,
  EvaluationDetail,
  Experiment,
  Metrics,
  Page,
  Stats,
} from '../src/api/types'

const time = '2026-09-14T12:00:00Z'
export const page = <T>(items: T[], next_cursor: string | null = null): Page<T> => ({
  items,
  next_cursor,
  total: items.length,
  remaining: 0,
  limit: 200,
  snapshot_at: time,
})
const diagnostics = {
  last_activity_at: time,
  heartbeat_age_seconds: 0,
  unresponsive: false,
  unresponsive_after_seconds: 300,
}
const stats = (wins = 0): Stats => ({
  games: 2,
  wins,
  draws: 0,
  losses: 2 - wins,
  score_difference_sum: wins - (2 - wins),
  match_score_numerator: wins * 2,
  match_score_denominator: 4,
  match_score: wins / 2,
  mean_score_difference: (wins - (2 - wins)) / 2,
})
export function checkpoint(
  id: string,
  episode: number,
  attempt_id: string | null = null,
  parent_checkpoint_id: string | null = null,
): CheckpointDetail {
  return {
    id,
    episode,
    attempt_id,
    parent_checkpoint_id,
    experiment_id: 'experiment',
    checkpoint_blob_id: 'blob-' + id,
    environment_steps: episode * 4,
    optimization_steps: episode,
    board: { rows: 7, columns: 7 },
    model: { channels: 64, blocks: 3, kind: 'dqn' },
    origin: attempt_id ? 'training' : 'import',
    save_sequence: episode,
    candidate_index: null,
    is_periodic_save: false,
    is_screening_candidate: !!attempt_id,
    is_best_in_attempt: id === 'champion',
    is_final_in_attempt: id === 'tail',
    created_at: time,
    blob: {
      available: true,
      sha256: 'a'.repeat(64),
      byte_length: 3658549,
      format: 'pytorch',
      format_version: 1,
    },
    counts: {
      child_checkpoints: 0,
      attempts_from_checkpoint: 0,
      evaluation_batches: 0,
      decisions: 0,
    },
    recent_evaluations: [],
    recent_decisions: [],
    evaluations: page([]),
  }
}
export function attempt(id: string, start: number, target: number, parent: string): AttemptDetail {
  return {
    id,
    experiment_id: 'experiment',
    starting_checkpoint_id: parent,
    champion_at_start_assignment_id: 'assignment',
    start_episode: start,
    target_episode: target,
    latest_episode: target,
    phase: 'finished',
    status: 'completed',
    outcome: 'promoted',
    error: null,
    stop_reason: null,
    created_at: time,
    updated_at: time,
    started_at: time,
    ended_at: time,
    heartbeat_at: time,
    last_interrupted_at: null,
    last_resumed_at: null,
    error_at: null,
    diagnostics,
    counts: { checkpoints: 0, qualified_checkpoints: 1, challenge_batches: 1, promotions: 1 },
    config: { seed: '9223372036854775811', batch_size: 64 },
    checkpoints: page([]),
  }
}
export function fixture(): { data: DashboardData; experiment: Experiment } {
  const root = checkpoint('root', 12500)
  const before = checkpoint('before', 12750, 'attempt-1', 'root')
  const champion = checkpoint('champion', 13250, 'attempt-1', 'before')
  const tail = checkpoint('tail', 13500, 'attempt-1', 'champion')
  const child = checkpoint('child', 13500, 'attempt-2', 'champion')
  const first = attempt('attempt-1', 12500, 13500, root.id)
  const second = {
    ...attempt('attempt-2', 13250, 14250, champion.id),
    latest_episode: 13600,
    status: 'running' as const,
    phase: 'training' as const,
    outcome: null,
    ended_at: null,
    counts: { checkpoints: 1, qualified_checkpoints: 0, challenge_batches: 0, promotions: 0 },
  }
  first.checkpoints = page([before, champion, tail])
  first.counts.checkpoints = 3
  second.checkpoints = page([child])
  const batch: EvaluationDetail = {
    id: 'evaluation-1',
    experiment_id: 'experiment',
    attempt_id: second.id,
    checkpoint_id: child.id,
    opponent_checkpoint_id: null,
    opponent_kind: 'random',
    purpose: 'screening',
    status: 'running',
    planned_suite_count: 2,
    recorded_suite_count: 2,
    completed_suite_count: 1,
    expected_games: 8,
    completed_games: 4,
    score_orientation: 'subject',
    is_complete: false,
    aggregate_is_partial: true,
    aggregate: {
      overall: {
        ...stats(),
        games: 4,
        losses: 4,
        match_score_denominator: 8,
        score_difference_sum: -4,
      },
      as_player_0: stats(),
      as_player_1: stats(),
    },
    error: null,
    created_at: time,
    started_at: time,
    ended_at: null,
    heartbeat_at: time,
    diagnostics,
    config: {},
    participants: [{ ...child }],
    decisions: page([]),
    suites: page(
      [0, 1].map((i) => ({
        id: 'suite-' + i,
        batch_id: 'evaluation-1',
        suite_index: i,
        definition: { game_seeds: ['9223372036854775815'], rows: 7, columns: 7 },
        suite_fingerprint: String(i).repeat(64),
        expected_games: 4,
        expected_player_0_games: 2,
        expected_player_1_games: 2,
        status: i ? 'planned' : 'completed',
        elapsed_seconds: i ? null : 2,
        error: null,
        started_at: time,
        ended_at: i ? null : time,
        result: i
          ? null
          : {
              overall: {
                ...stats(),
                games: 4,
                losses: 4,
                match_score_denominator: 8,
                score_difference_sum: -4,
              },
              as_player_0: stats(),
              as_player_1: stats(),
            },
      })),
    ),
  }
  child.evaluations = page([batch])
  child.recent_evaluations = [batch]
  child.counts.evaluation_batches = 1
  const assignment = {
    id: 'assignment',
    experiment_id: 'experiment',
    generation: 1,
    checkpoint_id: champion.id,
    previous_assignment_id: 'initial',
    attempt_id: first.id,
    decision_id: null,
    reason: 'Promotion',
    created_at: time,
  }
  const current_champion = {
    assignment,
    checkpoint: champion,
    branches: { attempts: 1, qualified_checkpoints: 0, challenge_batches: 0, promotions: 0 },
  }
  const experiment: Experiment = {
    id: 'experiment',
    name: 'default',
    game_config: { rows: 7, columns: 7 },
    game_config_fingerprint: 'game',
    current_champion_assignment_id: assignment.id,
    created_at: time,
    updated_at: time,
    counts: { attempts: 2, checkpoints: 5, evaluation_batches: 1 },
    current_champion,
  }
  const checkpoints = [root, before, champion, tail, child]
  const metric = (attempt_id: string): Metrics => ({
    attempt_id,
    total_samples: 2,
    omitted_samples: 0,
    stride: 1,
    method: 'uniform_samples',
    snapshot_at: time,
    episode_min: null,
    episode_max: null,
    items: [0, 1].map((i) => ({
      attempt_id,
      sample_sequence: i,
      episode: 13250 + i * 100,
      created_at: time,
      metrics: { loss: 0.12 - i * 0.01, games_per_second: 0.4 + i * 0.05, epsilon: 0.1 },
    })),
  })
  return {
    experiment,
    data: {
      lineage: {
        experiment,
        snapshot_at: time,
        counts: experiment.counts,
        current_champion,
        root_checkpoint_id: null,
        episode_min: null,
        episode_max: null,
        attempts: page([first, second]),
        checkpoints: page(
          checkpoints.map((c) => ({ ...c, is_boundary: false, children_outside_slice: 0 })),
        ),
        boundary_checkpoints: [],
        champion_history: [
          {
            ...assignment,
            id: 'initial',
            generation: 0,
            checkpoint_id: root.id,
            previous_assignment_id: null,
            attempt_id: null,
          },
          assignment,
        ],
        edges: checkpoints.flatMap((c) =>
          c.parent_checkpoint_id
            ? [{ parent_checkpoint_id: c.parent_checkpoint_id, child_checkpoint_id: c.id }]
            : [],
        ),
        truncated: false,
      },
      attempts: [first, second],
      checkpoints,
      evaluations: [batch],
      metrics: { [first.id]: metric(first.id), [second.id]: metric(second.id) },
    },
  }
}
