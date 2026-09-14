import type {
  AttemptDetail,
  CheckpointDetail,
  DashboardData,
  Evaluation,
  EvaluationDetail,
  Experiment,
  Lineage,
  Metrics,
  Page,
} from './types'

const prefix = '/api/v1'
export async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal, headers: { Accept: 'application/json' } })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.error?.message ?? `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

function query(path: string, values: Record<string, string | number | null | undefined>) {
  const params = new URLSearchParams()
  Object.entries(values).forEach(([key, value]) => {
    if (value != null) params.set(key, String(value))
  })
  return `${prefix}${path}?${params}`
}

export async function readPages<T>(fetchPage: (cursor: string | null) => Promise<Page<T>>) {
  const items: T[] = []
  const seen = new Set<string>()
  let cursor: string | null = null
  do {
    const page = await fetchPage(cursor)
    items.push(...page.items)
    cursor = page.next_cursor
    if (cursor && seen.has(cursor)) throw new Error('The API returned a repeated page cursor.')
    if (cursor) seen.add(cursor)
  } while (cursor)
  return items
}

export const listExperiments = (signal?: AbortSignal) =>
  readPages<Experiment>((cursor) => request(query('/experiments', { limit: 200, cursor }), signal))

// Checkpoint cursors belong to one attempt window. Finish that window before
// advancing the attempt cursor, retaining boundary nodes until their full page arrives.
export async function readLineage(experiment: string, signal: AbortSignal): Promise<Lineage> {
  let initial: Lineage | undefined
  const attempts = new Map<string, Lineage['attempts']['items'][number]>()
  const checkpoints = new Map<string, Lineage['checkpoints']['items'][number]>()
  const boundaries = new Map<string, Lineage['boundary_checkpoints'][number]>()
  const history = new Map<string, Lineage['champion_history'][number]>()
  const edges = new Map<string, Lineage['edges'][number]>()
  let attemptCursor: string | null = null
  const seenAttempts = new Set<string>()
  do {
    let checkpointCursor: string | null = null
    let nextAttempt: string | null = null
    const seenCheckpoints = new Set<string>()
    do {
      const page: Lineage = await request(
        query(`/experiments/${encodeURIComponent(experiment)}/lineage`, {
          attempt_limit: 100,
          checkpoint_limit: 500,
          attempt_cursor: attemptCursor,
          checkpoint_cursor: checkpointCursor,
        }),
        signal,
      )
      initial ??= page
      page.attempts.items.forEach((item) => attempts.set(item.id, item))
      page.checkpoints.items.forEach((item) => checkpoints.set(item.id, item))
      page.boundary_checkpoints.forEach((item) => boundaries.set(item.id, item))
      page.champion_history.forEach((item) => history.set(item.id, item))
      page.edges.forEach((item) => edges.set(item.child_checkpoint_id, item))
      nextAttempt = page.attempts.next_cursor
      checkpointCursor = page.checkpoints.next_cursor
      if (checkpointCursor && seenCheckpoints.has(checkpointCursor))
        throw new Error('Repeated checkpoint page.')
      if (checkpointCursor) seenCheckpoints.add(checkpointCursor)
    } while (checkpointCursor)
    attemptCursor = nextAttempt
    if (attemptCursor && seenAttempts.has(attemptCursor)) throw new Error('Repeated attempt page.')
    if (attemptCursor) seenAttempts.add(attemptCursor)
  } while (attemptCursor)
  return {
    ...initial!,
    attempts: {
      ...initial!.attempts,
      items: [...attempts.values()],
      next_cursor: null,
      remaining: 0,
    },
    checkpoints: {
      ...initial!.checkpoints,
      items: [...checkpoints.values()],
      next_cursor: null,
      remaining: 0,
    },
    boundary_checkpoints: [...boundaries.values()].filter((item) => !checkpoints.has(item.id)),
    champion_history: [...history.values()],
    edges: [...edges.values()],
    truncated: false,
  }
}

export class ReadCache {
  private values = new Map<string, { time: number; version: string; value: unknown }>()
  async get<T>(key: string, version: string, live: boolean, load: () => Promise<T>): Promise<T> {
    const cached = this.values.get(key)
    if (cached && !live && cached.version === version && Date.now() - cached.time < 30_000) {
      return cached.value as T
    }
    const value = await load()
    this.values.set(key, { value, version, time: Date.now() })
    return value
  }
}

async function pool<T, R>(items: T[], task: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let next = 0
  await Promise.all(
    Array.from({ length: Math.min(items.length, 4) }, async () => {
      while (next < items.length) {
        const index = next++
        results[index] = await task(items[index])
      }
    }),
  )
  return results
}

async function readAttempt(id: string, signal: AbortSignal): Promise<AttemptDetail> {
  let first: AttemptDetail | undefined
  const items = await readPages<AttemptDetail['checkpoints']['items'][number]>(async (cursor) => {
    const detail: AttemptDetail = await request(
      query(`/attempts/${id}`, {
        checkpoint_limit: 200,
        checkpoint_cursor: cursor,
      }),
      signal,
    )
    first ??= detail
    return detail.checkpoints
  })
  return {
    ...first!,
    checkpoints: { ...first!.checkpoints, items, next_cursor: null, remaining: 0 },
  }
}

async function readCheckpoint(id: string, signal: AbortSignal): Promise<CheckpointDetail> {
  let first: CheckpointDetail | undefined
  const items = await readPages<Evaluation>(async (cursor) => {
    const detail: CheckpointDetail = await request(
      query(`/checkpoints/${id}`, {
        evaluation_limit: 200,
        evaluation_cursor: cursor,
      }),
      signal,
    )
    first ??= detail
    return detail.evaluations
  })
  return {
    ...first!,
    evaluations: { ...first!.evaluations, items, next_cursor: null, remaining: 0 },
  }
}

async function readEvaluation(id: string, signal: AbortSignal): Promise<EvaluationDetail> {
  const detail: EvaluationDetail = await request(
    query(`/evaluations/${id}`, {
      suite_limit: 200,
      decision_limit: 200,
    }),
    signal,
  )
  const suites = await readPages<EvaluationDetail['suites']['items'][number]>(async (cursor) => {
    if (!cursor) return detail.suites
    const page: EvaluationDetail = await request(
      query(`/evaluations/${id}`, {
        suite_limit: 200,
        suite_cursor: cursor,
        decision_limit: 1,
      }),
      signal,
    )
    return page.suites
  })
  const decisions = await readPages<EvaluationDetail['decisions']['items'][number]>(
    async (cursor) => {
      if (!cursor) return detail.decisions
      const page: EvaluationDetail = await request(
        query(`/evaluations/${id}`, {
          suite_limit: 1,
          decision_limit: 200,
          decision_cursor: cursor,
        }),
        signal,
      )
      return page.decisions
    },
  )
  return {
    ...detail,
    suites: { ...detail.suites, items: suites, remaining: 0, next_cursor: null },
    decisions: { ...detail.decisions, items: decisions, remaining: 0, next_cursor: null },
  }
}

export async function loadDashboard(
  experiment: string,
  signal: AbortSignal,
  cache: ReadCache,
): Promise<DashboardData> {
  const lineage = await readLineage(experiment, signal)
  const active = new Set(
    lineage.attempts.items.filter((a) => a.status === 'running').map((a) => a.id),
  )
  const attemptResults = await pool(lineage.attempts.items, async (attempt) => {
    const version = `${attempt.updated_at}:${attempt.counts.checkpoints}:${attempt.latest_episode}`
    const detail = await cache.get(`attempt:${attempt.id}`, version, active.has(attempt.id), () =>
      readAttempt(attempt.id, signal),
    )
    const metrics = await cache.get<Metrics>(
      `metrics:${attempt.id}`,
      version,
      active.has(attempt.id),
      () => request(query(`/attempts/${attempt.id}/metrics`, { max_points: 120 }), signal),
    )
    return { detail, metrics }
  })
  const nodes = [...lineage.checkpoints.items, ...lineage.boundary_checkpoints]
  const checkpoints = await pool(nodes, (node) =>
    cache.get(
      `checkpoint:${node.id}`,
      JSON.stringify([
        node.counts,
        node.recent_evaluations,
        node.is_best_in_attempt,
        node.is_final_in_attempt,
      ]),
      active.has(node.attempt_id ?? ''),
      () => readCheckpoint(node.id, signal),
    ),
  )
  const summaries = new Map<string, Evaluation>()
  checkpoints.forEach((checkpoint) =>
    checkpoint.evaluations.items.forEach((e) => summaries.set(e.id, e)),
  )
  const evaluations = await pool([...summaries.values()], (batch) =>
    cache.get(
      `evaluation:${batch.id}`,
      `${batch.status}:${batch.completed_suite_count}:${batch.heartbeat_at}`,
      batch.status === 'running' || active.has(batch.attempt_id ?? ''),
      () => readEvaluation(batch.id, signal),
    ),
  )
  return {
    lineage,
    checkpoints,
    evaluations,
    attempts: attemptResults.map((a) => a.detail),
    metrics: Object.fromEntries(attemptResults.map((a) => [a.detail.id, a.metrics])),
  }
}
