import { useMemo, useRef, useState } from 'react'
import {
  ArrowDownToLine,
  ArrowUpRight,
  ChevronRight,
  Crown,
  Focus,
  GitBranch,
  Minus,
  Plus,
} from 'lucide-react'
import type { ReactNode } from 'react'
import type {
  AttemptDetail,
  CheckpointDetail,
  DashboardData,
  EvaluationDetail,
  Metrics,
  Stats,
} from './api/types'
import { layoutLineage, nodeHeight, nodeWidth } from './layout'
import {
  bytes,
  date,
  label,
  metricValue,
  number,
  percent,
  record,
  scoreLabel,
  shortId,
  signed,
} from './presentation'

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: string }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}
export function Status({ value }: { value: string }) {
  const tone = ['running', 'training', 'screening', 'challenging'].includes(value)
    ? 'live'
    : ['completed', 'promoted', 'qualified', 'passed', 'selected'].includes(value)
      ? 'green'
      : ['failed', 'rejected', 'not_qualified'].includes(value)
        ? 'red'
        : 'neutral'
  return <Badge tone={tone}>{label(value)}</Badge>
}
export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty-state">{children}</div>
}
export function SectionTitle({
  id,
  number: index,
  title,
  subtitle,
  right,
}: {
  id: string
  number: string
  title: string
  subtitle: string
  right?: ReactNode
}) {
  return (
    <div id={id} className="section-heading">
      <div>
        <div className="eyebrow">
          {index} / {title}
        </div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
      {right}
    </div>
  )
}
export function JsonDetails({ title, value }: { title: string; value: unknown }) {
  return (
    <details className="json-details">
      <summary>{title}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  )
}
function CheckpointLink({ id, data }: { id: string; data: DashboardData }) {
  const checkpoint = data.checkpoints.find((c) => c.id === id)
  return (
    <a href={`#checkpoint-${id}`} className="checkpoint-link" title={id}>
      {checkpoint ? `ep ${number(checkpoint.episode)}` : shortId(id)}
      <ArrowUpRight size={12} />
    </a>
  )
}
export function Overview({ data }: { data: DashboardData }) {
  const champion = data.lineage.current_champion
  const branches = champion?.branches
  return (
    <div className="overview-grid">
      <div className="stat-card champion-stat">
        <div className="stat-label">
          <Crown size={17} /> Current champion
        </div>
        <div className="stat-value">
          {champion ? number(champion.checkpoint.episode) : '—'}
          <span>episodes</span>
        </div>
        <div className="stat-foot">
          {champion ? (
            <>
              Generation {champion.assignment.generation} <span>·</span>{' '}
              {champion.checkpoint.board.rows} × {champion.checkpoint.board.columns} board
            </>
          ) : (
            'No champion assigned yet'
          )}
        </div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Attempts from champion</div>
        <div className="stat-value">{branches ? number(branches.attempts) : '—'}</div>
        <div className="stat-foot">
          {number(data.lineage.counts.attempts)} attempts in this experiment
        </div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Qualified contenders</div>
        <div className="stat-value">{branches ? number(branches.qualified_checkpoints) : '—'}</div>
        <div className="stat-foot">From the current champion’s branches</div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Champion challenges</div>
        <div className="stat-value">{branches ? number(branches.challenge_batches) : '—'}</div>
        <div className="stat-foot">From the current champion’s branches</div>
      </div>
    </div>
  )
}

export function LineageView({
  data,
  selected,
  onSelect,
}: {
  data: DashboardData
  selected: string | null
  onSelect: (id: string) => void
}) {
  const [zoom, setZoom] = useState(1)
  const viewport = useRef<HTMLDivElement>(null)
  const graph = useMemo(() => layoutLineage(data.lineage, zoom), [data.lineage, zoom])
  const championId = data.lineage.current_champion?.checkpoint.id
  const champions = new Set(data.lineage.champion_history.map((c) => c.checkpoint_id))
  const focusChampion = () => {
    const node = graph.nodes.find((n) => n.checkpoint.id === championId)
    if (node && viewport.current) {
      viewport.current.scrollTo({ left: Math.max(0, node.x - 60), behavior: 'smooth' })
      onSelect(node.checkpoint.id)
    }
  }
  return (
    <section>
      <SectionTitle
        id="lineage"
        number="01"
        title="Weight lineage"
        subtitle="Episodes move right. Each training attempt gets its own branch."
        right={
          <div className="toolbar">
            <button onClick={focusChampion} disabled={!championId}>
              <Focus size={15} /> Champion
            </button>
            <div className="button-group">
              <button
                aria-label="Zoom out"
                onClick={() => setZoom((z) => Math.max(0.8, z - 0.2))}
                disabled={zoom <= 0.8}
              >
                <Minus size={15} />
              </button>
              <span>{Math.round(zoom * 100)}%</span>
              <button
                aria-label="Zoom in"
                onClick={() => setZoom((z) => Math.min(2, z + 0.2))}
                disabled={zoom >= 2}
              >
                <Plus size={15} />
              </button>
            </div>
          </div>
        }
      />
      <div className="panel graph-panel">
        <div className="graph-meta">
          <span>
            <GitBranch size={15} /> {number(data.lineage.counts.checkpoints)} saved checkpoints
          </span>
          <div className="legend">
            <span>
              <i className="legend-champion" /> Champion
            </span>
            <span>
              <i className="legend-best" /> Best in attempt
            </span>
            <span>
              <i /> Checkpoint
            </span>
            <span>
              <i className="legend-progress" /> Unsaved progress
            </span>
          </div>
        </div>
        <div className="graph-layout">
          <div className="lane-labels" style={{ height: graph.height }}>
            <div className="axis-caption">TRAINING ATTEMPTS</div>
            {graph.lanes.map((lane) => (
              <div
                className="lane-label"
                key={lane.attempt?.id ?? 'root'}
                style={{ top: lane.y, height: lane.height }}
              >
                <strong>
                  {lane.attempt
                    ? `Attempt ${String(lane.number).padStart(2, '0')}`
                    : 'Starting weights'}
                </strong>
                {lane.attempt ? (
                  <>
                    <a href={`#attempt-${lane.attempt.id}`} className="muted mono">
                      {shortId(lane.attempt.id)} <ChevronRight size={11} />
                    </a>
                    <Status
                      value={
                        lane.attempt.status === 'running' ? lane.attempt.phase : lane.attempt.status
                      }
                    />
                  </>
                ) : (
                  <span className="muted">Imported / earlier lineage</span>
                )}
              </div>
            ))}
          </div>
          <div
            className="graph-viewport"
            ref={viewport}
            tabIndex={0}
            aria-label="Scrollable checkpoint lineage"
          >
            <div className="graph-canvas" style={{ width: graph.width, height: graph.height }}>
              <svg width={graph.width} height={graph.height} aria-hidden="true">
                {graph.ticks.map((tick) => (
                  <g key={tick}>
                    <line
                      className="grid-line"
                      x1={graph.x(tick) + nodeWidth / 2}
                      x2={graph.x(tick) + nodeWidth / 2}
                      y1={38}
                      y2={graph.height}
                    />
                    <text
                      className="axis-label"
                      x={graph.x(tick) + nodeWidth / 2}
                      y={25}
                      textAnchor="middle"
                    >
                      {number(tick)}
                    </text>
                  </g>
                ))}
                {graph.lanes.map((lane) => (
                  <line
                    className="lane-rule"
                    key={lane.attempt?.id ?? 'root'}
                    x1={0}
                    x2={graph.width}
                    y1={lane.y + lane.height - 14}
                    y2={lane.y + lane.height - 14}
                  />
                ))}
                {graph.edges.map((edge) => (
                  <path className="lineage-edge" key={edge.child_checkpoint_id} d={edge.path} />
                ))}
                {graph.lanes
                  .filter((lane) => lane.attempt?.status === 'running')
                  .map((lane) => {
                    const attempt = lane.attempt!
                    const saved = graph.nodes
                      .filter((n) => n.checkpoint.attempt_id === attempt.id)
                      .at(-1)
                    const start = saved?.checkpoint.episode ?? attempt.start_episode
                    if (attempt.latest_episode <= start) return null
                    const y = saved ? saved.y + nodeHeight / 2 : lane.y + nodeHeight / 2
                    return (
                      <g key={attempt.id}>
                        <line
                          className="progress-edge"
                          x1={graph.x(start) + nodeWidth / 2}
                          x2={graph.x(attempt.latest_episode) + nodeWidth / 2}
                          y1={y}
                          y2={y}
                        />
                        <circle
                          fill="#327962"
                          cx={graph.x(attempt.latest_episode) + nodeWidth / 2}
                          cy={y}
                          r={4}
                        />
                        <text
                          className="progress-label"
                          x={graph.x(attempt.latest_episode) + nodeWidth / 2}
                          y={y + 45}
                          textAnchor="middle"
                        >
                          {number(attempt.latest_episode)} · unsaved
                        </text>
                      </g>
                    )
                  })}
              </svg>
              {graph.nodes.map(({ checkpoint, x, y }) => {
                const current = checkpoint.id === championId
                const former = champions.has(checkpoint.id) && !current
                return (
                  <button
                    key={checkpoint.id}
                    onClick={() => onSelect(checkpoint.id)}
                    data-checkpoint-id={checkpoint.id}
                    aria-pressed={selected === checkpoint.id}
                    aria-label={`Checkpoint ${checkpoint.episode}, ${current ? 'current champion' : former ? 'former champion' : shortId(checkpoint.id)}`}
                    className={`checkpoint-node ${current ? 'node-champion' : checkpoint.is_best_in_attempt ? 'node-best' : ''} ${selected === checkpoint.id ? 'node-selected' : ''}`}
                    style={{ left: x, top: y, width: nodeWidth, height: nodeHeight }}
                  >
                    <span className="node-heading">
                      {number(checkpoint.episode)}{' '}
                      {current || former ? <Crown size={14} /> : <span className="node-dot" />}
                    </span>
                    <span className="node-caption">
                      {current
                        ? 'Current champion'
                        : former
                          ? 'Former champion'
                          : checkpoint.is_best_in_attempt
                            ? 'Best in attempt'
                            : checkpoint.is_final_in_attempt
                              ? 'Final checkpoint'
                              : checkpoint.is_screening_candidate
                                ? 'Screening candidate'
                                : 'Saved checkpoint'}
                    </span>
                  </button>
                )
              })}
              {!graph.nodes.length && (
                <div className="graph-empty">
                  Checkpoints will appear here when weights are saved.
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="graph-footer">
          <span>Connections follow saved parent checkpoints.</span>
          <span>
            Click a checkpoint to inspect its weights and scores <ArrowUpRight size={13} />
          </span>
        </div>
      </div>
      {selected && (
        <CheckpointInspector
          checkpoint={data.checkpoints.find((c) => c.id === selected)}
          data={data}
        />
      )}
      <div className="champion-history">
        <Crown size={14} />
        <span>Champion history</span>
        {data.lineage.champion_history.length ? (
          [...data.lineage.champion_history]
            .sort((a, b) => a.generation - b.generation)
            .map((assignment) => (
              <span
                key={assignment.id}
                className="history-item"
                title={`${date(assignment.created_at)} · ${assignment.reason}`}
              >
                <ChevronRight size={13} />
                <CheckpointLink id={assignment.checkpoint_id} data={data} />
                <span className="muted">G{assignment.generation}</span>
              </span>
            ))
        ) : (
          <span className="muted">No assignments yet</span>
        )}
      </div>
    </section>
  )
}

function CheckpointInspector({
  checkpoint,
  data,
}: {
  checkpoint?: CheckpointDetail
  data: DashboardData
}) {
  if (!checkpoint) return null
  const screening = checkpoint.evaluations.items.filter((e) => e.purpose === 'screening').at(-1)
  const challenge = checkpoint.evaluations.items
    .filter((e) => e.purpose === 'head_to_head' && e.checkpoint_id === checkpoint.id)
    .at(-1)
  return (
    <div className="checkpoint-inspector">
      <div>
        <div className="eyebrow">Selected checkpoint</div>
        <strong>Episode {number(checkpoint.episode)}</strong>
        <span className="mono muted">{shortId(checkpoint.id)}</span>
      </div>
      <div>
        <span className="muted">Parent</span>
        {checkpoint.parent_checkpoint_id ? (
          <CheckpointLink id={checkpoint.parent_checkpoint_id} data={data} />
        ) : (
          <strong>Imported weights</strong>
        )}
      </div>
      <div>
        <span className="muted">Screening score</span>
        <strong>{screening ? scoreLabel(screening) : '—'}</strong>
      </div>
      <div>
        <span className="muted">Challenge score</span>
        <strong>{challenge ? scoreLabel(challenge) : '—'}</strong>
      </div>
      <a className="button" href={`#checkpoint-${checkpoint.id}`}>
        View evidence <ArrowDownToLine size={14} />
      </a>
    </div>
  )
}

function Sparkline({ samples, metric, name }: { samples: Metrics; metric: string; name: string }) {
  const values = samples.items.map((item) => ({
    episode: item.episode,
    value: metricValue(item.metrics, metric),
  }))
  const valid = values.filter((v): v is { episode: number; value: number } => v.value !== null)
  if (!valid.length)
    return (
      <div className="spark">
        <div className="spark-label">
          {name}
          <strong>—</strong>
        </div>
        <span className="muted">No samples yet</span>
      </div>
    )
  const lo = Math.min(...valid.map((v) => v.value)),
    hi = Math.max(...valid.map((v) => v.value))
  const x0 = Math.min(...valid.map((v) => v.episode)),
    x1 = Math.max(...valid.map((v) => v.episode))
  let penDown = false
  const path = values
    .map((point) => {
      if (point.value === null) {
        penDown = false
        return ''
      }
      const command = penDown ? 'L' : 'M'
      penDown = true
      return `${command} ${5 + ((point.episode - x0) / (x1 - x0 || 1)) * 290} ${58 - ((point.value - lo) / (hi - lo || 1)) * 48}`
    })
    .join(' ')
  const last = valid.at(-1)!
  return (
    <div className="spark">
      <div className="spark-label">
        {name}
        <strong>{last.value.toFixed(metric === 'loss' ? 5 : 3)}</strong>
      </div>
      <svg
        viewBox="0 0 300 68"
        role="img"
        aria-label={`${name} over episodes ${x0} to ${x1}. Latest ${last.value}. Range ${lo} to ${hi}.`}
      >
        <line x1="0" x2="300" y1="60" y2="60" stroke="#e4eae5" />
        <path
          d={path}
          fill="none"
          stroke="#327962"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
        {valid.length === 1 && <circle cx="5" cy="58" r="3" fill="#327962" />}
      </svg>
      <div className="spark-axis">
        <span>{number(x0)}</span>
        <span>episodes</span>
        <span>{number(x1)}</span>
      </div>
    </div>
  )
}

export function AttemptsView({ data }: { data: DashboardData }) {
  const attempts = [...data.attempts].sort(
    (a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id),
  )
  return (
    <section>
      <SectionTitle
        id="attempts"
        number="02"
        title="Training attempts"
        subtitle="Every branch, its progress, and the decision it produced."
        right={<span className="section-count">{number(attempts.length)} attempts</span>}
      />
      {!attempts.length && (
        <Empty>No attempts recorded. New training branches will appear automatically.</Empty>
      )}
      <div className="attempt-list">
        {attempts.map((attempt, i) => (
          <AttemptCard key={attempt.id} attempt={attempt} index={i + 1} data={data} />
        ))}
      </div>
    </section>
  )
}
function AttemptCard({
  attempt,
  index,
  data,
}: {
  attempt: AttemptDetail
  index: number
  data: DashboardData
}) {
  const progress = Math.max(
    0,
    Math.min(
      100,
      ((attempt.latest_episode - attempt.start_episode) /
        Math.max(1, attempt.target_episode - attempt.start_episode)) *
        100,
    ),
  )
  const metrics = data.metrics[attempt.id]
  const decisions = [
    ...new Map(
      data.evaluations
        .flatMap((e) => e.decisions.items)
        .filter((d) => d.attempt_id === attempt.id)
        .map((d) => [d.id, d]),
    ).values(),
  ]
  return (
    <article className="panel attempt-card" id={`attempt-${attempt.id}`}>
      <div className="attempt-top">
        <div className="attempt-title">
          <span className="attempt-index">{String(index).padStart(2, '0')}</span>
          <div>
            <h3>Attempt {String(index).padStart(2, '0')}</h3>
            <span className="muted mono" title={attempt.id}>
              {shortId(attempt.id)}
            </span>
          </div>
          <Status value={attempt.status} />
          {attempt.status === 'running' && <Badge>{attempt.phase}</Badge>}
        </div>
        <span className="muted">
          {date(attempt.started_at)} <span className="time-separator">→</span>{' '}
          {attempt.ended_at ? date(attempt.ended_at) : 'in progress'}
        </span>
      </div>
      <div className="attempt-body">
        <div className="attempt-progress">
          <div className="eyebrow">Episode progression</div>
          <div className="episode-progression">
            {number(attempt.start_episode)} <span>→</span> {number(attempt.latest_episode)}{' '}
            <small>/ {number(attempt.target_episode)}</small>
          </div>
          <div
            className="progress-track"
            role="progressbar"
            aria-label={`Attempt ${index} training progress`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progress)}
          >
            <span style={{ width: `${progress}%` }} />
          </div>
          <p className="muted">
            {number(attempt.counts.checkpoints)} checkpoints ·{' '}
            {number(attempt.counts.qualified_checkpoints)} qualified ·{' '}
            {number(attempt.counts.challenge_batches)} challenges
          </p>
          <div className="attempt-origin">
            Starting weights{' '}
            {attempt.starting_checkpoint_id ? (
              <CheckpointLink id={attempt.starting_checkpoint_id} data={data} />
            ) : (
              <span>New model</span>
            )}
          </div>
        </div>
        {metrics ? (
          <div className="sparklines">
            <Sparkline samples={metrics} metric="loss" name="Training loss" />
            <Sparkline samples={metrics} metric="games_per_second" name="Games / second" />
            <Sparkline samples={metrics} metric="epsilon" name="Exploration ε" />
          </div>
        ) : (
          <p className="muted">No training metrics recorded.</p>
        )}
      </div>
      <div className="attempt-result">
        <span>Outcome</span>
        <strong>
          {attempt.outcome
            ? label(attempt.outcome)
            : attempt.status === 'running'
              ? 'In progress'
              : 'Not recorded'}
        </strong>
        {attempt.stop_reason && <span>{label(attempt.stop_reason)}</span>}
        {metrics && (
          <span className="sample-note">
            {number(metrics.items.length)} of {number(metrics.total_samples)} metric samples
            {metrics.omitted_samples > 0 ? ' · uniformly sampled' : ''}
          </span>
        )}
      </div>
      {attempt.diagnostics.unresponsive && (
        <p className="inline-warning">
          No recent activity since {date(attempt.diagnostics.last_activity_at)}. Recorded status:{' '}
          {attempt.status}.
        </p>
      )}
      {attempt.error && <p className="inline-error">{attempt.error}</p>}
      <div className="attempt-details">
        {decisions.length > 0 && (
          <div className="decision-strip">
            {decisions
              .sort((a, b) => a.created_at.localeCompare(b.created_at))
              .map((d) => (
                <div key={d.id}>
                  <CheckpointLink id={d.checkpoint_id} data={data} />
                  <span>{label(d.stage)}</span>
                  <Status value={d.result} />
                  {d.reason && <span className="muted">{d.reason}</span>}
                </div>
              ))}
          </div>
        )}
        <JsonDetails
          title="Training configuration & provenance"
          value={{ attempt_id: attempt.id, ...attempt.config }}
        />
      </div>
    </article>
  )
}

function Score({ stats, partial = false }: { stats?: Stats | null; partial?: boolean }) {
  return (
    <div className="score">
      <strong>{percent(stats?.match_score)}</strong>
      {partial && <Badge tone="amber">partial</Badge>}
    </div>
  )
}
export function CheckpointsView({
  data,
  selected,
}: {
  data: DashboardData
  selected: string | null
}) {
  const checkpoints = [...data.checkpoints].sort(
    (a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id),
  )
  return (
    <section>
      <SectionTitle
        id="checkpoints"
        number="03"
        title="Checkpoint ledger"
        subtitle="Saved weights, screening scores, and head-to-head results. A dash means no measured score."
        right={<span className="section-count">{number(checkpoints.length)} checkpoints</span>}
      />
      <div className="panel table-scroll">
        <table className="checkpoint-table">
          <thead>
            <tr>
              <th>Checkpoint</th>
              <th>Role</th>
              <th>Parent</th>
              <th>Screening</th>
              <th>Vs champion</th>
              <th>Weights</th>
            </tr>
          </thead>
          <tbody>
            {checkpoints.map((checkpoint) => {
              const screening = checkpoint.evaluations.items
                .filter((e) => e.purpose === 'screening')
                .at(-1)
              const challenge = checkpoint.evaluations.items
                .filter((e) => e.purpose === 'head_to_head' && e.checkpoint_id === checkpoint.id)
                .at(-1)
              return (
                <tr
                  id={`checkpoint-${checkpoint.id}`}
                  key={checkpoint.id}
                  className={selected === checkpoint.id ? 'selected-row' : ''}
                >
                  <td>
                    <strong className="episode-value">{number(checkpoint.episode)}</strong>
                    <span className="cell-note mono" title={checkpoint.id}>
                      {shortId(checkpoint.id)}
                    </span>
                    <span className="cell-note">{date(checkpoint.created_at)}</span>
                  </td>
                  <td>
                    <div className="role-list">
                      {checkpoint.id === data.lineage.current_champion?.checkpoint.id && (
                        <Badge tone="green">
                          <Crown size={12} /> Champion
                        </Badge>
                      )}
                      {checkpoint.is_best_in_attempt && <Badge tone="green">Best in attempt</Badge>}
                      {checkpoint.is_final_in_attempt && <Badge>Final</Badge>}
                      {checkpoint.is_screening_candidate && <Badge>Candidate</Badge>}
                      {!checkpoint.attempt_id && <Badge>Imported</Badge>}
                      {checkpoint.is_periodic_save && <Badge>Periodic save</Badge>}
                    </div>
                  </td>
                  <td>
                    {checkpoint.parent_checkpoint_id ? (
                      <CheckpointLink id={checkpoint.parent_checkpoint_id} data={data} />
                    ) : (
                      <span className="muted">No recorded parent</span>
                    )}
                  </td>
                  <td>
                    {screening ? (
                      <>
                        <a href={`#evaluation-${screening.id}`}>
                          <Score
                            stats={screening.aggregate?.overall}
                            partial={screening.aggregate_is_partial}
                          />
                        </a>
                        <span className="cell-note">
                          {screening.completed_games} / {screening.expected_games} games
                        </span>
                      </>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td>
                    {challenge ? (
                      <>
                        <a href={`#evaluation-${challenge.id}`}>
                          <Score
                            stats={challenge.aggregate?.overall}
                            partial={challenge.aggregate_is_partial}
                          />
                        </a>
                        <span className="cell-note">
                          vs{' '}
                          {data.checkpoints.find((c) => c.id === challenge.opponent_checkpoint_id)
                            ?.episode ?? shortId(challenge.opponent_checkpoint_id ?? '')}
                        </span>
                      </>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td>
                    {checkpoint.blob.available ? (
                      <a
                        className="download-link"
                        href={`/api/v1/checkpoints/${checkpoint.id}/download`}
                        download
                      >
                        <ArrowDownToLine size={14} /> {bytes(checkpoint.blob.byte_length)}
                      </a>
                    ) : (
                      <span className="muted">Unavailable</span>
                    )}
                    <details className="weight-details">
                      <summary>Metadata</summary>
                      <dl>
                        <dt>SHA-256</dt>
                        <dd className="mono">{checkpoint.blob.sha256 ?? '—'}</dd>
                        <dt>Model</dt>
                        <dd>
                          {checkpoint.model.channels} channels / {checkpoint.model.blocks} blocks
                        </dd>
                        <dt>Board</dt>
                        <dd>
                          {checkpoint.board.rows} × {checkpoint.board.columns}
                        </dd>
                        <dt>Environment steps</dt>
                        <dd>{number(checkpoint.environment_steps)}</dd>
                        <dt>Optimization steps</dt>
                        <dd>{number(checkpoint.optimization_steps)}</dd>
                        <dt>Full checkpoint ID</dt>
                        <dd className="mono">{checkpoint.id}</dd>
                      </dl>
                    </details>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {!checkpoints.length && <Empty>No saved checkpoints yet.</Empty>}
      </div>
      <p className="section-note">
        Scores show the latest recorded batch for each purpose. All batches and suite results appear
        below. Weights download only when requested.
      </p>
    </section>
  )
}

export function EvaluationsView({ data }: { data: DashboardData }) {
  const evaluations = [...data.evaluations].sort(
    (a, b) => b.created_at.localeCompare(a.created_at) || b.id.localeCompare(a.id),
  )
  return (
    <section>
      <SectionTitle
        id="evidence"
        number="04"
        title="Evaluation evidence"
        subtitle="Every evaluation batch and suite. Scores belong to the subject checkpoint; draws count as half a win."
        right={<span className="section-count">{number(evaluations.length)} batches</span>}
      />
      {!evaluations.length && (
        <Empty>Evaluation results will appear after the first screening or challenge.</Empty>
      )}
      <div className="evaluation-list">
        {evaluations.map((evaluation) => (
          <EvaluationCard key={evaluation.id} evaluation={evaluation} data={data} />
        ))}
      </div>
    </section>
  )
}
function EvaluationCard({
  evaluation: e,
  data,
}: {
  evaluation: EvaluationDetail
  data: DashboardData
}) {
  return (
    <article className="panel evaluation-card" id={`evaluation-${e.id}`}>
      <div className="evaluation-top">
        <div className="evaluation-title">
          <Badge>{label(e.purpose)}</Badge>
          <h3>
            <CheckpointLink id={e.checkpoint_id} data={data} />
            <span className="muted">vs</span>{' '}
            {e.opponent_checkpoint_id ? (
              <CheckpointLink id={e.opponent_checkpoint_id} data={data} />
            ) : (
              <span>{label(e.opponent_kind)}</span>
            )}
          </h3>
          <Status value={e.status} />
        </div>
        <span className="muted">{date(e.started_at)}</span>
      </div>
      <div className="evaluation-summary">
        <div>
          <span className="muted">Match score</span>
          <Score stats={e.aggregate?.overall} partial={e.aggregate_is_partial} />
        </div>
        <div>
          <span className="muted">Wins / draws / losses</span>
          <strong className="mono">{record(e.aggregate?.overall)}</strong>
        </div>
        <div>
          <span className="muted">Mean score difference</span>
          <strong className="mono">{signed(e.aggregate?.overall.mean_score_difference)}</strong>
        </div>
        <div>
          <span className="muted">As player 0 / player 1</span>
          <strong className="mono">
            {percent(e.aggregate?.as_player_0.match_score)} /{' '}
            {percent(e.aggregate?.as_player_1.match_score)}
          </strong>
        </div>
        <div>
          <span className="muted">Completed games</span>
          <strong className="mono">
            {number(e.completed_games)} <span className="muted">/ {number(e.expected_games)}</span>
          </strong>
        </div>
      </div>
      {e.aggregate_is_partial && (
        <p className="inline-warning">
          Partial aggregate · includes completed suites only ({e.completed_suite_count} /{' '}
          {e.planned_suite_count}).
        </p>
      )}
      {!e.is_complete && !e.aggregate_is_partial && (
        <p className="inline-warning">Evaluation is not complete; no completed suite scores yet.</p>
      )}
      {e.diagnostics.unresponsive && (
        <p className="inline-warning">
          No recent activity since {date(e.diagnostics.last_activity_at)}. Recorded status:{' '}
          {e.status}.
        </p>
      )}
      {e.error && <p className="inline-error">{e.error}</p>}
      <div className="table-scroll">
        <table className="suite-table">
          <thead>
            <tr>
              <th>Suite</th>
              <th>Status</th>
              <th>Games</th>
              <th>Match score</th>
              <th>W / D / L</th>
              <th>Player 0</th>
              <th>Player 1</th>
              <th>Mean difference</th>
            </tr>
          </thead>
          <tbody>
            {e.suites.items.map((suite) => (
              <tr key={suite.id}>
                <td>
                  <strong>{String(suite.suite_index + 1).padStart(2, '0')}</strong>
                  <span className="cell-note mono" title={suite.suite_fingerprint}>
                    {shortId(suite.suite_fingerprint)}
                  </span>
                </td>
                <td>
                  <Status value={suite.status} />
                  {suite.error && <span className="cell-note inline-error">{suite.error}</span>}
                </td>
                <td className="mono">
                  {suite.result?.overall.games ?? 0} / {suite.expected_games}
                </td>
                <td>
                  <Score stats={suite.result?.overall} />
                </td>
                <td className="mono">{record(suite.result?.overall)}</td>
                <td>{percent(suite.result?.as_player_0.match_score)}</td>
                <td>{percent(suite.result?.as_player_1.match_score)}</td>
                <td>{signed(suite.result?.overall.mean_score_difference)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {e.decisions.items.length > 0 && (
        <div className="evaluation-decisions">
          {e.decisions.items.map((d) => (
            <div key={d.id} className="decision-evidence">
              <div>
                <strong>{label(d.stage)}</strong>
                <Status value={d.result} />
                {d.rank !== null && <Badge>Rank {d.rank}</Badge>}
                <CheckpointLink id={d.checkpoint_id} data={data} />
                {d.reason && <span>{d.reason}</span>}
              </div>
              <div className="decision-links">
                {d.candidate_evaluation_id && (
                  <a href={`#evaluation-${d.candidate_evaluation_id}`}>
                    Candidate evaluation <ArrowUpRight size={12} />
                  </a>
                )}
                {d.champion_evaluation_id && (
                  <a href={`#evaluation-${d.champion_evaluation_id}`}>
                    Champion baseline <ArrowUpRight size={12} />
                  </a>
                )}
              </div>
              <JsonDetails title="Decision policy" value={d.policy} />
            </div>
          ))}
        </div>
      )}
      <div className="evaluation-details">
        <JsonDetails
          title="Evaluation configuration & suite definitions"
          value={{
            evaluation_id: e.id,
            config: e.config,
            suites: e.suites.items.map((s) => ({
              id: s.id,
              fingerprint: s.suite_fingerprint,
              definition: s.definition,
            })),
          }}
        />
      </div>
    </article>
  )
}
