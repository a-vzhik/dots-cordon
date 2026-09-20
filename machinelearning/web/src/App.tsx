import { useCallback, useEffect, useState } from 'react'
import {
  Activity,
  ArrowUpRight,
  BookOpen,
  Database,
  GitBranch,
  Layers3,
  Pause,
  Play,
  RefreshCw,
  Trophy,
} from 'lucide-react'
import { listExperiments } from './api/client'
import type { Experiment } from './api/types'
import {
  AttemptsView,
  CheckpointsView,
  Empty,
  EvaluationsView,
  JsonDetails,
  LineageView,
  Overview,
} from './components'
import { useDashboard } from './useDashboard'
import { currentPage, pageHref } from './navigation'

function updateUrl(key: string, value: string | null) {
  const url = new URL(window.location.href)
  if (value) url.searchParams.set(key, value)
  else url.searchParams.delete(key)
  history.replaceState(null, '', url)
}

export default function App() {
  const page = currentPage(location.pathname)
  const pageTitle = {
    overview: 'Overview',
    lineage: 'Weight lineage',
    attempts: 'Attempts',
    checkpoints: 'Checkpoints',
    evaluations: 'Evaluations',
  }[page]
  const href = (target: typeof page) => pageHref(target, location.search)
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [experiment, setExperiment] = useState('')
  const [listError, setListError] = useState<string | null>(null)
  const [listLoading, setListLoading] = useState(true)
  const [listRetry, setListRetry] = useState(0)
  const [paused, setPaused] = useState(false)
  const [selected, setSelected] = useState<string | null>(() =>
    new URLSearchParams(location.search).get('checkpoint'),
  )
  const [selectedAttempt, setSelectedAttempt] = useState<string | null>(() =>
    new URLSearchParams(location.search).get('attempt'),
  )
  useEffect(() => {
    const controller = new AbortController()
    setListLoading(true)
    listExperiments(controller.signal)
      .then((items) => {
        if (controller.signal.aborted) return
        setExperiments(items)
        setListError(null)
        const requested = new URLSearchParams(location.search).get('experiment')
        const active =
          items.find((e) => e.id === requested || e.name === requested) ??
          items.find((e) => e.name === 'default') ??
          items[0]
        setExperiment(active?.id ?? '')
        if (active) updateUrl('experiment', active.id)
      })
      .catch((err) => {
        if (!controller.signal.aborted)
          setListError(err instanceof Error ? err.message : 'Could not load experiments.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setListLoading(false)
      })
    return () => controller.abort()
  }, [listRetry])
  const { data, error, refreshing, updatedAt, refresh } = useDashboard(experiment, paused, page)
  const chooseCheckpoint = useCallback((id: string | null) => {
    setSelected(id)
    updateUrl('checkpoint', id)
  }, [])
  const chooseAttempt = useCallback((id: string | null) => {
    setSelectedAttempt(id)
    setSelected(null)
    updateUrl('attempt', id)
    updateUrl('checkpoint', null)
  }, [])
  const visibleData = data?.experiment === experiment && data.page === page ? data : null
  useEffect(() => {
    if (!data || !location.hash) return
    document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView()
  }, [data?.page, data?.experiment])
  const failure = listError ?? error
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href={href('overview')}>
          <span className="brand-mark">
            <i />
            <i />
            <i />
            <i />
          </span>
          <span>
            dots<span className="brand-light">cordon</span>
            <small>TRAINING LAB</small>
          </span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Dashboard sections">
          <a href={href('overview')} aria-current={page === 'overview' ? 'page' : undefined}>
            <Activity size={17} /> Overview
          </a>
          <a href={href('lineage')} aria-current={page === 'lineage' ? 'page' : undefined}>
            <GitBranch size={17} /> Weight lineage
          </a>
          <a href={href('attempts')} aria-current={page === 'attempts' ? 'page' : undefined}>
            <Layers3 size={17} /> Attempts
          </a>
          <a href={href('checkpoints')} aria-current={page === 'checkpoints' ? 'page' : undefined}>
            <Database size={17} /> Checkpoints
          </a>
          <a href={href('evaluations')} aria-current={page === 'evaluations' ? 'page' : undefined}>
            <Trophy size={17} /> Evaluations
          </a>
        </nav>
        <div className="sidebar-footer">
          <div>
            <span className="connection-dot" /> Training audit
          </div>
          <p>
            A record of every branch.
            <br />A path to better weights.
          </p>
          <a href="/docs" target="_blank" rel="noreferrer">
            <BookOpen size={14} /> API reference <ArrowUpRight size={13} />
          </a>
        </div>
      </aside>
      <main>
        <header className="page-header">
          <div>
            <div className="breadcrumb">
              Dots Cordon <span>/</span> Machine learning
            </div>
            <h1>
              {pageTitle}
              <span>.</span>
            </h1>
            <p>Follow the weights. See what made the champion.</p>
          </div>
          <div className="header-controls">
            <label className="experiment-control">
              EXPERIMENT
              <select
                aria-label="Experiment"
                value={experiment}
                onChange={(e) => {
                  setExperiment(e.target.value)
                  setSelected(null)
                  setSelectedAttempt(null)
                  updateUrl('experiment', e.target.value)
                  updateUrl('checkpoint', null)
                  updateUrl('attempt', null)
                }}
                disabled={!experiments.length}
              >
                {experiments.length ? (
                  experiments.map((e) => (
                    <option key={e.id} value={e.id}>
                      {e.name}
                    </option>
                  ))
                ) : (
                  <option value="">No experiments</option>
                )}
              </select>
            </label>
            <div className="refresh-controls">
              <button
                className={`live-button ${paused ? 'is-paused' : ''}`}
                onClick={() => setPaused((p) => !p)}
                aria-label={paused ? 'Resume live updates' : 'Pause live updates'}
                aria-pressed={!paused}
              >
                {paused ? <Play size={13} /> : <Pause size={13} />}{' '}
                {paused ? 'Paused' : 'Live updates'}
              </button>
              <button
                className="icon-button"
                aria-label="Refresh data"
                disabled={refreshing || listLoading}
                onClick={() => {
                  if (!experiment || listError) setListRetry((r) => r + 1)
                  else refresh()
                }}
              >
                <RefreshCw size={15} className={refreshing || listLoading ? 'spinning' : ''} />
              </button>
            </div>
          </div>
        </header>
        <div className="workspace-meta">
          <span>
            <span
              className={`connection-dot ${failure ? 'dot-error' : paused ? 'dot-paused' : ''}`}
            />
            {failure
              ? 'Connection needs attention'
              : paused
                ? 'Updates paused'
                : 'Auto-refresh every 3 seconds'}
          </span>
          <span>
            {updatedAt
              ? `Last read ${new Date(updatedAt).toLocaleTimeString('en-GB')}`
              : 'Waiting for first read'}
          </span>
        </div>
        {failure && (
          <div className="error-banner" role="alert">
            <div>
              <strong>Couldn’t refresh training data.</strong>
              <p>
                {failure}
                {visibleData ? ' Showing the last successful read.' : ''}
              </p>
            </div>
            <button
              onClick={() => {
                if (listError) setListRetry((r) => r + 1)
                else refresh()
              }}
            >
              Retry <RefreshCw size={14} />
            </button>
          </div>
        )}
        {(listLoading || (experiment && !visibleData && !error)) && (
          <div className="loading-state" role="status">
            <RefreshCw size={21} className="spinning" />
            <strong>Reading training history…</strong>
            <span>Loading {pageTitle.toLowerCase()}.</span>
          </div>
        )}
        {!listLoading && !listError && !experiments.length && (
          <Empty>
            No experiments yet. Start an audited training attempt, then refresh this page.
          </Empty>
        )}
        {visibleData?.page === 'overview' && (
          <>
            <Overview data={visibleData.data} />
            <div className="experiment-details">
              <JsonDetails title="Experiment configuration" value={visibleData.data} />
            </div>
          </>
        )}
        {visibleData && visibleData.page !== 'overview' && (
          <>
            {visibleData.page === 'lineage' && (
              <LineageView
                data={visibleData.data}
                selected={selected}
                onSelect={chooseCheckpoint}
              />
            )}
            {visibleData.page === 'attempts' && <AttemptsView data={visibleData.data} />}
            {visibleData.page === 'checkpoints' && (
              <CheckpointsView data={visibleData.data} selected={selected} />
            )}
            {visibleData.page === 'evaluations' && (
              <EvaluationsView
                data={visibleData.data}
                selected={selected}
                onSelect={chooseCheckpoint}
                selectedAttempt={selectedAttempt}
                onSelectAttempt={chooseAttempt}
              />
            )}
          </>
        )}
        <footer className="page-footer">
          <span>
            Dots Cordon <span>·</span> Training audit
          </span>
          <span>
            Scores and lineage from the training database <Database size={12} />
          </span>
        </footer>
      </main>
    </div>
  )
}
