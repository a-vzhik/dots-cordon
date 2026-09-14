import { useCallback, useEffect, useRef, useState } from 'react'
import { loadDashboard, ReadCache } from './api/client'
import type { DashboardData } from './api/types'

export function useDashboard(experiment: string, paused: boolean) {
  const [data, setData] = useState<DashboardData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [updatedAt, setUpdatedAt] = useState<number | null>(null)
  const refreshRef = useRef<() => void>(() => {})
  const previousExperiment = useRef<string | null>(null)
  useEffect(() => {
    if (!experiment) return
    const changed = previousExperiment.current !== experiment
    previousExperiment.current = experiment
    if (changed) {
      setData(null)
      setUpdatedAt(null)
      setError(null)
    }
    const cache = new ReadCache()
    let disposed = false
    let busy = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let controller: AbortController | undefined
    const refresh = async () => {
      if (disposed || busy || document.hidden) return
      clearTimeout(timer)
      controller = new AbortController()
      const signal = controller.signal
      busy = true
      setRefreshing(true)
      try {
        const next = await loadDashboard(experiment, signal, cache)
        if (!disposed && !signal.aborted) {
          setData(next)
          setUpdatedAt(Date.now())
          setError(null)
        }
      } catch (err) {
        if (!disposed && !signal.aborted) {
          setError(err instanceof Error ? err.message : 'Could not read training data.')
          controller.abort()
        }
      } finally {
        busy = false
        if (!disposed) {
          setRefreshing(false)
          if (!paused && !document.hidden) timer = setTimeout(refresh, 3_000)
        }
      }
    }
    const visibility = () => {
      if (document.hidden) {
        clearTimeout(timer)
        controller?.abort()
      } else if (!paused) void refresh()
    }
    refreshRef.current = () => {
      void refresh()
    }
    document.addEventListener('visibilitychange', visibility)
    if (changed || !paused) void refresh()
    return () => {
      disposed = true
      clearTimeout(timer)
      controller?.abort()
      document.removeEventListener('visibilitychange', visibility)
    }
  }, [experiment, paused])
  const refresh = useCallback(() => refreshRef.current(), [])
  return { data, error, refreshing, updatedAt, refresh }
}
