import type { Attempt, CheckpointNode, Lineage } from './api/types'

export const nodeWidth = 148
export const nodeHeight = 60
const laneHeight = 100
export interface PositionedNode {
  checkpoint: CheckpointNode
  x: number
  y: number
}
export interface Lane {
  attempt: Attempt | null
  number: number
  y: number
  height: number
}

export function layoutLineage(lineage: Lineage, zoom = 1) {
  const attempts = [...lineage.attempts.items].sort(
    (a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id),
  )
  const nodes = [...lineage.checkpoints.items, ...lineage.boundary_checkpoints].sort(
    (a, b) =>
      a.save_sequence - b.save_sequence ||
      a.created_at.localeCompare(b.created_at) ||
      a.id.localeCompare(b.id),
  )
  const episodes = [
    ...nodes.map((n) => n.episode),
    ...attempts.flatMap((a) => [a.start_episode, a.latest_episode, a.target_episode]),
  ]
  const min = episodes.length ? Math.min(...episodes) : 0
  const max = episodes.length ? Math.max(...episodes) : min + 1000
  const distinct = [...new Set(nodes.map((n) => n.episode))].sort((a, b) => a - b)
  const smallestGap = distinct.slice(1).reduce((gap, n, i) => Math.min(gap, n - distinct[i]), 250)
  const scale = Math.max(0.8, (nodeWidth + 35) / Math.max(1, smallestGap)) * zoom
  const x = (episode: number) => 40 + (episode - min) * scale
  const lanes: Lane[] = []
  const positions: PositionedNode[] = []
  let top = 62
  const placeLane = (attempt: Attempt | null, laneNodes: CheckpointNode[], index: number) => {
    const stacks = new Map<number, number>()
    let depth = 1
    laneNodes.forEach((checkpoint) => {
      const stack = stacks.get(checkpoint.episode) ?? 0
      stacks.set(checkpoint.episode, stack + 1)
      depth = Math.max(depth, stack + 1)
      positions.push({ checkpoint, x: x(checkpoint.episode), y: top + stack * (nodeHeight + 12) })
    })
    const height = laneHeight + (depth - 1) * (nodeHeight + 12)
    lanes.push({ attempt, number: index, y: top, height })
    top += height
  }
  const attemptIds = new Set(attempts.map((a) => a.id))
  const roots = nodes.filter((n) => !n.attempt_id || !attemptIds.has(n.attempt_id))
  if (roots.length || !attempts.length) placeLane(null, roots, 0)
  attempts.forEach((attempt, i) =>
    placeLane(
      attempt,
      nodes.filter((n) => n.attempt_id === attempt.id),
      i + 1,
    ),
  )
  const byId = new Map(positions.map((p) => [p.checkpoint.id, p]))
  const edges = lineage.edges.flatMap((edge) => {
    const from = byId.get(edge.parent_checkpoint_id)
    const to = byId.get(edge.child_checkpoint_id)
    if (!from || !to) return []
    // A shared checkpoint remains in its original lane even after promotion.
    // Branches leave its bottom edge; continuation checkpoints stay in that lane.
    const sameLane = from.y === to.y
    const startX = from.x + (sameLane && from.x < to.x ? nodeWidth : nodeWidth / 2)
    const startY = from.y + (sameLane && from.x < to.x ? nodeHeight / 2 : nodeHeight)
    const endX = to.x
    const endY = to.y + nodeHeight / 2
    const path =
      sameLane && from.x < to.x
        ? `M ${startX} ${startY} H ${endX}`
        : `M ${startX} ${startY} V ${endY - 12} Q ${startX} ${endY} ${startX + 12} ${endY} H ${endX}`
    return [{ ...edge, path }]
  })
  const tickStep = Math.max(250, Math.ceil((nodeWidth + 35) / scale / 250) * 250)
  const ticks = [min]
  for (let tick = Math.ceil((min + 1) / tickStep) * tickStep; tick <= max; tick += tickStep)
    ticks.push(tick)
  return {
    lanes,
    nodes: positions,
    edges,
    ticks,
    x,
    min,
    max,
    width: Math.max(850, x(max) + nodeWidth + 40),
    height: Math.max(top + 20, 210),
  }
}
