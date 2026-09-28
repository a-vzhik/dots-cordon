import { describe, expect, it } from 'vitest'
import { checkpoint, fixture, page } from '../tests/fixtures'
import { layoutLineage, nodeHeight, nodeWidth } from './layout'

describe('weight lineage', () => {
  it('keeps a promoted intermediate checkpoint in its original attempt while new attempts branch from it', () => {
    const { data } = fixture()
    const graph = layoutLineage(data.lineage)
    const node = (id: string) => graph.nodes.find((n) => n.checkpoint.id === id)!
    expect(node('champion').y).toBe(node('tail').y)
    expect(node('child').y).toBeGreaterThan(node('champion').y)
    expect(node('child').x).toBe(node('tail').x)
    expect(
      graph.edges
        .filter((e) => e.parent_checkpoint_id === 'champion')
        .map((e) => e.child_checkpoint_id)
        .sort(),
    ).toEqual(['child', 'tail'])
    expect(graph.nodes).toHaveLength(5)
    expect(graph.nodes.some((n) => n.checkpoint.episode === 13600)).toBe(false)
  })
  it('retains lane order across polls and stacks repeated saves without inventing episode distance', () => {
    const { data } = fixture()
    const original = layoutLineage(data.lineage)
    data.lineage.attempts.items.reverse()
    data.lineage.checkpoints.items.push({
      ...data.lineage.checkpoints.items[2],
      id: 'same-episode',
      save_sequence: 20000,
    })
    const graph = layoutLineage(data.lineage)
    expect(graph.lanes.map((l) => l.attempt?.id)).toEqual(original.lanes.map((l) => l.attempt?.id))
    const a = graph.nodes.find((n) => n.checkpoint.id === 'champion')!
    const b = graph.nodes.find((n) => n.checkpoint.id === 'same-episode')!
    expect(a.x).toBe(b.x)
    expect(a.y).not.toBe(b.y)
  })
  it('does not blow up the horizontal scale as unsaved progress approaches a checkpoint', () => {
    const { data } = fixture()
    const width = layoutLineage(data.lineage).width
    data.lineage.attempts.items[1].latest_episode = 13501
    expect(layoutLineage(data.lineage).width).toBe(width)
  })

  it.each([0.8, 1, 2])(
    'keeps an off-cadence save compact without overlapping cards at zoom %s',
    (zoom) => {
      const { data } = fixture()
      const lineage = data.lineage
      lineage.attempts.items = [
        {
          ...lineage.attempts.items[0],
          start_episode: 0,
          latest_episode: 900,
          target_episode: 1000,
        },
      ]
      lineage.checkpoints = page(
        Array.from({ length: 10 }, (_, i) => ({
          ...checkpoint(`saved-${i * 100}`, i * 100, 'attempt-1'),
          is_boundary: false,
          children_outside_slice: 0,
        })),
      )
      const before = layoutLineage(lineage, zoom)
      lineage.checkpoints.items.push({
        ...lineage.checkpoints.items.at(-1)!,
        id: 'interrupted-save',
        episode: 905,
        save_sequence: 905,
      })
      lineage.attempts.items[0].latest_episode = 905
      const graph = layoutLineage(lineage, zoom)
      expect(graph.width).toBe(before.width)
      expect(graph.x(100) - graph.x(0)).toBeCloseTo(183 * zoom)
      for (const previous of before.nodes) {
        expect(graph.nodes.find((node) => node.checkpoint.id === previous.checkpoint.id)!.x).toBe(
          previous.x,
        )
      }
      for (const [i, a] of graph.nodes.entries()) {
        for (const b of graph.nodes.slice(i + 1)) {
          expect(Math.abs(a.x - b.x) >= nodeWidth || Math.abs(a.y - b.y) >= nodeHeight).toBe(true)
        }
      }
      expect(
        graph.nodes.find((node) => node.checkpoint.id === 'interrupted-save')!.y,
      ).toBeGreaterThan(graph.nodes.find((node) => node.checkpoint.id === 'saved-900')!.y)
    },
  )
})
