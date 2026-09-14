import { describe, expect, it } from 'vitest'
import { fixture } from '../tests/fixtures'
import { layoutLineage } from './layout'

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
})
