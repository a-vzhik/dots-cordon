import { describe, expect, it } from 'vitest'
import { rejectedChallengeFixture } from '../tests/fixtures'
import { challengeEvidence, precisePercent } from './presentation'

describe('challenge explanation', () => {
  it('explains the rejected 400 vs 300 example despite a positive points margin', () => {
    const evaluation = rejectedChallengeFixture().data.evaluations[0]
    const gate = challengeEvidence(evaluation)!
    expect(evaluation.aggregate!.overall.mean_score_difference).toBeGreaterThan(0)
    expect(gate).toMatchObject({
      minimumScore: 0.52,
      minimumSuiteWins: 2,
      suiteWins: 1,
      scoreMet: false,
      suitesMet: false,
    })
    expect(gate.decision.result).toBe('rejected')
  })

  it('uses recorded extended thresholds without rounding them down to 50%', () => {
    const evaluation = rejectedChallengeFixture().data.evaluations[0]
    evaluation.decisions.items[0].policy = {
      promotion_min_match_score: 0.50005,
      promotion_min_suite_wins: 1,
    }
    const gate = challengeEvidence(evaluation)!
    expect(precisePercent(gate.minimumScore)).toBe('50.005%')
    expect(gate.suitesMet).toBe(true)
  })

  it('does not count a drawn suite as a win', () => {
    const evaluation = rejectedChallengeFixture().data.evaluations[0]
    evaluation.suites.items[2].result!.overall.match_score = 0.5
    expect(challengeEvidence(evaluation)!.suiteWins).toBe(0)
  })

  it.each([
    [0.5, false],
    [0.50005, true],
    [0.51, true],
    [0.52, true],
  ])('shows whether %s meets the above-50%% requirement', (score, met) => {
    const evaluation = rejectedChallengeFixture().data.evaluations[0]
    evaluation.aggregate!.overall.match_score = score
    const gate = challengeEvidence(evaluation)!
    expect(gate.scoreMet).toBe(met)
    expect(gate.minimumScore).toBe(0.52)
  })

  it('does not invent thresholds or explain another candidate’s decision', () => {
    const evaluation = rejectedChallengeFixture().data.evaluations[0]
    evaluation.decisions.items[0].checkpoint_id = 'other'
    expect(challengeEvidence(evaluation)).toBeNull()
    evaluation.decisions.items[0].checkpoint_id = evaluation.checkpoint_id
    evaluation.decisions.items[0].policy = {}
    expect(challengeEvidence(evaluation)).toBeNull()
  })

  it('does not present incomplete evidence or screening as a completed challenge', () => {
    const evaluation = rejectedChallengeFixture().data.evaluations[0]
    evaluation.is_complete = false
    expect(challengeEvidence(evaluation)).toBeNull()
    evaluation.is_complete = true
    evaluation.purpose = 'screening'
    expect(challengeEvidence(evaluation)).toBeNull()
    evaluation.purpose = 'head_to_head'
    evaluation.suites.items.pop()
    expect(challengeEvidence(evaluation)).toBeNull()
  })
})
