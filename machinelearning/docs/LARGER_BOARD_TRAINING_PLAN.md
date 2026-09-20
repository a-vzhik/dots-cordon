# Larger-board search training

Status: proposed implementation plan, 2026-09-19. No training or implementation
changes are part of this document.

## Intended result

Start a new larger-board experiment from the existing 7×7 search champion,
optionally expand its network without changing its initial predictions, and
run a supervising CLI that invokes bounded training, evaluation and promotion
commands as child processes.

The initial workflow is sequential:

```text
train 100 episodes with search → save checkpoint → random screening
    → initial head-to-head against current champion
    → extended head-to-head if borderline better
    → promote if the applicable gate passes
    → train the next 100 episodes
```

A rejected candidate also proceeds to the next training round. The learner
continues from its latest training checkpoint, preserving weights, optimizer,
replay, RNG and counters. Champion selection is separate from learner continuity.
Training does not advance while its candidate is being evaluated.

The unit-of-work boundaries could later be invoked by queue consumers. SQS,
LocalStack, message routing, concurrent training/evaluation and a distributed
supervisor are outside this implementation. No background checkpoint watcher is
needed for the initial workflow: the supervisor already knows which checkpoint
its training child produced.

This is a code-change plan. Benchmarking, hyperparameter selection and launching
the actual long run are separate operational work. Proposed architecture:
64 channels and seven residual blocks, subject to the source checkpoint's actual
architecture. The board and round size are configurable; 100 episodes is the
initial round-size default. There is no 15×15 training limit. Seven blocks give
a 31×31 spatial receptive field, sufficient to receive whole-board spatial
information at every move on 15×15, without guaranteeing playing strength.

## Isolated features and dependencies

Each feature is a separately reviewable implementation change. Verification and
necessary usage documentation belong with their feature, not separate features.

| ID | Feature | Depends on | Code deliverable |
| --- | --- | --- | --- |
| F1 | Transfer a search checkpoint to a new board | — | Explicit new-experiment initialization |
| F2 | Expand network depth while preserving predictions | F1 | Identity-initialized extra residual blocks |
| F3 | Support search baselines and bootstrap the target champion | F1; F2 for expanded baseline | Compatible opponent loading and initial champion |
| F4 | Bounded training CLI | F1 | One resumable training round with a durable checkpoint result |
| F5 | Parameterized evaluation CLI | F3 | One evaluation job with durable results, no promotion |
| F6 | Promotion CLI | F5 | Validate evidence and atomically update the champion |
| F7 | Supervising CLI | F4–F6 | Sequential subprocess orchestration and recovery |

F1–F3 prepare the larger-board model and initial champion. F4–F7 implement the
requested train/checkpoint/evaluate/promote loop using independently callable
units. Existing modules and commands should be reused where practical; filenames
such as `training.py`, `evaluation.py` and `promotion.py` describe responsibilities,
not a requirement to rename the current modules.

## F1 — Transfer search weights across board sizes

Problem: `search_train.py` accepts only DQN checkpoints for `--initialize-from`
and requires the source and destination boards to match. `--resume` restores
replay, optimizer and counters, making it unsuitable for a board-size change.

Implementation:

- Extend initialization to accept policy/value checkpoints. Copy the complete
  network, including both heads, when architecture matches.
- Allow destination dimensions to differ during explicit transfer initialization.
  Keep resume restricted to compatible board, architecture and rules.
- Start fresh optimizer, empty replay, fresh RNG seeded by the new command, and
  zero training counters. Never mix old-board examples into the new replay.
- Resolve the source reference once to immutable bytes. Record source checkpoint
  ID when available, SHA-256, source board/model/episode, destination board/model,
  and transfer method in checkpoint and audit configuration.
- Preserve the original checkpoint and experiment. Cross-experiment provenance
  must not masquerade as an ordinary same-experiment resume or fabricated training.
  Use existing configuration metadata where possible; introduce schema changes
  only if required by an explicit storage invariant.
- Retain existing DQN feature-transfer behavior as a distinct initialization mode.

Primary files: `search_train.py`, `checkpoint.py`, `audit/integration.py`, and
`tests/test_search_training.py`, under `machinelearning/` / `dots_cordon_ml/`
as appropriate.

Acceptance:

- A 7×7 policy/value checkpoint initializes a new 10×15 or 15×15 experiment with
  identical learned tensors, empty replay and zero counters.
- The transferred model produces the same outputs as the source when both are
  given identical inputs at the same dimensions, including rectangular inputs.
- A committed episode-zero checkpoint is available before optimization begins.
- The new run can save and resume normally; existing deterministic CPU resume
  coverage still passes. Incompatible resume remains rejected.

## F2 — Expand depth without losing the initial policy

Problem: `--blocks` supports deeper fresh networks, but checkpoint loading
requires exact depth. Randomly adding blocks would change the champion's policy.

Implementation:

- Permit destination depth greater than source depth during transfer only.
  Initially require identical channel counts; reject shrinking and width changes
  with an actionable error.
- Copy the stem, existing blocks, policy head and value head. Append new residual
  blocks before the heads.
- Initialize each added block as an identity on the network's nonnegative feature
  activations: initialize the first convolution normally and zero the second
  convolution's weights and bias. Do not zero both convolutions.
- Record the expansion method and source/destination block counts. Use a fresh
  optimizer; extra blocks remain trainable.

Primary files: `search.py`, the F1 initialization path, `tests/test_search.py`,
and `tests/test_search_training.py`.

Acceptance:

- Before optimization, expanded and original networks agree on policy logits and
  values within numerical tolerance across 7×7, 10×15 and 15×15 inputs.
- A nondegenerate learning example produces gradients and updates in the added
  blocks; expansion does not create permanently inactive layers.
- Expanded models save, reload, resume and export with the correct architecture.
- Performance is measured separately from prediction-preservation correctness.

## F3 — Support the search baseline and target champion

Problem: search training and standalone search evaluation currently restrict
the frozen baseline to DQN and label it `greedy_dqn`. The target experiment also
requires a board-compatible initial champion before candidates can challenge it.

Implementation:

- Reuse the shared checkpoint player loader to accept either DQN or policy/value
  opponents in `search_train.py` and `search_evaluate.py`.
- Record the actual opponent model kind and playing mode. Continue separating
  policy-only and search-assisted subject evaluations; the baseline plays greedily.
- Use the transferred episode-zero checkpoint as the explicit initial champion
  of the new experiment. With F2, its initial predictions match the source
  champion on the target board. Mark it as a transferred bootstrap, not a winner
  of a promotion contest or a model already trained on that board.
- Bootstrap atomically only when the target experiment has no champion. A restart
  must never overwrite a later promoted champion.
- Keep the training diagnostic opponent frozen to its initial checkpoint ID.
  Promotion challenges use the current target-experiment champion instead.
- Make the initial policy available to the target-board evaluation command.
  Evaluation records must specify the board/rules actually played; do not infer
  larger-board performance from 7×7 scores.

Primary files: `search_train.py`, `search_evaluate.py`,
`promotion_evaluation.py`, audit integration, and evaluation tests.

Acceptance:

- Both opponent kinds load and produce correctly labeled evaluation records.
- The original 7×7 champion assignment remains unchanged; the new experiment has
  a traceable episode-zero champion with destination board/rules metadata.
- Restarting cannot re-bootstrap over an incumbent. Frozen diagnostics keep the
  same opponent even after a promotion.

## F4 — Expose one bounded training round as a CLI unit

Implementation:

- Reuse `search_train.py` for search-assisted self-play. Accept an immutable source
  checkpoint, explicit effective training configuration, and an absolute target
  episode. The supervisor translates a 100-episode round into targets 100, 200,
  300, etc.; an interrupted round retains its original target.
- Disable inline evaluations in this mode. Train to the target, commit a complete
  checkpoint, then exit with a machine-readable result containing attempt ID,
  checkpoint ID/hash, reached episode and completion status.
- Save network, optimizer, replay, RNG and counters so process boundaries do not
  reset learning. Carry the complete effective configuration into later rounds;
  do not let omitted CLI defaults silently change training behavior.
- Support graceful interruption with a resumable partial checkpoint. Report partial
  versus completed rounds explicitly. A partial round is resumed to its existing
  target before its normal evaluation starts.
- Accept a stable operation ID and record the final result durably, allowing a
  retry to find an already completed round instead of training it again.
- Preserve existing attempt semantics: each invocation can create a new attempt
  with explicit resume ancestry; one attempt per uninterrupted process is fine.

Primary files: `search_train.py`, checkpoint/audit integration, and training tests.

Acceptance:

- Separate 100-episode rounds continue the same learner, including replay and
  optimizer. Small deterministic CPU fixtures verify equivalence to uninterrupted
  training with the same effective settings.
- A result is reported complete only after its full checkpoint is committed.
- Repeating a completed operation does not add episodes; interrupt/resume reaches
  the original target and preserves an honest attempt history.

## F5 — Expose evaluation as a parameterized CLI unit

Implementation:

- Extract/reuse shared evaluation functions from `promotion_evaluation.py` and
  `promote_run.py`. The CLI runs one requested evaluation stage and exits without
  promoting or deciding which stage should run next.
- Accept immutable subject checkpoint IDs, opponent kind (`random` or checkpoint),
  explicit opponent ID where applicable, board/rules, seeds, paired-seat/opening
  settings, games/suites, playing mode, device and worker limit.
- Random screening evaluates champion and candidate on matching suites. Initial
  and extended head-to-head evaluations use independently reserved fresh suites.
  Record the complete job definition and actual participants.
- The supervisor resolves “latest champion” to an assignment and checkpoint ID
  at the start of a contest. Children use those immutable references throughout
  screening and challenge stages.
- Keep policy-only evaluation as the promotion mode, matching current gameplay
  and saved-checkpoint promotion. Optional MCTS diagnostics remain separate and
  cannot be mistaken for evidence supporting a policy-only promotion.
- Return evaluation IDs and structured results, including W/D/L, match score,
  per-suite and per-seat results, and completion status. Match score counts a draw
  as half a win.
- Reuse completed results for the same operation ID. Preserve failed/interrupted
  evidence and record replacement suites explicitly; incomplete batches cannot
  support promotion. Preserve the existing fresh-suite retry convention.

Primary files: `promotion_evaluation.py`, `search_evaluate.py`, `promote_run.py`,
CLI registration if needed, and evaluation/audit tests.

Acceptance:

- Random screening, initial head-to-head and extended head-to-head can each be
  invoked independently with explicit settings and no champion mutation.
- Policy/value and DQN participants are labeled correctly. Replaying a completed
  operation does not silently generate new results.
- Parameter, seed, participant and completion metadata are sufficient to validate
  a promotion without parsing terminal output.

## F6 — Expose promotion as a CLI unit

Implementation:

- Accept candidate ID, expected incumbent assignment ID, completed screening and
  challenge evaluation IDs, exact gate configuration, and operation ID.
- Validate evidence in the shared application layer: correct experiment, board,
  participants, playing mode, completed suites and applicable thresholds. A caller
  cannot bypass gates by passing an unsupported “passed” boolean.
- Preserve existing gates: screening within configured random-score regression;
  normal head-to-head match score at least 0.52 with at least two of the default
  three suites above 0.5; extended evaluation only for an initial score strictly
  between 0.5 and 0.52. Extended promotion requires a combined extended-suite
  match score strictly above 0.5, even if the margin is small. Use the existing
  gate functions and configurable settings, not a second implementation.
- Atomically update the champion only if the expected incumbent is still current.
  Record the evidence and decision. A stale incumbent requires a new contest.
- Make a repeated successful operation return its existing result without another
  champion generation. Keep database promotion authoritative if file export fails;
  retry that export independently of the promotion decision.
- The command does no training and plays no evaluation games.

Primary files: shared promotion application logic, audit service, a thin CLI entry
point, and promotion integration tests. Existing stopped-run promotion should
reuse this logic so both workflows apply identical gates.

Acceptance:

- Normal and marginal extended wins promote; ties, failed screens, incomplete or
  mismatched evidence do not. Default threshold boundaries are covered explicitly.
- Stale-incumbent attempts fail without modifying the champion.
- Repeated requests and export failures cannot produce duplicate promotions.

## F7 — Add the supervising CLI

Implementation:

- Own the sequential state machine and invoke F4–F6 as subprocesses. Start with
  a configurable 100-episode training round, await its completed checkpoint, and
  evaluate immediately before starting the next round.
- Capture the current incumbent and request matching random screening. Reject if
  the candidate exceeds allowed regression; otherwise request the initial
  head-to-head stage.
- Use the shared gate functions to choose normal promotion, extended head-to-head,
  or rejection. Request promotion after a passing normal or extended result.
  Initial scores at or below 0.5 are rejected. A score at or above 0.52 that fails
  the normal suite-win requirement remains rejected, matching existing behavior.
- After a completed decision, resume the learner from the round's training
  checkpoint regardless of whether it became champion. Do not reload an older
  champion after rejection or discard learner replay.
- Persist run/round IDs, operation IDs, effective configuration, target episode,
  learner checkpoint, captured incumbent, evaluation IDs, stage and decision using
  the existing audit infrastructure where practical. Add a migration only if the
  current records cannot represent the supervisor state reliably.
- On restart, reconcile durable child results before retrying. Recover a committed
  training checkpoint or promotion even if its child died before reporting success.
  Resume the pending stage rather than starting another training round.
- A child failure pauses progress with a clear recoverable status. Forward shutdown
  signals to the active child, wait for its graceful exit, and preserve the pending
  stage. Do not interpret an interrupted evaluation as candidate rejection.
- Keep one supervisor active for a run and guard champion writes with F6's atomic
  incumbent check. If another actor changes the champion during evaluation,
  invalidate that contest and reevaluate against the new incumbent.
- Communicate with children through explicit structured request/result data and
  exit codes. Persist authoritative artifacts in the audit store; logs are for
  people. Use subprocess argument arrays, not shell-composed commands.
- Expose round size, total training target and the existing evaluation settings.
  Record the actual gate configuration; defaults and prior-run overrides must not
  be conflated. Do not add duplicate blocking diagnostic evaluations between the
  requested stages. Handle a shorter final round at a configured total target.

Primary files: a new supervisor module, CLI registration, shared audit integration,
focused subprocess tests and `docs/SEARCH_TRAINING.md` usage/recovery instructions.

Acceptance:

- A bounded integration test exercises train → screen → initial challenge →
  optional extended challenge → promote/reject → next training round, with actual
  process boundaries and deterministic fixtures for the possible decisions.
- No training child advances to the next round before evaluation and its decision
  finish. The next round starts from the learner checkpoint, while its next contest
  uses the current champion.
- Recovery is tested after training commit, during evaluation, and after promotion
  commit but before result acknowledgement; completed work is not duplicated.
- A small real-engine scenario verifies target-board training, checkpoint resume
  and the command contracts. Export/native inference compatibility for the deeper
  model remains part of F2 verification, not a separate feature.

## CLI contracts and future queue compatibility

Each unit has a stable operation identity, explicit inputs, durable outputs and
clear success/interruption/failure status. Requests can be represented as JSON
without depending on a queue library. The supervisor owns transitions; units own
only their bounded work and validate their inputs. Runtime model and engine code
remain shared Python modules, avoiding divergent implementations behind the CLIs.

These boundaries leave room for a future queue-based supervisor and workers.
Designing message delivery, queue topology, distributed ownership, visibility
extensions and retries belongs to that later project. The present implementation
uses local child processes and the existing game server/audit database.

## Deferred work

SQS/LocalStack integration, background checkpoint discovery, concurrent pipeline
stages, mixed-board replay, channel widening, dilated convolutions, attention/global
policy features, batched multi-game MCTS, subtree reuse and distributed training.
Performance benchmarking and selecting the actual long-run settings follow the
implementation; they are not code features or authorization to start training.
