# Search-assisted self-play

This trainer learns entirely from its own games. A policy/value network guides
PUCT Monte Carlo tree search through the existing Go engine. Search visit
counts teach the policy; the eventual win/draw/loss (+1/0/−1), from each acting
player's perspective, teaches the value function. There are no tactical
templates, demonstrations, capture bonuses, or scripted opponents in training.

The learner updates between games and retains replay across evaluations.
Champion acceptance is separate: this command never replaces the DQN champion.

## Start a run from the existing champion

Start an updated server in a separate terminal, from the repository root:

```sh
go run ./runners/grpc --listen=127.0.0.1:50052 --max-games=4 --quiet
```

Port 50052 lets the new server run alongside an existing DQN server. The new
trainer requires `SimulateMove`; an older server reports `UNIMPLEMENTED`.

From `machinelearning/`:

```sh
uv sync
uv run dots-cordon-audit db upgrade

uv run dots-cordon-search-train \
  --server 127.0.0.1:50052 \
  --experiment search-warm-7x7 \
  --initialize-from champion:default \
  --eval-opponent champion:default \
  --episodes 5000 \
  --simulations 64 \
  --eval-every 100 --eval-games 40 --eval-simulations 64 \
  --checkpoint-dir checkpoints/search-warm-7x7
```

This copies the DQN's convolutional stem and residual blocks. The policy and
value heads, optimizer, replay, and episode counters start fresh. The old Q
head and target network are not reused. Model width/depth are inferred from
the source. Board dimensions must match; both default to 7.

`champion:default` resolves once per command. Its immutable bytes are used for
the whole run, even if the old champion loop promotes another checkpoint.
Initialization records the source reference, SHA-256, and original episode in
the saved checkpoint and audit configuration. This is a new training lineage
at episode zero, not a continuation of DQN episode numbering.

The default device is `auto`. For a controlled throughput comparison, run a
short attempt with `--device cpu` or `--device mps`. Search evaluates individual
positions, so GPU availability alone does not establish which is faster.

## Train from scratch

Omit `--initialize-from` and use a separate experiment/directory:

```sh
uv run dots-cordon-search-train \
  --server 127.0.0.1:50052 \
  --experiment search-fresh-7x7 \
  --eval-opponent champion:default \
  --episodes 5000 --simulations 64 \
  --checkpoint-dir checkpoints/search-fresh-7x7
```

Match architecture, search budget, evaluation seeds, and training budget when
comparing this with transferred features. A warm start is an initialization
experiment, not a guarantee that the new policy will initially play as well as
the DQN: its move-selection head starts untrained.

## Resume and inspect

Ctrl-C finishes the current game or evaluation, saves the endpoint, and marks
the attempt interrupted. Resume with the same rules and desired settings:

```sh
uv run dots-cordon-search-train \
  --server 127.0.0.1:50052 \
  --experiment search-warm-7x7 \
  --resume checkpoints/search-warm-7x7/search-latest.pt \
  --eval-opponent champion:default \
  --episodes 10000 --simulations 64 \
  --checkpoint-dir checkpoints/search-warm-7x7
```

`--episodes` is the total search-training episode number, not an additional
count. `--resume` restores network weights, Adam state, replay, training RNG,
and counters. The CLI learning rate applies after optimizer restoration.
Other runtime settings come from the command line; repeat non-default settings
when resuming. Changing replay capacity truncates older examples if necessary.
CPU tests verify an uninterrupted run and a resumed run produce identical
weights and replay when settings match. Bitwise agreement across devices is
not promised.

Checkpoints are `search-NNNNNNN.pt` and `search-latest.pt`. Saved files include
replay, so they are larger than the old DQN checkpoints (roughly 24 MB of replay
at the default 20,000-position capacity on 7×7, plus model/optimizer data).
Periodic evaluation saves the evaluated weights first. Replay remains live
through evaluations; failed evaluations do not cause a rollback to a champion.
A failed process can be resumed from its last committed checkpoint.

Each completed training game's moves and final scores are appended to
`games.jsonl` in the checkpoint directory. These are row-major action indices,
with board dimensions, turn limit, episode, and attempt ID. The usual audit
dashboard shows attempts, metrics, checkpoint ancestry, and evaluations.
Model metadata identifies `kind: policy_value`; no database migration is
needed beyond the existing audit schema. The policy and value losses are
stored separately as well as their sum.

Database checkpoint references (`checkpoint:UUID`) work for resume. Use
`--database-url` or `DOTS_CORDON_DATABASE_URL` as before. `--no-audit` supports
file-only runs; database references require auditing. Search checkpoints must
be used with the search commands, not the DQN champion loop/evaluators.

## Evaluation

Training periodically tests two players separately:

- `mode=policy`: the neural policy chooses greedily, without search.
- `mode=search`: the same network chooses through MCTS, with no root noise and
  the most visited move selected.

Both play paired-seat random suites, plus a frozen DQN baseline when
`--eval-opponent` is supplied. Baseline matches use four paired random opening
moves by default. The baseline is only an evaluator, not a training opponent.
Search settings are stored in each suite definition so results from different
search budgets are distinguishable. Evaluations use a separate random stream
and cannot perturb subsequent training.

For a larger, fresh-seed comparison after selecting a checkpoint:

```sh
uv run dots-cordon-search-evaluate \
  --server 127.0.0.1:50052 \
  --experiment search-warm-7x7 \
  --games 1000 --seed 20260919 --simulations 128 \
  --opponent champion:default \
  checkpoints/search-warm-7x7/search-latest.pt
```

Use `--simulations 0` to evaluate only the neural policy. Both modes report
W/D/L, match score, score difference, and per-seat counts. A better search
player is not by itself evidence of a better network-only policy. Small
development suites reveal large changes; they cannot certify tiny loss rates.

## Promote saved checkpoints in episode order

After stopping a search-training run, evaluate its saved policy networks with
the same random screening and head-to-head gates used by the DQN champion loop:

```sh
uv run dots-cordon-promote-run \
  --source-attempt SOURCE_ATTEMPT_UUID \
  --policy-from-attempt PREVIOUS_CHAMPION_LOOP_ATTEMPT_UUID \
  --server 127.0.0.1:50051 \
  --output checkpoints/search-warm-7x7/policy-champion.pt
```

`--policy-from-attempt` copies evaluation settings (including device and worker
limit) from that attempt, overriding evaluation flags. Omit it to use the shared
champion-loop defaults or explicit flags. This command evaluates **policy alone**;
it does not train or use MCTS. It visits every saved episode greater than zero,
including the final interrupted checkpoint, in episode order. Each candidate
challenges the currently registered champion, which changes immediately after a
successful promotion.

The target experiment is the source run's experiment. If it has no champion,
the run's unique frozen evaluation opponent becomes its initial champion; use
`--baseline checkpoint:UUID` to choose an explicit starting opponent. Other
experiments' champion assignments are unaffected.

The existing gates are shared in `promotion_evaluation.py`:

- Screen champion and candidate on matching random suites; reject excessive
  match-score regression. The previous run used three 1,000-game suites and a
  0.005 tolerance (the CLI default remains 0.003).
- Challenge on three fresh 1,000-game paired-seat suites; promote at combined
  match score >= 0.52 with at least two suites above 0.5.
- Only initial scores strictly between 0.5 and 0.52 receive ten fresh 1,000-game
  suites. Promote when this separate extended evaluation scores strictly above
  0.5. Initial scores <= 0.5 are rejected.
- Suites run concurrently up to `--evaluation-workers`. Seeds are reserved in
  the database so subsequent candidates use new suites.

Each candidate gets an evaluation-only audit attempt linked to its original
checkpoint, with identical stored model bytes and no invented training metrics.
The source training attempt stays interrupted. Evaluations, rejections, and
champion assignments appear in the source experiment's audit page as they finish.
The promoted checkpoint is also exported to `--output`.

Rerun the same command to continue: completed candidates are skipped; an
interrupted candidate is evaluated again with fresh suites, retaining its earlier
evidence. A changed incumbent during an incomplete attempt stops the command
instead of promoting against stale evidence. A completed promotion is retained
even if the compatibility file export subsequently fails.

## Defaults and initial limits

| Setting | Default |
| --- | ---: |
| PUCT simulations per move | 64 |
| Exploration coefficient | 1.5 |
| Root Dirichlet alpha / mixture | 0.3 / 0.25 |
| Moves sampled from search visits before greedy play | 12 |
| Replay positions | 20,000 |
| Positions before optimization | 512 |
| Batch size / updates per game | 128 / 8 |
| Learning rate | 0.0003 |
| Checkpoint / evaluation interval | 100 games |

Training samples are augmented with square-board rotations/reflections, with
actions, legal masks, and policy targets transformed together. Rectangular
boards use only dimension-preserving symmetries.

This first version runs one game and one search simulation at a time. It caches
successors within a move's search, rebuilds the tree for the next move, and uses
one RPC for each new search edge. It does not batch neural inference across
games or reuse subtrees between moves. Increase simulation budgets only after
measuring actual game throughput and learning progress. Correct small runs
establish that the pipeline works; improved playing strength requires training
and independent evaluation.

## Smoke test

Using the updated server above, this exercises training, checkpointing, and
both evaluation modes with a small board and network:

```sh
uv run dots-cordon-search-train \
  --server 127.0.0.1:50052 --no-audit \
  --rows 4 --columns 4 --channels 8 --blocks 1 --device cpu \
  --episodes 3 --simulations 8 \
  --batch-size 8 --learning-starts 8 --updates-per-episode 2 \
  --log-every 1 --checkpoint-every 1 \
  --eval-every 3 --eval-games 2 --eval-simulations 8 \
  --checkpoint-dir /tmp/dots-cordon-search-smoke
```

Python tests run with `uv run pytest`. Set `DOTS_CORDON_TEST_SERVER` to an
updated server to include real-engine search integration tests. Go simulation
tests run with `go test -timeout=10s ./runners/grpc/server` from the repository
root. They compare simulated and actual moves through captures, dead territory,
and turn-limit termination, and verify that search leaves the live game intact.
