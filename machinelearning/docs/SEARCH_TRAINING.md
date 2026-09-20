# Search-assisted self-play

This trainer learns entirely from its own games. A policy/value network guides
PUCT Monte Carlo tree search through the existing Go engine. Search visit
counts teach the policy; the eventual win/draw/loss (+1/0/−1), from each acting
player's perspective, teaches the value function. There are no tactical
templates, demonstrations, capture bonuses, or scripted opponents in training.

The learner updates between games and retains replay across evaluations.
Champion acceptance is separate: a training child never replaces an existing
champion. Use `dots-cordon-search-loop` for the sequential train/evaluate/promote
workflow below. The standalone `dots-cordon-search-train` remains available for
continuous training and optional diagnostic evaluations.

## Run the train/evaluate/promote supervisor

`dots-cordon-search-loop` invokes bounded training, evaluation and promotion as
separate processes. It trains 100 episodes per round in the example below,
checkpoints, screens against random, challenges the current champion, and uses
extended head-to-head only for borderline improvements. It finishes the decision
before training the next round. Promotion evaluates the **neural policy alone**;
training uses MCTS. Learner state continues after either promotion or rejection.

Start the updated Go server from the repository root:

```sh
go run ./runners/grpc --listen=127.0.0.1:50052 --max-games=4 --quiet
```

From `machinelearning/`, install CLI entrypoints and upgrade the existing audit
schema (including the bounded-operation journal):

```sh
uv sync
uv run dots-cordon-audit db upgrade
```

Create the request **once**. Adjust the source experiment, destination, target
board and budgets below before generating it. The example resolves the source
champion to an immutable ID, keeps its width, expands to at least seven blocks,
and freezes all effective settings into JSON. It assumes an audited policy/value
champion; import a file first with `dots-cordon-audit import --experiment SOURCE
--checkpoint /absolute/path/search.pt`, and use that returned checkpoint ID if
there is no registered source champion. Use `DOTS_CORDON_DATABASE_URL` consistently
for this preparation and the loop when using a non-default database.

```sh
uv run python - <<'PYTHON'
import json
from pathlib import Path
from dots_cordon_ml.audit import AuditService
from dots_cordon_ml.search_train import parse_args
from dots_cordon_ml.training import TRAINING_CONFIG_FIELDS
from dots_cordon_ml.search_loop import validate_request

source_experiment = "search-warm-7x7"
target_experiment = "search-warm-15x15"
with AuditService() as audit:
    source = audit.current_champion(audit.get_experiment(source_experiment)["id"])
    if source is None:
        raise ValueError("Source experiment has no champion")
    checkpoint = audit.get_checkpoint(source["checkpoint_id"])
    if checkpoint["model"].get("kind") != "policy_value":
        raise ValueError("This bootstrap example requires a search champion")

output = Path("checkpoints") / target_experiment
args = parse_args([
    "--server", "127.0.0.1:50052", "--device", "cpu",
    "--rows", "15", "--columns", "15", "--max-turns", "0",
    "--channels", str(checkpoint["model"]["channels"]),
    "--blocks", str(max(7, checkpoint["model"]["blocks"])),
    "--simulations", "64", "--checkpoint-every", "100",
    "--checkpoint-dir", str(output.resolve()), "--bootstrap-champion",
    "--initialize-from", "checkpoint:" + checkpoint["id"],
])
training = {key: getattr(args, key) for key in TRAINING_CONFIG_FIELDS}
training["checkpoint_dir"] = str(training["checkpoint_dir"])
evaluation = {}
for stage, suites, seed in [
    ("screening", 3, 30000001),
    ("head_to_head", 3, 40000001),
    ("extended_head_to_head", 10, 50000001),
]:
    evaluation[stage] = {
        "server": args.server, "rows": args.rows, "columns": args.columns,
        "max_turns": args.max_turns, "mode": "policy", "device": args.device,
        "games": 1000, "suites": suites, "seed": seed,
        "opening_random_moves": 0 if stage == "screening" else 4,
        "paired_seats": True, "evaluation_workers": 4, "rpc_timeout": 10.0,
    }
request = {
    "version": 1, "run_id": target_experiment + "-run-001",
    "experiment": target_experiment,
    "source": {"mode": "initialize", "checkpoint_id": checkpoint["id"]},
    "total_episode": 5000, "round_episodes": 100,
    "training_config": training, "evaluation_config": evaluation,
    "gates": {"screen_max_regression": 0.003,
              "promotion_min_match_score": 0.52, "promotion_min_suite_wins": 2},
    "work_dir": str((output / "operations").resolve()),
}
validate_request(request)
Path("search-loop-request.json").write_text(json.dumps(request, indent=2) + "\n")
PYTHON

uv run dots-cordon-search-loop \
  --request search-loop-request.json --result search-loop-result.json
```

These are example budgets, not throughput recommendations. `games` is per suite,
so the default screening plays twice the configured games/suites (champion and candidate).
For an existing champion with durable screening evidence, set
`evaluation_config.screening.reuse_champion_screening: true`, `games: 150`,
`suites: 4`, and `evaluation_workers: 4` to play only 600 candidate games in
four parallel suites. The champion’s previous result supplies the baseline.
The complete generated request is reviewable before starting. The generator
captures current defaults once; later rounds use the saved settings, not changing
CLI defaults. Gates do not inherit overrides from older experiments. Set the
regression tolerance explicitly to `0.005` if retaining that previous-run policy.
The lower-level unit JSON formats are documented below.

The request is immutable for its `run_id`. `total_episode` is an absolute learner
target; the final round is shorter when necessary. After transfer, targets start
at 100, 200, etc. To continue an already completed loop, create a new request with
a new `run_id`, a larger total, `source.mode: "resume"`, the previous result's
`learner_checkpoint_id`, and `bootstrap_champion: false`. Carry forward all other
effective settings. Resume retains network, optimizer, replay, RNG and counters.
The promoted evaluation checkpoint is a separate audit record: learner continuity
always uses the original training checkpoint even when a candidate is rejected.

### Stop and recover the supervisor

SIGINT/Ctrl-C and SIGTERM are forwarded to the active child. The supervisor waits
for it to exit gracefully and writes `status: "paused"`; training saves a partial
checkpoint after its current episode. Child failure also pauses, with the pending
stage and error in the result. Exit code 0 means the complete target and its last
evaluation decision finished; a paused loop exits 1. Fix the underlying failure,
then rerun **the same request** to resume that stage and its original target.

The database operation journal is authoritative. Child requests are saved before
launch, and completed checkpoints, evaluation batches and promotions are recovered
even if a child died before acknowledgement. Completed work is not repeated. A
partially completed evaluation is replaced with fresh suites while preserving its
earlier evidence. Recovery reconciles a pending promotion before checking whether
the champion changed; its own successful promotion is not mistaken for an external
change. A genuinely changed incumbent starts a fresh contest with the same learner
checkpoint and new seeds, without retraining that round.

One local supervisor per run and one process per child operation are enforced by
OS locks (same OS user and canonical database URL). If the supervisor is forcibly
killed while a child is still alive, a restart cannot execute that child twice:
it pauses on the child lock; retry after the old child exits. This is local process
orchestration, with no queue, distributed lease, watcher, or concurrent training.
Request/result files in `work_dir` are transport artifacts; the audit database owns
the persisted configuration, pending stage, round history and decisions. Normal
audit attempts and evaluations remain visible in the existing dashboard.

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
the source. Board dimensions default to 7; explicit transfer can change them.

`champion:default` resolves once. With auditing enabled, the diagnostic opponent's
immutable checkpoint ID is saved and reused on resume, even if the original
champion changes. A supplied `--eval-opponent` on resume does not replace that
saved diagnostic opponent. File-only runs freeze opponents for that invocation.
Initialization records the source reference, checkpoint ID when available,
SHA-256, original episode, and source/destination board and architecture in
the saved checkpoint and audit configuration. This is a new training lineage
at episode zero, not a continuation of DQN episode numbering.

The default device is `auto`. For a controlled throughput comparison, run a
short attempt with `--device cpu` or `--device mps`. Search evaluates individual
positions, so GPU availability alone does not establish which is faster.

## Transfer a search champion to a larger board

`--initialize-from` also accepts policy/value checkpoints. It copies the entire
network, including policy and value heads, while resetting optimizer, replay,
training RNG (using the new `--seed`) and counters. For example:

```sh
uv run dots-cordon-search-train \
  --server 127.0.0.1:50052 \
  --experiment search-warm-10x15 \
  --initialize-from champion:search-warm-7x7 \
  --rows 10 --columns 15 \
  --episodes 5000 --simulations 64 \
  --checkpoint-dir checkpoints/search-warm-10x15
```

Use a new experiment and checkpoint directory to preserve the original run.
Architecture is inferred from the source. Width must match; policy/value transfer
can increase depth with `--blocks` (DQN feature transfer still requires matching
depth).
Source bytes are read once, and their exact SHA-256 and transfer provenance are
saved in checkpoints and audit configuration. No source replay crosses boards.
The trainer commits `search-0000000.pt` before playing the first training game;
this contains the transferred weights before any optimization. Transfer creates
a new episode-zero lineage, not resume ancestry in the source experiment.

After transfer, resume the destination checkpoint with matching board,
architecture and `--max-turns`; `--resume` never changes those properties.

### Bootstrap the larger-board champion

Add `--bootstrap-champion` to a policy/value transfer to register its committed
episode-zero checkpoint as the target experiment's initial champion. This is an
explicit opt-in and requires auditing. It records `transferred_bootstrap`, not
promotion evidence or larger-board training. The original experiment is unchanged.
If the target already has a champion, its assignment is preserved atomically.
Recovery can also use this flag with `--resume` of the transferred episode-zero
checkpoint; later checkpoints, random starts and DQN feature transfers cannot
bootstrap through this option.

The initial policy is available as `champion:TARGET_EXPERIMENT` before training
updates begin, or by its immutable `checkpoint:UUID`. Standalone search evaluation
accepts either DQN or policy/value baselines, with matching board and `max-turns`
rules. For a 10×15 experiment, use the transferred 10×15 initial checkpoint as
the baseline; the source 7×7 checkpoint is not compatible evaluation evidence.

### Expand the search network's receptive field

Add `--blocks 7` to the larger-board transfer command to expand a shallower
search champion to seven residual blocks. The stem, existing blocks and both
heads are copied. Each extra block keeps its normally initialized first
convolution and starts with zero weights and bias in its second convolution.
Because incoming features are nonnegative, these blocks initially pass features
through unchanged: policy logits and values match the source on identical
inputs, including larger boards. The second convolution learns immediately;
the first receives gradients as the second moves away from zero.

This increases the spatial receptive field from 15×15 at three blocks to 31×31
at seven blocks. It preserves initial predictions, not a guarantee of playing
strength or training speed. Expansion uses a fresh optimizer and records the
`zero_second_convolution_v1` method and old/new block counts in transfer
provenance. Shrinking depth and changing width are rejected. Subsequent resume
must use the expanded architecture, inferred when `--blocks` is omitted.

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

Both play paired-seat random suites, plus a frozen DQN or policy/value baseline when
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

## Bounded training process

`dots-cordon-training` runs one search-training round without inline evaluation.
It requires an audited immutable source checkpoint and a JSON request with every
effective training setting. Apply the latest schema with `dots-cordon-audit db
upgrade` before using it. Database credentials belong in `--database-url` or
`DOTS_CORDON_DATABASE_URL`, never in the request.

Example `training-request.json` (replace the checkpoint ID and absolute output
path; these settings illustrate the contract, not a measured training budget):

```json
{
  "version": 1,
  "operation_id": "larger-board-round-001",
  "experiment": "larger-board",
  "source": {"mode": "initialize", "checkpoint_id": "SOURCE_CHECKPOINT_UUID"},
  "target_episode": 100,
  "config": {
    "server": "127.0.0.1:50051",
    "rows": 10, "columns": 15, "max_turns": 0,
    "seed": 7, "device": "cpu", "channels": 64, "blocks": 7,
    "learning_rate": 0.0003, "batch_size": 128,
    "replay_capacity": 20000, "learning_starts": 512,
    "updates_per_episode": 8, "simulations": 64,
    "c_puct": 1.5, "dirichlet_alpha": 0.3, "noise_fraction": 0.25,
    "temperature_moves": 12, "log_every": 10, "checkpoint_every": 100,
    "checkpoint_dir": "/absolute/path/checkpoints/larger-board",
    "rpc_timeout": 10, "bootstrap_champion": true
  }
}
```

```sh
uv run dots-cordon-training --request training-request.json --result training-result.json
```

The result includes `operation_id`, `status`, `attempt_id`, `checkpoint_id`,
`checkpoint_sha256`, reached `episode`, absolute `target_episode` and `config`.
Exit code 0 means completed, 130 means gracefully interrupted with a committed
partial checkpoint, and 1 means a command failure. The audit operation record
is authoritative; the result file is an atomic acknowledgement that can be
recreated by retrying the same request.

For the next round, give the request a new operation ID, change source to
`{"mode":"resume","checkpoint_id":"PREVIOUS_RESULT_CHECKPOINT_UUID"}`, set
`target_episode` to 200, and set `bootstrap_champion` to false. Carry forward the
complete configuration. A rejected candidate still supplies the next learner
checkpoint. Resume restores optimizer, replay, RNG and counters.

SIGINT and SIGTERM finish the current episode, commit its complete state, and
return an interrupted result unless the target was already reached. Retry the
**unchanged request and operation ID** to finish its original target. A completed
operation returns its stored result without retraining. Recovery also recognizes
a checkpoint committed before the child saved its final result. Failed and
interrupted attempts remain in the audit history; retries create new attempts
with explicit checkpoint ancestry. An operation ID cannot be reused with changed
settings. A local OS lock prevents two same-user processes from executing the
same operation against the same database URL concurrently; distributed worker
ownership is outside this command's scope.

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

### Bounded evaluation jobs

`dots-cordon-evaluation --request evaluation.json --result evaluation-result.json
--database-url sqlite:////absolute/path/audit.sqlite3` runs one policy-only stage.
It plays no training games, makes no gate decisions, and never changes the champion.
Use the same `contest` object for screening, initial challenge and extended challenge,
and a distinct stable `operation_id` for each stage:

```json
{
  "version": 1,
  "operation_id": "round-100-screen",
  "experiment": "larger-board",
  "contest": {
    "id": "round-100-contest",
    "candidate_checkpoint_id": "IMMUTABLE_LEARNER_CHECKPOINT_ID",
    "expected_assignment_id": "CAPTURED_CHAMPION_ASSIGNMENT_ID",
    "champion_checkpoint_id": "CAPTURED_CHAMPION_CHECKPOINT_ID"
  },
  "stage": "screening",
  "config": {
    "server": "127.0.0.1:50051",
    "rows": 15,
    "columns": 15,
    "max_turns": 0,
    "mode": "policy",
    "device": "cpu",
    "games": 1000,
    "suites": 3,
    "seed": 30000001,
    "opening_random_moves": 0,
    "paired_seats": true,
    "evaluation_workers": 4,
    "rpc_timeout": 10.0
  }
}
```

All shown configuration fields are required. `reuse_champion_screening` is an
optional boolean supported only on screening jobs. `games` is the even number of games per
suite, split evenly between seats. `seed` is the explicit base for durable seed
reservation; the result records the actual reserved suite seeds, exact game seeds,
seat schedules and openings. By default, screening compares both captured participants against
random on identical suites and requires zero random opening moves. For an initial
challenge, use `stage: "head_to_head"`, a new operation ID, seed `40000001` and
`opening_random_moves: 4`. For extended validation, use
`stage: "extended_head_to_head"`, another operation ID, seed `50000001` and the
chosen larger suite count. The unit does not decide whether extended validation is
warranted. Custom overlapping seed bases still receive fresh, disjoint suites.
MCTS evidence is rejected here; use `dots-cordon-search-evaluate` for diagnostics.

With `reuse_champion_screening: true`, the unit executes only the candidate batch.
It prefers the champion's own qualifying screening from its promotion attempt;
for bootstrap/manual champions it selects the earliest compatible completed batch.
If no valid durable baseline exists, the job fails before playing games. It never
silently launches a champion evaluation. The selected baseline ID is saved before
execution and remains pinned across retries.

The historical baseline must describe the exact champion, board, rules, greedy
policy and paired-seat random opponent. Its completed operation, suite definitions
and actual counts are validated against its original budget. It may contain
3 × 1,000 games while a new candidate plays 4 × 150. Their aggregate match scores
are compared with the **unchanged** `screen_max_regression` tolerance; this mode
compares independent samples rather than matched seeds. Head-to-head budgets,
thresholds and borderline extension rules are unchanged. Promotion naturally makes
the successful candidate's existing screening the next champion's baseline.

The contest creates one audit attempt and one candidate record pointing to the same
immutable model bytes as the original learner checkpoint. Every stage shares that
candidate and its captured incumbent assignment. The result distinguishes
`source_checkpoint_id` (continue training from this learner) from
`candidate_checkpoint_id` (use this contest-owned candidate for promotion evidence).
`evaluation_ids` are ordered champion then candidate for screening and contain one
candidate-versus-champion batch for either challenge stage. Results include model
kinds, full definitions, W/D/L, match score (draws count half), and per-seat/per-suite
statistics. In cached mode the champion entry has `reused: true`;
`reused_evaluation_ids` and `executed_evaluation_ids` distinguish historical evidence
from newly played batches. No historical batch is copied or reassigned to the new
contest. `evaluation.batch_results` reconstructs complete audit results for the
existing shared gate functions.

The audit database owns operation results; the JSON file is an atomic acknowledgement.
Repeat exactly the same operation request to recover a lost result file or a process
that died after all suites committed. Completed games are not replayed in that case.
An interrupted or failed partial job retains its batches and retries the whole job
with fresh reserved suites, recording replacement batch IDs. Local locks prevent
simultaneous delivery of one operation or concurrent stages within one contest.
SIGINT/SIGTERM mark interrupted evidence; retry the same request after recovery.

### Bounded promotion jobs

`dots-cordon-promotion --request promotion.json --result promotion-result.json
--database-url sqlite:////absolute/path/audit.sqlite3` validates completed evaluation
evidence and applies the shared saved-run gates. It plays no games and never trains.
Use the contest-owned candidate and attempt returned by the evaluation unit:

```json
{
  "version": 1,
  "operation_id": "round-100-promotion",
  "experiment": "larger-board",
  "attempt_id": "EVALUATION_CONTEST_ATTEMPT_ID",
  "candidate_checkpoint_id": "EVALUATION_CONTEST_CANDIDATE_ID",
  "expected_assignment_id": "CAPTURED_CHAMPION_ASSIGNMENT_ID",
  "evidence": {
    "champion_screening": "CHAMPION_RANDOM_EVALUATION_ID",
    "candidate_screening": "CANDIDATE_RANDOM_EVALUATION_ID",
    "initial_head_to_head": "INITIAL_CHALLENGE_EVALUATION_ID",
    "extended_head_to_head": null
  },
  "config": {
    "rows": 15, "columns": 15, "max_turns": 0, "mode": "policy",
    "screen_max_regression": 0.003,
    "promotion_min_match_score": 0.52, "promotion_min_suite_wins": 2,
    "opening_random_moves": 4,
    "screen_games": 1000, "screen_suites": 3,
    "screen_seeds": [30000001, 30000002, 30000003],
    "head_to_head_games": 1000, "head_to_head_suites": 3,
    "head_to_head_seeds": [40000001, 40000002, 40000003],
    "extended_head_to_head_games": 1000, "extended_head_to_head_suites": 10,
    "extended_head_to_head_seeds": []
  }
}
```

All fields are required. Seed lists are the **actual reserved** `suite_seeds` from
evaluation results, not the requested base seeds. For extended validation supply
its evaluation ID and actual seeds; otherwise use `null` and an empty list.
Evidence must match the requested participants, contest, policy mode, board,
turn limit, game counts, paired seats, openings, full game-seed definitions and
completed evaluation generation. Challenge stages must have independent seeds.
The command also validates immutable checkpoint contents and screening provenance.
For cached screening, add `reuse_champion_screening: true` to promotion `config`.
The supervisor forwards this automatically. `screen_games`, `screen_suites` and
`screen_seeds` then describe the candidate batch; `champion_screening` must be the
exact historical baseline pinned by that candidate's completed screening operation.
Without the flag, the original requirement for one matched screening operation and
identical suite definitions remains in force.

The result has `status: "completed"` and `decision: "promoted"`, `"rejected"` or
`"extended_required"`; promoted results include the committed `assignment`.
Normal promotion requires the configured match score and suite wins (defaults
0.52 and two of three suites). Only a qualified initial score strictly above 0.5
and below the normal threshold can use extended evidence. Extended suites must
score strictly above 0.5, however small the advantage. Their combined result is
assessed separately from the initial suites. An `extended_required` result leaves
the contest open: run extended evaluation and submit a **new operation ID** with
that additional evidence. A failed screen or challenge is a completed rejection;
invalid evidence or a changed incumbent is an error, not a rejection.

Exit 0 means a completed decision; exit 1 indicates failure, and 130 interruption.
The database result is authoritative. Retrying an unchanged successful operation
returns the existing decision without creating another champion generation, even
if the process died after promotion but before recording its result. Reusing an
operation ID with changed evidence or settings is rejected. Champion writes use
an atomic captured-incumbent check; a changed champion requires a fresh contest.

Optional `--output promoted.pt` atomically exports the promoted checkpoint after
the database decision is committed. Export failure makes the command fail but
leaves promotion committed and recoverable; retry the identical request with the
same or a corrected output path. The export path is outside the immutable request.
A completed rejected operation does not export a model. Export and result files
can always be repaired without replaying evaluation or creating another promotion.
