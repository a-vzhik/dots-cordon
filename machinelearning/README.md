# Dots Cordon training

This directory contains a server-backed convolutional DQN trainer. It supports
continuous self-play and an optional mixture of games against a uniform-random
opponent. Board states are encoded relative to the player about to act, and
illegal actions are masked before exploration or inference.

The Bellman transition spans two moves: a player's placement and the
opponent's reply. Its reward is:

```text
points captured by this player - points captured by the opponent's reply
```

The last transition also receives a small win/loss bonus. This keeps the model
objective aligned with final score while retaining the server's denser capture
reward.

## Run training

From the repository root, start the game server in one terminal:

```sh
go run ./runners/grpc --max-games=100
```

In another terminal:

```sh
cd machinelearning
uv sync
uv run dots-cordon-audit db upgrade
uv run dots-cordon-train --episodes 10000
```

The default setup trains exclusively through self-play on a 7x7 board,
evaluates both seats against a random opponent every 250 episodes, and records
training history and full PyTorch checkpoints in the audit database. It also
writes atomic checkpoint files under `checkpoints/`. Every evaluation uses the
same paired random seeds with swapped seats, making checkpoint results
directly comparable. The starting policy is evaluated before training and the
strongest observed policy is also saved as `dqn-best.pt`; match score is the
primary ranking and mean score difference breaks ties.
Stop with Ctrl-C; the current episode finishes and
`checkpoints/dqn-latest.pt` is saved.

For a fast end-to-end smoke run:

```sh
uv run dots-cordon-train \
  --experiment smoke-4x4 \
  --rows 4 --columns 4 \
  --episodes 3 \
  --channels 8 --blocks 1 \
  --batch-size 4 --learning-starts 4 \
  --log-every 1 --eval-every 0 \
  --checkpoint-dir /tmp/dots-cordon-smoke
```

Resume a compatible checkpoint with:

```sh
uv run dots-cordon-train \
  --episodes 20000 \
  --resume checkpoints/dqn-latest.pt
```

`--episodes` is the total desired episode number, not an additional count.
Board dimensions and the model's channel/block counts must match the
checkpoint. Run `uv run dots-cordon-train --help` for all hyperparameters.

Checkpoints restore the network, target network, optimizer, and counters. The
replay buffer is intentionally not stored because a full buffer is roughly
hundreds of megabytes; a resumed run therefore refills a fresh replay buffer
before optimization restarts. New checkpoints also restore the exploration,
opponent, and replay-sampler random streams. Older checkpoints without random
state remain compatible and start those streams from the command-line seed.

## Durable training history

Auditing is enabled by default for training, the champion loop, and both
standalone evaluators. Initialize the schema with `dots-cordon-audit db upgrade`
before starting them. Training does not require a website or HTTP server.

Each experiment keeps a compatible board/rules configuration and its own
champion. Use `--experiment NAME` on runners to select one; the default is
`default`. For example, the 4x4 smoke run above uses a separate experiment from
normal 7x7 training.

An attempt is an independent training branch. Its checkpoints have explicit
parent links to the preceding saved state, and the attempt identifies the
checkpoint from which that branch started. Starting another branch from the
same champion creates another attempt, even when the episode range is identical.
The number of command launches does not determine the lineage.
`--resume` always starts a new attempt from the selected weights; continuing
an interrupted attempt under its original ID is not implemented yet.

The database stores the exact serialized checkpoint as a BLOB: online and target
weights, optimizer, training counters, model/board metadata, and saved random
state. The replay buffer remains excluded. Recorded checkpoints can be restored
or exported after their original `.pt` files have been removed. Evaluation
records retain exact per-seat counts, score-difference sums, seeds, and test
settings; champion-loop decisions retain their thresholds and evidence.

Ctrl-C finishes the current training episode, saves its endpoint, and records
the attempt as interrupted. A hard kill preserves only previously committed
checkpoints/results and can leave the attempt's last recorded status as
running. Automatic recovery or reconciliation after a hard kill is not yet
implemented; start a new attempt from a recorded checkpoint to continue work.

Use a database reference anywhere a runner accepts a checkpoint:

```sh
uv run dots-cordon-train \
  --resume checkpoint:CHECKPOINT_UUID --episodes 20000

uv run dots-cordon-evaluate --games 1000 champion:default

uv run dots-cordon-head-to-head --games 1000 \
  champion:default checkpoint:CHECKPOINT_UUID
```

A `checkpoint:` reference identifies immutable saved bytes. A `champion:`
reference resolves the experiment's current champion when work starts, so a
later promotion cannot change the opponent in an existing evaluation.
Existing file paths are still accepted; imported files have unknown prior
ancestry.

Set `DOTS_CORDON_DATABASE_URL` or pass `--database-url URL` to choose the SQL
database. The default SQLite file is `machinelearning/audit/training.sqlite3`.
For example, from `machinelearning/`:

```sh
export DOTS_CORDON_DATABASE_URL=sqlite:///audit/training.sqlite3
uv run dots-cordon-audit db upgrade
```

The administration CLI can inspect schema and experiment status, import an
existing file, and export exact checkpoint bytes:

```sh
uv run dots-cordon-audit db status
uv run dots-cordon-audit status --experiment default
uv run dots-cordon-audit import \
  --experiment default --checkpoint checkpoints/dqn-latest.pt
uv run dots-cordon-audit export CHECKPOINT_UUID exported-model.pt
```

Pass the administration command's `--database-url` before its subcommand, or
use the environment variable for all commands. Imports preserve available
checkpoint metadata without inventing missing parents or training scores.
When importing or bootstrapping an experiment with a turn limit, pass the same
`--max-turns` value used by its runners; the default is zero.

Keep SQLite on local disk. Use SQLite's online backup mechanism for a backup
while training is active; copying only the main file can omit WAL data.

Pass `--no-audit` for the previous file-only behavior. This mode does not update
the database champion or retain database evidence.

The implementation includes database migrations, administration commands,
runner integration, a read-only HTTP API, and a single-page training dashboard.

## Training dashboard

Build the frontend once (Node.js 22.12+ and npm are required). From
`machinelearning/`:

```sh
uv sync
cd web
npm ci
npm run build
cd ..
uv run dots-cordon-audit serve --port 8081
```

Open **http://127.0.0.1:8081/**. The page uses the server's configured database
and shows weight ancestry, champion history, attempts and training metrics,
checkpoint downloads, and all evaluation batches/suites. Episodes run
horizontally; attempts occupy separate rows. Click a checkpoint to inspect it.
Scores always describe the evaluated checkpoint, and incomplete aggregates
are labeled partial.

For champion-loop attempts, **Best in attempt** marks the highest-ranked
candidate after all screening evaluations finish: match score first, mean score
difference next, then the earlier candidate on an exact tie. It is selected even
when every candidate fails qualification or loses to the champion. Champion
status is separate. Standalone training selects its best checkpoint using its
periodic training evaluations.

The page refreshes after each completed read with a three-second interval,
pauses when hidden, and keeps the last successful data on a connection error.
You can pause updates or refresh manually. Experiment/checkpoint selection is
stored in the URL. Full configuration, provenance, policies, and exact suite
seeds are available in expandable details.

This first version fetches all metadata pages and renders them together.
Historical details are cached for up to 30 seconds; active attempts and
evaluations are refreshed each cycle. Training charts show up to 120 uniformly
sampled metric records per attempt. Large histories will need pagination or
virtualization later. Checkpoint BLOBs are fetched only through explicit
download links.

For frontend development, leave the API running on port 8081 and run
`npm run dev` in `machinelearning/web/`. Open http://127.0.0.1:5173/;
Vite proxies API requests to that server. `npm run build` regenerates TypeScript
types from the local FastAPI OpenAPI contract and writes the production bundle
to `dots_cordon_ml/audit/web/static/`. Build before making a Python distribution
to include these assets in the wheel. API-only use works without a frontend
build; `/` then returns build instructions.

Frontend checks, from `machinelearning/web/`:

```sh
npm test
npm run build
npx playwright install chromium
npm run test:browser
```

Alternatively, use an installed Google Chrome with
`PLAYWRIGHT_CHANNEL=chrome npm run test:browser`. Set
`AUDIT_LIVE_URL=http://127.0.0.1:8081` to include a read-only smoke test against
your running server. The other browser tests use synthetic API responses.

## Read API

Run this in a separate terminal from `machinelearning/`:

```sh
uv sync
uv run dots-cordon-audit serve
```

The API listens on `http://127.0.0.1:8080`. Interactive API documentation is at
`/docs`, and the typed OpenAPI contract is at `/openapi.json`. The server uses
the same default database and `DOTS_CORDON_DATABASE_URL` as the training
commands. An explicit URL goes before the subcommand:

```sh
uv run dots-cordon-audit --database-url sqlite:///audit/training.sqlite3 \
  serve --host 127.0.0.1 --port 8080
```

The API opens SQLite in read-only mode. It can run alongside training, without
connecting to the game server. `/health` returns HTTP 503 when the database is
unavailable or its schema needs migration. Apply migrations separately with
`dots-cordon-audit db upgrade`.

| Route | Response |
| --- | --- |
| `GET /health` | Database connectivity and schema compatibility. |
| `GET /api/v1/experiments` | Experiments, counts, and current champion metadata. |
| `GET /api/v1/experiments/{id}/lineage` | Checkpoints, attempt lanes, ancestry edges, boundary nodes, champion history, and branch counts. The ID may also be an experiment name, such as `default`. |
| `GET /api/v1/attempts` | Attempts filtered by `experiment_id`, `starting_checkpoint_id`, `status`, `phase`, `outcome`, `created_after`, and `created_before`. |
| `GET /api/v1/attempts/{id}` | Configuration/provenance, diagnostics, counts, and a checkpoint page with recent scores and decisions. |
| `GET /api/v1/attempts/{id}/metrics` | Bounded progress samples preserving their original training windows. |
| `GET /api/v1/checkpoints/{id}` | Checkpoint metadata, BLOB availability/hash, and evaluation summaries. |
| `GET /api/v1/checkpoints/{id}/download` | Verified checkpoint bytes, download filename, content length, SHA-256, and ETag. |
| `GET /api/v1/evaluations/{id}` | Participants, paginated exact suite definitions/results and decisions, and aggregate evidence. |

For the current training lineage:

```sh
curl http://127.0.0.1:8080/api/v1/experiments/default/lineage
curl 'http://127.0.0.1:8080/api/v1/attempts?experiment_id=default&status=running'
```

List pages return `items`, `total`, `remaining`, `limit`, `next_cursor`, and
`snapshot_at`. Pass the returned cursor unchanged with the same filters to get
the next page; omit it to poll fresh data. Cursors bound append-only history by
the initial page's timestamp and ordering key. Mutable status, flags, and
evaluation completeness are read again on each request. Default list limits
are 50, with a maximum of 200.

Nested pages have their own parameters: attempt details use
`checkpoint_limit`/`checkpoint_cursor`, checkpoint details use
`evaluation_limit`/`evaluation_cursor`, and evaluation details use
`suite_limit`/`suite_cursor` plus `decision_limit`/`decision_cursor`.
Evaluation aggregates always cover **all completed suites in the batch**,
independently of the suite page. Missing results are `null`; incomplete
aggregates have `aggregate_is_partial: true`. Scores identify the subject
checkpoint and include their numerator and denominator. Large random seeds
are decimal strings so JavaScript can preserve them exactly.

Lineage accepts `root_checkpoint_id`, `episode_min`, `episode_max`,
`attempt_limit` (default 50, maximum 100), and `checkpoint_limit` (default 200,
maximum 500). Its attempt and checkpoint pages have separate cursors. Preserve
the attempt cursor while paging checkpoints within those lanes; advance the
attempt cursor with no checkpoint cursor to load the next lanes. Checkpoint
limits apply to full nodes; immediate boundary nodes are included separately
so edges always have both endpoints. `truncated`, `remaining`, and each node's
`children_outside_slice` identify omitted history. A root inside an older
attempt can include that attempt's later checkpoints, while
`current_champion.branches.attempts` counts only attempts whose starting
checkpoint is that champion.

Metrics accept an episode range and `max_points` (default 500, maximum 2000).
They select evenly spaced stored samples, preserving the first and last
samples. `stride`, `total_samples`, and `omitted_samples` explain sampling;
the returned metrics are original values, not averages of omitted samples.

Diagnostics mark a running record `unresponsive` after five minutes without
a stored heartbeat. For attempts this also considers evaluation activity.
This is a stale-heartbeat indication, not a change to the recorded status or
proof that a long-running evaluation has stopped.

Metadata requests do not load checkpoint payloads. Downloads verify the
stored checksum and stream a temporary file, with two concurrent downloads
per server process by default (`--download-workers` changes this). Excess
downloads receive HTTP 429 with `Retry-After`; `If-None-Match` supports HTTP
304. The API has no write routes. Keep the default loopback binding for local
use; authentication and the frontend are not part of this API release.

## Early stopping

For champion continuations, enable patience-based early stopping so training
does not continue long after a policy peak:

```sh
uv run dots-cordon-train \
  --episodes 20000 \
  --resume checkpoints/dqn-champion.pt \
  --random-opponent-probability 0.50 \
  --eval-games 200 \
  --early-stop-patience 6 \
  --early-stop-min-delta 0.005 \
  --checkpoint-dir checkpoints/next-candidate \
  2>&1 | tee logs/train-next-candidate.log
```

Patience counts evaluations, not episodes. With the default evaluation
interval, six evaluations allow 1,500 episodes without a match-score
improvement of at least `0.005`. A smaller improvement can still replace
`dqn-best.pt`, but does not reset patience. Early stopping is disabled when
`--early-stop-patience` is zero. Promote a best checkpoint to the global
champion only after a separate, larger evaluation suite.

## Mixed self-play and random-opponent training

`--random-opponent-probability` selects the fraction of training episodes in
which only one side is learned and the other side chooses uniform-random legal
moves. The learner alternates seats in those games. Zero preserves pure
self-play; one trains only against random.

For a second 10,000-episode phase starting from the first run's final weights:

```sh
uv run dots-cordon-train \
  --episodes 20000 \
  --resume checkpoints/dqn-0010000.pt \
  --random-opponent-probability 0.25 \
  --checkpoint-dir checkpoints/mixed-25
```

`--episodes 20000` means train from checkpoint episode 10,000 through episode
20,000. Using a new checkpoint directory preserves every checkpoint from the
first run.

## Compare checkpoints

The standalone evaluator loads one or more checkpoints and plays every model
against the same reproducible random-opponent suite:

```sh
uv run dots-cordon-evaluate \
  --games 1000 \
  --seed 10007 \
  checkpoints/dqn-0003000.pt \
  checkpoints/dqn-0005500.pt \
  checkpoints/dqn-0006500.pt \
  checkpoints/dqn-0007250.pt \
  checkpoints/dqn-0008000.pt \
  checkpoints/dqn-0010000.pt
```

Results include overall W/D/L, match score (`win=1`, `draw=0.5`), mean score
difference, and separate statistics for playing as Player 0 and Player 1. Keep
the game count and seed unchanged when adding later checkpoints to the
comparison.

## Compare two checkpoints head to head

The head-to-head evaluator assigns one DQN to each player and swaps their
seats over paired games:

```sh
uv run dots-cordon-head-to-head \
  --games 1000 \
  --seed 20260917 \
  --opening-random-moves 4 \
  checkpoints/dqn-champion.pt \
  checkpoints/early-stop-from-11750-seed8/dqn-0012000.pt
```

Each pair starts with the same reproducible random opening, after which both
DQNs select every remaining move greedily. Four opening moves provide varied
games without introducing a random opponent. Use an even game count so every
opening is tested with the checkpoint seats swapped. Setting
`--opening-random-moves 0` plays pure greedy games, but then every game with
the same seat assignment is identical.

## Run the champion loop

The champion-loop command automates training, screening, direct challenges,
and promotion. Import the initial champion into the experiment first:

```sh
uv run dots-cordon-audit bootstrap \
  --experiment default --checkpoint checkpoints/dqn-champion.pt
```

Bootstrap records an initial champion assignment; it does not claim that the
checkpoint won an evaluation. Then run the loop:

```sh
set -o pipefail

uv run dots-cordon-champion-loop \
  --server 127.0.0.1:50051 \
  --champion checkpoints/dqn-champion.pt \
  --run-dir checkpoints/champion-loop \
  --fresh-training-rng \
  2>&1 | tee logs/champion-loop.log
```

Each round reads the current champion from the database, retains an immutable
file copy, and produces four candidates, 250 episodes apart. `--champion` names
the compatibility file exported after a database promotion. The database
champion remains authoritative even if that export fails. By default, half of
the training games use a uniform-random opponent and half use the round's frozen champion; the
learner alternates seats. Use `--training-opponent self-play` to retain the
older random/self-play mixture instead.

The champion and all four candidates are screened on three shared, fresh
1,000-game random-opponent suites. Candidates whose aggregate match score is
no more than `0.003` below the champion are challenged in screen-rank order.
Each challenge uses three different 1,000-game paired-opening suites. A
challenger is promoted when it wins at least two suites and has a combined
match score of at least `0.52`. If a challenger fails, the next
screen-qualified candidate is tested. If none qualifies or passes, the
command stops without changing the champion. Otherwise it starts another
round from the promoted checkpoint.

Random screening evaluates the champion and four candidates concurrently,
and the three suites in each head-to-head challenge also run concurrently.
The default `--evaluation-workers 5` matches the default random-screen size;
set it to `1` to disable parallel evaluation or lower it when memory is tight.
Each worker owns one model instance and one server-side game, so start the
server with `--max-games` at least as large as the worker count.

`--max-rounds 0`, the default, continues until a round produces no promotion.
Set a positive limit to cap one invocation. Every round retains
`champion-before.pt`, all candidate checkpoints, and a machine-readable
`results.json` under the run directory. Checkpoints and completed evaluation
suites are also recorded incrementally in the database, so an interrupted round
retains its completed evidence even without a final `results.json`.
Screening and head-to-head seed ranges advance in the experiment database
across rounds and command launches. Incomplete evaluations cannot promote a
candidate. A checkpoint trained after the eventual winner remains on its
original attempt's branch.

Pass `--fresh-training-rng` when starting another branch from an unchanged
champion. It keeps the checkpoint's weights, optimizer, episode count, and
environment-step count, but initializes exploration, replay sampling,
opponent selection/actions, and seat alternation from a new seed. By default,
the loop uses the current Unix timestamp in seconds, prints it, and saves
it in the round's `results.json`; pass `--training-seed NUMBER` only when you
want a reproducible value. Without `--fresh-training-rng`, resuming restores
the random streams from the checkpoint; the replay buffer still refills. The
loop increments the resolved training seed between rounds in one invocation.

## Tests

Run unit tests with:

```sh
uv run pytest
```

To include the real-server lifecycle test, start the server and run:

```sh
DOTS_CORDON_TEST_SERVER=127.0.0.1:50051 uv run pytest
```

## Audit implementation

`dots_cordon_ml/audit/service.py` exposes `AuditService`, the application entry
point shared by runners and the HTTP API. It validates checkpoint lineage,
evaluation evidence, and champion promotion. Runners pass ordinary Python
values and do not access SQL directly.

`audit/repository.py` owns SQL queries and checkpoint BLOB access.
`AuditService.reader()` exposes the bounded read views in `audit/reader.py`;
`audit/read_repository.py` implements their SQL projections and pagination.
Each read scope uses one consistent database snapshot, with batched metadata
queries and no writer lock. HTTP response models live in `audit/web/schemas.py`.
`audit/database.py` owns SQLAlchemy engine configuration, transaction boundaries,
and the Alembic lifecycle. Migrations live in `audit/migrations/versions/`.
Changing `audit/schema.py` requires a new migration containing the explicit
schema change; released migration definitions stay frozen so an old database
can upgrade reliably. Run `dots-cordon-audit db upgrade` explicitly before
starting a runner against the changed schema.

SQLite integration is tested. An optional PostgreSQL driver is available with
`uv sync --extra postgres`; configure a `postgresql+psycopg://...` database URL
to use it. PostgreSQL integration has not yet been verified, and changing the
URL does not copy data from an existing SQLite database. Automatic transfer
between database engines is not implemented.

## Input representation

The model receives five planes:

1. current player's live dots;
2. opponent's live dots;
3. all dead territory;
4. current player's normalized score difference;
5. normalized turn progress.

The checked-in Python protobuf message module is generated from
`../api/dotscordon/v1/game.proto`; `game_pb2_grpc.py` is a client-only service
stub for the five existing RPCs.
