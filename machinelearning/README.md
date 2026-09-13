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
uv run dots-cordon-train --episodes 10000
```

The default setup trains exclusively through self-play on a 7x7 board,
evaluates both seats against a random opponent every 250 episodes, and writes
atomic PyTorch checkpoints under `checkpoints/`. Every evaluation uses the
same paired random seeds with swapped seats, making checkpoint results
directly comparable. The starting policy is evaluated before training and the
strongest observed policy is also saved as `dqn-best.pt`; match score is the
primary ranking and mean score difference breaks ties.
Stop with Ctrl-C; the current episode finishes and
`checkpoints/dqn-latest.pt` is saved.

For a fast end-to-end smoke run:

```sh
uv run dots-cordon-train \
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

## Tests

Run unit tests with:

```sh
uv run pytest
```

To include the real-server lifecycle test, start the server and run:

```sh
DOTS_CORDON_TEST_SERVER=127.0.0.1:50051 uv run pytest
```

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
