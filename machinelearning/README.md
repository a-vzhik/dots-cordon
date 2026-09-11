# Dots Cordon self-play baseline

This directory contains a runnable first training loop backed by the stateful
gRPC game server. One convolutional DQN controls both players. Board states are
encoded relative to the player about to act, and illegal actions are masked
before exploration or inference.

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

The default setup trains on a 7x7 board, evaluates both seats against a random
opponent every 250 episodes, and writes atomic PyTorch checkpoints under
`checkpoints/`. Stop with Ctrl-C; the current episode finishes and
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
