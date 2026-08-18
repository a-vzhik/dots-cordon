# ML training interface

This document describes how a Python training stack integrates with the Go
game server. The server owns rules and legality; the trainer owns models and
self-play.

## Responsibilities

| Component | Owns |
|-----------|------|
| Go server | Board state, turns, cordon detection, captures, dead territory, score |
| Python trainer | Self-play, replay buffer, network training, ONNX export |

## Game server API (sketch)

### Create game

```
POST /games
{ "width": 20, "height": 20, "first_player": "A" }

→ { "game_id": "..." }
```

### Get state (current player's perspective)

```
GET /games/{id}/state?viewer=A

→ {
  "width": 20,
  "height": 20,
  "to_move": "A",
  "scores": { "A": 3, "B": 1 },
  "cells": [ ... ],               // row-major, y-down
  "legal_moves": [[x, y], ...],
  "last_move": {
    "player": "B",
    "x": 4,
    "y": 7
  },
  "last_turn": {
    "move": { "player": "B", "x": 4, "y": 7 },
    "cordons": [ ... ],
    "points_scored": 2
  },
  "terminal": false,
  "winner": null
}
```

For training, an in-process Go library call is faster than HTTP, but the JSON
shape should stay the same.

## Tensor layout (20×20 default)

Always encode from the **current player's** point of view.

| Channel | Value |
|---------|-------|
| 0 | My live dots |
| 1 | Enemy live dots |
| 2 | My dead territory |
| 3 | Enemy dead territory |
| 4 | Legal-move mask |
| 5 | Plane of 1s (side to move) |
| 6 | My last move (1 at cell) |
| 7 | Enemy last move (1 at cell) |

Action space: `action = y * width + x`. Mask illegal cells before softmax.

## Rewards

Per step from the current player's view:

1. `+N` when you capture `N` enemy dots this turn.
2. `-N` when the opponent captures `N` dots on their turn.
3. Terminal: `+1` win, `-1` loss (or score difference if using a fixed turn limit).

Value head target: final outcome from the acting player's perspective.

## Training loop (AlphaZero-lite)

1. Play a game: current policy (+ optional MCTS) vs itself or a checkpoint pool.
2. At each ply, store `(state_tensor, policy_target, z)` where `z` is the game
   outcome from that player's view.
3. Train policy head (cross-entropy) and value head (MSE).
4. Export ONNX for Go inference at play time.

## Inference in Go

1. Train in PyTorch.
2. Export `model.onnx`.
3. Load with onnxruntime-go in the game client or server for AI moves.
4. Same tensor layout as training.

## First milestones

1. Random vs random through the server → verify JSON and tensor encoding.
2. Small ResNet beats random > 90%.
3. Add MCTS → self-play 10k games.
4. Human vs AI in the Go client using ONNX.
