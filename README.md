# dots-cordon

A two-player mathematical territory game and an engine for ML/self-play experiments.

## Stateful gRPC server

The gRPC runner keeps games in memory and exposes one compact protobuf API for
Go, Python, and other training clients. Start it with:

```sh
go run ./runners/grpc
```

It listens on `127.0.0.1:50051` by default. Use `--listen` to change the address
and `--max-games` to bound the number of live sessions. The server is plaintext
and intended for local/trusted networks.

The service supports this episode lifecycle:

1. `CreateGame` creates a board and returns its random `game_id`.
2. `MakeMove` applies an action from the explicitly supplied `player` and returns the new state,
   reward (`scored_points`), killed cells, and cordons.
3. `ResetGame` starts another episode with the same ID and configuration.
4. `DeleteGame` releases the in-memory session.

Every move must explicitly supply `player` (0 or 1), matching the game
state’s `next_turn_by`. Wrong-player moves return gRPC `FailedPrecondition`;
occupied or unavailable cells return `InvalidArgument`. A stale request is
accepted if its player and position are legal when the server executes it.
There is no observation-version check or stored turn counter. `max_turns` is
carried in `GameState`; zero disables it. Limits and move totals count owned
dots, including captured dots, and exclude dead empty cells.

The server stores protobuf game state directly under a keyed lock. Each move
restores a private sequential engine and publishes its resulting state.
Clients must regenerate bindings and supply `player`; legacy requests that
omit it are rejected.
`Board.cells` contains one row-major byte per cell, making it suitable for a
direct `uint8` tensor conversion. See [game.proto](api/dotscordon/v1/game.proto)
for the cell values and complete contract.

Server reflection and the standard gRPC health service are enabled. For
example, with `grpcurl` installed:

```sh
grpcurl -plaintext \
  -d '{"rows":7,"columns":7,"maxTurns":200}' \
  127.0.0.1:50051 dotscordon.v1.GameService/CreateGame
```

Games are intentionally process-local: restarting the server discards them.
Clients should set RPC deadlines and explicitly delete sessions they no longer
need.

`SimulateMove` applies a move to a supplied board snapshot using an isolated
instance of the same engine. It returns the successor and move result without
modifying a session or allocating another game ID. Search requests identify their player and carry the
turn limit in the supplied game state. Use `--quiet` on the server during search to
suppress per-capture logs. See
[search-assisted training](machinelearning/SEARCH_TRAINING.md) for the new learner.

### Regenerating Go bindings

Generated bindings are checked in. To regenerate them after editing the proto:

```sh
go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.12
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.6.2
PATH="$(go env GOPATH)/bin:$PATH" go generate ./api/dotscordon/v1
```

## CLI runner

To play through the terminal:

```sh
go run ./runners/cli --board=7x7 --player0=human --player1=random
```

The CLI starts a private gRPC server on an ephemeral loopback port and plays
through the same `GameService` API used by external agents. It shuts the server
down when the game finishes, input ends, the user quits, or an error stops the
game. CLI games continue to be written to timestamped `game-*.json` records.

The model player loads an ONNX policy into Go and runs inference in the CLI
process using [onnxruntime_go](https://github.com/yalue/onnxruntime_go).
Python is used once to export a training checkpoint; gameplay needs only the Go
binary, the exported model, and the native ONNX Runtime library.

Export the current champion (from the repository root):

```sh
uv sync --project machinelearning --extra export
machinelearning/.venv/bin/dots-cordon-export-policy \
  champion:search-warm-7x7 \
  machinelearning/checkpoints/search-warm-7x7/policy-champion.onnx
```

The exporter also accepts `.pt` paths and `checkpoint:UUID`, with
`--database-url` or `DOTS_CORDON_DATABASE_URL` to select an audit database.
It reads the database without modifying training history. Export again to play
a newly promoted champion; an exported file is a fixed snapshot.

Install the **ONNX Runtime 1.29.0** shared library for your OS and architecture
from the [official release](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0).
The pinned Go wrapper is v1.36.0 and requires that runtime API version. Building
the CLI requires Go with CGO enabled and a C compiler. On Apple Silicon, the
wrapper's module also includes the matching runtime, which can be copied locally:

```sh
go mod download github.com/yalue/onnxruntime_go
mkdir -p machinelearning/runtime
cp "$(go env GOMODCACHE)/github.com/yalue/onnxruntime_go@v1.36.0/test_data/onnxruntime_arm64.dylib" \
  machinelearning/runtime/libonnxruntime.1.29.0.dylib
```

Play from the repository root (Apple Silicon example):

```sh
go run ./runners/cli --board=7x7 \
  --player0=human --player1=model \
  --player1-weights=machinelearning/checkpoints/search-warm-7x7/policy-champion.onnx \
  --onnxruntime=machinelearning/runtime/libonnxruntime.1.29.0.dylib
```

Alternatively set `ONNXRUNTIME_SHARED_LIBRARY_PATH` to the native library path.
Player 0 moves first. Enter zero-based `row col` coordinates (for example,
`3 4`); enter `Q` to quit. Swap the player types and use `--player0-weights`
for the model to move first.

Both DQN and policy/value models use greedy action selection, without search or
exploration. The CLI reports the episode and loads the model once per game.
Choose any supported board size with `--board`, including `10x15`, `12x12`, or
`5x15`. One exported model accepts different row and column counts; no retraining
or separate export per size is required. Training dimensions are recorded as
metadata, and playing strength on other sizes depends on what the model learned.
Older fixed-size ONNX files need a one-time re-export to support other sizes.
Checkpoints trained with a turn limit are rejected at export because CLI games
use the full board. No training server or audit
database is needed during gameplay. The `agent` player type still takes manually
entered moves; use `model` for a trained opponent.

Native inference parity tests compare Go encoding, action scores, and selected
moves with checked-in PyTorch fixtures for both architectures across square and
rectangular boards, including changing dimensions within a loaded model:

```sh
ONNXRUNTIME_SHARED_LIBRARY_PATH="$PWD/machinelearning/runtime/libonnxruntime.1.29.0.dylib" \
  go test -timeout=10s ./inference ./runners ./runners/cli
```

Without the environment variable, native runtime tests skip; encoding and
move-selection tests still run. Regenerate fixtures with
`machinelearning/.venv/bin/python inference/testdata/generate.py`.

## Native desktop runner

The Ebitengine runner uses a local 10-row × 15-column game. You play player 0
and move first; player 1 uses a required exported ONNX policy. Export a model
using the CLI runner directions above (the same `dots-cordon-export-policy`
command supports this board). Older fixed-size exports must be re-exported if
not already 10×15.

Build or run from the repository root:

```sh
go build -o /tmp/dots-cordon-desktop ./runners/ebiten
go run ./runners/ebiten \
  --player1-weights=machinelearning/checkpoints/search-warm-7x7/policy-champion.onnx \
  --onnxruntime=machinelearning/runtime/libonnxruntime.1.29.0.dylib
# The built binary accepts the same flags:
/tmp/dots-cordon-desktop --player1-weights=exported.onnx
```

Building requires CGO enabled, a C compiler, and native Ebitengine desktop
platform dependencies. Gameplay requires the ONNX Runtime shared library
matching the pinned Go wrapper; see the runtime installation directions above.
Use `--onnxruntime` or set `ONNXRUNTIME_SHARED_LIBRARY_PATH`. The model is loaded
once and its board compatibility checked before opening the game window.
Inference runs on one background worker; the UI applies completed moves.
Closing the window waits for any in-flight native inference before releasing
the model. An inference error stops play.

This runner is native desktop only. A future browser runner needs its own
inference adapter; the session/controller contract has no Ebitengine or native
runtime dependency. No browser build is provided here.

Run desktop tests, including the optional real-model integration test:

```sh
DOTS_CORDON_NATIVE_TEST_MODEL="$PWD/machinelearning/checkpoints/search-warm-7x7/policy-champion.onnx" \
ONNXRUNTIME_SHARED_LIBRARY_PATH="$PWD/machinelearning/runtime/libonnxruntime.1.29.0.dylib" \
  go test -timeout=10s ./runners/ebiten/...
```

Without both environment variables, the real-model integration test skips.
Desktop view tests may require a graphical session.
