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
2. `MakeMove` applies one action for `current_player` and returns the new state,
   reward (`scored_points`), killed cells, and cordons.
3. `ResetGame` starts another episode with the same ID and configuration.
4. `DeleteGame` releases the in-memory session.

Every move includes `expected_turn`, copied from the observation it was based
on. A stale or duplicate move is rejected with gRPC `FailedPrecondition`.
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
modifying a session or allocating another game ID. Search clients supply the
same turn limit as live play. Use `--quiet` on the server during search to
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
