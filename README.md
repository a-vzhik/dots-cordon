# dots-cordon

Two-player grid game where dots form cordons to capture enemy pieces.

## Docs

- [game_rules.md](game_rules.md) — player-facing rules
- [docs/ml_training.md](docs/ml_training.md) — ML / server interface sketch

## Go package: `cordon`

Cordon detection for the game server:

```go
results := cordon.Detect(board, cordon.PlayerA)
for _, r := range results {
    // r.Pocket, r.Captured, r.Wall, r.Mover
}
```

Escape uses **4-connected** movement; offensive walls use **8-connected** groups.

### Run tests

```bash
go test ./...
```
