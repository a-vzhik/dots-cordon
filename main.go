package main

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/a-vzhik/dots-cordon/engine"
)

func configureLogger(logLevel slog.Level) {
	handler := slog.NewTextHandler(os.Stdout,
		&slog.HandlerOptions{
			Level: &logLevel,
		})

	slog.SetDefault(slog.New(handler))
}

func main() {
	configureLogger(slog.LevelInfo)

	players := []*engine.Player{
		{
			Score: 0,
			Color: engine.BlueColor,
		},
		{
			Score: 0,
			Color: engine.RedColor,
		},
	}
	game := engine.NewGame(
		engine.NewGameField(20, 20),
		players)

	_ = game.Move(0, 10, 9)
	_ = game.Move(1, 10, 10)
	_ = game.Move(0, 9, 10)
	_ = game.Move(1, 1, 1)
	_ = game.Move(0, 10, 11)
	_ = game.Move(1, 2, 2)
	_ = game.Move(0, 11, 10)

	slog.Info(fmt.Sprintf("Score %d : %d", game.Players[0].Score, game.Players[1].Score))
}
