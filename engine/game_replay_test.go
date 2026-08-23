package engine

import (
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

type recordedAction struct {
	Type        string      `json:"type"`
	Width       uint8       `json:"width"`
	Height      uint8       `json:"height"`
	PlayerIndex PlayerIndex `json:"playerIndex"`
	Row         uint8       `json:"row"`
	Col         uint8       `json:"col"`
}

func TestGameReplay_MoveAtFiveFive(t *testing.T) {
	game := replayRecordedGame(t)

	slog.Info("========")

	_, err := game.Move(PlayerIndex(1), 5, 5)
	require.NoError(t, err)
}

func replayRecordedGame(t *testing.T) *Game {
	t.Helper()
	filePath := filepath.Join("..", "game-2026-08-23T22-47-25.json")
	contents, err := os.ReadFile(filePath)
	require.NoError(t, err)

	var actions []recordedAction
	require.NoError(t, json.Unmarshal(contents, &actions))
	require.NotEmpty(t, actions)
	require.Equal(t, "start", actions[0].Type)

	game := NewGame(
		NewGameField(actions[0].Width, actions[0].Height),
		[]*Player{{Color: RedColor}, {Color: BlueColor}},
		NoopGameRecorder{},
	)

	for actionIdx, action := range actions[1:] {
		require.Equal(t, "move", action.Type, "action %d", actionIdx+1)
		_, err := game.Move(action.PlayerIndex, action.Row, action.Col)
		require.NoError(
			t,
			err,
			"replay action %d: player %d at (%d, %d)",
			actionIdx+1,
			action.PlayerIndex,
			action.Row,
			action.Col,
		)
	}

	return game
}
