package engine

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type recordedAction struct {
	Type        string      `json:"type"`
	Width       uint8       `json:"width"`
	Height      uint8       `json:"height"`
	PlayerIndex PlayerIndex `json:"playerIndex"`
	Row         uint8       `json:"row"`
	Col         uint8       `json:"col"`
	Result      MoveResult  `json:"result"`
}

func TestGameReplay_FourCaptures(t *testing.T) {
	_, captureMoveCount := replayRecordedGameAndCountCaptures(
		t,
		"game-2026-08-29T14-35-29.json",
	)
	assert.Equal(t, 4, captureMoveCount)
}

func TestGameReplay_CompleteGame1(t *testing.T) {
	_, captureMoveCount := replayRecordedGameAndCountCaptures(
		t,
		"game-2026-08-29T15-25-12.json",
	)
	assert.Equal(t, 3, captureMoveCount)
}

func TestGameReplay_CompleteGame2(t *testing.T) {
	_, captureMoveCount := replayRecordedGameAndCountCaptures(
		t,
		"game-2026-08-29T15-49-00.json",
	)
	assert.Equal(t, 2, captureMoveCount)
}

func TestGameReplay_NoCaptures(t *testing.T) {
	_, captureMoveCount := replayRecordedGameAndCountCaptures(
		t,
		"game-2026-08-29T15-32-15.json",
	)
	assert.Equal(t, 0, captureMoveCount)
}

func TestGameReplay_WrongCaptureRegression(t *testing.T) {
	_, captureMoveCount := replayRecordedGameAndCountCaptures(
		t,
		"game-2026-08-30T14-36-49.json",
	)
	assert.Equal(t, 4, captureMoveCount)
}

func TestGameReplay_MoveAtFiveFive(t *testing.T) {
	game, _ := replayRecordedGameAndCountCaptures(t, "game-2026-08-23T22-47-25.json")

	slog.Info("========")

	_, err := game.Move(PlayerIndex(1), 5, 5)
	require.NoError(t, err)
}

func TestGameReplay_MissedCounterCaptureRegression(t *testing.T) {
	_, totalCaptures := replayRecordedGameAndCountCaptures(t, "game-2026-08-30T16-00-39.json")

	assert.Equal(t, 2, totalCaptures)
}

func replayRecordedGameAndCountCaptures(t *testing.T, fileName string) (*Game, int) {
	t.Helper()
	filePath := filepath.Join("testdata", fileName)
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

	captureMoveCount := 0
	for actionIdx, action := range actions[1:] {
		require.Equal(t, "move", action.Type, "action %d", actionIdx+1)
		slog.Info(fmt.Sprintf("Move player %d: %d, %d", action.PlayerIndex, action.Row, action.Col))
		result, err := game.Move(action.PlayerIndex, action.Row, action.Col)

		slog.Info(game.GameField.ToString())

		require.NoError(
			t,
			err,
			"replay action %d: player %d at (%d, %d)",
			actionIdx+1,
			action.PlayerIndex,
			action.Row,
			action.Col,
		)
		assert.Equal(
			t,
			&action.Result,
			result,
			"replay action %d: player %d at (%d, %d)",
			actionIdx+1,
			action.PlayerIndex,
			action.Row,
			action.Col,
		)
		if result.ScoredPoints > 0 {
			captureMoveCount++
		}
	}

	return game, captureMoveCount
}
