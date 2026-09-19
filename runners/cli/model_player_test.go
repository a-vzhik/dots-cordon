package main

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/runners"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNativeModelAcceptsDifferentBoardSizes(t *testing.T) {
	if os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH") == "" {
		t.Skip("set ONNXRUNTIME_SHARED_LIBRARY_PATH to test native inference")
	}
	for _, board := range [][2]uint8{{10, 15}, {12, 12}, {5, 15}} {
		t.Run(fmt.Sprint(board), func(t *testing.T) {
			model, info, err := startModelPlayer(runners.GameOptions{
				BoardRows: board[0], BoardCols: board[1],
				Weights: [2]string{"../../inference/testdata/policy_value.onnx", ""},
			}, 0)
			require.NoError(t, err)
			defer model.Close()
			assert.Equal(t, 3, info.Rows) // Trained on 3x4, played on the chosen board.
			cells := bytes.Repeat([]byte{1}, int(board[0])*int(board[1]))
			cells[len(cells)-1] = 0
			move, err := model.Move(&dotscordonv1.GameState{
				Board:  &dotscordonv1.Board{Rows: uint32(board[0]), Columns: uint32(board[1]), Cells: cells},
				Scores: []uint32{0, 0}, NextTurnBy: 1,
			})
			require.NoError(t, err)
			assert.Equal(t, uint32(board[0]-1), move.Row)
			assert.Equal(t, uint32(board[1]-1), move.Column)
		})
	}
}

type testModel struct {
	seat  uint32
	calls int
}

func (model *testModel) Move(game *dotscordonv1.GameState) (*dotscordonv1.Coordinate, error) {
	if game.GetNextTurnBy() != model.seat {
		return nil, fmt.Errorf("wrong model seat")
	}
	model.calls++
	for i, cell := range game.Board.Cells {
		if cell == 0 {
			return &dotscordonv1.Coordinate{Column: uint32(i)}, nil
		}
	}
	return nil, fmt.Errorf("no empty cell")
}

func TestModelPlaysEitherSeatThroughCLI(t *testing.T) {
	for _, seat := range []uint32{0, 1} {
		t.Run(fmt.Sprint(seat), func(t *testing.T) {
			client, game := newTestGame(t, 1, 2)
			players := [2]runners.PlayerType{runners.Human, runners.Human}
			players[seat] = runners.Model
			model := &testModel{seat: seat}
			var models [2]moveSelector
			models[seat] = model
			input := "0 1\n"
			if seat == 1 {
				input = "0 0\n"
			}
			var output bytes.Buffer
			err := runGame(client, game, players, bufio.NewScanner(strings.NewReader(input)), &output, models)
			require.NoError(t, err)
			assert.Equal(t, 1, model.calls)
			assert.Contains(t, output.String(), "Model move:")
			assert.Contains(t, output.String(), "Game over.")
		})
	}
}
