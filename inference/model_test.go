package inference

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	pb "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

type parityCase struct {
	Game    []byte
	Encoded []float32
	Scores  []float32
	Action  int
}

func TestModelMatchesPyTorch(t *testing.T) {
	if os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH") == "" {
		t.Skip("set ONNXRUNTIME_SHARED_LIBRARY_PATH to test native inference")
	}
	dirs := []string{"testdata"}
	if dir := os.Getenv("DOTS_CORDON_PARITY_DIR"); dir != "" {
		dirs = append(dirs, dir)
	}
	for _, dir := range dirs {
		files, err := filepath.Glob(filepath.Join(dir, "*.onnx"))
		require.NoError(t, err)
		require.NotEmpty(t, files)
		for _, file := range files {
			t.Run(filepath.Base(file), func(t *testing.T) {
				model, err := LoadWeights(file, "")
				require.NoError(t, err)
				defer model.Close()
				second, err := LoadWeights(file, "")
				require.NoError(t, err)
				second.Close() // The first model must remain usable.
				second.Close()
				data, err := os.ReadFile(strings.TrimSuffix(file, ".onnx") + ".json")
				require.NoError(t, err)
				var cases []parityCase
				require.NoError(t, json.Unmarshal(data, &cases))
				var maxError float64
				for _, test := range cases {
					game := &pb.GameState{}
					require.NoError(t, proto.Unmarshal(test.Game, game))
					rows, columns := int(game.Board.Rows), int(game.Board.Columns)
					encoded, err := EncodeState(game, rows, columns)
					require.NoError(t, err)
					require.Equal(t, test.Encoded, encoded)
					scores, err := model.Infer(encoded, rows, columns)
					require.NoError(t, err)
					require.Len(t, scores, len(test.Scores))
					for i, expected := range test.Scores {
						delta := math.Abs(float64(scores[i] - expected))
						maxError = max(maxError, delta)
						require.LessOrEqual(t, delta, 1e-5+1e-5*math.Abs(float64(expected)))
					}
					move, err := SelectMove(scores, game.Board)
					require.NoError(t, err)
					require.Equal(t, test.Action, int(move.Row*game.Board.Columns+move.Column))
				}
				t.Logf("%d positions: identical moves; maximum score error %.9g", len(cases), maxError)
				_, err = model.Infer(nil, 7, 7)
				require.Error(t, err)
				model.Close()
				_, err = model.Infer(cases[0].Encoded, 7, 7)
				require.ErrorContains(t, err, "closed")
			})
		}
	}
}

func TestBoardSizeValidation(t *testing.T) {
	dynamic := &Model{info: Metadata{Version: 2, Rows: 7, Columns: 7}}
	for _, board := range [][2]int{{10, 15}, {12, 12}, {5, 15}, {255, 255}, {1, 1}} {
		require.NoError(t, dynamic.ValidateBoardSize(board[0], board[1]))
	}
	for _, board := range [][2]int{{0, 7}, {7, -1}, {256, 7}, {7, 256}} {
		require.Error(t, dynamic.ValidateBoardSize(board[0], board[1]))
		_, err := dynamic.Infer(nil, board[0], board[1])
		require.Error(t, err)
	}
	legacy := &Model{info: Metadata{Version: 1, Rows: 7, Columns: 7}}
	require.NoError(t, legacy.ValidateBoardSize(7, 7))
	require.ErrorContains(t, legacy.ValidateBoardSize(10, 15), "re-export")
}

func TestStateValidationAndMoveMask(t *testing.T) {
	game := &pb.GameState{Board: &pb.Board{Rows: 1, Columns: 6, Cells: []byte{1, 2, 3, 4, 5, 0}}, Scores: []uint32{1, 3}, NextTurnBy: 0}
	encoded, err := EncodeState(game, 1, 6)
	require.NoError(t, err)
	require.Equal(t, float32(-2.0/6), encoded[18])
	require.Equal(t, float32(4.0/6), encoded[24])
	move, err := SelectMove([]float32{10, 10, 10, 10, 10, -1}, game.Board)
	require.NoError(t, err)
	require.Equal(t, uint32(5), move.Column)
	game.Board.Cells = []byte{0, 0, 0, 0, 0, 0}
	move, err = SelectMove([]float32{0, 1, 1, 0, 0, 0}, game.Board)
	require.NoError(t, err)
	require.Equal(t, uint32(1), move.Column) // First maximum on ties.
	_, err = SelectMove([]float32{float32(math.NaN()), 0, 0, 0, 0, 0}, game.Board)
	require.Error(t, err)
	game.Terminal = true
	_, err = EncodeState(game, 1, 6)
	require.ErrorContains(t, err, "already over")
	game.Terminal = false
	game.NextTurnBy = 2
	_, err = EncodeState(game, 1, 6)
	require.Error(t, err)
	game.NextTurnBy = 0
	game.MaxTurns = 1
	_, err = EncodeState(game, 1, 6)
	require.Error(t, err)
	_, err = LoadWeights("champion:play", "")
	require.ErrorContains(t, err, "export")
}
