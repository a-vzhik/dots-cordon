package opponent

import (
	"os"
	"testing"

	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/a-vzhik/dots-cordon/inference"
	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/session"
	"github.com/stretchr/testify/require"
)

func TestSnapshotConversion(t *testing.T) {
	s := session.New().Snapshot()
	s.Dots[0][0] = engine.Dot{}
	s.Dots[0][1] = engine.Dot{Owned: true, Owner: 0}
	s.Dots[0][2] = engine.Dot{Owned: true, Owner: 1}
	s.Dots[0][3] = engine.Dot{Killed: true}
	s.Dots[0][4] = engine.Dot{Owned: true, Owner: 0, Killed: true}
	s.Dots[0][5] = engine.Dot{Owned: true, Owner: 1, Killed: true}
	s.Scores = [2]uint32{2, 3}
	s.NextPlayer = 1
	game, err := toProto(s)
	require.NoError(t, err)
	require.Equal(t, []byte{0, 1, 2, 3, 4, 5}, game.Board.Cells[:6])
	require.Equal(t, uint32(10), game.Board.Rows)
	require.Equal(t, uint32(15), game.Board.Columns)
	require.Equal(t, []uint32{2, 3}, game.Scores)
	encoded, err := inference.EncodeState(game, session.Rows, session.Columns)
	require.NoError(t, err)
	require.Equal(t, float32(1), encoded[2])
	require.Equal(t, float32(1), encoded[150+1])
	require.Equal(t, float32(1), encoded[300+3])
	require.Equal(t, float32(1.0/150), encoded[450])
	require.Equal(t, float32(4.0/150), encoded[600])
	s.Dots[0][1].Owner = 1
	s.Scores[0] = 100
	require.Equal(t, byte(1), game.Board.Cells[1])
	require.Equal(t, uint32(2), game.Scores[0])
	s.Terminal = true
	game, err = toProto(s)
	require.NoError(t, err)
	require.True(t, game.Terminal)
}

func TestRejectMalformedSnapshot(t *testing.T) {
	s := session.New().Snapshot()
	s.Dots = s.Dots[:1]
	_, err := toProto(s)
	require.Error(t, err)
	s = session.New().Snapshot()
	s.Dots[0] = nil
	_, err = toProto(s)
	require.Error(t, err)
	s = session.New().Snapshot()
	s.Dots[0][0] = engine.Dot{Owned: true, Owner: 2}
	_, err = toProto(s)
	require.Error(t, err)
}

func TestNativeChampionAtDesktopSize(t *testing.T) {
	path := os.Getenv("DOTS_CORDON_NATIVE_TEST_MODEL")
	if path == "" || os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH") == "" {
		t.Skip("set DOTS_CORDON_NATIVE_TEST_MODEL and ONNXRUNTIME_SHARED_LIBRARY_PATH for native integration")
	}
	model, err := Load(path, "")
	require.NoError(t, err)
	defer model.Close()
	game := session.New()
	require.NoError(t, game.Move(0, 0))
	// Reuse one native model across successive human/AI turns.
	for range 3 {
		move, err := model.Move(game.Snapshot())
		require.NoError(t, err)
		require.NoError(t, game.Move(move.Row, move.Col))
		snapshot := game.Snapshot()
		require.Equal(t, engine.PlayerIndex(0), snapshot.NextPlayer)
		moved := false
		for row, dots := range snapshot.Dots {
			for col, dot := range dots {
				if !dot.Owned && !dot.Killed {
					require.NoError(t, game.Move(uint8(row), uint8(col)))
					moved = true
					break
				}
			}
			if moved {
				break
			}
		}
	}
	model.Close()
	_, err = model.Move(game.Snapshot())
	require.ErrorContains(t, err, "closed")
}
