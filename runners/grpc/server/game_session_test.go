package grpcserver

import (
	"context"
	"testing"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

func TestSessionOperationsRequireLock(t *testing.T) {
	operations := map[string]func(*testing.T, *gameSession) error{
		"move": func(t *testing.T, session *gameSession) error {
			response, err := session.move(&dotscordonv1.Coordinate{})
			require.Nil(t, response)
			return err
		},
		"snapshot": func(t *testing.T, session *gameSession) error {
			response, err := session.snapshot()
			require.Nil(t, response)
			return err
		},
		"reset": func(t *testing.T, session *gameSession) error {
			return session.reset()
		},
	}
	for name, operation := range operations {
		t.Run(name, func(t *testing.T) {
			// Missing locks and uninitialized state must return an error before
			// an operation tries to access the game.
			require.ErrorIs(t, operation(t, &gameSession{}), ErrLockNotAcquired)
			require.ErrorIs(t, operation(t, &gameSession{Lock: &inProcessLock{}}), ErrLockNotAcquired)

			session, err := newGameSession("test", 2, 2, 0, nil)
			require.NoError(t, err)
			require.False(t, session.IsAcquired())
			require.True(t, session.TryAcquireLock())
			before, err := session.move(&dotscordonv1.Coordinate{Row: 1, Column: 1})
			require.NoError(t, err)
			require.NoError(t, session.ReleaseLock())

			require.ErrorIs(t, operation(t, session), ErrLockNotAcquired)

			require.True(t, session.TryAcquireLock())
			defer session.ReleaseLock()
			after, err := session.snapshot()
			require.NoError(t, err)
			require.True(t, proto.Equal(before.Game, after), "unlocked operation changed state")
		})
	}
}

func TestSessionOperationsWithLock(t *testing.T) {
	session, err := newGameSession("test", 2, 2, 0, nil)
	require.NoError(t, err)
	require.True(t, session.TryAcquireLock())
	defer session.ReleaseLock()

	moved, err := session.move(&dotscordonv1.Coordinate{})
	require.NoError(t, err)
	require.Equal(t, uint32(1), moved.Game.Turn)

	snapshot, err := session.snapshot()
	require.NoError(t, err)
	require.True(t, proto.Equal(moved.Game, snapshot))

	require.NoError(t, session.reset())
	reset, err := session.snapshot()
	require.NoError(t, err)
	require.Zero(t, reset.Turn)
	require.Equal(t, []byte{0, 0, 0, 0}, reset.Board.Cells)
}

func newTestLockedSession(t *testing.T) (*Service, *gameSession) {
	t.Helper()
	service := NewService(1)
	created, err := service.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows: 2, Columns: 2,
	})
	require.NoError(t, err)
	session, err := service.getSession(created.Game.GameId)
	require.NoError(t, err)
	require.True(t, session.TryAcquireLock())
	t.Cleanup(func() {
		if session.IsAcquired() {
			require.NoError(t, session.ReleaseLock())
		}
	})
	return service, session
}
