package grpcserver

import (
	"context"
	"sync"
	"testing"
	"time"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestDeleteGameRemovesGame(t *testing.T) {
	client := newTestClient(t, 10)

	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    2,
		Columns: 3,
	})
	require.NoError(t, err)
	gameID := created.GetGame().GetGameId()

	_, err = client.DeleteGame(context.Background(), &dotscordonv1.DeleteGameRequest{GameId: gameID})
	require.NoError(t, err)

	_, err = client.GetGame(context.Background(), &dotscordonv1.GetGameRequest{GameId: gameID})
	assert.Equal(t, codes.NotFound, status.Code(err))
}

func TestQueuedGetObservesDeletion(t *testing.T) {
	service := NewService(1)
	created, err := service.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{Rows: 2, Columns: 2})
	require.NoError(t, err)
	gate := &pauseFirstAcquisition{
		manager:  service.lock,
		acquired: make(chan struct{}),
		proceed:  make(chan struct{}),
	}
	service.lock = gate
	var resumeOnce sync.Once
	resume := func() { resumeOnce.Do(func() { close(gate.proceed) }) }
	t.Cleanup(resume)

	deletion := startTestCall(func() (*dotscordonv1.DeleteGameResponse, error) {
		return service.DeleteGame(context.Background(), &dotscordonv1.DeleteGameRequest{GameId: created.Game.GameId})
	})
	<-gate.acquired // Delete owns the lock, but has not removed the row yet.
	read := startTestCall(func() (*dotscordonv1.GetGameResponse, error) {
		return service.GetGame(context.Background(), &dotscordonv1.GetGameRequest{GameId: created.Game.GameId})
	})
	requireCallPending(t, read)
	resume()
	require.NoError(t, (<-deletion).err)
	result := <-read
	require.Nil(t, result.response)
	require.Equal(t, codes.NotFound, status.Code(result.err))
	requireLockMapEmpty(t, gate.manager.(*inProcessLock))
}

type pauseFirstAcquisition struct {
	manager  Lock
	once     sync.Once
	acquired chan struct{}
	proceed  chan struct{}
}

func (gate *pauseFirstAcquisition) TryAcquireLock(gameID string, timeout time.Duration) (AcquiredLock, error) {
	lock, err := gate.manager.TryAcquireLock(gameID, timeout)
	if err != nil {
		return nil, err
	}
	gate.once.Do(func() {
		close(gate.acquired)
		<-gate.proceed
	})
	return lock, nil
}

func TestDeleteGameWaitsForInFlightMove(t *testing.T) {
	service, gameID, finishMove := newTestBlockedGame(t)
	pending := startTestCall(func() (*dotscordonv1.DeleteGameResponse, error) {
		return service.DeleteGame(context.Background(), &dotscordonv1.DeleteGameRequest{GameId: gameID})
	})
	requireCallPending(t, pending)
	first := finishMove()
	require.Equal(t, uint32(1), first.Game.Turn)
	result := <-pending
	require.NoError(t, result.err)
	_, err := service.GetGame(context.Background(), &dotscordonv1.GetGameRequest{GameId: gameID})
	require.Equal(t, codes.NotFound, status.Code(err))
}

func TestWaitingDeleteDoesNotBlockOtherGames(t *testing.T) {
	service, gameID, finishMove := newTestBlockedGame(t)
	deletion := startTestCall(func() (*dotscordonv1.DeleteGameResponse, error) {
		return service.DeleteGame(context.Background(), &dotscordonv1.DeleteGameRequest{GameId: gameID})
	})
	requireCallPending(t, deletion)

	// Creation and moving another game must succeed before the first move ends.
	created, err := service.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{Rows: 2, Columns: 2})
	require.NoError(t, err)
	moved, err := service.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId: created.Game.GameId, Position: &dotscordonv1.Coordinate{},
	})
	require.NoError(t, err)
	require.Equal(t, uint32(1), moved.Game.Turn)

	finishMove()
	require.NoError(t, (<-deletion).err)
}
