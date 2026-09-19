package grpcserver

import (
	"context"
	"testing"
	"time"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestCreateGameInitialState(t *testing.T) {
	client := newTestClient(t, 10)

	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    2,
		Columns: 3,
	})
	require.NoError(t, err)
	require.NotNil(t, created.GetGame())
	gameID := created.GetGame().GetGameId()
	assert.Len(t, gameID, 32)
	assert.Equal(t, uint32(2), created.GetGame().GetBoard().GetRows())
	assert.Equal(t, uint32(3), created.GetGame().GetBoard().GetColumns())
	assert.Equal(t, []byte{0, 0, 0, 0, 0, 0}, created.GetGame().GetBoard().GetCells())
	assert.Equal(t, []uint32{0, 0}, created.GetGame().GetScores())
	assert.Zero(t, created.GetGame().GetTurn())
	assert.Zero(t, created.GetGame().GetCurrentPlayer())
}

func TestConcurrentCreateGameHonorsLimit(t *testing.T) {
	const limit = 3
	const attempts = 16
	service := NewService(limit)
	for range 2 {
		results := make(chan testCallResult[*dotscordonv1.CreateGameResponse], attempts)
		start := make(chan struct{})
		for range attempts {
			go func() {
				<-start
				response, err := service.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{Rows: 2, Columns: 2})
				results <- testCallResult[*dotscordonv1.CreateGameResponse]{response: response, err: err}
			}()
		}
		close(start)
		var gameIDs []string
		for range attempts {
			result := <-results
			if result.err == nil {
				gameIDs = append(gameIDs, result.response.Game.GameId)
			} else {
				require.Equal(t, codes.ResourceExhausted, status.Code(result.err))
			}
		}
		require.Len(t, gameIDs, limit)
		deletions := make(chan error, limit)
		for _, gameID := range gameIDs {
			go func() {
				_, err := service.DeleteGame(context.Background(), &dotscordonv1.DeleteGameRequest{GameId: gameID})
				deletions <- err
			}()
		}
		for range limit {
			require.NoError(t, <-deletions)
		}
	}
}

func TestFailedCreateReturnsReservedSlot(t *testing.T) {
	service := NewService(1)
	service.lock = unavailableLock{}
	request := &dotscordonv1.CreateGameRequest{Rows: 2, Columns: 2}
	_, err := service.CreateGame(context.Background(), request)
	require.ErrorIs(t, err, ErrLockTimeout)

	service.lock = &inProcessLock{}
	_, err = service.CreateGame(context.Background(), request)
	require.NoError(t, err)
}

type unavailableLock struct{}

func (unavailableLock) TryAcquireLock(string, time.Duration) (AcquiredLock, error) {
	return nil, ErrLockTimeout
}

func TestCreateGameValidatesDimensionsAndLimit(t *testing.T) {
	client := newTestClient(t, 1)

	_, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    256,
		Columns: 5,
	})
	assert.Equal(t, codes.InvalidArgument, status.Code(err))

	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    5,
		Columns: 5,
	})
	require.NoError(t, err)

	_, err = client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    5,
		Columns: 5,
	})
	assert.Equal(t, codes.ResourceExhausted, status.Code(err))

	_, err = client.DeleteGame(context.Background(), &dotscordonv1.DeleteGameRequest{
		GameId: created.GetGame().GetGameId(),
	})
	require.NoError(t, err)

	_, err = client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    5,
		Columns: 5,
	})
	require.NoError(t, err)
}
