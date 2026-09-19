package grpcserver

import (
	"context"
	"testing"

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
