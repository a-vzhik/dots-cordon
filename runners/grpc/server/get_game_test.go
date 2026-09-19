package grpcserver

import (
	"context"
	"testing"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetGameReturnsCurrentState(t *testing.T) {
	client := newTestClient(t, 10)

	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    2,
		Columns: 3,
	})
	require.NoError(t, err)
	gameID := created.GetGame().GetGameId()

	_, err = client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:       gameID,
		ExpectedTurn: 0,
		Position:     &dotscordonv1.Coordinate{Row: 0, Column: 1},
	})
	require.NoError(t, err)

	got, err := client.GetGame(context.Background(), &dotscordonv1.GetGameRequest{GameId: gameID})
	require.NoError(t, err)
	assert.Equal(t, uint32(1), got.GetGame().GetTurn())
}

func TestGetGameWaitsForInFlightMove(t *testing.T) {
	service, gameID, finishMove := newTestBlockedGame(t)
	pending := startTestCall(func() (*dotscordonv1.GetGameResponse, error) {
		return service.GetGame(context.Background(), &dotscordonv1.GetGameRequest{GameId: gameID})
	})
	requireCallPending(t, pending)
	first := finishMove()
	require.Equal(t, uint32(1), first.Game.Turn)
	result := <-pending
	require.NoError(t, result.err)
	require.Equal(t, uint32(1), result.response.Game.Turn)
	require.Equal(t, []byte{1, 0, 0, 0}, result.response.Game.Board.Cells)
}
