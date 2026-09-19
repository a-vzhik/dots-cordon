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

func TestGetGameRejectsBusySession(t *testing.T) {
	service, session := newTestLockedSession(t)
	request := &dotscordonv1.GetGameRequest{GameId: session.id}
	response, err := service.GetGame(context.Background(), request)
	require.Nil(t, response)
	require.ErrorIs(t, err, ErrSessionBusy)
	require.True(t, session.IsAcquired(), "failed acquisition released another caller's lock")

	stored, err := service.getSession(session.id)
	require.NoError(t, err)
	require.Same(t, session, stored)
	require.False(t, session.deleted)
	require.Zero(t, session.turn)

	require.NoError(t, session.ReleaseLock())
	_, err = service.GetGame(context.Background(), request)
	require.NoError(t, err)
	require.False(t, session.IsAcquired())
}
