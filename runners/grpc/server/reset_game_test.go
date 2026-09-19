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

func TestResetGameClearsBoardAndTurn(t *testing.T) {
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

	reset, err := client.ResetGame(context.Background(), &dotscordonv1.ResetGameRequest{GameId: gameID})
	require.NoError(t, err)
	assert.Equal(t, gameID, reset.GetGame().GetGameId())
	assert.Zero(t, reset.GetGame().GetTurn())
	assert.Equal(t, []byte{0, 0, 0, 0, 0, 0}, reset.GetGame().GetBoard().GetCells())
}

func TestTerminationReasonAndReset(t *testing.T) {
	client := newTestClient(t, 10)
	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:     2,
		Columns:  2,
		MaxTurns: 1,
	})
	require.NoError(t, err)

	moved, err := client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:       created.GetGame().GetGameId(),
		ExpectedTurn: 0,
		Position:     &dotscordonv1.Coordinate{},
	})
	require.NoError(t, err)
	assert.True(t, moved.GetGame().GetTerminal())
	assert.Equal(
		t,
		dotscordonv1.TerminationReason_TERMINATION_REASON_TURN_LIMIT,
		moved.GetGame().GetTerminationReason(),
	)

	_, err = client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:       created.GetGame().GetGameId(),
		ExpectedTurn: 1,
		Position:     &dotscordonv1.Coordinate{Row: 0, Column: 1},
	})
	assert.Equal(t, codes.FailedPrecondition, status.Code(err))

	reset, err := client.ResetGame(context.Background(), &dotscordonv1.ResetGameRequest{
		GameId: created.GetGame().GetGameId(),
	})
	require.NoError(t, err)
	assert.False(t, reset.GetGame().GetTerminal())
	assert.Equal(
		t,
		dotscordonv1.TerminationReason_TERMINATION_REASON_UNSPECIFIED,
		reset.GetGame().GetTerminationReason(),
	)
}
