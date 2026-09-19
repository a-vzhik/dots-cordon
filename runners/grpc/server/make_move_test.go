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

func TestMakeMoveUpdatesStateAndRejectsInvalidMoves(t *testing.T) {
	client := newTestClient(t, 10)

	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    2,
		Columns: 3,
	})
	require.NoError(t, err)
	gameID := created.GetGame().GetGameId()

	moved, err := client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:       gameID,
		ExpectedTurn: 0,
		Position:     &dotscordonv1.Coordinate{Row: 0, Column: 1},
	})
	require.NoError(t, err)
	assert.Equal(t, uint32(0), moved.GetResult().GetPlayer())
	assert.Equal(t, uint32(1), moved.GetGame().GetTurn())
	assert.Equal(t, uint32(1), moved.GetGame().GetCurrentPlayer())
	assert.Equal(t, []byte{0, 1, 0, 0, 0, 0}, moved.GetGame().GetBoard().GetCells())

	_, err = client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:       gameID,
		ExpectedTurn: 0,
		Position:     &dotscordonv1.Coordinate{Row: 1, Column: 1},
	})
	assert.Equal(t, codes.FailedPrecondition, status.Code(err))

	_, err = client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:       gameID,
		ExpectedTurn: 1,
		Position:     &dotscordonv1.Coordinate{Row: 0, Column: 1},
	})
	assert.Equal(t, codes.InvalidArgument, status.Code(err))
}

func TestMakeMoveReturnsCaptureAndCompactBoard(t *testing.T) {
	client := newTestClient(t, 10)
	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    5,
		Columns: 5,
	})
	require.NoError(t, err)

	moves := []*dotscordonv1.Coordinate{
		{Row: 1, Column: 2},
		{Row: 2, Column: 2},
		{Row: 2, Column: 1},
		{Row: 3, Column: 1},
		{Row: 3, Column: 2},
		{Row: 3, Column: 3},
		{Row: 2, Column: 3},
	}

	var response *dotscordonv1.MakeMoveResponse
	for turn, move := range moves {
		response, err = client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
			GameId:       created.GetGame().GetGameId(),
			ExpectedTurn: uint32(turn),
			Position:     move,
		})
		require.NoError(t, err, "turn %d", turn)
	}

	assert.Equal(t, uint32(0), response.GetResult().GetPlayer())
	assert.Equal(t, uint32(1), response.GetResult().GetScoredPoints())
	assert.Equal(t, []*dotscordonv1.Coordinate{{Row: 2, Column: 2}}, response.GetResult().GetKilledCells())
	assert.Len(t, response.GetResult().GetCordons(), 1)
	assert.Equal(t, []uint32{1, 0}, response.GetGame().GetScores())
	assert.Equal(
		t,
		byte(dotscordonv1.Cell_CELL_DEAD_PLAYER_1),
		response.GetGame().GetBoard().GetCells()[2*5+2],
	)
}

func TestBoardFullTerminationTakesPrecedence(t *testing.T) {
	client := newTestClient(t, 10)
	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:     1,
		Columns:  1,
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
		dotscordonv1.TerminationReason_TERMINATION_REASON_BOARD_FULL,
		moved.GetGame().GetTerminationReason(),
	)
}

func TestConcurrentMovesFromSameObservationApplyOnlyOnce(t *testing.T) {
	client := newTestClient(t, 10)
	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    2,
		Columns: 2,
	})
	require.NoError(t, err)

	errorsByMove := make(chan error, 2)
	for column := uint32(0); column < 2; column++ {
		go func(column uint32) {
			_, moveErr := client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
				GameId:       created.GetGame().GetGameId(),
				ExpectedTurn: 0,
				Position:     &dotscordonv1.Coordinate{Column: column},
			})
			errorsByMove <- moveErr
		}(column)
	}

	statusCounts := map[codes.Code]int{}
	for range 2 {
		statusCounts[status.Code(<-errorsByMove)]++
	}
	assert.Equal(t, 1, statusCounts[codes.OK])
	assert.Equal(t, 1, statusCounts[codes.FailedPrecondition])

	got, err := client.GetGame(context.Background(), &dotscordonv1.GetGameRequest{
		GameId: created.GetGame().GetGameId(),
	})
	require.NoError(t, err)
	assert.Equal(t, uint32(1), got.GetGame().GetTurn())
}

func TestMakeMoveRejectsBusySession(t *testing.T) {
	service, session := newTestLockedSession(t)
	request := &dotscordonv1.MakeMoveRequest{GameId: session.id, Position: &dotscordonv1.Coordinate{}}
	response, err := service.MakeMove(context.Background(), request)
	require.Nil(t, response)
	require.ErrorIs(t, err, ErrSessionBusy)
	require.True(t, session.IsAcquired(), "failed acquisition released another caller's lock")

	stored, err := service.getSession(session.id)
	require.NoError(t, err)
	require.Same(t, session, stored)
	require.False(t, session.deleted)
	require.Zero(t, session.turn)

	require.NoError(t, session.ReleaseLock())
	_, err = service.MakeMove(context.Background(), request)
	require.NoError(t, err)
	require.False(t, session.IsAcquired())
}
