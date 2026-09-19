package grpcserver

import (
	"context"
	"testing"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
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
		GameId:   gameID,
		Player:   proto.Uint32(0),
		Position: &dotscordonv1.Coordinate{Row: 0, Column: 1},
	})
	require.NoError(t, err)
	assert.Equal(t, uint32(0), moved.GetResult().GetPlayer())
	assert.Equal(t, uint32(1), moved.GetGame().GetNextTurnBy())
	assert.Equal(t, []byte{0, 1, 0, 0, 0, 0}, moved.GetGame().GetBoard().GetCells())

	_, err = client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:   gameID,
		Player:   proto.Uint32(0),
		Position: &dotscordonv1.Coordinate{Row: 1, Column: 1},
	})
	assert.Equal(t, codes.FailedPrecondition, status.Code(err))

	_, err = client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:   gameID,
		Player:   proto.Uint32(1),
		Position: &dotscordonv1.Coordinate{Row: 0, Column: 1},
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
			GameId:   created.GetGame().GetGameId(),
			Player:   proto.Uint32(uint32(turn % 2)),
			Position: move,
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
		GameId:   created.GetGame().GetGameId(),
		Player:   proto.Uint32(0),
		Position: &dotscordonv1.Coordinate{},
	})
	require.NoError(t, err)
	assert.True(t, moved.GetGame().GetTerminal())
	assert.Equal(
		t,
		dotscordonv1.TerminationReason_TERMINATION_REASON_BOARD_FULL,
		moved.GetGame().GetTerminationReason(),
	)
}

func TestConcurrentMovesBySamePlayerApplyOnlyOnce(t *testing.T) {
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
				GameId:   created.GetGame().GetGameId(),
				Player:   proto.Uint32(0),
				Position: &dotscordonv1.Coordinate{Column: column},
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
	assert.Equal(t, uint32(1), got.GetGame().GetNextTurnBy())
}

func TestMakeMoveWaitsForInFlightMove(t *testing.T) {
	service, gameID, finishMove := newTestBlockedGame(t)
	pending := startTestCall(func() (*dotscordonv1.MakeMoveResponse, error) {
		return service.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{GameId: gameID, Player: proto.Uint32(1), Position: &dotscordonv1.Coordinate{Column: 1}})
	})
	requireCallPending(t, pending)
	first := finishMove()
	require.Equal(t, uint32(1), first.Game.NextTurnBy)
	result := <-pending
	require.NoError(t, result.err)
	require.Equal(t, uint32(0), result.response.Game.NextTurnBy)
	require.Equal(t, []byte{1, 2, 0, 0}, result.response.Game.Board.Cells)
}

func TestMakeMoveAcceptsStaleButLegalRequest(t *testing.T) {
	client := newTestClient(t, 1)
	ctx := context.Background()
	created, err := client.CreateGame(ctx, &dotscordonv1.CreateGameRequest{Rows: 2, Columns: 3})
	require.NoError(t, err)
	// Prepare this using the initial observation, then let both players move.
	stale := &dotscordonv1.MakeMoveRequest{
		GameId: created.Game.GameId, Player: proto.Uint32(created.Game.NextTurnBy),
		Position: &dotscordonv1.Coordinate{Column: 2},
	}
	for _, player := range []uint32{0, 1} {
		_, err := client.MakeMove(ctx, &dotscordonv1.MakeMoveRequest{
			GameId: created.Game.GameId, Player: proto.Uint32(player),
			Position: &dotscordonv1.Coordinate{Column: player},
		})
		require.NoError(t, err)
	}
	response, err := client.MakeMove(ctx, stale)
	require.NoError(t, err)
	require.Equal(t, uint32(0), response.Result.Player)
	require.Equal(t, uint32(1), response.Game.NextTurnBy)
	require.Equal(t, []byte{1, 2, 1, 0, 0, 0}, response.Game.Board.Cells)
}

func TestRejectedMovesPreserveProtocolState(t *testing.T) {
	client := newTestClient(t, 1)
	ctx := context.Background()
	created, err := client.CreateGame(ctx, &dotscordonv1.CreateGameRequest{Rows: 2, Columns: 3, MaxTurns: 5})
	require.NoError(t, err)
	first, err := client.MakeMove(ctx, &dotscordonv1.MakeMoveRequest{
		GameId: created.Game.GameId, Player: proto.Uint32(0), Position: &dotscordonv1.Coordinate{},
	})
	require.NoError(t, err)
	for _, test := range []struct {
		name   string
		player *uint32
		column uint32
		code   codes.Code
	}{
		{"missing player", nil, 1, codes.InvalidArgument},
		{"unknown player", proto.Uint32(2), 1, codes.InvalidArgument},
		{"same player twice", proto.Uint32(0), 1, codes.FailedPrecondition},
		{"occupied cell", proto.Uint32(1), 0, codes.InvalidArgument},
		{"outside board", proto.Uint32(1), 256, codes.InvalidArgument},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := client.MakeMove(ctx, &dotscordonv1.MakeMoveRequest{
				GameId: created.Game.GameId, Player: test.player,
				Position: &dotscordonv1.Coordinate{Column: test.column},
			})
			require.Equal(t, test.code, status.Code(err))
			current, err := client.GetGame(ctx, &dotscordonv1.GetGameRequest{GameId: created.Game.GameId})
			require.NoError(t, err)
			require.True(t, proto.Equal(first.Game, current.Game))
		})
	}
}
