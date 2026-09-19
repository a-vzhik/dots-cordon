package grpcserver

import (
	"context"
	"testing"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
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
		GameId:   gameID,
		Player:   proto.Uint32(0),
		Position: &dotscordonv1.Coordinate{Row: 0, Column: 1},
	})
	require.NoError(t, err)

	reset, err := client.ResetGame(context.Background(), &dotscordonv1.ResetGameRequest{GameId: gameID})
	require.NoError(t, err)
	assert.Equal(t, gameID, reset.GetGame().GetGameId())
	assert.Zero(t, reset.GetGame().GetNextTurnBy())
	assert.Equal(t, []byte{0, 0, 0, 0, 0, 0}, reset.GetGame().GetBoard().GetCells())
}

func TestResetGamePreservesConfigurationAndRecreatesRecorder(t *testing.T) {
	var recordedIDs []string
	recorder := &countingRecorder{}
	service := NewService(1, WithRecorderFactory(func(gameID string) engine.Recorder {
		recordedIDs = append(recordedIDs, gameID)
		return recorder
	}))
	ctx := context.Background()
	created, err := service.CreateGame(ctx, &dotscordonv1.CreateGameRequest{Rows: 2, Columns: 3, MaxTurns: 1})
	require.NoError(t, err)
	request := &dotscordonv1.MakeMoveRequest{Player: proto.Uint32(0), GameId: created.Game.GameId, Position: &dotscordonv1.Coordinate{}}
	_, err = service.MakeMove(ctx, request)
	require.NoError(t, err)

	reset, err := service.ResetGame(ctx, &dotscordonv1.ResetGameRequest{GameId: created.Game.GameId})
	require.NoError(t, err)
	require.Equal(t, created.Game.GameId, reset.Game.GameId)
	require.Equal(t, uint32(2), reset.Game.Board.Rows)
	require.Equal(t, uint32(3), reset.Game.Board.Columns)
	require.Equal(t, []uint32{0, 0}, reset.Game.Scores)
	require.Equal(t, []string{created.Game.GameId, created.Game.GameId}, recordedIDs)
	require.Equal(t, int32(2), recorder.starts.Load())

	moved, err := service.MakeMove(ctx, request)
	require.NoError(t, err)
	require.True(t, moved.Game.Terminal)
	require.Equal(t, dotscordonv1.TerminationReason_TERMINATION_REASON_TURN_LIMIT, moved.Game.TerminationReason)
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
		GameId:   created.GetGame().GetGameId(),
		Player:   proto.Uint32(0),
		Position: &dotscordonv1.Coordinate{},
	})
	require.NoError(t, err)
	assert.True(t, moved.GetGame().GetTerminal())
	assert.Equal(
		t,
		dotscordonv1.TerminationReason_TERMINATION_REASON_TURN_LIMIT,
		moved.GetGame().GetTerminationReason(),
	)

	_, err = client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:   created.GetGame().GetGameId(),
		Player:   proto.Uint32(1),
		Position: &dotscordonv1.Coordinate{Row: 0, Column: 1},
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

func TestResetGameWaitsForInFlightMove(t *testing.T) {
	service, gameID, finishMove := newTestBlockedGame(t)
	pending := startTestCall(func() (*dotscordonv1.ResetGameResponse, error) {
		return service.ResetGame(context.Background(), &dotscordonv1.ResetGameRequest{GameId: gameID})
	})
	requireCallPending(t, pending)
	first := finishMove()
	require.Equal(t, uint32(1), first.Game.NextTurnBy)
	result := <-pending
	require.NoError(t, result.err)
	require.Equal(t, gameID, result.response.Game.GameId)
	require.Zero(t, result.response.Game.NextTurnBy)
	require.Equal(t, []byte{0, 0, 0, 0}, result.response.Game.Board.Cells)
}
