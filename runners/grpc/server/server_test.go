package grpcserver

import (
	"context"
	"net"
	"testing"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

func TestGameLifecycle(t *testing.T) {
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

	got, err := client.GetGame(context.Background(), &dotscordonv1.GetGameRequest{GameId: gameID})
	require.NoError(t, err)
	assert.Equal(t, uint32(1), got.GetGame().GetTurn())

	reset, err := client.ResetGame(context.Background(), &dotscordonv1.ResetGameRequest{GameId: gameID})
	require.NoError(t, err)
	assert.Equal(t, gameID, reset.GetGame().GetGameId())
	assert.Zero(t, reset.GetGame().GetTurn())
	assert.Equal(t, []byte{0, 0, 0, 0, 0, 0}, reset.GetGame().GetBoard().GetCells())

	_, err = client.DeleteGame(context.Background(), &dotscordonv1.DeleteGameRequest{GameId: gameID})
	require.NoError(t, err)

	_, err = client.GetGame(context.Background(), &dotscordonv1.GetGameRequest{GameId: gameID})
	assert.Equal(t, codes.NotFound, status.Code(err))
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

func newTestClient(t *testing.T, maxGames int) dotscordonv1.GameServiceClient {
	t.Helper()

	listener := bufconn.Listen(1024 * 1024)
	server := grpc.NewServer()
	dotscordonv1.RegisterGameServiceServer(server, NewService(maxGames))

	serveErrors := make(chan error, 1)
	go func() {
		serveErrors <- server.Serve(listener)
	}()

	connection, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return listener.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(t, err)

	t.Cleanup(func() {
		require.NoError(t, connection.Close())
		server.Stop()
		require.NoError(t, listener.Close())
		assert.NoError(t, <-serveErrors)
	})

	return dotscordonv1.NewGameServiceClient(connection)
}
