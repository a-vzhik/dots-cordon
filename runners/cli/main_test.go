package main

import (
	"bufio"
	"bytes"
	"context"
	"net"
	"strings"
	"testing"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/runners"
	grpcserver "github.com/a-vzhik/dots-cordon/runners/grpc/server"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

func TestRunPrintsUsageForMissingOptions(t *testing.T) {
	var output bytes.Buffer
	var errorOutput bytes.Buffer

	exitCode := run(nil, strings.NewReader(""), &output, &errorOutput)

	assert.Equal(t, 2, exitCode)
	assert.Empty(t, output.String())
	assert.Contains(t, errorOutput.String(), "missing required options: --board, --player0, --player1")
	assert.Contains(t, errorOutput.String(), "Usage:")
}

func TestRunPrintsUsageForInvalidOptions(t *testing.T) {
	var output bytes.Buffer
	var errorOutput bytes.Buffer

	exitCode := run(
		[]string{"--board=invalid", "--player0=agent", "--player1=human"},
		strings.NewReader(""),
		&output,
		&errorOutput,
	)

	assert.Equal(t, 2, exitCode)
	assert.Empty(t, output.String())
	assert.Contains(t, errorOutput.String(), `invalid board "invalid"`)
	assert.Contains(t, errorOutput.String(), "Usage:")
}

func TestRunPrintsUsageForHelp(t *testing.T) {
	var output bytes.Buffer
	var errorOutput bytes.Buffer

	exitCode := run([]string{"--help"}, strings.NewReader(""), &output, &errorOutput)

	assert.Equal(t, 0, exitCode)
	assert.Contains(t, output.String(), "Usage:")
	assert.Empty(t, errorOutput.String())
}

func TestMakeHumanMoveIdentifiesPlayer(t *testing.T) {
	client, game := newTestGame(t, 2, 2)
	firstMove, err := client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:       game.GetGameId(),
		ExpectedTurn: game.GetTurn(),
		Position:     &dotscordonv1.Coordinate{},
	})
	require.NoError(t, err)

	input := bufio.NewScanner(strings.NewReader("0 1\n"))
	var output bytes.Buffer
	_, move, err := makeHumanMove(client, firstMove.GetGame(), input, &output)

	require.NoError(t, err)
	assert.Equal(t, uint32(0), move.GetRow())
	assert.Equal(t, uint32(1), move.GetColumn())
	assert.Equal(
		t,
		"Player 1 move (<row> <col>) OR <Q> to finish the game: ",
		output.String(),
	)
}

func TestRunGameAcceptsAgentPlayerThroughGRPC(t *testing.T) {
	client, game := newTestGame(t, 1, 1)
	input := bufio.NewScanner(strings.NewReader("0 0\n"))
	var output bytes.Buffer

	err := runGame(
		client,
		game,
		[2]runners.PlayerType{runners.Agent, runners.RandomAI},
		input,
		&output,
	)

	require.NoError(t, err)
	got, err := client.GetGame(context.Background(), &dotscordonv1.GetGameRequest{
		GameId: game.GetGameId(),
	})
	require.NoError(t, err)
	assert.Equal(t, []byte{byte(dotscordonv1.Cell_CELL_PLAYER_0)}, got.GetGame().GetBoard().GetCells())
	assert.True(t, got.GetGame().GetTerminal())
	assert.Contains(t, output.String(), "Player 0 (Agent)")
	assert.Contains(t, output.String(), "Game over.")
}

func TestMakeRandomMoveExpandsSearchBeforeFallingBackToWholeField(t *testing.T) {
	client, game := newTestGame(t, 20, 20)
	occupiedCoordinates := []*dotscordonv1.Coordinate{
		{Row: 8, Column: 8},
		{Row: 9, Column: 9},
		{Row: 10, Column: 10},
		{Row: 6, Column: 6},
		{Row: 7, Column: 7},
	}
	for _, occupied := range occupiedCoordinates {
		response, err := client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
			GameId:       game.GetGameId(),
			ExpectedTurn: game.GetTurn(),
			Position:     occupied,
		})
		require.NoError(t, err)
		game = response.GetGame()
	}

	randomValues := []int{
		0, 0, // Radius 2, attempt 1: (8, 8).
		1, 1, // Radius 2, attempt 2: (9, 9).
		2, 2, // Radius 2, attempt 3: (10, 10).
		0, 0, // Radius 4, attempt 1: (6, 6).
		1, 1, // Radius 4, attempt 2: (7, 7).
		19, 19, // Whole field: (19, 19).
	}
	intn := func(limit int) int {
		require.NotEmpty(t, randomValues)
		value := randomValues[0]
		randomValues = randomValues[1:]
		require.Less(t, value, limit)
		return value
	}

	var output bytes.Buffer
	lastOpponentMove := &dotscordonv1.Coordinate{Row: 10, Column: 10}
	response, move, err := makeRandomMoveWithIntn(
		client,
		game,
		lastOpponentMove,
		&output,
		intn,
	)

	require.NoError(t, err)
	assert.Equal(t, uint32(19), move.GetRow())
	assert.Equal(t, uint32(19), move.GetColumn())
	assert.Equal(
		t,
		byte(dotscordonv1.Cell_CELL_PLAYER_1),
		response.GetGame().GetBoard().GetCells()[19*20+19],
	)
	assert.Empty(t, randomValues)
	assert.Equal(t, "RandomAI move: 8 8\n"+
		"RandomAI move: 9 9\n"+
		"RandomAI move: 10 10\n"+
		"RandomAI move: 6 6\n"+
		"RandomAI move: 7 7\n"+
		"RandomAI move: 19 19\n", output.String())
}

func TestRandomCoordinatesNearClipsSearchRadiusToField(t *testing.T) {
	values := []int{0, 2}
	intn := func(limit int) int {
		value := values[0]
		values = values[1:]
		require.Less(t, value, limit)
		return value
	}

	row, column := randomCoordinatesNear(
		&dotscordonv1.Coordinate{Row: 0, Column: 6},
		2,
		7,
		7,
		intn,
	)

	assert.Equal(t, uint32(0), row)
	assert.Equal(t, uint32(6), column)
}

func TestBoardToStringRendersAllCellStates(t *testing.T) {
	board := &dotscordonv1.Board{
		Rows:    1,
		Columns: 6,
		Cells: []byte{
			byte(dotscordonv1.Cell_CELL_EMPTY),
			byte(dotscordonv1.Cell_CELL_PLAYER_0),
			byte(dotscordonv1.Cell_CELL_PLAYER_1),
			byte(dotscordonv1.Cell_CELL_DEAD_EMPTY),
			byte(dotscordonv1.Cell_CELL_DEAD_PLAYER_0),
			byte(dotscordonv1.Cell_CELL_DEAD_PLAYER_1),
		},
	}

	assert.Equal(t, "    00 01 02 03 04 05\n00  .  0  1  -  x  X", boardToString(board))
}

func TestEmbeddedGameServerCloseStopsTheClientConnection(t *testing.T) {
	listener := bufconn.Listen(1024 * 1024)
	gameServer, err := startGameServerOnListener(
		listener,
		"passthrough:///bufnet",
		grpcserver.NewService(1),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return listener.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(t, err)

	_, err = gameServer.client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    2,
		Columns: 2,
	})
	require.NoError(t, err)
	require.NoError(t, gameServer.Close())
	require.NoError(t, gameServer.Close())

	_, err = gameServer.client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    2,
		Columns: 2,
	})
	assert.Equal(t, codes.Canceled, status.Code(err))
}

func newTestGame(
	t *testing.T,
	rows uint32,
	columns uint32,
) (dotscordonv1.GameServiceClient, *dotscordonv1.GameState) {
	t.Helper()

	listener := bufconn.Listen(1024 * 1024)
	gameServer, err := startGameServerOnListener(
		listener,
		"passthrough:///bufnet",
		grpcserver.NewService(10),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return listener.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, gameServer.Close())
	})

	created, err := gameServer.client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    rows,
		Columns: columns,
	})
	require.NoError(t, err)
	return gameServer.client, created.GetGame()
}
