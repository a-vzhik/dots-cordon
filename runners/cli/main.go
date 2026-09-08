package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/a-vzhik/dots-cordon/runners"
	grpcserver "github.com/a-vzhik/dots-cordon/runners/grpc/server"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	closeMoveRadius    = 2
	closeMoveAttempts  = 3
	wideMoveRadius     = 4
	wideMoveAttempts   = 2
	serverStartTimeout = 5 * time.Second
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, input io.Reader, output io.Writer, errorOutput io.Writer) (exitCode int) {
	options, err := runners.ParseGameOptions(args)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			runners.PrintGameOptionsUsage(output)
			return 0
		}

		fmt.Fprintf(errorOutput, "Invalid game options: %v\n\n", err)
		runners.PrintGameOptionsUsage(errorOutput)
		return 2
	}

	recordFilePath := fmt.Sprintf(
		"game-%s.json",
		time.Now().Format("2006-01-02T15-04-05"),
	)
	gameServer, err := startEmbeddedGameServer(
		grpcserver.NewService(
			1,
			grpcserver.WithRecorderFactory(func(string) engine.Recorder {
				return engine.NewJsonGameRecorder(recordFilePath)
			}),
		),
	)
	if err != nil {
		fmt.Fprintf(errorOutput, "Failed to start game server: %v\n", err)
		return 1
	}
	defer func() {
		if closeErr := gameServer.Close(); closeErr != nil {
			fmt.Fprintf(errorOutput, "Failed to stop game server: %v\n", closeErr)
			if exitCode == 0 {
				exitCode = 1
			}
		}
	}()

	startContext, cancelStart := context.WithTimeout(context.Background(), serverStartTimeout)
	created, err := gameServer.client.CreateGame(startContext, &dotscordonv1.CreateGameRequest{
		Rows:    uint32(options.BoardRows),
		Columns: uint32(options.BoardCols),
	})
	cancelStart()
	if err != nil {
		fmt.Fprintf(errorOutput, "Failed to create game: %v\n", err)
		return 1
	}

	err = runGame(
		gameServer.client,
		created.GetGame(),
		options.PlayerTypes,
		bufio.NewScanner(input),
		output,
	)
	if err != nil && !errors.Is(err, io.EOF) {
		fmt.Fprintf(errorOutput, "Game stopped: %v\n", err)
		return 1
	}

	return 0
}

func runGame(
	client dotscordonv1.GameServiceClient,
	game *dotscordonv1.GameState,
	playerTypes [2]runners.PlayerType,
	input *bufio.Scanner,
	output io.Writer,
) error {
	var lastMove *dotscordonv1.Coordinate

	for {
		printGame(output, game, playerTypes)

		playerIndex := game.GetCurrentPlayer()
		if playerIndex >= uint32(len(playerTypes)) {
			return fmt.Errorf("unsupported player index: %d", playerIndex)
		}

		var (
			response    *dotscordonv1.MakeMoveResponse
			currentMove *dotscordonv1.Coordinate
			err         error
		)

		playerType := playerTypes[playerIndex]
		switch playerType {
		case runners.Human, runners.Agent:
			response, currentMove, err = makeHumanMove(client, game, input, output)
		case runners.RandomAI:
			response, currentMove, err = makeRandomMove(client, game, lastMove, output)
		default:
			return fmt.Errorf("unsupported player type: %d", playerType)
		}
		if err != nil {
			return err
		}

		printMoveResult(output, response.GetResult())
		game = response.GetGame()
		if game.GetTerminal() {
			printGame(output, game, playerTypes)
			fmt.Fprintln(output, "Game over.")
			return nil
		}

		lastMove = currentMove
	}
}

func printGame(
	output io.Writer,
	game *dotscordonv1.GameState,
	playerTypes [2]runners.PlayerType,
) {
	fmt.Fprintf(
		output,
		"Score: Player 0 (%s) %d - Player 1 (%s) %d\n%s\n",
		playerTypeName(playerTypes[0]),
		scoreAt(game, 0),
		playerTypeName(playerTypes[1]),
		scoreAt(game, 1),
		boardToString(game.GetBoard()),
	)
}

func printMoveResult(output io.Writer, result *dotscordonv1.MoveResult) {
	fmt.Fprintf(
		output,
		"MoveResult: {Player:%d ScoredPoints:%d KilledCells:%v Cordons:%v}\n",
		result.GetPlayer(),
		result.GetScoredPoints(),
		result.GetKilledCells(),
		result.GetCordons(),
	)
}

func scoreAt(game *dotscordonv1.GameState, player int) uint32 {
	if player >= len(game.GetScores()) {
		return 0
	}
	return game.GetScores()[player]
}

func boardToString(board *dotscordonv1.Board) string {
	var result strings.Builder

	result.WriteString("    ")
	for column := uint32(0); column < board.GetColumns(); column++ {
		if column > 0 {
			result.WriteByte(' ')
		}
		fmt.Fprintf(&result, "%02d", column)
	}

	for row := uint32(0); row < board.GetRows(); row++ {
		fmt.Fprintf(&result, "\n%02d  ", row)
		for column := uint32(0); column < board.GetColumns(); column++ {
			if column > 0 {
				result.WriteString("  ")
			}
			result.WriteByte(boardCellSymbol(board, row, column))
		}
	}

	return result.String()
}

func boardCellSymbol(board *dotscordonv1.Board, row, column uint32) byte {
	index := int(row*board.GetColumns() + column)
	if index >= len(board.GetCells()) {
		return '?'
	}

	switch dotscordonv1.Cell(board.GetCells()[index]) {
	case dotscordonv1.Cell_CELL_PLAYER_0:
		return '0'
	case dotscordonv1.Cell_CELL_PLAYER_1:
		return '1'
	case dotscordonv1.Cell_CELL_DEAD_EMPTY:
		return '-'
	case dotscordonv1.Cell_CELL_DEAD_PLAYER_0:
		return 'x'
	case dotscordonv1.Cell_CELL_DEAD_PLAYER_1:
		return 'X'
	case dotscordonv1.Cell_CELL_EMPTY:
		return '.'
	default:
		return '?'
	}
}

func playerTypeName(playerType runners.PlayerType) string {
	switch playerType {
	case runners.Human:
		return "Human"
	case runners.Agent:
		return "Agent"
	case runners.RandomAI:
		return "RandomAI"
	default:
		return fmt.Sprintf("Unknown:%d", playerType)
	}
}

func makeHumanMove(
	client dotscordonv1.GameServiceClient,
	game *dotscordonv1.GameState,
	input *bufio.Scanner,
	output io.Writer,
) (*dotscordonv1.MakeMoveResponse, *dotscordonv1.Coordinate, error) {
	for {
		fmt.Fprintf(
			output,
			"Player %d move (<row> <col>) OR <Q> to finish the game: ",
			game.GetCurrentPlayer(),
		)
		if !input.Scan() {
			return nil, nil, scannerError(input)
		}

		text := input.Text()
		if text == "Q" {
			return nil, nil, errors.New("game stopped by user")
		}

		row, column, err := parseCoordinates(text)
		if err != nil {
			fmt.Fprintf(output, "Invalid input: %v\n", err)
			continue
		}

		move := &dotscordonv1.Coordinate{Row: uint32(row), Column: uint32(column)}
		response, err := client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
			GameId:       game.GetGameId(),
			ExpectedTurn: game.GetTurn(),
			Position:     move,
		})
		if err != nil {
			if status.Code(err) != codes.InvalidArgument {
				return nil, nil, err
			}
			fmt.Fprintf(output, "Invalid move: %v\n", status.Convert(err).Message())
			continue
		}

		return response, move, nil
	}
}

func parseCoordinates(input string) (uint8, uint8, error) {
	coordinates := strings.Fields(input)
	if len(coordinates) != 2 {
		return 0, 0, errors.New("enter exactly two numbers")
	}

	row, err := strconv.ParseUint(coordinates[0], 10, 8)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid row %q", coordinates[0])
	}

	column, err := strconv.ParseUint(coordinates[1], 10, 8)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid column %q", coordinates[1])
	}

	return uint8(row), uint8(column), nil
}

func makeRandomMove(
	client dotscordonv1.GameServiceClient,
	game *dotscordonv1.GameState,
	lastOpponentMove *dotscordonv1.Coordinate,
	output io.Writer,
) (*dotscordonv1.MakeMoveResponse, *dotscordonv1.Coordinate, error) {
	return makeRandomMoveWithIntn(client, game, lastOpponentMove, output, rand.Intn)
}

func makeRandomMoveWithIntn(
	client dotscordonv1.GameServiceClient,
	game *dotscordonv1.GameState,
	lastOpponentMove *dotscordonv1.Coordinate,
	output io.Writer,
	intn func(int) int,
) (*dotscordonv1.MakeMoveResponse, *dotscordonv1.Coordinate, error) {
	board := game.GetBoard()
	if lastOpponentMove != nil {
		attempts := []struct {
			radius uint32
			count  int
		}{
			{radius: closeMoveRadius, count: closeMoveAttempts},
			{radius: wideMoveRadius, count: wideMoveAttempts},
		}

		for _, attempt := range attempts {
			for range attempt.count {
				row, column := randomCoordinatesNear(
					lastOpponentMove,
					attempt.radius,
					board.GetRows(),
					board.GetColumns(),
					intn,
				)

				response, move, err := tryRandomMove(client, game, row, column, output)
				if err == nil {
					return response, move, nil
				}
				if status.Code(err) != codes.InvalidArgument {
					return nil, nil, err
				}
			}
		}
	}

	for {
		row := uint32(intn(int(board.GetRows())))
		column := uint32(intn(int(board.GetColumns())))

		response, move, err := tryRandomMove(client, game, row, column, output)
		if err == nil {
			return response, move, nil
		}
		if status.Code(err) != codes.InvalidArgument {
			return nil, nil, err
		}
	}
}

func randomCoordinatesNear(
	center *dotscordonv1.Coordinate,
	radius uint32,
	height uint32,
	width uint32,
	intn func(int) int,
) (uint32, uint32) {
	minRow := max(0, int(center.GetRow())-int(radius))
	maxRow := min(int(height)-1, int(center.GetRow())+int(radius))
	minColumn := max(0, int(center.GetColumn())-int(radius))
	maxColumn := min(int(width)-1, int(center.GetColumn())+int(radius))

	row := minRow + intn(maxRow-minRow+1)
	column := minColumn + intn(maxColumn-minColumn+1)
	return uint32(row), uint32(column)
}

func tryRandomMove(
	client dotscordonv1.GameServiceClient,
	game *dotscordonv1.GameState,
	row uint32,
	column uint32,
	output io.Writer,
) (*dotscordonv1.MakeMoveResponse, *dotscordonv1.Coordinate, error) {
	fmt.Fprintf(output, "RandomAI move: %d %d\n", row, column)
	move := &dotscordonv1.Coordinate{Row: row, Column: column}
	response, err := client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:       game.GetGameId(),
		ExpectedTurn: game.GetTurn(),
		Position:     move,
	})
	return response, move, err
}

func scannerError(input *bufio.Scanner) error {
	if err := input.Err(); err != nil {
		return fmt.Errorf("read input: %w", err)
	}
	return io.EOF
}
