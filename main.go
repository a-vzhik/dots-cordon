package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/a-vzhik/dots-cordon/engine"
)

const (
	closeMoveRadius   = 2
	closeMoveAttempts = 3
	wideMoveRadius    = 4
	wideMoveAttempts  = 2
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, input io.Reader, output io.Writer, errorOutput io.Writer) int {
	options, err := parseGameOptions(args)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			printGameOptionsUsage(output)
			return 0
		}

		fmt.Fprintf(errorOutput, "Invalid game options: %v\n\n", err)
		printGameOptionsUsage(errorOutput)
		return 2
	}

	recordFilePath := fmt.Sprintf(
		"game-%s.json",
		time.Now().Format("2006-01-02T15-04-05"),
	)
	game := engine.NewGame(
		engine.NewGameField(options.BoardCols, options.BoardRows),
		[]*engine.Player{
			{Color: engine.BlueColor},
			{Color: engine.RedColor},
		},
		engine.NewJsonGameRecorder(recordFilePath),
	)

	err = runGame(game, options.PlayerTypes, bufio.NewScanner(input), output)
	if err != nil && !errors.Is(err, io.EOF) {
		fmt.Fprintf(errorOutput, "Game stopped: %v\n", err)
		return 1
	}

	return 0
}

func runGame(
	game *engine.Game,
	playerTypes [2]PlayerType,
	input *bufio.Scanner,
	output io.Writer,
) error {
	currentPlayer := engine.PlayerIndex(0)
	var lastMove *engine.Coord

	for {
		printGame(output, game, playerTypes)

		var (
			result      *engine.MoveResult
			currentMove engine.Coord
			err         error
		)

		playerType := playerTypes[currentPlayer]
		switch playerType {
		case Human, Agent:
			result, currentMove, err = makeHumanMove(game, currentPlayer, input, output)
		case RandomAI:
			result, currentMove, err = makeRandomMove(game, currentPlayer, lastMove, output)
		default:
			return fmt.Errorf("unsupported player type: %d", playerType)
		}
		if err != nil {
			return err
		}

		fmt.Fprintf(output, "MoveResult: %+v\n", *result)
		//if err := waitForEnter(input, output); err != nil {
		//	return err
		//}

		if result.IsTerminal {
			printGame(output, game, playerTypes)
			fmt.Fprintln(output, "Game over.")
			return nil
		}

		lastMove = &currentMove
		currentPlayer = currentPlayer.EnemyIndex()
	}
}

func printGame(output io.Writer, game *engine.Game, playerTypes [2]PlayerType) {
	fmt.Fprintf(
		output,
		"Score: Player 0 (%s) %d - Player 1 (%s) %d\n%s\n",
		playerTypeName(playerTypes[0]),
		game.Players[0].Score,
		playerTypeName(playerTypes[1]),
		game.Players[1].Score,
		game.GameField.ToString(),
	)
}

func playerTypeName(playerType PlayerType) string {
	switch playerType {
	case Human:
		return "Human"
	case Agent:
		return "Agent"
	case RandomAI:
		return "RandomAI"
	default:
		return fmt.Sprintf("Unknown:%d", playerType)
	}
}

func makeHumanMove(
	game *engine.Game,
	playerIndex engine.PlayerIndex,
	input *bufio.Scanner,
	output io.Writer,
) (*engine.MoveResult, engine.Coord, error) {
	for {
		fmt.Fprintf(
			output,
			"Player %d move (<row> <col>) OR <Q> to finish the game: ",
			playerIndex,
		)
		if !input.Scan() {
			return nil, engine.Coord{}, scannerError(input)
		}

		text := input.Text()
		if text == "Q" {
			return nil, engine.Coord{}, errors.New("game stopped by user")
		}

		row, col, err := parseCoordinates(text)
		if err != nil {
			fmt.Fprintf(output, "Invalid input: %v\n", err)
			continue
		}

		result, err := game.Move(playerIndex, row, col)
		if err != nil {
			fmt.Fprintf(output, "Invalid move: %v\n", err)
			continue
		}

		return result, engine.Coord{Row: row, Col: col}, nil
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

	col, err := strconv.ParseUint(coordinates[1], 10, 8)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid column %q", coordinates[1])
	}

	return uint8(row), uint8(col), nil
}

func makeRandomMove(
	game *engine.Game,
	playerIndex engine.PlayerIndex,
	lastOpponentMove *engine.Coord,
	output io.Writer,
) (*engine.MoveResult, engine.Coord, error) {
	return makeRandomMoveWithIntn(game, playerIndex, lastOpponentMove, output, rand.Intn)
}

func makeRandomMoveWithIntn(
	game *engine.Game,
	playerIndex engine.PlayerIndex,
	lastOpponentMove *engine.Coord,
	output io.Writer,
	intn func(int) int,
) (*engine.MoveResult, engine.Coord, error) {
	if lastOpponentMove != nil {
		attempts := []struct {
			radius uint8
			count  int
		}{
			{radius: closeMoveRadius, count: closeMoveAttempts},
			{radius: wideMoveRadius, count: wideMoveAttempts},
		}

		for _, attempt := range attempts {
			for range attempt.count {
				row, col := randomCoordinatesNear(
					*lastOpponentMove,
					attempt.radius,
					game.GameField.Height,
					game.GameField.Width,
					intn,
				)

				result, err := tryRandomMove(game, playerIndex, row, col, output)
				if err == nil {
					return result, engine.Coord{Row: row, Col: col}, nil
				}
			}
		}
	}

	for {
		row := uint8(intn(int(game.GameField.Height)))
		col := uint8(intn(int(game.GameField.Width)))

		result, err := tryRandomMove(game, playerIndex, row, col, output)
		if err == nil {
			return result, engine.Coord{Row: row, Col: col}, nil
		}
	}
}

func randomCoordinatesNear(
	center engine.Coord,
	radius uint8,
	height uint8,
	width uint8,
	intn func(int) int,
) (uint8, uint8) {
	minRow := max(0, int(center.Row)-int(radius))
	maxRow := min(int(height)-1, int(center.Row)+int(radius))
	minCol := max(0, int(center.Col)-int(radius))
	maxCol := min(int(width)-1, int(center.Col)+int(radius))

	row := minRow + intn(maxRow-minRow+1)
	col := minCol + intn(maxCol-minCol+1)
	return uint8(row), uint8(col)
}

func tryRandomMove(
	game *engine.Game,
	playerIndex engine.PlayerIndex,
	row uint8,
	col uint8,
	output io.Writer,
) (*engine.MoveResult, error) {
	fmt.Fprintf(output, "RandomAI move: %d %d\n", row, col)
	return game.Move(playerIndex, row, col)
}

func waitForEnter(input *bufio.Scanner, output io.Writer) error {
	fmt.Fprint(output, "Press <Enter> to continue...")
	if !input.Scan() {
		return scannerError(input)
	}
	fmt.Fprintln(output)
	return nil
}

func scannerError(input *bufio.Scanner) error {
	if err := input.Err(); err != nil {
		return fmt.Errorf("read input: %w", err)
	}
	return io.EOF
}
