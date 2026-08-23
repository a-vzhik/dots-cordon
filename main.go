package main

import (
	"bufio"
	"errors"
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
	gameFieldWidth  uint8 = 7
	gameFieldHeight uint8 = 7
)

func main() {
	recordFilePath := fmt.Sprintf(
		"game-%s.json",
		time.Now().Format("2006-01-02T15-04-05"),
	)
	game := engine.NewGame(
		engine.NewGameField(gameFieldWidth, gameFieldHeight),
		[]*engine.Player{
			{Color: engine.BlueColor, PlayerType: engine.Human},
			{Color: engine.RedColor, PlayerType: engine.RandomAI},
		},
		engine.NewJsonGameRecorder(recordFilePath),
	)

	err := runGame(game, bufio.NewScanner(os.Stdin), os.Stdout)
	if err != nil && !errors.Is(err, io.EOF) {
		fmt.Fprintf(os.Stderr, "Game stopped: %v\n", err)
	}
}

func runGame(game *engine.Game, input *bufio.Scanner, output io.Writer) error {
	currentPlayer := engine.PlayerIndex(0)

	for {
		printGame(output, game)

		var (
			result *engine.MoveResult
			err    error
		)

		switch game.Players[currentPlayer].PlayerType {
		case engine.Human:
			result, err = makeHumanMove(game, currentPlayer, input, output)
		case engine.RandomAI:
			result, err = makeRandomMove(game, currentPlayer, output)
		default:
			return fmt.Errorf("unsupported player type: %d", game.Players[currentPlayer].PlayerType)
		}
		if err != nil {
			return err
		}

		fmt.Fprintf(output, "MoveResult: %+v\n", *result)
		if err := waitForEnter(input, output); err != nil {
			return err
		}

		if result.IsTerminal {
			printGame(output, game)
			fmt.Fprintln(output, "Game over.")
			return nil
		}

		currentPlayer = currentPlayer.EnemyIndex()
	}
}

func printGame(output io.Writer, game *engine.Game) {
	fmt.Fprintf(
		output,
		"Score: Human %d - RandomAI %d\n%s\n",
		game.Players[0].Score,
		game.Players[1].Score,
		game.GameField.ToString(),
	)
}

func makeHumanMove(
	game *engine.Game,
	playerIndex engine.PlayerIndex,
	input *bufio.Scanner,
	output io.Writer,
) (*engine.MoveResult, error) {
	for {
		fmt.Fprint(output, "Your move (<row> <col>): ")
		if !input.Scan() {
			return nil, scannerError(input)
		}

		row, col, err := parseCoordinates(input.Text())
		if err != nil {
			fmt.Fprintf(output, "Invalid input: %v\n", err)
			continue
		}

		result, err := game.Move(playerIndex, row, col)
		if err != nil {
			fmt.Fprintf(output, "Invalid move: %v\n", err)
			continue
		}

		return result, nil
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
	output io.Writer,
) (*engine.MoveResult, error) {
	for {
		row := uint8(rand.Intn(int(game.GameField.Height)))
		col := uint8(rand.Intn(int(game.GameField.Width)))

		fmt.Fprintf(output, "RandomAI move: %d %d\n", row, col)

		result, err := game.Move(playerIndex, row, col)
		if err != nil {
			continue
		}

		return result, nil
	}
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
