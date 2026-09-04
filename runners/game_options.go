package runners

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strconv"
	"strings"
)

type PlayerType uint8

const (
	Human PlayerType = iota
	RandomAI
	Agent
)

type GameOptions struct {
	BoardRows   uint8
	BoardCols   uint8
	PlayerTypes [2]PlayerType
}

type gameOptionsInput struct {
	board   string
	player0 string
	player1 string
}

func newGameOptionsFlagSet(output io.Writer, input *gameOptionsInput) *flag.FlagSet {
	flags := flag.NewFlagSet("dots-cordon", flag.ContinueOnError)
	flags.SetOutput(output)
	flags.StringVar(&input.board, "board", "", "board dimensions as <rows>x<cols>")
	flags.StringVar(&input.player0, "player0", "", "player 0 controller: human, agent, or random")
	flags.StringVar(&input.player1, "player1", "", "player 1 controller: human, agent, or random")
	flags.Usage = func() {
		fmt.Fprintln(output, "Usage:")
		fmt.Fprintln(output, "  dots-cordon --board=<rows>x<cols> --player0=<type> --player1=<type>")
		fmt.Fprintln(output)
		fmt.Fprintln(output, "Player types: human, agent, random")
	}
	return flags
}

func PrintGameOptionsUsage(output io.Writer) {
	flags := newGameOptionsFlagSet(output, &gameOptionsInput{})
	flags.Usage()
}

func ParseGameOptions(args []string) (GameOptions, error) {
	input := gameOptionsInput{}
	flags := newGameOptionsFlagSet(io.Discard, &input)
	if err := flags.Parse(args); err != nil {
		return GameOptions{}, err
	}
	if flags.NArg() != 0 {
		return GameOptions{}, fmt.Errorf("unexpected arguments: %s", strings.Join(flags.Args(), " "))
	}

	missing := make([]string, 0, 3)
	if input.board == "" {
		missing = append(missing, "--board")
	}
	if input.player0 == "" {
		missing = append(missing, "--player0")
	}
	if input.player1 == "" {
		missing = append(missing, "--player1")
	}
	if len(missing) != 0 {
		return GameOptions{}, fmt.Errorf("missing required options: %s", strings.Join(missing, ", "))
	}

	rows, cols, err := parseBoardSize(input.board)
	if err != nil {
		return GameOptions{}, fmt.Errorf("invalid board %q: %w", input.board, err)
	}

	player0Type, err := parsePlayerType(input.player0)
	if err != nil {
		return GameOptions{}, fmt.Errorf("invalid player0 %q: %w", input.player0, err)
	}

	player1Type, err := parsePlayerType(input.player1)
	if err != nil {
		return GameOptions{}, fmt.Errorf("invalid player1 %q: %w", input.player1, err)
	}

	return GameOptions{
		BoardRows:   rows,
		BoardCols:   cols,
		PlayerTypes: [2]PlayerType{player0Type, player1Type},
	}, nil
}

func parseBoardSize(value string) (uint8, uint8, error) {
	dimensions := strings.Split(strings.ToLower(strings.TrimSpace(value)), "x")
	if len(dimensions) != 2 {
		return 0, 0, errors.New("must be formatted as <rows>x<cols>")
	}

	parseDimension := func(name, value string) (uint8, error) {
		parsed, err := strconv.ParseUint(strings.TrimSpace(value), 10, 8)
		if err != nil || parsed == 0 {
			return 0, fmt.Errorf("%s must be an integer between 1 and 255", name)
		}
		return uint8(parsed), nil
	}

	rows, err := parseDimension("rows", dimensions[0])
	if err != nil {
		return 0, 0, err
	}

	cols, err := parseDimension("cols", dimensions[1])
	if err != nil {
		return 0, 0, err
	}

	return rows, cols, nil
}

func parsePlayerType(value string) (PlayerType, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "human":
		return Human, nil
	case "agent":
		return Agent, nil
	case "random":
		return RandomAI, nil
	default:
		return 0, errors.New(`must be "human", "agent", or "random"`)
	}
}
