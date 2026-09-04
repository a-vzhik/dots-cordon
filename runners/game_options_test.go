package runners

import (
	"bytes"
	"flag"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseGameOptions(t *testing.T) {
	t.Run("requires all options", func(t *testing.T) {
		_, err := ParseGameOptions(nil)

		assert.EqualError(t, err, "missing required options: --board, --player0, --player1")
	})

	t.Run("custom", func(t *testing.T) {
		options, err := ParseGameOptions([]string{
			"--board=15x25",
			"--player0", "agent",
			"--player1", "human",
		})

		require.NoError(t, err)
		assert.Equal(t, GameOptions{
			BoardRows:   15,
			BoardCols:   25,
			PlayerTypes: [2]PlayerType{Agent, Human},
		}, options)
	})

	t.Run("invalid value", func(t *testing.T) {
		_, err := ParseGameOptions([]string{
			"--board=15x25",
			"--player0=agent",
			"--player1=robot",
		})

		assert.EqualError(t, err, `invalid player1 "robot": must be "human", "agent", or "random"`)
	})

	t.Run("unexpected argument", func(t *testing.T) {
		_, err := ParseGameOptions([]string{"extra"})

		assert.EqualError(t, err, "unexpected arguments: extra")
	})

	t.Run("help", func(t *testing.T) {
		_, err := ParseGameOptions([]string{"--help"})

		assert.ErrorIs(t, err, flag.ErrHelp)
	})
}

func TestPrintGameOptionsUsage(t *testing.T) {
	var output bytes.Buffer

	PrintGameOptionsUsage(&output)

	assert.Equal(t, "Usage:\n"+
		"  dots-cordon --board=<rows>x<cols> --player0=<type> --player1=<type>\n"+
		"\n"+
		"Player types: human, agent, random\n", output.String())
}

func TestParseBoardSize(t *testing.T) {
	testCases := []struct {
		name         string
		input        string
		expectedRows uint8
		expectedCols uint8
	}{
		{name: "rectangular", input: "15x25", expectedRows: 15, expectedCols: 25},
		{name: "maximum dimension", input: "1x255", expectedRows: 1, expectedCols: 255},
		{name: "case and whitespace", input: " 7 X 11 ", expectedRows: 7, expectedCols: 11},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			rows, cols, err := parseBoardSize(testCase.input)

			require.NoError(t, err)
			assert.Equal(t, testCase.expectedRows, rows)
			assert.Equal(t, testCase.expectedCols, cols)
		})
	}

	for _, input := range []string{"", "15", "15x25x30", "0x25", "15x0", "256x25", "rowsx25"} {
		t.Run("invalid "+input, func(t *testing.T) {
			_, _, err := parseBoardSize(input)

			assert.Error(t, err)
		})
	}
}

func TestParsePlayerType(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected PlayerType
	}{
		{name: "human", input: "human", expected: Human},
		{name: "agent", input: "agent", expected: Agent},
		{name: "random", input: "random", expected: RandomAI},
		{name: "case and whitespace", input: " Human ", expected: Human},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			actual, err := parsePlayerType(testCase.input)

			require.NoError(t, err)
			assert.Equal(t, testCase.expected, actual)
		})
	}

	_, err := parsePlayerType("robot")
	assert.EqualError(t, err, `must be "human", "agent", or "random"`)
}
