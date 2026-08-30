package main

import (
	"bytes"
	"testing"

	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMakeRandomMoveExpandsSearchBeforeFallingBackToWholeField(t *testing.T) {
	game := engine.NewGame(
		engine.NewGameField(20, 20),
		[]*engine.Player{{}, {}},
		engine.NoopGameRecorder{},
	)

	occupiedCoordinates := []engine.Coord{
		{Row: 8, Col: 8},
		{Row: 9, Col: 9},
		{Row: 10, Col: 10},
		{Row: 6, Col: 6},
		{Row: 7, Col: 7},
	}
	for _, occupied := range occupiedCoordinates {
		game.GameField.Transform(occupied.Row, occupied.Col, func(dot engine.Dot) engine.Dot {
			return dot.WithOwner(0)
		})
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
	lastOpponentMove := engine.Coord{Row: 10, Col: 10}
	_, move, err := makeRandomMoveWithIntn(game, 1, &lastOpponentMove, &output, intn)

	require.NoError(t, err)
	assert.Equal(t, engine.Coord{Row: 19, Col: 19}, move)
	assert.True(t, game.GameField.Dots[19][19].IsOwnedBy(1))
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

	row, col := randomCoordinatesNear(
		engine.Coord{Row: 0, Col: 6},
		2,
		7,
		7,
		intn,
	)

	assert.Equal(t, uint8(0), row)
	assert.Equal(t, uint8(6), col)
}
