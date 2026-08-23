package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGameMoveMarksDotOnGrid(t *testing.T) {
	field := NewGameField(3, 2)
	game := NewGame(field, []*Player{
		{Color: RedColor},
		{Color: BlueColor},
	})

	_, err := game.Move(PlayerIndex(1), 1, 2)

	want := NewGameField(3, 2)
	want.Dots[1][2] = want.Dots[1][2].WithOwner(PlayerIndex(1))
	assert.NoError(t, err)
	assert.Equal(t, want, field)
}

func TestMove_CapturesDotInDiamond(t *testing.T) {
	// Game board after eight turns:
	//       c0 c1 c2 c3 c4
	// r0    .  .  .  .  .
	// r1    .  .  0  .  .
	// r2    .  0  x  0  .
	// r3    .  1  0  1  .
	// r4    .  .  1  .  .
	//
	// 0 = player 0 dot, 1 = player 1 dot, x = killed player 1 dot, . = empty.
	type move struct {
		player PlayerIndex
		row    uint8
		col    uint8
	}
	moves := []move{
		{player: 0, row: 1, col: 2},
		{player: 1, row: 2, col: 2},
		{player: 0, row: 2, col: 1},
		{player: 1, row: 3, col: 1},
		{player: 0, row: 3, col: 2},
		{player: 1, row: 3, col: 3},
		{player: 0, row: 2, col: 3},
	}

	game := NewGame(NewGameField(5, 5), []*Player{
		{Color: RedColor},
		{Color: BlueColor},
	})
	var result *MoveResult
	for _, move := range moves {
		var err error
		result, err = game.Move(move.player, move.row, move.col)
		require.NoError(t, err)
	}

	assert.Equal(t, &MoveResult{
		ScoredPoints: 1,
		Cordons: [][]Dot{{
			{Row: 1, Col: 2, Owned: true, Owner: 0},
			{Row: 2, Col: 3, Owned: true, Owner: 0},
			{Row: 3, Col: 2, Owned: true, Owner: 0},
			{Row: 2, Col: 1, Owned: true, Owner: 0},
			{Row: 1, Col: 2, Owned: true, Owner: 0},
		}},
		KilledDots: []Dot{{
			Col:   2,
			Row:   2,
			Owned: true,
			Owner: 1,
		}},
	}, result)

	result, err := game.Move(1, 4, 2)
	require.NoError(t, err)
	assert.Equal(t, &MoveResult{}, result)
}
