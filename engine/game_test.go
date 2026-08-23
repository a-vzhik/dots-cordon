package engine

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGameMoveMarksDotOnGrid(t *testing.T) {
	field := NewGameField(3, 2)
	game := NewGame(field, []*Player{
		{Color: RedColor},
		{Color: BlueColor},
	}, NoopGameRecorder{})

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
	}, NoopGameRecorder{})
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

func TestMove_CapturesTwoGroupsInSequence(t *testing.T) {
	// Final game board after 23 alternating turns:
	//
	//     00 01 02 03 04 05 06
	// 00  .  0  .  .  1  .  1
	// 01  0  K  0  .  .  .  .
	// 02  0  K  0  .  .  .  1
	// 03  .  0  .  .  .  0  .
	// 04  1  .  .  .  0  K  0
	// 05  .  .  .  .  0  K  0
	// 06  1  .  1  1  .  0  .
	//
	// Player 0 captures the top-left group on turn 11 and the bottom-right
	// group on turn 23. K marks a killed player 1 dot.
	type move struct {
		row  uint8
		col  uint8
		want *MoveResult
	}

	moves := []move{
		{row: 1, col: 0},
		{row: 1, col: 1},
		{row: 2, col: 0},
		{row: 2, col: 1},
		{row: 3, col: 1},
		{row: 0, col: 4},
		{row: 2, col: 2},
		{row: 0, col: 6},
		{row: 1, col: 2},
		{row: 2, col: 6},
		{
			row: 0,
			col: 1,
			want: &MoveResult{
				ScoredPoints: 2,
				Cordons: [][]Dot{{
					{Row: 0, Col: 1, Owned: true, Owner: 0},
					{Row: 1, Col: 2, Owned: true, Owner: 0},
					{Row: 2, Col: 2, Owned: true, Owner: 0},
					{Row: 3, Col: 1, Owned: true, Owner: 0},
					{Row: 2, Col: 0, Owned: true, Owner: 0},
					{Row: 1, Col: 0, Owned: true, Owner: 0},
					{Row: 0, Col: 1, Owned: true, Owner: 0},
				}},
				KilledDots: []Dot{
					{Row: 1, Col: 1, Owned: true, Owner: 1},
					{Row: 2, Col: 1, Owned: true, Owner: 1},
				},
			},
		},
		{row: 4, col: 0},
		{row: 3, col: 5},
		{row: 4, col: 5},
		{row: 4, col: 4},
		{row: 5, col: 5},
		{row: 4, col: 6},
		{row: 6, col: 0},
		{row: 5, col: 4},
		{row: 6, col: 2},
		{row: 5, col: 6},
		{row: 6, col: 3},
		{
			row: 6,
			col: 5,
			want: &MoveResult{
				ScoredPoints: 2,
				Cordons: [][]Dot{{
					{Row: 3, Col: 5, Owned: true, Owner: 0},
					{Row: 4, Col: 6, Owned: true, Owner: 0},
					{Row: 5, Col: 6, Owned: true, Owner: 0},
					{Row: 6, Col: 5, Owned: true, Owner: 0},
					{Row: 5, Col: 4, Owned: true, Owner: 0},
					{Row: 4, Col: 4, Owned: true, Owner: 0},
					{Row: 3, Col: 5, Owned: true, Owner: 0},
				}},
				KilledDots: []Dot{
					{Row: 4, Col: 5, Owned: true, Owner: 1},
					{Row: 5, Col: 5, Owned: true, Owner: 1},
				},
			},
		},
	}

	game := NewGame(NewGameField(7, 7), []*Player{
		{Color: RedColor},
		{Color: BlueColor},
	}, NoopGameRecorder{})
	placedDots := [2]int{}

	for turnIdx, move := range moves {
		playerIdx := PlayerIndex(turnIdx % 2)
		result, err := game.Move(playerIdx, move.row, move.col)
		require.NoError(t, err, "turn %d", turnIdx+1)

		placedDots[playerIdx]++
		assert.LessOrEqual(t, placedDots[0], placedDots[1]+1, "turn %d", turnIdx+1)
		assert.LessOrEqual(t, placedDots[1], placedDots[0]+1, "turn %d", turnIdx+1)

		if move.want == nil {
			assert.Equal(t, &MoveResult{}, result, "turn %d", turnIdx+1)
			continue
		}
		assert.Equal(t, move.want, result, "turn %d", turnIdx+1)
	}

	assert.Equal(t, uint32(4), game.Players[0].Score)

	fmt.Print(game.GameField.ToString())
}
