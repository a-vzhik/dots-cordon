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
		Cordons: [][]CordonIndexKey{{
			{Row: 1, Col: 2},
			{Row: 2, Col: 3},
			{Row: 3, Col: 2},
			{Row: 2, Col: 1},
			{Row: 1, Col: 2},
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

func TestGameFieldTransformFiltersDots(t *testing.T) {
	t.Run("single dot", func(t *testing.T) {
		field := NewGameField(3, 2)

		field.TransformFunc(
			func(dot Dot) bool {
				return dot.Row == 1 && dot.Col == 2
			},
			func(dot Dot) Dot {
				return dot.WithOwner(PlayerIndex(1))
			},
		)

		want := NewGameField(3, 2)
		want.Dots[1][2] = want.Dots[1][2].WithOwner(PlayerIndex(1))
		assert.Equal(t, want, field)
	})

	t.Run("group of dots", func(t *testing.T) {
		field := NewGameField(3, 3)

		field.TransformFunc(
			func(dot Dot) bool {
				return dot.Row == 1
			},
			func(dot Dot) Dot {
				return dot.WithKilled()
			},
		)

		want := NewGameField(3, 3)
		for colIdx, dot := range want.Dots[1] {
			want.Dots[1][colIdx] = dot.WithKilled()
		}
		assert.Equal(t, want, field)
	})
}

func TestGameFieldFind(t *testing.T) {
	tests := []struct {
		name   string
		filter func(Dot) bool
		want   []Dot
	}{
		{
			name: "nothing",
			filter: func(Dot) bool {
				return false
			},
			want: []Dot{},
		},
		{
			name: "one dot",
			filter: func(dot Dot) bool {
				return dot.Row == 1 && dot.Col == 2
			},
			want: []Dot{
				{Col: 2, Row: 1},
			},
		},
		{
			name: "multiple dots",
			filter: func(dot Dot) bool {
				return dot.Row == 1
			},
			want: []Dot{
				{Col: 0, Row: 1},
				{Col: 1, Row: 1},
				{Col: 2, Row: 1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			field := NewGameField(3, 2)

			got := field.Find(tt.filter)

			assert.Equal(t, tt.want, got)
		})
	}
}
