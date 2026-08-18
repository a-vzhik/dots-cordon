package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGameMoveMarksDotOnGrid(t *testing.T) {
	field := NewGameField(3, 2)
	game := NewGame(field, []*Player{
		{Color: RedColor},
		{Color: BlueColor},
	})

	err := game.Move(PlayerIndex(1), 1, 2)

	want := NewGameField(3, 2)
	want.Dots[1][2] = want.Dots[1][2].WithOwner(PlayerIndex(1))
	assert.NoError(t, err)
	assert.Equal(t, want, field)
}

func TestGameFieldTransformFiltersDots(t *testing.T) {
	t.Run("single dot", func(t *testing.T) {
		field := NewGameField(3, 2)

		field.Transform(
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

		field.Transform(
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
