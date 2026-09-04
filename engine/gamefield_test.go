package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGameFieldToString(t *testing.T) {
	field := NewGameField(3, 2)
	field.Dots[0][1] = field.Dots[0][1].WithOwner(PlayerIndex(0))
	field.Dots[0][2] = field.Dots[0][2].WithOwner(PlayerIndex(1)).WithKilled()
	field.Dots[1][0] = field.Dots[1][0].WithOwner(PlayerIndex(1))
	field.Dots[1][1] = field.Dots[1][1].WithOwner(PlayerIndex(0)).WithKilled()
	field.Dots[1][2] = field.Dots[1][2].WithKilled()

	want := "    00 01 02\n" +
		"00  .  0  X\n" +
		"01  1  x  -"
	assert.Equal(t, want, field.ToString())
}

func TestGameFieldIsFull(t *testing.T) {
	t.Run("false when an empty dot remains", func(t *testing.T) {
		field := NewGameField(2, 1)
		field.Dots[0][0] = field.Dots[0][0].WithOwner(PlayerIndex(0))

		assert.False(t, field.IsFull())
	})

	t.Run("true when every dot is unavailable", func(t *testing.T) {
		field := NewGameField(2, 1)
		field.Dots[0][0] = field.Dots[0][0].WithOwner(PlayerIndex(0))
		field.Dots[0][1] = field.Dots[0][1].WithKilled()

		assert.True(t, field.IsFull())
	})
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
				{Coord: Coord{Col: 2, Row: 1}},
			},
		},
		{
			name: "multiple dots",
			filter: func(dot Dot) bool {
				return dot.Row == 1
			},
			want: []Dot{
				{Coord: Coord{Col: 0, Row: 1}},
				{Coord: Coord{Col: 1, Row: 1}},
				{Coord: Coord{Col: 2, Row: 1}},
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
