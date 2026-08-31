package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEightNeighbours(t *testing.T) {
	const (
		width  = uint8(4)
		height = uint8(3)
	)

	tests := []struct {
		name  string
		coord Coord
		want  []Coord
	}{
		{
			name:  "interior",
			coord: Coord{Row: 1, Col: 1},
			want: []Coord{
				{Row: 0, Col: 0},
				{Row: 0, Col: 1},
				{Row: 0, Col: 2},
				{Row: 1, Col: 2},
				{Row: 2, Col: 2},
				{Row: 2, Col: 1},
				{Row: 2, Col: 0},
				{Row: 1, Col: 0},
			},
		},
		{
			name:  "top edge",
			coord: Coord{Row: 0, Col: 1},
			want: []Coord{
				{Row: 0, Col: 2},
				{Row: 1, Col: 2},
				{Row: 1, Col: 1},
				{Row: 1, Col: 0},
				{Row: 0, Col: 0},
			},
		},
		{
			name:  "right edge",
			coord: Coord{Row: 1, Col: 3},
			want: []Coord{
				{Row: 0, Col: 2},
				{Row: 0, Col: 3},
				{Row: 2, Col: 3},
				{Row: 2, Col: 2},
				{Row: 1, Col: 2},
			},
		},
		{
			name:  "bottom edge",
			coord: Coord{Row: 2, Col: 1},
			want: []Coord{
				{Row: 1, Col: 0},
				{Row: 1, Col: 1},
				{Row: 1, Col: 2},
				{Row: 2, Col: 2},
				{Row: 2, Col: 0},
			},
		},
		{
			name:  "left edge",
			coord: Coord{Row: 1, Col: 0},
			want: []Coord{
				{Row: 0, Col: 0},
				{Row: 0, Col: 1},
				{Row: 1, Col: 1},
				{Row: 2, Col: 1},
				{Row: 2, Col: 0},
			},
		},
		{
			name:  "top-left corner",
			coord: Coord{Row: 0, Col: 0},
			want: []Coord{
				{Row: 0, Col: 1},
				{Row: 1, Col: 1},
				{Row: 1, Col: 0},
			},
		},
		{
			name:  "top-right corner",
			coord: Coord{Row: 0, Col: 3},
			want: []Coord{
				{Row: 1, Col: 3},
				{Row: 1, Col: 2},
				{Row: 0, Col: 2},
			},
		},
		{
			name:  "bottom-right corner",
			coord: Coord{Row: 2, Col: 3},
			want: []Coord{
				{Row: 1, Col: 2},
				{Row: 1, Col: 3},
				{Row: 2, Col: 2},
			},
		},
		{
			name:  "bottom-left corner",
			coord: Coord{Row: 2, Col: 0},
			want: []Coord{
				{Row: 1, Col: 0},
				{Row: 1, Col: 1},
				{Row: 2, Col: 1},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.want, eightNeighbours(test.coord, width, height))
		})
	}
}
