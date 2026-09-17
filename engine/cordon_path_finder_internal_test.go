package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCordonAreaSize(t *testing.T) {
	tests := []struct {
		name       string
		cordon     []Coord
		wantWidth  uint8
		wantHeight uint8
	}{
		{
			name:   "nil cordon",
			cordon: nil,
		},
		{
			name:   "empty cordon",
			cordon: []Coord{},
		},
		{
			name:   "single point",
			cordon: []Coord{{Row: 7, Col: 11}},
		},
		{
			name: "horizontal line",
			cordon: []Coord{
				{Row: 4, Col: 9},
				{Row: 4, Col: 3},
				{Row: 4, Col: 6},
			},
			wantWidth: 6,
		},
		{
			name: "vertical line",
			cordon: []Coord{
				{Row: 8, Col: 5},
				{Row: 2, Col: 5},
				{Row: 4, Col: 5},
			},
			wantHeight: 6,
		},
		{
			name: "offset rectangle with repeated closing point",
			cordon: []Coord{
				{Row: 3, Col: 8},
				{Row: 3, Col: 13},
				{Row: 6, Col: 13},
				{Row: 6, Col: 8},
				{Row: 3, Col: 8},
			},
			wantWidth:  5,
			wantHeight: 3,
		},
		{
			name: "solid block from hanging game",
			cordon: []Coord{
				{Row: 2, Col: 3},
				{Row: 2, Col: 4},
				{Row: 3, Col: 4},
				{Row: 3, Col: 3},
				{Row: 2, Col: 3},
			},
			wantWidth:  1,
			wantHeight: 1,
		},
		{
			name: "diamond enclosing one dot",
			cordon: []Coord{
				{Row: 1, Col: 5},
				{Row: 2, Col: 6},
				{Row: 3, Col: 5},
				{Row: 2, Col: 4},
				{Row: 1, Col: 5},
			},
			wantWidth:  2,
			wantHeight: 2,
		},
		{
			name: "loop with unit width",
			cordon: []Coord{
				{Row: 2, Col: 3},
				{Row: 2, Col: 4},
				{Row: 5, Col: 4},
				{Row: 5, Col: 3},
				{Row: 2, Col: 3},
			},
			wantWidth:  1,
			wantHeight: 3,
		},
		{
			name: "loop with unit height",
			cordon: []Coord{
				{Row: 2, Col: 3},
				{Row: 2, Col: 6},
				{Row: 3, Col: 6},
				{Row: 3, Col: 3},
				{Row: 2, Col: 3},
			},
			wantWidth:  3,
			wantHeight: 1,
		},
		{
			name: "unordered points with separate row and column extremes",
			cordon: []Coord{
				{Row: 7, Col: 12},
				{Row: 12, Col: 5},
				{Row: 4, Col: 10},
				{Row: 9, Col: 2},
				{Row: 6, Col: 6},
			},
			wantWidth:  10,
			wantHeight: 8,
		},
		{
			name: "full uint8 coordinate range",
			cordon: []Coord{
				{Row: 255, Col: 0},
				{Row: 0, Col: 255},
				{Row: 255, Col: 255},
				{Row: 0, Col: 0},
			},
			wantWidth:  255,
			wantHeight: 255,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			width, height := cordonAreaSize(test.cordon)
			assert.Equal(t, test.wantWidth, width, "width")
			assert.Equal(t, test.wantHeight, height, "height")
		})
	}
}

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
