package engine

import (
	"fmt"
	"strings"
)

type GameField struct {
	Width  uint8
	Height uint8
	Dots   [][]Dot
}

func NewGameField(width uint8, height uint8) *GameField {
	dots := make([][]Dot, int(height))
	for rowIdx := range dots {
		row := make([]Dot, int(width))
		for colIdx := range row {
			row[colIdx] = Dot{
				Coord: Coord{
					Col: uint8(colIdx),
					Row: uint8(rowIdx),
				},
			}
		}

		dots[rowIdx] = row
	}

	return &GameField{
		Width:  width,
		Height: height,
		Dots:   dots,
	}
}

func (gf *GameField) ToString() string {
	var result strings.Builder

	result.WriteString("    ")
	for colIdx := 0; colIdx < int(gf.Width); colIdx++ {
		if colIdx > 0 {
			result.WriteByte(' ')
		}
		fmt.Fprintf(&result, "%02d", colIdx)
	}

	for rowIdx, row := range gf.Dots {
		fmt.Fprintf(&result, "\n%02d  ", rowIdx)
		for colIdx, dot := range row {
			if colIdx > 0 {
				result.WriteString("  ")
			}

			switch {
			case dot.Killed:
				result.WriteByte('K')
			case dot.Owned:
				fmt.Fprintf(&result, "%d", dot.Owner)
			default:
				result.WriteByte('.')
			}
		}
	}

	return result.String()
}

func (gf *GameField) GetAt(coord Coord) Dot {
	return gf.Dots[coord.Row][coord.Col]
}

func (gf *GameField) Transform(row uint8, col uint8, applyFunc func(Dot) Dot) {
	oldDot := gf.Dots[row][col]
	gf.Dots[row][col] = applyFunc(oldDot)
}

func (gf *GameField) TransformMany(sourceDots []Dot, applyFunc func(Dot) Dot) {
	for _, source := range sourceDots {
		gf.Dots[source.Row][source.Col] = applyFunc(source)
	}
}

func (gf *GameField) TransformFunc(filterFunc func(Dot) bool, applyFunc func(Dot) Dot) {
	for rowIdx, row := range gf.Dots {
		for colIdx, oldDot := range row {

			if !filterFunc(oldDot) {
				continue
			}

			newDot := applyFunc(oldDot)
			gf.Dots[rowIdx][colIdx] = newDot
		}
	}
}

func (gf *GameField) IsFull() bool {
	emptyDots := gf.Find(func(dot Dot) bool {
		return !dot.Killed && !dot.Owned
	})

	return len(emptyDots) == 0
}

func (gf *GameField) Find(filterFunc func(Dot) bool) []Dot {
	filtered := make([]Dot, 0)
	for _, row := range gf.Dots {
		for _, dot := range row {

			if !filterFunc(dot) {
				continue
			}

			filtered = append(filtered, dot)

		}
	}
	return filtered
}
