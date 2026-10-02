// Package geometry provides display-independent board coordinates.
package geometry

import "math"

const (
	Spacing = 44
	Left    = 48
	Top     = 64
)

func Position(row, col int) (float32, float32) {
	return float32(Left + col*Spacing), float32(Top + row*Spacing)
}

// Hit returns the nearest intersection only inside its circular picking radius.
func Hit(x, y, rows, columns int) (row, col int, ok bool) {
	col = int(math.Round(float64(x-Left) / Spacing))
	row = int(math.Round(float64(y-Top) / Spacing))
	if row < 0 || row >= rows || col < 0 || col >= columns {
		return 0, 0, false
	}
	px, py := Position(row, col)
	dx, dy := float64(x)-float64(px), float64(y)-float64(py)
	return row, col, dx*dx+dy*dy <= float64(Spacing*Spacing)/9
}
