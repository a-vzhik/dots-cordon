package view

import (
	"image/color"

	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/session"
	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/view/geometry"
	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/view/pixel"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

func rect(dst *ebiten.Image, x, y, w, h int, c color.Color) {
	vector.DrawFilledRect(dst, float32(x), float32(y), float32(w), float32(h), c, false)
}

// Integer Bresenham edges preserve crisp staircase diagonals at the art resolution.
func pixelLine(dst *ebiten.Image, x, y, ex, ey int, c color.Color) {
	dx, dy := ex-x, ey-y
	sx, sy := 1, 1
	if dx < 0 {
		dx = -dx
		sx = -1
	}
	if dy < 0 {
		dy = -dy
		sy = -1
	}
	err := dx - dy
	for {
		rect(dst, x, y, 1, 1, c)
		if x == ex && y == ey {
			break
		}
		e := 2 * err
		if e > -dy {
			err -= dy
			x += sx
		}
		if e < dx {
			err += dx
			y += sy
		}
	}
}

func (g *Game) makeBackground() *ebiten.Image {
	dst := ebiten.NewImage(Width/2, Height/2)
	dst.Fill(pixel.Background)
	rect(dst, 4, 4, 348, 237, pixel.Ink)
	rect(dst, 5, 5, 346, 235, pixel.Panel)
	rect(dst, 8, 7, 340, 1, pixel.Muted)
	pixel.Text(dst, "DOTS CORDON", 13, 11, 1, pixel.Cream)
	pixel.Text(dst, "ENCLOSE / CAPTURE", 237, 11, 1, pixel.Gold)
	rect(dst, 13, 22, 330, 219, pixel.Ink)
	rect(dst, 14, 23, 328, 217, pixel.Gold)
	rect(dst, 16, 25, 324, 213, pixel.Board)
	// The board inset is a multiple of the shared eight-pixel texture tile.
	for y := 26; y < 234; y += 8 {
		for x := 18; x < 338; x += 8 {
			g.atlas.DrawTile(dst, x, y)
		}
	}
	for row := 0; row < session.Rows; row++ {
		x, y := geometry.Position(row, 0)
		end, _ := geometry.Position(row, session.Columns-1)
		rect(dst, int(x)/2, int(y)/2, int(end-x)/2+1, 1, pixel.Grid)
	}
	for col := 0; col < session.Columns; col++ {
		x, y := geometry.Position(0, col)
		_, end := geometry.Position(session.Rows-1, col)
		rect(dst, int(x)/2, int(y)/2, 1, int(end-y)/2+1, pixel.Grid)
	}
	// Rivets and corner brackets belong to the housing, outside the picking area.
	for _, x := range []int{8, 346} {
		for _, y := range []int{26, 234} {
			rect(dst, x-1, y-1, 3, 3, pixel.Ink)
			rect(dst, x, y, 1, 1, pixel.Gold)
		}
	}
	return dst
}
