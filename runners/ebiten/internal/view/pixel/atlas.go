package pixel

import (
	"image"
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
)

// Hand-authored masks: outline, highlight, body, and lower bevel. Dots are
// transparent in the center, so territory and grid remain visible through O.
var ring = []string{
	"...#####...",
	"..#HHHHH#..",
	".#HHbbbbs#.",
	"#Hb#####ss#",
	"#Hb#...#bs#",
	"#Hb#...#bs#",
	"#bb#...#bs#",
	"#sb#####ss#",
	".#sbbbbss#.",
	"..#sssss#..",
	"...#####...",
}

var cross = []string{
	".##.....##.",
	"#HH#...#Hb#",
	"#Hbb#.#Hbs#",
	".#Hbb#Hbs#.",
	"..#Hbbbs#..",
	"...#Hbs#...",
	"..#Hbbbs#..",
	".#Hbs#bbs#.",
	"#Hbs#.#bbs#",
	"#ss#...#ss#",
	".##.....##.",
}

var reticle = []string{
	"#####.......#####",
	"#HHH#.......#HHH#",
	"#Hb##.......##bH#",
	"#Hb#.........#bH#",
	"####.........####",
	".................",
	".......###.......",
	"......#HHb#......",
	"......#H.b#......",
	"......#bbs#......",
	".......###.......",
	".................",
	"####.........####",
	"#Hb#.........#bs#",
	"#Hb##.......##bs#",
	"#bbb#.......#bss#",
	"#####.......#####",
}

var blocked = []string{
	"#####.......#####",
	"#bbb#.......#bbb#",
	"#bs##.......##sb#",
	"#bs#.........#sb#",
	"####.........####",
	".................",
	"......##.##......",
	"......#HbH#......",
	".......#b#.......",
	"......#bsb#......",
	"......##.##......",
	".................",
	"####.........####",
	"#bs#.........#sb#",
	"#bs##.......##sb#",
	"#sss#.......#sss#",
	"#####.......#####",
}

// Atlas owns reusable GPU images; create it once, not once per frame.
type Atlas struct {
	dots    [2][2]*ebiten.Image
	cursors [2]*ebiten.Image
	tile    *ebiten.Image
}

func bitmap(rows []string, colors map[byte]color.NRGBA) *ebiten.Image {
	im := image.NewNRGBA(image.Rect(0, 0, len(rows[0]), len(rows)))
	for y, row := range rows {
		for x := range row {
			if c, ok := colors[row[x]]; ok {
				im.SetNRGBA(x, y, c)
			}
		}
	}
	return ebiten.NewImageFromImage(im)
}

// NewAtlas builds the two player O/X pairs, hover states, and an 8x8 board tile.
// Palette changes should be made before construction.
func NewAtlas() *Atlas {
	a := &Atlas{}
	for player, body := range []color.NRGBA{Red, Blue} {
		highlight := color.NRGBA{R: 255, G: 168, B: 151, A: 255}
		shadow := color.NRGBA{R: 151, G: 39, B: 66, A: 255}
		if player == 1 {
			highlight = color.NRGBA{R: 167, G: 227, B: 255, A: 255}
			shadow = color.NRGBA{R: 37, G: 80, B: 165, A: 255}
		}
		colors := map[byte]color.NRGBA{'#': Ink, 'H': highlight, 'b': body, 's': shadow}
		a.dots[player][0] = bitmap(ring, colors)
		a.dots[player][1] = bitmap(cross, colors)
	}
	a.cursors[0] = bitmap(blocked, map[byte]color.NRGBA{'#': Ink, 'H': Cream, 'b': Muted, 's': Panel})
	a.cursors[1] = bitmap(reticle, map[byte]color.NRGBA{'#': Ink, 'H': Cream, 'b': Gold, 's': {R: 163, G: 113, B: 47, A: 255}})
	a.tile = bitmap([]string{
		"........", "..l.....", "......d.", "........",
		"........", ".d......", ".....l..", "........",
	}, map[byte]color.NRGBA{
		'.': Board,
		'l': {R: 197, G: 195, B: 172, A: 255},
		'd': {R: 186, G: 185, B: 162, A: 255},
	})
	return a
}

func draw(dst, sprite *ebiten.Image, x, y int) {
	op := &ebiten.DrawImageOptions{Filter: ebiten.FilterNearest}
	op.GeoM.Translate(float64(x), float64(y))
	dst.DrawImage(sprite, op)
}

// DrawDot draws an 11x11 O (live) or X (dead), centered at x,y.
// Player zero is red; player one (and other values) is blue.
func (a *Atlas) DrawDot(dst *ebiten.Image, x, y, player int, dead bool) {
	p, state := 0, 0
	if player != 0 {
		p = 1
	}
	if dead {
		state = 1
	}
	draw(dst, a.dots[p][state], x-5, y-5)
}

// DrawCursor draws a centered 17x17 gold preview or muted unavailable reticle.
func (a *Atlas) DrawCursor(dst *ebiten.Image, x, y int, available bool) {
	state := 0
	if available {
		state = 1
	}
	draw(dst, a.cursors[state], x-8, y-8)
}

// DrawTile draws the opaque, repeating 8x8 board tile with its top left at x,y.
func (a *Atlas) DrawTile(dst *ebiten.Image, x, y int) {
	draw(dst, a.tile, x, y)
}
