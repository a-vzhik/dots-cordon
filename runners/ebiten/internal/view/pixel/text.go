package pixel

import (
	"image/color"
	"unicode"

	"github.com/hajimehoshi/ebiten/v2"
)

// Each row stores five authored pixels, most significant bit on the left.
var glyphs = map[rune][7]byte{
	' ': {},
	'A': {14, 17, 17, 31, 17, 17, 17}, 'B': {30, 17, 17, 30, 17, 17, 30},
	'C': {14, 17, 16, 16, 16, 17, 14}, 'D': {30, 17, 17, 17, 17, 17, 30},
	'E': {31, 16, 16, 30, 16, 16, 31}, 'F': {31, 16, 16, 30, 16, 16, 16},
	'G': {14, 17, 16, 23, 17, 17, 15}, 'H': {17, 17, 17, 31, 17, 17, 17},
	'I': {14, 4, 4, 4, 4, 4, 14}, 'J': {7, 2, 2, 2, 18, 18, 12},
	'K': {17, 18, 20, 24, 20, 18, 17}, 'L': {16, 16, 16, 16, 16, 16, 31},
	'M': {17, 27, 21, 21, 17, 17, 17}, 'N': {17, 25, 21, 19, 17, 17, 17},
	'O': {14, 17, 17, 17, 17, 17, 14}, 'P': {30, 17, 17, 30, 16, 16, 16},
	'Q': {14, 17, 17, 17, 21, 18, 13}, 'R': {30, 17, 17, 30, 20, 18, 17},
	'S': {15, 16, 16, 14, 1, 1, 30}, 'T': {31, 4, 4, 4, 4, 4, 4},
	'U': {17, 17, 17, 17, 17, 17, 14}, 'V': {17, 17, 17, 17, 17, 10, 4},
	'W': {17, 17, 17, 21, 21, 21, 10}, 'X': {17, 17, 10, 4, 10, 17, 17},
	'Y': {17, 17, 10, 4, 4, 4, 4}, 'Z': {31, 1, 2, 4, 8, 16, 31},
	'0': {14, 17, 19, 21, 25, 17, 14}, '1': {4, 12, 4, 4, 4, 4, 14},
	'2': {14, 17, 1, 2, 4, 8, 31}, '3': {30, 1, 1, 14, 1, 1, 30},
	'4': {2, 6, 10, 18, 31, 2, 2}, '5': {31, 16, 16, 30, 1, 1, 30},
	'6': {14, 16, 16, 30, 17, 17, 14}, '7': {31, 1, 2, 4, 8, 8, 8},
	'8': {14, 17, 17, 14, 17, 17, 14}, '9': {14, 17, 17, 15, 1, 1, 14},
	'.': {0, 0, 0, 0, 0, 12, 12}, ',': {0, 0, 0, 0, 4, 4, 8},
	':': {0, 12, 12, 0, 12, 12, 0}, ';': {0, 12, 12, 0, 4, 4, 8},
	'!': {4, 4, 4, 4, 4, 0, 4}, '?': {14, 17, 1, 2, 4, 0, 4},
	'-': {0, 0, 0, 31, 0, 0, 0}, '_': {0, 0, 0, 0, 0, 0, 31},
	'+': {0, 4, 4, 31, 4, 4, 0}, '=': {0, 0, 31, 0, 31, 0, 0},
	'/': {1, 1, 2, 4, 8, 16, 16}, '\\': {16, 16, 8, 4, 2, 1, 1},
	'(': {2, 4, 8, 8, 8, 4, 2}, ')': {8, 4, 2, 2, 2, 4, 8},
	'[': {14, 8, 8, 8, 8, 8, 14}, ']': {14, 2, 2, 2, 2, 2, 14},
	'{': {3, 4, 4, 8, 4, 4, 3}, '}': {24, 4, 4, 2, 4, 4, 24},
	'<': {1, 2, 4, 8, 4, 2, 1}, '>': {16, 8, 4, 2, 4, 8, 16},
	'\'': {4, 4, 8, 0, 0, 0, 0}, '"': {10, 10, 10, 0, 0, 0, 0},
	'`': {8, 4, 2, 0, 0, 0, 0}, '*': {0, 21, 14, 31, 14, 21, 0},
	'#': {10, 10, 31, 10, 31, 10, 10}, '%': {24, 25, 2, 4, 8, 19, 3},
	'&': {12, 18, 20, 8, 21, 18, 13}, '@': {14, 17, 23, 21, 23, 16, 14},
	'$': {4, 15, 20, 14, 5, 30, 4}, '|': {4, 4, 4, 4, 4, 4, 4},
	'^': {4, 10, 17, 0, 0, 0, 0}, '~': {0, 0, 9, 22, 0, 0, 0},
}

var fontImages map[rune]*ebiten.Image

func font() map[rune]*ebiten.Image {
	if fontImages != nil {
		return fontImages
	}
	fontImages = make(map[rune]*ebiten.Image, len(glyphs))
	for r, rows := range glyphs {
		pixels := make([]byte, 5*7*4)
		for y, row := range rows {
			for x := 0; x < 5; x++ {
				if row&(1<<(4-x)) != 0 {
					i := (y*5 + x) * 4
					pixels[i], pixels[i+1], pixels[i+2], pixels[i+3] = 255, 255, 255, 255
				}
			}
		}
		im := ebiten.NewImage(5, 7)
		im.WritePixels(pixels)
		fontImages[r] = im
	}
	return fontImages
}

// Text draws 5x7 bitmap text from its top left, with a six-pixel advance and
// eight-pixel newline. Lowercase maps to uppercase; unsupported runes use '?'.
// Nonpositive scale draws nothing. Call from the Ebitengine rendering goroutine.
func Text(dst *ebiten.Image, text string, x, y, scale int, c color.Color) {
	if scale <= 0 || text == "" {
		return
	}
	images := font()
	start := x
	for _, r := range text {
		if r == '\n' {
			x = start
			y += 8 * scale
			continue
		}
		im, ok := images[unicode.ToUpper(r)]
		if !ok {
			im = images['?']
		}
		op := &ebiten.DrawImageOptions{Filter: ebiten.FilterNearest}
		op.GeoM.Scale(float64(scale), float64(scale))
		op.GeoM.Translate(float64(x), float64(y))
		op.ColorScale.ScaleWithColor(c)
		dst.DrawImage(im, op)
		x += 6 * scale
	}
}

// TextWidth returns the longest line's layout width (six pixels per rune,
// including the final spacing pixel), or zero for a nonpositive scale.
func TextWidth(text string, scale int) int {
	if scale <= 0 {
		return 0
	}
	longest, current := 0, 0
	for _, r := range text {
		if r == '\n' {
			current = 0
			continue
		}
		current += 6 * scale
		if current > longest {
			longest = current
		}
	}
	return longest
}
