// Package pixel supplies authored bitmap art for the 356x302 game canvas.
// Coordinates and text scales are integral canvas pixels. The caller scales
// the completed canvas by two with ebiten.FilterNearest and can disable window
// smoothing with ebiten.SetScreenFilterEnabled(false).
package pixel

import "image/color"

var (
	Ink        = color.NRGBA{R: 12, G: 18, B: 33, A: 255}
	Background = color.NRGBA{R: 20, G: 29, B: 48, A: 255}
	Panel      = color.NRGBA{R: 35, G: 48, B: 69, A: 255}
	Board      = color.NRGBA{R: 192, G: 190, B: 166, A: 255}
	Grid       = color.NRGBA{R: 130, G: 145, B: 146, A: 255}
	Muted      = color.NRGBA{R: 142, G: 157, B: 171, A: 255}
	Cream      = color.NRGBA{R: 255, G: 239, B: 202, A: 255}
	Gold       = color.NRGBA{R: 239, G: 187, B: 83, A: 255}
	Red        = color.NRGBA{R: 241, G: 75, B: 88, A: 255}
	Blue       = color.NRGBA{R: 66, G: 161, B: 255, A: 255}
)
