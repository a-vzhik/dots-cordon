// Package view draws the game using portable Ebitengine APIs.
package view

import (
	"fmt"
	"image/color"
	"strings"
	"unicode"

	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/session"
	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/view/geometry"
	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/view/pixel"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const (
	Width  = geometry.Left*2 + (session.Columns-1)*geometry.Spacing
	Height = geometry.Top + (session.Rows-1)*geometry.Spacing + 144
)

type Game struct {
	controller *session.Controller
	snapshot   session.Snapshot
	moveErr    error
	canvas     *ebiten.Image
	background *ebiten.Image
	atlas      *pixel.Atlas
}

var _ ebiten.Game = (*Game)(nil)

func New(controller *session.Controller) *Game {
	ebiten.SetScreenFilterEnabled(false)
	g := &Game{controller: controller, snapshot: controller.Snapshot(), canvas: ebiten.NewImage(Width/2, Height/2), atlas: pixel.NewAtlas()}
	g.background = g.makeBackground()
	return g
}

func (g *Game) Layout(_, _ int) (int, int) { return Width, Height }

func (g *Game) canMove() bool {
	return !g.snapshot.Terminal && g.snapshot.NextPlayer == 0 && !g.controller.Thinking() && g.controller.Err() == nil && g.moveErr == nil
}

func (g *Game) legal(row, col int) bool {
	return row < len(g.snapshot.Dots) && col < len(g.snapshot.Dots[row]) && !g.snapshot.Dots[row][col].Owned && !g.snapshot.Dots[row][col].Killed
}

func (g *Game) Update() error {
	// Check availability before polling too: clicks made during an AI turn
	// must not become moves just because its result arrived in the same tick.
	available := g.canMove()
	g.controller.Update()
	g.snapshot = g.controller.Snapshot()
	if available && g.canMove() && inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		x, y := ebiten.CursorPosition()
		if row, col, ok := geometry.Hit(x, y, session.Rows, session.Columns); ok && g.legal(row, col) {
			g.moveErr = g.controller.HumanMove(uint8(row), uint8(col))
			g.controller.Update()
			g.snapshot = g.controller.Snapshot()
		}
	}
	return nil
}

func playerColor(player engine.PlayerIndex) color.NRGBA {
	if player == 0 {
		return pixel.Red
	}
	return pixel.Blue
}

func (g *Game) Draw(screen *ebiten.Image) {
	dst := g.canvas
	dst.DrawImage(g.background, nil)
	// Capture order matters: newer translucent territory overlays older territory.
	for _, cordon := range g.snapshot.Cordons {
		if len(cordon.Points) < 3 {
			continue
		}
		var path vector.Path
		for i, p := range cordon.Points {
			x, y := geometry.Position(int(p.Row), int(p.Col))
			if i == 0 {
				path.MoveTo(x/2, y/2)
			} else {
				path.LineTo(x/2, y/2)
			}
		}
		path.Close()
		c := playerColor(cordon.Player)
		c.A = 64
		opts := &vector.DrawPathOptions{AntiAlias: false}
		opts.ColorScale.ScaleWithColor(c)
		vector.FillPath(dst, &path, &vector.FillOptions{FillRule: vector.FillRuleEvenOdd}, opts)
		// Explicitly close the stepped outline, including the last-to-first edge.
		for i, p := range cordon.Points {
			q := cordon.Points[(i+1)%len(cordon.Points)]
			x, y := geometry.Position(int(p.Row), int(p.Col))
			xx, yy := geometry.Position(int(q.Row), int(q.Col))
			pixelLine(dst, int(x)/2, int(y)/2, int(xx)/2, int(yy)/2, playerColor(cordon.Player))
		}
	}
	// All symbols sit above all territory; dead empty intersections have no glyph.
	for row, dots := range g.snapshot.Dots {
		for col, dot := range dots {
			if !dot.Owned {
				continue
			}
			x, y := geometry.Position(row, col)
			g.atlas.DrawDot(dst, int(x)/2, int(y)/2, int(dot.Owner), dot.Killed)
		}
	}
	x, y := ebiten.CursorPosition()
	if row, col, ok := geometry.Hit(x, y, session.Rows, session.Columns); ok {
		px, py := geometry.Position(row, col)
		g.atlas.DrawCursor(dst, int(px)/2, int(py)/2, g.canMove() && g.legal(row, col))
	}
	g.drawStatus(dst)
	op := &ebiten.DrawImageOptions{Filter: ebiten.FilterNearest}
	op.GeoM.Scale(2, 2)
	screen.DrawImage(dst, op)
}

func (g *Game) drawStatus(dst *ebiten.Image) {
	const top = 246
	for player := 0; player < 2; player++ {
		x := 12 + player*170
		c := playerColor(engine.PlayerIndex(player))
		rect(dst, x, top, 162, 26, pixel.Ink)
		rect(dst, x+1, top+1, 160, 24, pixel.Panel)
		rect(dst, x+1, top+1, 2, 24, c)
		g.atlas.DrawDot(dst, x+13, top+13, player, false)
		label := "RED / YOU"
		if player == 1 {
			label = "BLUE / CPU"
		}
		pixel.Text(dst, label, x+24, top+4, 1, pixel.Cream)
		if !g.snapshot.Terminal && int(g.snapshot.NextPlayer) == player {
			rect(dst, x+24, top+15, 49, 9, pixel.Gold)
			pixel.Text(dst, "TURN", x+36, top+16, 1, pixel.Ink)
		} else {
			pixel.Text(dst, "CAPTURED", x+24, top+16, 1, pixel.Muted)
		}
		score := fmt.Sprintf("%d", g.snapshot.Scores[player])
		pixel.Text(dst, score, x+153-pixel.TextWidth(score, 2), top+6, 2, pixel.Cream)
	}
	status := "YOUR TURN - CLICK AN EMPTY POINT"
	hint := "ENCLOSE RIVALS TO SCORE  /  O LIVE  X LOST"
	c := pixel.Cream
	var err error
	switch {
	case g.controller.Err() != nil:
		err = g.controller.Err()
	case g.moveErr != nil:
		err = g.moveErr
	case g.snapshot.Terminal:
		status = "GAME OVER - DRAW"
		if g.snapshot.Scores[0] > g.snapshot.Scores[1] {
			status = "GAME OVER - RED WINS"
		}
		if g.snapshot.Scores[1] > g.snapshot.Scores[0] {
			status = "GAME OVER - BLUE WINS"
		}
		hint = "FINAL SCORE ABOVE  /  THANKS FOR PLAYING"
		c = pixel.Gold
	case g.controller.Thinking():
		status = "CPU TURN - THINKING..."
	case g.snapshot.NextPlayer == 1:
		status = "CPU TURN - BLUE TO PLAY"
	}
	if err != nil {
		pixel.Text(dst, wrapStatus("ERROR: "+err.Error(), 55, 3), 12, 276, 1, pixel.Cream)
		return
	}
	pixel.Text(dst, status, 12, 277, 1, c)
	pixel.Text(dst, hint, 12, 289, 1, pixel.Cream)
}

// Keep long errors within the fixed panel, including unbroken paths and URLs.
func wrapStatus(text string, width, height int) string {
	text = strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return ' '
		}
		if r < 32 || r > 126 {
			return '?'
		}
		return unicode.ToUpper(r)
	}, text)
	var lines []string
	line := ""
	for _, word := range strings.Fields(text) {
		if line != "" && len(line)+1+len(word) > width {
			lines = append(lines, line)
			line = ""
		}
		for len(word) > width {
			lines = append(lines, word[:width])
			word = word[width:]
		}
		if line != "" {
			line += " "
		}
		line += word
	}
	if line != "" {
		lines = append(lines, line)
	}
	if len(lines) > height {
		lines = lines[:height]
		last := lines[height-1]
		if len(last) > width-3 {
			last = last[:width-3]
		}
		lines[height-1] = last + "..."
	}
	return strings.Join(lines, "\n")
}
