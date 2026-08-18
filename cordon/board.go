package cordon

// Player identifies one of the two sides.
type Player uint8

const (
	PlayerA Player = 1
	PlayerB Player = 2
)

// Opponent returns the other player.
func (p Player) Opponent() Player {
	if p == PlayerA {
		return PlayerB
	}
	return PlayerA
}

// Cell is the contents of a board intersection.
type Cell uint8

const (
	CellEmpty Cell = iota
	CellLiveA
	CellLiveB
	CellDeadA
	CellDeadB
)

// Point is a board intersection. Y increases downward.
type Point struct {
	X int
	Y int
}

// Board is a rectangular grid of intersections.
type Board struct {
	Width  int
	Height int
	Cells  []Cell
}

// NewBoard returns an empty board with the given dimensions.
func NewBoard(width, height int) *Board {
	if width <= 0 || height <= 0 {
		panic("cordon: invalid board size")
	}
	return &Board{
		Width:  width,
		Height: height,
		Cells:  make([]Cell, width*height),
	}
}

func (b *Board) index(p Point) int {
	return p.Y*b.Width + p.X
}

func (b *Board) inBounds(p Point) bool {
	return p.X >= 0 && p.X < b.Width && p.Y >= 0 && p.Y < b.Height
}

// Get returns the cell at p.
func (b *Board) Get(p Point) Cell {
	if !b.inBounds(p) {
		panic("cordon: point out of bounds")
	}
	return b.Cells[b.index(p)]
}

// Set writes the cell at p.
func (b *Board) Set(p Point, c Cell) {
	if !b.inBounds(p) {
		panic("cordon: point out of bounds")
	}
	b.Cells[b.index(p)] = c
}

func liveCell(player Player) Cell {
	if player == PlayerA {
		return CellLiveA
	}
	return CellLiveB
}

func deadCell(player Player) Cell {
	if player == PlayerA {
		return CellDeadA
	}
	return CellDeadB
}

func (b *Board) isWalkableForEscape(c Cell, escaper Player) bool {
	switch c {
	case CellEmpty:
		return true
	case CellLiveA:
		return escaper == PlayerA
	case CellLiveB:
		return escaper == PlayerB
	default:
		return false
	}
}

func (b *Board) isMoverDot(c Cell, mover Player) bool {
	return c == liveCell(mover)
}

var cardinals = []Point{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}

var neighbors8 = []Point{
	{-1, -1}, {0, -1}, {1, -1},
	{-1, 0}, {1, 0},
	{-1, 1}, {0, 1}, {1, 1},
}
