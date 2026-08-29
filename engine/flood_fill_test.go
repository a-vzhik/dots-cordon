package engine_test

import (
	"testing"

	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/stretchr/testify/assert"
)

func allDotsHaveFloodFillState(
	dots []engine.Dot,
	floodFillGrid [][]engine.FloodFillCellState,
	desiredState engine.FloodFillCellState,
) bool {
	for _, dot := range dots {
		if floodFillGrid[dot.Row][dot.Col] != desiredState {
			return false
		}
	}

	return true
}

func newFloodFillGrid(width uint8, height uint8) [][]engine.FloodFillCellState {
	floodFillGrid := make([][]engine.FloodFillCellState, int(height))
	for rowIdx := range floodFillGrid {
		floodFillGrid[rowIdx] = make([]engine.FloodFillCellState, int(width))
	}

	return floodFillGrid
}

func TestRunFloodFill_SupportsRectangularGrid(t *testing.T) {
	field := engine.NewGameField(10, 20)
	field.Dots[2][7] = field.Dots[2][7].WithOwner(engine.PlayerIndex(0))
	defenderDots := []engine.Dot{field.Dots[2][7]}

	floodFillGrid := engine.RunFloodFill(field, defenderDots, engine.PlayerIndex(0))

	assert.Len(t, floodFillGrid, 20)
	for _, row := range floodFillGrid {
		assert.Len(t, row, 10)
	}
	assert.Equal(t, engine.FloodFillCellStateEscaped, floodFillGrid[2][7])
}

func TestRunFloodFill_MarksEscapedRegion(t *testing.T) {
	// Grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4
	// r0    .  .  .  .  .
	// r1    .  .  .  .  .
	// r2    R  R  R  R  R
	// r3    .  .  .  .  .
	// r4    B  B  B  B  B
	//
	// R = red dot, B = blue dot, . = unowned dot.
	const (
		redPlayerIdx engine.PlayerIndex = iota
		bluePlayerIdx
	)

	field := engine.NewGameField(5, 5)
	for colIdx := uint8(0); colIdx < field.Width; colIdx++ {
		field.Dots[2][colIdx] = field.Dots[2][colIdx].WithOwner(redPlayerIdx)
		field.Dots[4][colIdx] = field.Dots[4][colIdx].WithOwner(bluePlayerIdx)
	}
	redDots := field.Find(func(dot engine.Dot) bool {
		return dot.Owned && dot.Owner == redPlayerIdx
	})

	floodFillGrid := engine.RunFloodFill(field, redDots, redPlayerIdx)

	assert.True(
		t,
		allDotsHaveFloodFillState(redDots, floodFillGrid, engine.FloodFillCellStateEscaped),
		"expected all red dots to have escaped",
	)
}

func TestRunFloodFill_MarksEnclosedRegionBlocked(t *testing.T) {
	// Grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4 c5
	// r0    .  .  .  .  .  .
	// r1    .  .  B  B  .  .
	// r2    .  B  R  R  B  .
	// r3    .  B  R  R  B  .
	// r4    .  .  B  B  .  .
	// r5    .  .  .  .  .  .
	//
	// R = red dot, B = blue dot, . = unowned dot.
	const (
		redPlayerIdx engine.PlayerIndex = iota
		bluePlayerIdx
	)

	field := engine.NewGameField(6, 6)
	for colIdx := uint8(2); colIdx <= 3; colIdx++ {
		field.Dots[1][colIdx] = field.Dots[1][colIdx].WithOwner(bluePlayerIdx)
		field.Dots[4][colIdx] = field.Dots[4][colIdx].WithOwner(bluePlayerIdx)
	}
	for rowIdx := uint8(2); rowIdx <= 3; rowIdx++ {
		field.Dots[rowIdx][1] = field.Dots[rowIdx][1].WithOwner(bluePlayerIdx)
		field.Dots[rowIdx][4] = field.Dots[rowIdx][4].WithOwner(bluePlayerIdx)
		for colIdx := uint8(2); colIdx <= 3; colIdx++ {
			field.Dots[rowIdx][colIdx] = field.Dots[rowIdx][colIdx].WithOwner(redPlayerIdx)
		}
	}
	redDots := field.Find(func(dot engine.Dot) bool {
		return dot.Owned && dot.Owner == redPlayerIdx
	})

	floodFillGrid := engine.RunFloodFill(field, redDots, redPlayerIdx)

	engine.PrintFloodFillGrid(floodFillGrid)

	assert.True(
		t,
		allDotsHaveFloodFillState(redDots, floodFillGrid, engine.FloodFillCellStateBlocked),
		"expected all red dots to be blocked",
	)
}

func TestRunFloodFill_SeparatesEnclosedAndFreeRedDots(t *testing.T) {
	// Grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4
	// r0    R  .  B  .  R
	// r1    .  B  R  B  .
	// r2    B  R  R  R  B
	// r3    .  B  .  B  .
	// r4    R  .  B  .  R
	//
	// R = red dot, B = blue dot, . = unowned dot.
	// The four red corner dots are free; the other four are enclosed.
	const (
		redPlayerIdx engine.PlayerIndex = iota
		bluePlayerIdx
	)
	type position struct {
		row uint8
		col uint8
	}

	field := engine.NewGameField(5, 5)
	placeDots := func(positions []position, owner engine.PlayerIndex) []engine.Dot {
		dots := make([]engine.Dot, 0, len(positions))
		for _, position := range positions {
			dot := field.Dots[position.row][position.col].WithOwner(owner)
			field.Dots[position.row][position.col] = dot
			dots = append(dots, dot)
		}

		return dots
	}

	_ = placeDots([]position{
		{row: 0, col: 2},
		{row: 1, col: 1},
		{row: 1, col: 3},
		{row: 2, col: 0},
		{row: 2, col: 4},
		{row: 3, col: 1},
		{row: 3, col: 3},
		{row: 4, col: 2},
	}, bluePlayerIdx)
	enclosedRedDots := placeDots([]position{
		{row: 1, col: 2},
		{row: 2, col: 1},
		{row: 2, col: 2},
		{row: 2, col: 3},
	}, redPlayerIdx)
	freeDots := placeDots([]position{
		{row: 0, col: 0},
		{row: 0, col: 4},
		{row: 4, col: 0},
		{row: 4, col: 4},
	}, redPlayerIdx)
	redDots := field.Find(func(dot engine.Dot) bool {
		return dot.Owned && dot.Owner == redPlayerIdx
	})

	floodFillGrid := engine.RunFloodFill(field, redDots, redPlayerIdx)

	assert.True(
		t,
		allDotsHaveFloodFillState(enclosedRedDots, floodFillGrid, engine.FloodFillCellStateBlocked),
		"expected all enclosed red dots to be blocked",
	)
	assert.True(
		t,
		allDotsHaveFloodFillState(freeDots, floodFillGrid, engine.FloodFillCellStateEscaped),
		"expected all free red dots to be escaped",
	)
}

func TestRunFloodFill_EscapesAllRedDotsThroughOpenRhombusHatch(t *testing.T) {
	// Grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4
	// r0    R  .  B  .  R
	// r1    .  B  R  B  .
	// r2    .  R  R  R  B
	// r3    .  B  .  B  .
	// r4    R  .  B  .  .
	//
	// R = red dot, B = blue dot, . = unowned dot.
	// Cell (2, 0) is the open hatch. There are seven dots per player.
	const (
		redPlayerIdx engine.PlayerIndex = iota
		bluePlayerIdx
	)
	type position struct {
		row uint8
		col uint8
	}

	field := engine.NewGameField(5, 5)
	placeDots := func(positions []position, owner engine.PlayerIndex) []engine.Dot {
		dots := make([]engine.Dot, 0, len(positions))
		for _, position := range positions {
			dot := field.Dots[position.row][position.col].WithOwner(owner)
			field.Dots[position.row][position.col] = dot
			dots = append(dots, dot)
		}

		return dots
	}

	blueDots := placeDots([]position{
		{row: 0, col: 2},
		{row: 1, col: 1},
		{row: 1, col: 3},
		{row: 2, col: 4},
		{row: 3, col: 1},
		{row: 3, col: 3},
		{row: 4, col: 2},
	}, bluePlayerIdx)
	_ = placeDots([]position{
		{row: 1, col: 2},
		{row: 2, col: 1},
		{row: 2, col: 2},
		{row: 2, col: 3},
		{row: 0, col: 0},
		{row: 0, col: 4},
		{row: 4, col: 0},
	}, redPlayerIdx)
	redDots := field.Find(func(dot engine.Dot) bool {
		return dot.Owned && dot.Owner == redPlayerIdx
	})

	assert.Len(t, redDots, len(blueDots), "expected equal red and blue dot counts")
	floodFillGrid := engine.RunFloodFill(field, redDots, redPlayerIdx)

	assert.True(
		t,
		allDotsHaveFloodFillState(redDots, floodFillGrid, engine.FloodFillCellStateEscaped),
		"expected all red dots to escape through the open hatch",
	)
}

func TestRunFloodFill_HandlesTwoRectanglesSharingCorner(t *testing.T) {
	// Grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4
	// r0    B  B  B  R  R
	// r1    B  R  B  R  R
	// r2    B  B  B  B  B
	// r3    R  R  B  R  B
	// r4    R  R  B  B  B
	//
	// R = red dot, B = blue dot.
	// The blue rectangles span (0,0)-(2,2) and (2,2)-(4,4),
	// sharing the corner at (2,2). Red dots (1,1) and (3,3)
	// are enclosed; every other red dot is outside both rectangles.
	const (
		redPlayerIdx engine.PlayerIndex = iota
		bluePlayerIdx
	)
	type position struct {
		row uint8
		col uint8
	}

	field := engine.NewGameField(5, 5)
	bluePositions := []position{
		{row: 0, col: 0},
		{row: 0, col: 1},
		{row: 0, col: 2},
		{row: 1, col: 0},
		{row: 1, col: 2},
		{row: 2, col: 0},
		{row: 2, col: 1},
		{row: 2, col: 2},
		{row: 2, col: 3},
		{row: 2, col: 4},
		{row: 3, col: 2},
		{row: 3, col: 4},
		{row: 4, col: 2},
		{row: 4, col: 3},
		{row: 4, col: 4},
	}
	for _, position := range bluePositions {
		field.Dots[position.row][position.col] = field.Dots[position.row][position.col].WithOwner(bluePlayerIdx)
	}

	enclosedRedDots := make([]engine.Dot, 0, 2)
	for _, position := range []position{
		{row: 1, col: 1},
		{row: 3, col: 3},
	} {
		dot := field.Dots[position.row][position.col].WithOwner(redPlayerIdx)
		field.Dots[position.row][position.col] = dot
		enclosedRedDots = append(enclosedRedDots, dot)
	}

	freeRedDots := make([]engine.Dot, 0)
	for rowIdx, row := range field.Dots {
		for colIdx, dot := range row {
			if dot.Owned {
				continue
			}

			redDot := dot.WithOwner(redPlayerIdx)
			field.Dots[rowIdx][colIdx] = redDot
			freeRedDots = append(freeRedDots, redDot)
		}
	}

	redDots := field.Find(func(dot engine.Dot) bool {
		return dot.Owned && dot.Owner == redPlayerIdx
	})

	floodFillGrid := engine.RunFloodFill(field, redDots, redPlayerIdx)

	engine.PrintFloodFillGrid(floodFillGrid)
	assert.True(
		t,
		allDotsHaveFloodFillState(enclosedRedDots, floodFillGrid, engine.FloodFillCellStateBlocked),
		"expected the red dot inside each rectangle to be blocked",
	)
	assert.True(
		t,
		allDotsHaveFloodFillState(freeRedDots, floodFillGrid, engine.FloodFillCellStateEscaped),
		"expected all red dots outside the rectangles to be escaped",
	)
}

func TestRunFloodFill_HandlesTwoDiamondsSharingCorner(t *testing.T) {
	// Grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4
	// r0    R  R  B  R  R
	// r1    R  B  R  B  R
	// r2    R  R  B  R  R
	// r3    R  B  R  B  R
	// r4    R  R  B  R  R
	//
	// R = red dot, B = blue dot. The blue diamonds enclose the red dots at
	// (1,2) and (3,2), sharing the blue corner at (2,2). Every other red dot
	// is outside both diamonds.
	const (
		redPlayerIdx engine.PlayerIndex = iota
		bluePlayerIdx
	)
	type position struct {
		row uint8
		col uint8
	}

	field := engine.NewGameField(5, 5)
	for _, position := range []position{
		{row: 0, col: 2},
		{row: 1, col: 1},
		{row: 1, col: 3},
		{row: 2, col: 2},
		{row: 3, col: 1},
		{row: 3, col: 3},
		{row: 4, col: 2},
	} {
		field.Dots[position.row][position.col] = field.Dots[position.row][position.col].WithOwner(bluePlayerIdx)
	}

	enclosedRedDots := make([]engine.Dot, 0, 2)
	for _, position := range []position{
		{row: 1, col: 2},
		{row: 3, col: 2},
	} {
		dot := field.Dots[position.row][position.col].WithOwner(redPlayerIdx)
		field.Dots[position.row][position.col] = dot
		enclosedRedDots = append(enclosedRedDots, dot)
	}

	freeRedDots := make([]engine.Dot, 0)
	for rowIdx, row := range field.Dots {
		for colIdx, dot := range row {
			if dot.Owned {
				continue
			}

			redDot := dot.WithOwner(redPlayerIdx)
			field.Dots[rowIdx][colIdx] = redDot
			freeRedDots = append(freeRedDots, redDot)
		}
	}

	redDots := field.Find(func(dot engine.Dot) bool {
		return dot.Owned && dot.Owner == redPlayerIdx
	})

	floodFillGrid := engine.RunFloodFill(field, redDots, redPlayerIdx)

	assert.True(
		t,
		allDotsHaveFloodFillState(enclosedRedDots, floodFillGrid, engine.FloodFillCellStateBlocked),
		"expected the red dot inside each diamond to be blocked",
	)
	assert.True(
		t,
		allDotsHaveFloodFillState(freeRedDots, floodFillGrid, engine.FloodFillCellStateEscaped),
		"expected all red dots outside the diamonds to be escaped",
	)
}

func TestRunFloodFill_DoesNotCloseDiamondThroughKilledDot(t *testing.T) {
	// Grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4
	// r0    .  .  .  .  .
	// r1    .  .  R  .  .
	// r2    .  R  x  R  .
	// r3    .  B  R  B  .
	// r4    .  .  B  .  .
	//
	// R = red dot, B = blue dot, x = killed blue dot, . = unowned dot.
	const (
		redPlayerIdx engine.PlayerIndex = iota
		bluePlayerIdx
	)
	type position struct {
		row uint8
		col uint8
	}

	field := engine.NewGameField(5, 5)
	for _, position := range []position{
		{row: 1, col: 2},
		{row: 2, col: 1},
		{row: 2, col: 3},
		{row: 3, col: 2},
	} {
		field.Dots[position.row][position.col] = field.Dots[position.row][position.col].WithOwner(redPlayerIdx)
	}
	for _, position := range []position{
		{row: 3, col: 1},
		{row: 3, col: 3},
		{row: 4, col: 2},
	} {
		field.Dots[position.row][position.col] = field.Dots[position.row][position.col].WithOwner(bluePlayerIdx)
	}
	field.Dots[2][2] = field.Dots[2][2].WithOwner(bluePlayerIdx).WithKilled()

	redDots := field.Find(func(dot engine.Dot) bool {
		return !dot.Killed && dot.Owned && dot.Owner == redPlayerIdx
	})
	floodFillGrid := engine.RunFloodFill(field, redDots, redPlayerIdx)

	engine.PrintFloodFillGrid(floodFillGrid)

	const (
		N = engine.FloodFillCellStateNone
		B = engine.FloodFillCellStateBlocked
		E = engine.FloodFillCellStateEscaped
		K = engine.FloodFillCellStateKilled
		P = engine.FloodFillCellStatePotentialBlocked
	)
	assert.Equal(t, [][]engine.FloodFillCellState{
		{N, E, E, E, N},
		{N, E, E, E, N},
		{N, E, E, E, N},
		{N, P, E, P, N},
		{N, N, P, N, N},
	}, floodFillGrid)
}

func TestRunFloodFill_EscapesAllRedDotsThroughBorderHatch(t *testing.T) {
	// Grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4
	// r0    B  B  .  B  B
	// r1    B  R  R  R  B
	// r2    B  R  B  R  B
	// r3    B  R  R  R  B
	// r4    B  B  B  B  B
	//
	// R = red dot, B = blue dot, . = unowned dot.
	// The blue grid border is open at (0,2), allowing the red region to
	// escape around the additional blue dot at (2,2).
	const (
		redPlayerIdx engine.PlayerIndex = iota
		bluePlayerIdx
	)

	field := engine.NewGameField(5, 5)
	for rowIdx, row := range field.Dots {
		for colIdx, dot := range row {
			isHatch := rowIdx == 0 && colIdx == 2
			if isHatch {
				continue
			}

			isBorder := rowIdx == 0 || rowIdx == len(field.Dots)-1 || colIdx == 0 || colIdx == len(row)-1
			isInnerBlue := rowIdx == 2 && colIdx == 2
			if isBorder || isInnerBlue {
				field.Dots[rowIdx][colIdx] = dot.WithOwner(bluePlayerIdx)
				continue
			}

			field.Dots[rowIdx][colIdx] = dot.WithOwner(redPlayerIdx)
		}
	}

	redDots := field.Find(func(dot engine.Dot) bool {
		return dot.Owned && dot.Owner == redPlayerIdx
	})

	assert.Len(t, redDots, 8)
	floodFillGrid := engine.RunFloodFill(field, redDots, redPlayerIdx)

	assert.True(
		t,
		allDotsHaveFloodFillState(redDots, floodFillGrid, engine.FloodFillCellStateEscaped),
		"expected all red dots to escape through the border hatch",
	)
}

func TestRunFloodFill_ClassifiesRandomSevenBySevenGrid(t *testing.T) {
	// Fixed snapshot of a random alternating allocation.
	//
	//       c0 c1 c2 c3 c4 c5 c6
	// r0    R  B  R  R  R  B  R
	// r1    R  B  R  R  B  R  B
	// r2    R  B  R  B  R  R  B
	// r3    B  R  B  B  B  R  B
	// r4    R  B  B  R  B  B  B
	// r5    R  B  B  B  R  B  R
	// r6    R  R  R  R  B  B  B
	const (
		redPlayerIdx engine.PlayerIndex = iota
		bluePlayerIdx
	)
	type position struct {
		row uint8
		col uint8
	}

	allocation := []string{
		"RBRRRBR",
		"RBRRBRB",
		"RBRBRRB",
		"BRBBBRB",
		"RBBRBBB",
		"RBBBRBR",
		"RRRRBBB",
	}
	field := engine.NewGameField(7, 7)
	for rowIdx, row := range allocation {
		for colIdx := range row {
			owner := redPlayerIdx
			if row[colIdx] == 'B' {
				owner = bluePlayerIdx
			}
			field.Dots[rowIdx][colIdx] = field.Dots[rowIdx][colIdx].WithOwner(owner)
		}
	}

	trappedPositions := map[position]struct{}{
		{row: 1, col: 5}: {},
		{row: 2, col: 4}: {},
		{row: 2, col: 5}: {},
		{row: 3, col: 1}: {},
		{row: 3, col: 5}: {},
		{row: 4, col: 3}: {},
		{row: 5, col: 4}: {},
	}
	redDots := field.Find(func(dot engine.Dot) bool {
		return dot.Owned && dot.Owner == redPlayerIdx
	})
	trappedRedDots := make([]engine.Dot, 0, len(trappedPositions))
	escapedRedDots := make([]engine.Dot, 0, len(redDots)-len(trappedPositions))
	for _, dot := range redDots {
		_, isTrapped := trappedPositions[position{row: dot.Row, col: dot.Col}]
		if isTrapped {
			trappedRedDots = append(trappedRedDots, dot)
			continue
		}
		escapedRedDots = append(escapedRedDots, dot)
	}

	assert.Len(t, trappedRedDots, 7)
	assert.Len(t, escapedRedDots, 17)
	floodFillGrid := engine.RunFloodFill(field, redDots, redPlayerIdx)

	assert.True(
		t,
		allDotsHaveFloodFillState(trappedRedDots, floodFillGrid, engine.FloodFillCellStateBlocked),
		"expected the seven cordoned red dots to be blocked",
	)
	assert.True(
		t,
		allDotsHaveFloodFillState(escapedRedDots, floodFillGrid, engine.FloodFillCellStateEscaped),
		"expected every other red dot to be escaped",
	)
}

func TestRunFloodFill_ClassifiesSwappedRandomSevenBySevenGrid(t *testing.T) {
	// Fixed snapshot of a random alternating allocation after swapping
	// (6,3) and (6,4) to close the lower-middle cordon.
	//
	//       c0 c1 c2 c3 c4 c5 c6
	// r0    R  B  B  R  B  B  R
	// r1    R  B  R  B  R  B  B
	// r2    R  B  R  B  B  R  R
	// r3    B  R  R  B  B  R  R
	// r4    R  B  B  R  R  B  R
	// r5    B  B  R  R  B  R  R
	// r6    R  R  B  B  R  B  B
	const (
		redPlayerIdx engine.PlayerIndex = iota
		bluePlayerIdx
	)
	type position struct {
		row uint8
		col uint8
	}

	allocation := []string{
		"RBBRBBR",
		"RBRBRBB",
		"RBRBBRR",
		"BRRBBRR",
		"RBBRRBR",
		"BBRRBRR",
		"RRBBRBB",
	}
	field := engine.NewGameField(7, 7)
	for rowIdx, row := range allocation {
		for colIdx := range row {
			owner := redPlayerIdx
			if row[colIdx] == 'B' {
				owner = bluePlayerIdx
			}
			field.Dots[rowIdx][colIdx] = field.Dots[rowIdx][colIdx].WithOwner(owner)
		}
	}

	trappedPositions := map[position]struct{}{
		{row: 1, col: 2}: {},
		{row: 1, col: 4}: {},
		{row: 2, col: 2}: {},
		{row: 3, col: 1}: {},
		{row: 3, col: 2}: {},
		{row: 4, col: 3}: {},
		{row: 4, col: 4}: {},
		{row: 5, col: 2}: {},
		{row: 5, col: 3}: {},
	}
	redDots := field.Find(func(dot engine.Dot) bool {
		return dot.Owned && dot.Owner == redPlayerIdx
	})
	trappedRedDots := make([]engine.Dot, 0, len(trappedPositions))
	escapedRedDots := make([]engine.Dot, 0, len(redDots)-len(trappedPositions))
	for _, dot := range redDots {
		_, isTrapped := trappedPositions[position{row: dot.Row, col: dot.Col}]
		if isTrapped {
			trappedRedDots = append(trappedRedDots, dot)
			continue
		}
		escapedRedDots = append(escapedRedDots, dot)
	}

	assert.Len(t, trappedRedDots, 9)
	assert.Len(t, escapedRedDots, 16)
	floodFillGrid := engine.RunFloodFill(field, redDots, redPlayerIdx)

	assert.True(
		t,
		allDotsHaveFloodFillState(trappedRedDots, floodFillGrid, engine.FloodFillCellStateBlocked),
		"expected the nine cordoned red dots to be blocked",
	)
	assert.True(
		t,
		allDotsHaveFloodFillState(escapedRedDots, floodFillGrid, engine.FloodFillCellStateEscaped),
		"expected every other red dot to be escaped",
	)
}
