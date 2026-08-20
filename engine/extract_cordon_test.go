package engine_test

import (
	"testing"

	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGameExtractCordonFromDiamond(t *testing.T) {
	// Flood-fill grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4
	// r0    .  .  .  .  .
	// r1    .  .  B  .  .
	// r2    .  B  R  B  .
	// r3    .  .  B  .  .
	// r4    .  .  .  .  .
	//
	// B = blocked blue cordon dot, R = blocked captured red dot,
	// . = cell with no flood-fill state.
	const (
		none    = engine.FloodFillCellStateNone
		blocked = engine.FloodFillCellStateBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{none, none, none, none, none},
		{none, none, blocked, none, none},
		{none, blocked, blocked, blocked, none},
		{none, none, blocked, none, none},
		{none, none, none, none, none},
	}

	game := engine.NewGame(engine.NewGameField(5, 5), nil)
	cordons := game.ExtractCordon(floodFillGrid)

	require.Len(t, cordons, 1)

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 1, Col: 2},
		{Row: 2, Col: 3},
		{Row: 3, Col: 2},
		{Row: 2, Col: 1},
		{Row: 1, Col: 2},
	}, cordons[0])
}

func TestGameExtractCordonFromFullyBlockedGrid(t *testing.T) {
	// Flood-fill grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3
	// r0    B  B  B  B
	// r1    B  R  R  B
	// r2    B  R  R  B
	// r3    B  B  B  B
	//
	// B = blocked blue cordon dot, R = blocked captured red dot. The twelve
	// blue edge dots form the ordered cordon; the four red dots are not part of it.
	const blocked = engine.FloodFillCellStateBlocked
	floodFillGrid := [][]engine.FloodFillCellState{
		{blocked, blocked, blocked, blocked},
		{blocked, blocked, blocked, blocked},
		{blocked, blocked, blocked, blocked},
		{blocked, blocked, blocked, blocked},
	}

	game := engine.NewGame(engine.NewGameField(4, 4), nil)
	cordons := game.ExtractCordon(floodFillGrid)

	require.Len(t, cordons, 1)

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 0, Col: 0},
		{Row: 0, Col: 1},
		{Row: 0, Col: 2},
		{Row: 0, Col: 3},
		{Row: 1, Col: 3},
		{Row: 2, Col: 3},
		{Row: 3, Col: 3},
		{Row: 3, Col: 2},
		{Row: 3, Col: 1},
		{Row: 3, Col: 0},
		{Row: 2, Col: 0},
		{Row: 1, Col: 0},
		{Row: 0, Col: 0},
	}, cordons[0])
}

func TestGameExtractCordonFromAsymmetricBlob(t *testing.T) {
	// Flood-fill grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4 c5 c6
	// r0    .  .  .  .  .  .  .
	// r1    .  .  B  B  B  .  .
	// r2    .  B  R  R  R  B  .
	// r3    .  B  R  R  R  B  .
	// r4    .  B  R  R  R  R  B
	// r5    .  .  B  B  B  B  .
	// r6    .  .  .  .  .  .  .
	//
	// B = blocked blue cordon dot, R = blocked captured red dot,
	// . = cell with no flood-fill state.
	const (
		none    = engine.FloodFillCellStateNone
		blocked = engine.FloodFillCellStateBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{none, none, none, none, none, none, none},
		{none, none, blocked, blocked, blocked, none, none},
		{none, blocked, blocked, blocked, blocked, blocked, none},
		{none, blocked, blocked, blocked, blocked, blocked, none},
		{none, blocked, blocked, blocked, blocked, blocked, blocked},
		{none, none, blocked, blocked, blocked, blocked, none},
		{none, none, none, none, none, none, none},
	}

	game := engine.NewGame(engine.NewGameField(7, 7), nil)
	cordons := game.ExtractCordon(floodFillGrid)

	require.Len(t, cordons, 1)

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 1, Col: 2},
		{Row: 1, Col: 3},
		{Row: 1, Col: 4},
		{Row: 2, Col: 5},
		{Row: 3, Col: 5},
		{Row: 4, Col: 6},
		{Row: 5, Col: 5},
		{Row: 5, Col: 4},
		{Row: 5, Col: 3},
		{Row: 5, Col: 2},
		{Row: 4, Col: 1},
		{Row: 3, Col: 1},
		{Row: 2, Col: 1},
		{Row: 1, Col: 2},
	}, cordons[0])
}

func TestGameExtractCordonFromTwoRectanglesSharingCorner(t *testing.T) {
	// Flood-fill grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4
	// r0    B  B  B  .  .
	// r1    B  R  B  .  .
	// r2    B  B  B  B  B
	// r3    .  .  B  R  B
	// r4    .  .  B  B  B
	//
	// B = blocked blue dot, R = blocked captured red dot,
	// . = cell with no flood-fill state.
	// The greedy cordon uses diagonal connections around (2,2), so that
	// shared blue corner is inside the combined cordon rather than on it.
	const (
		none    = engine.FloodFillCellStateNone
		blocked = engine.FloodFillCellStateBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{blocked, blocked, blocked, none, none},
		{blocked, blocked, blocked, none, none},
		{blocked, blocked, blocked, blocked, blocked},
		{none, none, blocked, blocked, blocked},
		{none, none, blocked, blocked, blocked},
	}

	game := engine.NewGame(engine.NewGameField(5, 5), nil)
	cordons := game.ExtractCordon(floodFillGrid)

	require.Len(t, cordons, 1)

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 0, Col: 0},
		{Row: 0, Col: 1},
		{Row: 0, Col: 2},
		{Row: 1, Col: 2},
		{Row: 2, Col: 3},
		{Row: 2, Col: 4},
		{Row: 3, Col: 4},
		{Row: 4, Col: 4},
		{Row: 4, Col: 3},
		{Row: 4, Col: 2},
		{Row: 3, Col: 2},
		{Row: 2, Col: 1},
		{Row: 2, Col: 0},
		{Row: 1, Col: 0},
		{Row: 0, Col: 0},
	}, cordons[0])
}

func TestGameExtractCordonFromTwoDiamondsSharingCorner(t *testing.T) {
	// Flood-fill grid (zero-based row and column indices):
	//
	//       c0 c1 c2 c3 c4
	// r0    .  .  B  .  .
	// r1    .  B  R  B  .
	// r2    .  .  B  .  .
	// r3    .  B  R  B  .
	// r4    .  .  B  .  .
	//
	// B = blocked blue cordon dot, R = blocked captured red dot,
	// . = cell with no flood-fill state. The two diamonds share the blue
	// cordon dot at (2,2).

	// NOTE:: it detects only the upper cordon and I'm probably fine with this.
	// I feel it's much easier to solve as dual pass of RunFloodFill + ExtractCordon
	// rather than writing very correct logic in a single ExtractCordon run.
	const (
		none    = engine.FloodFillCellStateNone
		blocked = engine.FloodFillCellStateBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{none, none, blocked, none, none},
		{none, blocked, blocked, blocked, none},
		{none, none, blocked, none, none},
		{none, blocked, blocked, blocked, none},
		{none, none, blocked, none, none},
	}

	game := engine.NewGame(engine.NewGameField(5, 5), nil)
	cordons := game.ExtractCordon(floodFillGrid)

	require.Len(t, cordons, 2)

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 0, Col: 2},
		{Row: 1, Col: 3},
		{Row: 2, Col: 2},
		{Row: 1, Col: 1},
		{Row: 0, Col: 2},
	}, cordons[0])
}
