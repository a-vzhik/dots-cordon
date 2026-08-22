package engine_test

import (
	"testing"

	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGameExtractCordonFromDiamond(t *testing.T) {
	// Game board:
	//       c0 c1 c2 c3 c4
	// r0    .  .  .  .  .
	// r1    .  .  B  .  .
	// r2    .  B  R  B  .
	// r3    .  .  B  .  .
	// r4    .  .  .  .  .
	//
	// B = blue dot, R = red dot, . = empty.
	const (
		N = engine.FloodFillCellStateNone
		B = engine.FloodFillCellStateBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, N, N, N},
		{N, N, B, N, N},
		{N, B, B, B, N},
		{N, N, B, N, N},
		{N, N, N, N, N},
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
	// Game board:
	//       c0 c1 c2 c3
	// r0    B  B  B  B
	// r1    B  R  R  B
	// r2    B  R  R  B
	// r3    B  B  B  B
	//
	// B = blue dot, R = red dot, . = empty.
	const B = engine.FloodFillCellStateBlocked
	floodFillGrid := [][]engine.FloodFillCellState{
		{B, B, B, B},
		{B, B, B, B},
		{B, B, B, B},
		{B, B, B, B},
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
	// Game board:
	//       c0 c1 c2 c3 c4 c5 c6
	// r0    .  .  .  .  .  .  .
	// r1    .  .  B  B  B  .  .
	// r2    .  B  R  R  R  B  .
	// r3    .  B  R  R  R  B  .
	// r4    .  B  R  R  R  R  B
	// r5    .  .  B  B  B  B  .
	// r6    .  .  .  .  .  .  .
	//
	// B = blue dot, R = red dot, . = empty.
	const (
		N = engine.FloodFillCellStateNone
		B = engine.FloodFillCellStateBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, N, N, N, N, N},
		{N, N, B, B, B, N, N},
		{N, B, B, B, B, B, N},
		{N, B, B, B, B, B, N},
		{N, B, B, B, B, B, B},
		{N, N, B, B, B, B, N},
		{N, N, N, N, N, N, N},
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

func TestGameExtractCordonFromZigzagSShape(t *testing.T) {
	// Game board:
	//       c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10
	// r0    .  .  B  B  B  B  .  .  B  .  .
	// r1    .  B  R  R  R  R  B  B  R  B  .
	// r2    B  B  B  R  R  R  R  R  B  .  .
	// r3    .  .  .  B  B  R  R  R  R  B  B
	// r4    .  .  .  .  .  B  R  R  R  R  B
	// r5    .  .  .  B  B  R  R  R  R  B  .
	// r6    .  B  B  R  R  R  R  B  B  .  .
	// r7    B  R  R  B  R  R  B  .  .  .  .
	// r8    .  B  R  R  R  B  .  .  .  .  .
	// r9    B  R  R  R  B  R  B  .  .  .  .
	// r10   .  B  R  R  R  R  R  B  B  .  .
	// r11   .  .  B  R  R  R  R  R  R  B  .
	// r12   .  .  .  B  B  B  B  B  B  B  B
	//
	// B = blue dot, R = red dot, . = empty.
	const (
		N = engine.FloodFillCellStateNone
		B = engine.FloodFillCellStateBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, B, B, B, B, N, N, B, N, N},
		{N, B, B, B, B, B, B, B, B, B, N},
		{B, B, B, B, B, B, B, B, B, N, N},
		{N, N, N, B, B, B, B, B, B, B, B},
		{N, N, N, N, N, B, B, B, B, B, B},
		{N, N, N, B, B, B, B, B, B, B, N},
		{N, B, B, B, B, B, B, B, B, N, N},
		{B, B, B, B, B, B, B, N, N, N, N},
		{N, B, B, B, B, B, N, N, N, N, N},
		{B, B, B, B, B, B, B, N, N, N, N},
		{N, B, B, B, B, B, B, B, B, N, N},
		{N, N, B, B, B, B, B, B, B, B, N},
		{N, N, N, B, B, B, B, B, B, B, B},
	}

	game := engine.NewGame(engine.NewGameField(11, 13), nil)
	cordons := game.ExtractCordon(floodFillGrid)

	require.Len(t, cordons, 2)

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 1, Col: 7},
		{Row: 0, Col: 8},
		{Row: 1, Col: 9},
		{Row: 2, Col: 8},
		{Row: 1, Col: 7},
	}, cordons[0])

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 0, Col: 2},
		{Row: 0, Col: 3},
		{Row: 0, Col: 4},
		{Row: 0, Col: 5},
		{Row: 1, Col: 6},
		{Row: 1, Col: 7},
		{Row: 2, Col: 8},
		{Row: 3, Col: 9},
		{Row: 4, Col: 10},
		{Row: 5, Col: 9},
		{Row: 6, Col: 8},
		{Row: 6, Col: 7},
		{Row: 7, Col: 6},
		{Row: 8, Col: 5},
		{Row: 9, Col: 6},
		{Row: 10, Col: 7},
		{Row: 10, Col: 8},
		{Row: 11, Col: 9},
		{Row: 12, Col: 9},
		{Row: 12, Col: 8},
		{Row: 12, Col: 7},
		{Row: 12, Col: 6},
		{Row: 12, Col: 5},
		{Row: 12, Col: 4},
		{Row: 12, Col: 3},
		{Row: 11, Col: 2},
		{Row: 10, Col: 1},
		{Row: 9, Col: 0},
		{Row: 8, Col: 1},
		{Row: 7, Col: 0},
		{Row: 6, Col: 1},
		{Row: 6, Col: 2},
		{Row: 5, Col: 3},
		{Row: 5, Col: 4},
		{Row: 4, Col: 5},
		{Row: 3, Col: 4},
		{Row: 3, Col: 3},
		{Row: 2, Col: 2},
		{Row: 2, Col: 1},
		{Row: 1, Col: 1},
		{Row: 0, Col: 2},
	}, cordons[1])
}

func TestGameExtractCordonFromTwoRectanglesSharingCorner(t *testing.T) {
	// Game board:
	//       c0 c1 c2 c3 c4
	// r0    B  B  B  .  .
	// r1    B  R  B  R  .
	// r2    B  B  B  B  B
	// r3    .  R  B  R  B
	// r4    .  .  B  B  B
	//
	// B = blue dot, R = red dot, . = empty.
	const (
		B = engine.FloodFillCellStateBlocked
		E = engine.FloodFillCellStateEscaped
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{B, B, B, E, E},
		{B, B, B, E, E},
		{B, B, B, B, B},
		{E, E, B, B, B},
		{E, E, B, B, B},
	}

	game := engine.NewGame(engine.NewGameField(5, 5), nil)
	cordons := game.ExtractCordon(floodFillGrid)

	require.Len(t, cordons, 1)

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 0, Col: 0},
		{Row: 0, Col: 1},
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
	// Game board:
	//       c0 c1 c2 c3 c4
	// r0    .  .  B  .  .
	// r1    .  B  R  B  .
	// r2    .  .  B  .  .
	// r3    .  B  R  B  .
	// r4    .  .  B  .  .
	//
	// B = blue dot, R = red dot, . = empty.
	const (
		N = engine.FloodFillCellStateNone
		B = engine.FloodFillCellStateBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, B, N, N},
		{N, B, B, B, N},
		{N, N, B, N, N},
		{N, B, B, B, N},
		{N, N, B, N, N},
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

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 2, Col: 2},
		{Row: 3, Col: 3},
		{Row: 4, Col: 2},
		{Row: 3, Col: 1},
		{Row: 2, Col: 2},
	}, cordons[1])
}

func TestGameExtractCordonFromFourDiamondsSharingCenter(t *testing.T) {
	// Game board:
	//       c0 c1 c2 c3 c4
	// r0    .  .  B  .  .
	// r1    .  B  R  B  .
	// r2    B  R  B  R  B
	// r3    R  B  R  B  .
	// r4    .  .  B  .  .
	//
	// B = blue dot, R = red dot, . = empty.
	const (
		N = engine.FloodFillCellStateNone
		B = engine.FloodFillCellStateBlocked
		E = engine.FloodFillCellStateEscaped
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, B, N, N},
		{N, B, B, B, N},
		{B, B, B, B, B},
		{E, B, B, B, N},
		{E, E, B, N, N},
	}

	game := engine.NewGame(engine.NewGameField(5, 5), nil)
	cordons := game.ExtractCordon(floodFillGrid)

	require.Len(t, cordons, 1)

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 0, Col: 2},
		{Row: 1, Col: 3},
		{Row: 2, Col: 4},
		{Row: 3, Col: 3},
		{Row: 4, Col: 2},
		{Row: 3, Col: 1},
		{Row: 2, Col: 0},
		{Row: 1, Col: 1},
		{Row: 0, Col: 2},
	}, cordons[0])
}

func TestGameExtractCordonFromTwoRaggedPocketsSharingCorner(t *testing.T) {
	// Game board:
	//       c0 c1 c2 c3 c4 c5 c6 c7
	// r0    .  .  .  .  B  B  B  .
	// r1    .  .  .  B  R  R  R  B
	// r2    .  .  .  B  R  B  B  B
	// r3    .  .  .  .  B  .  .  .
	// r4    .  B  .  B  R  B  .  .
	// r5    B  R  B  R  B  .  .  .
	// r6    B  R  R  R  B  B  .  .
	// r7    .  B  B  B  .  .  .  .
	//
	// B = blue dot, R = red dot, . = empty.
	const (
		N = engine.FloodFillCellStateNone
		B = engine.FloodFillCellStateBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, N, N, B, B, B, N},
		{N, N, N, B, B, B, B, B},
		{N, N, N, B, B, B, B, B},
		{N, N, N, N, B, N, N, N},
		{N, B, N, B, B, B, N, N},
		{B, B, B, B, B, N, N, N},
		{B, B, B, B, B, B, N, N},
		{N, B, B, B, N, N, N, N},
	}

	game := engine.NewGame(engine.NewGameField(8, 8), nil)
	cordons := game.ExtractCordon(floodFillGrid)

	require.Len(t, cordons, 2)

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 0, Col: 4},
		{Row: 0, Col: 5},
		{Row: 0, Col: 6},
		{Row: 1, Col: 7},
		{Row: 2, Col: 7},
		{Row: 2, Col: 6},
		{Row: 2, Col: 5},
		{Row: 3, Col: 4},
		{Row: 2, Col: 3},
		{Row: 1, Col: 3},
		{Row: 0, Col: 4},
	}, cordons[0])

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 3, Col: 4},
		{Row: 4, Col: 5},
		{Row: 5, Col: 4},
		{Row: 6, Col: 4},
		{Row: 7, Col: 3},
		{Row: 7, Col: 2},
		{Row: 7, Col: 1},
		{Row: 6, Col: 0},
		{Row: 5, Col: 0},
		{Row: 4, Col: 1},
		{Row: 5, Col: 2},
		{Row: 4, Col: 3},
		{Row: 3, Col: 4},
	}, cordons[1])
}

func TestGameExtractCordonFromBalancedRandomBoard(t *testing.T) {
	// Game board:
	//       c0 c1 c2 c3 c4 c5 c6 c7
	// r0    B  R  B  B  R  B  R  R
	// r1    R  B  R  R  B  R  B  R
	// r2    B  B  B  B  R  R  B  R
	// r3    R  R  R  R  R  R  B  R
	// r4    B  R  B  B  B  R  B  B
	// r5    R  B  R  B  R  B  R  B
	// r6    B  B  R  B  R  B  B  R
	// r7    R  B  B  R  B  R  R  B
	//
	// B = blue dot, R = red dot, . = empty.
	const (
		N = engine.FloodFillCellStateNone
		B = engine.FloodFillCellStateBlocked
		E = engine.FloodFillCellStateEscaped
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, E, B, B, E, N, E, E},
		{E, B, B, B, B, E, N, E},
		{N, B, B, B, E, E, N, E},
		{E, E, E, E, E, E, B, E},
		{B, E, B, B, B, E, B, N},
		{E, B, B, B, B, B, B, B},
		{N, B, B, B, B, B, B, E},
		{E, N, B, E, B, E, E, N},
	}

	game := engine.NewGame(engine.NewGameField(8, 8), nil)
	cordons := game.ExtractCordon(floodFillGrid)

	require.Len(t, cordons, 3)

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 0, Col: 2},
		{Row: 0, Col: 3},
		{Row: 1, Col: 4},
		{Row: 2, Col: 3},
		{Row: 2, Col: 2},
		{Row: 2, Col: 1},
		{Row: 1, Col: 1},
		{Row: 0, Col: 2},
	}, cordons[0])

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 6, Col: 5},
		{Row: 5, Col: 5},
		{Row: 4, Col: 4},
		{Row: 4, Col: 3},
		{Row: 4, Col: 2},
		{Row: 5, Col: 1},
		{Row: 6, Col: 1},
		{Row: 7, Col: 2},
		{Row: 6, Col: 3},
		{Row: 7, Col: 4},
		{Row: 6, Col: 5},
	}, cordons[1])

	assert.Equal(t, []engine.CordonIndexKey{
		{Row: 4, Col: 6},
		{Row: 5, Col: 7},
		{Row: 6, Col: 6},
		{Row: 6, Col: 5},
		{Row: 5, Col: 5},
		{Row: 4, Col: 6},
	}, cordons[2])
}
