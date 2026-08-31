package engine_test

import (
	"fmt"
	"log/slog"
	"testing"

	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFindCordons_Diamond(t *testing.T) {
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
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, N, N, N},
		{N, N, P, N, N},
		{N, P, B, P, N},
		{N, N, P, N, N},
		{N, N, N, N, N},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	require.Len(t, cordons, 1)

	assert.Equal(t, []engine.Coord{
		{Row: 1, Col: 2},
		{Row: 2, Col: 3},
		{Row: 3, Col: 2},
		{Row: 2, Col: 1},
		{Row: 1, Col: 2},
	}, cordons[0])
}

func TestFindCordons_FullyBlockedGrid(t *testing.T) {
	// Game board:
	//       c0 c1 c2 c3
	// r0    B  B  B  B
	// r1    B  R  R  B
	// r2    B  R  R  B
	// r3    B  B  B  B
	//
	// B = blue dot, R = red dot, . = empty.
	const (
		B = engine.FloodFillCellStateBlocked
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{P, P, P, P},
		{P, B, B, P},
		{P, B, B, P},
		{P, P, P, P},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	require.Len(t, cordons, 1)

	assert.Equal(t, []engine.Coord{
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

func TestFindCordons_AsymmetricBlob(t *testing.T) {
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
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, N, N, N, N, N},
		{N, N, P, P, P, N, N},
		{N, P, B, B, B, P, N},
		{N, P, B, B, B, P, N},
		{N, P, B, B, B, B, P},
		{N, N, P, P, P, P, N},
		{N, N, N, N, N, N, N},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	require.Len(t, cordons, 1)

	assert.Equal(t, []engine.Coord{
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

func TestFindCordons_ZigzagSShape(t *testing.T) {
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
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, P, P, P, P, N, N, P, N, N},
		{N, P, B, B, B, B, P, P, B, P, N},
		{N, N, P, B, B, B, B, B, P, N, N},
		{N, N, N, P, P, B, B, B, B, P, N},
		{N, N, N, N, N, P, B, B, B, B, P},
		{N, N, N, P, P, B, B, B, B, P, N},
		{N, P, P, B, B, B, B, P, P, N, N},
		{P, B, B, P, B, B, P, N, N, N, N},
		{N, P, B, B, B, P, N, N, N, N, N},
		{P, B, B, B, P, B, P, N, N, N, N},
		{N, P, B, B, B, B, B, P, P, N, N},
		{N, N, P, B, B, B, B, B, B, P, N},
		{N, N, N, P, P, P, P, P, P, N, N},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	require.Len(t, cordons, 2)

	assert.Equal(t, []engine.Coord{
		{Row: 1, Col: 7},
		{Row: 0, Col: 8},
		{Row: 1, Col: 9},
		{Row: 2, Col: 8},
		{Row: 1, Col: 7},
	}, cordons[0])

	assert.Equal(t, []engine.Coord{
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
		{Row: 1, Col: 1},
		{Row: 0, Col: 2},
	}, cordons[1])
}

func TestFindCordons_TwoRectanglesSharingCorner(t *testing.T) {
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
		N = engine.FloodFillCellStateNone
		B = engine.FloodFillCellStateBlocked
		E = engine.FloodFillCellStateEscaped
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, P, N, E, N},
		{P, B, P, E, N},
		{N, P, N, P, N},
		{N, E, P, B, P},
		{N, E, N, P, N},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	slog.Info(fmt.Sprintf("%+v", cordons))

	require.Len(t, cordons, 1)

	assert.Equal(t, [][]engine.Coord{
		{
			{Row: 0, Col: 1},
			{Row: 1, Col: 2},
			{Row: 2, Col: 3},
			{Row: 3, Col: 4},
			{Row: 4, Col: 3},
			{Row: 3, Col: 2},
			{Row: 2, Col: 1},
			{Row: 1, Col: 0},
			{Row: 0, Col: 1},
		},
	}, cordons)
}

func Test_WrongCordonRegression(t *testing.T) {
	/*
		Flood-fill grid regenerated from the final board in
		game-2026-08-30T14-36-49.json:
		E E . E . . .
		. . . . . . .
		E . . . E . .
		. E E E E E .
		. E E E E E E
		E E E E E E E
		E E P E E E .
	*/
	const (
		N = engine.FloodFillCellStateNone
		E = engine.FloodFillCellStateEscaped
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{E, E, N, E, N, N, N},
		{N, N, N, N, N, N, N},
		{E, N, N, N, E, N, N},
		{N, E, E, E, E, E, N},
		{N, E, E, E, E, E, E},
		{E, E, E, E, E, E, E},
		{E, E, P, E, E, E, N},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	slog.Info(fmt.Sprintf("Cordons: %+v", cordons))

	require.Len(t, cordons, 0)
}

func Test_123(t *testing.T) {
	/*
		2026/08/30 13:39:20 INFO Floodfill Grid:
		2026/08/30 13:39:20 INFO E . E P E P .
		2026/08/30 13:39:20 INFO E . E E E E E
		2026/08/30 13:39:20 INFO . P E E E E E
		2026/08/30 13:39:20 INFO . P P E E E .
		2026/08/30 13:39:20 INFO P B P E E E P
		2026/08/30 13:39:20 INFO . P . E E P .
		2026/08/30 13:39:20 INFO E E E P E . .
	*/
	const (
		N = engine.FloodFillCellStateNone
		B = engine.FloodFillCellStateBlocked
		E = engine.FloodFillCellStateEscaped
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{E, N, E, P, E, P, N},
		{E, N, E, E, E, E, E},
		{N, P, E, E, E, E, E},
		{N, P, P, E, E, E, N},
		{P, B, P, E, E, E, P},
		{N, P, N, E, E, P, N},
		{E, E, E, P, E, N, N},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	require.Len(t, cordons, 1)
}

func TestFindCordons_TwoDiamondsSharingCorner(t *testing.T) {
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
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, P, N, N},
		{N, P, B, P, N},
		{N, N, P, N, N},
		{N, P, B, P, N},
		{N, N, P, N, N},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	require.Len(t, cordons, 2)

	assert.Equal(t, []engine.Coord{
		{Row: 0, Col: 2},
		{Row: 1, Col: 3},
		{Row: 2, Col: 2},
		{Row: 1, Col: 1},
		{Row: 0, Col: 2},
	}, cordons[0])

	assert.Equal(t, []engine.Coord{
		{Row: 2, Col: 2},
		{Row: 3, Col: 3},
		{Row: 4, Col: 2},
		{Row: 3, Col: 1},
		{Row: 2, Col: 2},
	}, cordons[1])
}

func TestFindCordons_FourDiamondsSharingCenter(t *testing.T) {
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
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, P, N, N},
		{N, P, B, P, N},
		{P, B, P, B, P},
		{E, P, B, P, N},
		{N, N, P, N, N},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	require.Len(t, cordons, 2)

	assert.Equal(t, [][]engine.Coord{
		{
			{Row: 0, Col: 2},
			{Row: 1, Col: 3},
			{Row: 2, Col: 4},
			{Row: 3, Col: 3},
			{Row: 2, Col: 2},
			{Row: 1, Col: 1},
			{Row: 0, Col: 2},
		},
		{
			{Row: 1, Col: 1},
			{Row: 2, Col: 2},
			{Row: 3, Col: 3},
			{Row: 4, Col: 2},
			{Row: 3, Col: 1},
			{Row: 2, Col: 0},
			{Row: 1, Col: 1},
		},
	}, cordons)
}

func TestFindCordons_TwoRaggedPocketsSharingCorner(t *testing.T) {
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
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, N, N, N, P, P, P, N},
		{N, N, N, P, B, B, B, P},
		{N, N, N, P, B, P, P, N},
		{N, N, N, N, P, N, N, N},
		{N, P, N, P, B, P, N, N},
		{P, B, P, B, P, N, N, N},
		{P, B, B, B, P, N, N, N},
		{N, P, P, P, N, N, N, N},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	require.Len(t, cordons, 2)

	assert.Equal(t, []engine.Coord{
		{Row: 0, Col: 4},
		{Row: 0, Col: 5},
		{Row: 0, Col: 6},
		{Row: 1, Col: 7},
		{Row: 2, Col: 6},
		{Row: 2, Col: 5},
		{Row: 3, Col: 4},
		{Row: 2, Col: 3},
		{Row: 1, Col: 3},
		{Row: 0, Col: 4},
	}, cordons[0])

	assert.Equal(t, []engine.Coord{
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

func TestFindCordons_BalancedRandomBoard(t *testing.T) {
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
		P = engine.FloodFillCellStatePotentialBlocked
	)
	floodFillGrid := [][]engine.FloodFillCellState{
		{N, E, P, P, E, N, E, E},
		{E, P, B, B, P, E, N, E},
		{N, P, P, P, E, E, N, E},
		{E, E, E, E, E, E, P, E},
		{P, E, P, P, P, E, P, N},
		{E, P, B, P, B, P, B, P},
		{N, P, B, P, B, P, P, E},
		{E, N, P, E, P, E, E, N},
	}

	pathFinder := engine.NewCordonPathFinder()
	cordons := pathFinder.FindCordons(floodFillGrid)

	slog.Info(fmt.Sprintf("CORDONS: %+v", cordons))

	require.Len(t, cordons, 4)

	assert.Equal(t, [][]engine.Coord{
		{
			{Row: 0, Col: 2},
			{Row: 0, Col: 3},
			{Row: 1, Col: 4},
			{Row: 2, Col: 3},
			{Row: 2, Col: 2},
			{Row: 2, Col: 1},
			{Row: 1, Col: 1},
			{Row: 0, Col: 2},
		},
		{
			{Row: 5, Col: 5},
			{Row: 6, Col: 5},
			{Row: 6, Col: 6},
			{Row: 5, Col: 7},
			{Row: 4, Col: 6},
			{Row: 5, Col: 5},
		},
		{
			{Row: 4, Col: 3},
			{Row: 4, Col: 4},
			{Row: 5, Col: 5},
			{Row: 6, Col: 5},
			{Row: 7, Col: 4},
			{Row: 6, Col: 3},
			{Row: 5, Col: 3},
			{Row: 4, Col: 3},
		},
		{
			{Row: 4, Col: 2},
			{Row: 4, Col: 3},
			{Row: 5, Col: 3},
			{Row: 6, Col: 3},
			{Row: 7, Col: 2},
			{Row: 6, Col: 1},
			{Row: 5, Col: 1},
			{Row: 4, Col: 2},
		},
	}, cordons)
}
