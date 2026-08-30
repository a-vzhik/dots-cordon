package engine

import (
	"cmp"
	"fmt"
	"log/slog"
	"slices"
)

var EmptyValue any

type CordonPathFinder struct {
	Index   map[Coord]any
	Ordered []Coord
}

func NewCordonPathFinder() *CordonPathFinder {
	return &CordonPathFinder{
		Index: make(map[Coord]any),
	}
}

func (cpf *CordonPathFinder) BuildIndex(floodFillGrid [][]FloodFillCellState) {
	height := uint8(len(floodFillGrid))
	width := uint8(len(floodFillGrid[0]))

	for rowIdx := range height {
		for colIdx := range width {
			state := floodFillGrid[rowIdx][colIdx]
			if state != FloodFillCellStateBlocked {
				continue
			}

			isCordon := false
			if rowIdx == 0 || colIdx == 0 || rowIdx == height-1 || colIdx == width-1 {
				// If a blocked cell is on the edge -> it's always a part of the cordon.
				isCordon = true
			} else {
				// Otherwise I need to check if at least one of the 4-directional neighbours is not blocked.
				// If yes -> it's part of the cordon.
				prevRowIdx := rowIdx - 1
				prevColIdx := colIdx - 1
				nextRowIdx := rowIdx + 1
				nextColIdx := colIdx + 1

				isCordon = floodFillGrid[rowIdx][prevColIdx] != FloodFillCellStateBlocked ||
					floodFillGrid[rowIdx][nextColIdx] != FloodFillCellStateBlocked ||
					floodFillGrid[prevRowIdx][colIdx] != FloodFillCellStateBlocked ||
					floodFillGrid[nextRowIdx][colIdx] != FloodFillCellStateBlocked
			}

			if !isCordon {
				continue
			}

			key := Coord{
				Row: rowIdx,
				Col: colIdx,
			}
			cpf.Index[key] = EmptyValue
		}
	}

	slog.Debug(fmt.Sprintf("Cordon index: %+v", cpf.Index))
}

func (cpf *CordonPathFinder) FindCordons(floodFillGrid [][]FloodFillCellState) [][]Coord {
	// Firstly I need to build an index of all BLOCKED dots which are on the outer edge.
	cpf.BuildIndex(floodFillGrid)

	// In the real game it's possible to close 2 cordons with a single move (when 2 diamond shapes are getting connected).
	// However I believe 2 is the max. I can't imagine how it can be 3 or more cordons.
	//
	// But to solve the problem I prefer a generic loop instead of 2 hardcoded runs.
	// Each iteration extracts 1 cordon. This cordon may contain joints which also belong to another cordon.
	// Non-joints are deleted from the index before the next run.
	// The loop stops if the index is empty or no cordon was extracted during the run.
	capturingCordons := make([][]Coord, 0)
	for {
		startFromKey, found := findTopLeftCoord(cpf.Index)
		if !found {
			break
		}

		slog.Debug(fmt.Sprintf("Index:  %+v", cpf.Index))

		cpf.Ordered = make([]Coord, 0, len(cpf.Index))
		cordonFound := cpf.WalkCordonStep(startFromKey, startFromKey)
		if !cordonFound {
			// This run has not found any cordon - give up.
			break
		}

		extractedCordon := cpf.Ordered
		if len(extractedCordon) == 0 {
			// This run has not found any cordon - give up.
			break
		}

		extractedCordon = trimToClosedCordon(extractedCordon)

		slog.Debug(fmt.Sprintf("Cordon: len=%d, %+v", len(extractedCordon), extractedCordon))

		// Cordon of 4 has effectively 3 points and doesn't capture anything.
		// We still need remove such cordon from the index, but we won't return it.
		if len(extractedCordon) > 4 {
			slog.Debug(fmt.Sprintf("Capturing Cordon: len=%d, %+v", len(extractedCordon), extractedCordon))
			capturingCordons = append(capturingCordons, extractedCordon)
		}

		cpf.UpdateIndex(extractedCordon)

		if len(cpf.Index) == 0 {
			break
		}
	}

	return capturingCordons
}

func (cpf *CordonPathFinder) UpdateIndex(extractedCordon []Coord) {
	// Delete all cordon keys from the index.
	for _, key := range extractedCordon {
		delete(cpf.Index, key)
	}

	// Check if we deleted any cordon keys which have remaining neighbours.
	// If yes, remember them.
	jointsToRestore := make([]Coord, 0)
	for _, key := range extractedCordon {
		neighbours := existingNeighboursOf(key, cpf.Index)
		if len(neighbours) == 0 {
			continue
		}
		jointsToRestore = append(jointsToRestore, key)
	}

	// Restore those joints in the index, because they are needed for other cordons.
	for _, joint := range jointsToRestore {
		cpf.Index[joint] = EmptyValue
	}
}

func (cordon *CordonPathFinder) WalkCordonStep(toKey Coord, fromKey Coord) bool {
	cordon.Ordered = append(cordon.Ordered, toKey)

	existingNeighbours := existingNeighboursOf(toKey, cordon.Index)
	slices.SortStableFunc(existingNeighbours, func(k1, k2 Coord) int {
		dc1 := toKey.Col - k1.Col
		dr1 := toKey.Row - k1.Row
		distance1 := dc1*dc1 + dr1*dr1

		dc2 := toKey.Col - k2.Col
		dr2 := toKey.Row - k2.Row
		distance2 := dc2*dc2 + dr2*dr2

		return cmp.Compare(distance1, distance2)
	})

	for _, n := range existingNeighbours {
		if n == fromKey {
			// Ignore the direction from where we came.
			continue
		}

		if slices.Contains(cordon.Ordered, n) {
			// Ordered cordon has enclosed segment,
			// which could be the entire cordon or part of it
			cordon.Ordered = append(cordon.Ordered, n)
			return true
		}

		if cordon.WalkCordonStep(n, toKey) {
			return true
		}
	}
	return false
}

func findTopLeftCoord(source map[Coord]any) (Coord, bool) {
	if len(source) == 0 {
		return Coord{}, false
	}

	var minKey *Coord

	for k := range source {
		if minKey == nil {
			minKey = &k
		} else if k.Row < minKey.Row {
			minKey = &k
		} else if k.Row == minKey.Row && k.Col < minKey.Col {
			minKey = &k
		}
	}

	return *minKey, true
}

func allNeighboursOf(currentKey Coord) []Coord {
	neighbours := []Coord{
		{Row: currentKey.Row - 1, Col: currentKey.Col - 1},
		{Row: currentKey.Row - 1, Col: currentKey.Col},
		{Row: currentKey.Row - 1, Col: currentKey.Col + 1},
		{Row: currentKey.Row, Col: currentKey.Col + 1},
		{Row: currentKey.Row + 1, Col: currentKey.Col + 1},
		{Row: currentKey.Row + 1, Col: currentKey.Col},
		{Row: currentKey.Row + 1, Col: currentKey.Col - 1},
		{Row: currentKey.Row, Col: currentKey.Col - 1},
	}

	return neighbours
}

func existingNeighboursOf(currentKey Coord, cordonIndex map[Coord]any) []Coord {
	neighbours := allNeighboursOf(currentKey)

	existingNeighbours := slices.DeleteFunc(neighbours, func(n Coord) bool {
		_, found := cordonIndex[n]
		return !found
	})

	return existingNeighbours
}

func trimToClosedCordon(extractedCordon []Coord) []Coord {
	trimmedCordon := extractedCordon

	for trimmedCordon[0] != trimmedCordon[len(trimmedCordon)-1] {
		trimmedCordon = trimmedCordon[1:]
	}

	return trimmedCordon
}
