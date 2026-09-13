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

	sourceFloodFillGrid [][]FloodFillCellState
}

func NewCordonPathFinder() *CordonPathFinder {
	return &CordonPathFinder{
		Index: make(map[Coord]any),
	}
}

func eightNeighbours(coord Coord, width uint8, height uint8) []Coord {
	neighbours := make([]Coord, 0, 8)

	// Row above
	if coord.Row > 0 {
		if coord.Col > 0 {
			neighbours = append(neighbours, Coord{Row: coord.Row - 1, Col: coord.Col - 1})
		}

		neighbours = append(neighbours, Coord{Row: coord.Row - 1, Col: coord.Col})

		if coord.Col < width-1 {
			neighbours = append(neighbours, Coord{Row: coord.Row - 1, Col: coord.Col + 1})
		}
	}

	// Right neighbour in the same row
	if coord.Col < width-1 {
		neighbours = append(neighbours, Coord{Row: coord.Row, Col: coord.Col + 1})
	}

	// Row below
	if coord.Row < height-1 {

		if coord.Col < width-1 {
			neighbours = append(neighbours, Coord{Row: coord.Row + 1, Col: coord.Col + 1})
		}

		neighbours = append(neighbours, Coord{Row: coord.Row + 1, Col: coord.Col})

		if coord.Col > 0 {
			neighbours = append(neighbours, Coord{Row: coord.Row + 1, Col: coord.Col - 1})
		}

	}

	// Left neighbour in the same row
	if coord.Col > 0 {
		neighbours = append(neighbours, Coord{Row: coord.Row, Col: coord.Col - 1})
	}

	return neighbours
}

func (cpf *CordonPathFinder) BuildIndex(floodFillGrid [][]FloodFillCellState) {
	height := uint8(len(floodFillGrid))
	width := uint8(len(floodFillGrid[0]))

	for rowIdx := range height {
		for colIdx := range width {
			state := floodFillGrid[rowIdx][colIdx]
			if state != FloodFillCellStatePotentialBlocked {
				continue
			}

			neighbours := eightNeighbours(Coord{Row: rowIdx, Col: colIdx}, width, height)
			isCordon := false
			for _, n := range neighbours {
				if floodFillGrid[n.Row][n.Col] == FloodFillCellStateBlocked {
					isCordon = true
					break
				}
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
	cpf.sourceFloodFillGrid = floodFillGrid
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

		if len(cpf.Index) < 4 {
			break
		}

		slog.Debug(fmt.Sprintf("Index:  %+v", cpf.Index))

		cpf.Ordered = make([]Coord, 0, len(cpf.Index))
		cordonFound, deadEnd := cpf.WalkCordonStep(startFromKey, startFromKey)
		if !cordonFound && !deadEnd {
			// This run has not found any cordon - give up.
			slog.Info("Break the loop")
			break
		}

		if deadEnd {
			delete(cpf.Index, startFromKey)
			continue
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
	/*

		// Game board:
		//       c0 c1 c2 c3 c4
		// r0    B  B  B  .  .
		// r1    B  R  B  R  .
		// r2    B  B  B  B  B
		// r3    .  R  B  R  B
		// r4    .  .  B  B  B
		//

		   * 2026/08/30 17:33:02 INFO Cordon: len=4, [{Row:1 Col:2} {Row:2 Col:2} {Row:2 Col:3} {Row:1 Col:2}]
		   2026/08/30 17:33:02 INFO Index:  map[{Row:0 Col:0}:<nil> {Row:0 Col:1}:<nil>
		   {Row:0 Col:2}:<nil> {Row:1 Col:0}:<nil> {Row:1 Col:2}:<nil>
		   {Row:2 Col:0}:<nil> {Row:2 Col:1}:<nil> {Row:2 Col:2}:<nil>
		   {Row:2 Col:3}:<nil> {Row:3 Col:2}:<nil>]

	*/

	/**
	  Pruned index:
	  map[{Row:0 Col:0}:<nil> {Row:0 Col:1}:<nil>
	  {Row:0 Col:2}:<nil> {Row:1 Col:0}:<nil>
	  {Row:2 Col:0}:<nil> {Row:2 Col:1}:<nil>  {Row:3 Col:2}:<nil>]

		(1,2) => 01, 02, 21 => mark kept.
		(2,2) => 21 32 => mark kept
		(2,3) =>

	*/
	// Delete all cordon keys from the index.
	for _, key := range extractedCordon {
		delete(cpf.Index, key)
	}

	width := len(cpf.sourceFloodFillGrid[0])
	height := len(cpf.sourceFloodFillGrid)

	// Check if we deleted any cordon keys which have remaining neighbours.
	// If yes, remember them.
	jointsToRestore := make([]Coord, 0)
	for _, key := range extractedCordon {
		neighbours := existingNeighboursOf(key, cpf.Index, uint8(width), uint8(height))
		neighboursOutsideCordon := slices.DeleteFunc(neighbours, func(n Coord) bool {
			return slices.Contains(extractedCordon, n)
		})
		if len(neighboursOutsideCordon) == 0 {
			continue
		}

		jointsToRestore = append(jointsToRestore, key)
	}

	// Restore those joints in the index, because they are needed for other cordons.
	for _, joint := range jointsToRestore {
		cpf.Index[joint] = EmptyValue
	}
}

func (cpf *CordonPathFinder) WalkCordonStep(toKey Coord, fromKey Coord) (bool, bool) {
	loopStart := slices.Index(cpf.Ordered, toKey)
	cpf.Ordered = append(cpf.Ordered, toKey)

	if loopStart >= 0 {
		// Ordered cordon has enclosed segment,
		// which could be the entire cordon or part of it

		if len(cpf.Ordered)-loopStart <= 4 {
			slog.Info(fmt.Sprintf("Small loop dead end found: %+v", cpf.Ordered))
			return false, true
		}

		return true, false
	}

	width := uint8(len(cpf.sourceFloodFillGrid[0]))
	height := uint8(len(cpf.sourceFloodFillGrid))

	existingNeighbours := existingNeighboursOf(toKey, cpf.Index, width, height)
	slices.SortStableFunc(existingNeighbours, func(k1, k2 Coord) int {
		dc1 := toKey.Col - k1.Col
		dr1 := toKey.Row - k1.Row
		distance1 := dc1*dc1 + dr1*dr1

		dc2 := toKey.Col - k2.Col
		dr2 := toKey.Row - k2.Row
		distance2 := dc2*dc2 + dr2*dr2

		return cmp.Compare(distance1, distance2)
	})

	slog.Debug(fmt.Sprintf("Walk %+v", toKey))

	if len(existingNeighbours) == 1 {
		slog.Info(fmt.Sprintf("%+v 1 neighbour left: %+v", toKey, existingNeighbours))
		return false, true
	}

	for _, n := range existingNeighbours {
		if n == fromKey {
			// Ignore the direction from where we came.
			continue
		}

		found, deadEnd := cpf.WalkCordonStep(n, toKey)
		if found {
			return found, deadEnd
		}
		if deadEnd {
			slog.Info(fmt.Sprintf("Dead end loop %+v when checking %+v", cpf.Ordered, n))
			for cpf.Ordered[len(cpf.Ordered)-1] != n {
				//delete(cpf.Index, cpf.Ordered[len(cpf.Ordered)-1])
				cpf.Ordered = cpf.Ordered[:len(cpf.Ordered)-1]
			}

			//delete(cpf.Index, cpf.Ordered[len(cpf.Ordered)-1])
			cpf.Ordered = cpf.Ordered[:len(cpf.Ordered)-1]

			slog.Info(fmt.Sprintf("Dead end unwound %+v", cpf.Ordered))
		}
	}
	return false, true
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

func existingNeighboursOf(currentKey Coord, cordonIndex map[Coord]any, width uint8, height uint8) []Coord {
	neighbours := eightNeighbours(currentKey, width, height)

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
