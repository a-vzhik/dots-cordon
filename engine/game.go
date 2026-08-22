// Package engine contains game primitives and compute game state after each move.
package engine

import (
	"cmp"
	"fmt"
	"log/slog"
	"slices"
	"strings"
)

type GameField struct {
	Width  uint8
	Height uint8
	Dots   [][]Dot
}

type CordonIndexKey struct {
	Row uint8
	Col uint8
}

type Cordon struct {
	Index   map[CordonIndexKey]any
	Ordered []CordonIndexKey
}

func NewCordon() *Cordon {
	return &Cordon{
		Index: make(map[CordonIndexKey]any),
	}
}

func NewGameField(width uint8, height uint8) *GameField {
	dots := make([][]Dot, int(height))
	for rowIdx := range dots {
		row := make([]Dot, int(width))
		for colIdx := range row {
			row[colIdx] = Dot{
				Col: uint8(colIdx),
				Row: uint8(rowIdx),
			}
		}

		dots[rowIdx] = row
	}

	return &GameField{
		Width:  width,
		Height: height,
		Dots:   dots,
	}
}

func (gf *GameField) Transform(filterFunc func(Dot) bool, applyFunc func(Dot) Dot) {
	for rowIdx, row := range gf.Dots {
		for colIdx, oldDot := range row {

			if !filterFunc(oldDot) {
				continue
			}

			newDot := applyFunc(oldDot)
			gf.Dots[rowIdx][colIdx] = newDot
		}
	}
}

type FloodFillCellState uint8

const (
	FloodFillCellStateNone FloodFillCellState = iota
	FloodFillCellStateBlocked
	FloodFillCellStateEscaped
	FloodFillCellStateEscapeCandidate
)

func (gf *GameField) ToFloodFillGrid() [][]FloodFillCellState {
	floodFillGrid := make([][]FloodFillCellState, gf.Height)
	for rowIdx := 0; rowIdx < int(gf.Height); rowIdx++ {
		row := make([]FloodFillCellState, gf.Width)
		for colIdx := 0; colIdx < int(gf.Width); colIdx++ {
			state := FloodFillCellStateNone
			row[colIdx] = state
		}
		floodFillGrid[rowIdx] = row

	}
	return floodFillGrid
}

func (gf *GameField) Find(filterFunc func(Dot) bool) []Dot {
	filtered := make([]Dot, 0)
	for _, row := range gf.Dots {
		for _, dot := range row {

			if !filterFunc(dot) {
				continue
			}

			filtered = append(filtered, dot)

		}
	}
	return filtered
}

type Game struct {
	GameField   *GameField
	Players     []*Player
	PlayerCount PlayerIndex
}

func NewGame(gameField *GameField, players []*Player) *Game {
	game := &Game{
		GameField: gameField,
		Players:   players,
	}

	game.PlayerCount = PlayerIndex(len(players))
	return game
}

func (g *Game) HasDotEscaped(floodFillGrid [][]FloodFillCellState, defenderIdx PlayerIndex, currRowIdx uint8, currColIdx uint8) bool {
	knownState := floodFillGrid[currRowIdx][currColIdx]
	switch knownState {
	case FloodFillCellStateBlocked:
		return false
	case FloodFillCellStateEscaped:
		return true
	case FloodFillCellStateEscapeCandidate:
		return false
	}

	dot := g.GameField.Dots[currRowIdx][currColIdx]
	if dot.Killed {
		floodFillGrid[currRowIdx][currColIdx] = FloodFillCellStateBlocked
		return false
	}
	if !dot.Owned {
		floodFillGrid[currRowIdx][currColIdx] = FloodFillCellStateEscapeCandidate
	}
	if dot.Owned && dot.Owner == defenderIdx {
		floodFillGrid[currRowIdx][currColIdx] = FloodFillCellStateEscapeCandidate
	}
	if dot.Owned && dot.Owner != defenderIdx {
		floodFillGrid[currRowIdx][currColIdx] = FloodFillCellStateBlocked
		slog.Debug(fmt.Sprintf("found the other player: %d, %d", currRowIdx, currColIdx))
		return false
	}

	if currColIdx == 0 || currRowIdx == 0 || currRowIdx == g.GameField.Height-1 || currColIdx == g.GameField.Width-1 {
		slog.Debug(fmt.Sprintf("reached the border: %d, %d", currRowIdx, currColIdx))
		return true
	}

	prevRowIdx := currRowIdx - 1
	prevColIdx := currColIdx - 1
	nextRowIdx := currRowIdx + 1
	nextColIdx := currColIdx + 1

	if currRowIdx < g.GameField.Height-1 {
		if g.HasDotEscaped(floodFillGrid, defenderIdx, nextRowIdx, currColIdx) {
			return true
		}
	}

	if currRowIdx > 0 {
		if g.HasDotEscaped(floodFillGrid, defenderIdx, prevRowIdx, currColIdx) {
			return true
		}
	}

	if currColIdx < g.GameField.Width-1 {
		if g.HasDotEscaped(floodFillGrid, defenderIdx, currRowIdx, nextColIdx) {
			return true
		}
	}

	if currColIdx > 0 {
		if g.HasDotEscaped(floodFillGrid, defenderIdx, currRowIdx, prevColIdx) {
			return true
		}
	}

	return false
}

func setEscapeCandidatesToState(floodFillGrid [][]FloodFillCellState, state FloodFillCellState) {
	slog.Debug(fmt.Sprintf("Marking all escape candidate with %d: %+v\n", state, floodFillGrid))
	for i := range floodFillGrid {
		for j := range floodFillGrid {
			if floodFillGrid[i][j] != FloodFillCellStateEscapeCandidate {
				continue
			}
			floodFillGrid[i][j] = state
		}
	}
}

func PrintFloodFillGrid(floodFillGrid [][]FloodFillCellState) {
	slog.Info("Floodfill Grid: ")

	for _, row := range floodFillGrid {
		var builder strings.Builder
		for _, s := range row {
			rune := '.'
			switch s {
			case FloodFillCellStateBlocked:
				rune = 'B'
			case FloodFillCellStateEscapeCandidate:
				rune = 'C'
			case FloodFillCellStateEscaped:
				rune = 'E'
			}
			builder.WriteRune(rune)
			builder.WriteByte(' ')
		}
		slog.Info(builder.String())
	}
}

func (g *Game) RunFloodFill(floodFillGrid [][]FloodFillCellState, defenderDots []Dot, defenderIdx PlayerIndex) {
	//
	// Create  a flood fill grid of the same size as GameField. This grid contains states: UNKNOWN, BLOCKED, ESCAPED, ESCAPE_CANDIDATE(transient)
	// Iterate over defender's dots.
	// For each dot:
	// 		- if the dot lands on a cell marked as ESCAPED
	// 			- immediately means the dot escapes
	// 		- if it lands on BLOCKED
	// 		  - immediately means the dot is BLOCKED
	// 		- if the cell is UNKNOWN -> start the floodFill in all directions.
	// 		  - mark each reachable defenender's dot or empty dot as ESCAPE_CANDIDATE
	// 		  - stop if blocked by an offender dot
	//      - stop if reached ANY border (return true), brnaching should propagate true correctly to the starting floodfill call.
	//    - If the floodfill for the dot finished with true
	//      - Mark all ESCAPE_CANDIDATES as ESCAPED
	//    -if the floodfill for the dot finishes with false
	//      - Mark all ESCAPE_CANDIDATE as BLOCKED
	//
	// Match BLOCKED cells to defenderDots => they are pockets
	// Match BLOCKED cells to offenderDots => they are cordons

	for _, dot := range defenderDots {
		slog.Debug(fmt.Sprintf("FloodFill dot [%d, %d]...", dot.Row, dot.Col))
		if g.HasDotEscaped(floodFillGrid, defenderIdx, dot.Row, dot.Col) {
			setEscapeCandidatesToState(floodFillGrid, FloodFillCellStateEscaped)
		} else {
			setEscapeCandidatesToState(floodFillGrid, FloodFillCellStateBlocked)
		}
	}
}

func findTopLeftCordonIndexKey(source map[CordonIndexKey]any) (CordonIndexKey, bool) {
	if len(source) == 0 {
		return CordonIndexKey{}, false
	}

	var minKey *CordonIndexKey

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

func ExtractLoop(extractedCordon []CordonIndexKey) []CordonIndexKey {
	trimmedCordon := extractedCordon

	for trimmedCordon[0] != trimmedCordon[len(trimmedCordon)-1] {
		trimmedCordon = trimmedCordon[1:]
	}

	return trimmedCordon
}

func (g *Game) ExtractCordon(floodFillGrid [][]FloodFillCellState) [][]CordonIndexKey {
	cordonExtractor := NewCordon()

	// Firstly I need to build an index of all BLOCKED dots which are on the outer edge.
	for rowIdx := uint8(0); rowIdx < g.GameField.Height; rowIdx++ {
		for colIdx := uint8(0); colIdx < g.GameField.Width; colIdx++ {
			state := floodFillGrid[rowIdx][colIdx]
			if state != FloodFillCellStateBlocked {
				continue
			}

			isCordon := false
			if rowIdx == 0 || colIdx == 0 || rowIdx == g.GameField.Height-1 || colIdx == g.GameField.Width-1 {
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

			key := CordonIndexKey{
				Row: rowIdx,
				Col: colIdx,
			}
			var value any
			cordonExtractor.Index[key] = value
		}
	}

	slog.Debug(fmt.Sprintf("Cordon index: %+v", cordonExtractor.Index))

	// In the real game it's possible to close 2 cordons with a single move (when 2 diamond shapes are getting connected).
	// However I believe 2 is the max. I can't imagine how it can be 3 or more cordons.
	//
	// But to solve the problem I prefer a generic loop instead of 2 hardcoded runs.
	// Each iteration extracts 1 cordon. This cordon may contain joints which also belong to another cordon.
	// Non-joints are deleted from the index before the next run.
	// The loop stops if the index is empty or no cordon was extracted during the run.
	capturingCordons := make([][]CordonIndexKey, 0)
	for {
		startFromKey, found := findTopLeftCordonIndexKey(cordonExtractor.Index)
		if !found {
			break
		}

		cordonExtractor.Ordered = make([]CordonIndexKey, 0, len(cordonExtractor.Index))
		cordonFound := g.AdvanceCordon(cordonExtractor, startFromKey, startFromKey)
		if !cordonFound {
			// This run has not found any cordon - give up.
			break
		}

		extractedCordon := cordonExtractor.Ordered
		if len(extractedCordon) == 0 {
			// This run has not found any cordon - give up.
			break
		}

		extractedCordon = ExtractLoop(extractedCordon)

		slog.Debug(fmt.Sprintf("Cordon: len=%d, %+v", len(extractedCordon), extractedCordon))

		// Cordon of 4 has effectively 3 points and doesn't capture anything.
		// We still need remove such cordon from the index, but we won't return
		// it as a cordon
		if len(extractedCordon) > 4 {
			slog.Info(fmt.Sprintf("Capturing Cordon: len=%d, %+v", len(extractedCordon), extractedCordon))
			capturingCordons = append(capturingCordons, extractedCordon)
		}

		joints := make([]CordonIndexKey, 0)
		for _, key := range extractedCordon {
			neighbours := Neighbours(key, cordonExtractor.Index)
			remainingNeighbours := slices.DeleteFunc(neighbours, func(candidate CordonIndexKey) bool {
				return slices.Contains(extractedCordon, candidate)
			})
			if len(remainingNeighbours) > 0 {
				joints = append(joints, key)
			}
		}

		for _, key := range extractedCordon {
			if slices.Contains(joints, key) {
				continue
			}

			delete(cordonExtractor.Index, key)
		}

		if len(cordonExtractor.Index) == 0 {
			break
		}
	}

	return capturingCordons
}

func Neighbours(currentKey CordonIndexKey, cordonIndex map[CordonIndexKey]any) []CordonIndexKey {
	neighbours := []CordonIndexKey{
		{Row: currentKey.Row - 1, Col: currentKey.Col - 1},
		{Row: currentKey.Row - 1, Col: currentKey.Col},
		{Row: currentKey.Row - 1, Col: currentKey.Col + 1},
		{Row: currentKey.Row, Col: currentKey.Col + 1},
		{Row: currentKey.Row + 1, Col: currentKey.Col + 1},
		{Row: currentKey.Row + 1, Col: currentKey.Col},
		{Row: currentKey.Row + 1, Col: currentKey.Col - 1},
		{Row: currentKey.Row, Col: currentKey.Col - 1},
	}

	existingNeighbours := slices.DeleteFunc(neighbours, func(n CordonIndexKey) bool {
		_, found := cordonIndex[n]
		return !found
	})

	return existingNeighbours
}

func (g *Game) AdvanceCordon(cordon *Cordon, currentKey CordonIndexKey, previousKey CordonIndexKey) bool {
	cordon.Ordered = append(cordon.Ordered, currentKey)

	existingNeighbours := Neighbours(currentKey, cordon.Index)
	slices.SortStableFunc(existingNeighbours, func(k1, k2 CordonIndexKey) int {
		dc1 := currentKey.Col - k1.Col
		dr1 := currentKey.Row - k1.Row
		distance1 := dc1*dc1 + dr1*dr1

		dc2 := currentKey.Col - k2.Col
		dr2 := currentKey.Row - k2.Row
		distance2 := dc2*dc2 + dr2*dr2

		return cmp.Compare(distance1, distance2)
	})

	for _, n := range existingNeighbours {
		if n == previousKey {
			continue
		}

		if slices.Contains(cordon.Ordered, n) {
			// Ordered cordon has enclosed segment,
			// which could be the entire cordon or part of it
			cordon.Ordered = append(cordon.Ordered, n)
			return true
		}

		if g.AdvanceCordon(cordon, n, currentKey) {
			return true
		}
	}
	return false
}

func (g *Game) Move(offenderIndex PlayerIndex, row uint8, col uint8) error {
	if offenderIndex >= g.PlayerCount {
		return fmt.Errorf("invalid player: %w", ErrUnknownPlayer)
	}

	g.GameField.Transform(
		func(candidate Dot) bool {
			return col == candidate.Col && row == candidate.Row && !candidate.Killed && !candidate.Owned
		},
		func(oldDot Dot) Dot {
			return oldDot.WithOwner(offenderIndex)
		})

	for defenderIdx := PlayerIndex(0); defenderIdx < g.PlayerCount; defenderIdx++ {
		if defenderIdx == offenderIndex {
			continue
		}

		defenderDots := g.GameField.Find(
			func(dot Dot) bool {
				return dot.Owned && dot.Owner == PlayerIndex(defenderIdx)
			})

		floodFillGrid := g.GameField.ToFloodFillGrid()
		g.RunFloodFill(floodFillGrid, defenderDots, defenderIdx)
		//g.ExtractCordon(floodFillGrid)

		defenderTrappedFilter := func(dot Dot) bool {
			return dot.Owned && dot.Owner == defenderIdx && floodFillGrid[dot.Row][dot.Col] == FloodFillCellStateBlocked
		}

		trappedDots := g.GameField.Find(
			defenderTrappedFilter)
		g.Players[offenderIndex].Score += uint32(len(trappedDots))

		g.GameField.Transform(
			defenderTrappedFilter,
			func(dot Dot) Dot {
				return dot.WithKilled()
			})

		/*
			offenderDotsFilter := func(dot Dot) bool {
				return !dot.Killed && dot.Owned && dot.Owner == offenderIndex
			}

				cordonDots := g.GameField.Find(offenderDotsFilter)
				if len(cordonDots) > 0 {
					topLeftDot := cordonDots[0]
				}
		*/
	}

	return nil
}
