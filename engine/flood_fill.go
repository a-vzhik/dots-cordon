package engine

import (
	"fmt"
	"log/slog"
	"strings"
)

type FloodFillCellState uint8

const (
	FloodFillCellStateNone FloodFillCellState = iota
	FloodFillCellStateBlocked
	FloodFillCellStateEscaped
	FloodFillCellStateEscapeCandidate
	FloodFillCellStateKilled
)

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

func ToFloodFillGrid(gf *GameField) [][]FloodFillCellState {
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
			case FloodFillCellStateKilled:
				rune = 'x'
			}
			builder.WriteRune(rune)
			builder.WriteByte(' ')
		}
		slog.Info(builder.String())
	}
}

func RunFloodFill(gameField *GameField, defenderDots []Dot, defenderIdx PlayerIndex) [][]FloodFillCellState {
	floodFillGrid := ToFloodFillGrid(gameField)

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
		if HasDotEscaped(gameField, floodFillGrid, defenderIdx, dot.Row, dot.Col) {
			setEscapeCandidatesToState(floodFillGrid, FloodFillCellStateEscaped)
		} else {
			setEscapeCandidatesToState(floodFillGrid, FloodFillCellStateBlocked)
		}
	}
	return floodFillGrid
}

func HasDotEscaped(gameField *GameField, floodFillGrid [][]FloodFillCellState, defenderIdx PlayerIndex, currRowIdx uint8, currColIdx uint8) bool {
	knownState := floodFillGrid[currRowIdx][currColIdx]
	switch knownState {
	case FloodFillCellStateKilled:
		return false
	case FloodFillCellStateBlocked:
		return false
	case FloodFillCellStateEscaped:
		return true
	case FloodFillCellStateEscapeCandidate:
		return false
	}

	dot := gameField.Dots[currRowIdx][currColIdx]
	if dot.Killed {
		floodFillGrid[currRowIdx][currColIdx] = FloodFillCellStateKilled
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

	if currColIdx == 0 || currRowIdx == 0 || currRowIdx == gameField.Height-1 || currColIdx == gameField.Width-1 {
		slog.Debug(fmt.Sprintf("reached the border: %d, %d", currRowIdx, currColIdx))
		return true
	}

	prevRowIdx := currRowIdx - 1
	prevColIdx := currColIdx - 1
	nextRowIdx := currRowIdx + 1
	nextColIdx := currColIdx + 1

	if currRowIdx < gameField.Height-1 {
		if HasDotEscaped(gameField, floodFillGrid, defenderIdx, nextRowIdx, currColIdx) {
			return true
		}
	}

	if currRowIdx > 0 {
		if HasDotEscaped(gameField, floodFillGrid, defenderIdx, prevRowIdx, currColIdx) {
			return true
		}
	}

	if currColIdx < gameField.Width-1 {
		if HasDotEscaped(gameField, floodFillGrid, defenderIdx, currRowIdx, nextColIdx) {
			return true
		}
	}

	if currColIdx > 0 {
		if HasDotEscaped(gameField, floodFillGrid, defenderIdx, currRowIdx, prevColIdx) {
			return true
		}
	}

	return false
}
