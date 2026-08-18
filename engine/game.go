// Package engine contains game primitives and compute game state after each move.
package engine

import (
	"fmt"
	"log"
)

type GameField struct {
	Width  uint8
	Height uint8
	Dots   [][]Dot
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
	log.Printf("CanEscape [%d, %d]: %d\n", currRowIdx, currColIdx, floodFillGrid[currRowIdx][currColIdx])

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
	if !dot.Owned {
		floodFillGrid[currRowIdx][currColIdx] = FloodFillCellStateEscapeCandidate
	}
	if dot.Owned && dot.Owner == defenderIdx {
		floodFillGrid[currRowIdx][currColIdx] = FloodFillCellStateEscapeCandidate
	}
	if dot.Owned && dot.Owner != defenderIdx {
		floodFillGrid[currRowIdx][currColIdx] = FloodFillCellStateBlocked
		log.Printf("found the other player: %d, %d", currRowIdx, currColIdx)
		return false
	}

	if currColIdx == 0 || currRowIdx == 0 || currRowIdx == g.GameField.Height-1 || currColIdx == g.GameField.Width-1 {
		log.Printf("reached the border: %d, %d", currRowIdx, currColIdx)
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
	log.Printf("Marking all escape candidate with %d: %+v\n", state, floodFillGrid)
	for i := range floodFillGrid {
		for j := range floodFillGrid {
			if floodFillGrid[i][j] != FloodFillCellStateEscapeCandidate {
				continue
			}
			floodFillGrid[i][j] = state
		}
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
		log.Printf("FloodFill dot [%d, %d]...", dot.Row, dot.Col)
		if g.HasDotEscaped(floodFillGrid, defenderIdx, dot.Row, dot.Col) {
			setEscapeCandidatesToState(floodFillGrid, FloodFillCellStateEscaped)
		} else {
			setEscapeCandidatesToState(floodFillGrid, FloodFillCellStateBlocked)
		}
	}
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

		escapedDots := make([]Dot, 0, len(defenderDots))
		for _, defenderDot := range defenderDots {
			escapedDots = append(escapedDots, defenderDot)
		}

	}

	activePlayer := g.Players[offenderIndex]
	activePlayer.Score = 0
	return nil
}
