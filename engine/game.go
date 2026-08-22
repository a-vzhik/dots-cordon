// Package engine contains game primitives and compute game state after each move.
package engine

import (
	"fmt"
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

		floodFillGrid := RunFloodFill(g.GameField, defenderDots, defenderIdx)
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
