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

type Game struct {
	GameField   *GameField
	Players     []*Player
	PlayerCount uint8
}

func NewGame(gameField *GameField, players []*Player) *Game {
	game := &Game{
		GameField: gameField,
		Players:   players,
	}

	game.PlayerCount = uint8(len(players))
	return game
}

func (g *Game) Move(playerIndex PlayerIndex, row uint8, col uint8) error {
	if uint8(playerIndex) >= g.PlayerCount {
		return fmt.Errorf("invalid player: %w", ErrUnknownPlayer)
	}

	g.GameField.Transform(
		func(candidate Dot) bool {
			return col == candidate.Col && row == candidate.Row && !candidate.Killed && !candidate.Owned
		},
		func(oldDot Dot) Dot {
			return oldDot.WithOwner(playerIndex)
		})

	activePlayer := g.Players[playerIndex]
	activePlayer.Score = 0
	return nil
}
