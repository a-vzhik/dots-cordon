// Package engine contains game primitives and compute game state after each move.
package engine

import (
	"fmt"
	"slices"
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

func (gf *GameField) Transform(row uint8, col uint8, applyFunc func(Dot) Dot) {
	oldDot := gf.Dots[row][col]
	gf.Dots[row][col] = applyFunc(oldDot)
}

func (gf *GameField) TransformMany(sourceDots []Dot, applyFunc func(Dot) Dot) {
	for _, source := range sourceDots {
		gf.Dots[source.Row][source.Col] = applyFunc(source)
	}
}

func (gf *GameField) TransformFunc(filterFunc func(Dot) bool, applyFunc func(Dot) Dot) {
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

type MoveResult struct {
	ScoredPoints uint32
	Cordons      [][]Dot
	KilledDots   []Dot
	IsTerminal   bool
}

func NewGame(gameField *GameField, players []*Player) *Game {
	game := &Game{
		GameField: gameField,
		Players:   players,
	}

	game.PlayerCount = PlayerIndex(len(players))
	return game
}

func (g *Game) Move(offenderIndex PlayerIndex, row uint8, col uint8) (*MoveResult, error) {
	if offenderIndex >= g.PlayerCount {
		return nil, fmt.Errorf("invalid player: %w", ErrUnknownPlayer)
	}

	if row >= g.GameField.Height || col >= g.GameField.Width {
		return nil, fmt.Errorf("invalid move: %w", ErrInvalidMove)
	}

	attemptedDot := g.GameField.Dots[row][col]
	if attemptedDot.Killed || attemptedDot.Owned {
		return nil, fmt.Errorf("dot must be empty: %w", ErrNonEmptyDot)
	}

	// Mark claimed dot.
	g.GameField.Transform(row, col, func(oldDot Dot) Dot {
		return oldDot.WithOwner(offenderIndex)
	})

	// Find active denfender dots and find cordons.
	defenderIdx := offenderIndex.EnemyIndex()
	activeDefenderDots := g.GameField.Find(
		func(dot Dot) bool {
			return !dot.Killed && dot.IsOwnedBy(defenderIdx)
		})

	floodFillGrid := RunFloodFill(g.GameField, activeDefenderDots, defenderIdx)
	cordonPathFinder := NewCordonPathFinder()
	cordons := cordonPathFinder.FindCordons(floodFillGrid)

	// Return if no cordons found.
	if len(cordons) == 0 {
		return &MoveResult{}, nil
	}

	// Find all killed dots inside all cordons (these can be empty, offender's or defender's)
	killedDots := g.GameField.Find(func(dot Dot) bool {
		if dot.Killed {
			return false
		}

		if floodFillGrid[dot.Row][dot.Col] != FloodFillCellStateBlocked {
			return false
		}

		for _, cordon := range cordons {
			if slices.Contains(cordon, CordonIndexKey{Row: dot.Row, Col: dot.Col}) {
				return false
			}
		}

		return true
	})

	// Update scored points based on defender's killed dots only.
	scoredPoints := uint32(0)
	for _, trappedDot := range killedDots {
		if trappedDot.IsOwnedBy(defenderIdx) {
			scoredPoints++
		}
	}
	g.Players[offenderIndex].Score += uint32(scoredPoints)

	// Mark all killed dots as killed.
	g.GameField.TransformMany(killedDots, func(dot Dot) Dot {
		return dot.WithKilled()
	})

	// Check if there are further available moves.
	emptyDots := g.GameField.Find(func(dot Dot) bool {
		return !dot.Killed && !dot.Owned
	})

	// Coonvert internal cordons to cordons of Dot
	dotCordons := make([][]Dot, 0, len(cordons))
	for _, cordon := range cordons {
		dotCordon := make([]Dot, 0, len(cordon))
		for _, cordonKey := range cordon {
			dotCordon = append(dotCordon, g.GameField.Dots[cordonKey.Row][cordonKey.Col])
		}
		dotCordons = append(dotCordons, dotCordon)
	}

	return &MoveResult{
		ScoredPoints: scoredPoints,
		KilledDots:   killedDots,
		Cordons:      dotCordons,
		IsTerminal:   len(emptyDots) == 0,
	}, nil
}
