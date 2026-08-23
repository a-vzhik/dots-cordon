// Package engine contains game primitives and compute game state after each move.
package engine

import (
	"fmt"
	"slices"
)

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
