// Package engine contains game primitives and compute game state after each move.
package engine

import (
	"fmt"
	"log/slog"
)

type Game struct {
	GameField   *GameField
	Players     []*Player
	PlayerCount PlayerIndex
	recorder    Recorder
}

type MoveResult struct {
	ScoredPoints uint32
	Cordons      [][]Dot
	KilledDots   []Dot
	IsTerminal   bool
}

func NewGame(gameField *GameField, players []*Player, recorder Recorder) *Game {
	game := &Game{
		GameField: gameField,
		Players:   players,
		recorder:  recorder,
	}

	game.PlayerCount = PlayerIndex(len(players))
	game.recorder.RecordStart(gameField.Width, gameField.Height)
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
	activeNonOffenderDots := g.GameField.Find(
		func(dot Dot) bool {
			return !dot.Killed && (dot.Owned == false || dot.IsOwnedBy(defenderIdx))
		})

	floodFillGrid := RunFloodFill(g.GameField, activeNonOffenderDots, defenderIdx)
	slog.Debug(FloodFillGridToString(floodFillGrid))

	cordonPathFinder := NewCordonPathFinder()
	cordons := cordonPathFinder.FindCordons(floodFillGrid)

	// Return if no cordons found.
	if len(cordons) == 0 {
		return g.recordMove(offenderIndex, row, col, &MoveResult{
			IsTerminal: g.GameField.IsFull(),
		})
	}

	// Find all killed dots inside all cordons (these can be empty, offender's or defender's)
	killedDots := g.GameField.Find(func(dot Dot) bool {
		if dot.Killed {
			return false
		}

		if floodFillGrid[dot.Row][dot.Col] != FloodFillCellStateBlocked {
			return false
		}
		/*
			fourNeigbours := []Coord{
				{Row: dot.Row - 1, Col: dot.Col},
				{Row: dot.Row + 1, Col: dot.Col},
				{Row: dot.Row, Col: dot.Col - 1},
				{Row: dot.Row, Col: dot.Col + 1},
			}

			insideCordon := true
			for _, n := range fourNeigbours {
				if n.Col < 0 || n.Row < 0 || n.Row >= uint8(len(floodFillGrid)) || n.Col >= uint8(len(floodFillGrid)) {
					continue
				}

				if floodFillGrid[n.Row][n.Col] != FloodFillCellStateBlocked {
					insideCordon = false
					break
				}
			}

			if !insideCordon {
				return false
			}

			for _, cordon := range cordons {
				if slices.Contains(cordon, Coord{Row: dot.Row, Col: dot.Col}) {
					return false
				}
			}
		*/
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

	// Coonvert internal cordons to cordons of Dot
	dotCordons := make([][]Dot, 0, len(cordons))
	for _, cordon := range cordons {
		dotCordon := make([]Dot, 0, len(cordon))
		for _, cordonKey := range cordon {
			dotCordon = append(dotCordon, g.GameField.Dots[cordonKey.Row][cordonKey.Col])
		}
		dotCordons = append(dotCordons, dotCordon)
	}

	return g.recordMove(offenderIndex, row, col, &MoveResult{
		ScoredPoints: scoredPoints,
		KilledDots:   killedDots,
		Cordons:      dotCordons,
		IsTerminal:   g.GameField.IsFull(),
	})
}

func (g *Game) recordMove(
	playerIdx PlayerIndex,
	row uint8,
	col uint8,
	result *MoveResult,
) (*MoveResult, error) {
	g.recorder.RecordMove(playerIdx, row, col, *result)
	return result, nil
}
