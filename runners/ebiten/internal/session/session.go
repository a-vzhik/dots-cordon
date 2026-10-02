// Package session owns game state independently of the native UI.
package session

import "github.com/a-vzhik/dots-cordon/engine"

const (
	Rows    = 10
	Columns = 15
)

type Cordon struct {
	Player engine.PlayerIndex
	Points []engine.Coord
}

// Snapshot is a detached copy of the state; callers may modify it freely.
type Snapshot struct {
	Dots       [][]engine.Dot
	Scores     [2]uint32
	NextPlayer engine.PlayerIndex
	Terminal   bool
	Cordons    []Cordon
}

type Opponent interface {
	Move(Snapshot) (engine.Coord, error)
}

// Session is used by one goroutine. Share snapshots with other goroutines.
type Session struct {
	game       *engine.Game
	nextPlayer engine.PlayerIndex
	terminal   bool
	cordons    []Cordon
}

func New() *Session {
	return &Session{game: engine.NewGame(
		engine.NewGameField(Columns, Rows),
		[]*engine.Player{{Color: engine.RedColor}, {Color: engine.BlueColor}},
		engine.NoopGameRecorder{},
	)}
}

func (s *Session) Snapshot() Snapshot {
	result := Snapshot{
		Dots:       make([][]engine.Dot, Rows),
		Scores:     [2]uint32{s.game.Players[0].Score, s.game.Players[1].Score},
		NextPlayer: s.nextPlayer,
		Terminal:   s.terminal,
		Cordons:    make([]Cordon, len(s.cordons)),
	}
	for row, dots := range s.game.GameField.Dots {
		result.Dots[row] = append([]engine.Dot(nil), dots...)
	}
	for i, cordon := range s.cordons {
		result.Cordons[i] = Cordon{Player: cordon.Player, Points: append([]engine.Coord(nil), cordon.Points...)}
	}
	return result
}

// Move places the current player's dot and advances the turn only on success.
func (s *Session) Move(row, col uint8) error {
	result, err := s.game.Move(s.nextPlayer, row, col)
	if err != nil {
		return err
	}
	for _, dots := range result.Cordons {
		cordon := Cordon{Player: s.nextPlayer, Points: make([]engine.Coord, len(dots))}
		for i, dot := range dots {
			cordon.Points[i] = dot.Coord
		}
		s.cordons = append(s.cordons, cordon)
	}
	s.terminal = result.IsTerminal
	s.nextPlayer = s.nextPlayer.EnemyIndex()
	return nil
}
