// Package opponent adapts native ONNX inference to the desktop session contract.
package opponent

import (
	"fmt"

	pb "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/a-vzhik/dots-cordon/inference"
	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/session"
)

type Model struct{ model *inference.Model }

var _ session.Opponent = (*Model)(nil)

// Load loads once and validates the desktop board before the window opens.
func Load(weights, runtime string) (*Model, error) {
	model, err := inference.LoadWeights(weights, runtime)
	if err != nil {
		return nil, err
	}
	if err := model.ValidateBoardSize(session.Rows, session.Columns); err != nil {
		model.Close()
		return nil, err
	}
	return &Model{model: model}, nil
}

func (m *Model) Close() { m.model.Close() }

func (m *Model) Move(snapshot session.Snapshot) (engine.Coord, error) {
	game, err := toProto(snapshot)
	if err != nil {
		return engine.Coord{}, err
	}
	state, err := inference.EncodeState(game, session.Rows, session.Columns)
	if err != nil {
		return engine.Coord{}, err
	}
	scores, err := m.model.Infer(state, session.Rows, session.Columns)
	if err != nil {
		return engine.Coord{}, err
	}
	move, err := inference.SelectMove(scores, game.Board)
	if err != nil {
		return engine.Coord{}, err
	}
	return engine.Coord{Row: uint8(move.Row), Col: uint8(move.Column)}, nil
}

func toProto(snapshot session.Snapshot) (*pb.GameState, error) {
	if len(snapshot.Dots) != session.Rows {
		return nil, fmt.Errorf("expected %d rows", session.Rows)
	}
	cells := make([]byte, 0, session.Rows*session.Columns)
	for _, row := range snapshot.Dots {
		if len(row) != session.Columns {
			return nil, fmt.Errorf("expected %d columns", session.Columns)
		}
		for _, dot := range row {
			var cell byte
			if dot.Owned {
				if dot.Owner > 1 {
					return nil, fmt.Errorf("invalid dot owner %d", dot.Owner)
				}
				cell = byte(dot.Owner) + 1
			}
			if dot.Killed {
				cell += 3
			}
			cells = append(cells, cell)
		}
	}
	return &pb.GameState{Board: &pb.Board{Rows: session.Rows, Columns: session.Columns, Cells: cells}, Scores: []uint32{snapshot.Scores[0], snapshot.Scores[1]}, NextTurnBy: uint32(snapshot.NextPlayer), Terminal: snapshot.Terminal}, nil
}
