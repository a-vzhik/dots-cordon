package inference

import (
	"errors"
	"fmt"
	"math"

	pb "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
)

// EncodeState mirrors dots_cordon_ml.encoding.encode_state, including captured
// dots in episode progress and excluding captured empty intersections.
func EncodeState(game *pb.GameState, rows, columns int) ([]float32, error) {
	board := game.GetBoard()
	if rows < 1 || rows > 255 || columns < 1 || columns > 255 || int(board.GetRows()) != rows || int(board.GetColumns()) != columns {
		return nil, fmt.Errorf("invalid game dimensions for encoding %dx%d", rows, columns)
	}
	n := rows * columns
	if game.GetTerminal() {
		return nil, errors.New("game is already over")
	}
	if game.GetMaxTurns() != 0 {
		return nil, errors.New("policy play requires a full-board game")
	}
	seat := game.GetNextTurnBy()
	if len(board.Cells) != n || seat > 1 || len(game.GetScores()) != 2 {
		return nil, errors.New("invalid game state")
	}
	state := make([]float32, 5*n)
	moves := 0
	for i, cell := range board.Cells {
		if cell > 5 {
			return nil, errors.New("invalid cell value")
		}
		if cell == byte(seat+1) {
			state[i] = 1
		}
		if cell == byte(2-seat) {
			state[n+i] = 1
		}
		if cell >= 3 {
			state[2*n+i] = 1
		}
		if cell == 1 || cell == 2 || cell == 4 || cell == 5 {
			moves++
		}
	}
	// Convert through float64 just as Python does before filling float32 planes.
	score := float32((float64(game.Scores[seat]) - float64(game.Scores[1-seat])) / float64(n))
	progress := float32(float64(moves) / float64(n))
	for i := range n {
		state[3*n+i], state[4*n+i] = score, progress
	}
	return state, nil
}

func SelectMove(scores []float32, board *pb.Board) (*pb.Coordinate, error) {
	if board == nil || board.Columns == 0 || len(scores) != len(board.Cells) {
		return nil, errors.New("invalid action scores or board")
	}
	best := -1
	for i, score := range scores {
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			return nil, errors.New("non-finite model output")
		}
		if board.Cells[i] == 0 && (best < 0 || score > scores[best]) {
			best = i
		}
	}
	if best < 0 {
		return nil, errors.New("no legal moves")
	}
	return &pb.Coordinate{Row: uint32(best) / board.Columns, Column: uint32(best) % board.Columns}, nil
}
