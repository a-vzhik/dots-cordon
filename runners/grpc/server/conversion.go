package grpcserver

import (
	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
)

func cellToProto(dot engine.Dot) dotscordonv1.Cell {
	if dot.Killed {
		if !dot.Owned {
			return dotscordonv1.Cell_CELL_DEAD_EMPTY
		}
		if dot.IsOwnedBy(0) {
			return dotscordonv1.Cell_CELL_DEAD_PLAYER_0
		}
		return dotscordonv1.Cell_CELL_DEAD_PLAYER_1
	}
	if !dot.Owned {
		return dotscordonv1.Cell_CELL_EMPTY
	}
	if dot.IsOwnedBy(0) {
		return dotscordonv1.Cell_CELL_PLAYER_0
	}
	return dotscordonv1.Cell_CELL_PLAYER_1
}

func moveResultToProto(
	player engine.PlayerIndex,
	result *engine.MoveResult,
) *dotscordonv1.MoveResult {
	converted := &dotscordonv1.MoveResult{
		Player:       uint32(player),
		ScoredPoints: result.ScoredPoints,
		KilledCells:  make([]*dotscordonv1.Coordinate, 0, len(result.KilledDots)),
		Cordons:      make([]*dotscordonv1.Cordon, 0, len(result.Cordons)),
	}
	for _, dot := range result.KilledDots {
		converted.KilledCells = append(converted.KilledCells, coordinateToProto(dot.Coord))
	}
	for _, cordon := range result.Cordons {
		convertedCordon := &dotscordonv1.Cordon{
			Positions: make([]*dotscordonv1.Coordinate, 0, len(cordon)),
		}
		for _, dot := range cordon {
			convertedCordon.Positions = append(
				convertedCordon.Positions,
				coordinateToProto(dot.Coord),
			)
		}
		converted.Cordons = append(converted.Cordons, convertedCordon)
	}
	return converted
}

func coordinateToProto(coord engine.Coord) *dotscordonv1.Coordinate {
	return &dotscordonv1.Coordinate{
		Row:    uint32(coord.Row),
		Column: uint32(coord.Col),
	}
}

func gameStateToProto(gameID string, game *engine.Game, turn, maxTurns uint32) *dotscordonv1.GameState {
	field := game.GameField
	cells := make([]byte, 0, int(field.Width)*int(field.Height))
	for _, row := range field.Dots {
		for _, dot := range row {
			cells = append(cells, byte(cellToProto(dot)))
		}
	}
	termination := gameTerminationReason(game, turn, maxTurns)
	return &dotscordonv1.GameState{
		GameId: gameID,
		Board: &dotscordonv1.Board{
			Rows:    uint32(field.Height),
			Columns: uint32(field.Width),
			Cells:   cells,
		},
		Scores:            []uint32{game.Players[0].Score, game.Players[1].Score},
		CurrentPlayer:     turn % 2,
		Turn:              turn,
		Terminal:          termination != dotscordonv1.TerminationReason_TERMINATION_REASON_UNSPECIFIED,
		TerminationReason: termination,
	}
}

func gameTerminationReason(game *engine.Game, turn, maxTurns uint32) dotscordonv1.TerminationReason {
	if game.GameField.IsFull() {
		return dotscordonv1.TerminationReason_TERMINATION_REASON_BOARD_FULL
	}
	if maxTurns > 0 && turn >= maxTurns {
		return dotscordonv1.TerminationReason_TERMINATION_REASON_TURN_LIMIT
	}
	return dotscordonv1.TerminationReason_TERMINATION_REASON_UNSPECIFIED
}
