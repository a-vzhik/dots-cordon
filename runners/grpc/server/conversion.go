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
