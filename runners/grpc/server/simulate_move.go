package grpcserver

import (
	"context"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// SimulateMove reconstructs an isolated engine state. Search does not allocate
// server records, consume maxGames, invoke recorders, or mutate any live game.
// Structural validation does not certify that a supplied position is reachable.
func (s *Service) SimulateMove(ctx context.Context, request *dotscordonv1.SimulateMoveRequest) (*dotscordonv1.MakeMoveResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, status.FromContextError(err).Err()
	}
	if request == nil || request.Game == nil || request.Game.Board == nil || request.Position == nil {
		return nil, status.Error(codes.InvalidArgument, "game, board, and position are required")
	}
	if request.Player == nil || request.GetPlayer() > 1 {
		return nil, status.Error(codes.InvalidArgument, "player must be explicitly set to 0 or 1")
	}
	state := request.Game
	board := state.Board
	if err := validateDimensions(board.Rows, board.Columns); err != nil {
		return nil, err
	}
	if len(board.Cells) != int(board.Rows*board.Columns) || len(state.Scores) != 2 || state.NextTurnBy > 1 {
		return nil, status.Error(codes.InvalidArgument, "invalid search state dimensions, scores, or next player")
	}
	if uint64(state.Scores[0])+uint64(state.Scores[1]) > uint64(board.Rows*board.Columns) {
		return nil, status.Error(codes.InvalidArgument, "scores exceed board size")
	}
	for _, value := range board.Cells {
		if value > byte(dotscordonv1.Cell_CELL_DEAD_PLAYER_1) {
			return nil, status.Error(codes.InvalidArgument, "invalid cell value")
		}
	}
	return applyMove(state, request.GetPlayer(), request.Position, nil)
}
