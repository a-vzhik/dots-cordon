package grpcserver

import (
	"context"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// SimulateMove reconstructs an isolated engine state. Search does not allocate
// sessions, consume maxGames, invoke recorders, or mutate any live game.
// Structural validation does not certify that a supplied position is reachable.
func (s *Service) SimulateMove(ctx context.Context, request *dotscordonv1.SimulateMoveRequest) (*dotscordonv1.MakeMoveResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, status.FromContextError(err).Err()
	}
	if request == nil || request.Game == nil || request.Game.Board == nil || request.Position == nil {
		return nil, status.Error(codes.InvalidArgument, "game, board, and position are required")
	}
	state := request.Game
	board := request.Game.Board
	if err := validateDimensions(board.Rows, board.Columns); err != nil {
		return nil, err
	}
	size := board.Rows * board.Columns
	if len(board.Cells) != int(size) || len(state.Scores) != 2 || state.CurrentPlayer > 1 || state.Turn > size || state.CurrentPlayer != state.Turn%2 {
		return nil, status.Error(codes.InvalidArgument, "invalid search state dimensions, scores, player, or turn")
	}
	if uint64(state.Scores[0])+uint64(state.Scores[1]) > uint64(size) {
		return nil, status.Error(codes.InvalidArgument, "scores exceed board size")
	}
	if state.Terminal || (request.MaxTurns > 0 && state.Turn >= request.MaxTurns) {
		return nil, status.Error(codes.FailedPrecondition, "cannot simulate a terminal position")
	}
	session, err := newGameSession("", uint8(board.Rows), uint8(board.Columns), request.MaxTurns, nil)
	if err != nil {
		return nil, err
	}
	if !session.TryAcquireLock() {
		return nil, ErrSessionBusy
	}
	defer session.ReleaseLock()
	placed := uint32(0)
	for index, value := range board.Cells {
		if value > byte(dotscordonv1.Cell_CELL_DEAD_PLAYER_1) {
			return nil, status.Error(codes.InvalidArgument, "invalid cell value")
		}
		dot := &session.game.GameField.Dots[index/int(board.Columns)][index%int(board.Columns)]
		dot.Killed = value >= byte(dotscordonv1.Cell_CELL_DEAD_EMPTY)
		switch dotscordonv1.Cell(value) {
		case dotscordonv1.Cell_CELL_PLAYER_0, dotscordonv1.Cell_CELL_DEAD_PLAYER_0:
			dot.Owned, dot.Owner = true, engine.PlayerIndex(0)
			placed++
		case dotscordonv1.Cell_CELL_PLAYER_1, dotscordonv1.Cell_CELL_DEAD_PLAYER_1:
			dot.Owned, dot.Owner = true, engine.PlayerIndex(1)
			placed++
		}
	}
	if placed != state.Turn {
		return nil, status.Error(codes.InvalidArgument, "turn does not match placed dots")
	}

	session.turn = state.Turn
	session.current = engine.PlayerIndex(state.CurrentPlayer)
	session.game.Players[0].Score = state.Scores[0]
	session.game.Players[1].Score = state.Scores[1]
	return session.move(request.Position)
}
