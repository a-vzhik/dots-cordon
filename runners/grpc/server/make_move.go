package grpcserver

import (
	"context"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (s *Service) MakeMove(
	_ context.Context,
	request *dotscordonv1.MakeMoveRequest,
) (*dotscordonv1.MakeMoveResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}
	if request.GetPosition() == nil {
		return nil, status.Error(codes.InvalidArgument, "position is required")
	}

	record, err := s.tryLockGame(request.GetGameId(), gameLockTimeout)
	if err != nil {
		return nil, err
	}
	defer record.lock.ReleaseLock()
	if gameTerminationReason(record.game, record.turn, record.maxTurns) != dotscordonv1.TerminationReason_TERMINATION_REASON_UNSPECIFIED {
		return nil, status.Error(codes.FailedPrecondition, "game is terminal; reset it before moving")
	}
	if request.GetExpectedTurn() != record.turn {
		return nil, status.Errorf(
			codes.FailedPrecondition,
			"expected turn %d, current turn is %d",
			request.GetExpectedTurn(),
			record.turn,
		)
	}

	response, err := applyMove(record.game, request.GetGameId(), record.turn, record.maxTurns, request.GetPosition())
	if err != nil {
		return nil, err
	}
	record.turn++
	return response, nil
}

// applyMove applies a sequential engine move and converts its response for both
// live play and isolated simulations. The caller owns access to the game.
func applyMove(game *engine.Game, gameID string, turn, maxTurns uint32, position *dotscordonv1.Coordinate) (*dotscordonv1.MakeMoveResponse, error) {
	field := game.GameField
	// Validate before narrowing protobuf coordinates to the engine's uint8.
	if position.GetRow() >= uint32(field.Height) || position.GetColumn() >= uint32(field.Width) {
		return nil, status.Errorf(
			codes.InvalidArgument,
			"position (%d, %d) is outside the %dx%d board",
			position.GetRow(), position.GetColumn(), field.Height, field.Width,
		)
	}
	player := engine.PlayerIndex(turn % 2)
	result, err := game.Move(player, uint8(position.GetRow()), uint8(position.GetColumn()))
	if err != nil {
		return nil, moveStatus(err)
	}
	return &dotscordonv1.MakeMoveResponse{
		Game:   gameStateToProto(gameID, game, turn+1, maxTurns),
		Result: moveResultToProto(player, result),
	}, nil
}
