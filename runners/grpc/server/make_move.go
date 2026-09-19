package grpcserver

import (
	"context"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

func (s *Service) MakeMove(
	_ context.Context,
	request *dotscordonv1.MakeMoveRequest,
) (*dotscordonv1.MakeMoveResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}
	if request.Player == nil || request.GetPlayer() > 1 {
		return nil, status.Error(codes.InvalidArgument, "player must be explicitly set to 0 or 1")
	}
	if request.GetPosition() == nil {
		return nil, status.Error(codes.InvalidArgument, "position is required")
	}

	lock, err := s.tryLockGame(request.GetGameId(), gameLockTimeout)
	if err != nil {
		return nil, err
	}
	defer lock.Release()
	state, err := s.getGame(request.GetGameId())
	if err != nil {
		return nil, err
	}
	response, err := applyMove(state, request.GetPlayer(), request.GetPosition(), func(player engine.PlayerIndex, result *engine.MoveResult) {
		if recorder, ok := s.recorders.Load(request.GetGameId()); ok {
			recorder.(engine.Recorder).RecordMove(player, uint8(request.Position.Row), uint8(request.Position.Column), *result)
		}
	})
	if err != nil {
		return nil, err
	}
	s.games.Store(request.GetGameId(), proto.Clone(response.Game).(*dotscordonv1.GameState))
	return response, nil
}

// applyMove checks protocol rules, applies an engine move, and returns the new
// protocol state. It does not mutate the supplied state or derive player identity.
func applyMove(state *dotscordonv1.GameState, player uint32, position *dotscordonv1.Coordinate, recordMove func(engine.PlayerIndex, *engine.MoveResult)) (*dotscordonv1.MakeMoveResponse, error) {
	if state.Terminal {
		return nil, status.Error(codes.FailedPrecondition, "game is terminal; reset it before moving")
	}
	if player != state.NextTurnBy {
		return nil, status.Errorf(codes.FailedPrecondition, "player %d is out of turn; next turn is by player %d", player, state.NextTurnBy)
	}
	if position.GetRow() >= state.Board.Rows || position.GetColumn() >= state.Board.Columns {
		return nil, status.Errorf(codes.InvalidArgument, "position (%d, %d) is outside the %dx%d board", position.GetRow(), position.GetColumn(), state.Board.Rows, state.Board.Columns)
	}
	game := gameFromProto(state)
	if gameTerminationReason(game, state.MaxTurns) != dotscordonv1.TerminationReason_TERMINATION_REASON_UNSPECIFIED {
		return nil, status.Error(codes.FailedPrecondition, "game is terminal; reset it before moving")
	}
	movingPlayer := engine.PlayerIndex(player)
	result, err := game.Move(movingPlayer, uint8(position.GetRow()), uint8(position.GetColumn()))
	if err != nil {
		return nil, moveStatus(err)
	}
	if recordMove != nil {
		recordMove(movingPlayer, result)
	}
	return &dotscordonv1.MakeMoveResponse{
		Game:   gameStateToProto(state.GameId, game, uint32(movingPlayer.EnemyIndex()), state.MaxTurns),
		Result: moveResultToProto(movingPlayer, result),
	}, nil
}
