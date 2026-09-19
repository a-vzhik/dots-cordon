package grpcserver

import (
	"context"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (s *Service) ResetGame(
	_ context.Context,
	request *dotscordonv1.ResetGameRequest,
) (*dotscordonv1.ResetGameResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}

	lock, err := s.tryLockGame(request.GetGameId(), gameLockTimeout)
	if err != nil {
		return nil, err
	}
	defer lock.Release()
	record, err := s.getGame(request.GetGameId())
	if err != nil {
		return nil, err
	}
	field := record.game.GameField
	record.game = s.newGame(request.GetGameId(), field.Height, field.Width)
	record.turn = 0
	return &dotscordonv1.ResetGameResponse{
		Game: gameStateToProto(request.GetGameId(), record.game, record.turn, record.maxTurns),
	}, nil
}
