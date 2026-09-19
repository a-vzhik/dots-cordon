package grpcserver

import (
	"context"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (s *Service) GetGame(
	_ context.Context,
	request *dotscordonv1.GetGameRequest,
) (*dotscordonv1.GetGameResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}

	record, err := s.tryLockGame(request.GetGameId(), gameLockTimeout)
	if err != nil {
		return nil, err
	}
	defer record.lock.ReleaseLock()
	return &dotscordonv1.GetGameResponse{
		Game: gameStateToProto(request.GetGameId(), record.game, record.turn, record.maxTurns),
	}, nil
}
