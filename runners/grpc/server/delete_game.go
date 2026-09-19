package grpcserver

import (
	"context"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (s *Service) DeleteGame(
	_ context.Context,
	request *dotscordonv1.DeleteGameRequest,
) (*dotscordonv1.DeleteGameResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}
	lock, err := s.tryLockGame(request.GetGameId(), gameLockTimeout)
	if err != nil {
		return nil, err
	}
	defer lock.Release()
	_, err = s.getGame(request.GetGameId())
	if err != nil {
		return nil, err
	}

	s.games.Delete(request.GetGameId())
	s.gameCount.Add(-1)
	return &dotscordonv1.DeleteGameResponse{}, nil
}
