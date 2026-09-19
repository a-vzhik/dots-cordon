package grpcserver

import (
	"context"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

func (s *Service) GetGame(
	_ context.Context,
	request *dotscordonv1.GetGameRequest,
) (*dotscordonv1.GetGameResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
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
	return &dotscordonv1.GetGameResponse{
		Game: proto.Clone(state).(*dotscordonv1.GameState),
	}, nil
}
