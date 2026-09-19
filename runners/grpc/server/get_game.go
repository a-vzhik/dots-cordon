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

	session, err := s.getSession(request.GetGameId())
	if err != nil {
		return nil, err
	}

	session.mu.Lock()
	defer session.mu.Unlock()
	if session.deleted {
		return nil, gameNotFound(request.GetGameId())
	}

	return &dotscordonv1.GetGameResponse{Game: session.snapshotLocked()}, nil
}
