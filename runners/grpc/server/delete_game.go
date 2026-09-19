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
	if request.GetGameId() == "" {
		return nil, status.Error(codes.InvalidArgument, "game_id is required")
	}

	s.mu.Lock()
	session := s.games[request.GetGameId()]
	if session == nil {
		s.mu.Unlock()
		return nil, gameNotFound(request.GetGameId())
	}
	delete(s.games, request.GetGameId())

	// Holding the store lock while marking the session deleted makes deletion
	// linearizable with a concurrent getSession call.
	session.mu.Lock()
	session.deleted = true
	session.mu.Unlock()
	s.mu.Unlock()

	return &dotscordonv1.DeleteGameResponse{}, nil
}
