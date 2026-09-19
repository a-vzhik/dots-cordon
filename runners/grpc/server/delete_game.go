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
	record, err := s.tryLockGame(request.GetGameId(), gameLockTimeout)
	if err != nil {
		return nil, err
	}
	defer record.lock.ReleaseLock()

	// Requests that already found this record must observe deletion when they
	// acquire its lock. The map lock is held only for removal, never waiting.
	record.deleted = true
	s.mu.Lock()
	delete(s.games, request.GetGameId())
	s.mu.Unlock()
	return &dotscordonv1.DeleteGameResponse{}, nil
}
