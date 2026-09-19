package grpcserver

import (
	"context"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
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

	session, err := s.getSession(request.GetGameId())
	if err != nil {
		return nil, err
	}

	session.mu.Lock()
	defer session.mu.Unlock()
	if session.deleted {
		return nil, gameNotFound(request.GetGameId())
	}
	if session.terminal {
		return nil, status.Error(codes.FailedPrecondition, "game is terminal; reset it before moving")
	}
	if request.GetExpectedTurn() != session.turn {
		return nil, status.Errorf(
			codes.FailedPrecondition,
			"expected turn %d, current turn is %d",
			request.GetExpectedTurn(),
			session.turn,
		)
	}

	return session.moveLocked(request.GetPosition())
}
