package grpcserver

import (
	"context"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
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
	state, err := s.getGame(request.GetGameId())
	if err != nil {
		return nil, err
	}
	reset := s.newGame(request.GetGameId(), uint8(state.Board.Rows), uint8(state.Board.Columns), state.MaxTurns)
	s.games.Store(request.GetGameId(), reset)
	return &dotscordonv1.ResetGameResponse{Game: proto.Clone(reset).(*dotscordonv1.GameState)}, nil
}
