package grpcserver

import (
	"context"
	"crypto/rand"
	"encoding/hex"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func randomGameID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(value[:]), nil
}

func (s *Service) CreateGame(
	_ context.Context,
	request *dotscordonv1.CreateGameRequest,
) (*dotscordonv1.CreateGameResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}
	if err := validateDimensions(request.GetRows(), request.GetColumns()); err != nil {
		return nil, err
	}

	gameID, err := randomGameID()
	if err != nil {
		return nil, status.Errorf(codes.Internal, "generate game ID: %v", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.maxGames > 0 && len(s.games) >= s.maxGames {
		return nil, status.Errorf(
			codes.ResourceExhausted,
			"concurrent game limit of %d reached",
			s.maxGames,
		)
	}

	for s.games[gameID] != nil {
		gameID, err = randomGameID()
		if err != nil {
			return nil, status.Errorf(codes.Internal, "generate game ID: %v", err)
		}
	}

	session := newGameSession(
		gameID,
		uint8(request.GetRows()),
		uint8(request.GetColumns()),
		request.GetMaxTurns(),
		s.recorderFactory,
	)
	s.games[gameID] = session

	return &dotscordonv1.CreateGameResponse{Game: session.snapshotLocked()}, nil
}
