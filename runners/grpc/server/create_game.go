package grpcserver

import (
	"context"
	"crypto/rand"
	"encoding/hex"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
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

	record := &gameRecord{
		lock:     &inProcessLock{},
		game:     s.newGame(gameID, uint8(request.GetRows()), uint8(request.GetColumns())),
		maxTurns: request.GetMaxTurns(),
	}
	s.games[gameID] = record
	return &dotscordonv1.CreateGameResponse{
		Game: gameStateToProto(gameID, record.game, record.turn, record.maxTurns),
	}, nil
}

func (s *Service) newGame(gameID string, rows, columns uint8) *engine.Game {
	recorder := engine.Recorder(engine.NoopGameRecorder{})
	if s.recorderFactory != nil {
		if configured := s.recorderFactory(gameID); configured != nil {
			recorder = configured
		}
	}
	return engine.NewGame(
		engine.NewGameField(columns, rows),
		[]*engine.Player{{Color: engine.BlueColor}, {Color: engine.RedColor}},
		recorder,
	)
}
