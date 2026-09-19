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

	if !s.reserveGameSlot() {
		return nil, status.Errorf(codes.ResourceExhausted, "concurrent game limit of %d reached", s.maxGames)
	}
	created := false
	defer func() {
		if !created {
			s.gameCount.Add(-1)
		}
	}()

	for {
		gameID, err := randomGameID()
		if err != nil {
			return nil, status.Errorf(codes.Internal, "generate game ID: %v", err)
		}
		lock, err := s.tryLockGame(gameID, gameLockTimeout)
		if err != nil {
			return nil, err
		}
		if _, exists := s.games.Load(gameID); exists {
			lock.Release()
			continue
		}
		defer lock.Release()

		record := &gameRecord{
			game:     s.newGame(gameID, uint8(request.GetRows()), uint8(request.GetColumns())),
			maxTurns: request.GetMaxTurns(),
		}
		s.games.Store(gameID, record)
		created = true
		return &dotscordonv1.CreateGameResponse{
			Game: gameStateToProto(gameID, record.game, record.turn, record.maxTurns),
		}, nil
	}
}

// reserveGameSlot enforces the limit across simultaneous creations without a
// global map lock. Failed creations return their reservation in CreateGame.
func (s *Service) reserveGameSlot() bool {
	for {
		count := s.gameCount.Load()
		if s.maxGames > 0 && count >= int64(s.maxGames) {
			return false
		}
		if s.gameCount.CompareAndSwap(count, count+1) {
			return true
		}
	}
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
