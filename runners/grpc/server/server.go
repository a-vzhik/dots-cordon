package grpcserver

import (
	"sync"
	"time"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const gameLockTimeout = 5 * time.Second

// Service owns game lifecycle and serializes requests for each game.
// Requests for different games can execute concurrently. Requests waiting for
// the same game return ErrLockTimeout if they cannot acquire its lock in time.
type Service struct {
	dotscordonv1.UnimplementedGameServiceServer

	mu              sync.RWMutex
	games           map[string]*gameRecord
	maxGames        int
	recorderFactory RecorderFactory
}

// RecorderFactory creates an engine recorder for a game episode. The default
// service uses NoopGameRecorder.
type RecorderFactory func(gameID string) engine.Recorder

// Option configures a Service.
type Option func(*Service)

// WithRecorderFactory records each newly created or reset game with the
// recorder returned by factory. Returning nil disables recording for that
// episode.
func WithRecorderFactory(factory RecorderFactory) Option {
	return func(service *Service) {
		service.recorderFactory = factory
	}
}

// NewService creates a stateful game service. maxGames <= 0 disables the
// concurrent-game limit.
func NewService(maxGames int, options ...Option) *Service {
	service := &Service{
		games:    make(map[string]*gameRecord),
		maxGames: maxGames,
	}
	for _, option := range options {
		option(service)
	}
	return service
}

var _ dotscordonv1.GameServiceServer = (*Service)(nil)

// gameRecord is server storage, with no gameplay or lifecycle methods.
// lock protects the game and its metadata; Service.mu protects the games map.
type gameRecord struct {
	lock     Lock
	game     *engine.Game
	turn     uint32
	maxTurns uint32
	deleted  bool
}

// tryLockGame waits up to timeout for exclusive access to a live game. On
// success, the caller must release record.lock after response conversion.
func (s *Service) tryLockGame(gameID string, timeout time.Duration) (*gameRecord, error) {
	if gameID == "" {
		return nil, status.Error(codes.InvalidArgument, "game_id is required")
	}
	s.mu.RLock()
	record := s.games[gameID]
	s.mu.RUnlock()
	if record == nil {
		return nil, gameNotFound(gameID)
	}

	// Never hold the map lock while waiting for an individual game.
	if !record.lock.TryAcquireLock(timeout) {
		return nil, ErrLockTimeout
	}
	if record.deleted {
		record.lock.ReleaseLock()
		return nil, gameNotFound(gameID)
	}
	return record, nil
}
