package grpcserver

import (
	"sync"
	"sync/atomic"
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

	lock            Lock
	games           sync.Map     // game ID -> *dotscordonv1.GameState
	recorders       sync.Map     // game ID -> engine.Recorder
	gameCount       atomic.Int64 // includes reserved creation slots
	maxGames        int
	recorderFactory RecorderFactory
}

var _ dotscordonv1.GameServiceServer = (*Service)(nil)

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
		lock:     &inProcessLock{},
		maxGames: maxGames,
	}
	for _, option := range options {
		option(service)
	}
	return service
}

// tryLockGame acquires exclusive access to an ID without reading game state.
func (s *Service) tryLockGame(gameID string, timeout time.Duration) (AcquiredLock, error) {
	if gameID == "" {
		return nil, status.Error(codes.InvalidArgument, "game_id is required")
	}
	return s.lock.TryAcquireLock(gameID, timeout)
}

// getGame must be called while holding the lock for gameID. The lock must remain
// held while the returned state is accessed, including response conversion.
func (s *Service) getGame(gameID string) (*dotscordonv1.GameState, error) {
	state, ok := s.games.Load(gameID)
	if !ok {
		return nil, gameNotFound(gameID)
	}
	return state.(*dotscordonv1.GameState), nil
}
