package grpcserver

import (
	"sync"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
)

// Service stores independent game sessions in memory. It is safe for
// concurrent use; moves within one game are serialized while separate games
// can advance in parallel.
type Service struct {
	dotscordonv1.UnimplementedGameServiceServer

	mu              sync.RWMutex
	games           map[string]*gameSession
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
// concurrent-session limit.
func NewService(maxGames int, options ...Option) *Service {
	service := &Service{
		games:    make(map[string]*gameSession),
		maxGames: maxGames,
	}
	for _, option := range options {
		option(service)
	}
	return service
}

var _ dotscordonv1.GameServiceServer = (*Service)(nil)
