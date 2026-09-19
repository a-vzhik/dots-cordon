package grpcserver

import (
	"errors"

	"github.com/a-vzhik/dots-cordon/engine"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var (
	// ErrLockNotAcquired indicates an operation requires an acquired session lock.
	ErrLockNotAcquired = status.Error(codes.FailedPrecondition, "session lock must be acquired")
	// ErrSessionBusy indicates the session lock is already acquired.
	ErrSessionBusy = status.Error(codes.FailedPrecondition, "session lock is already acquired")
)

func gameNotFound(gameID string) error {
	return status.Errorf(codes.NotFound, "game %q not found", gameID)
}

func moveStatus(err error) error {
	switch {
	case errors.Is(err, engine.ErrInvalidMove), errors.Is(err, engine.ErrNonEmptyDot):
		return status.Error(codes.InvalidArgument, err.Error())
	case errors.Is(err, engine.ErrUnknownPlayer):
		return status.Error(codes.Internal, err.Error())
	default:
		return status.Error(codes.Internal, "apply move")
	}
}
