package grpcserver

import (
	"errors"

	"github.com/a-vzhik/dots-cordon/engine"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ErrLockNotAcquired indicates an operation requires an acquired game lock.
var ErrLockNotAcquired = status.Error(codes.FailedPrecondition, "game lock must be acquired")

// ErrLockTimeout indicates the game lock could not be acquired within the timeout.
var ErrLockTimeout = status.Error(codes.DeadlineExceeded, "timed out waiting for game lock")

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
