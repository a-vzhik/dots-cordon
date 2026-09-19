package grpcserver

import "time"

// Lock provides exclusive access to a game. Callers must successfully acquire
// it before accessing game state and release it when finished.
type Lock interface {
	// TryAcquireLock waits up to timeout for exclusive access. A nonpositive
	// timeout makes a single immediate attempt.
	TryAcquireLock(timeout time.Duration) bool
	ReleaseLock() error
	// IsAcquired reports lock state, not ownership by the calling goroutine.
	IsAcquired() bool
}
