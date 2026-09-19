package grpcserver

import "time"

// Lock manages exclusive access by game ID, independently of game storage.
type Lock interface {
	// TryAcquireLock waits up to timeout. A nonpositive timeout makes a single
	// immediate attempt. The caller must release a successful acquisition.
	TryAcquireLock(gameID string, timeout time.Duration) (AcquiredLock, error)
}

// AcquiredLock represents one acquisition, not the game or its state.
type AcquiredLock interface {
	// Release removes this acquisition and wakes waiters. Releasing an already
	// released handle returns ErrLockNotAcquired without affecting a new holder.
	Release() error
}
