package grpcserver

// Lock provides exclusive access to a session. Callers must successfully acquire
// it before accessing session state and release it when finished.
type Lock interface {
	// TryAcquireLock returns immediately, reporting whether acquisition succeeded.
	TryAcquireLock() bool
	ReleaseLock() error
	// IsAcquired reports lock state, not ownership by the calling goroutine.
	IsAcquired() bool
}
