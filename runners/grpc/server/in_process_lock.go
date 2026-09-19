package grpcserver

import (
	"sync"
	"sync/atomic"
)

// inProcessLock must not be copied after first use. Its zero value is unlocked.
type inProcessLock struct {
	mu       sync.Mutex
	acquired atomic.Bool
}

func (lock *inProcessLock) TryAcquireLock() bool {
	if !lock.mu.TryLock() {
		return false
	}
	lock.acquired.Store(true)
	return true
}

func (lock *inProcessLock) ReleaseLock() error {
	if !lock.acquired.Swap(false) {
		return ErrLockNotAcquired
	}
	lock.mu.Unlock()
	return nil
}

func (lock *inProcessLock) IsAcquired() bool {
	return lock.acquired.Load()
}

var _ Lock = (*inProcessLock)(nil)
