package grpcserver

import (
	"sync"
	"sync/atomic"
	"time"
)

// inProcessLock must not be copied after first use. Its zero value is unlocked.
type inProcessLock struct {
	mu       sync.Mutex
	acquired atomic.Bool
}

func (lock *inProcessLock) TryAcquireLock(timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	if lock.mu.TryLock() {
		lock.acquired.Store(true)
		return true
	}

	// Mutex acquisition cannot be canceled. Retry in this goroutine so an
	// expired attempt cannot acquire the lock later and leave it held.
	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return false
		}
		time.Sleep(min(remaining, time.Millisecond))
		if time.Now().Before(deadline) && lock.mu.TryLock() {
			lock.acquired.Store(true)
			return true
		}
	}
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
