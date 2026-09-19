package grpcserver

import (
	"sync"
	"time"
)

// inProcessLock must not be copied after first use. Its zero value is ready to use.
type inProcessLock struct {
	locks sync.Map // game ID -> *lockEntry; contains only active acquisitions
}

type lockEntry struct {
	released chan struct{}
}

type acquiredLock struct {
	manager *inProcessLock
	gameID  string
	entry   *lockEntry
}

func (manager *inProcessLock) TryAcquireLock(gameID string, timeout time.Duration) (AcquiredLock, error) {
	deadline := time.Now().Add(timeout)
	entry := &lockEntry{released: make(chan struct{})}
	actual, loaded := manager.locks.LoadOrStore(gameID, entry)
	if !loaded {
		return &acquiredLock{manager: manager, gameID: gameID, entry: entry}, nil
	}
	if timeout <= 0 {
		return nil, ErrLockTimeout
	}

	timer := time.NewTimer(time.Until(deadline))
	defer timer.Stop()
	for {
		select {
		case <-actual.(*lockEntry).released:
			// All waiters retry against the map; no waiter can acquire a
			// detached entry. Keep the original deadline across retries.
			if !time.Now().Before(deadline) {
				return nil, ErrLockTimeout
			}
			actual, loaded = manager.locks.LoadOrStore(gameID, entry)
			if !loaded {
				return &acquiredLock{manager: manager, gameID: gameID, entry: entry}, nil
			}
		case <-timer.C:
			return nil, ErrLockTimeout
		}
	}
}

func (lock *acquiredLock) Release() error {
	if !lock.manager.locks.CompareAndDelete(lock.gameID, lock.entry) {
		return ErrLockNotAcquired
	}
	close(lock.entry.released)
	return nil
}

var _ Lock = (*inProcessLock)(nil)
var _ AcquiredLock = (*acquiredLock)(nil)
