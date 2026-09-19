package grpcserver

import (
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"
)

func TestInProcessLockLifecycle(t *testing.T) {
	var manager inProcessLock
	for range 2 {
		lock, err := manager.TryAcquireLock("game", 0)
		require.NoError(t, err)
		blocked, err := manager.TryAcquireLock("game", 0)
		require.Nil(t, blocked)
		require.ErrorIs(t, err, ErrLockTimeout)
		require.NoError(t, lock.Release())
		require.ErrorIs(t, lock.Release(), ErrLockNotAcquired)
		requireLockMapEmpty(t, &manager)
	}
}

func TestInProcessLockDifferentGamesAreIndependent(t *testing.T) {
	var manager inProcessLock
	first, err := manager.TryAcquireLock("first", 0)
	require.NoError(t, err)
	defer first.Release()
	second, err := manager.TryAcquireLock("second", 0)
	require.NoError(t, err)
	require.NoError(t, second.Release())
	blocked, err := manager.TryAcquireLock("first", -time.Second)
	require.Nil(t, blocked)
	require.ErrorIs(t, err, ErrLockTimeout)
}

func TestInProcessLockStaleReleaseCannotUnlockNewHolder(t *testing.T) {
	var manager inProcessLock
	old, err := manager.TryAcquireLock("game", 0)
	require.NoError(t, err)
	require.NoError(t, old.Release())
	current, err := manager.TryAcquireLock("game", 0)
	require.NoError(t, err)
	defer current.Release()

	require.ErrorIs(t, old.Release(), ErrLockNotAcquired)
	blocked, err := manager.TryAcquireLock("game", 0)
	require.Nil(t, blocked)
	require.ErrorIs(t, err, ErrLockTimeout)
}

func TestInProcessLockConcurrentReleaseSucceedsOnce(t *testing.T) {
	var manager inProcessLock
	lock, err := manager.TryAcquireLock("game", 0)
	require.NoError(t, err)
	results := make(chan error, 16)
	for range cap(results) {
		go func() { results <- lock.Release() }()
	}
	released := 0
	for range cap(results) {
		if err := <-results; err == nil {
			released++
		} else {
			require.ErrorIs(t, err, ErrLockNotAcquired)
		}
	}
	require.Equal(t, 1, released)
	requireLockMapEmpty(t, &manager)
}

func TestInProcessLockWaitersAcquireSeriallyAfterRelease(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var manager inProcessLock
		owner, err := manager.TryAcquireLock("game", 0)
		require.NoError(t, err)
		defer owner.Release()
		const waiters = 16
		results := make(chan error, waiters)
		moves := 0 // Deliberately protected only by the keyed lock.
		for range waiters {
			go func() {
				lock, err := manager.TryAcquireLock("game", time.Second)
				if err == nil {
					moves++
					err = lock.Release()
				}
				results <- err
			}()
		}
		synctest.Wait()
		require.Empty(t, results, "waiters must not enter while the owner holds the lock")
		require.NoError(t, owner.Release())
		for range waiters {
			require.NoError(t, <-results)
		}
		require.Equal(t, waiters, moves)
		requireLockMapEmpty(t, &manager)
	})
}

func TestInProcessLockTimeoutPreservesHolderAndDoesNotLeak(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var manager inProcessLock
		owner, err := manager.TryAcquireLock("game", 0)
		require.NoError(t, err)
		defer owner.Release()
		const timeout = 20 * time.Millisecond
		started := time.Now()
		blocked, err := manager.TryAcquireLock("game", timeout)
		require.Nil(t, blocked)
		require.ErrorIs(t, err, ErrLockTimeout)
		require.Equal(t, timeout, time.Since(started))
		blocked, err = manager.TryAcquireLock("game", 0)
		require.Nil(t, blocked)
		require.ErrorIs(t, err, ErrLockTimeout)

		require.NoError(t, owner.Release())
		requireLockMapEmpty(t, &manager)
		next, err := manager.TryAcquireLock("game", 0)
		require.NoError(t, err)
		require.NoError(t, next.Release())
		requireLockMapEmpty(t, &manager)
	})
}

func requireLockMapEmpty(t *testing.T, manager *inProcessLock) {
	t.Helper()
	manager.locks.Range(func(key, value any) bool {
		t.Errorf("lock entry for %v was not cleaned up", key)
		return true
	})
}
