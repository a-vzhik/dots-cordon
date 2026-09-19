package grpcserver

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestInProcessLockLifecycle(t *testing.T) {
	var lock inProcessLock
	require.False(t, lock.IsAcquired())
	require.ErrorIs(t, lock.ReleaseLock(), ErrLockNotAcquired)

	for range 2 {
		require.True(t, lock.TryAcquireLock(0))
		require.True(t, lock.IsAcquired())
		require.False(t, lock.TryAcquireLock(0))
		require.True(t, lock.IsAcquired())
		require.NoError(t, lock.ReleaseLock())
		require.False(t, lock.IsAcquired())
		require.ErrorIs(t, lock.ReleaseLock(), ErrLockNotAcquired)
	}
}

func TestInProcessLockAllowsOnlyOneConcurrentAcquisition(t *testing.T) {
	var lock inProcessLock
	const contenders = 16
	start := make(chan struct{})
	results := make(chan bool, contenders)
	for range contenders {
		go func() {
			<-start
			results <- lock.TryAcquireLock(0)
		}()
	}
	close(start)

	acquired := 0
	for range contenders {
		if <-results {
			acquired++
		}
	}
	require.Equal(t, 1, acquired)
	require.NoError(t, lock.ReleaseLock())
	require.True(t, lock.TryAcquireLock(0))
	require.NoError(t, lock.ReleaseLock())
}

func TestInProcessLockTryAcquireWaitsForRelease(t *testing.T) {
	var lock inProcessLock
	require.True(t, lock.TryAcquireLock(0))
	require.True(t, lock.IsAcquired())
	require.False(t, lock.TryAcquireLock(0))
	held := true
	defer func() {
		if held {
			require.NoError(t, lock.ReleaseLock())
		}
	}()

	started := make(chan struct{})
	pending := startTestCall(func() (bool, error) {
		close(started)
		if !lock.TryAcquireLock(time.Second) {
			return false, nil
		}
		return lock.IsAcquired(), lock.ReleaseLock()
	})
	<-started
	requireCallPending(t, pending)
	require.NoError(t, lock.ReleaseLock())
	held = false

	result := <-pending
	require.NoError(t, result.err)
	require.True(t, result.response)
	require.False(t, lock.IsAcquired())
}

func TestInProcessLockTryAcquireTimesOut(t *testing.T) {
	var lock inProcessLock
	require.True(t, lock.TryAcquireLock(0))
	t.Cleanup(func() {
		if lock.IsAcquired() {
			require.NoError(t, lock.ReleaseLock())
		}
	})

	const timeout = 20 * time.Millisecond
	started := time.Now()
	require.False(t, lock.TryAcquireLock(timeout))
	require.GreaterOrEqual(t, time.Since(started), timeout)
	require.True(t, lock.IsAcquired(), "timeout must not release the existing lock")
	require.False(t, lock.TryAcquireLock(-time.Second))

	require.NoError(t, lock.ReleaseLock())
	require.True(t, lock.TryAcquireLock(timeout), "a timed-out attempt must not retain the lock")
	require.NoError(t, lock.ReleaseLock())
}
