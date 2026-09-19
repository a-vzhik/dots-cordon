package grpcserver

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestInProcessLockLifecycle(t *testing.T) {
	var lock inProcessLock
	require.False(t, lock.IsAcquired())
	require.ErrorIs(t, lock.ReleaseLock(), ErrLockNotAcquired)

	for range 2 {
		require.True(t, lock.TryAcquireLock())
		require.True(t, lock.IsAcquired())
		require.False(t, lock.TryAcquireLock())
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
			results <- lock.TryAcquireLock()
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
	require.True(t, lock.TryAcquireLock())
	require.NoError(t, lock.ReleaseLock())
}
