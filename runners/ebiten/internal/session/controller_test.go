package session

import (
	"errors"
	"runtime"
	"testing"
	"time"

	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/stretchr/testify/require"
)

type opponentFunc func(Snapshot) (engine.Coord, error)

func (f opponentFunc) Move(s Snapshot) (engine.Coord, error) { return f(s) }

func poll(t *testing.T, c *Controller) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for c.Thinking() && time.Now().Before(deadline) {
		c.Update()
		runtime.Gosched()
	}
	require.False(t, c.Thinking(), "worker did not finish")
}

func TestControllerOneRequestAndUIAppliesResult(t *testing.T) {
	started := make(chan Snapshot, 2)
	release := make(chan struct{})
	c := NewController(opponentFunc(func(s Snapshot) (engine.Coord, error) {
		started <- s
		<-release
		return engine.Coord{Row: 0, Col: 1}, nil
	}))
	defer c.Close()
	defer close(release)
	c.Update()
	require.False(t, c.Thinking())
	require.NoError(t, c.HumanMove(0, 0))
	c.Update()
	s := <-started
	require.Equal(t, engine.PlayerIndex(1), s.NextPlayer)
	for range 100 {
		c.Update()
	}
	require.True(t, c.Thinking())
	require.Error(t, c.HumanMove(0, 2))
	require.Len(t, started, 0)
	// A detached worker snapshot cannot alter the live session.
	s.Dots[0][0].Owned = false
	require.True(t, c.Snapshot().Dots[0][0].Owned)
	release <- struct{}{}
	poll(t, c)
	require.NoError(t, c.Err())
	require.Equal(t, engine.PlayerIndex(0), c.Snapshot().NextPlayer)
	require.Equal(t, engine.PlayerIndex(1), c.Snapshot().Dots[0][1].Owner)
	require.Error(t, c.HumanMove(0, 0))
	require.Equal(t, engine.PlayerIndex(0), c.Snapshot().NextPlayer)
}

func TestControllerErrorsStopPlay(t *testing.T) {
	for _, test := range []struct {
		name string
		move engine.Coord
		err  error
	}{
		{name: "inference", err: errors.New("inference failed")},
		{name: "occupied", move: engine.Coord{Row: 0, Col: 0}},
		{name: "out of bounds", move: engine.Coord{Row: 255, Col: 255}},
	} {
		t.Run(test.name, func(t *testing.T) {
			calls := make(chan struct{}, 2)
			c := NewController(opponentFunc(func(Snapshot) (engine.Coord, error) { calls <- struct{}{}; return test.move, test.err }))
			defer c.Close()
			require.NoError(t, c.HumanMove(0, 0))
			c.Update()
			poll(t, c)
			require.Error(t, c.Err())
			if test.err != nil {
				require.ErrorIs(t, c.Err(), test.err)
			}
			require.Error(t, c.HumanMove(0, 2))
			for range 10 {
				c.Update()
			}
			require.Len(t, calls, 1)
			require.Equal(t, engine.PlayerIndex(1), c.Snapshot().NextPlayer)
		})
	}
}

func TestControllerCloseJoinsWorker(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	finished := make(chan struct{})
	c := NewController(opponentFunc(func(Snapshot) (engine.Coord, error) {
		close(started)
		<-release
		close(finished)
		return engine.Coord{Row: 0, Col: 1}, nil
	}))
	require.NoError(t, c.HumanMove(0, 0))
	c.Update()
	<-started
	closed := make(chan struct{})
	go func() { c.Close(); close(closed) }()
	select {
	case <-closed:
		t.Fatal("Close returned during inference")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("Close blocked on undrained result")
	}
	<-finished
	c.Close()
	c.Update()
	require.False(t, c.Thinking())
	require.Error(t, c.HumanMove(0, 2))
	require.Equal(t, engine.PlayerIndex(1), c.Snapshot().NextPlayer)
}

func TestControllerNilOpponent(t *testing.T) {
	c := NewController(nil)
	defer c.Close()
	require.Error(t, c.Err())
	require.Error(t, c.HumanMove(0, 0))
	c.Update()
	require.False(t, c.Thinking())
}
