package session

import (
	"errors"
	"fmt"

	"github.com/a-vzhik/dots-cordon/engine"
)

type moveResult struct {
	coord engine.Coord
	err   error
}

// Controller belongs to the UI goroutine. Only its worker calls Opponent.Move;
// the worker receives detached snapshots and never accesses the live session.
// All public methods, including Close, must be called on the UI goroutine.
type Controller struct {
	session  *Session
	requests chan Snapshot
	results  chan moveResult
	done     chan struct{}
	thinking bool
	closed   bool
	err      error
}

func NewController(opponent Opponent) *Controller {
	c := &Controller{session: New(), requests: make(chan Snapshot, 1), results: make(chan moveResult, 1), done: make(chan struct{})}
	if opponent == nil {
		c.err = errors.New("opponent is required")
	}
	go func() {
		defer close(c.done)
		for snapshot := range c.requests {
			coord, err := opponent.Move(snapshot)
			c.results <- moveResult{coord, err}
		}
	}()
	return c
}

func (c *Controller) Snapshot() Snapshot { return c.session.Snapshot() }
func (c *Controller) Thinking() bool     { return c.thinking }
func (c *Controller) Err() error         { return c.err }

func (c *Controller) HumanMove(row, col uint8) error {
	if c.closed {
		return errors.New("controller is closed")
	}
	if c.err != nil {
		return c.err
	}
	snapshot := c.session.Snapshot()
	if c.thinking || snapshot.NextPlayer != 0 {
		return errors.New("it is not the human player's turn")
	}
	if snapshot.Terminal {
		return errors.New("game is over")
	}
	return c.session.Move(row, col)
}

// Update polls for completion and schedules at most one outstanding AI move.
func (c *Controller) Update() {
	if c.closed || c.err != nil {
		return
	}
	if c.thinking {
		select {
		case result := <-c.results:
			c.thinking = false
			if result.err != nil {
				c.err = fmt.Errorf("opponent move: %w", result.err)
				return
			}
			if err := c.session.Move(result.coord.Row, result.coord.Col); err != nil {
				c.err = fmt.Errorf("apply opponent move: %w", err)
				return
			}
		default:
			return
		}
	}
	snapshot := c.session.Snapshot()
	if !snapshot.Terminal && snapshot.NextPlayer == 1 {
		c.thinking = true
		c.requests <- snapshot
	}
}

// Close joins the worker before its model may be released. Native inference
// cannot be canceled, so this may wait for the current call to finish.
func (c *Controller) Close() {
	if c.closed {
		return
	}
	c.closed = true
	close(c.requests)
	<-c.done
	c.thinking = false
}
