package engine

import (
	"errors"
)

var (
	ErrUnknownPlayer = errors.New("unknown player index")
	ErrInvalidMove   = errors.New("move outside the game field")
	ErrNonEmptyDot   = errors.New("the cell is not empty")
)
