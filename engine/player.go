package engine

type PlayerIndex uint8

type PlayerColor int

const (
	RedColor PlayerColor = iota
	BlueColor
)

type Player struct {
	Score uint32
	Color PlayerColor
}
