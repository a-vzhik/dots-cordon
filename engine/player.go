package engine

type PlayerIndex uint8

func (p PlayerIndex) EnemyIndex() PlayerIndex {
	if p == PlayerIndex(0) {
		return PlayerIndex(1)
	} else {
		return PlayerIndex(0)
	}
}

type PlayerColor int

const (
	RedColor PlayerColor = iota
	BlueColor
)

type PlayerType int

const (
	Human PlayerType = iota
	RandomAI
	TrainedAI
)

type Player struct {
	Score      uint32
	Color      PlayerColor
	PlayerType PlayerType
}
