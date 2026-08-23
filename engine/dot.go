package engine

type Dot struct {
	Col    uint8
	Row    uint8
	Owned  bool
	Owner  PlayerIndex
	Killed bool
}

func (source Dot) WithOwner(playerIndex PlayerIndex) Dot {
	result := source
	result.Owned = true
	result.Owner = playerIndex
	return result
}

func (source Dot) WithKilled() Dot {
	result := source
	result.Killed = true
	return result
}

func (dot Dot) IsOwnedBy(playerIndex PlayerIndex) bool {
	return dot.Owned && dot.Owner == playerIndex
}
