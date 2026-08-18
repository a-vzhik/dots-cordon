package main

import (
	"log"

	"github.com/a-vzhik/dots-cordon/engine"
)

func main() {
	players := []*engine.Player{
		{
			Score: 0,
			Color: engine.BlueColor,
		},
		{
			Score: 0,
			Color: engine.RedColor,
		},
	}
	game := engine.NewGame(
		engine.NewGameField(20, 20),
		players)

	err1 := game.Move(0, 10, 9)
	err2 := game.Move(1, 10, 10)

	log.Printf("The program running: %v %v", err1, err2)
}
