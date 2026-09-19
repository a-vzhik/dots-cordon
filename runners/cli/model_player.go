package main

import (
	pb "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/inference"
	"github.com/a-vzhik/dots-cordon/runners"
)

type moveSelector interface {
	Move(*pb.GameState) (*pb.Coordinate, error)
}
type modelPlayer struct{ model *inference.Model }

func startModelPlayer(options runners.GameOptions, player int) (*modelPlayer, inference.Metadata, error) {
	model, err := inference.LoadWeights(options.Weights[player], options.ONNXRuntime)
	if err != nil {
		return nil, inference.Metadata{}, err
	}
	info := model.Metadata()
	if err := model.ValidateBoardSize(int(options.BoardRows), int(options.BoardCols)); err != nil {
		model.Close()
		return nil, info, err
	}
	return &modelPlayer{model}, info, nil
}

func (p *modelPlayer) Move(game *pb.GameState) (*pb.Coordinate, error) {
	rows, columns := int(game.GetBoard().GetRows()), int(game.GetBoard().GetColumns())
	state, err := inference.EncodeState(game, rows, columns)
	if err != nil {
		return nil, err
	}
	scores, err := p.model.Infer(state, rows, columns)
	if err != nil {
		return nil, err
	}
	return inference.SelectMove(scores, game.GetBoard())
}

func (p *modelPlayer) Close() { p.model.Close() }
