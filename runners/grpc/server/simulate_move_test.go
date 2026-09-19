package grpcserver

import (
	"context"
	"math/rand"
	"testing"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

func TestSimulationMatchesLivePlayAndLeavesGameUntouched(t *testing.T) {
	ctx := context.Background()
	service := NewService(1)
	rng := rand.New(rand.NewSource(17))
	for _, limit := range []uint32{0, 12} {
		created, err := service.CreateGame(ctx, &dotscordonv1.CreateGameRequest{Rows: 7, Columns: 7, MaxTurns: limit})
		require.NoError(t, err)
		game := created.Game
		// Force a capture, then exercise reconstruction with dead cells and scores.
		opening := []int{9, 16, 15, 22, 23, 30, 17}
		turn := 0
		for !game.Terminal {
			legal := []int{}
			for i, cell := range game.Board.Cells {
				if cell == 0 {
					legal = append(legal, i)
				}
			}
			action := legal[rng.Intn(len(legal))]
			if turn < len(opening) {
				action = opening[turn]
			}
			position := &dotscordonv1.Coordinate{Row: uint32(action / 7), Column: uint32(action % 7)}
			before := proto.Clone(game).(*dotscordonv1.GameState)
			simulated, err := service.SimulateMove(ctx, &dotscordonv1.SimulateMoveRequest{Game: game, Position: position, Player: proto.Uint32(game.NextTurnBy)})
			require.NoError(t, err)
			require.True(t, proto.Equal(before, game), "input snapshot was mutated")
			live, err := service.GetGame(ctx, &dotscordonv1.GetGameRequest{GameId: game.GameId})
			require.NoError(t, err)
			require.True(t, proto.Equal(before, live.Game), "live game was mutated")
			actual, err := service.MakeMove(ctx, &dotscordonv1.MakeMoveRequest{GameId: game.GameId, Player: proto.Uint32(game.NextTurnBy), Position: position})
			require.NoError(t, err)
			simulated.Game.GameId = actual.Game.GameId
			require.True(t, proto.Equal(actual, simulated), "simulation differs on turn %d", turn)
			game = actual.Game
			turn++
			if turn == 7 {
				require.Equal(t, uint32(1), game.Scores[0])
			}
		}
		_, err = service.DeleteGame(ctx, &dotscordonv1.DeleteGameRequest{GameId: game.GameId})
		require.NoError(t, err)
	}
}

func TestSimulationRejectsInvalidStates(t *testing.T) {
	ctx := context.Background()
	service := NewService(1)
	created, err := service.CreateGame(ctx, &dotscordonv1.CreateGameRequest{Rows: 3, Columns: 3})
	require.NoError(t, err)
	for _, change := range []func(*dotscordonv1.SimulateMoveRequest){
		func(r *dotscordonv1.SimulateMoveRequest) { r.Game = nil },
		func(r *dotscordonv1.SimulateMoveRequest) { r.Position = nil },
		func(r *dotscordonv1.SimulateMoveRequest) { r.Game.Board.Cells = []byte{0} },
		func(r *dotscordonv1.SimulateMoveRequest) { r.Game.Board.Cells[0] = 99 },
		func(r *dotscordonv1.SimulateMoveRequest) { r.Game.Scores = nil },
		func(r *dotscordonv1.SimulateMoveRequest) { r.Player = nil },
		func(r *dotscordonv1.SimulateMoveRequest) { r.Game.NextTurnBy = 2 },
		func(r *dotscordonv1.SimulateMoveRequest) { r.Position.Row = 3 },
	} {
		r := &dotscordonv1.SimulateMoveRequest{Player: proto.Uint32(0), Game: proto.Clone(created.Game).(*dotscordonv1.GameState), Position: &dotscordonv1.Coordinate{}}
		change(r)
		_, err = service.SimulateMove(ctx, r)
		require.Equal(t, codes.InvalidArgument, status.Code(err))
	}
	created.Game.Terminal = true
	_, err = service.SimulateMove(ctx, &dotscordonv1.SimulateMoveRequest{Player: proto.Uint32(0), Game: created.Game, Position: &dotscordonv1.Coordinate{}})
	require.Equal(t, codes.FailedPrecondition, status.Code(err))
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = service.SimulateMove(cancelled, nil)
	require.Equal(t, codes.Canceled, status.Code(err))
}

func TestSimulationUsesExplicitNextPlayer(t *testing.T) {
	service := NewService(1)
	state := &dotscordonv1.GameState{
		Board:  &dotscordonv1.Board{Rows: 2, Columns: 2, Cells: []byte{0, 0, 0, 0}},
		Scores: []uint32{0, 0}, NextTurnBy: 1,
	}
	response, err := service.SimulateMove(context.Background(), &dotscordonv1.SimulateMoveRequest{
		Game: state, Player: proto.Uint32(1), Position: &dotscordonv1.Coordinate{},
	})
	require.NoError(t, err)
	require.Equal(t, uint32(1), response.Result.Player)
	require.Equal(t, uint32(0), response.Game.NextTurnBy)
	require.Equal(t, []byte{2, 0, 0, 0}, response.Game.Board.Cells)
}

func TestMaxTurnsCountsCapturedOwnedDotsAndExcludesDeadEmptyCells(t *testing.T) {
	service := NewService(1)
	state := &dotscordonv1.GameState{
		Board:  &dotscordonv1.Board{Rows: 1, Columns: 5, Cells: []byte{3, 4, 5, 0, 0}},
		Scores: []uint32{1, 1}, NextTurnBy: 0, MaxTurns: 3,
	}
	request := &dotscordonv1.SimulateMoveRequest{
		Game: state, Player: proto.Uint32(0), Position: &dotscordonv1.Coordinate{Column: 3},
	}
	response, err := service.SimulateMove(context.Background(), request)
	require.NoError(t, err)
	require.True(t, response.Game.Terminal)
	require.Equal(t, uint32(3), response.Game.MaxTurns)
	require.Equal(t, dotscordonv1.TerminationReason_TERMINATION_REASON_TURN_LIMIT, response.Game.TerminationReason)
	state.MaxTurns = 2
	_, err = service.SimulateMove(context.Background(), request)
	require.Equal(t, codes.FailedPrecondition, status.Code(err))
}
