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

func TestSimulationMatchesLivePlayAndLeavesSessionUntouched(t *testing.T) {
	ctx := context.Background()
	service := NewService(1)
	rng := rand.New(rand.NewSource(17))
	for _, limit := range []uint32{0, 12} {
		created, err := service.CreateGame(ctx, &dotscordonv1.CreateGameRequest{Rows: 7, Columns: 7, MaxTurns: limit})
		require.NoError(t, err)
		game := created.Game
		// Force a capture, then exercise reconstruction with dead cells and scores.
		opening := []int{9, 16, 15, 22, 23, 30, 17}
		for !game.Terminal {
			legal := []int{}
			for i, cell := range game.Board.Cells {
				if cell == 0 {
					legal = append(legal, i)
				}
			}
			action := legal[rng.Intn(len(legal))]
			if int(game.Turn) < len(opening) {
				action = opening[game.Turn]
			}
			position := &dotscordonv1.Coordinate{Row: uint32(action / 7), Column: uint32(action % 7)}
			before := proto.Clone(game).(*dotscordonv1.GameState)
			simulated, err := service.SimulateMove(ctx, &dotscordonv1.SimulateMoveRequest{Game: game, Position: position, MaxTurns: limit})
			require.NoError(t, err)
			require.True(t, proto.Equal(before, game), "input snapshot was mutated")
			live, err := service.GetGame(ctx, &dotscordonv1.GetGameRequest{GameId: game.GameId})
			require.NoError(t, err)
			require.True(t, proto.Equal(before, live.Game), "live game was mutated")
			actual, err := service.MakeMove(ctx, &dotscordonv1.MakeMoveRequest{GameId: game.GameId, ExpectedTurn: game.Turn, Position: position})
			require.NoError(t, err)
			simulated.Game.GameId = actual.Game.GameId
			require.True(t, proto.Equal(actual, simulated), "simulation differs on turn %d", game.Turn)
			game = actual.Game
			if game.Turn == 7 {
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
		func(r *dotscordonv1.SimulateMoveRequest) { r.Game.Turn = 2 },
		func(r *dotscordonv1.SimulateMoveRequest) { r.Game.CurrentPlayer = 1 },
		func(r *dotscordonv1.SimulateMoveRequest) { r.Position.Row = 3 },
	} {
		r := &dotscordonv1.SimulateMoveRequest{Game: proto.Clone(created.Game).(*dotscordonv1.GameState), Position: &dotscordonv1.Coordinate{}}
		change(r)
		_, err = service.SimulateMove(ctx, r)
		require.Equal(t, codes.InvalidArgument, status.Code(err))
	}
	created.Game.Terminal = true
	_, err = service.SimulateMove(ctx, &dotscordonv1.SimulateMoveRequest{Game: created.Game, Position: &dotscordonv1.Coordinate{}})
	require.Equal(t, codes.FailedPrecondition, status.Code(err))
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = service.SimulateMove(cancelled, nil)
	require.Equal(t, codes.Canceled, status.Code(err))
}
