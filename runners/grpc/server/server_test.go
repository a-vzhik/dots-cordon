package grpcserver

import (
	"context"
	"net"
	"sync/atomic"
	"testing"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

func TestRecorderFactoryRecordsServerGame(t *testing.T) {
	recorder := &countingRecorder{}
	client := newTestClientWithService(t, NewService(
		1,
		WithRecorderFactory(func(string) engine.Recorder {
			return recorder
		}),
	))

	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    2,
		Columns: 2,
	})
	require.NoError(t, err)
	_, err = client.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
		GameId:       created.GetGame().GetGameId(),
		ExpectedTurn: 0,
		Position:     &dotscordonv1.Coordinate{},
	})
	require.NoError(t, err)

	assert.Equal(t, int32(1), recorder.starts.Load())
	assert.Equal(t, int32(1), recorder.moves.Load())
}

func newTestClient(t *testing.T, maxGames int) dotscordonv1.GameServiceClient {
	t.Helper()
	return newTestClientWithService(t, NewService(maxGames))
}

func newTestClientWithService(
	t *testing.T,
	service dotscordonv1.GameServiceServer,
) dotscordonv1.GameServiceClient {
	t.Helper()

	listener := bufconn.Listen(1024 * 1024)
	server := grpc.NewServer()
	dotscordonv1.RegisterGameServiceServer(server, service)

	serveErrors := make(chan error, 1)
	go func() {
		serveErrors <- server.Serve(listener)
	}()

	connection, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return listener.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	require.NoError(t, err)

	t.Cleanup(func() {
		require.NoError(t, connection.Close())
		server.Stop()
		require.NoError(t, listener.Close())
		assert.NoError(t, <-serveErrors)
	})

	return dotscordonv1.NewGameServiceClient(connection)
}

type countingRecorder struct {
	starts atomic.Int32
	moves  atomic.Int32
}

func (recorder *countingRecorder) RecordStart(uint8, uint8) {
	recorder.starts.Add(1)
}

func (recorder *countingRecorder) RecordMove(
	engine.PlayerIndex,
	uint8,
	uint8,
	engine.MoveResult,
) {
	recorder.moves.Add(1)
}
