package grpcserver

import (
	"context"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

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

func TestTryLockGameReturnsErrorOnTimeout(t *testing.T) {
	service := NewService(1)
	created, err := service.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{Rows: 2, Columns: 2})
	require.NoError(t, err)
	lock, err := service.tryLockGame(created.Game.GameId, 0)
	require.NoError(t, err)
	defer lock.Release()

	blocked, err := service.tryLockGame(created.Game.GameId, 20*time.Millisecond)
	require.Nil(t, blocked)
	require.ErrorIs(t, err, ErrLockTimeout)
	record, err := service.getGame(created.Game.GameId)
	require.NoError(t, err)
	require.Zero(t, record.turn)

	require.NoError(t, lock.Release())
	retried, err := service.tryLockGame(created.Game.GameId, time.Second)
	require.NoError(t, err)
	require.NoError(t, retried.Release())
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

// newTestBlockedGame holds the first move inside its recorder, after the engine
// changes the board but before the server updates the turn and returns a result.
func newTestBlockedGame(t *testing.T) (*Service, string, func() *dotscordonv1.MakeMoveResponse) {
	t.Helper()
	recorder := &blockingRecorder{started: make(chan struct{}), release: make(chan struct{})}
	var recorderUsed atomic.Bool
	service := NewService(2, WithRecorderFactory(func(string) engine.Recorder {
		if recorderUsed.CompareAndSwap(false, true) {
			return recorder
		}
		return nil
	}))
	created, err := service.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{Rows: 2, Columns: 2})
	require.NoError(t, err)
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(recorder.release) }) }
	t.Cleanup(release)
	pending := startTestCall(func() (*dotscordonv1.MakeMoveResponse, error) {
		return service.MakeMove(context.Background(), &dotscordonv1.MakeMoveRequest{
			GameId: created.Game.GameId, Position: &dotscordonv1.Coordinate{},
		})
	})
	select {
	case <-recorder.started:
	case <-time.After(time.Second):
		t.Fatal("move did not reach the recorder")
	}
	return service, created.Game.GameId, func() *dotscordonv1.MakeMoveResponse {
		release()
		result := <-pending
		require.NoError(t, result.err)
		return result.response
	}
}

type blockingRecorder struct {
	once    sync.Once
	started chan struct{}
	release chan struct{}
}

func (*blockingRecorder) RecordStart(uint8, uint8) {}

func (recorder *blockingRecorder) RecordMove(engine.PlayerIndex, uint8, uint8, engine.MoveResult) {
	recorder.once.Do(func() {
		close(recorder.started)
		<-recorder.release
	})
}

type testCallResult[T any] struct {
	response T
	err      error
}

func startTestCall[T any](call func() (T, error)) <-chan testCallResult[T] {
	pending := make(chan testCallResult[T], 1)
	go func() {
		response, err := call()
		pending <- testCallResult[T]{response: response, err: err}
	}()
	return pending
}

func requireCallPending[T any](t *testing.T, pending <-chan testCallResult[T]) {
	t.Helper()
	select {
	case result := <-pending:
		t.Fatalf("call returned before the in-flight move completed: %v", result.err)
	case <-time.After(30 * time.Millisecond):
	}
}
