package main

import (
	"errors"
	"net"
	"sync"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// embeddedGameServer is a private gRPC server owned by one CLI process.
type embeddedGameServer struct {
	client dotscordonv1.GameServiceClient

	connection  *grpc.ClientConn
	server      *grpc.Server
	serveErrors <-chan error
	closeOnce   sync.Once
	closeErr    error
}

func startEmbeddedGameServer(
	gameService dotscordonv1.GameServiceServer,
) (*embeddedGameServer, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}

	return startGameServerOnListener(
		listener,
		"passthrough:///"+listener.Addr().String(),
		gameService,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
}

func startGameServerOnListener(
	listener net.Listener,
	target string,
	gameService dotscordonv1.GameServiceServer,
	dialOptions ...grpc.DialOption,
) (*embeddedGameServer, error) {
	server := grpc.NewServer()
	dotscordonv1.RegisterGameServiceServer(server, gameService)

	serveErrors := make(chan error, 1)
	go func() {
		serveErrors <- server.Serve(listener)
	}()

	connection, err := grpc.NewClient(target, dialOptions...)
	if err != nil {
		server.Stop()
		_ = listener.Close()
		<-serveErrors
		return nil, err
	}

	return &embeddedGameServer{
		client:      dotscordonv1.NewGameServiceClient(connection),
		connection:  connection,
		server:      server,
		serveErrors: serveErrors,
	}, nil
}

func (server *embeddedGameServer) Close() error {
	server.closeOnce.Do(func() {
		server.server.GracefulStop()
		connectionErr := server.connection.Close()
		serveErr := <-server.serveErrors
		if errors.Is(serveErr, grpc.ErrServerStopped) {
			serveErr = nil
		}
		server.closeErr = errors.Join(connectionErr, serveErr)
	})
	return server.closeErr
}
