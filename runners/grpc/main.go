package main

import (
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	grpcserver "github.com/a-vzhik/dots-cordon/runners/grpc/server"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

func main() {
	listenAddress := flag.String("listen", "127.0.0.1:50051", "TCP address to listen on")
	maxGames := flag.Int("max-games", 10000, "maximum concurrent games; zero disables the limit")
	quiet := flag.Bool("quiet", false, "suppress informational engine logs during search/training")
	flag.Parse()
	if *quiet {
		slog.SetLogLoggerLevel(slog.LevelWarn)
	}

	if *maxGames < 0 {
		fmt.Fprintln(os.Stderr, "--max-games must be non-negative")
		os.Exit(2)
	}

	listener, err := net.Listen("tcp", *listenAddress)
	if err != nil {
		slog.Error("listen", "address", *listenAddress, "error", err)
		os.Exit(1)
	}

	server := grpc.NewServer()
	dotscordonv1.RegisterGameServiceServer(server, grpcserver.NewService(*maxGames))

	healthServer := health.NewServer()
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
	healthServer.SetServingStatus(
		dotscordonv1.GameService_ServiceDesc.ServiceName,
		grpc_health_v1.HealthCheckResponse_SERVING,
	)
	grpc_health_v1.RegisterHealthServer(server, healthServer)
	reflection.Register(server)

	stopSignals := make(chan os.Signal, 1)
	signal.Notify(stopSignals, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stopSignals
		healthServer.Shutdown()
		server.GracefulStop()
	}()

	fmt.Fprintf(os.Stderr, "gRPC game server listening address=%s\n", listener.Addr())
	if err := server.Serve(listener); err != nil {
		slog.Error("serve", "error", err)
		os.Exit(1)
	}
}
