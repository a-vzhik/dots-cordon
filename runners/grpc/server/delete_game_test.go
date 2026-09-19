package grpcserver

import (
	"context"
	"testing"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestDeleteGameRemovesSession(t *testing.T) {
	client := newTestClient(t, 10)

	created, err := client.CreateGame(context.Background(), &dotscordonv1.CreateGameRequest{
		Rows:    2,
		Columns: 3,
	})
	require.NoError(t, err)
	gameID := created.GetGame().GetGameId()

	_, err = client.DeleteGame(context.Background(), &dotscordonv1.DeleteGameRequest{GameId: gameID})
	require.NoError(t, err)

	_, err = client.GetGame(context.Background(), &dotscordonv1.GetGameRequest{GameId: gameID})
	assert.Equal(t, codes.NotFound, status.Code(err))
}
