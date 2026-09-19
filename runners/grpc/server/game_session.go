package grpcserver

import (
	"sync"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type gameSession struct {
	mu sync.Mutex

	id              string
	rows            uint8
	columns         uint8
	maxTurns        uint32
	game            *engine.Game
	turn            uint32
	current         engine.PlayerIndex
	terminal        bool
	termination     dotscordonv1.TerminationReason
	deleted         bool
	recorderFactory RecorderFactory
}

func newGameSession(
	id string,
	rows uint8,
	columns uint8,
	maxTurns uint32,
	recorder RecorderFactory,
) *gameSession {
	session := &gameSession{
		id:              id,
		rows:            rows,
		columns:         columns,
		maxTurns:        maxTurns,
		recorderFactory: recorder,
	}
	session.resetLocked()
	return session
}

func (s *Service) getSession(gameID string) (*gameSession, error) {
	if gameID == "" {
		return nil, status.Error(codes.InvalidArgument, "game_id is required")
	}

	s.mu.RLock()
	session := s.games[gameID]
	s.mu.RUnlock()
	if session == nil {
		return nil, gameNotFound(gameID)
	}
	return session, nil
}

// moveLocked is shared by live play and isolated search simulations.
func (session *gameSession) moveLocked(position *dotscordonv1.Coordinate) (*dotscordonv1.MakeMoveResponse, error) {
	if position.GetRow() >= uint32(session.rows) || position.GetColumn() >= uint32(session.columns) {
		return nil, status.Errorf(
			codes.InvalidArgument,
			"position (%d, %d) is outside the %dx%d board",
			position.GetRow(),
			position.GetColumn(),
			session.rows,
			session.columns,
		)
	}

	player := session.current
	result, moveErr := session.game.Move(
		player,
		uint8(position.GetRow()),
		uint8(position.GetColumn()),
	)
	if moveErr != nil {
		return nil, moveStatus(moveErr)
	}

	session.turn++
	session.current = session.current.EnemyIndex()
	switch {
	case result.IsTerminal:
		session.terminal = true
		session.termination = dotscordonv1.TerminationReason_TERMINATION_REASON_BOARD_FULL
	case session.maxTurns > 0 && session.turn >= session.maxTurns:
		session.terminal = true
		session.termination = dotscordonv1.TerminationReason_TERMINATION_REASON_TURN_LIMIT
	}

	return &dotscordonv1.MakeMoveResponse{
		Game:   session.snapshotLocked(),
		Result: moveResultToProto(player, result),
	}, nil
}

func (session *gameSession) resetLocked() {
	recorder := engine.Recorder(engine.NoopGameRecorder{})
	if session.recorderFactory != nil {
		if configured := session.recorderFactory(session.id); configured != nil {
			recorder = configured
		}
	}

	session.game = engine.NewGame(
		engine.NewGameField(session.columns, session.rows),
		[]*engine.Player{
			{Color: engine.BlueColor},
			{Color: engine.RedColor},
		},
		recorder,
	)
	session.turn = 0
	session.current = 0
	session.terminal = false
	session.termination = dotscordonv1.TerminationReason_TERMINATION_REASON_UNSPECIFIED
}

func (session *gameSession) snapshotLocked() *dotscordonv1.GameState {
	field := session.game.GameField
	cells := make([]byte, 0, int(field.Width)*int(field.Height))
	for _, row := range field.Dots {
		for _, dot := range row {
			cells = append(cells, byte(cellToProto(dot)))
		}
	}

	return &dotscordonv1.GameState{
		GameId: session.id,
		Board: &dotscordonv1.Board{
			Rows:    uint32(field.Height),
			Columns: uint32(field.Width),
			Cells:   cells,
		},
		Scores: []uint32{
			session.game.Players[0].Score,
			session.game.Players[1].Score,
		},
		CurrentPlayer:     uint32(session.current),
		Turn:              session.turn,
		Terminal:          session.terminal,
		TerminationReason: session.termination,
	}
}
