package grpcserver

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"sync"

	dotscordonv1 "github.com/a-vzhik/dots-cordon/api/dotscordon/v1"
	"github.com/a-vzhik/dots-cordon/engine"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Service stores independent game sessions in memory. It is safe for
// concurrent use; moves within one game are serialized while separate games
// can advance in parallel.
type Service struct {
	dotscordonv1.UnimplementedGameServiceServer

	mu              sync.RWMutex
	games           map[string]*gameSession
	maxGames        int
	recorderFactory RecorderFactory
}

// RecorderFactory creates an engine recorder for a game episode. The default
// service uses NoopGameRecorder.
type RecorderFactory func(gameID string) engine.Recorder

// Option configures a Service.
type Option func(*Service)

// WithRecorderFactory records each newly created or reset game with the
// recorder returned by factory. Returning nil disables recording for that
// episode.
func WithRecorderFactory(factory RecorderFactory) Option {
	return func(service *Service) {
		service.recorderFactory = factory
	}
}

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

// NewService creates a stateful game service. maxGames <= 0 disables the
// concurrent-session limit.
func NewService(maxGames int, options ...Option) *Service {
	service := &Service{
		games:    make(map[string]*gameSession),
		maxGames: maxGames,
	}
	for _, option := range options {
		option(service)
	}
	return service
}

func (s *Service) CreateGame(
	_ context.Context,
	request *dotscordonv1.CreateGameRequest,
) (*dotscordonv1.CreateGameResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}
	if err := validateDimensions(request.GetRows(), request.GetColumns()); err != nil {
		return nil, err
	}

	gameID, err := randomGameID()
	if err != nil {
		return nil, status.Errorf(codes.Internal, "generate game ID: %v", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.maxGames > 0 && len(s.games) >= s.maxGames {
		return nil, status.Errorf(
			codes.ResourceExhausted,
			"concurrent game limit of %d reached",
			s.maxGames,
		)
	}

	for s.games[gameID] != nil {
		gameID, err = randomGameID()
		if err != nil {
			return nil, status.Errorf(codes.Internal, "generate game ID: %v", err)
		}
	}

	session := newGameSession(
		gameID,
		uint8(request.GetRows()),
		uint8(request.GetColumns()),
		request.GetMaxTurns(),
		s.recorderFactory,
	)
	s.games[gameID] = session

	return &dotscordonv1.CreateGameResponse{Game: session.snapshotLocked()}, nil
}

func (s *Service) GetGame(
	_ context.Context,
	request *dotscordonv1.GetGameRequest,
) (*dotscordonv1.GetGameResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}

	session, err := s.getSession(request.GetGameId())
	if err != nil {
		return nil, err
	}

	session.mu.Lock()
	defer session.mu.Unlock()
	if session.deleted {
		return nil, gameNotFound(request.GetGameId())
	}

	return &dotscordonv1.GetGameResponse{Game: session.snapshotLocked()}, nil
}

func (s *Service) MakeMove(
	_ context.Context,
	request *dotscordonv1.MakeMoveRequest,
) (*dotscordonv1.MakeMoveResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}
	if request.GetPosition() == nil {
		return nil, status.Error(codes.InvalidArgument, "position is required")
	}

	session, err := s.getSession(request.GetGameId())
	if err != nil {
		return nil, err
	}

	session.mu.Lock()
	defer session.mu.Unlock()
	if session.deleted {
		return nil, gameNotFound(request.GetGameId())
	}
	if session.terminal {
		return nil, status.Error(codes.FailedPrecondition, "game is terminal; reset it before moving")
	}
	if request.GetExpectedTurn() != session.turn {
		return nil, status.Errorf(
			codes.FailedPrecondition,
			"expected turn %d, current turn is %d",
			request.GetExpectedTurn(),
			session.turn,
		)
	}

	position := request.GetPosition()
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

func (s *Service) ResetGame(
	_ context.Context,
	request *dotscordonv1.ResetGameRequest,
) (*dotscordonv1.ResetGameResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}

	session, err := s.getSession(request.GetGameId())
	if err != nil {
		return nil, err
	}

	session.mu.Lock()
	defer session.mu.Unlock()
	if session.deleted {
		return nil, gameNotFound(request.GetGameId())
	}

	session.resetLocked()
	return &dotscordonv1.ResetGameResponse{Game: session.snapshotLocked()}, nil
}

func (s *Service) DeleteGame(
	_ context.Context,
	request *dotscordonv1.DeleteGameRequest,
) (*dotscordonv1.DeleteGameResponse, error) {
	if request == nil {
		return nil, status.Error(codes.InvalidArgument, "request is required")
	}
	if request.GetGameId() == "" {
		return nil, status.Error(codes.InvalidArgument, "game_id is required")
	}

	s.mu.Lock()
	session := s.games[request.GetGameId()]
	if session == nil {
		s.mu.Unlock()
		return nil, gameNotFound(request.GetGameId())
	}
	delete(s.games, request.GetGameId())

	// Holding the store lock while marking the session deleted makes deletion
	// linearizable with a concurrent getSession call.
	session.mu.Lock()
	session.deleted = true
	session.mu.Unlock()
	s.mu.Unlock()

	return &dotscordonv1.DeleteGameResponse{}, nil
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

func validateDimensions(rows, columns uint32) error {
	if rows == 0 || rows > 255 {
		return status.Error(codes.InvalidArgument, "rows must be between 1 and 255")
	}
	if columns == 0 || columns > 255 {
		return status.Error(codes.InvalidArgument, "columns must be between 1 and 255")
	}
	return nil
}

func randomGameID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(value[:]), nil
}

func gameNotFound(gameID string) error {
	return status.Errorf(codes.NotFound, "game %q not found", gameID)
}

func moveStatus(err error) error {
	switch {
	case errors.Is(err, engine.ErrInvalidMove), errors.Is(err, engine.ErrNonEmptyDot):
		return status.Error(codes.InvalidArgument, err.Error())
	case errors.Is(err, engine.ErrUnknownPlayer):
		return status.Error(codes.Internal, err.Error())
	default:
		return status.Error(codes.Internal, "apply move")
	}
}

func cellToProto(dot engine.Dot) dotscordonv1.Cell {
	if dot.Killed {
		if !dot.Owned {
			return dotscordonv1.Cell_CELL_DEAD_EMPTY
		}
		if dot.IsOwnedBy(0) {
			return dotscordonv1.Cell_CELL_DEAD_PLAYER_0
		}
		return dotscordonv1.Cell_CELL_DEAD_PLAYER_1
	}
	if !dot.Owned {
		return dotscordonv1.Cell_CELL_EMPTY
	}
	if dot.IsOwnedBy(0) {
		return dotscordonv1.Cell_CELL_PLAYER_0
	}
	return dotscordonv1.Cell_CELL_PLAYER_1
}

func moveResultToProto(
	player engine.PlayerIndex,
	result *engine.MoveResult,
) *dotscordonv1.MoveResult {
	converted := &dotscordonv1.MoveResult{
		Player:       uint32(player),
		ScoredPoints: result.ScoredPoints,
		KilledCells:  make([]*dotscordonv1.Coordinate, 0, len(result.KilledDots)),
		Cordons:      make([]*dotscordonv1.Cordon, 0, len(result.Cordons)),
	}
	for _, dot := range result.KilledDots {
		converted.KilledCells = append(converted.KilledCells, coordinateToProto(dot.Coord))
	}
	for _, cordon := range result.Cordons {
		convertedCordon := &dotscordonv1.Cordon{
			Positions: make([]*dotscordonv1.Coordinate, 0, len(cordon)),
		}
		for _, dot := range cordon {
			convertedCordon.Positions = append(
				convertedCordon.Positions,
				coordinateToProto(dot.Coord),
			)
		}
		converted.Cordons = append(converted.Cordons, convertedCordon)
	}
	return converted
}

func coordinateToProto(coord engine.Coord) *dotscordonv1.Coordinate {
	return &dotscordonv1.Coordinate{
		Row:    uint32(coord.Row),
		Column: uint32(coord.Col),
	}
}

var _ dotscordonv1.GameServiceServer = (*Service)(nil)
