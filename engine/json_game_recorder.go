package engine

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
)

type JsonGameRecorder struct {
	FilePath string
}

type jsonGameStartAction struct {
	Type   string `json:"type"`
	Width  uint8  `json:"width"`
	Height uint8  `json:"height"`
}

type jsonGameMoveAction struct {
	Type        string      `json:"type"`
	PlayerIndex PlayerIndex `json:"playerIndex"`
	Row         uint8       `json:"row"`
	Col         uint8       `json:"col"`
	Result      MoveResult  `json:"result"`
}

var _ Recorder = (*JsonGameRecorder)(nil)

func NewJsonGameRecorder(filePath string) *JsonGameRecorder {
	return &JsonGameRecorder{FilePath: filePath}
}

func (recorder *JsonGameRecorder) RecordStart(width uint8, height uint8) {
	recorder.record(jsonGameStartAction{
		Type:   "start",
		Width:  width,
		Height: height,
	})
}

func (recorder *JsonGameRecorder) RecordMove(
	playerIdx PlayerIndex,
	row uint8,
	col uint8,
	result MoveResult,
) {
	recorder.record(jsonGameMoveAction{
		Type:        "move",
		PlayerIndex: playerIdx,
		Row:         row,
		Col:         col,
		Result:      result,
	})
}

func (recorder *JsonGameRecorder) record(action any) {
	if err := recorder.appendAction(action); err != nil {
		slog.Error(
			"failed to record game action",
			"path", recorder.FilePath,
			"error", err,
		)
	}
}

func (recorder *JsonGameRecorder) appendAction(action any) error {
	actions, err := recorder.readActions()
	if err != nil {
		return err
	}

	encodedAction, err := json.Marshal(action)
	if err != nil {
		return fmt.Errorf("encode action: %w", err)
	}
	actions = append(actions, encodedAction)

	contents, err := json.MarshalIndent(actions, "", "  ")
	if err != nil {
		return fmt.Errorf("encode game record: %w", err)
	}
	contents = append(contents, '\n')

	if err := os.WriteFile(recorder.FilePath, contents, 0o644); err != nil {
		return fmt.Errorf("write game record: %w", err)
	}

	return nil
}

func (recorder *JsonGameRecorder) readActions() ([]json.RawMessage, error) {
	contents, err := os.ReadFile(recorder.FilePath)
	if errors.Is(err, os.ErrNotExist) {
		return []json.RawMessage{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read game record: %w", err)
	}
	if len(bytes.TrimSpace(contents)) == 0 {
		return []json.RawMessage{}, nil
	}

	var actions []json.RawMessage
	if err := json.Unmarshal(contents, &actions); err != nil {
		return nil, fmt.Errorf("decode game record: %w", err)
	}

	return actions, nil
}
