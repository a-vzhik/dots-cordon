package engine

type Recorder interface {
	RecordStart(width uint8, height uint8)
	RecordMove(playedIdx PlayerIndex, row uint8, col uint8, result MoveResult)
}

type NoopGameRecorder struct{}

var _ Recorder = NoopGameRecorder{}

func (NoopGameRecorder) RecordStart(_ uint8, _ uint8) {}

func (NoopGameRecorder) RecordMove(_ PlayerIndex, _ uint8, _ uint8, _ MoveResult) {}
