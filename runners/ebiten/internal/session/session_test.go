package session

import (
	"errors"
	"reflect"
	"testing"

	"github.com/a-vzhik/dots-cordon/engine"
)

func play(t *testing.T, s *Session, coords ...engine.Coord) {
	t.Helper()
	for _, coord := range coords {
		if err := s.Move(coord.Row, coord.Col); err != nil {
			t.Fatalf("Move(%d, %d): %v", coord.Row, coord.Col, err)
		}
	}
}

func TestNew(t *testing.T) {
	s := New()
	snap := s.Snapshot()
	if len(snap.Dots) != Rows || snap.NextPlayer != 0 || snap.Terminal || snap.Scores != [2]uint32{} || len(snap.Cordons) != 0 {
		t.Fatalf("unexpected initial state: %+v", snap)
	}
	for row, dots := range snap.Dots {
		if len(dots) != Columns {
			t.Fatalf("row %d has %d columns", row, len(dots))
		}
		for col, dot := range dots {
			if dot != (engine.Dot{Coord: engine.Coord{Row: uint8(row), Col: uint8(col)}}) {
				t.Fatalf("unexpected dot: %+v", dot)
			}
		}
	}
	if s.game.Players[0].Color != engine.RedColor || s.game.Players[1].Color != engine.BlueColor {
		t.Fatal("incorrect player colors")
	}
}

func TestInvalidMovesDoNotAdvanceOrChangeState(t *testing.T) {
	s := New()
	play(t, s, engine.Coord{Row: 0, Col: 0})
	for _, tc := range []struct {
		row, col uint8
		want     error
	}{
		{Rows, 0, engine.ErrInvalidMove}, {0, Columns, engine.ErrInvalidMove},
		{255, 255, engine.ErrInvalidMove}, {0, 0, engine.ErrNonEmptyDot},
	} {
		before := s.Snapshot()
		if err := s.Move(tc.row, tc.col); !errors.Is(err, tc.want) {
			t.Fatalf("got %v, want %v", err, tc.want)
		}
		if !reflect.DeepEqual(before, s.Snapshot()) {
			t.Fatal("invalid move changed state")
		}
	}
	play(t, s, engine.Coord{Row: 0, Col: 1})
	if snap := s.Snapshot(); snap.NextPlayer != 0 || !snap.Dots[0][1].IsOwnedBy(1) {
		t.Fatal("turn did not alternate")
	}
}

func TestCaptureScoreHistoryAndSnapshotIsolation(t *testing.T) {
	s := New()
	initial := s.Snapshot()
	play(t, s, engine.Coord{Row: 1, Col: 2}, engine.Coord{Row: 2, Col: 2},
		engine.Coord{Row: 2, Col: 1}, engine.Coord{Row: 9, Col: 14},
		engine.Coord{Row: 2, Col: 3}, engine.Coord{Row: 9, Col: 13})
	beforeCapture := s.Snapshot()
	play(t, s, engine.Coord{Row: 3, Col: 2})
	captured := s.Snapshot()
	if captured.Scores != [2]uint32{1, 0} || !captured.Dots[2][2].Killed || captured.NextPlayer != 1 || captured.Terminal {
		t.Fatalf("unexpected capture: %+v", captured)
	}
	if len(captured.Cordons) != 1 || captured.Cordons[0].Player != 0 || len(captured.Cordons[0].Points) < 4 {
		t.Fatalf("unexpected cordons: %+v", captured.Cordons)
	}
	for _, point := range captured.Cordons[0].Points {
		if !captured.Dots[point.Row][point.Col].IsOwnedBy(0) {
			t.Fatalf("cordon point not owned: %+v", point)
		}
	}
	if initial.Dots[1][2].Owned || beforeCapture.Dots[2][2].Killed || beforeCapture.Scores[0] != 0 || len(beforeCapture.Cordons) != 0 {
		t.Fatal("old snapshot changed")
	}
	if err := s.Move(2, 2); !errors.Is(err, engine.ErrNonEmptyDot) {
		t.Fatalf("captured dot accepted: %v", err)
	}
	if !reflect.DeepEqual(captured, s.Snapshot()) {
		t.Fatal("invalid capture move changed state")
	}
	play(t, s, engine.Coord{Row: 8, Col: 14})
	if !reflect.DeepEqual(captured.Cordons, s.Snapshot().Cordons) {
		t.Fatal("cordon history lost on later move")
	}
	clean := s.Snapshot()
	mutated := s.Snapshot()
	mutated.Dots[0][0].Owned = true
	mutated.Dots[1] = nil
	mutated.Scores[0] = 999
	mutated.NextPlayer = 1
	mutated.Terminal = true
	mutated.Cordons[0].Player = 1
	mutated.Cordons[0].Points[0] = engine.Coord{Row: 255, Col: 255}
	mutated.Cordons = append(mutated.Cordons, Cordon{})
	if !reflect.DeepEqual(clean, s.Snapshot()) {
		t.Fatal("snapshot mutation leaked into session")
	}
	if !reflect.DeepEqual(captured.Cordons, clean.Cordons) {
		t.Fatal("snapshots share cordon storage")
	}
}

func TestFullBoardTerminal(t *testing.T) {
	s := New()
	for row := uint8(0); row < Rows; row++ {
		for col := uint8(0); col < Columns; col++ {
			if s.Snapshot().Terminal {
				t.Fatalf("terminal before board filled at %d,%d", row, col)
			}
			play(t, s, engine.Coord{Row: row, Col: col})
		}
	}
	final := s.Snapshot()
	if !final.Terminal {
		t.Fatal("full board not terminal")
	}
	if err := s.Move(0, 0); err == nil {
		t.Fatal("accepted move after terminal")
	}
	if !reflect.DeepEqual(final, s.Snapshot()) {
		t.Fatal("terminal move changed state")
	}
}
