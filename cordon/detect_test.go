package cordon_test

import (
	"testing"

	"github.com/a-vzhik/dots-cordon/cordon"
)

func placeLive(b *cordon.Board, player cordon.Player, pts ...cordon.Point) {
	cell := cordon.CellLiveA
	if player == cordon.PlayerB {
		cell = cordon.CellLiveB
	}
	for _, p := range pts {
		b.Set(p, cell)
	}
}

func TestDetect_WhenHexRingClosesTwoDotsThenBothCaptured(t *testing.T) {
	// y-down grid from game_rules walkthrough (4x5).
	b := cordon.NewBoard(4, 5)

	placeLive(b, cordon.PlayerA,
		cordon.Point{2, 4},
		cordon.Point{1, 3}, cordon.Point{3, 3},
		cordon.Point{1, 2}, cordon.Point{3, 2},
		cordon.Point{2, 1},
	)
	placeLive(b, cordon.PlayerB,
		cordon.Point{2, 3}, cordon.Point{2, 2},
		cordon.Point{0, 4}, cordon.Point{1, 4}, cordon.Point{3, 0},
	)

	results := cordon.Detect(b, cordon.PlayerA)
	if len(results) != 1 {
		t.Fatalf("expected 1 cordon, got %d", len(results))
	}

	got := results[0]
	if len(got.Captured) != 2 {
		t.Fatalf("expected 2 captures, got %v", got.Captured)
	}
	if len(got.Wall) != 6 {
		t.Fatalf("expected wall of 6 dots, got %d: %v", len(got.Wall), got.Wall)
	}
}

func TestDetect_WhenCardinalCrossClosesThenOneCaptured(t *testing.T) {
	b := cordon.NewBoard(5, 5)

	placeLive(b, cordon.PlayerA,
		cordon.Point{2, 1},
		cordon.Point{1, 2}, cordon.Point{3, 2},
		cordon.Point{2, 3},
	)
	placeLive(b, cordon.PlayerB, cordon.Point{2, 2})

	results := cordon.Detect(b, cordon.PlayerA)
	if len(results) != 1 {
		t.Fatalf("expected 1 cordon, got %d", len(results))
	}
	if len(results[0].Captured) != 1 {
		t.Fatalf("expected 1 capture, got %v", results[0].Captured)
	}
}

func TestDetect_WhenCornerRingHasPinholesThenNoCapture(t *testing.T) {
	b := cordon.NewBoard(5, 5)

	placeLive(b, cordon.PlayerA,
		cordon.Point{1, 1}, cordon.Point{3, 1},
		cordon.Point{1, 3}, cordon.Point{3, 3},
	)
	placeLive(b, cordon.PlayerB, cordon.Point{2, 2})

	results := cordon.Detect(b, cordon.PlayerA)
	if len(results) != 0 {
		t.Fatalf("expected no cordon, got %v", results)
	}
}

func TestDetect_WhenWaistDotClosesTwoPocketsThenTwoCordons(t *testing.T) {
	// Vertical cage split by a waist dot at (2, 4).
	b := cordon.NewBoard(5, 9)

	placeLive(b, cordon.PlayerA,
		cordon.Point{2, 1},
		cordon.Point{1, 2}, cordon.Point{3, 2},
		cordon.Point{1, 3}, cordon.Point{3, 3},
		cordon.Point{1, 5}, cordon.Point{3, 5},
		cordon.Point{1, 6}, cordon.Point{3, 6},
		cordon.Point{2, 7},
		cordon.Point{2, 4}, // waist
	)
	placeLive(b, cordon.PlayerB,
		cordon.Point{2, 2}, cordon.Point{2, 3},
		cordon.Point{2, 5}, cordon.Point{2, 6},
	)

	results := cordon.Detect(b, cordon.PlayerA)
	if len(results) != 2 {
		t.Fatalf("expected 2 cordons, got %d: %+v", len(results), results)
	}

	totalCaptured := 0
	for _, r := range results {
		totalCaptured += len(r.Captured)
	}
	if totalCaptured != 4 {
		t.Fatalf("expected 4 total captures, got %d", totalCaptured)
	}
}

func TestDetect_WhenOpponentOnBorderThenNotCaptured(t *testing.T) {
	b := cordon.NewBoard(4, 5)

	placeLive(b, cordon.PlayerA,
		cordon.Point{2, 4},
		cordon.Point{1, 3}, cordon.Point{3, 3},
		cordon.Point{1, 2}, cordon.Point{3, 2},
		cordon.Point{2, 1},
	)
	placeLive(b, cordon.PlayerB,
		cordon.Point{2, 3}, cordon.Point{2, 2},
		cordon.Point{0, 4},
	)

	results := cordon.Detect(b, cordon.PlayerA)
	if len(results) != 1 || len(results[0].Captured) != 2 {
		t.Fatalf("unexpected cordons: %+v", results)
	}
}

func TestDetect_WhenPlayerBClosesThenMoverIsB(t *testing.T) {
	b := cordon.NewBoard(5, 5)

	placeLive(b, cordon.PlayerB,
		cordon.Point{2, 1},
		cordon.Point{1, 2}, cordon.Point{3, 2},
		cordon.Point{2, 3},
	)
	placeLive(b, cordon.PlayerA, cordon.Point{2, 2})

	results := cordon.Detect(b, cordon.PlayerB)
	if len(results) != 1 {
		t.Fatalf("expected 1 cordon, got %d", len(results))
	}
	if results[0].Mover != cordon.PlayerB {
		t.Fatalf("expected mover B, got %v", results[0].Mover)
	}
}
