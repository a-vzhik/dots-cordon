package geometry

import "testing"

func TestHit(t *testing.T) {
	for _, tt := range []struct {
		name           string
		x, y, row, col int
		ok             bool
	}{
		{"first", Left, Top, 0, 0, true},
		{"last", Left + 14*Spacing, Top + 9*Spacing, 9, 14, true},
		{"near", Left + 3*Spacing + 14, Top + 2*Spacing, 2, 3, true},
		{"outside radius", Left + 15, Top, 0, 0, false},
		{"diagonal outside", Left + 11, Top + 11, 0, 0, false},
		{"between", Left + Spacing/2, Top, 0, 0, false},
		{"left margin inside", Left - 14, Top, 0, 0, true},
		{"outside board", Left - Spacing, Top, 0, 0, false},
		{"status", Left, Top + 10*Spacing, 0, 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			row, col, ok := Hit(tt.x, tt.y, 10, 15)
			if ok != tt.ok || ok && (row != tt.row || col != tt.col) {
				t.Fatalf("Hit = (%d,%d,%v), want (%d,%d,%v)", row, col, ok, tt.row, tt.col, tt.ok)
			}
		})
	}
}
