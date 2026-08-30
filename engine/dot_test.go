package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDotTransformationsDoNotMutateSource(t *testing.T) {
	tests := []struct {
		name   string
		source Dot
		apply  func(Dot) Dot
		want   Dot
	}{
		{
			name: "with owner",
			source: Dot{
				Coord: Coord{
					Col: 2,
					Row: 3,
				},
				Killed: true,
			},
			apply: func(dot Dot) Dot {
				return dot.WithOwner(PlayerIndex(1))
			},
			want: Dot{
				Coord: Coord{
					Col: 2,
					Row: 3,
				}, Owned: true,
				Owner:  PlayerIndex(1),
				Killed: true,
			},
		},
		{
			name: "with killed",
			source: Dot{
				Coord: Coord{
					Col: 4,
					Row: 5,
				}, Owned: true,
				Owner: PlayerIndex(1),
			},
			apply: func(dot Dot) Dot {
				return dot.WithKilled()
			},
			want: Dot{
				Coord: Coord{
					Col: 4,
					Row: 5,
				}, Owned: true,
				Owner:  PlayerIndex(1),
				Killed: true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := tt.source

			got := tt.apply(tt.source)

			assert.Equal(t, original, tt.source)
			assert.Equal(t, tt.want, got)
		})
	}
}
