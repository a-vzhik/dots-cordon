package engine

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestJsonGameRecorderAppendsActions(t *testing.T) {
	filePath := filepath.Join(t.TempDir(), "game.json")
	NewJsonGameRecorder(filePath).RecordStart(7, 7)

	moveResult := MoveResult{
		ScoredPoints: 2,
		Cordons: [][]Dot{{
			{Row: 0, Col: 1, Owned: true, Owner: 0},
		}},
		KilledDots: []Dot{
			{Row: 1, Col: 1, Owned: true, Owner: 1},
		},
	}
	NewJsonGameRecorder(filePath).RecordMove(PlayerIndex(0), 0, 1, moveResult)

	contents, err := os.ReadFile(filePath)
	require.NoError(t, err)
	assert.JSONEq(t, `
		[
			{
				"type": "start",
				"width": 7,
				"height": 7
			},
			{
				"type": "move",
				"playerIndex": 0,
				"row": 0,
				"col": 1,
				"result": {
					"ScoredPoints": 2,
					"Cordons": [[
						{
							"Col": 1,
							"Row": 0,
							"Owned": true,
							"Owner": 0,
							"Killed": false
						}
					]],
					"KilledDots": [
						{
							"Col": 1,
							"Row": 1,
							"Owned": true,
							"Owner": 1,
							"Killed": false
						}
					],
					"IsTerminal": false
				}
			}
		]
	`, string(contents))
}
