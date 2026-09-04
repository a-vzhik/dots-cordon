---
name: play-dots-cordon
description: Play Dots Cordon against the user through a persistent interactive CLI session in the local dots-cordon repository. Use when the user asks to start, resume, or take turns in a human-vs-Codex Dots Cordon match; includes the rules and relay protocol so source inspection is unnecessary.
---

# Play Dots Cordon

Play as Player 0 while the user plays as Player 1. Operate the live CLI, relay the board through chat, and keep strategy private during the match. Do not reread the source code as a prerequisite; this skill contains the required rules and operating procedure.

## Start or resume a match

Run commands from the root of the dots-cordon repository that contains this skill.

Start a new two-player game with:

```bash
go run . --board=7x11 --player0 agent --player1 human
```

Run it in a persistent interactive terminal with a TTY and a short initial yield. Retain the returned session identifier across chat turns and send moves to that session's stdin as `row col\n`. If the local Go wrapper fails because `/bin/ps` is sandboxed, retry the same command with the platform's required escalation flow. Wait through a toolchain download or wrapper warning; the useful output begins with the score and board.

Confirm that startup reaches a `Player 0 move` prompt and labels Player 0 as Agent and Player 1 as Human. All three command-line options are required; omitting or mistyping one prints usage and stops the runner. Do not start a second process while a match is active.

Player 0 moves first. Choose and submit the opening move, wait until `Player 1 move` appears, then show the resulting board and ask the user for `row col`.

For every later user turn:

1. If the message is a coordinate, submit that exact move for Player 1. Never invent, alter, or queue moves for the user.
2. Wait for and inspect `MoveResult`, the updated score, the board, and the `Player 0 move` prompt. Captures can materially change the position.
3. Choose and submit Player 0's move.
4. Wait for the next `Player 1 move` prompt.
5. Reply with Player 0's coordinate, the current board in a code block, the score, and `Your move.`

If the user asks a question instead of supplying a coordinate, answer it without advancing the process. If their move is invalid, the CLI re-prompts Player 1; report the error and request another move rather than choosing for them. If Player 0's chosen move is invalid, select another legal move internally. Translate a clear request to quit into uppercase `Q` and report that the match stopped.

Keep the terminal alive while awaiting the next chat turn. If its session is lost, do not reconstruct or continue the position from memory. The current CLI records actions but does not resume an interactive game from the record; tell the user and offer a fresh match.

The game writes `game-<timestamp>.json` in the repository. Preserve it. At game end, report the final score and optionally print the recorded game file.

## Board and input

- The board is N rows by M columns. Rows are `00` - `N-1`; columns are `00` - `M-1`.
- Input order is **row, then column**. For example, `3 5` claims row 3, column 5.
- Player 0 is Codex and is rendered as `0`. Player 1 is the user and is rendered as `1`.
- `.` is playable empty space.
- `x` is a killed Player 0 dot; `X` is a killed Player 1 dot; `-` is killed empty space.
- A move must be in bounds and target an empty, non-killed cell. Players alternate, with Player 0 first.
- Killed cells cannot be played again and killed dots no longer form cordon walls.

## Capture rules

A player's active dots form walls. Wall dots connect in all eight directions, so horizontal, vertical, and diagonal neighbors can form a cordon. The smallest scoring cordon is a diamond: four wall dots north, east, south, and west of one enemy dot. Those four dots touch diagonally and enclose the center.

To determine whether an enemy group escapes, imagine flood fill moving only north, east, south, and west. It can move through empty cells, enemy dots, and killed cells, but not through the current player's active wall dots. If it reaches any outer row or column, the group escapes. Otherwise it is enclosed.

When a move leaves an enemy component without an orthogonal path to an edge:

- the enclosing player's cordon is reported;
- enclosed enemy dots and enclosed empty cells are killed;
- the mover scores one point per enclosed enemy dot;
- boundary dots remain active.

A cordon can enclose several dots, and a larger loop is often needed when the opponent occupies a missing point of a planned diamond. Edge dots always escape. An orthogonal path through killed cells also remains an escape path, which can make old enemy groups permanently difficult or impossible to capture.

Capture detection runs for the player who just moved. If the opponent plays inside an already closed empty cage, that dot is normally captured after the cage owner makes their next legal move anywhere. A terminal move ends the game immediately, so there is no following turn if the opponent fills the last playable cell.

The game ends when no playable empty cells remain. Highest score wins; ordinary territory and connected stones are not worth points.

## Play well

Before every Player 0 move, privately perform this order of analysis:

1. Apply the user's move and read its actual result. Never reason from the previous board after a capture.
2. Audit immediate danger first. Check whether the user's last dot completed a diamond or a larger eight-connected loop around any active Player 0 dots. Inspect whole components, not only local four-point diamonds.
3. Look for a capture now: a missing boundary point that would close a cordon around one or more active Player 1 dots.
4. Look for one-move threats on both sides. Block a genuine opponent closure unless a favorable scoring counter-capture is available.
5. If the opponent occupies a desired boundary point, recalculate a larger ring around the expanded enemy component rather than following the obsolete plan.
6. When no current enemy dot can be captured, build purposeful near-complete cages, restrict safe future cells, or manage late-game move parity.

Do not value connectivity or visual density by itself. A move should capture, prevent a capture, create a concrete enclosure threat, build a specific cage, or serve a clear endgame purpose. Be especially skeptical of passive reinforcement while behind.

Track orthogonal escape routes explicitly. Enemy dots connected to an edge through active enemy dots, empty cells, or killed cells cannot be scored until every such route is blocked. A defender dot actually on an edge cannot be captured.

Near the end, classify remaining cells by who would be captured after playing there. Account for turn parity and the fact that the game stops on the terminal move.

## Match communication

The user is the opponent, not a student. During play, do not reveal planned tactics, threats, candidate moves, or move rationale. Keep tool-progress messages minimal and neutral. It is fine to report observable results such as captured coordinates and score changes.

If the user explicitly asks why a particular move was made, answer candidly about that move only, then return to keeping strategy private. Do not coach the user or warn them about threats.

Raw output may contain Go wrapper warnings and verbose `INFO` cordon-walk logs. Omit that noise from chat. Relay the board, move, captures, score, invalid-move errors, and terminal result accurately.
