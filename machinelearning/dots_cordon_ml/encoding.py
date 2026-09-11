"""Convert protobuf game states into player-relative model inputs."""

from __future__ import annotations

import numpy as np

from .proto import game_pb2

INPUT_CHANNELS = 5


def legal_action_mask(game: game_pb2.GameState) -> np.ndarray:
    """Return a flat mask whose true entries are legal placement actions."""
    board = game.board
    cells = _cells(game)
    return (cells.reshape(board.rows * board.columns) == game_pb2.CELL_EMPTY).copy()


def encode_state(game: game_pb2.GameState, player: int | None = None) -> np.ndarray:
    """Encode a state from one player's perspective as ``[5, rows, columns]``.

    The channels are own live dots, opponent live dots, dead territory, current
    score difference, and episode progress. Player-relative encoding lets one
    network control both sides during self-play.
    """
    if player is None:
        player = game.current_player
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player}")

    board = game.board
    cells = _cells(game)
    own_cell = game_pb2.CELL_PLAYER_0 if player == 0 else game_pb2.CELL_PLAYER_1
    opponent_cell = game_pb2.CELL_PLAYER_1 if player == 0 else game_pb2.CELL_PLAYER_0

    encoded = np.empty((INPUT_CHANNELS, board.rows, board.columns), dtype=np.float32)
    encoded[0] = cells == own_cell
    encoded[1] = cells == opponent_cell
    encoded[2] = cells >= game_pb2.CELL_DEAD_EMPTY

    own_score = game.scores[player] if len(game.scores) > player else 0
    opponent_score = game.scores[1 - player] if len(game.scores) > 1 - player else 0
    board_size = max(1, board.rows * board.columns)
    encoded[3].fill((own_score - opponent_score) / board_size)
    encoded[4].fill(min(game.turn / board_size, 1.0))
    return encoded


def _cells(game: game_pb2.GameState) -> np.ndarray:
    board = game.board
    expected = board.rows * board.columns
    cells = np.frombuffer(board.cells, dtype=np.uint8)
    if cells.size != expected:
        raise ValueError(
            f"board contains {cells.size} cells; expected {expected} "
            f"for {board.rows}x{board.columns}"
        )
    return cells.reshape(board.rows, board.columns)

