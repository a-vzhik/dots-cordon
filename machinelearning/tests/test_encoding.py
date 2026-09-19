import numpy as np
import pytest

from dots_cordon_ml.encoding import encode_state, legal_action_mask, move_count
from dots_cordon_ml.proto import game_pb2


def game_state() -> game_pb2.GameState:
    return game_pb2.GameState(
        board=game_pb2.Board(
            rows=2,
            columns=3,
            cells=bytes(
                [
                    game_pb2.CELL_EMPTY,
                    game_pb2.CELL_PLAYER_0,
                    game_pb2.CELL_PLAYER_1,
                    game_pb2.CELL_DEAD_EMPTY,
                    game_pb2.CELL_DEAD_PLAYER_0,
                    game_pb2.CELL_DEAD_PLAYER_1,
                ]
            ),
        ),
        scores=[3, 1],
        next_turn_by=1,
    )


def test_encoding_is_player_relative() -> None:
    game = game_state()
    player_zero = encode_state(game, player=0)
    player_one = encode_state(game, player=1)

    assert player_zero.shape == (5, 2, 3)
    np.testing.assert_array_equal(player_zero[0], player_one[1])
    np.testing.assert_array_equal(player_zero[1], player_one[0])
    np.testing.assert_array_equal(player_zero[2], player_one[2])
    np.testing.assert_allclose(player_zero[3], -player_one[3])
    np.testing.assert_allclose(player_zero[4], 4 / 6)


def test_only_empty_cells_are_legal() -> None:
    np.testing.assert_array_equal(
        legal_action_mask(game_state()),
        np.array([True, False, False, False, False, False]),
    )


def test_encoding_rejects_malformed_board() -> None:
    game = game_pb2.GameState(
        board=game_pb2.Board(rows=2, columns=2, cells=b"\x00")
    )
    with pytest.raises(ValueError, match="expected 4"):
        encode_state(game)



def test_move_count_includes_captured_dots_but_not_dead_empty_cells():
    assert move_count(game_state()) == 4
