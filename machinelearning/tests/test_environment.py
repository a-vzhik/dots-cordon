from __future__ import annotations

import os

import pytest

from dots_cordon_ml.encoding import legal_action_mask
from dots_cordon_ml.environment import GameEnvironment


@pytest.mark.skipif(
    "DOTS_CORDON_TEST_SERVER" not in os.environ,
    reason="set DOTS_CORDON_TEST_SERVER to run the server integration test",
)
def test_game_server_lifecycle() -> None:
    with GameEnvironment(
        target=os.environ["DOTS_CORDON_TEST_SERVER"],
        rows=2,
        columns=2,
    ) as environment:
        initial = environment.reset()
        assert initial.turn == 0
        assert legal_action_mask(initial).all()

        moved = environment.step(1)
        assert moved.player == 0
        assert moved.game.turn == 1
        assert not legal_action_mask(moved.game)[1]

