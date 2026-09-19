from __future__ import annotations

import os

import pytest
import numpy as np

from dots_cordon_ml.encoding import legal_action_mask
from dots_cordon_ml.environment import GameEnvironment
from dots_cordon_ml.search import MCTS


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


@pytest.mark.skipif(
    "DOTS_CORDON_TEST_SERVER" not in os.environ,
    reason="set DOTS_CORDON_TEST_SERVER to run search integration tests",
)
def test_real_search_finds_terminal_capture_and_blocks_losing_reply():
    target = os.environ["DOTS_CORDON_TEST_SERVER"]
    with GameEnvironment(target, 7, 7, max_turns=7) as environment:
        for action in [9, 16, 15, 22, 23, 30]:
            environment.step(action)
        before = environment.game.SerializeToString()

        def uniform(game):
            mask = legal_action_mask(game)
            return mask.astype(float) / mask.sum(), 0

        policy = MCTS(uniform, environment.simulate, simulations=160).search(
            environment.game, np.random.default_rng(3)
        )
        assert policy.argmax() == 17
        assert environment.game.SerializeToString() == before
        result = environment.step(int(policy.argmax()))
        assert result.game.terminal
        assert tuple(result.game.scores) == (1, 0)

    with GameEnvironment(target, 7, 7, max_turns=8) as environment:
        for action in [8, 7, 13, 9, 27, 15]:
            environment.step(action)

        def priors(game):
            mask = legal_action_mask(game)
            policy = mask.astype(float) / mask.sum()
            if game.turn == 6:
                policy[:] = 0
                policy[[1, 48]] = 0.5  # compare the block and observed champion mistake
            elif mask[1]:
                policy[:] = 0
                policy[1] = 1  # opponent examines the available capture
            return policy, 0

        policy = MCTS(priors, environment.simulate, simulations=32).search(
            environment.game, np.random.default_rng(3)
        )
        assert policy.argmax() == 1
