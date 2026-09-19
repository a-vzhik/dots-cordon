from copy import deepcopy

import numpy as np
import pytest
import torch

from dots_cordon_ml.dqn import QNetwork
from dots_cordon_ml.proto import game_pb2 as pb
from dots_cordon_ml.search import (
    MCTS,
    PolicyValueAgent,
    PolicyValueNetwork,
    SearchReplay,
    SearchSample,
    augment,
    collect_episode,
)


def game(
    player=0, turn=0, scores=(0, 0), terminal=False, cells=b"\0\0\0\0", name="root"
):
    return pb.GameState(
        game_id=name,
        board=pb.Board(rows=2, columns=2, cells=cells),
        current_player=player,
        turn=turn,
        scores=scores,
        terminal=terminal,
    )


@pytest.mark.parametrize("player", [0, 1])
def test_search_backpropagates_terminal_result_in_correct_perspective(player):
    def simulate(state, action):
        scores = [0, 0]
        scores[player if action == 2 else 1 - player] = 1
        return game(1 - player, scores=scores, terminal=True)

    search = MCTS(lambda _: (np.ones(4) / 4, 0), simulate, simulations=80)
    policy = search.search(game(player), np.random.default_rng(3))
    assert policy.argmax() == 2
    assert policy[2] > 0.8
    assert np.isclose(policy.sum(), 1)


def test_search_accounts_for_opponents_best_reply_not_cooperative_reply():
    # Action 0 lets the opponent win by action 1, or lose by action 0.
    # Root action 1 draws. Search must prefer the draw over an exploitable move.
    def predict(state):
        return np.array([0.5, 0.5, 0, 0]), 0.0

    def simulate(state, action):
        if state.game_id == "root":
            return game(1, 1, terminal=action == 1, name="reply", cells=b"\0\0\1\1")
        return game(0, 2, scores=(1, 0) if action == 0 else (0, 1), terminal=True)

    search = MCTS(predict, simulate, simulations=160)
    policy = search.search(game(cells=b"\0\0\1\1"), np.random.default_rng(3))
    assert policy[1] > 0.8


def test_search_masks_illegal_actions_and_does_not_consume_evaluation_rng():
    rng = np.random.default_rng(3)
    before = deepcopy(rng.bit_generator.state)
    actions = []

    def simulate(state, action):
        actions.append(action)
        return game(1, scores=(1, 0), terminal=True)

    search = MCTS(lambda _: (np.array([100, 1, 100, 100]), 0), simulate, simulations=3)
    policy = search.search(game(cells=b"\1\0\2\3"), rng)
    np.testing.assert_array_equal(policy, [0, 1, 0, 0])
    assert actions == [1]  # cached successor, not another RPC on every visit
    assert before == rng.bit_generator.state


def test_root_noise_is_reproducible_and_affects_self_play():
    search = MCTS(
        lambda _: (np.ones(4) / 4, 0),
        lambda *_: game(1, terminal=True),
        simulations=16,
        noise_fraction=1,
    )
    a = search.search(game(), np.random.default_rng(1), explore=True)
    b = search.search(game(), np.random.default_rng(1), explore=True)
    c = search.search(game(), np.random.default_rng(2), explore=True)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_warm_start_copies_features_only():
    torch.manual_seed(2)
    dqn = QNetwork(8, 1)
    network = PolicyValueNetwork(8, 1)
    heads = {k: v.clone() for k, v in network.state_dict().items() if "head" in k}
    network.initialize_from_dqn(dqn.state_dict())
    for key, value in network.state_dict().items():
        torch.testing.assert_close(
            value, heads[key] if "head" in key else dqn.state_dict()[key]
        )
    with pytest.raises(ValueError, match="architecture"):
        PolicyValueNetwork(8, 0).initialize_from_dqn(dqn.state_dict())


@pytest.mark.parametrize("shape", [(3, 3), (3, 5)])
def test_augmentation_keeps_policy_mask_and_board_aligned(shape):
    rows, cols = shape
    state = np.zeros((5, rows, cols), np.float32)
    state[0, 0, 1] = 1
    mask = state[0].astype(bool).ravel()
    sample = SearchSample(state, mask, mask.astype(np.float32), -1)
    rng = np.random.default_rng(2)
    for _ in range(20):
        result = augment(sample, rng)
        assert result.state.shape == state.shape
        np.testing.assert_array_equal(result.state[0].ravel(), result.policy)
        np.testing.assert_array_equal(result.mask, result.policy.astype(bool))
        assert result.value == -1


def test_policy_value_update_and_replay_roundtrip():
    torch.manual_seed(3)
    agent = PolicyValueAgent(torch.device("cpu"), 4, 0, learning_rate=0.01)
    replay = SearchReplay(10)
    state = np.zeros((5, 2, 2), np.float32)
    sample = SearchSample(
        state,
        np.array([True, True, False, True]),
        np.array([0, 1, 0, 0], np.float32),
        1,
    )
    replay.items.append(sample)
    before = deepcopy(agent.online.state_dict())
    losses = agent.optimize(replay, 1, np.random.default_rng(3))
    assert all(np.isfinite(x) for x in losses.values())
    assert any(
        not torch.equal(before[k], v) for k, v in agent.online.state_dict().items()
    )
    restored = SearchReplay(10)
    restored.restore(replay.state_dict())
    np.testing.assert_array_equal(restored.items[0].state, sample.state)
    np.testing.assert_array_equal(restored.items[0].policy, sample.policy)
    assert restored.items[0].value == 1


def test_episode_labels_each_seat_from_final_outcome():
    class Environment:
        def reset(self):
            self.game = game()
            return self.game

        def step(self, action):
            self.game = game(
                1 - self.game.current_player,
                self.game.turn + 1,
                scores=(3, 1),
                terminal=self.game.turn == 1,
            )
            return type("Step", (), {"game": self.game})()

    class Search:
        def search(self, *args, **kwargs):
            return np.array([0, 1, 0, 0], np.float32)

    samples, scores, moves = collect_episode(
        Environment(), Search(), np.random.default_rng(1)
    )
    assert [s.value for s in samples] == [1, -1]
    assert scores == (3, 1)
    assert moves == [1, 1]
