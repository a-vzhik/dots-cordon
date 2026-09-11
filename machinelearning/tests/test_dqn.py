import numpy as np
import torch

from dots_cordon_ml.dqn import DQNAgent, QNetwork, ReplayBuffer, Transition


def test_network_returns_one_value_per_cell() -> None:
    network = QNetwork(channels=8, blocks=1)
    output = network(torch.zeros((2, 5, 4, 6)))
    assert output.shape == (2, 24)


def test_agent_never_selects_an_illegal_action() -> None:
    agent = DQNAgent(
        device=torch.device("cpu"),
        learning_rate=1e-3,
        gamma=0.99,
        seed=3,
        channels=8,
        blocks=0,
    )
    state = np.zeros((5, 2, 3), dtype=np.float32)
    mask = np.array([False, False, True, False, False, False])
    assert agent.select_action(state, mask, epsilon=0.0) == 2
    assert agent.select_action(state, mask, epsilon=1.0) == 2


def test_optimize_handles_terminal_state_without_legal_moves() -> None:
    agent = DQNAgent(
        device=torch.device("cpu"),
        learning_rate=1e-3,
        gamma=0.99,
        seed=3,
        channels=8,
        blocks=0,
    )
    replay = ReplayBuffer(capacity=2, seed=3)
    state = np.zeros((5, 2, 2), dtype=np.float32)
    replay.add(
        Transition(
            state=state,
            action=0,
            reward=1.0,
            next_state=state,
            next_legal_mask=np.zeros(4, dtype=bool),
            done=True,
        )
    )
    loss = agent.optimize(replay, batch_size=1)
    assert np.isfinite(loss)

