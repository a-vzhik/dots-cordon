from pathlib import Path

import torch

from dots_cordon_ml.checkpoint import read_checkpoint, restore_agent
from dots_cordon_ml.dqn import DQNAgent


def agent(seed: int) -> DQNAgent:
    return DQNAgent(
        device=torch.device("cpu"),
        learning_rate=1e-3,
        gamma=0.99,
        seed=seed,
        channels=8,
        blocks=1,
    )


def test_checkpoint_metadata_and_weights_are_restored(tmp_path: Path) -> None:
    source = agent(seed=1)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "online": source.online.state_dict(),
            "target": source.target.state_dict(),
            "optimizer": source.optimizer.state_dict(),
            "training_state": {
                "episode": 123,
                "environment_steps": 456,
                "optimization_steps": 321,
            },
            "board": {"rows": 7, "columns": 7},
            "model": {"channels": 8, "blocks": 1},
        },
        path,
    )

    checkpoint, metadata = read_checkpoint(path, map_location="cpu")
    destination = agent(seed=2)
    restore_agent(destination, checkpoint, restore_optimizer=True)

    assert metadata.board == (7, 7)
    assert metadata.model == (8, 1)
    assert metadata.episode == 123
    for expected, actual in zip(
        source.online.parameters(), destination.online.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)

