"""Loading and validation shared by training and evaluation commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .dqn import DQNAgent


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    rows: int
    columns: int
    channels: int
    blocks: int
    episode: int
    environment_steps: int
    optimization_steps: int

    @property
    def board(self) -> tuple[int, int]:
        return (self.rows, self.columns)

    @property
    def model(self) -> tuple[int, int]:
        return (self.channels, self.blocks)


def read_checkpoint(
    path: Path,
    map_location: str | torch.device,
) -> tuple[dict[str, Any], CheckpointMetadata]:
    """Read a trusted trainer checkpoint and extract its required metadata."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint {path} does not contain a dictionary")

    try:
        board = checkpoint["board"]
        model = checkpoint["model"]
        training = checkpoint["training_state"]
        metadata = CheckpointMetadata(
            rows=int(board["rows"]),
            columns=int(board["columns"]),
            channels=int(model["channels"]),
            blocks=int(model["blocks"]),
            episode=int(training["episode"]),
            environment_steps=int(training["environment_steps"]),
            optimization_steps=int(training["optimization_steps"]),
        )
        checkpoint["online"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint {path} has invalid or missing metadata") from exc

    return checkpoint, metadata


def restore_agent(
    agent: DQNAgent,
    checkpoint: dict[str, Any],
    *,
    restore_optimizer: bool,
) -> None:
    agent.online.load_state_dict(checkpoint["online"])
    if "target" in checkpoint:
        agent.target.load_state_dict(checkpoint["target"])
    else:
        agent.sync_target()
    if restore_optimizer:
        try:
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
        except KeyError as exc:
            raise ValueError("checkpoint does not contain optimizer state") from exc

