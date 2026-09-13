"""Convolutional Q-network, replay memory, and DQN optimization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .encoding import INPUT_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.conv1(inputs))
        return F.relu(inputs + self.conv2(hidden))


class QNetwork(nn.Module):
    """A fully convolutional network producing one Q-value per board cell."""

    def __init__(self, channels: int = 64, blocks: int = 3) -> None:
        super().__init__()
        if channels < 4:
            raise ValueError("channels must be at least 4")
        if blocks < 0:
            raise ValueError("blocks must be non-negative")

        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.stem(inputs))
        return self.head(hidden).flatten(start_dim=1)


@dataclass(frozen=True, slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_legal_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, count: int) -> list[Transition]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        return self._random.sample(self._items, count)

    def random_state(self) -> tuple[object, ...]:
        """Return the sampler state so checkpointed runs can resume reproducibly."""
        return self._random.getstate()

    def restore_random_state(self, state: tuple[object, ...]) -> None:
        self._random.setstate(state)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterable[Transition]:
        return iter(self._items)


class DQNAgent:
    def __init__(
        self,
        device: torch.device,
        learning_rate: float,
        gamma: float,
        seed: int,
        channels: int = 64,
        blocks: int = 3,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.random = np.random.default_rng(seed)
        self.online = QNetwork(channels=channels, blocks=blocks).to(device)
        self.target = QNetwork(channels=channels, blocks=blocks).to(device)
        self.sync_target()
        self.target.eval()
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=learning_rate)

    def select_action(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        legal_actions = np.flatnonzero(legal_mask)
        if legal_actions.size == 0:
            raise ValueError("cannot select an action without any legal moves")
        # Greedy evaluation must not advance the exploration random stream. If it
        # did, changing evaluation frequency would also change subsequent training.
        if epsilon > 0.0 and self.random.random() < epsilon:
            return int(self.random.choice(legal_actions))

        self.online.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.online(state_tensor).squeeze(0)
            mask_tensor = torch.from_numpy(legal_mask).to(self.device)
            q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
            action = int(q_values.argmax().item())
        self.online.train()
        return action

    def optimize(self, replay: ReplayBuffer, batch_size: int) -> float:
        transitions = replay.sample(batch_size)
        states = torch.from_numpy(np.stack([item.state for item in transitions])).to(self.device)
        actions = torch.tensor(
            [item.action for item in transitions], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [item.reward for item in transitions], dtype=torch.float32, device=self.device
        )
        next_states = torch.from_numpy(
            np.stack([item.next_state for item in transitions])
        ).to(self.device)
        next_masks = torch.from_numpy(
            np.stack([item.next_legal_mask for item in transitions])
        ).to(self.device)
        dones = torch.tensor(
            [item.done for item in transitions], dtype=torch.bool, device=self.device
        )

        predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: choose with the online network, evaluate with target.
            next_online = self.online(next_states).masked_fill(~next_masks, -torch.inf)
            next_actions = next_online.argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            next_values = torch.where(dones, torch.zeros_like(next_values), next_values)
            expected = rewards + self.gamma * next_values

        loss = F.smooth_l1_loss(predicted, expected)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
