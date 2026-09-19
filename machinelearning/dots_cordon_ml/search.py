"""Policy/value learning and PUCT search using the real game's simulator.

Values are always from the player-to-move's perspective. Rewards are terminal
win/draw/loss only; there are no tactical rules or handcrafted move bonuses.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dqn import ResidualBlock
from .encoding import INPUT_CHANNELS, encode_state, legal_action_mask
from .proto import game_pb2


class PolicyValueNetwork(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 3) -> None:
        super().__init__()
        if channels < 4 or blocks < 0:
            raise ValueError("channels must be >= 4 and blocks must be >= 0")
        # Names and shapes deliberately match the DQN feature extractor.
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1), nn.ReLU()
        )
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy_head = nn.Conv2d(channels, 1, 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.blocks(self.stem(inputs))
        return self.policy_head(features).flatten(1), self.value_head(features).squeeze(
            1
        )

    def initialize_from_dqn(self, weights: dict[str, torch.Tensor]) -> None:
        features = {
            key: value
            for key, value in weights.items()
            if key.startswith(("stem.", "blocks."))
        }
        expected = {
            key for key in self.state_dict() if key.startswith(("stem.", "blocks."))
        }
        if set(features) != expected:
            raise ValueError("DQN feature extractor does not match this architecture")
        self.load_state_dict({**self.state_dict(), **features})


class PolicyValueAgent:
    def __init__(
        self,
        device: torch.device,
        channels: int,
        blocks: int,
        learning_rate: float = 3e-4,
    ) -> None:
        self.device = device
        self.online = PolicyValueNetwork(channels, blocks).to(device)
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=learning_rate)

    def predict(self, game: game_pb2.GameState) -> tuple[np.ndarray, float]:
        mask = legal_action_mask(game)
        if game.terminal or not mask.any():
            raise ValueError("network prediction requires a nonterminal position")
        self.online.eval()
        with torch.inference_mode():
            logits, value = self.online(
                torch.from_numpy(encode_state(game))[None].to(self.device)
            )
            logits = logits[0].masked_fill(
                ~torch.from_numpy(mask).to(self.device), -torch.inf
            )
            return logits.softmax(0).cpu().numpy(), float(value.item())

    def select_action(
        self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float = 0.0
    ) -> int:
        if epsilon != 0:
            raise ValueError(
                "policy evaluation is greedy; use MCTS exploration for training"
            )
        if not legal_mask.any():
            raise ValueError("no legal actions")
        self.online.eval()
        with torch.inference_mode():
            logits, _ = self.online(torch.from_numpy(state)[None].to(self.device))
            return int(
                logits[0]
                .masked_fill(~torch.from_numpy(legal_mask).to(self.device), -torch.inf)
                .argmax()
            )

    def optimize(
        self, replay: SearchReplay, batch_size: int, random: np.random.Generator
    ) -> dict[str, float]:
        samples = replay.sample(batch_size, random)
        augmented = [augment(sample, random) for sample in samples]
        states = torch.from_numpy(np.stack([x.state for x in augmented])).to(
            self.device
        )
        masks = torch.from_numpy(np.stack([x.mask for x in augmented])).to(self.device)
        policies = torch.from_numpy(np.stack([x.policy for x in augmented])).to(
            self.device
        )
        values = torch.tensor(
            [x.value for x in augmented], dtype=torch.float32, device=self.device
        )
        self.online.train()
        logits, predicted_values = self.online(states)
        # Finite masking avoids 0 * -inf in policy cross entropy.
        log_probs = F.log_softmax(logits.masked_fill(~masks, -1e9), dim=1)
        policy_loss = -(policies * log_probs).sum(1).mean()
        value_loss = F.mse_loss(predicted_values, values)
        loss = policy_loss + value_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }


@dataclass
class Node:
    game: game_pb2.GameState
    prior: np.ndarray | None = None
    visits: np.ndarray | None = None
    totals: np.ndarray | None = None
    children: dict[int, Node] = field(default_factory=dict)


class MCTS:
    def __init__(
        self,
        predict: Callable,
        simulate: Callable,
        simulations: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        noise_fraction: float = 0.25,
    ) -> None:
        if (
            simulations < 1
            or c_puct <= 0
            or dirichlet_alpha <= 0
            or not 0 <= noise_fraction <= 1
        ):
            raise ValueError("invalid search configuration")
        self.predict, self.simulate = predict, simulate
        self.simulations, self.c_puct = simulations, c_puct
        self.dirichlet_alpha, self.noise_fraction = dirichlet_alpha, noise_fraction

    def _expand(self, node: Node) -> float:
        prior, value = self.predict(node.game)
        mask = legal_action_mask(node.game)
        prior = np.asarray(prior, dtype=np.float64).copy()
        if (
            prior.shape != mask.shape
            or not np.isfinite(prior).all()
            or (prior < 0).any()
            or not np.isfinite(value)
        ):
            raise ValueError("invalid policy/value prediction")
        prior[~mask] = 0
        if prior.sum() <= 0:
            raise ValueError("policy has no legal probability mass")
        node.prior = prior / prior.sum()
        node.visits = np.zeros(len(prior), dtype=np.int64)
        node.totals = np.zeros(len(prior), dtype=np.float64)
        return float(value)

    def search(
        self,
        game: game_pb2.GameState,
        random: np.random.Generator,
        *,
        explore: bool = False,
    ) -> np.ndarray:
        if game.terminal or not legal_action_mask(game).any():
            raise ValueError("cannot search a terminal position")
        root = Node(game)
        self._expand(root)
        if explore and self.noise_fraction:
            legal = np.flatnonzero(legal_action_mask(game))
            noise = random.dirichlet(np.full(len(legal), self.dirichlet_alpha))
            root.prior[legal] = (1 - self.noise_fraction) * root.prior[
                legal
            ] + self.noise_fraction * noise
        for _ in range(self.simulations):
            node, path = root, []
            while node.prior is not None and not node.game.terminal:
                mean = np.divide(
                    node.totals,
                    node.visits,
                    out=np.zeros_like(node.totals),
                    where=node.visits > 0,
                )
                bonus = (
                    self.c_puct
                    * node.prior
                    * np.sqrt(1 + node.visits.sum())
                    / (1 + node.visits)
                )
                scores = np.where(legal_action_mask(node.game), mean + bonus, -np.inf)
                action = int(scores.argmax())
                path.append((node, action))
                if action not in node.children:
                    node.children[action] = Node(self.simulate(node.game, action))
                node = node.children[action]
            if node.game.terminal:
                player = node.game.current_player
                value = float(
                    np.sign(
                        int(node.game.scores[player])
                        - int(node.game.scores[1 - player])
                    )
                )
            else:
                value = self._expand(node)
            # Every placement changes player. An edge stores its parent's value.
            for parent, action in reversed(path):
                value = -value
                parent.visits[action] += 1
                parent.totals[action] += value
        return (root.visits / root.visits.sum()).astype(np.float32)


@dataclass(frozen=True)
class SearchSample:
    state: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    value: float


def augment(sample: SearchSample, random: np.random.Generator) -> SearchSample:
    rows, columns = sample.state.shape[-2:]
    rotations = (
        int(random.integers(4)) if rows == columns else int(random.integers(2)) * 2
    )
    reflect = bool(random.integers(2))

    def transform(array):
        result = np.rot90(array, rotations, axes=(-2, -1))
        if reflect:
            result = np.flip(result, axis=-1)
        return np.ascontiguousarray(result)

    return SearchSample(
        transform(sample.state),
        transform(sample.mask.reshape(rows, columns)).ravel(),
        transform(sample.policy.reshape(rows, columns)).ravel(),
        sample.value,
    )


class SearchReplay:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.items: deque[SearchSample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, count: int, random: np.random.Generator) -> list[SearchSample]:
        return [
            self.items[int(i)]
            for i in random.choice(len(self.items), count, replace=False)
        ]

    def state_dict(self) -> dict:
        if not self.items:
            return {}
        return {
            "states": torch.from_numpy(np.stack([x.state for x in self.items])),
            "masks": torch.from_numpy(np.stack([x.mask for x in self.items])),
            "policies": torch.from_numpy(np.stack([x.policy for x in self.items])),
            "values": torch.tensor([x.value for x in self.items], dtype=torch.float32),
        }

    def restore(self, state: dict) -> None:
        self.items.clear()
        if state:
            for s, m, p, v in zip(
                state["states"],
                state["masks"],
                state["policies"],
                state["values"],
                strict=True,
            ):
                self.items.append(
                    SearchSample(
                        s.cpu().numpy().copy(),
                        m.cpu().numpy().copy(),
                        p.cpu().numpy().copy(),
                        float(v),
                    )
                )


def collect_episode(
    environment, search: MCTS, random: np.random.Generator, temperature_moves: int = 12
) -> tuple[list[SearchSample], tuple[int, int], list[int]]:
    game = environment.reset()
    pending, moves = [], []
    while not game.terminal:
        policy = search.search(game, random, explore=True)
        pending.append(
            (encode_state(game), legal_action_mask(game), policy, game.current_player)
        )
        action = (
            int(
                random.choice(
                    len(policy),
                    p=policy.astype(np.float64) / policy.sum(dtype=np.float64),
                )
            )
            if game.turn < temperature_moves
            else int(policy.argmax())
        )
        moves.append(action)
        game = environment.step(action).game
    scores = (int(game.scores[0]), int(game.scores[1]))
    samples = [
        SearchSample(s, m, p, float(np.sign(scores[player] - scores[1 - player])))
        for s, m, p, player in pending
    ]
    return samples, scores, moves


class SearchPlayer:
    """Adapter for existing paired-seat evaluators; search never advances the game."""

    def __init__(
        self,
        agent: PolicyValueAgent,
        environment,
        simulations: int,
        seed: int = 0,
        c_puct: float = 1.5,
    ) -> None:
        self.searcher = MCTS(agent.predict, environment.simulate, simulations, c_puct)
        self.environment, self.random = environment, np.random.default_rng(seed)

    def select_action(self, state, legal_mask, epsilon=0.0) -> int:
        if epsilon != 0:
            raise ValueError("evaluation search must not explore")
        return int(self.searcher.search(self.environment.game, self.random).argmax())
