"""Self-play data collection and evaluation against a random policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .dqn import DQNAgent, ReplayBuffer, Transition
from .encoding import encode_state, legal_action_mask
from .environment import GameEnvironment


@dataclass(slots=True)
class _PendingTransition:
    state: np.ndarray
    action: int
    reward: float


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    moves: int
    scores: tuple[int, int]
    transitions: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    wins: int
    draws: int
    losses: int
    mean_score_difference: float


def collect_self_play_episode(
    environment: GameEnvironment,
    agent: DQNAgent,
    replay: ReplayBuffer,
    epsilon: float,
    terminal_win_bonus: float = 1.0,
    on_transition: Callable[[], None] | None = None,
) -> EpisodeResult:
    """Play both sides and add player-to-same-player transitions to replay.

    A transition spans the opponent's reply. Its reward is the acting player's
    captured points minus points captured by that reply. This makes the next
    state another decision for the same player and keeps the Bellman update in
    one canonical perspective.
    """
    game = environment.reset()
    pending: list[_PendingTransition | None] = [None, None]
    transition_count = 0

    def add(transition: Transition) -> None:
        nonlocal transition_count
        replay.add(transition)
        transition_count += 1
        if on_transition is not None:
            on_transition()

    while not game.terminal:
        player = game.current_player
        state = encode_state(game, player)
        legal_mask = legal_action_mask(game)

        previous = pending[player]
        if previous is not None:
            add(
                Transition(
                    state=previous.state,
                    action=previous.action,
                    reward=previous.reward,
                    next_state=state,
                    next_legal_mask=legal_mask,
                    done=False,
                )
            )
            pending[player] = None

        action = agent.select_action(state, legal_mask, epsilon)
        step = environment.step(action)
        opponent = 1 - player
        if pending[opponent] is not None:
            pending[opponent].reward -= step.reward
        pending[player] = _PendingTransition(state, action, step.reward)
        game = step.game

    final_scores = (int(game.scores[0]), int(game.scores[1]))
    final_masks = legal_action_mask(game)
    for player, previous in enumerate(pending):
        if previous is None:
            continue
        score_difference = final_scores[player] - final_scores[1 - player]
        if score_difference > 0:
            previous.reward += terminal_win_bonus
        elif score_difference < 0:
            previous.reward -= terminal_win_bonus
        add(
            Transition(
                state=previous.state,
                action=previous.action,
                reward=previous.reward,
                next_state=encode_state(game, player),
                next_legal_mask=final_masks,
                done=True,
            )
        )

    return EpisodeResult(
        moves=int(game.turn),
        scores=final_scores,
        transitions=transition_count,
    )


def evaluate_against_random(
    environment: GameEnvironment,
    agent: DQNAgent,
    games: int,
    random: np.random.Generator,
) -> EvaluationResult:
    wins = draws = losses = 0
    score_differences: list[int] = []

    for game_index in range(games):
        model_player = game_index % 2
        game = environment.reset()
        while not game.terminal:
            mask = legal_action_mask(game)
            if game.current_player == model_player:
                action = agent.select_action(
                    encode_state(game, model_player), mask, epsilon=0.0
                )
            else:
                action = int(random.choice(np.flatnonzero(mask)))
            game = environment.step(action).game

        difference = int(game.scores[model_player]) - int(game.scores[1 - model_player])
        score_differences.append(difference)
        if difference > 0:
            wins += 1
        elif difference < 0:
            losses += 1
        else:
            draws += 1

    return EvaluationResult(
        wins=wins,
        draws=draws,
        losses=losses,
        mean_score_difference=float(np.mean(score_differences)),
    )

