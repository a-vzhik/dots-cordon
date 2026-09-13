from __future__ import annotations

import numpy as np

from dots_cordon_ml.dqn import ReplayBuffer
from dots_cordon_ml.environment import StepResult
from dots_cordon_ml.proto import game_pb2
from dots_cordon_ml.self_play import (
    collect_against_random_episode,
    collect_self_play_episode,
    evaluate_against_random,
    random_game_seeds,
)


def state(
    cells: list[int],
    *,
    turn: int,
    player: int,
    scores: tuple[int, int] = (0, 0),
    terminal: bool = False,
) -> game_pb2.GameState:
    return game_pb2.GameState(
        board=game_pb2.Board(rows=1, columns=3, cells=bytes(cells)),
        scores=scores,
        current_player=player,
        turn=turn,
        terminal=terminal,
    )


class FirstLegalAgent:
    def select_action(
        self, _state: np.ndarray, legal_mask: np.ndarray, epsilon: float
    ) -> int:
        del epsilon
        return int(np.flatnonzero(legal_mask)[0])


class ScriptedEnvironment:
    def __init__(self) -> None:
        self.initial = state([0, 0, 0], turn=0, player=0)
        self.steps = iter(
            [
                StepResult(
                    state([1, 0, 0], turn=1, player=1, scores=(1, 0)),
                    player=0,
                    reward=1.0,
                ),
                StepResult(
                    state([1, 2, 0], turn=2, player=0, scores=(1, 2)),
                    player=1,
                    reward=2.0,
                ),
                StepResult(
                    state(
                        [1, 2, 1],
                        turn=3,
                        player=1,
                        scores=(1, 2),
                        terminal=True,
                    ),
                    player=0,
                    reward=0.0,
                ),
            ]
        )

    def reset(self) -> game_pb2.GameState:
        return self.initial

    def step(self, _action: int) -> StepResult:
        return next(self.steps)


def test_self_play_transitions_include_opponent_reply_and_terminal_result() -> None:
    replay = ReplayBuffer(capacity=10, seed=1)
    result = collect_self_play_episode(
        ScriptedEnvironment(),
        FirstLegalAgent(),
        replay,
        epsilon=0.5,
        terminal_win_bonus=1.0,
    )

    transitions = list(replay)
    assert result.moves == 3
    assert result.scores == (1, 2)
    assert result.transitions == 3
    assert result.opponent == "self-play"
    assert result.learner_player is None
    assert [item.reward for item in transitions] == [-1.0, -1.0, 3.0]
    assert [item.done for item in transitions] == [False, True, True]


def test_random_opponent_episode_records_only_learner_transitions() -> None:
    replay = ReplayBuffer(capacity=10, seed=1)
    result = collect_against_random_episode(
        ScriptedEnvironment(),
        FirstLegalAgent(),
        replay,
        epsilon=0.5,
        random=np.random.default_rng(4),
        learner_player=0,
        terminal_win_bonus=1.0,
    )

    transitions = list(replay)
    assert result.transitions == 2
    assert result.opponent == "random"
    assert result.learner_player == 0
    assert [item.reward for item in transitions] == [-1.0, -1.0]
    assert [item.done for item in transitions] == [False, True]


class OneMoveEvaluationEnvironment:
    def __init__(self) -> None:
        self._scores = iter([(2, 0), (1, 1), (0, 1), (0, 3)])

    def reset(self) -> game_pb2.GameState:
        return game_pb2.GameState(
            board=game_pb2.Board(rows=1, columns=1, cells=b"\x00"),
            scores=(0, 0),
            current_player=0,
        )

    def step(self, _action: int) -> StepResult:
        scores = next(self._scores)
        return StepResult(
            game=game_pb2.GameState(
                board=game_pb2.Board(rows=1, columns=1, cells=b"\x01"),
                scores=scores,
                current_player=1,
                turn=1,
                terminal=True,
            ),
            player=0,
            reward=0,
        )


def test_evaluation_reports_reproducible_overall_and_seat_results() -> None:
    seeds = random_game_seeds(4, seed=99)
    assert seeds == random_game_seeds(4, seed=99)
    assert seeds != random_game_seeds(4, seed=100)
    assert seeds[0] == seeds[1]
    assert seeds[2] == seeds[3]

    result = evaluate_against_random(
        OneMoveEvaluationEnvironment(), FirstLegalAgent(), seeds
    )

    assert (result.wins, result.draws, result.losses) == (2, 1, 1)
    assert result.mean_score_difference == 1.0
    assert result.overall.match_score == 0.625
    assert (result.as_player_0.wins, result.as_player_0.losses) == (1, 1)
    assert result.as_player_0.mean_score_difference == 0.5
    assert (result.as_player_1.wins, result.as_player_1.draws) == (1, 1)
    assert result.as_player_1.mean_score_difference == 1.5
