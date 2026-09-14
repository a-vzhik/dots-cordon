from __future__ import annotations

import numpy as np

from dots_cordon_ml.dqn import ReplayBuffer
from dots_cordon_ml.environment import StepResult
from dots_cordon_ml.proto import game_pb2
from dots_cordon_ml.self_play import (
    EvaluationResult,
    MatchStats,
    collect_against_agent_episode,
    collect_against_random_episode,
    collect_self_play_episode,
    combine_evaluation_results,
    evaluate_against_random,
    evaluate_head_to_head,
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


class FixedActionAgent:
    def __init__(self, action: int) -> None:
        self.action = action
        self.calls = 0

    def select_action(
        self, _state: np.ndarray, legal_mask: np.ndarray, epsilon: float
    ) -> int:
        assert epsilon == 0.0
        assert legal_mask[self.action]
        self.calls += 1
        return self.action


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


def test_frozen_opponent_episode_records_only_learner_transitions() -> None:
    replay = ReplayBuffer(capacity=10, seed=1)
    opponent = FixedActionAgent(1)
    result = collect_against_agent_episode(
        ScriptedEnvironment(),
        FirstLegalAgent(),
        opponent,
        replay,
        epsilon=0.5,
        learner_player=0,
        terminal_win_bonus=1.0,
    )

    transitions = list(replay)
    assert result.transitions == 2
    assert result.opponent == "frozen"
    assert result.learner_player == 0
    assert opponent.calls == 1
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


def test_evaluation_results_are_combined_by_game_count() -> None:
    first_stats = MatchStats(2, 1, 1, 0, 2.0)
    second_stats = MatchStats(4, 2, 0, 2, -1.0)
    first = EvaluationResult(first_stats, first_stats, first_stats)
    second = EvaluationResult(second_stats, second_stats, second_stats)

    combined = combine_evaluation_results((first, second))

    assert combined.overall.games == 6
    assert (combined.wins, combined.draws, combined.losses) == (3, 1, 2)
    assert combined.overall.match_score == 3.5 / 6
    assert combined.mean_score_difference == 0.0


class OneMoveHeadToHeadEnvironment:
    def __init__(self) -> None:
        self.actions: list[int] = []

    def reset(self) -> game_pb2.GameState:
        return game_pb2.GameState(
            board=game_pb2.Board(rows=1, columns=2, cells=b"\x00\x00"),
            scores=(0, 0),
            current_player=0,
        )

    def step(self, action: int) -> StepResult:
        self.actions.append(action)
        scores = (2, 0) if action == 0 else (0, 2)
        cells = b"\x01\x00" if action == 0 else b"\x00\x01"
        return StepResult(
            game=game_pb2.GameState(
                board=game_pb2.Board(rows=1, columns=2, cells=cells),
                scores=scores,
                current_player=1,
                turn=1,
                terminal=True,
            ),
            player=0,
            reward=0,
        )


def test_head_to_head_swaps_candidate_seats() -> None:
    environment = OneMoveHeadToHeadEnvironment()
    candidate_a = FixedActionAgent(0)
    candidate_b = FixedActionAgent(1)

    result = evaluate_head_to_head(
        environment,
        candidate_a,
        candidate_b,
        random_game_seeds(2, seed=20),
        opening_random_moves=0,
    )

    assert (result.wins, result.draws, result.losses) == (2, 0, 0)
    assert result.mean_score_difference == 2.0
    assert result.as_player_0.wins == 1
    assert result.as_player_1.wins == 1
    assert candidate_a.calls == 1
    assert candidate_b.calls == 1


def test_head_to_head_reuses_opening_for_swapped_seat_pair() -> None:
    environment = OneMoveHeadToHeadEnvironment()
    candidate_a = FixedActionAgent(0)
    candidate_b = FixedActionAgent(1)

    evaluate_head_to_head(
        environment,
        candidate_a,
        candidate_b,
        random_game_seeds(2, seed=21),
        opening_random_moves=1,
    )

    assert environment.actions[0] == environment.actions[1]
    assert candidate_a.calls == 0
    assert candidate_b.calls == 0
