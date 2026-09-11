from __future__ import annotations

import numpy as np

from dots_cordon_ml.dqn import ReplayBuffer
from dots_cordon_ml.environment import StepResult
from dots_cordon_ml.proto import game_pb2
from dots_cordon_ml.self_play import collect_self_play_episode


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
        self, _state: np.ndarray, legal_mask: np.ndarray, _epsilon: float
    ) -> int:
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
        ScriptedEnvironment(),  # type: ignore[arg-type]
        FirstLegalAgent(),  # type: ignore[arg-type]
        replay,
        epsilon=0.5,
        terminal_win_bonus=1.0,
    )

    transitions = list(replay)
    assert result.moves == 3
    assert result.scores == (1, 2)
    assert result.transitions == 3
    assert [item.reward for item in transitions] == [-1.0, -1.0, 3.0]
    assert [item.done for item in transitions] == [False, True, True]

