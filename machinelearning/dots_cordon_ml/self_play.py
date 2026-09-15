"""Training episodes and reproducible evaluation against a random policy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .dqn import ReplayBuffer, Transition
from .encoding import encode_state, legal_action_mask
from .environment import StepResult
from .proto import game_pb2


class ActionSelector(Protocol):
    def select_action(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
        epsilon: float,
    ) -> int: ...


class Environment(Protocol):
    def reset(self) -> game_pb2.GameState: ...

    def step(self, action: int) -> StepResult: ...


Opponent = Literal["self-play", "random", "frozen"]


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
    opponent: Opponent
    learner_player: int | None


@dataclass(frozen=True, slots=True)
class MatchStats:
    games: int
    wins: int
    draws: int
    losses: int
    mean_score_difference: float
    # Older callers can still construct summaries from means. Real evaluations
    # retain the integer total so persistence and combination never round scores.
    score_difference_sum: int | None = None

    @property
    def match_score(self) -> float:
        """Return match points per game, where a draw is worth half a win."""
        if not self.games:
            return float("nan")
        return (self.wins + 0.5 * self.draws) / self.games


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    overall: MatchStats
    as_player_0: MatchStats
    as_player_1: MatchStats

    # Preserve the original convenient fields used by the training logger.
    @property
    def wins(self) -> int:
        return self.overall.wins

    @property
    def draws(self) -> int:
        return self.overall.draws

    @property
    def losses(self) -> int:
        return self.overall.losses

    @property
    def mean_score_difference(self) -> float:
        return self.overall.mean_score_difference


@dataclass(slots=True)
class _MatchAccumulator:
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    score_difference_sum: int = 0

    def record(self, difference: int) -> None:
        self.games += 1
        self.score_difference_sum += difference
        if difference > 0:
            self.wins += 1
        elif difference < 0:
            self.losses += 1
        else:
            self.draws += 1

    def result(self) -> MatchStats:
        mean = self.score_difference_sum / self.games if self.games else float("nan")
        return MatchStats(
            games=self.games,
            wins=self.wins,
            draws=self.draws,
            losses=self.losses,
            mean_score_difference=mean,
            score_difference_sum=self.score_difference_sum,
        )


def collect_self_play_episode(
    environment: Environment,
    agent: ActionSelector,
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
            add(_continuing_transition(previous, state, legal_mask))
            pending[player] = None

        action = agent.select_action(state, legal_mask, epsilon)
        step = environment.step(action)
        opponent = 1 - player
        if pending[opponent] is not None:
            pending[opponent].reward -= step.reward
        pending[player] = _PendingTransition(state, action, step.reward)
        game = step.game

    final_scores = _scores(game)
    final_mask = legal_action_mask(game)
    for player, previous in enumerate(pending):
        if previous is None:
            continue
        previous.reward += _win_bonus(final_scores, player, terminal_win_bonus)
        add(_terminal_transition(previous, game, player, final_mask))

    return EpisodeResult(
        moves=int(game.turn),
        scores=final_scores,
        transitions=transition_count,
        opponent="self-play",
        learner_player=None,
    )


def collect_against_random_episode(
    environment: Environment,
    agent: ActionSelector,
    replay: ReplayBuffer,
    epsilon: float,
    random: np.random.Generator,
    learner_player: int,
    terminal_win_bonus: float = 1.0,
    on_transition: Callable[[], None] | None = None,
) -> EpisodeResult:
    """Train one player against a uniform-random opponent.

    Only the learner's decisions enter replay. Each transition still includes
    the random opponent's reply, using learner captures minus reply captures as
    its reward.
    """
    if learner_player not in (0, 1):
        raise ValueError(f"learner_player must be 0 or 1, got {learner_player}")

    game = environment.reset()
    pending: _PendingTransition | None = None
    transition_count = 0

    def add(transition: Transition) -> None:
        nonlocal transition_count
        replay.add(transition)
        transition_count += 1
        if on_transition is not None:
            on_transition()

    while not game.terminal:
        legal_mask = legal_action_mask(game)
        if game.current_player == learner_player:
            state = encode_state(game, learner_player)
            if pending is not None:
                add(_continuing_transition(pending, state, legal_mask))
            action = agent.select_action(state, legal_mask, epsilon)
            step = environment.step(action)
            pending = _PendingTransition(state, action, step.reward)
        else:
            action = int(random.choice(np.flatnonzero(legal_mask)))
            step = environment.step(action)
            if pending is not None:
                pending.reward -= step.reward
        game = step.game

    final_scores = _scores(game)
    if pending is not None:
        pending.reward += _win_bonus(final_scores, learner_player, terminal_win_bonus)
        add(
            _terminal_transition(
                pending,
                game,
                learner_player,
                legal_action_mask(game),
            )
        )

    return EpisodeResult(
        moves=int(game.turn),
        scores=final_scores,
        transitions=transition_count,
        opponent="random",
        learner_player=learner_player,
    )


def collect_against_agent_episode(
    environment: Environment,
    agent: ActionSelector,
    opponent: ActionSelector,
    replay: ReplayBuffer,
    epsilon: float,
    learner_player: int,
    opening_random: np.random.Generator | None = None,
    opening_random_moves: int = 0,
    terminal_win_bonus: float = 1.0,
    on_transition: Callable[[], None] | None = None,
) -> EpisodeResult:
    """Train one player against a frozen greedy agent.

    Only the learner's decisions enter replay. Optional random opening moves
    happen before either policy acts and do not enter replay. The frozen
    opponent observes the board from its own perspective and never explores or
    receives updates.
    """
    if learner_player not in (0, 1):
        raise ValueError(f"learner_player must be 0 or 1, got {learner_player}")
    if opening_random_moves < 0:
        raise ValueError("opening_random_moves must be non-negative")
    if opening_random_moves and opening_random is None:
        raise ValueError("opening_random is required when opening_random_moves > 0")

    game = environment.reset()
    if opening_random is not None:
        for _ in range(opening_random_moves):
            if game.terminal:
                break
            legal_mask = legal_action_mask(game)
            action = int(opening_random.choice(np.flatnonzero(legal_mask)))
            game = environment.step(action).game

    pending: _PendingTransition | None = None
    transition_count = 0

    def add(transition: Transition) -> None:
        nonlocal transition_count
        replay.add(transition)
        transition_count += 1
        if on_transition is not None:
            on_transition()

    while not game.terminal:
        legal_mask = legal_action_mask(game)
        if game.current_player == learner_player:
            state = encode_state(game, learner_player)
            if pending is not None:
                add(_continuing_transition(pending, state, legal_mask))
            action = agent.select_action(state, legal_mask, epsilon)
            step = environment.step(action)
            pending = _PendingTransition(state, action, step.reward)
        else:
            action = opponent.select_action(
                encode_state(game, game.current_player),
                legal_mask,
                epsilon=0.0,
            )
            step = environment.step(action)
            if pending is not None:
                pending.reward -= step.reward
        game = step.game

    final_scores = _scores(game)
    if pending is not None:
        pending.reward += _win_bonus(final_scores, learner_player, terminal_win_bonus)
        add(
            _terminal_transition(
                pending,
                game,
                learner_player,
                legal_action_mask(game),
            )
        )

    return EpisodeResult(
        moves=int(game.turn),
        scores=final_scores,
        transitions=transition_count,
        opponent="frozen",
        learner_player=learner_player,
    )


def random_game_seeds(games: int, seed: int) -> tuple[int, ...]:
    """Create reproducible paired seeds for swapped-seat evaluations."""
    if games <= 0:
        raise ValueError("games must be positive")
    random = np.random.default_rng(seed)
    pair_count = (games + 1) // 2
    values = random.integers(
        0,
        np.iinfo(np.int64).max,
        size=pair_count,
        dtype=np.int64,
    )
    paired = [int(value) for value in values for _ in range(2)]
    return tuple(paired[:games])


def evaluate_against_random(
    environment: Environment,
    agent: ActionSelector,
    game_seeds: Sequence[int],
) -> EvaluationResult:
    """Evaluate greedily on a reproducible suite, alternating model seats."""
    if not game_seeds:
        raise ValueError("at least one game seed is required")

    overall = _MatchAccumulator()
    by_player = (_MatchAccumulator(), _MatchAccumulator())

    for game_index, game_seed in enumerate(game_seeds):
        model_player = game_index % 2
        opponent_random = np.random.default_rng(game_seed)
        game = environment.reset()
        while not game.terminal:
            mask = legal_action_mask(game)
            if game.current_player == model_player:
                action = agent.select_action(
                    encode_state(game, model_player), mask, epsilon=0.0
                )
            else:
                action = int(opponent_random.choice(np.flatnonzero(mask)))
            game = environment.step(action).game

        difference = int(game.scores[model_player]) - int(game.scores[1 - model_player])
        overall.record(difference)
        by_player[model_player].record(difference)

    return EvaluationResult(
        overall=overall.result(),
        as_player_0=by_player[0].result(),
        as_player_1=by_player[1].result(),
    )


def evaluate_head_to_head(
    environment: Environment,
    candidate_a: ActionSelector,
    candidate_b: ActionSelector,
    game_seeds: Sequence[int],
    opening_random_moves: int,
) -> EvaluationResult:
    """Evaluate candidate A against candidate B with paired swapped seats.

    Each adjacent pair uses the same random opening. After the opening both
    candidates act greedily, so the opponent is always the other DQN rather
    than a random policy.
    """
    if not game_seeds:
        raise ValueError("at least one game seed is required")
    if opening_random_moves < 0:
        raise ValueError("opening_random_moves must be non-negative")

    overall = _MatchAccumulator()
    by_player = (_MatchAccumulator(), _MatchAccumulator())

    for game_index, game_seed in enumerate(game_seeds):
        candidate_a_player = game_index % 2
        opening_random = np.random.default_rng(game_seed)
        game = environment.reset()
        opening_move = 0

        while not game.terminal:
            mask = legal_action_mask(game)
            if opening_move < opening_random_moves:
                action = int(opening_random.choice(np.flatnonzero(mask)))
                opening_move += 1
            else:
                agent = (
                    candidate_a
                    if game.current_player == candidate_a_player
                    else candidate_b
                )
                action = agent.select_action(
                    encode_state(game, game.current_player),
                    mask,
                    epsilon=0.0,
                )
            game = environment.step(action).game

        difference = int(game.scores[candidate_a_player]) - int(
            game.scores[1 - candidate_a_player]
        )
        overall.record(difference)
        by_player[candidate_a_player].record(difference)

    return EvaluationResult(
        overall=overall.result(),
        as_player_0=by_player[0].result(),
        as_player_1=by_player[1].result(),
    )


def combine_evaluation_results(
    results: Sequence[EvaluationResult],
) -> EvaluationResult:
    """Combine disjoint evaluation suites without averaging their averages."""
    if not results:
        raise ValueError("at least one evaluation result is required")

    def combine(stats: Sequence[MatchStats]) -> MatchStats:
        games = sum(item.games for item in stats)
        if not games:
            raise ValueError("evaluation results must contain games")
        exact_totals = [
            item.score_difference_sum
            for item in stats
            if item.score_difference_sum is not None
        ]
        exact_total = sum(exact_totals) if len(exact_totals) == len(stats) else None
        total = (
            exact_total
            if exact_total is not None
            else sum(item.mean_score_difference * item.games for item in stats)
        )
        return MatchStats(
            games=games,
            wins=sum(item.wins for item in stats),
            draws=sum(item.draws for item in stats),
            losses=sum(item.losses for item in stats),
            mean_score_difference=total / games,
            score_difference_sum=exact_total,
        )

    return EvaluationResult(
        overall=combine([item.overall for item in results]),
        as_player_0=combine([item.as_player_0 for item in results]),
        as_player_1=combine([item.as_player_1 for item in results]),
    )


def _continuing_transition(
    previous: _PendingTransition,
    next_state: np.ndarray,
    next_legal_mask: np.ndarray,
) -> Transition:
    return Transition(
        state=previous.state,
        action=previous.action,
        reward=previous.reward,
        next_state=next_state,
        next_legal_mask=next_legal_mask,
        done=False,
    )


def _terminal_transition(
    previous: _PendingTransition,
    game: game_pb2.GameState,
    player: int,
    final_mask: np.ndarray,
) -> Transition:
    return Transition(
        state=previous.state,
        action=previous.action,
        reward=previous.reward,
        next_state=encode_state(game, player),
        next_legal_mask=final_mask,
        done=True,
    )


def _scores(game: game_pb2.GameState) -> tuple[int, int]:
    return (int(game.scores[0]), int(game.scores[1]))


def _win_bonus(scores: tuple[int, int], player: int, amount: float) -> float:
    difference = scores[player] - scores[1 - player]
    if difference > 0:
        return amount
    if difference < 0:
        return -amount
    return 0.0
