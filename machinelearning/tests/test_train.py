from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dots_cordon_ml.dqn import DQNAgent, ReplayBuffer
from dots_cordon_ml.self_play import EvaluationResult, MatchStats
from dots_cordon_ml.train import (
    EvaluationMonitor,
    _capture_rng_state,
    _rng_source,
    _restore_rng_state,
    parse_args,
)


def evaluation(
    wins: int,
    draws: int,
    losses: int,
    mean_difference: float,
) -> EvaluationResult:
    stats = MatchStats(
        games=wins + draws + losses,
        wins=wins,
        draws=draws,
        losses=losses,
        mean_score_difference=mean_difference,
    )
    return EvaluationResult(
        overall=stats,
        as_player_0=stats,
        as_player_1=stats,
    )


def agent(seed: int) -> DQNAgent:
    return DQNAgent(
        device=torch.device("cpu"),
        learning_rate=1e-3,
        gamma=0.99,
        seed=seed,
        channels=8,
        blocks=0,
    )


def test_resume_rng_can_be_restored_or_reset_from_seed() -> None:
    restored = parse_args(["--resume", "checkpoint.pt"])
    reset = parse_args(
        ["--resume", "checkpoint.pt", "--reset-rng-on-resume", "--seed", "19"]
    )

    assert _rng_source(restored, {"saved": True}) == "checkpoint"
    assert _rng_source(reset, {"saved": True}) == "fresh-seed:19"


def test_monitor_saves_small_best_but_does_not_reset_patience() -> None:
    monitor = EvaluationMonitor(patience=2, min_delta=0.005)

    assert monitor.observe(100, evaluation(190, 0, 10, 3.0))
    assert monitor.observe(200, evaluation(190, 1, 9, 3.1))
    assert monitor.best_episode == 200
    assert monitor.evaluations_without_improvement == 1

    assert not monitor.observe(300, evaluation(189, 1, 10, 3.2))
    assert monitor.should_stop


def test_monitor_resets_patience_after_meaningful_improvement() -> None:
    monitor = EvaluationMonitor(patience=2, min_delta=0.005)

    monitor.observe(100, evaluation(190, 0, 10, 3.0))
    monitor.observe(200, evaluation(189, 2, 9, 3.1))
    assert monitor.evaluations_without_improvement == 1

    assert monitor.observe(300, evaluation(191, 0, 9, 3.2))
    assert monitor.evaluations_without_improvement == 0
    assert not monitor.should_stop


def test_training_random_state_round_trips_through_weights_only_checkpoint(
    tmp_path: Path,
) -> None:
    source_agent = agent(seed=1)
    source_replay = ReplayBuffer(capacity=10, seed=2)
    source_selection = np.random.default_rng(3)
    source_opponent = np.random.default_rng(4)
    source_agent.random.random()
    source_selection.random()
    source_opponent.random()
    saved = _capture_rng_state(
        source_agent,
        source_replay,
        source_selection,
        source_opponent,
        random_opponent_episodes=7,
    )
    path = tmp_path / "rng.pt"
    torch.save({"rng_state": saved}, path)
    loaded = torch.load(path, weights_only=True)["rng_state"]

    expected_agent_value = source_agent.random.random()
    expected_selection_value = source_selection.random()
    expected_opponent_value = source_opponent.random()

    destination_agent = agent(seed=10)
    destination_replay = ReplayBuffer(capacity=10, seed=11)
    destination_selection = np.random.default_rng(12)
    destination_opponent = np.random.default_rng(13)
    random_opponent_episodes, frozen_opponent_episodes = _restore_rng_state(
        loaded,
        destination_agent,
        destination_replay,
        destination_selection,
        destination_opponent,
    )

    assert random_opponent_episodes == 7
    assert frozen_opponent_episodes == 0
    assert destination_agent.random.random() == expected_agent_value
    assert destination_selection.random() == expected_selection_value
    assert destination_opponent.random() == expected_opponent_value
    assert destination_replay.random_state() == source_replay.random_state()
