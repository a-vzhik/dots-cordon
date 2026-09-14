from __future__ import annotations

from pathlib import Path

import torch
from pytest import MonkeyPatch

import dots_cordon_ml.champion_loop as champion_loop
from dots_cordon_ml.checkpoint import CheckpointMetadata
from dots_cordon_ml.champion_loop import (
    EvaluatedCheckpoint,
    _initial_training_seed,
    _passes_promotion,
    _passes_random_screen,
    _round_seeds,
    parse_args,
)
from dots_cordon_ml.self_play import EvaluationResult, MatchStats


def result(wins: int, draws: int, losses: int, difference: float) -> EvaluationResult:
    stats = MatchStats(wins + draws + losses, wins, draws, losses, difference)
    return EvaluationResult(stats, stats, stats)


def evaluated(
    episode: int,
    wins: int,
    draws: int,
    losses: int,
    difference: float,
) -> EvaluatedCheckpoint:
    evaluation = result(wins, draws, losses, difference)
    metadata = CheckpointMetadata(7, 7, 64, 3, episode, 0, 0)
    return EvaluatedCheckpoint(
        Path(f"dqn-{episode}.pt"), metadata, (evaluation,), evaluation
    )


def test_defaults_describe_four_candidate_thousand_episode_round() -> None:
    args = parse_args([])

    assert args.candidate_count == 4
    assert args.candidate_interval == 250
    assert args.training_seed is None
    assert not args.fresh_training_rng
    assert args.training_opponent == "frozen"
    assert args.screen_max_regression == 0.003
    assert args.screen_suites == 3
    assert args.screen_games == 1_000
    assert args.head_to_head_suites == 3
    assert args.head_to_head_games == 1_000


def test_fresh_training_rng_parameter_is_available() -> None:
    args = parse_args(["--fresh-training-rng", "--training-seed", "19"])

    assert args.fresh_training_rng
    assert args.training_seed == 19


def test_training_seed_defaults_to_unix_milliseconds(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(champion_loop.time, "time_ns", lambda: 1_234_567_890_000_000)

    assert _initial_training_seed(None, round_number=8) == 1_234_567_890


def test_explicit_training_seed_accounts_for_existing_rounds() -> None:
    assert _initial_training_seed(19, round_number=8) == 26


def test_round_seeds_are_fresh_and_non_overlapping() -> None:
    assert _round_seeds(100, 1, 3) == (100, 101, 102)
    assert _round_seeds(100, 2, 3) == (103, 104, 105)


def test_random_screen_allows_at_most_point_zero_zero_three_regression() -> None:
    champion = evaluated(100, 980, 0, 20, 3.0)

    assert _passes_random_screen(
        evaluated(200, 985, 0, 15, 2.0), champion, 0.003
    )
    assert _passes_random_screen(
        evaluated(200, 977, 0, 23, 2.0), champion, 0.003
    )
    assert not _passes_random_screen(
        evaluated(200, 976, 0, 24, 4.0), champion, 0.003
    )


def test_promotion_requires_combined_score_and_suite_wins() -> None:
    passing = (
        result(55, 0, 45, 0.2),
        result(54, 0, 46, 0.1),
        result(47, 0, 53, -0.1),
    )
    one_suite_win = (
        result(60, 0, 40, 0.2),
        result(49, 0, 51, -0.1),
        result(49, 0, 51, -0.1),
    )

    assert _passes_promotion(passing, 0.52, 2)
    assert not _passes_promotion(one_suite_win, 0.52, 2)


def test_loop_promotes_a_screened_head_to_head_winner(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    champion_path = tmp_path / "champion.pt"
    candidate_path = tmp_path / "candidate.pt"

    def save_checkpoint(path: Path, episode: int, marker: int) -> None:
        torch.save(
            {
                "online": {"marker": torch.tensor(marker)},
                "training_state": {
                    "episode": episode,
                    "environment_steps": 0,
                    "optimization_steps": 0,
                },
                "board": {"rows": 7, "columns": 7},
                "model": {"channels": 64, "blocks": 3},
            },
            path,
        )

    save_checkpoint(champion_path, episode=100, marker=1)
    save_checkpoint(candidate_path, episode=350, marker=2)

    class FakeEnvironment:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeEnvironment:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    def fake_training(*_args: object, **_kwargs: object) -> tuple[Path, ...]:
        return (candidate_path,)

    def fake_screen(
        _environment: object,
        path: Path,
        _device: object,
        _suite_seeds: object,
        _games: int,
    ) -> EvaluatedCheckpoint:
        if path.name == "champion-before.pt":
            item = evaluated(100, 50, 0, 50, 0.0)
        else:
            item = evaluated(350, 60, 0, 40, 0.5)
        return EvaluatedCheckpoint(path, item.metadata, item.suites, item.aggregate)

    def fake_head_to_head(*_args: object, **_kwargs: object) -> tuple[EvaluationResult, ...]:
        return (
            result(55, 0, 45, 0.2),
            result(54, 0, 46, 0.1),
            result(53, 0, 47, 0.1),
        )

    monkeypatch.setattr(champion_loop, "GameEnvironment", FakeEnvironment)
    monkeypatch.setattr(champion_loop, "_run_training_round", fake_training)
    monkeypatch.setattr(
        champion_loop,
        "_evaluate_against_random_suites",
        fake_screen,
    )
    monkeypatch.setattr(
        champion_loop,
        "_evaluate_head_to_head_suites",
        fake_head_to_head,
    )

    args = parse_args(
        [
            "--champion",
            str(champion_path),
            "--run-dir",
            str(tmp_path / "rounds"),
            "--candidate-count",
            "1",
            "--max-rounds",
            "1",
        ]
    )

    assert champion_loop.run(args) == 0
    promoted, promoted_metadata = champion_loop.read_checkpoint(
        champion_path, map_location="cpu"
    )
    assert promoted_metadata.episode == 350
    assert int(promoted["online"]["marker"]) == 2
    assert (tmp_path / "rounds/round-001-from-0000100/results.json").is_file()
