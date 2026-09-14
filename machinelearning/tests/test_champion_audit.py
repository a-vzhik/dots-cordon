from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Event

import pytest
import torch

from dots_cordon_ml import champion_loop, train
from dots_cordon_ml.audit import AuditService, Database
from dots_cordon_ml.checkpoint import read_checkpoint
from dots_cordon_ml.dqn import DQNAgent
from dots_cordon_ml.self_play import (
    EpisodeResult,
    EvaluationResult,
    MatchStats,
    combine_evaluation_results,
)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[tuple[str, AuditService]]:
    url = f"sqlite:///{tmp_path / 'audit.sqlite3'}"
    db = Database(url)
    db.upgrade()
    db.close()
    service = AuditService(url)
    try:
        yield url, service
    finally:
        service.close()


def initial_champion(path: Path) -> None:
    agent = DQNAgent(
        device=torch.device("cpu"),
        learning_rate=3e-4,
        gamma=0.99,
        seed=11,
        channels=8,
        blocks=0,
    )
    torch.save(
        {
            "board": {"rows": 3, "columns": 3},
            "model": {"channels": 8, "blocks": 0},
            "training_state": {
                "episode": 0,
                "environment_steps": 0,
                "optimization_steps": 0,
            },
            "online": agent.online.state_dict(),
            "target": agent.target.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
        },
        path,
    )


def result(wins_per_seat: int) -> EvaluationResult:
    losses = 2 - wins_per_seat
    difference = wins_per_seat - losses
    seat = MatchStats(2, wins_per_seat, 0, losses, difference / 2, difference)
    overall = MatchStats(
        4, wins_per_seat * 2, 0, losses * 2, difference / 2, difference * 2
    )
    return EvaluationResult(overall, seat, seat)


@pytest.fixture
def tiny_training(monkeypatch: pytest.MonkeyPatch) -> None:
    class Environment:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    def collect(_environment, agent, _replay, **_kwargs):
        # The trainer still creates real checkpoints, RNG state, and ancestry.
        with torch.no_grad():
            next(agent.online.parameters()).add_(0.01)
        return EpisodeResult(1, (1, 0), 0, "self-play", None)

    monkeypatch.setattr(train, "GameEnvironment", Environment)
    monkeypatch.setattr(train, "collect_self_play_episode", collect)


def arguments(url: str, champion: Path, run_dir: Path, *extra: str):
    return champion_loop.parse_args(
        [
            "--database-url",
            url,
            "--champion",
            str(champion),
            "--run-dir",
            str(run_dir),
            "--candidate-count",
            "3",
            "--candidate-interval",
            "1",
            "--max-rounds",
            "2",
            "--evaluation-workers",
            "1",
            "--training-seed",
            "23",
            "--fresh-training-rng",
            "--training-opponent",
            "self-play",
            "--random-opponent-probability",
            "0",
            "--training-log-every",
            "1",
            "--screen-games",
            "4",
            "--screen-suites",
            "2",
            "--screen-seed",
            "100",
            "--head-to-head-games",
            "4",
            "--head-to-head-suites",
            "2",
            "--head-to-head-seed",
            "200",
            "--opening-random-moves",
            "0",
            "--device",
            "cpu",
            *extra,
        ]
    )


def evaluated(path: Path, seeds: tuple[int, ...], wins: int):
    _, metadata = read_checkpoint(path, map_location="cpu")
    suites = tuple(result(wins) for _seed in seeds)
    return champion_loop.EvaluatedCheckpoint(
        path, metadata, suites, combine_evaluation_results(suites)
    )


def is_incumbent(service: AuditService, args, path: Path) -> bool:
    _, metadata = read_checkpoint(path, map_location="cpu")
    return (
        metadata.episode == service.get_attempt(args._audit_attempt_id)["start_episode"]
    )


def test_intermediate_champion_retains_tail_and_starts_new_branches_across_commands(
    tmp_path: Path,
    database: tuple[str, AuditService],
    tiny_training: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, service = database
    champion = tmp_path / "champion.pt"
    initial_champion(champion)

    def screen(args, path, _device, seeds, games):
        assert games == 4
        _, metadata = read_checkpoint(path, map_location="cpu")
        if is_incumbent(service, args, path):
            wins = 1 if metadata.episode == 0 else 2
        else:
            wins = {1: 1, 2: 2}.get(metadata.episode, 0)
        return evaluated(path, seeds, wins)

    def challenge(_args, candidate, incumbent, _device, _seed, _games, _opening):
        assert candidate.metadata.episode == 2
        assert incumbent.metadata.episode == 0
        return result(2)

    monkeypatch.setattr(champion_loop, "_evaluate_against_random_suites", screen)
    monkeypatch.setattr(champion_loop, "_evaluate_head_to_head_suite", challenge)
    assert champion_loop.run(arguments(url, champion, tmp_path / "rounds")) == 0

    experiment = service.get_experiment("default")
    attempts = service.list_attempts(experiment["id"])
    assert len(attempts) == 2
    first, second = attempts
    assert first["outcome"] == "promoted"
    assert second["outcome"] == "no_qualified_candidate"
    assert first["status"] == second["status"] == "completed"
    history = service.champion_history(experiment["id"])
    assert len(history) == 2
    promoted_id = history[-1]["checkpoint_id"]
    assert second["starting_checkpoint_id"] == promoted_id
    assert (second["start_episode"], second["target_episode"]) == (2, 5)

    first_checkpoints = service.list_checkpoints(first["id"])
    assert [item["episode"] for item in first_checkpoints] == [0, 1, 2, 3]
    by_episode = {item["episode"]: item for item in first_checkpoints}
    assert by_episode[2]["id"] == promoted_id
    assert by_episode[3]["parent_checkpoint_id"] == promoted_id
    assert by_episode[3]["is_final_in_attempt"]
    assert not by_episode[2]["is_final_in_attempt"]
    assert [item["candidate_index"] for item in first_checkpoints[1:]] == [1, 2, 3]
    assert [
        item["episode"] for item in first_checkpoints if item["is_best_in_attempt"]
    ] == [2]
    second_checkpoints = service.list_checkpoints(second["id"])
    assert second_checkpoints[0]["parent_checkpoint_id"] == promoted_id
    assert [item["episode"] for item in second_checkpoints] == [2, 3, 4, 5]
    assert second_checkpoints[1]["id"] != by_episode[3]["id"]
    # Every candidate lost its screen. The earliest of the tied candidates is
    # still the best within this attempt; the incumbent is not a candidate.
    assert [
        item["episode"] for item in second_checkpoints if item["is_best_in_attempt"]
    ] == [3]
    for chain in (first_checkpoints, second_checkpoints):
        for parent, child in zip(chain, chain[1:]):
            assert child["parent_checkpoint_id"] == parent["id"]

    decisions = service.list_decisions(first["id"])
    assert any(
        item["checkpoint_id"] == by_episode[1]["id"]
        and item["stage"] == "challenge"
        and item["result"] == "skipped"
        for item in decisions
    )
    assert any(
        item["checkpoint_id"] == by_episode[3]["id"]
        and item["stage"] == "screening"
        and item["result"] == "rejected"
        for item in decisions
    )
    assert read_checkpoint(champion, map_location="cpu")[1].episode == 2
    exported = service.export_checkpoint(promoted_id, tmp_path / "promoted.pt")
    assert champion.read_bytes() == exported.read_bytes()

    # A later invocation uses SQL even if the compatibility file was corrupted.
    champion.write_bytes(b"invalid stale compatibility checkpoint")
    assert champion_loop.run(arguments(url, champion, tmp_path / "another-dir")) == 0
    assert read_checkpoint(champion, map_location="cpu")[1].episode == 2
    attempts = service.list_attempts(experiment["id"])
    assert len(attempts) == 3
    assert attempts[-1]["starting_checkpoint_id"] == promoted_id
    assert len(service.champion_history(experiment["id"])) == 2
    batches = service.list_evaluations(experiment["id"])
    for index, attempt in enumerate(attempts):
        screening = [
            service.get_evaluation(item["id"])
            for item in batches
            if item["attempt_id"] == attempt["id"] and item["purpose"] == "screening"
        ]
        assert len(screening) == 4
        expected = [str(100 + index * 2), str(101 + index * 2)]
        for batch in screening:
            assert batch["status"] == "completed"
            assert [
                suite["definition"]["seed"] for suite in batch["suites"]
            ] == expected
        for suite_index in range(2):
            assert (
                len(
                    {
                        batch["suites"][suite_index]["suite_fingerprint"]
                        for batch in screening
                    }
                )
                == 1
            )
    assert service.get_experiment("default")["seed_offsets"] == {
        "screening": 6,
        "head_to_head": 6,
    }


def test_qualified_candidate_losing_challenge_retains_rejection_evidence(
    tmp_path: Path,
    database: tuple[str, AuditService],
    tiny_training: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, service = database
    champion = tmp_path / "champion.pt"
    initial_champion(champion)

    def screen(args, path, _device, seeds, _games):
        return evaluated(path, seeds, 1 if is_incumbent(service, args, path) else 2)

    monkeypatch.setattr(champion_loop, "_evaluate_against_random_suites", screen)
    monkeypatch.setattr(
        champion_loop, "_evaluate_head_to_head_suite", lambda *_: result(0)
    )
    args = arguments(url, champion, tmp_path / "rounds", "--candidate-count", "1")
    assert champion_loop.run(args) == 0

    experiment = service.get_experiment("default")
    (attempt,) = service.list_attempts(experiment["id"])
    assert attempt["outcome"] == "no_challenger_passed"
    assert attempt["status"] == "completed"
    assert [
        row["episode"]
        for row in service.list_checkpoints(attempt["id"])
        if row["is_best_in_attempt"]
    ] == [1]
    assert {
        (item["stage"], item["result"])
        for item in service.list_decisions(attempt["id"])
    } == {
        ("screening", "qualified"),
        ("challenge", "rejected"),
    }
    batches = service.list_evaluations(experiment["id"])
    (challenge,) = [item for item in batches if item["purpose"] == "head_to_head"]
    evidence = service.get_evaluation(challenge["id"])
    assert evidence["status"] == "completed"
    assert all(suite["player_0_losses"] == 2 for suite in evidence["suites"])
    assert len(service.champion_history(experiment["id"])) == 1
    assert read_checkpoint(champion, map_location="cpu")[1].episode == 0


def test_failed_challenge_preserves_completed_suite_without_promoting(
    tmp_path: Path,
    database: tuple[str, AuditService],
    tiny_training: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, service = database
    champion = tmp_path / "champion.pt"
    initial_champion(champion)
    first_suite_saved = Event()
    complete_suite = AuditService.complete_suite

    def record_then_notify(self, batch_id, index, stats):
        complete_suite(self, batch_id, index, stats)
        if self.get_evaluation(batch_id)["purpose"] == "head_to_head" and index == 0:
            first_suite_saved.set()

    def screen(args, path, _device, seeds, _games):
        return evaluated(path, seeds, 1 if is_incumbent(service, args, path) else 2)

    def challenge(_args, _candidate, _incumbent, _device, seed, _games, _opening):
        if seed == 201:
            assert first_suite_saved.wait(timeout=5)
            raise ConnectionError("second challenge suite disconnected")
        return result(2)

    monkeypatch.setattr(AuditService, "complete_suite", record_then_notify)
    monkeypatch.setattr(champion_loop, "_evaluate_against_random_suites", screen)
    monkeypatch.setattr(champion_loop, "_evaluate_head_to_head_suite", challenge)
    args = arguments(url, champion, tmp_path / "rounds", "--candidate-count", "1")
    with pytest.raises(ConnectionError, match="second challenge suite disconnected"):
        champion_loop.run(args)

    experiment = service.get_experiment("default")
    (attempt,) = service.list_attempts(experiment["id"])
    assert attempt["status"] == "failed"
    batches = service.list_evaluations(experiment["id"])
    (challenge,) = [item for item in batches if item["purpose"] == "head_to_head"]
    evidence = service.get_evaluation(challenge["id"])
    assert evidence["status"] == "failed"
    assert [suite["status"] for suite in evidence["suites"]] == ["completed", "failed"]
    assert evidence["suites"][0]["player_0_wins"] == 2
    assert evidence["suites"][1]["player_0_wins"] is None
    assert len(service.champion_history(experiment["id"])) == 1
    assert all(
        decision["stage"] == "screening"
        for decision in service.list_decisions(attempt["id"])
    )


def test_failed_compatibility_export_preserves_committed_champion(
    tmp_path: Path,
    database: tuple[str, AuditService],
    tiny_training: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, service = database
    champion = tmp_path / "champion.pt"
    initial_champion(champion)
    export_checkpoint = AuditService.export_checkpoint

    def fail_promoted_file(self, checkpoint_id, path):
        if (
            Path(path) == champion
            and self.get_checkpoint(checkpoint_id)["episode"] == 1
        ):
            raise OSError("compatibility export failed")
        return export_checkpoint(self, checkpoint_id, path)

    def screen(args, path, _device, seeds, _games):
        return evaluated(path, seeds, 1 if is_incumbent(service, args, path) else 2)

    monkeypatch.setattr(AuditService, "export_checkpoint", fail_promoted_file)
    monkeypatch.setattr(champion_loop, "_evaluate_against_random_suites", screen)
    monkeypatch.setattr(
        champion_loop, "_evaluate_head_to_head_suite", lambda *_: result(2)
    )
    args = arguments(
        url,
        champion,
        tmp_path / "rounds",
        "--candidate-count",
        "1",
        "--max-rounds",
        "1",
    )
    with pytest.raises(OSError, match="compatibility export failed"):
        champion_loop.run(args)

    experiment = service.get_experiment("default")
    promoted = service.current_champion(experiment["id"])
    assert service.get_checkpoint(promoted["checkpoint_id"])["episode"] == 1
    (attempt,) = service.list_attempts(experiment["id"])
    assert attempt["status"] == "completed"
    assert attempt["outcome"] == "promoted"
    assert read_checkpoint(champion, map_location="cpu")[1].episode == 0

    def reject_next_screen(args, path, _device, seeds, _games):
        return evaluated(path, seeds, 2 if is_incumbent(service, args, path) else 0)

    monkeypatch.setattr(AuditService, "export_checkpoint", export_checkpoint)
    monkeypatch.setattr(
        champion_loop, "_evaluate_against_random_suites", reject_next_screen
    )
    assert (
        champion_loop.run(
            arguments(url, champion, tmp_path / "another-dir", "--candidate-count", "1")
        )
        == 0
    )
    assert read_checkpoint(champion, map_location="cpu")[1].episode == 1
    assert (
        service.list_attempts(experiment["id"])[-1]["starting_checkpoint_id"]
        == promoted["checkpoint_id"]
    )


def test_screen_and_challenge_use_saved_blobs_after_round_files_are_overwritten(
    tmp_path: Path,
    database: tuple[str, AuditService],
    tiny_training: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, service = database
    champion = tmp_path / "champion.pt"
    initial_champion(champion)
    original_screen = champion_loop._evaluate_random_screen

    def overwrite_before_screen(args, paths, *rest, **kwargs):
        for path in paths:
            path.write_bytes(b"overwritten compatibility output")
        return original_screen(args, paths, *rest, **kwargs)

    def screen(args, path, _device, seeds, _games):
        return evaluated(path, seeds, 1 if is_incumbent(service, args, path) else 2)

    def challenge(_args, candidate, incumbent, _device, _seed, _games, _opening):
        assert read_checkpoint(candidate.path, map_location="cpu")[1].episode == 1
        assert read_checkpoint(incumbent.path, map_location="cpu")[1].episode == 0
        return result(2)

    monkeypatch.setattr(
        champion_loop, "_evaluate_random_screen", overwrite_before_screen
    )
    monkeypatch.setattr(champion_loop, "_evaluate_against_random_suites", screen)
    monkeypatch.setattr(champion_loop, "_evaluate_head_to_head_suite", challenge)
    args = arguments(
        url,
        champion,
        tmp_path / "rounds",
        "--candidate-count",
        "1",
        "--max-rounds",
        "1",
    )
    assert champion_loop.run(args) == 0
    assert read_checkpoint(champion, map_location="cpu")[1].episode == 1
    experiment = service.get_experiment("default")
    assert len(service.champion_history(experiment["id"])) == 2
