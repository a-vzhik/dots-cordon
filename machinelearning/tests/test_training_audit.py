from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import signal

import pytest
from pytest import MonkeyPatch
import torch

from dots_cordon_ml.audit import AuditService, Database
from dots_cordon_ml.self_play import EpisodeResult, EvaluationResult, MatchStats
import dots_cordon_ml.train as train


@pytest.fixture
def database_url(tmp_path: Path) -> str:
    url = f"sqlite:///{tmp_path / 'training.sqlite3'}"
    database = Database(url)
    try:
        database.upgrade()
    finally:
        database.close()
    return url


@pytest.fixture
def simulated_games(monkeypatch: MonkeyPatch) -> dict[str, int]:
    """Exercise real training/checkpoint persistence without opening an RPC."""
    progress = {"episodes": 0, "evaluations": 0}
    monkeypatch.setattr(train, "GameEnvironment", lambda **_: nullcontext(object()))

    def collect(*_: object, **__: object) -> EpisodeResult:
        progress["episodes"] += 1
        return EpisodeResult(2, (1, 0), 0, "self-play", None)

    def evaluate(*_: object, **__: object) -> EvaluationResult:
        progress["evaluations"] += 1
        difference = progress["evaluations"]
        # Two games, one per seat. Increasing score difference creates a new
        # best at each evaluation without changing the evaluation's game budget.
        seat = MatchStats(1, 1, 0, 0, float(difference), difference)
        overall = MatchStats(2, 2, 0, 0, float(difference), 2 * difference)
        return EvaluationResult(overall, seat, seat)

    monkeypatch.setattr(train, "collect_self_play_episode", collect)
    monkeypatch.setattr(train, "evaluate_against_random", evaluate)
    return progress


def training_args(database_url: str, directory: Path, *extra: str):
    return train.parse_args(
        [
            "--database-url",
            database_url,
            "--experiment",
            "training-tests",
            "--device",
            "cpu",
            "--channels",
            "4",
            "--blocks",
            "0",
            "--episodes",
            "4",
            "--replay-capacity",
            "8",
            "--batch-size",
            "2",
            "--learning-starts",
            "2",
            "--log-every",
            "1",
            "--eval-every",
            "1",
            "--eval-games",
            "2",
            "--checkpoint-every",
            "2",
            "--checkpoint-dir",
            str(directory),
            *extra,
        ]
    )


def attempt_rows(audit: AuditService) -> list[dict]:
    experiment = audit.get_experiment("training-tests")
    return audit.list_attempts(experiment["id"])


def checkpoint_rows(audit: AuditService, attempt_id: str) -> list[dict]:
    return sorted(
        audit.list_checkpoints(attempt_id), key=lambda row: row["save_sequence"]
    )


def test_training_requires_schema_upgrade_before_collecting_games(
    tmp_path: Path, simulated_games: dict[str, int]
) -> None:
    url = f"sqlite:///{tmp_path / 'uninitialized.sqlite3'}"
    with pytest.raises(ValueError, match="db upgrade"):
        train.run(training_args(url, tmp_path / "checkpoints"))
    assert simulated_games == {"episodes": 0, "evaluations": 0}
    assert not (tmp_path / "checkpoints").exists()


def test_explicit_file_only_training_needs_no_database(
    tmp_path: Path, simulated_games: dict[str, int]
) -> None:
    database_path = tmp_path / "unused.sqlite3"
    directory = tmp_path / "file-only"
    assert (
        train.run(
            training_args(
                f"sqlite:///{database_path}",
                directory,
                "--no-audit",
                "--episodes",
                "1",
                "--eval-every",
                "0",
            )
        )
        == 0
    )
    assert simulated_games["episodes"] == 1
    assert (directory / "dqn-latest.pt").is_file()
    assert not database_path.exists()


def test_training_records_evaluated_weights_and_coalesces_save_flags(
    tmp_path: Path, database_url: str, simulated_games: dict[str, int]
) -> None:
    directory = tmp_path / "checkpoints"
    assert train.run(training_args(database_url, directory)) == 0
    assert simulated_games == {"episodes": 4, "evaluations": 5}

    with AuditService(database_url) as audit:
        attempts = attempt_rows(audit)
        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt["status"] == "completed"
        assert attempt["outcome"] == "trained_only"
        checkpoints = checkpoint_rows(audit, attempt["id"])
        assert [row["episode"] for row in checkpoints] == [0, 1, 2, 3, 4]
        assert [row["parent_checkpoint_id"] for row in checkpoints] == [
            None,
            *[row["id"] for row in checkpoints[:-1]],
        ]
        assert [row["episode"] for row in checkpoints if row["is_periodic_save"]] == [
            2,
            4,
        ]
        assert [row["episode"] for row in checkpoints if row["is_best_in_attempt"]] == [
            4
        ]
        assert [
            row["episode"] for row in checkpoints if row["is_final_in_attempt"]
        ] == [4]

        evaluations = audit.list_evaluations(attempt["experiment_id"])
        assert len(evaluations) == 5
        assert {row["checkpoint_id"] for row in evaluations} == {
            row["id"] for row in checkpoints
        }
        episodes = {row["id"]: row["episode"] for row in checkpoints}
        fingerprints = set()
        for evaluation in evaluations:
            assert evaluation["status"] == "completed"
            assert evaluation["purpose"] == "training"
            details = audit.get_evaluation(evaluation["id"])
            assert len(details["suites"]) == 1
            suite = details["suites"][0]
            assert suite["player_0_wins"] == suite["player_1_wins"] == 1
            assert suite["player_0_draws"] == suite["player_1_draws"] == 0
            assert suite["player_0_losses"] == suite["player_1_losses"] == 0
            difference = episodes[evaluation["checkpoint_id"]] + 1
            assert suite["player_0_score_difference_sum"] == difference
            assert suite["player_1_score_difference_sum"] == difference
            assert suite["definition"]["game_seeds"] == [
                str(seed) for seed in train.random_game_seeds(2, 10_007)
            ]
            fingerprints.add(suite["suite_fingerprint"])
        assert len(fingerprints) == 1

        saved_bytes = (directory / "dqn-latest.pt").read_bytes()
        for path in directory.glob("*.pt"):
            path.unlink()
        exported = tmp_path / "from-database.pt"
        audit.export_checkpoint(checkpoints[-1]["id"], exported)
        assert exported.read_bytes() == saved_bytes
        payload = torch.load(exported, map_location="cpu", weights_only=True)
        assert {"online", "target", "optimizer", "rng_state"} <= payload.keys()
        assert payload["training_state"]["episode"] == 4
        assert payload["training_state"]["environment_steps"] == 8


def test_repeated_resume_from_one_checkpoint_creates_independent_branches(
    tmp_path: Path, database_url: str, simulated_games: dict[str, int]
) -> None:
    assert (
        train.run(
            training_args(
                database_url,
                tmp_path / "source",
                "--episodes",
                "2",
                "--eval-every",
                "0",
                "--checkpoint-every",
                "1",
            )
        )
        == 0
    )
    with AuditService(database_url) as audit:
        source = attempt_rows(audit)[0]
        parent = next(
            row for row in checkpoint_rows(audit, source["id"]) if row["episode"] == 1
        )

    for branch in range(2):
        assert (
            train.run(
                training_args(
                    database_url,
                    tmp_path / f"branch-{branch}",
                    "--episodes",
                    "2",
                    "--eval-every",
                    "0",
                    "--resume",
                    f"checkpoint:{parent['id']}",
                    "--reset-rng-on-resume",
                    "--seed",
                    str(10 + branch),
                    "--learning-rate",
                    "0.01",
                )
            )
            == 0
        )

    with AuditService(database_url) as audit:
        attempts = attempt_rows(audit)
        children = [
            row for row in attempts if row["starting_checkpoint_id"] == parent["id"]
        ]
        assert len(attempts) == 3
        assert len(children) == 2
        assert len({row["id"] for row in children}) == 2
        for child in children:
            assert child["status"] == "completed"
            assert child["config"]["learning_rate"] == 0.01
            assert child["config"]["effective_optimizer"][0]["lr"] == 3e-4
            assert child["config"]["rng_source"].startswith("fresh-seed:")
            chain = checkpoint_rows(audit, child["id"])
            assert chain[0]["parent_checkpoint_id"] == parent["id"]
            assert chain[-1]["episode"] == 2
            assert chain[-1]["is_final_in_attempt"]
            assert all(row["attempt_id"] == child["id"] for row in chain)


def test_interruption_records_saved_endpoint_without_completing_attempt(
    tmp_path: Path,
    database_url: str,
    simulated_games: dict[str, int],
    monkeypatch: MonkeyPatch,
) -> None:
    handlers = {signal.SIGINT: signal.SIG_DFL}

    def set_handler(number, handler):
        previous = handlers[number]
        handlers[number] = handler
        return previous

    def stop_after_episode(*_: object, **__: object) -> EpisodeResult:
        handlers[signal.SIGINT](signal.SIGINT, None)
        return EpisodeResult(2, (1, 0), 0, "self-play", None)

    monkeypatch.setattr(train.signal, "signal", set_handler)
    monkeypatch.setattr(train, "collect_self_play_episode", stop_after_episode)
    assert (
        train.run(
            training_args(
                database_url,
                tmp_path / "interrupted",
                "--eval-every",
                "0",
            )
        )
        == 130
    )

    with AuditService(database_url) as audit:
        attempt = attempt_rows(audit)[0]
        assert attempt["status"] == "interrupted"
        assert attempt["outcome"] is None
        assert attempt["last_interrupted_at"] is not None
        checkpoints = checkpoint_rows(audit, attempt["id"])
        assert checkpoints[-1]["episode"] == 1
        assert checkpoints[-1]["is_final_in_attempt"]
        assert audit.list_evaluations(attempt["experiment_id"]) == []


def test_training_failure_keeps_prior_saved_weights_and_failure_reason(
    tmp_path: Path,
    database_url: str,
    simulated_games: dict[str, int],
    monkeypatch: MonkeyPatch,
) -> None:
    calls = 0

    def fail_on_second_episode(*_: object, **__: object) -> EpisodeResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("game server disconnected")
        return EpisodeResult(2, (1, 0), 0, "self-play", None)

    monkeypatch.setattr(train, "collect_self_play_episode", fail_on_second_episode)
    with pytest.raises(ConnectionError, match="game server disconnected"):
        train.run(
            training_args(
                database_url,
                tmp_path / "failed",
                "--eval-every",
                "0",
                "--checkpoint-every",
                "1",
            )
        )

    with AuditService(database_url) as audit:
        attempt = attempt_rows(audit)[0]
        assert attempt["status"] == "failed"
        assert attempt["outcome"] is None
        assert "game server disconnected" in attempt["error"]
        checkpoints = checkpoint_rows(audit, attempt["id"])
        assert [row["episode"] for row in checkpoints] == [0, 1]
        assert checkpoints[1]["parent_checkpoint_id"] == checkpoints[0]["id"]


def test_logged_progress_advances_before_the_next_saved_checkpoint(
    tmp_path: Path,
    database_url: str,
    simulated_games: dict[str, int],
    monkeypatch: MonkeyPatch,
) -> None:
    calls = 0
    observed = {}

    def inspect_progress(*_: object, **__: object) -> EpisodeResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            with AuditService(database_url) as audit:
                attempt = attempt_rows(audit)[0]
                observed["episode"] = attempt["latest_episode"]
                observed["saved_episodes"] = [
                    row["episode"] for row in checkpoint_rows(audit, attempt["id"])
                ]
                observed["metrics"] = audit.list_metrics(attempt["id"])
        return EpisodeResult(2, (1, 0), 0, "self-play", None)

    monkeypatch.setattr(train, "collect_self_play_episode", inspect_progress)
    assert (
        train.run(
            training_args(
                database_url,
                tmp_path / "sparse-saves",
                "--episodes",
                "2",
                "--eval-every",
                "0",
                "--checkpoint-every",
                "0",
            )
        )
        == 0
    )
    assert observed["episode"] == 1
    assert observed["saved_episodes"] == [0]
    assert observed["metrics"][0]["episode"] == 1
    assert observed["metrics"][0]["metrics"]["loss"] is None


def test_training_evaluation_failure_keeps_policy_and_completed_evidence(
    tmp_path: Path,
    database_url: str,
    simulated_games: dict[str, int],
    monkeypatch: MonkeyPatch,
) -> None:
    initial_evaluate = train.evaluate_against_random
    calls = 0

    def fail_on_second_evaluation(*args, **kwargs) -> EvaluationResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("evaluation server disconnected")
        return initial_evaluate(*args, **kwargs)

    monkeypatch.setattr(train, "evaluate_against_random", fail_on_second_evaluation)
    with pytest.raises(ConnectionError, match="evaluation server disconnected"):
        train.run(training_args(database_url, tmp_path / "evaluation-failed"))

    with AuditService(database_url) as audit:
        attempt = attempt_rows(audit)[0]
        assert attempt["status"] == "failed"
        checkpoints = checkpoint_rows(audit, attempt["id"])
        assert [row["episode"] for row in checkpoints] == [0, 1]
        evaluations = audit.list_evaluations(attempt["experiment_id"])
        assert sorted(row["status"] for row in evaluations) == ["completed", "failed"]
        failed = next(row for row in evaluations if row["status"] == "failed")
        assert failed["checkpoint_id"] == checkpoints[-1]["id"]
        assert "evaluation server disconnected" in failed["error"]
