from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from dots_cordon_ml import evaluate, head_to_head
from dots_cordon_ml.audit import AuditService, Database
from dots_cordon_ml.dqn import DQNAgent
from dots_cordon_ml.environment import StepResult
from dots_cordon_ml.proto import game_pb2
from dots_cordon_ml.self_play import MatchStats, random_game_seeds


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


def checkpoint(tmp_path: Path, name: str, *, seed: int = 1) -> Path:
    agent = DQNAgent(
        device=torch.device("cpu"),
        learning_rate=3e-4,
        gamma=0.99,
        seed=seed,
        channels=8,
        blocks=0,
    )
    path = tmp_path / name
    torch.save(
        {
            "board": {"rows": 3, "columns": 3},
            "model": {"channels": 8, "blocks": 0},
            "training_state": {
                "episode": 250,
                "environment_steps": 2_250,
                "optimization_steps": 100,
            },
            "online": agent.online.state_dict(),
            "target": agent.target.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
        },
        path,
    )
    return path


class EvaluationEnvironment:
    """Exercise real evaluation code with deterministic, inexpensive games."""

    def __init__(self, **_kwargs: object) -> None:
        self.game_index = 0

    def __enter__(self) -> EvaluationEnvironment:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def reset(self) -> game_pb2.GameState:
        return game_pb2.GameState(
            board=game_pb2.Board(rows=3, columns=3, cells=bytes(9)),
            scores=(0, 0),
            current_player=0,
        )

    def step(self, _action: int) -> StepResult:
        scores = [(2, 0), (1, 1), (0, 1), (0, 3)][self.game_index % 4]
        self.game_index += 1
        return StepResult(
            game=game_pb2.GameState(
                board=game_pb2.Board(rows=3, columns=3, cells=bytes(9)),
                scores=scores,
                current_player=1,
                turn=1,
                terminal=True,
            ),
            player=0,
            reward=0,
        )


def test_random_evaluator_persists_exact_inputs_seeds_and_seat_results(
    tmp_path: Path,
    database: tuple[str, AuditService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, service = database
    path = checkpoint(tmp_path, "candidate.pt")
    expected_bytes = path.read_bytes()
    monkeypatch.setattr(evaluate, "GameEnvironment", EvaluationEnvironment)
    args = evaluate.parse_args(
        [
            str(path),
            "--database-url",
            url,
            "--games",
            "4",
            "--seed",
            "23",
            "--device",
            "cpu",
        ]
    )

    assert evaluate.run(args) == 0

    experiment = service.get_experiment("default")
    batches = service.list_evaluations(experiment["id"])
    assert len(batches) == 1
    batch = service.get_evaluation(batches[0]["id"])
    assert batch["status"] == "completed"
    assert batch["purpose"] == "standalone_random"
    assert batch["opponent_checkpoint_id"] is None
    suite = batch["suites"][0]
    assert suite["status"] == "completed"
    assert suite["definition"]["game_seeds"] == [
        str(seed) for seed in random_game_seeds(4, 23)
    ]
    assert suite["definition"]["seat_schedule"] == [0, 1, 0, 1]
    assert suite["player_0_score_difference_sum"] == 1
    assert suite["player_1_score_difference_sum"] == 3
    for seat, expected in ((0, (1, 0, 1)), (1, (1, 1, 0))):
        actual = tuple(
            suite[f"player_{seat}_{name}"] for name in ("wins", "draws", "losses")
        )
        assert actual == expected
    exported = service.export_checkpoint(batch["checkpoint_id"], tmp_path / "export.pt")
    assert exported.read_bytes() == expected_bytes


@pytest.mark.parametrize(
    "error_type,status",
    [(ConnectionError, "failed"), (KeyboardInterrupt, "interrupted")],
)
def test_evaluator_retains_completed_result_and_marks_remaining_planned_batch(
    tmp_path: Path,
    database: tuple[str, AuditService],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
    status: str,
) -> None:
    url, service = database
    first = checkpoint(tmp_path, "first.pt")
    second = checkpoint(tmp_path, "second.pt", seed=2)
    monkeypatch.setattr(evaluate, "GameEnvironment", EvaluationEnvironment)
    original = evaluate.evaluate_against_random
    calls = 0

    def evaluate_once(*args: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("evaluation stopped")
        return original(*args)

    monkeypatch.setattr(evaluate, "evaluate_against_random", evaluate_once)
    args = evaluate.parse_args(
        [
            str(first),
            str(second),
            "--database-url",
            url,
            "--games",
            "4",
            "--device",
            "cpu",
        ]
    )
    with pytest.raises(error_type, match="evaluation stopped"):
        evaluate.run(args)

    experiment = service.get_experiment("default")
    batches = service.list_evaluations(experiment["id"])
    assert sorted(batch["status"] for batch in batches) == sorted(["completed", status])
    unfinished = next(batch for batch in batches if batch["status"] == status)
    failed = service.get_evaluation(unfinished["id"])
    assert failed["suites"][0]["status"] == status
    assert failed["suites"][0]["player_0_wins"] is None


def test_head_to_head_pins_database_participants_and_retains_oriented_result(
    tmp_path: Path,
    database: tuple[str, AuditService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, service = database
    experiment = service.ensure_experiment(
        "default", {"rows": 3, "columns": 3, "max_turns": 0}
    )
    record_a = service.import_checkpoint(experiment["id"], checkpoint(tmp_path, "a.pt"))
    record_b = service.import_checkpoint(
        experiment["id"], checkpoint(tmp_path, "b.pt", seed=2)
    )
    service.bootstrap(experiment["id"], record_b["id"])
    monkeypatch.setattr(head_to_head, "GameEnvironment", EvaluationEnvironment)
    args = head_to_head.parse_args(
        [
            f"checkpoint:{record_a['id']}",
            "champion:default",
            "--database-url",
            url,
            "--games",
            "4",
            "--opening-random-moves",
            "0",
            "--device",
            "cpu",
        ]
    )

    assert head_to_head.run(args) == 0

    batches = service.list_evaluations(experiment["id"])
    assert len(batches) == 1
    batch = service.get_evaluation(batches[0]["id"])
    assert batch["checkpoint_id"] == record_a["id"]
    assert batch["opponent_checkpoint_id"] == record_b["id"]
    assert batch["purpose"] == "standalone_head_to_head"
    assert batch["status"] == "completed"
    suite = batch["suites"][0]
    assert suite["definition"]["kind"] == "head_to_head"
    assert suite["definition"]["opening_random_moves"] == 0
    assert suite["player_0_score_difference_sum"] == 1
    assert suite["player_1_score_difference_sum"] == 3


def test_head_to_head_records_environment_start_failure(
    tmp_path: Path,
    database: tuple[str, AuditService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, service = database
    path_a = checkpoint(tmp_path, "a.pt")
    path_b = checkpoint(tmp_path, "b.pt", seed=2)

    class UnavailableEnvironment(EvaluationEnvironment):
        def __enter__(self):
            raise ConnectionError("game server unavailable")

    monkeypatch.setattr(head_to_head, "GameEnvironment", UnavailableEnvironment)
    args = head_to_head.parse_args(
        [
            str(path_a),
            str(path_b),
            "--database-url",
            url,
            "--games",
            "4",
            "--device",
            "cpu",
        ]
    )
    with pytest.raises(ConnectionError, match="game server unavailable"):
        head_to_head.run(args)

    experiment = service.get_experiment("default")
    batches = service.list_evaluations(experiment["id"])
    assert len(batches) == 1
    assert batches[0]["status"] == "failed"


def test_evaluation_uses_saved_blob_when_source_file_changes_after_import(
    tmp_path: Path,
    database: tuple[str, AuditService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, _service = database
    source = checkpoint(tmp_path, "latest.pt")
    replacement = checkpoint(tmp_path, "next.pt", seed=2)
    expected_weights = torch.load(source, weights_only=True)["online"]
    original_record = evaluate.checkpoint_record
    original_evaluate = evaluate.evaluate_against_random

    def import_then_replace(*args: object, **kwargs: object):
        record = original_record(*args, **kwargs)
        source.write_bytes(replacement.read_bytes())
        return record

    def check_tested_weights(environment, agent, seeds):
        for name, tensor in agent.online.state_dict().items():
            assert torch.equal(tensor, expected_weights[name])
        return original_evaluate(environment, agent, seeds)

    monkeypatch.setattr(evaluate, "GameEnvironment", EvaluationEnvironment)
    monkeypatch.setattr(evaluate, "checkpoint_record", import_then_replace)
    monkeypatch.setattr(evaluate, "evaluate_against_random", check_tested_weights)
    args = evaluate.parse_args(
        [str(source), "--database-url", url, "--games", "4", "--device", "cpu"]
    )
    assert evaluate.run(args) == 0


def test_file_only_evaluation_does_not_require_a_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = checkpoint(tmp_path, "candidate.pt")
    monkeypatch.setattr(evaluate, "GameEnvironment", EvaluationEnvironment)
    args = evaluate.parse_args(
        [str(path), "--no-audit", "--games", "4", "--device", "cpu"]
    )
    assert evaluate.run(args) == 0


def test_opponent_stats_preserve_exact_score_total() -> None:
    stats = MatchStats(3, 2, 0, 1, 1 / 3, 1)
    opponent = head_to_head._opponent_stats(stats)
    assert (opponent.wins, opponent.draws, opponent.losses) == (1, 0, 2)
    assert opponent.score_difference_sum == -1
    assert opponent.mean_score_difference == -1 / 3
