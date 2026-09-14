from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import importlib

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
import sqlalchemy as sa
import torch

from dots_cordon_ml.audit import (
    AuditError,
    AuditService,
    Database,
    PromotionConflict,
    SchemaVersionError,
)
from dots_cordon_ml.audit import schema
from dots_cordon_ml.audit.repository import AuditRepository


@pytest.fixture
def database_url(tmp_path):
    url = f"sqlite:///{tmp_path / 'audit.sqlite3'}"
    database = Database(url)
    database.upgrade()
    database.close()
    return url


@pytest.fixture
def service(database_url):
    with AuditService(database_url) as service:
        yield service


def checkpoint_file(tmp_path, episode=12500, marker=1):
    path = tmp_path / f"{episode}-{marker}.pt"
    torch.save(
        {
            "online": {"weight": torch.tensor([marker])},
            "target": {"weight": torch.tensor([marker])},
            "optimizer": {"state": {}, "param_groups": []},
            "rng_state": {"seed": marker},
            "training_state": {
                "episode": episode,
                "environment_steps": episode * 3,
                "optimization_steps": episode,
            },
            "board": {"rows": 7, "columns": 7},
            "model": {"channels": 4, "blocks": 0},
        },
        path,
    )
    return path


def root(service, tmp_path):
    experiment = service.ensure_experiment("test", {"rows": 7, "columns": 7})
    checkpoint = service.import_checkpoint(experiment["id"], checkpoint_file(tmp_path))
    assignment = service.bootstrap(experiment["id"], checkpoint["id"])
    return experiment, checkpoint, assignment


def attempt(service, experiment, checkpoint, assignment):
    return service.create_attempt(
        experiment["id"],
        starting_checkpoint_id=checkpoint["id"],
        champion_at_start_assignment_id=assignment["id"],
        config={"seed": 17},
        target_episode=13500,
    )


def definition(seed=1):
    return {
        "definition_version": 1,
        "kind": "random",
        "rows": 7,
        "columns": 7,
        "expected_games": 2,
        "game_seeds": [str(seed), str(seed)],
        "seat_schedule": [0, 1],
    }


def result(wins=1):
    return {
        f"as_player_{seat}": {
            "wins": wins,
            "draws": 0,
            "losses": 1 - wins,
            "score_difference_sum": 2 if wins else -2,
        }
        for seat in (0, 1)
    }


def evaluated(
    service, experiment, subject, attempt, *, opponent=None, seed=1, complete=True
):
    batch = service.create_evaluation(
        experiment["id"],
        subject["id"],
        purpose="head_to_head" if opponent else "screening",
        attempt_id=attempt["id"],
        opponent_checkpoint_id=opponent["id"] if opponent else None,
        suite_definitions=[definition(seed)],
    )
    if complete:
        service.complete_suite(batch["id"], 0, result())
    return batch


def promotion_ready(service, tmp_path, experiment, champion, assignment, marker):
    branch = attempt(service, experiment, champion, assignment)
    candidate = service.import_checkpoint(
        experiment["id"],
        checkpoint_file(tmp_path, 12750, marker),
        attempt_id=branch["id"],
    )
    baseline = evaluated(service, experiment, champion, branch)
    screening = evaluated(service, experiment, candidate, branch)
    service.record_decision(
        branch["id"],
        candidate["id"],
        stage="screening",
        result="qualified",
        candidate_evaluation_id=screening["id"],
        champion_evaluation_id=baseline["id"],
        policy={"screen_max_regression": 0.003},
    )
    challenge = evaluated(service, experiment, candidate, branch, opponent=champion)
    return dict(
        experiment_id=experiment["id"],
        checkpoint_id=candidate["id"],
        attempt_id=branch["id"],
        expected_assignment_id=assignment["id"],
        candidate_evaluation_id=challenge["id"],
        champion_evaluation_id=baseline["id"],
        policy={"promotion_min_match_score": 0.55, "promotion_min_suite_wins": 1},
    )


def test_schema_is_explicitly_migrated_and_checked(tmp_path):
    url = f"sqlite:///{tmp_path / 'new.sqlite3'}"
    with pytest.raises(SchemaVersionError, match="db upgrade"):
        AuditService(url)
    database = Database(url)
    assert not database.status()["up_to_date"]
    assert database.upgrade()["up_to_date"]
    assert database.upgrade()["up_to_date"]
    with database.engine.connect() as connection:
        assert set(sa.inspect(connection).get_table_names()) == set(
            schema.metadata.tables
        ) | {"alembic_version"}
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert connection.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
    database.close()


def test_postgres_migration_defers_cyclic_foreign_keys():
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    revision = importlib.import_module(
        "dots_cordon_ml.audit.migrations.versions.0001_training_audit"
    )
    with Operations.context(context):
        revision.upgrade()
    sql = output.getvalue()
    assert "payload BYTEA" in sql
    assert sql.index("ALTER TABLE") > sql.rindex("CREATE TABLE")
    assert sql.count("CREATE TABLE") == 9
    assert "REFERENCES champion_history (id)" in sql


def test_full_blob_round_trip_and_branch_identity(service, tmp_path):
    experiment, champion, assignment = root(service, tmp_path)
    left = attempt(service, experiment, champion, assignment)
    right = attempt(service, experiment, champion, assignment)
    path = checkpoint_file(tmp_path, 12750)
    original = path.read_bytes()
    first = service.import_checkpoint(
        experiment["id"], path, attempt_id=left["id"], is_periodic_save=True
    )
    second = service.import_checkpoint(experiment["id"], path, attempt_id=right["id"])
    assert first["id"] != second["id"]
    assert first["checkpoint_blob_id"] == second["checkpoint_blob_id"]
    assert (
        first["parent_checkpoint_id"]
        == second["parent_checkpoint_id"]
        == champion["id"]
    )
    assert "payload" not in service.get_checkpoint(first["id"])
    assert "payload" not in str(service.list_checkpoints(left["id"]))
    path.unlink()
    exported = service.export_checkpoint(first["id"], tmp_path / "export.pt")
    assert exported.read_bytes() == original


def test_same_boundary_flags_and_immutable_ancestry(service, tmp_path):
    experiment, champion, assignment = root(service, tmp_path)
    branch = attempt(service, experiment, champion, assignment)
    path = checkpoint_file(tmp_path, 12750)
    first = service.import_checkpoint(
        experiment["id"], path, attempt_id=branch["id"], is_periodic_save=True
    )
    changed = service.import_checkpoint(
        experiment["id"],
        path,
        checkpoint_id=first["id"],
        attempt_id=branch["id"],
        is_best_in_attempt=True,
        is_final_in_attempt=True,
    )
    assert (
        changed["is_periodic_save"]
        and changed["is_best_in_attempt"]
        and changed["is_final_in_attempt"]
    )
    assert len(service.list_checkpoints(branch["id"])) == 1
    assert (
        service.import_checkpoint(
            experiment["id"], path, checkpoint_id=first["id"], attempt_id=branch["id"]
        )["id"]
        == first["id"]
    )
    with pytest.raises(AuditError, match="lineage"):
        service.import_checkpoint(
            experiment["id"],
            path,
            checkpoint_id=first["id"],
            attempt_id=branch["id"],
            parent_checkpoint_id=None,
        )
    later = service.import_checkpoint(
        experiment["id"],
        checkpoint_file(tmp_path, 13000),
        attempt_id=branch["id"],
        is_best_in_attempt=True,
        is_final_in_attempt=True,
    )
    assert later["parent_checkpoint_id"] == first["id"]
    assert not service.get_checkpoint(first["id"])["is_best_in_attempt"]
    with pytest.raises(AuditError, match="parent"):
        service.import_checkpoint(
            experiment["id"],
            checkpoint_file(tmp_path, 13250),
            attempt_id=branch["id"],
            parent_checkpoint_id=champion["id"],
        )


def test_wrong_board_and_earlier_parent_rollback(service, tmp_path):
    experiment, champion, assignment = root(service, tmp_path)
    branch = attempt(service, experiment, champion, assignment)
    with pytest.raises(AuditError, match="precede"):
        service.import_checkpoint(
            experiment["id"], checkpoint_file(tmp_path, 12000), attempt_id=branch["id"]
        )
    assert service.list_checkpoints(branch["id"]) == []
    other = service.ensure_experiment("large", {"rows": 15, "columns": 15})
    with pytest.raises(AuditError, match="board"):
        service.import_checkpoint(other["id"], checkpoint_file(tmp_path))


def test_metric_sequence_and_config_freeze(service, tmp_path):
    experiment, champion, assignment = root(service, tmp_path)
    branch = attempt(service, experiment, champion, assignment)
    service.update_attempt(branch["id"], config={"seed": 17, "device": "cpu"})
    service.record_metrics(branch["id"], 12750, {"loss": 0.1}, sample_sequence=1)
    service.record_metrics(branch["id"], 12750, {"loss": 0.1}, sample_sequence=1)
    assert len(service.list_metrics(branch["id"])) == 1
    with pytest.raises(AuditError, match="sequence reused"):
        service.record_metrics(branch["id"], 12750, {"loss": 0.2}, sample_sequence=1)
    with pytest.raises(AuditError, match="immutable"):
        service.update_attempt(branch["id"], config={"seed": 18})


def test_suite_counts_partial_failure_and_immutable_results(service, tmp_path):
    experiment, champion, assignment = root(service, tmp_path)
    branch = attempt(service, experiment, champion, assignment)
    batch = service.create_evaluation(
        experiment["id"],
        champion["id"],
        attempt_id=branch["id"],
        purpose="screening",
        suite_definitions=[definition(), definition(2)],
    )
    wrong = result()
    wrong["as_player_0"]["wins"] = 2
    with pytest.raises(AuditError, match="game counts"):
        service.complete_suite(batch["id"], 0, wrong)
    service.complete_suite(batch["id"], 0, result())
    service.complete_suite(batch["id"], 0, result())
    assert service.get_evaluation(batch["id"])["status"] == "running"
    with pytest.raises(AuditError, match="immutable"):
        service.complete_suite(batch["id"], 0, result(0))
    service.fail_evaluation(batch["id"], "worker stopped", status="interrupted")
    saved = service.get_evaluation(batch["id"])
    assert saved["suites"][0]["status"] == "completed"
    assert saved["suites"][1]["player_0_wins"] is None
    with pytest.raises(AuditError, match="stopped"):
        service.complete_suite(batch["id"], 1, result())


def test_screening_rejects_incomplete_and_mismatched_evidence(service, tmp_path):
    experiment, champion, assignment = root(service, tmp_path)
    branch = attempt(service, experiment, champion, assignment)
    candidate = service.import_checkpoint(
        experiment["id"], checkpoint_file(tmp_path, 12750), attempt_id=branch["id"]
    )
    baseline = evaluated(service, experiment, champion, branch)
    screening = evaluated(
        service, experiment, candidate, branch, seed=2, complete=False
    )
    kwargs = dict(
        stage="screening",
        result="qualified",
        candidate_evaluation_id=screening["id"],
        champion_evaluation_id=baseline["id"],
    )
    with pytest.raises(AuditError, match="complete"):
        service.record_decision(branch["id"], candidate["id"], **kwargs)
    service.complete_suite(screening["id"], 0, result())
    with pytest.raises(AuditError, match="identical"):
        service.record_decision(branch["id"], candidate["id"], **kwargs)


def test_atomic_promotion_and_idempotent_retry(service, tmp_path):
    experiment, champion, assignment = root(service, tmp_path)
    values = promotion_ready(service, tmp_path, experiment, champion, assignment, 2)
    promoted = service.promote(**values)
    assert promoted["generation"] == 2
    assert service.promote(**values) == promoted
    assert (
        service.current_champion(experiment["id"])["checkpoint_id"]
        == values["checkpoint_id"]
    )
    assert service.get_attempt(values["attempt_id"])["outcome"] == "promoted"
    assert len(service.champion_history(experiment["id"])) == 2


def test_screening_cannot_qualify_losing_evidence(service, tmp_path):
    experiment, champion, assignment = root(service, tmp_path)
    branch = attempt(service, experiment, champion, assignment)
    candidate = service.import_checkpoint(
        experiment["id"], checkpoint_file(tmp_path, 12750), attempt_id=branch["id"]
    )
    baseline = evaluated(service, experiment, champion, branch)
    screening = evaluated(service, experiment, candidate, branch, complete=False)
    service.complete_suite(screening["id"], 0, result(0))
    with pytest.raises(AuditError, match="qualification policy"):
        service.record_decision(
            branch["id"],
            candidate["id"],
            stage="screening",
            result="qualified",
            candidate_evaluation_id=screening["id"],
            champion_evaluation_id=baseline["id"],
            policy={"screen_max_regression": 0.003},
        )
    assert service.list_decisions(branch["id"]) == []


def test_promotion_threshold_and_transaction_rollback(service, tmp_path, monkeypatch):
    experiment, champion, assignment = root(service, tmp_path)
    values = promotion_ready(service, tmp_path, experiment, champion, assignment, 2)
    with pytest.raises(AuditError, match="policy"):
        service.promote(
            **{
                **values,
                "policy": {
                    "promotion_min_match_score": 1.01,
                    "promotion_min_suite_wins": 1,
                },
            }
        )
    monkeypatch.setattr(AuditRepository, "compare_and_set_champion", lambda *_: False)
    with pytest.raises(PromotionConflict):
        service.promote(**values)
    assert service.current_champion(experiment["id"])["id"] == assignment["id"]
    assert len(service.champion_history(experiment["id"])) == 1
    assert service.get_attempt(values["attempt_id"])["outcome"] is None
    assert not any(
        row["result"] == "promoted"
        for row in service.list_decisions(values["attempt_id"])
    )


def test_two_concurrent_promotions_have_exactly_one_winner(
    service, database_url, tmp_path
):
    experiment, champion, assignment = root(service, tmp_path)
    values = [
        promotion_ready(service, tmp_path, experiment, champion, assignment, marker)
        for marker in (2, 3)
    ]

    def compete(arguments):
        with AuditService(database_url) as contender:
            try:
                return contender.promote(**arguments)["checkpoint_id"]
            except PromotionConflict:
                return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(compete, values))
    assert results.count(None) == 1
    assert len(service.champion_history(experiment["id"])) == 2


def test_seed_reservations_survive_reopen(service, database_url, tmp_path):
    experiment, _, _ = root(service, tmp_path)
    assert service.reserve_suite_seeds(experiment["id"], "screen", 3, 100) == (
        100,
        101,
        102,
    )
    with AuditService(database_url) as reopened:
        assert reopened.reserve_suite_seeds(experiment["id"], "screen", 2, 100) == (
            103,
            104,
        )


def test_corrupt_blob_cannot_be_exported(service, tmp_path):
    _, champion, _ = root(service, tmp_path)
    with service.database.engine.begin() as connection:
        connection.execute(
            schema.checkpoint_blobs.update()
            .where(schema.checkpoint_blobs.c.id == champion["checkpoint_blob_id"])
            .values(payload=b"corrupt")
        )
    target = tmp_path / "export.pt"
    with pytest.raises(AuditError, match="checksum"):
        service.export_checkpoint(champion["id"], target)
    assert not target.exists()
