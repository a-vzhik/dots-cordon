from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
import sqlalchemy as sa
import torch

from dots_cordon_ml.audit import AuditService, Database
from dots_cordon_ml.audit import schema
from dots_cordon_ml.audit.read_repository import AuditReadRepository
from dots_cordon_ml.audit.web.app import create_app


@pytest.fixture
def audit(tmp_path):
    url = f"sqlite:///{tmp_path / 'history.sqlite3'}"
    database = Database(url)
    database.upgrade()
    database.close()
    with AuditService(url) as service:
        yield service, url


@pytest.fixture
def client(audit):
    with TestClient(create_app(audit[1])) as client:
        yield client


def save(service, experiment, directory, episode, attempt=None):
    path = directory / f"{uuid4()}.pt"
    torch.save(
        {
            "online": {"marker": torch.tensor(episode)},
            "target": {"marker": torch.tensor(episode)},
            "optimizer": {"state": {}, "param_groups": []},
            "rng_state": {"seed": 2**63 + 17},
            "training_state": {
                "episode": episode,
                "environment_steps": episode * 4,
                "optimization_steps": episode,
            },
            "board": {"rows": 7, "columns": 7},
            "model": {"channels": 4, "blocks": 0},
        },
        path,
    )
    checkpoint = service.import_checkpoint(
        experiment["id"], path, attempt_id=attempt["id"] if attempt else None
    )
    return checkpoint, path


@pytest.fixture
def history(audit, tmp_path):
    service, _ = audit
    experiment = service.ensure_experiment("test", {"rows": 7, "columns": 7})
    champion, path = save(service, experiment, tmp_path, 12500)
    assignment = service.bootstrap(experiment["id"], champion["id"])
    return experiment, champion, assignment, path


def branch(service, experiment, checkpoint, assignment):
    return service.create_attempt(
        experiment["id"],
        starting_checkpoint_id=checkpoint["id"],
        champion_at_start_assignment_id=assignment["id"],
        config={"seed": 2**63 + 3, "provenance": {"commit": "fixture", "dirty": False}},
        target_episode=checkpoint["episode"] + 1000,
    )


def suite(seed=2**63 + 7):
    return {
        "definition_version": 1,
        "kind": "random",
        "expected_games": 2,
        "game_seeds": [seed, seed],
        "seat_schedule": [0, 1],
        "rows": 7,
        "columns": 7,
    }


def score(wins=1, draws=0):
    return {
        f"as_player_{seat}": {
            "wins": wins,
            "draws": draws,
            "losses": 1 - wins - draws,
            "score_difference_sum": wins - (1 - wins - draws),
        }
        for seat in (0, 1)
    }


def evaluation(
    service,
    experiment,
    checkpoint,
    attempt=None,
    *,
    suites=1,
    opponent=None,
    purpose="screening",
):
    return service.create_evaluation(
        experiment["id"],
        checkpoint["id"],
        purpose=purpose,
        attempt_id=attempt["id"] if attempt else None,
        opponent_checkpoint_id=opponent["id"] if opponent else None,
        suite_definitions=[suite(2**63 + index + 1) for index in range(suites)],
    )


def get(client, path, **params):
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_health_openapi_and_read_only_routes(client, history):
    assert get(client, "/health")["schema"] == "compatible"
    spec = get(client, "/openapi.json")
    assert len(spec["paths"]) == 9
    assert all(set(methods) == {"get"} for methods in spec["paths"].values())
    assert spec["paths"]["/api/v1/checkpoints/{id}"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]["$ref"]
    assert client.post("/api/v1/attempts", json={}).status_code == 405
    assert (
        get(client, "/api/v1/experiments")["items"][0]["current_champion"][
            "checkpoint"
        ]["episode"]
        == 12500
    )
    with pytest.raises(sa.exc.OperationalError, match="readonly"):
        with client.app.state.audit.database.read_snapshot() as connection:
            connection.execute(schema.experiments.update().values(name="not-allowed"))


def test_missing_or_unmigrated_db_is_unavailable_without_creating_schema(tmp_path):
    path = tmp_path / "missing.sqlite3"
    with TestClient(create_app(f"sqlite:///{path}")) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["database"] == "unavailable"
        assert client.get("/api/v1/experiments").status_code == 503
    assert not path.exists()
    path.touch()
    with TestClient(create_app(f"sqlite:///{path}")) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["schema"] == "incompatible"
        assert (
            client.get("/api/v1/experiments").json()["error"]["code"]
            == "schema_incompatible"
        )
    import sqlite3

    with sqlite3.connect(path) as database:
        assert database.execute("select name from sqlite_master").fetchall() == []


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/attempts/no-such-id",
        "/api/v1/attempts/no-such-id/metrics",
        "/api/v1/checkpoints/no-such-id",
        "/api/v1/checkpoints/no-such-id/download",
        "/api/v1/evaluations/no-such-id",
        "/api/v1/experiments/no-such-id/lineage",
    ],
)
def test_unknown_records_have_structured_errors(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert "sql" not in response.text.lower()


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/attempts?limit=201",
        "/api/v1/attempts?status=made_up",
        "/api/v1/experiments?limit=0",
        "/api/v1/experiments/test/lineage?checkpoint_limit=501",
        "/api/v1/attempts/unused/metrics?max_points=1",
    ],
)
def test_request_limits_and_filters_are_validated(client, path):
    response = client.get(path)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_attempt_pagination_has_stable_bounds_and_filter_scope(client, audit, history):
    service, _ = audit
    experiment, champion, assignment, _ = history
    originals = [branch(service, experiment, champion, assignment) for _ in range(3)]
    first = get(client, "/api/v1/attempts", experiment_id="test", limit=1)
    assert first["total"] == 3
    branch(service, experiment, champion, assignment)
    second = get(
        client,
        "/api/v1/attempts",
        experiment_id="test",
        limit=1,
        cursor=first["next_cursor"],
    )
    third = get(
        client,
        "/api/v1/attempts",
        experiment_id="test",
        limit=1,
        cursor=second["next_cursor"],
    )
    assert [page["items"][0]["id"] for page in (first, second, third)] == [
        row["id"] for row in originals
    ]
    assert [page["total"] for page in (first, second, third)] == [3, 3, 3]
    assert third["next_cursor"] is None and third["remaining"] == 0
    assert get(client, "/api/v1/attempts", experiment_id="test")["total"] == 4
    bad_scope = client.get(
        "/api/v1/attempts",
        params={
            "experiment_id": "test",
            "status": "running",
            "cursor": first["next_cursor"],
        },
    )
    assert (
        bad_scope.status_code == 400
        and bad_scope.json()["error"]["code"] == "invalid_cursor"
    )
    assert client.get("/api/v1/attempts?cursor=garbage!").status_code == 400
    assert (
        client.get(
            "/api/v1/attempts", params={"created_after": "2026-09-14T00:00:00"}
        ).status_code
        == 400
    )
    service.update_attempt(
        originals[0]["id"], status="failed", error="RPC disconnected"
    )
    filtered = get(
        client,
        "/api/v1/attempts",
        starting_checkpoint_id=champion["id"],
        status="failed",
    )
    assert [row["id"] for row in filtered["items"]] == [originals[0]["id"]]


def test_attempt_detail_nested_pages_configuration_and_stale_status(
    client, audit, history, tmp_path
):
    service, _ = audit
    experiment, champion, assignment, _ = history
    attempt = branch(service, experiment, champion, assignment)
    checkpoints = [
        save(service, experiment, tmp_path, episode, attempt)[0]
        for episode in (12750, 13000, 13250)
    ]
    old_heartbeat = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(
        timespec="microseconds"
    )
    with service.database.transaction() as connection:
        connection.execute(
            schema.attempts.update()
            .where(schema.attempts.c.id == attempt["id"])
            .values(heartbeat_at=old_heartbeat)
        )
    first = get(client, f"/api/v1/attempts/{attempt['id']}", checkpoint_limit=1)
    assert first["status"] == "running" and first["diagnostics"]["unresponsive"] is True
    assert first["config"]["seed"] == str(2**63 + 3)
    assert first["config"]["provenance"]["commit"] == "fixture"
    assert first["checkpoints"]["items"][0]["id"] == checkpoints[0]["id"]
    second = get(
        client,
        f"/api/v1/attempts/{attempt['id']}",
        checkpoint_limit=1,
        checkpoint_cursor=first["checkpoints"]["next_cursor"],
    )
    assert second["checkpoints"]["items"][0]["id"] == checkpoints[1]["id"]
    assert service.get_attempt(attempt["id"])["status"] == "running"


def test_evaluation_aggregates_cover_all_completed_suites_not_only_page(
    client, audit, history
):
    service, _ = audit
    experiment, champion, _, _ = history
    batch = evaluation(service, experiment, champion, suites=3)
    path = f"/api/v1/evaluations/{batch['id']}"
    pending = get(client, path, suite_limit=1)
    assert pending["aggregate"] is None and pending["completed_games"] == 0
    assert pending["suites"]["items"][0]["result"] is None
    service.complete_suite(batch["id"], 0, score())
    service.complete_suite(batch["id"], 2, score(wins=0))
    partial = get(client, path, suite_limit=1)
    assert partial["aggregate_is_partial"] is True and partial["is_complete"] is False
    assert (
        partial["completed_suite_count"] == 2
        and partial["completed_games"] == 4
        and partial["expected_games"] == 6
    )
    overall = partial["aggregate"]["overall"]
    assert overall["wins"] == overall["losses"] == 2
    assert overall["score_difference_sum"] == 0
    assert (
        overall["match_score_numerator"] == 4
        and overall["match_score_denominator"] == 8
    )
    assert (
        partial["suites"]["items"][0]["definition"]["game_seeds"]
        == [str(2**63 + 1)] * 2
    )
    second = get(
        client, path, suite_limit=1, suite_cursor=partial["suites"]["next_cursor"]
    )
    assert second["suites"]["items"][0]["suite_index"] == 1
    assert second["suites"]["items"][0]["result"] is None
    service.complete_suite(batch["id"], 1, score(wins=0, draws=1))
    complete = get(client, path)
    assert complete["is_complete"] is True and complete["aggregate_is_partial"] is False
    assert complete["aggregate"]["overall"]["games"] == 6
    assert (
        complete["suites"]["items"][1]["result"]["overall"]["score_difference_sum"] == 0
    )


def test_failed_evaluation_keeps_partial_evidence_and_attempt_activity(
    client, audit, history
):
    service, _ = audit
    experiment, champion, assignment, _ = history
    attempt = branch(service, experiment, champion, assignment)
    batch = evaluation(service, experiment, champion, attempt, suites=2)
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat(
        timespec="microseconds"
    )
    with service.database.transaction() as connection:
        connection.execute(
            schema.attempts.update()
            .where(schema.attempts.c.id == attempt["id"])
            .values(heartbeat_at=old)
        )
    service.complete_suite(batch["id"], 0, score())
    service.fail_evaluation(batch["id"], "worker disconnected")
    detail = get(client, f"/api/v1/evaluations/{batch['id']}")
    assert detail["status"] == "failed"
    assert (
        detail["aggregate_is_partial"] is True
        and detail["aggregate"]["overall"]["games"] == 2
    )
    assert detail["suites"]["items"][1]["status"] == "failed"
    assert detail["suites"]["items"][1]["result"] is None
    active = get(client, f"/api/v1/attempts/{attempt['id']}")
    assert active["diagnostics"]["unresponsive"] is False
    assert active["status"] == "running"


def test_equal_timestamps_use_id_tie_breaker(client, audit, history):
    service, _ = audit
    experiment, champion, assignment, _ = history
    attempts = [branch(service, experiment, champion, assignment) for _ in range(3)]
    with service.database.transaction() as connection:
        connection.execute(
            schema.attempts.update().values(created_at=attempts[0]["created_at"])
        )
    cursor = None
    ids = []
    for _ in range(3):
        page = get(
            client,
            "/api/v1/attempts",
            limit=1,
            **({"cursor": cursor} if cursor else {}),
        )
        ids.append(page["items"][0]["id"])
        cursor = page["next_cursor"]
    assert ids == sorted(attempt["id"] for attempt in attempts)
    assert cursor is None


def test_checkpoint_evaluation_page_excludes_seed_definitions(client, audit, history):
    service, _ = audit
    experiment, champion, _, _ = history
    for _ in range(3):
        evaluation(service, experiment, champion)
    first = get(client, f"/api/v1/checkpoints/{champion['id']}", evaluation_limit=1)
    assert first["blob"]["available"] is True and first["evaluations"]["total"] == 3
    assert "game_seeds" not in json.dumps(first)
    second = get(
        client,
        f"/api/v1/checkpoints/{champion['id']}",
        evaluation_limit=1,
        evaluation_cursor=first["evaluations"]["next_cursor"],
    )
    assert (
        first["evaluations"]["items"][0]["id"]
        != second["evaluations"]["items"][0]["id"]
    )


def test_metrics_bound_samples_preserve_endpoints_and_original_windows(
    client, audit, history
):
    service, _ = audit
    experiment, champion, assignment, _ = history
    attempt = branch(service, experiment, champion, assignment)
    for index in range(50):
        service.record_metrics(
            attempt["id"],
            12501 + index,
            {
                "episode_start": 12501 + index,
                "loss": None if index == 0 else index / 10,
                "seed": 2**63 + index,
            },
        )
    response = get(client, f"/api/v1/attempts/{attempt['id']}/metrics", max_points=7)
    assert response["total_samples"] == 50 and len(response["items"]) <= 7
    assert response["omitted_samples"] == 50 - len(response["items"])
    assert (
        response["items"][0]["episode"] == 12501
        and response["items"][-1]["episode"] == 12550
    )
    assert response["items"][0]["metrics"]["loss"] is None
    assert response["items"][0]["metrics"]["seed"] == str(2**63)
    window = get(
        client,
        f"/api/v1/attempts/{attempt['id']}/metrics",
        episode_min=12510,
        episode_max=12520,
    )
    assert window["total_samples"] == 11


def test_download_exact_bytes_etag_integrity_and_bounded_slots(client, audit, history):
    service, _ = audit
    _, champion, _, path = history
    uri = f"/api/v1/checkpoints/{champion['id']}/download"
    response = client.get(uri)
    assert response.status_code == 200
    assert response.content == path.read_bytes()
    assert (
        response.headers["etag"] == f'"{hashlib.sha256(response.content).hexdigest()}"'
    )
    assert int(response.headers["content-length"]) == len(response.content)
    assert "12500" in response.headers["content-disposition"]
    cached = client.get(uri, headers={"If-None-Match": f"W/{response.headers['etag']}"})
    assert cached.status_code == 304 and not cached.content
    semaphore = client.app.state.download_slots
    assert semaphore.acquire(blocking=False) and semaphore.acquire(blocking=False)
    try:
        busy = client.get(uri)
        assert busy.status_code == 429 and busy.headers["retry-after"] == "1"
    finally:
        semaphore.release()
        semaphore.release()
    with service.database.transaction() as connection:
        connection.execute(
            schema.checkpoint_blobs.update()
            .where(schema.checkpoint_blobs.c.id == champion["checkpoint_blob_id"])
            .values(payload=b"corrupted")
        )
    assert client.get(uri).status_code == 409
    assert semaphore.acquire(blocking=False)
    semaphore.release()


def test_lineage_boundaries_never_have_dangling_edges(client, audit, history, tmp_path):
    service, _ = audit
    experiment, champion, assignment, _ = history
    attempt = branch(service, experiment, champion, assignment)
    checkpoints = [
        save(service, experiment, tmp_path, episode, attempt)[0]
        for episode in (12750, 13000, 13250)
    ]
    graph = get(
        client,
        "/api/v1/experiments/test/lineage",
        root_checkpoint_id=champion["id"],
        episode_min=13000,
        checkpoint_limit=1,
    )
    assert graph["truncated"] is True
    ids = {
        row["id"]
        for row in [*graph["checkpoints"]["items"], *graph["boundary_checkpoints"]]
    }
    assert graph["checkpoints"]["items"][0]["id"] == checkpoints[1]["id"]
    assert checkpoints[0]["id"] in ids and champion["id"] in ids
    assert all(
        edge["parent_checkpoint_id"] in ids and edge["child_checkpoint_id"] in ids
        for edge in graph["edges"]
    )
    assert graph["checkpoints"]["items"][0]["children_outside_slice"] == 1
    next_graph = get(
        client,
        "/api/v1/experiments/test/lineage",
        root_checkpoint_id=champion["id"],
        episode_min=13000,
        checkpoint_limit=1,
        checkpoint_cursor=graph["checkpoints"]["next_cursor"],
    )
    assert next_graph["checkpoints"]["items"][0]["id"] == checkpoints[2]["id"]
    assert (
        client.get(
            "/api/v1/experiments/test/lineage?episode_min=10&episode_max=1"
        ).status_code
        == 400
    )


def test_thirteen_attempt_tree_counts_branches_not_winning_attempt_tail(
    client, audit, history, tmp_path
):
    service, _ = audit
    experiment, champion, assignment, _ = history
    for _ in range(10):
        attempt = branch(service, experiment, champion, assignment)
        nodes = [
            save(service, experiment, tmp_path, episode, attempt)[0]
            for episode in (12750, 13000, 13250, 13500)
        ]
    winner = nodes[2]
    baseline = evaluation(service, experiment, champion, attempt)
    screened = evaluation(service, experiment, winner, attempt)
    challenge = evaluation(
        service, experiment, winner, attempt, opponent=champion, purpose="head_to_head"
    )
    for batch in (baseline, screened, challenge):
        service.complete_suite(batch["id"], 0, score())
    service.record_decision(
        attempt["id"],
        winner["id"],
        stage="screening",
        result="qualified",
        candidate_evaluation_id=screened["id"],
        champion_evaluation_id=baseline["id"],
        policy={"screen_max_regression": 0},
    )
    promoted = service.promote(
        experiment["id"],
        winner["id"],
        attempt_id=attempt["id"],
        expected_assignment_id=assignment["id"],
        candidate_evaluation_id=challenge["id"],
        champion_evaluation_id=baseline["id"],
        policy={"promotion_min_match_score": 0.52, "promotion_min_suite_wins": 1},
    )
    for _ in range(3):
        later = branch(service, experiment, winner, promoted)
        save(service, experiment, tmp_path, 13500, later)
    graph = get(client, "/api/v1/experiments/test/lineage")
    assert graph["counts"]["attempts"] == 13
    assert graph["current_champion"]["branches"]["attempts"] == 3
    assert graph["current_champion"]["checkpoint"]["counts"]["child_checkpoints"] == 4
    assert len(graph["attempts"]["items"]) == 13
    assert (
        len(
            get(client, "/api/v1/attempts", starting_checkpoint_id=champion["id"])[
                "items"
            ]
        )
        == 10
    )
    rooted = get(
        client, "/api/v1/experiments/test/lineage", root_checkpoint_id=winner["id"]
    )
    assert len(rooted["attempts"]["items"]) == 4
    assert nodes[3]["id"] in {row["id"] for row in rooted["checkpoints"]["items"]}
    challenge_detail = get(client, f"/api/v1/evaluations/{challenge['id']}")
    assert challenge_detail["opponent_checkpoint_id"] == champion["id"]
    assert any(
        item["result"] == "promoted" for item in challenge_detail["decisions"]["items"]
    )
    assert len(challenge_detail["participants"]) == 2


def test_graph_reads_one_snapshot_while_writer_commits(
    client, audit, history, monkeypatch
):
    service, _ = audit
    experiment, champion, assignment, _ = history
    original = AuditReadRepository.experiment
    writes = []

    def concurrent_write(repository, identifier):
        row = original(repository, identifier)
        with ThreadPoolExecutor(max_workers=1) as pool:
            writes.append(
                pool.submit(branch, service, experiment, champion, assignment).result(
                    timeout=3
                )
            )
        return row

    monkeypatch.setattr(AuditReadRepository, "experiment", concurrent_write)
    first = get(client, "/api/v1/experiments/test/lineage")
    assert first["counts"]["attempts"] == 0 and first["attempts"]["total"] == 0
    monkeypatch.setattr(AuditReadRepository, "experiment", original)
    assert get(client, "/api/v1/experiments/test/lineage")["counts"]["attempts"] == 1
    assert len(writes) == 1


def test_metadata_queries_are_batched_and_never_fetch_blob_payloads(
    client, audit, history, tmp_path
):
    service, _ = audit
    experiment, champion, assignment, _ = history
    attempt = branch(service, experiment, champion, assignment)
    for episode in range(12501, 12521):
        save(service, experiment, tmp_path, episode, attempt)
    statements = []

    def trace(_connection, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    engine = client.app.state.audit.database.engine
    sa.event.listen(engine, "before_cursor_execute", trace)
    try:
        get(client, "/api/v1/experiments/test/lineage", checkpoint_limit=5)
        small = len(statements)
        statements.clear()
        get(client, "/api/v1/experiments/test/lineage", checkpoint_limit=100)
        assert len(statements) <= small + 2
        assert not any(
            "checkpoint_blobs.payload" in statement for statement in statements
        )
    finally:
        sa.event.remove(engine, "before_cursor_execute", trace)
