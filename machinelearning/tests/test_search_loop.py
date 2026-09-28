import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest
from dots_cordon_ml import search_loop, search_train, training
from dots_cordon_ml.audit import AuditService, Database
from dots_cordon_ml.audit.integration import resolve_checkpoint_reference
from dots_cordon_ml.checkpoint import read_checkpoint
from dots_cordon_ml.operations import operation_lock, write_result
from search_loop_worker import FixtureChildren
from test_search_training import SmallEnvironment, arguments
from test_training_operation import assert_equal

@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(search_train, "GameEnvironment", SmallEnvironment)
    url = f"sqlite:///{tmp_path / 'audit.sqlite3'}"
    db = Database(url)
    db.upgrade()
    db.close()
    search_train.run(arguments(tmp_path / "source", "--episodes", "1"))
    with AuditService(url) as audit:
        exp = audit.ensure_experiment("source", {"rows": 2, "columns": 2, "max_turns": 0})
        source = audit.import_checkpoint(exp["id"], tmp_path / "source/search-latest.pt")
    config = {key: getattr(arguments(tmp_path / "rounds"), key) for key in training.TRAINING_CONFIG_FIELDS}
    config.update(checkpoint_dir=str(tmp_path / "rounds"), bootstrap_champion=True)
    eval_config = {}
    for i, stage in enumerate(search_loop.STAGES, 3):
        eval_config[stage] = dict(server="127.0.0.1:1", rows=2, columns=2, max_turns=0, mode="policy", device="cpu",
                                  games=100, suites=10 if i == 5 else 3, seed=i * 10000000 + 1,
                                  opening_random_moves=0, paired_seats=True, evaluation_workers=1, rpc_timeout=10.0)
    request = dict(version=1, run_id="fixture-loop", experiment="rounds", source=dict(mode="initialize", checkpoint_id=source["id"]),
                   total_episode=2, round_episodes=1, training_config=config, evaluation_config=eval_config,
                   gates=dict(screen_max_regression=.003, promotion_min_match_score=.52, promotion_min_suite_wins=2),
                   work_dir=str(tmp_path / "transport"))
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SEARCH_LOOP_TEST_TRACE", str(trace))
    return url, request, trace

def entries(trace):
    return [json.loads(line) for line in trace.read_text().splitlines()]

@pytest.mark.parametrize("branch,stages,decision", [
    ("screen_reject", ["screening"], "rejected"),
    ("initial_reject", ["screening", "head_to_head"], "rejected"),
    ("normal", ["screening", "head_to_head", "promotion"], "promoted"),
    ("extended", ["screening", "head_to_head", "extended_head_to_head", "promotion"], "promoted"),
    ("extended_reject", ["screening", "head_to_head", "extended_head_to_head"], "rejected"),
])
def test_real_subprocess_round_order_and_learner_continuity(setup, monkeypatch, branch, stages, decision):
    url, request, trace = setup
    monkeypatch.setenv("SEARCH_LOOP_TEST_BRANCH", branch)
    request.update(total_episode=3, round_episodes=2)
    response = search_loop.run_request(request, url, children=FixtureChildren())
    assert response["status"] == "completed", response
    assert [r["episode"] for r in response["rounds"]] == [2, 3]
    assert [r["decision"] for r in response["rounds"]] == [decision, decision]
    history = entries(trace)
    assert [e["request"].get("stage", e["kind"]) for e in history] == ["training", *stages] * 2
    assert all(e["pid"] != os.getpid() for e in history)
    train = [e["request"] for e in history if e["kind"] == "training"]
    first, second = response["rounds"]
    assert train[1]["source"] == dict(mode="resume", checkpoint_id=first["learner_checkpoint_id"])
    assert train[1]["config"]["bootstrap_champion"] is False
    assert train[0]["config"]["bootstrap_champion"] is True
    assert first["results"]["screening"]["candidate_checkpoint_id"] != train[1]["source"]["checkpoint_id"]
    expected_champion = first["results"]["promotion"]["candidate_checkpoint_id"] if decision == "promoted" else first["contest"]["champion_checkpoint_id"]
    assert second["contest"]["champion_checkpoint_id"] == expected_champion
    full = dict(version=1, operation_id="uninterrupted", experiment="full", source=request["source"],
                target_episode=3, config=copy.deepcopy(request["training_config"]))
    full["config"].update(checkpoint_dir=str(Path(request["work_dir"]) / "full"), bootstrap_champion=False)
    direct = training.run_request(full, url)
    with AuditService(url) as audit:
        actual, _ = read_checkpoint(resolve_checkpoint_reference("checkpoint:" + response["learner_checkpoint_id"], audit), map_location="cpu")
        expected, _ = read_checkpoint(resolve_checkpoint_reference("checkpoint:" + direct["checkpoint_id"], audit), map_location="cpu")
        for key in ("online", "optimizer", "replay", "rng_state", "training_state"):
            assert_equal(actual[key], expected[key])
    assert search_loop.run_request(request, url, children=FixtureChildren()) == response
    assert entries(trace) == history

def test_resume_seed_applies_only_to_first_round_and_preserves_continuity(setup):
    url, request, trace = setup
    with AuditService(url) as audit:
        experiment = audit.ensure_experiment("rounds", {"rows": 2, "columns": 2, "max_turns": 0})
        source_path = resolve_checkpoint_reference("checkpoint:" + request["source"]["checkpoint_id"], audit)
        source = audit.import_checkpoint(experiment["id"], source_path)
        audit.bootstrap(experiment["id"], source["id"])
    request.update(total_episode=3, source=dict(mode="resume", checkpoint_id=source["id"], resume_seed=123))
    request["training_config"]["bootstrap_champion"] = False

    response = search_loop.run_request(request, url, children=FixtureChildren())
    assert response["status"] == "completed", response
    assert [round_["episode"] for round_ in response["rounds"]] == [2, 3]
    trains = [entry["request"] for entry in entries(trace) if entry["kind"] == "training"]
    assert len(trains) == 2
    assert trains[0]["source"] == request["source"]
    assert trains[1]["source"] == dict(mode="resume", checkpoint_id=response["rounds"][0]["learner_checkpoint_id"])

    full = dict(version=1, operation_id="uninterrupted-seeded-resume", experiment=request["experiment"],
                source=copy.deepcopy(request["source"]), target_episode=3,
                config=copy.deepcopy(request["training_config"]))
    full["config"]["checkpoint_dir"] = str(Path(request["work_dir"]) / "full")
    direct = training.run_request(full, url)
    assert direct["status"] == "completed", direct
    with AuditService(url) as audit:
        actual, _ = read_checkpoint(resolve_checkpoint_reference("checkpoint:" + response["learner_checkpoint_id"], audit), map_location="cpu")
        expected, _ = read_checkpoint(resolve_checkpoint_reference("checkpoint:" + direct["checkpoint_id"], audit), map_location="cpu")
        for key in ("online", "optimizer", "replay", "rng_state", "training_state"):
            assert_equal(actual[key], expected[key])


@pytest.mark.parametrize("kind", ["training", "evaluation", "promotion"])
def test_recover_child_commit_before_completion_or_ack(setup, monkeypatch, kind):
    url, request, trace = setup
    request["total_episode"] = 1
    monkeypatch.setenv("SEARCH_LOOP_TEST_CRASH", kind)
    paused = search_loop.run_request(request, url, children=FixtureChildren())
    assert paused["status"] == "paused", paused
    with AuditService(url) as audit:
        exp = audit.get_experiment("rounds")
        if kind == "promotion":
            assert len(audit.champion_history(exp["id"])) == 2
    complete = search_loop.run_request(request, url, children=FixtureChildren())
    assert complete["status"] == "completed", complete
    assert complete["rounds"][0]["decision"] == "promoted"
    with AuditService(url) as audit:
        assert (len(audit.list_attempts(exp["id"])), len(audit.list_evaluations(exp["id"])), len(audit.champion_history(exp["id"]))) == (2, 3, 2)
        assert len(audit.list_decisions(complete["rounds"][0]["results"]["screening"]["attempt_id"])) == 3
    assert sum(e["kind"] == kind for e in entries(trace)) == (3 if kind == "evaluation" else 2)

@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_graceful_signal_forwards_waits_and_resumes_same_target(setup, tmp_path, monkeypatch, stop_signal):
    url, request, trace = setup
    request.update(total_episode=2, round_episodes=2)
    monkeypatch.setenv("SEARCH_LOOP_TEST_SLOW", "1")
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    write_result(request_path, request)
    adapter = Path(__file__).with_name("search_loop_worker.py")
    process = subprocess.Popen([sys.executable, str(adapter), "--kind", "search_loop", "--request", str(request_path),
                                "--result", str(result_path), "--database-url", url], start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        ready = Path(request["work_dir"]) / search_loop.identifier(request["run_id"], "supervisor") / "ready"
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(.05)
        assert ready.exists()
        process.send_signal(stop_signal)
        assert process.wait(timeout=20) == 1
        paused = json.loads(result_path.read_text())
        assert paused["status"] == "paused" and paused["stage"] == "train"
        with AuditService(url) as audit:
            op = audit.get_operation(search_loop.identifier(request["run_id"], "round:1:training"))
            assert op["status"] == "interrupted"
            assert op["operation_result"]["episode"] == 1
        resumed = search_loop.run_request(request, url, children=FixtureChildren())
        assert resumed["status"] == "completed", resumed
        assert resumed["episode"] == 2
        trains = [e["request"] for e in entries(trace) if e["kind"] == "training"]
        assert len(trains) == 2 and trains[0] == trains[1]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

def test_single_supervisor_and_immutable_config(setup):
    url, request, trace = setup
    op_id = search_loop.identifier(request["run_id"], "supervisor")
    with AuditService(url) as audit, operation_lock(audit, op_id):
        with pytest.raises(ValueError, match="already running"):
            search_loop.run_request(request, url)
    with AuditService(url) as audit:
        experiment = audit.ensure_experiment(request["experiment"], {"rows": 2, "columns": 2, "max_turns": 0})
        audit.ensure_operation(op_id, experiment["id"], "supervisor", request)
    request["total_episode"] += 1
    with pytest.raises(ValueError, match="different request"):
        search_loop.run_request(request, url)

@pytest.mark.parametrize("change", [
    lambda r: r.update(round_episodes=0), lambda r: r.update(total_episode=True),
    lambda r: r["training_config"].pop("learning_rate"),
    lambda r: r["evaluation_config"]["screening"].update(rows=3),
    lambda r: r["gates"].update(promotion_min_match_score=.5),
    lambda r: r["evaluation_config"]["extended_head_to_head"].update(opening_random_moves=1),
])
def test_reject_invalid_request_before_spawning(setup, change):
    url, request, trace = setup
    change(request)
    with pytest.raises(ValueError):
        search_loop.run_request(request, url)
    assert not trace.exists()

@pytest.mark.parametrize("kind", ["training", "evaluation", "promotion"])
def test_recover_completed_child_before_supervisor_acknowledgement(setup, kind):
    url, request, trace = setup
    request["total_episode"] = 1
    class StopBeforeAcknowledgement(FixtureChildren):
        def invoke(self, child_kind, *args):
            code = super().invoke(child_kind, *args)
            if child_kind == kind:
                raise search_loop.Paused("supervisor stopped before acknowledging child")
            return code
    paused = search_loop.run_request(request, url, children=StopBeforeAcknowledgement())
    assert paused["status"] == "paused"
    before = entries(trace)
    completed = search_loop.run_request(request, url, children=FixtureChildren())
    assert completed["status"] == "completed", completed
    repeated_id = before[-1]["request"]["operation_id"]
    assert sum(e["request"]["operation_id"] == repeated_id for e in entries(trace)) == 1
    with AuditService(url) as audit:
        assert len(audit.champion_history(audit.get_experiment("rounds")["id"])) == 2


def test_external_promotion_race_starts_new_contest_without_retraining(setup):
    url, request, trace = setup
    request["total_episode"] = 1
    class ExternalPromotion(FixtureChildren):
        changed = False
        def invoke(self, kind, request_path, result_path, database_url):
            if kind == "promotion" and not self.changed:
                self.changed = True
                external = json.loads(request_path.read_text())
                external["operation_id"] = "external-actor-promotion"
                path = request_path.with_name("external.request.json")
                write_result(path, external)
                assert super().invoke(kind, path, path.with_name("external.result.json"), database_url) == 0
            return super().invoke(kind, request_path, result_path, database_url)
    completed = search_loop.run_request(request, url, children=ExternalPromotion())
    assert completed["status"] == "completed", completed
    calls = entries(trace)
    assert sum(e["kind"] == "training" for e in calls) == 1
    screens = [e["request"] for e in calls if e["request"].get("stage") == "screening"]
    assert len(screens) == 2
    assert screens[0]["contest"]["candidate_checkpoint_id"] == screens[1]["contest"]["candidate_checkpoint_id"]
    assert screens[0]["contest"]["expected_assignment_id"] != screens[1]["contest"]["expected_assignment_id"]
    with AuditService(url) as audit:
        state = audit.get_operation(search_loop.identifier(request["run_id"], "supervisor"))["progress"]
        assert len(state["discarded_contests"]) == 1
        old = state["discarded_contests"][0]["results"]["screening"]
        assert audit.get_attempt(old["attempt_id"])["status"] == "completed"
        assert len(audit.champion_history(audit.get_experiment("rounds")["id"])) == 3


def test_interrupted_evaluation_pauses_and_retries_fresh_suites(setup, monkeypatch):
    url, request, trace = setup
    request["total_episode"] = 1
    monkeypatch.setenv("SEARCH_LOOP_TEST_PARTIAL_EVALUATION", "1")
    paused = search_loop.run_request(request, url, children=FixtureChildren())
    assert paused["status"] == "paused" and paused["stage"] == "screening"
    with AuditService(url) as audit:
        state = audit.get_operation(search_loop.identifier(request["run_id"], "supervisor"))["progress"]
        operation = audit.get_operation(state["pending"]["request"]["operation_id"])
        assert operation["status"] == "interrupted"
        previous_seeds = operation["progress"]["suite_seeds"]
        previous_ids = operation["progress"]["evaluation_ids"]
    completed = search_loop.run_request(request, url, children=FixtureChildren())
    assert completed["status"] == "completed", completed
    screen = completed["rounds"][0]["results"]["screening"]
    assert screen["generation"] == 2
    assert set(screen["suite_seeds"]).isdisjoint(previous_seeds)
    assert sum(e["kind"] == "training" for e in entries(trace)) == 1
    with AuditService(url) as audit:
        assert all(audit.get_evaluation(i)["status"] == "interrupted" for i in previous_ids)


def test_cached_screening_supervisor_reuses_promoted_evidence_across_rounds(setup):
    url, request, trace = setup
    request["total_episode"] = 1
    first = search_loop.run_request(request, url, children=FixtureChildren())
    assert first["status"] == "completed"
    original = first["rounds"][0]
    request.update(run_id="cached-loop", total_episode=3,
                   source=dict(mode="resume", checkpoint_id=original["learner_checkpoint_id"]))
    request["training_config"]["bootstrap_champion"] = False
    request["evaluation_config"]["screening"].update(reuse_champion_screening=True, suites=4, evaluation_workers=4)
    response = search_loop.run_request(request, url, children=FixtureChildren())
    assert response["status"] == "completed", response
    assert [r["decision"] for r in response["rounds"]] == ["promoted", "promoted"]
    baseline = original["results"]["screening"]["evaluation_ids"][1]
    with AuditService(url) as audit:
        for round_ in response["rounds"]:
            screen = round_["results"]["screening"]
            assert screen["evaluation_ids"][0] == baseline
            assert screen["reused_evaluation_ids"] == [baseline]
            assert len(screen["executed_evaluation_ids"]) == 1
            assert round_["results"]["promotion"]["config"]["reuse_champion_screening"] is True
            baseline = screen["evaluation_ids"][1]
            batches = [b for b in audit.list_evaluations(audit.get_experiment("rounds")["id"])
                       if b["config"].get("operation_id") == screen["operation_id"]]
            assert len(batches) == 1
