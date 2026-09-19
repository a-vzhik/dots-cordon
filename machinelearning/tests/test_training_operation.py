import copy
import signal

import pytest
import torch

from dots_cordon_ml import search_train, training
from dots_cordon_ml.audit import AuditService
from dots_cordon_ml.audit.database import Database
from dots_cordon_ml.operations import operation_lock, write_result
from test_search_training import SmallEnvironment, arguments, load


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(search_train, "GameEnvironment", SmallEnvironment)
    url = f"sqlite:///{tmp_path / 'audit.sqlite3'}"
    db = Database(url)
    db.upgrade()
    db.close()
    search_train.run(arguments(tmp_path / "source", "--episodes", "1"))
    with AuditService(url) as audit:
        source_exp = audit.ensure_experiment("source", {"rows": 2, "columns": 2})
        source = audit.import_checkpoint(source_exp["id"], tmp_path / "source/search-latest.pt")
    config = {key: getattr(arguments(tmp_path / "round"), key) for key in training.TRAINING_CONFIG_FIELDS}
    config["checkpoint_dir"] = str(config["checkpoint_dir"])
    request = dict(version=1, operation_id="round-1", experiment="rounds",
                   source={"mode": "initialize", "checkpoint_id": source["id"]},
                   target_episode=3, config=config)
    return url, request


def assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_equal(left, right)
    else:
        assert actual == expected


def test_round_boundaries_preserve_entire_learner_and_disable_evaluation(setup, tmp_path, monkeypatch):
    url, request = setup
    monkeypatch.setattr(search_train, "evaluate", lambda *a, **kw: pytest.fail("inline evaluation"))
    full = copy.deepcopy(request)
    full.update(operation_id="full", experiment="full")
    full["config"]["checkpoint_dir"] = str(tmp_path / "full")
    training.run_request(full, url)
    request["target_episode"] = 1
    first = training.run_request(request, url)
    second = copy.deepcopy(request)
    second.update(operation_id="round-2", target_episode=3,
                  source={"mode": "resume", "checkpoint_id": first["checkpoint_id"]})
    result = training.run_request(second, url)
    assert result["status"] == "completed"
    assert result["episode"] == 3
    assert len(result["checkpoint_sha256"]) == 64
    actual, expected = load(tmp_path / "round"), load(tmp_path / "full")
    for key in ("online", "optimizer", "replay", "rng_state", "training_state"):
        assert_equal(actual[key], expected[key])
    with AuditService(url) as audit:
        attempts = audit.list_attempts(audit.get_experiment("rounds")["id"])
        assert len(attempts) == 2
        assert attempts[-1]["starting_checkpoint_id"] == first["checkpoint_id"]
        assert attempts[-1]["start_episode"] == 1


def test_duplicate_and_changed_requests(setup, monkeypatch):
    url, request = setup
    first = training.run_request(request, url)
    monkeypatch.setattr(search_train, "run", lambda *a: pytest.fail("duplicate trained again"))
    assert training.run_request(request, url) == first
    request["config"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="different request"):
        training.run_request(request, url)


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_interrupted_retry_keeps_original_target_and_attempt_history(setup, monkeypatch, stop_signal):
    url, request = setup
    collect = search_train.collect_episode
    def stop_after_game(*args, **kwargs):
        result = collect(*args, **kwargs)
        signal.raise_signal(stop_signal)
        return result
    previous = signal.getsignal(stop_signal)
    monkeypatch.setattr(search_train, "collect_episode", stop_after_game)
    partial = training.run_request(request, url)
    assert signal.getsignal(stop_signal) == previous
    assert partial["status"] == "interrupted"
    assert partial["episode"] == 1
    assert partial["target_episode"] == 3
    monkeypatch.setattr(search_train, "collect_episode", collect)
    completed = training.run_request(request, url)
    assert completed["status"] == "completed"
    assert completed["episode"] == 3
    with AuditService(url) as audit:
        attempts = audit.list_attempts(audit.get_experiment("rounds")["id"])
        assert [a["status"] for a in attempts] == ["interrupted", "completed"]
        assert attempts[1]["starting_checkpoint_id"] == partial["checkpoint_id"]


def test_recovers_checkpoint_commit_before_result_and_attempt_completion(setup, monkeypatch):
    url, request = setup
    original = AuditService.update_attempt
    def crash(self, attempt_id, **fields):
        if fields.get("status") == "completed":
            raise RuntimeError("process died after checkpoint commit")
        return original(self, attempt_id, **fields)
    monkeypatch.setattr(AuditService, "update_attempt", crash)
    with pytest.raises(RuntimeError, match="process died"):
        training.run_request(request, url)
    monkeypatch.setattr(AuditService, "update_attempt", original)
    monkeypatch.setattr(search_train, "run", lambda *a: pytest.fail("committed round repeated"))
    completed = training.run_request(request, url)
    assert completed["status"] == "completed"
    assert completed["episode"] == 3
    with AuditService(url) as audit:
        assert len(audit.list_attempts(audit.get_experiment("rounds")["id"])) == 1
        assert audit.get_operation(request["operation_id"])["operation_result"] == completed


def test_recovers_periodic_checkpoint_after_failure_without_local_files(setup, tmp_path, monkeypatch):
    url, request = setup
    collect = search_train.collect_episode
    calls = 0
    def fail_second_game(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("engine disconnected")
        return collect(*args, **kwargs)
    monkeypatch.setattr(search_train, "collect_episode", fail_second_game)
    with pytest.raises(RuntimeError, match="engine disconnected"):
        training.run_request(request, url)
    with AuditService(url) as audit:
        operation = audit.get_operation(request["operation_id"])
        assert operation["status"] == "failed"
        partial = operation["operation_result"]
        assert partial["episode"] == 1
    for path in (tmp_path / "round").iterdir():
        path.unlink()
    monkeypatch.setattr(search_train, "collect_episode", collect)
    result = training.run_request(request, url)
    assert result["status"] == "completed"
    assert result["episode"] == 3
    with AuditService(url) as audit:
        resumed = audit.get_attempt(result["attempt_id"])
        assert resumed["starting_checkpoint_id"] == partial["checkpoint_id"]
        assert audit.get_attempt(partial["attempt_id"])["status"] == "failed"


def test_local_duplicate_lock_and_acknowledgement_recovery(setup, tmp_path):
    url, request = setup
    with AuditService(url) as audit, operation_lock(audit, request["operation_id"]):
        with pytest.raises(ValueError, match="already running"):
            training.run_request(request, url)
    result = training.run_request(request, url)
    bad = tmp_path / "directory"
    bad.mkdir()
    with pytest.raises(OSError):
        write_result(bad, result)
    assert training.run_request(request, url) == result


@pytest.mark.parametrize("change", [
    lambda r: r["config"].pop("learning_rate"),
    lambda r: r["config"].update(eval_every=1),
    lambda r: r["config"].update(device="auto"),
    lambda r: r["config"].update(checkpoint_dir="relative"),
    lambda r: r["config"].update(learning_rate=float("nan")),
    lambda r: r["config"].update(simulations=True),
    lambda r: r.update(config=4),
    lambda r: r.update(config=[]),
    lambda r: r.update(source={"mode": [], "checkpoint_id": "x"}),
    lambda r: r.update(source={"mode": "resume", "reference": "champion:source"}),
])
def test_requires_complete_explicit_request(setup, change):
    url, request = setup
    change(request)
    with pytest.raises(ValueError):
        training.run_request(request, url)
