import copy

import pytest

from dots_cordon_ml import evaluation, promotion_evaluation as shared
from dots_cordon_ml.audit import AuditService, Database
from dots_cordon_ml.checkpoint import read_checkpoint
from dots_cordon_ml.operations import write_result
from test_promote_run import checkpoint, result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'audit.sqlite3'}"
    db = Database(url)
    db.upgrade()
    db.close()
    with AuditService(url) as audit:
        experiment = audit.ensure_experiment("rounds", {"rows": 7, "columns": 7, "max_turns": 0})
        champion = audit.import_checkpoint(experiment["id"], checkpoint(tmp_path / "champion.pt", 0, False))
        candidate = audit.import_checkpoint(experiment["id"], checkpoint(tmp_path / "candidate.pt", 100))
        assignment = audit.bootstrap(experiment["id"], champion["id"])
    calls = []
    def random_suites(args, path, device, seeds, games):
        calls.append(("random", seeds, str(path)))
        _, metadata = read_checkpoint(path, map_location="cpu")
        suites = tuple(result(30) for _ in seeds)
        return shared.EvaluatedCheckpoint(path, metadata, suites, shared.combine_evaluation_results(suites))
    def challenge(args, candidate, champion, device, seed, games, opening_random_moves):
        calls.append(("head_to_head", (seed,), str(candidate.path), str(champion.path)))
        return result(26)
    monkeypatch.setattr(shared, "_evaluate_against_random_suites", random_suites)
    monkeypatch.setattr(shared, "_evaluate_head_to_head_suite", challenge)
    request = {
        "version": 1, "operation_id": "screen-1", "experiment": "rounds",
        "contest": {"id": "contest-1", "candidate_checkpoint_id": candidate["id"],
                    "expected_assignment_id": assignment["id"], "champion_checkpoint_id": champion["id"]},
        "stage": "screening",
        "config": {"server": "127.0.0.1:1", "rows": 7, "columns": 7, "max_turns": 0,
                   "mode": "policy", "device": "cpu", "games": 100, "suites": 3,
                   "seed": 30000001, "opening_random_moves": 0, "paired_seats": True,
                   "evaluation_workers": 2, "rpc_timeout": 10.0},
    }
    return url, request, calls


def test_screen_and_challenges_share_immutable_contest_and_never_promote(setup):
    url, request, calls = setup
    screen = evaluation.run_request(request, url)
    assert screen["status"] == "completed"
    assert screen["candidate_checkpoint_id"] != screen["source_checkpoint_id"]
    champion, candidate = screen["evaluations"]
    assert champion["model_kind"] == "dqn"
    assert candidate["model_kind"] == "policy_value"
    assert candidate["aggregate"]["overall"]["match_score"] == 0.6
    assert candidate["aggregate"]["as_player_0"]["games"] == 150
    assert [s["definition"] for s in champion["suites"]] == [s["definition"] for s in candidate["suites"]]
    for index, stage in enumerate(("head_to_head", "extended_head_to_head"), 4):
        job = copy.deepcopy(request)
        job.update(stage=stage, operation_id=stage)
        job["config"].update(seed=index * 10000000 + 1, opening_random_moves=4)
        response = evaluation.run_request(job, url)
        assert response["attempt_id"] == screen["attempt_id"]
        assert response["candidate_checkpoint_id"] == screen["candidate_checkpoint_id"]
        batch = response["evaluations"][0]
        assert batch["opponent_checkpoint_id"] == request["contest"]["champion_checkpoint_id"]
        assert batch["opponent_model_kind"] == "dqn"
        assert batch["aggregate"]["overall"]["match_score"] == 0.52
        assert batch["suites"][0]["definition"]["opening_random_moves"] == 4
    with AuditService(url) as audit:
        experiment = audit.get_experiment("rounds")
        assert audit.current_champion(experiment["id"])["id"] == request["contest"]["expected_assignment_id"]
        assert len(audit.champion_history(experiment["id"])) == 1
        assert audit.list_decisions(screen["attempt_id"]) == []
        cloned = audit.get_checkpoint(screen["candidate_checkpoint_id"])
        original = audit.get_checkpoint(screen["source_checkpoint_id"])
        assert cloned["parent_checkpoint_id"] == original["id"]
        assert cloned["checkpoint_blob_id"] == original["checkpoint_blob_id"]
        assert len(audit.list_attempts(experiment["id"])) == 1


def test_completed_operation_replay_and_result_file_failure(setup, tmp_path):
    url, request, calls = setup
    first = evaluation.run_request(request, url)
    count = len(calls)
    with pytest.raises(OSError):
        write_result(tmp_path, first)
    assert evaluation.run_request(request, url) == first
    assert len(calls) == count
    request["config"]["games"] = 200
    with pytest.raises(ValueError, match="different request"):
        evaluation.run_request(request, url)


def test_failed_pair_replacement_preserves_evidence_and_reserves_fresh_seeds(setup, monkeypatch):
    url, request, calls = setup
    execute = shared._evaluate_random_screen
    def interrupted(args, paths, device, seeds, games, *, audit_service, evaluation_ids):
        from dots_cordon_ml.audit.integration import evaluation_result_dict
        for index in range(len(seeds)):
            audit_service.complete_suite(evaluation_ids[0], index, evaluation_result_dict(result(30)))
        audit_service.complete_suite(evaluation_ids[1], 0, evaluation_result_dict(result(30)))
        raise KeyboardInterrupt
    monkeypatch.setattr(shared, "_evaluate_random_screen", interrupted)
    with pytest.raises(KeyboardInterrupt):
        evaluation.run_request(request, url)
    with AuditService(url) as audit:
        partial = audit.get_operation(request["operation_id"])
        assert partial["status"] == "interrupted"
        previous_ids = partial["progress"]["evaluation_ids"]
        assert audit.get_evaluation(previous_ids[0])["status"] == "completed"
        assert audit.get_evaluation(previous_ids[1])["status"] == "interrupted"
    monkeypatch.setattr(shared, "_evaluate_random_screen", execute)
    completed = evaluation.run_request(request, url)
    assert completed["generation"] == 2
    assert set(completed["suite_seeds"]).isdisjoint(partial["progress"]["suite_seeds"])
    for batch in completed["evaluations"]:
        assert batch["config"]["replaces"] == previous_ids
    with AuditService(url) as audit:
        assert len(audit.list_evaluations(audit.get_experiment("rounds")["id"])) == 4


def test_recovers_completed_batches_before_operation_result_commit(setup, monkeypatch):
    url, request, calls = setup
    update = AuditService.update_operation
    def crash(self, operation_id, **kwargs):
        if kwargs["status"] == "completed":
            raise RuntimeError("process died before result")
        return update(self, operation_id, **kwargs)
    monkeypatch.setattr(AuditService, "update_operation", crash)
    with pytest.raises(RuntimeError, match="process died"):
        evaluation.run_request(request, url)
    count = len(calls)
    monkeypatch.setattr(AuditService, "update_operation", update)
    completed = evaluation.run_request(request, url)
    assert completed["status"] == "completed"
    assert completed["generation"] == 1
    assert len(calls) == count


def test_death_between_batch_creations_uses_fresh_pair(setup, monkeypatch):
    url, request, calls = setup
    create = AuditService.create_evaluation
    count = 0
    def crash(self, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("died during batch creation")
        return create(self, *args, **kwargs)
    monkeypatch.setattr(AuditService, "create_evaluation", crash)
    with pytest.raises(RuntimeError, match="died during"):
        evaluation.run_request(request, url)
    monkeypatch.setattr(AuditService, "create_evaluation", create)
    response = evaluation.run_request(request, url)
    assert response["generation"] == 2
    assert response["suite_seeds"] == [30000004, 30000005, 30000006]
    with AuditService(url) as audit:
        batches = audit.list_evaluations(audit.get_experiment("rounds")["id"])
        assert [b["status"] for b in batches].count("interrupted") == 1


def test_same_contest_cannot_change_participants(setup):
    url, request, calls = setup
    evaluation.run_request(request, url)
    request["operation_id"] = "changed-contest"
    request["contest"]["candidate_checkpoint_id"] = request["contest"]["champion_checkpoint_id"]
    with pytest.raises(ValueError, match="different content"):
        evaluation.run_request(request, url)


@pytest.mark.parametrize("change", [
    lambda r: r["config"].update(mode="search"),
    lambda r: r["config"].update(paired_seats=False),
    lambda r: r["config"].update(opening_random_moves=4),
    lambda r: r["config"].update(rows=15),
    lambda r: r["config"].update(max_turns=3),
    lambda r: r["config"].update(games=3),
    lambda r: r["config"].update(games=True),
    lambda r: r["config"].update(suites=0),
    lambda r: r["config"].update(rpc_timeout=float("nan")),
    lambda r: r["config"].update(device="auto"),
    lambda r: r["config"].pop("seed"),
    lambda r: r["contest"].update(expected_assignment_id="unknown"),
])
def test_rejects_ambiguous_or_incompatible_requests(setup, change):
    url, request, calls = setup
    change(request)
    with pytest.raises(ValueError):
        evaluation.run_request(request, url)
    assert not calls


def test_custom_overlapping_seed_bases_still_produce_independent_stages(setup):
    url, request, calls = setup
    first = evaluation.run_request(request, url)
    request.update(operation_id="initial", stage="head_to_head")
    second = evaluation.run_request(request, url)
    request.update(operation_id="extended", stage="extended_head_to_head")
    third = evaluation.run_request(request, url)
    assert set(first["suite_seeds"]).isdisjoint(second["suite_seeds"])
    assert set(first["suite_seeds"]).isdisjoint(third["suite_seeds"])
    assert set(second["suite_seeds"]).isdisjoint(third["suite_seeds"])


def test_concurrent_operation_and_contest_locks(setup):
    from dots_cordon_ml.operations import operation_lock
    url, request, calls = setup
    with AuditService(url) as audit:
        with operation_lock(audit, request["operation_id"]):
            with pytest.raises(ValueError, match="already running"):
                evaluation.run_request(request, url)
        experiment = audit.get_experiment("rounds")
        contest_id = evaluation.contest_attempt_id(experiment["id"], request["contest"]["id"])
        with operation_lock(audit, "contest:" + contest_id):
            with pytest.raises(ValueError, match="already running"):
                evaluation.run_request(request, url)
    assert not calls
