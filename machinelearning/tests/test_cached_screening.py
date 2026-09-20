import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading

import pytest

from dots_cordon_ml import evaluation, promotion, promotion_evaluation as shared
from dots_cordon_ml.audit import AuditService
from dots_cordon_ml.audit.integration import evaluation_result_dict
from dots_cordon_ml.cached_screening import select_champion_baseline
from dots_cordon_ml.checkpoint import read_checkpoint
from dots_cordon_ml.self_play import EvaluationResult, MatchStats
from test_evaluation_operation import setup
from test_promotion_operation import evidence_request
from test_promote_run import result


def cached_job(setup):
    url, request, calls = setup
    baseline = evaluation.run_request(request, url)
    request = copy.deepcopy(request)
    request.update(operation_id="cached-screen", contest={**request["contest"], "id": "cached-contest"})
    request["config"]["reuse_champion_screening"] = True
    return baseline, request


def wins_result(games, losses=0):
    first = MatchStats(games // 2, games // 2 - losses, 0, losses, 0, 0)
    second = MatchStats(games // 2, games // 2, 0, 0, 0, 0)
    return EvaluationResult(MatchStats(games, games - losses, 0, losses, 0, 0), first, second)


def test_dispatches_only_four_candidate_suites_in_parallel(setup, monkeypatch):
    url, _, calls = setup
    baseline, job = cached_job(setup)
    job["config"].update(games=150, suites=4, evaluation_workers=4)
    dispatches, workers = [], []
    barrier = threading.Barrier(4, timeout=5)
    @contextmanager
    def executor(device, count):
        workers.append(count)
        with ThreadPoolExecutor(max_workers=count) as pool:
            yield pool
    def play(args, path, device, seeds, games):
        _, metadata = read_checkpoint(path, map_location="cpu")
        dispatches.append((metadata.episode, seeds, games))
        barrier.wait()
        suites = (wins_result(games),)
        return shared.EvaluatedCheckpoint(path, metadata, suites, suites[0])
    monkeypatch.setattr(shared, "_evaluation_executor", executor)
    monkeypatch.setattr(shared, "_evaluate_against_random_suites", play)
    response = evaluation.run_request(job, url)
    assert workers == [4]
    assert len(dispatches) == 4
    assert all(episode == 100 and len(seeds) == 1 and games == 150 for episode, seeds, games in dispatches)
    assert response["evaluation_ids"][0] == baseline["evaluation_ids"][0]
    assert response["reused_evaluation_ids"] == baseline["evaluation_ids"][:1]
    assert response["executed_evaluation_ids"] == response["evaluation_ids"][1:]
    assert [item["reused"] for item in response["evaluations"]] == [True, False]
    assert response["evaluations"][1]["aggregate"]["overall"]["games"] == 600
    assert evaluation.run_request(job, url) == response
    assert len(dispatches) == 4
    with AuditService(url) as audit:
        assert len(audit.list_evaluations(audit.get_experiment("rounds")["id"])) == 3


def test_no_baseline_fails_without_playing_games(setup):
    url, job, calls = setup
    job["config"]["reuse_champion_screening"] = True
    with pytest.raises(ValueError, match="No compatible completed champion"):
        evaluation.run_request(job, url)
    assert calls == []


def test_retry_keeps_pinned_baseline_and_replaces_only_candidate(setup, monkeypatch):
    url, _, calls = setup
    baseline, job = cached_job(setup)
    run = shared._evaluate_random_screen
    def interrupt(args, paths, device, seeds, games, *, audit_service, evaluation_ids):
        assert len(paths) == len(evaluation_ids) == 1
        audit_service.complete_suite(evaluation_ids[0], 0, evaluation_result_dict(result(30)))
        raise KeyboardInterrupt
    monkeypatch.setattr(shared, "_evaluate_random_screen", interrupt)
    with pytest.raises(KeyboardInterrupt):
        evaluation.run_request(job, url)
    with AuditService(url) as audit:
        old = audit.get_operation(job["operation_id"])["progress"]
        assert old["champion_screening_id"] == baseline["evaluation_ids"][0]
    monkeypatch.setattr(shared, "_evaluate_random_screen", run)
    newer = copy.deepcopy(setup[1])
    newer.update(operation_id="later-screen", contest={**newer["contest"], "id": "later-contest"})
    evaluation.run_request(newer, url)
    # Resume must not re-select, regardless of any newly available evidence.
    monkeypatch.setattr(evaluation, "select_champion_baseline", lambda *a: pytest.fail("baseline was reselected"))
    response = evaluation.run_request(job, url)
    assert response["generation"] == 2
    assert response["evaluation_ids"][0] == old["champion_screening_id"]
    assert response["evaluation_ids"][1] != old["evaluation_ids"][1]
    with AuditService(url) as audit:
        assert audit.get_evaluation(old["champion_screening_id"])["status"] == "completed"


def cached_promotion_request(setup, monkeypatch, **kwargs):
    _, job = cached_job(setup)
    request = evidence_request((setup[0], job, setup[2]), monkeypatch, **kwargs)
    request["config"]["reuse_champion_screening"] = True
    return request


@pytest.mark.parametrize("initial,draws,extra,expected", [
    (26, 0, None, "promoted"), (25, 0, None, "rejected"),
    (25, 1, None, "extended_required"), (25, 1, (25, 1), "promoted"),
    (25, 1, (25, 0), "rejected"), (24, 0, None, "rejected"),
])
def test_cached_mode_keeps_h2h_criteria(setup, monkeypatch, initial, draws, extra, expected):
    request = cached_promotion_request(setup, monkeypatch, initial=initial, initial_draws=draws, extended=extra)
    assert promotion.run_request(request, setup[0])["decision"] == expected


@pytest.mark.parametrize("losses,expected", [(3, "promoted"), (4, "rejected")])
def test_600_game_score_tolerance_boundary_unchanged(setup, monkeypatch, losses, expected):
    url, job, calls = setup
    def play(args, path, device, seeds, games):
        _, metadata = read_checkpoint(path, map_location="cpu")
        suites = tuple(wins_result(games, losses if games == 150 and seed == selected_seed[0] else 0) for seed in seeds)
        return shared.EvaluatedCheckpoint(path, metadata, suites, shared.combine_evaluation_results(suites))
    selected_seed = [30000004]
    monkeypatch.setattr(shared, "_evaluate_against_random_suites", play)
    baseline, cached = cached_job(setup)
    cached["config"].update(games=150, suites=4)
    screen = evaluation.run_request(cached, url)
    # Build a normal challenge/promotion request, replacing only the screening evidence budget.
    request = evidence_request((url, job, calls), monkeypatch)
    challenge_job = copy.deepcopy(cached)
    challenge_job.update(operation_id="cached-h2h", stage="head_to_head")
    challenge_job["config"].update(games=100, suites=3, seed=40000001, opening_random_moves=4)
    challenge_job["config"].pop("reuse_champion_screening")
    challenge = evaluation.run_request(challenge_job, url)
    request.update(attempt_id=screen["attempt_id"], candidate_checkpoint_id=screen["candidate_checkpoint_id"])
    request["evidence"].update(champion_screening=screen["evaluation_ids"][0], candidate_screening=screen["evaluation_ids"][1],
                               initial_head_to_head=challenge["evaluation_ids"][0])
    request["config"].update(reuse_champion_screening=True, screen_games=150, screen_suites=4,
                             screen_seeds=screen["suite_seeds"], head_to_head_seeds=challenge["suite_seeds"], screen_max_regression=.005)
    assert promotion.run_request(request, url)["decision"] == expected


@pytest.mark.parametrize("change", [
    lambda b: b.update(status="running"),
    lambda b: b.update(checkpoint_id="wrong"),
    lambda b: b["config"].update(mode="search"),
    lambda b: b["config"].update(max_turns=1),
    lambda b: b["config"].update(operation_id="missing"),
    lambda b: b["suites"][0].update(player_0_losses=0),
    lambda b: b["suites"][0]["definition"].update(seed=123),
])
def test_tampered_historical_baseline_cannot_promote(setup, monkeypatch, change):
    request = cached_promotion_request(setup, monkeypatch)
    get = AuditService.get_evaluation
    def altered(self, identifier):
        batch = get(self, identifier)
        if identifier == request["evidence"]["champion_screening"]:
            change(batch)
        return batch
    monkeypatch.setattr(AuditService, "get_evaluation", altered)
    with pytest.raises(ValueError):
        promotion.run_request(request, setup[0])


def test_cannot_substitute_another_valid_baseline_after_screening(setup, monkeypatch):
    request = cached_promotion_request(setup, monkeypatch)
    other = copy.deepcopy(setup[1])
    other.update(operation_id="later-screen", contest={**other["contest"], "id": "later-contest"})
    later = evaluation.run_request(other, setup[0])
    request["evidence"]["champion_screening"] = later["evaluation_ids"][0]
    with pytest.raises(ValueError, match="pinned champion baseline"):
        promotion.run_request(request, setup[0])


def test_prefers_champions_own_qualifying_screen_over_later_rescreen(setup, monkeypatch):
    request = evidence_request(setup, monkeypatch)
    promoted = promotion.run_request(request, setup[0])
    later = copy.deepcopy(setup[1])
    later.update(operation_id="promoted-rescreen", contest={**later["contest"], "id": "promoted-rescreen",
                  "champion_checkpoint_id": promoted["assignment"]["checkpoint_id"],
                  "expected_assignment_id": promoted["assignment"]["id"]})
    rescreen = evaluation.run_request(later, setup[0])
    with AuditService(setup[0]) as audit:
        baseline = select_champion_baseline(audit, audit.get_experiment("rounds")["id"], promoted["assignment"], request["config"])
        assert baseline["id"] == request["evidence"]["candidate_screening"]
        assert baseline["id"] != rescreen["evaluation_ids"][0]


@pytest.mark.parametrize("field,value", [("status", "interrupted"), ("opponent_kind", "checkpoint")])
def test_invalid_available_baseline_fails_before_candidate_execution(setup, monkeypatch, field, value):
    baseline, job = cached_job(setup)
    get = AuditService.get_evaluation
    def altered(self, identifier):
        batch = get(self, identifier)
        if identifier == baseline["evaluation_ids"][0]:
            batch[field] = value
        return batch
    monkeypatch.setattr(AuditService, "get_evaluation", altered)
    count = len(setup[2])
    with pytest.raises(ValueError, match="No compatible completed champion"):
        evaluation.run_request(job, setup[0])
    assert len(setup[2]) == count


def test_audit_cached_policy_cannot_bypass_matched_operation_provenance(setup):
    url, job, calls = setup
    screen = evaluation.run_request(job, url)
    with AuditService(url) as audit:
        with pytest.raises(ValueError, match="pinned champion baseline"):
            audit.record_decision(screen["attempt_id"], screen["candidate_checkpoint_id"],
                stage="screening", result="qualified", candidate_evaluation_id=screen["evaluation_ids"][1],
                champion_evaluation_id=screen["evaluation_ids"][0],
                policy={"screen_max_regression": .005, "reuse_champion_screening": True})


def test_cached_complete_batch_recovers_after_operation_acknowledgement_failure(setup, monkeypatch):
    url, _, calls = setup
    baseline, job = cached_job(setup)
    update = AuditService.update_operation
    def crash(self, operation_id, **kwargs):
        if operation_id == job["operation_id"] and kwargs["status"] == "completed":
            raise RuntimeError("died before cached acknowledgement")
        return update(self, operation_id, **kwargs)
    monkeypatch.setattr(AuditService, "update_operation", crash)
    with pytest.raises(RuntimeError, match="died before cached"):
        evaluation.run_request(job, url)
    count = len(calls)
    monkeypatch.setattr(AuditService, "update_operation", update)
    response = evaluation.run_request(job, url)
    assert response["status"] == "completed"
    assert response["generation"] == 1
    assert response["evaluation_ids"][0] == baseline["evaluation_ids"][0]
    assert len(calls) == count
    with AuditService(url) as audit:
        assert len(audit.list_evaluations(audit.get_experiment("rounds")["id"])) == 3
