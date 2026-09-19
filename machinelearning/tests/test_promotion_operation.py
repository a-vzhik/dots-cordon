import copy

import pytest

from dots_cordon_ml import evaluation, promotion, promotion_evaluation as shared
from dots_cordon_ml.audit import AuditService
from dots_cordon_ml.audit.database import PromotionConflict
from test_evaluation_operation import setup
from test_promote_run import result


def evidence_request(setup, monkeypatch, *, initial=26, initial_draws=0, extended=None):
    url, job, calls = setup
    screen = evaluation.run_request(job, url)
    monkeypatch.setattr(shared, "_evaluate_head_to_head_suite", lambda *args: result(initial, initial_draws))
    challenge_job = copy.deepcopy(job)
    challenge_job.update(operation_id="initial", stage="head_to_head")
    challenge_job["config"].update(seed=40000001, opening_random_moves=4)
    challenge = evaluation.run_request(challenge_job, url)
    extra = None
    if extended is not None:
        monkeypatch.setattr(shared, "_evaluate_head_to_head_suite", lambda *args: result(*extended))
        extended_job = copy.deepcopy(challenge_job)
        extended_job.update(operation_id="extended", stage="extended_head_to_head")
        extended_job["config"].update(seed=50000001, suites=10)
        extra = evaluation.run_request(extended_job, url)
    return {
        "version": 1, "operation_id": "promote-1", "experiment": "rounds",
        "attempt_id": screen["attempt_id"], "candidate_checkpoint_id": screen["candidate_checkpoint_id"],
        "expected_assignment_id": screen["expected_assignment_id"],
        "evidence": {"champion_screening": screen["evaluation_ids"][0], "candidate_screening": screen["evaluation_ids"][1],
                     "initial_head_to_head": challenge["evaluation_ids"][0],
                     "extended_head_to_head": extra["evaluation_ids"][0] if extra else None},
        "config": {"rows": 7, "columns": 7, "max_turns": 0, "mode": "policy", "screen_max_regression": .003,
                   "promotion_min_match_score": .52, "promotion_min_suite_wins": 2, "opening_random_moves": 4,
                   "screen_games": 100, "screen_suites": 3, "screen_seeds": screen["suite_seeds"],
                   "head_to_head_games": 100, "head_to_head_suites": 3, "head_to_head_seeds": challenge["suite_seeds"],
                   "extended_head_to_head_games": 100, "extended_head_to_head_suites": 10,
                   "extended_head_to_head_seeds": extra["suite_seeds"] if extra else []},
    }


@pytest.mark.parametrize("initial,draws,extra,expected", [
    (26, 0, None, "promoted"), (25, 0, None, "rejected"),
    (25, 1, None, "extended_required"), (25, 1, (25, 1), "promoted"),
    (25, 1, (25, 0), "rejected"), (24, 0, None, "rejected"),
])
def test_gate_boundaries_and_idempotence(setup, monkeypatch, initial, draws, extra, expected):
    url, _, calls = setup
    request = evidence_request(setup, monkeypatch, initial=initial, initial_draws=draws, extended=extra)
    before = len(calls)
    response = promotion.run_request(request, url)
    assert response["decision"] == expected
    assert promotion.run_request(request, url) == response
    assert len(calls) == before
    with AuditService(url) as audit:
        assert len(audit.champion_history(audit.get_experiment("rounds")["id"])) == (2 if expected == "promoted" else 1)
        assert audit.get_attempt(request["attempt_id"])["status"] == ("running" if expected == "extended_required" else "completed")
        assert audit.list_decisions(request["attempt_id"])[0]["stage"] == "screening"


def test_extended_requires_borderline_initial(setup, monkeypatch):
    request = evidence_request(setup, monkeypatch, extended=(30, 0))
    with pytest.raises(ValueError, match="borderline"):
        promotion.run_request(request, setup[0])


def test_postcommit_recovery_and_export_repair(setup, monkeypatch, tmp_path):
    url, _, _ = setup
    request = evidence_request(setup, monkeypatch)
    update = AuditService.update_operation
    def crash(self, operation_id, **kwargs):
        if kwargs["status"] == "completed":
            raise RuntimeError("died before acknowledgement")
        return update(self, operation_id, **kwargs)
    monkeypatch.setattr(AuditService, "update_operation", crash)
    with pytest.raises(RuntimeError, match="died before"):
        promotion.run_request(request, url)
    monkeypatch.setattr(AuditService, "update_operation", update)
    with pytest.raises(OSError):
        promotion.run_request(request, url, output=tmp_path)
    with AuditService(url) as audit:
        assert audit.get_operation(request["operation_id"])["status"] == "completed"
        assert len(audit.champion_history(audit.get_experiment("rounds")["id"])) == 2
    output = tmp_path / "repair.pt"
    response = promotion.run_request(request, url, output=output)
    assert response["decision"] == "promoted" and output.is_file()
    changed = copy.deepcopy(request)
    changed["config"]["screen_max_regression"] = 0.1
    with pytest.raises(ValueError, match="different request"):
        promotion.run_request(changed, url)


@pytest.mark.parametrize("change", [
    lambda r: r.update(passed=True), lambda r: r["config"].update(mode="search"),
    lambda r: r["config"].update(max_turns=2), lambda r: r["config"].update(screen_games=200),
    lambda r: r["config"].update(opening_random_moves=3),
    lambda r: r["config"].update(head_to_head_seeds=[1, 2, 3]),
    lambda r: r["config"].update(head_to_head_seeds=r["config"]["screen_seeds"]),
    lambda r: r["evidence"].update(candidate_screening=r["evidence"]["champion_screening"]),
    lambda r: r["evidence"].update(initial_head_to_head=r["evidence"]["candidate_screening"]),
    lambda r: r.update(expected_assignment_id="missing"),
    lambda r: r["config"].update(screen_max_regression=float("nan")),
    lambda r: r["config"].update(promotion_min_suite_wins=0),
])
def test_rejects_mismatched_or_malformed_evidence(setup, monkeypatch, change):
    request = evidence_request(setup, monkeypatch)
    change(request)
    with pytest.raises(ValueError):
        promotion.run_request(request, setup[0])
    with AuditService(setup[0]) as audit:
        assert len(audit.champion_history(audit.get_experiment("rounds")["id"])) == 1


def test_stale_contest_fails_without_another_generation(setup, monkeypatch):
    request = evidence_request(setup, monkeypatch)
    promotion.run_request(request, setup[0])
    other = copy.deepcopy(request)
    other["operation_id"] = "another-promotion"
    with pytest.raises(PromotionConflict):
        promotion.run_request(other, setup[0])
    with AuditService(setup[0]) as audit:
        assert len(audit.champion_history(audit.get_experiment("rounds")["id"])) == 2


@pytest.mark.parametrize("initial", [24, 25])
def test_losing_or_tied_initial_cannot_use_extended_win(setup, monkeypatch, initial):
    request = evidence_request(setup, monkeypatch, initial=initial, extended=(30, 0))
    with pytest.raises(ValueError, match="borderline"):
        promotion.run_request(request, setup[0])


def test_normal_score_with_insufficient_suite_wins_is_rejected(setup, monkeypatch):
    url, job, _ = setup
    # 60%, 48%, 48%: 52% aggregate, but only one suite won.
    evaluation.run_request(job, url)
    monkeypatch.setattr(shared, "_evaluate_head_to_head_suite",
                        lambda args, candidate, champion, device, seed, games, opening: result(30 if seed == 40000001 else 24))
    # Generate the request normally, then replace the challenge worker override.
    initial = evaluation.run_request
    def run(request, database_url):
        if request["stage"] == "head_to_head":
            monkeypatch.setattr(shared, "_evaluate_head_to_head_suite",
                                lambda args, candidate, champion, device, seed, games, opening: result(30 if seed == 40000001 else 24))
        return initial(request, database_url)
    monkeypatch.setattr(evaluation, "run_request", run)
    request = evidence_request(setup, monkeypatch)
    assert promotion.run_request(request, url)["decision"] == "rejected"


def test_exact_smallest_extended_margin_passes(setup, monkeypatch):
    url = setup[0]
    run = evaluation.run_request
    def execute(job, database_url):
        if job["stage"] == "extended_head_to_head":
            from dots_cordon_ml.self_play import EvaluationResult, MatchStats
            def smallest_margin(args, candidate, champion, device, seed, games, opening):
                if seed != 50000001:
                    return result(25)
                # Exactly one loss becomes a draw in one seat of one suite.
                player_0 = MatchStats(50, 25, 1, 24, 1 / 50, 1)
                player_1 = result(25).as_player_1
                return EvaluationResult(MatchStats(100, 50, 1, 49, 1 / 100, 1), player_0, player_1)
            monkeypatch.setattr(shared, "_evaluate_head_to_head_suite", smallest_margin)
        return run(job, database_url)
    monkeypatch.setattr(evaluation, "run_request", execute)
    request = evidence_request(setup, monkeypatch, initial=25, initial_draws=1, extended=(25, 1))
    with AuditService(url) as audit:
        suites = evaluation.batch_results(audit.get_evaluation(request["evidence"]["extended_head_to_head"]))
        assert shared.combine_evaluation_results(suites).overall.match_score == 500.5 / 1000
    assert promotion.run_request(request, url)["decision"] == "promoted"


def test_failed_random_screen_rejects_good_challenge(setup, monkeypatch):
    run = shared._evaluate_against_random_suites
    def weak(args, path, device, seeds, games):
        item = run(args, path, device, seeds, games)
        if item.metadata.kind == "policy_value":
            suites = tuple(result(29) for _ in seeds)
            return shared.EvaluatedCheckpoint(item.path, item.metadata, suites, shared.combine_evaluation_results(suites))
        return item
    monkeypatch.setattr(shared, "_evaluate_against_random_suites", weak)
    request = evidence_request(setup, monkeypatch)
    assert promotion.run_request(request, setup[0])["decision"] == "rejected"


@pytest.mark.parametrize("change", [
    lambda b: b.update(status="running"),
    lambda b: b["suites"][0].update(status="failed"),
    lambda b: b["suites"][0]["definition"].update(game_seeds=["wrong"]),
    lambda b: b["suites"][0].update(player_0_losses=0),
    lambda b: b["config"].update(paired_seats=False),
    lambda b: b["config"].update(generation=2),
])
def test_incomplete_or_invalid_persisted_evidence_cannot_promote(setup, monkeypatch, change):
    request = evidence_request(setup, monkeypatch)
    get = AuditService.get_evaluation
    def altered(self, identifier):
        batch = get(self, identifier)
        if identifier == request["evidence"]["initial_head_to_head"]:
            change(batch)
        return batch
    monkeypatch.setattr(AuditService, "get_evaluation", altered)
    with pytest.raises(ValueError):
        promotion.run_request(request, setup[0])


def test_atomic_promote_rechecks_incumbent_after_precheck(setup, monkeypatch):
    request = evidence_request(setup, monkeypatch)
    promote = AuditService.promote
    def racing(self, *args, **kwargs):
        # Another operation wins after application validation, before the actual CAS.
        other_policy = {**kwargs["policy"], "actor": "other"}
        promote(self, *args, **{**kwargs, "policy": other_policy})
        return promote(self, *args, **kwargs)
    monkeypatch.setattr(AuditService, "promote", racing)
    with pytest.raises(ValueError, match="different evidence"):
        promotion.run_request(request, setup[0])
    with AuditService(setup[0]) as audit:
        assert len(audit.champion_history(audit.get_experiment("rounds")["id"])) == 2
        assert audit.get_operation(request["operation_id"])["status"] == "failed"
