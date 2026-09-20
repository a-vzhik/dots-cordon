"""Validate completed policy evaluation evidence and atomically select a champion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

from .audit import AuditService
from .audit.database import PromotionConflict
from .audit.integration import close_audit, resolve_checkpoint_reference
from .checkpoint import read_checkpoint
from .cached_screening import validate_cached_pair
from .evaluation import batch_results, contest_attempt_id, request_arguments as evaluation_arguments
from .operations import operation_lock, write_result
from . import promotion_evaluation as shared


CONFIG_FIELDS = {
    "rows", "columns", "max_turns", "mode", "screen_max_regression",
    "promotion_min_match_score", "promotion_min_suite_wins", "opening_random_moves",
    *{f"{prefix}_{suffix}" for prefix in ("screen", "head_to_head", "extended_head_to_head")
      for suffix in ("games", "suites", "seeds")},
}
EVIDENCE_FIELDS = {"champion_screening", "candidate_screening", "initial_head_to_head", "extended_head_to_head"}


def _identifier(value):
    return isinstance(value, str) and 1 <= len(value) <= 255


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {
        "version", "operation_id", "experiment", "attempt_id", "candidate_checkpoint_id",
        "expected_assignment_id", "evidence", "config",
    }:
        raise ValueError("Complete explicit promotion request required")
    if type(request["version"]) is not int or request["version"] != 1:
        raise ValueError("Unsupported promotion request version")
    if not all(_identifier(request[key]) for key in (
        "operation_id", "experiment", "attempt_id", "candidate_checkpoint_id", "expected_assignment_id"
    )):
        raise ValueError("Promotion identifiers must be nonempty strings")
    evidence, config = request["evidence"], request["config"]
    if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_FIELDS:
        raise ValueError("Explicit screening and challenge evaluation IDs required")
    for key, value in evidence.items():
        if not _identifier(value) and not (key == "extended_head_to_head" and value is None):
            raise ValueError("Invalid evidence ID")
    if not isinstance(config, dict) or set(config) not in (CONFIG_FIELDS, CONFIG_FIELDS | {"reuse_champion_screening"}):
        raise ValueError("Complete explicit promotion config required")
    if "reuse_champion_screening" in config and type(config["reuse_champion_screening"]) is not bool:
        raise ValueError("reuse_champion_screening must be a boolean")
    for key in CONFIG_FIELDS - {"mode", "screen_max_regression", "promotion_min_match_score"}:
        value = config[key]
        if key.endswith("_seeds"):
            if not isinstance(value, list) or any(type(v) is not int or v < 0 for v in value):
                raise ValueError("Suite seeds must be explicit nonnegative integer lists")
            if len(set(value)) != len(value):
                raise ValueError("Suite seeds must be distinct")
        elif type(value) is not int:
            raise ValueError(f"{key} must be an integer")
    if config["mode"] != "policy" or any(not 2 <= config[k] <= 255 for k in ("rows", "columns")):
        raise ValueError("Promotion requires policy mode and valid board dimensions")
    if config["max_turns"] < 0 or not 0 <= config["opening_random_moves"] < config["rows"] * config["columns"]:
        raise ValueError("Invalid turn limit or opening length")
    for key in ("screen_max_regression", "promotion_min_match_score"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError("Gate thresholds must be finite numbers")
    if config["screen_max_regression"] < 0 or not 0.5 < config["promotion_min_match_score"] <= 1:
        raise ValueError("Invalid gate thresholds")
    if not 1 <= config["promotion_min_suite_wins"] <= config["head_to_head_suites"]:
        raise ValueError("Invalid suite-win threshold")
    for prefix in ("screen", "head_to_head", "extended_head_to_head"):
        games, suites, seeds = (config[f"{prefix}_{suffix}"] for suffix in ("games", "suites", "seeds"))
        if games < 2 or games % 2 or suites < 1:
            raise ValueError("Suite counts must be positive and games positive and even")
        expected = 0 if prefix == "extended_head_to_head" and evidence[prefix] is None else suites
        if len(seeds) != expected:
            raise ValueError("Expected suite seed count differs from configured evidence")
    groups = [set(config[f"{prefix}_seeds"]) for prefix in ("screen", "head_to_head", "extended_head_to_head")]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Challenge stages require fresh independent suite seeds")


def _batch(audit, experiment, request, key, subject, opponent):
    """Rebuild expected definitions and verify actual counts, not claimed summaries."""
    batch = audit.get_evaluation(request["evidence"][key])
    stage = "screening" if key.endswith("screening") else (
        "head_to_head" if key == "initial_head_to_head" else "extended_head_to_head")
    prefix = {"screening": "screen", "head_to_head": "head_to_head",
              "extended_head_to_head": "extended_head_to_head"}[stage]
    config = request["config"]
    if (batch["experiment_id"] != experiment["id"] or batch["attempt_id"] != request["attempt_id"]
            or batch["checkpoint_id"] != subject or batch["opponent_checkpoint_id"] != opponent
            or batch["opponent_kind"] != ("checkpoint" if opponent else "random")
            or batch["purpose"] != ("screening" if stage == "screening" else "head_to_head")):
        raise ValueError("Evaluation evidence has incorrect participants, purpose or attempt")
    settings = batch["config"]
    operation = audit.get_operation(settings.get("operation_id"))
    if (not operation or operation["kind"] != "evaluation" or operation["experiment_id"] != experiment["id"]
            or operation["status"] != "completed"
            or batch["id"] not in operation["progress"].get("evaluation_ids", [])):
        raise ValueError("Evidence must belong to a completed current evaluation operation")
    job = operation["operation_request"]
    evaluation_arguments(job)
    attempt = audit.get_attempt(request["attempt_id"])
    if (job["contest"] != attempt["config"].get("contest")
            or contest_attempt_id(experiment["id"], job["contest"]["id"]) != attempt["id"]
            or job["stage"] != stage
            or any(settings.get(k) != value for k, value in job["config"].items())
            or settings.get("generation") != operation["progress"].get("generation")
            or settings.get("stage") != stage):
        raise ValueError("Evaluation operation does not match this contest and stage")
    expected_config = {k: config[k] for k in ("rows", "columns", "max_turns", "mode")}
    expected_config.update(games=config[f"{prefix}_games"], suites=config[f"{prefix}_suites"],
                           paired_seats=True, opening_random_moves=0 if stage == "screening" else config["opening_random_moves"])
    if any(settings.get(k) != value for k, value in expected_config.items()):
        raise ValueError("Evaluation configuration differs from promotion requirements")
    seeds = config[f"{prefix}_seeds"]
    if operation["progress"].get("suite_seeds") != seeds:
        raise ValueError("Evaluation suite seeds differ from promotion requirements")
    definitions = shared.suite_definitions(SimpleNamespace(**config), SimpleNamespace(**config), seeds,
                                           config[f"{prefix}_games"], head_to_head=opponent is not None)
    if (len(batch["suites"]) != len(definitions)
            or any(s["suite_index"] != i or s["definition"] != definition
                   for i, (s, definition) in enumerate(zip(batch["suites"], definitions)))):
        raise ValueError("Evaluation suite definitions differ from promotion requirements")
    results = batch_results(batch)
    for suite in results:
        if any(seat.games != config[f"{prefix}_games"] // 2 for seat in (suite.as_player_0, suite.as_player_1)):
            raise ValueError("Evaluation evidence has incomplete or unpaired game counts")
    return results


def _decide(audit, experiment, request):
    config, evidence = request["config"], request["evidence"]
    attempt = audit.get_attempt(request["attempt_id"])
    candidate = audit.get_checkpoint(request["candidate_checkpoint_id"])
    assignment = next((row for row in audit.champion_history(experiment["id"])
                       if row["id"] == request["expected_assignment_id"]), None)
    if (assignment is None or attempt["experiment_id"] != experiment["id"]
            or attempt["champion_at_start_assignment_id"] != assignment["id"]
            or candidate["experiment_id"] != experiment["id"] or candidate["attempt_id"] != attempt["id"]):
        raise ValueError("Candidate, contest and captured incumbent do not match")
    if any(experiment["game_config"].get(k, 0) != config[k] for k in ("rows", "columns", "max_turns")):
        raise ValueError("Promotion board/rules differ from the experiment")
    champion_id = assignment["checkpoint_id"]
    contest = attempt["config"].get("contest", {})
    if (attempt["config"].get("algorithm") != "evaluation_contest_v1"
            or contest.get("expected_assignment_id") != assignment["id"]
            or contest.get("champion_checkpoint_id") != champion_id
            or candidate["parent_checkpoint_id"] != contest.get("candidate_checkpoint_id")):
        raise ValueError("Candidate does not belong to the captured evaluation contest")
    source = audit.get_checkpoint(contest["candidate_checkpoint_id"])
    if source["checkpoint_blob_id"] != candidate["checkpoint_blob_id"]:
        raise ValueError("Contest candidate differs from the immutable learner checkpoint")
    for checkpoint_id in (candidate["id"], champion_id):
        path = resolve_checkpoint_reference(f"checkpoint:{checkpoint_id}", audit)
        payload, metadata = read_checkpoint(path, map_location="cpu")
        if (metadata.board != (config["rows"], config["columns"])
                or payload.get("game_config", {}).get("max_turns", 0) != config["max_turns"]
                or metadata.kind not in {"dqn", "policy_value"}):
            raise ValueError("Participant model board/rules differ from promotion requirements")
    screens = [audit.get_evaluation(evidence[key]) for key in ("champion_screening", "candidate_screening")]
    reuse = config.get("reuse_champion_screening", False)
    if reuse:
        validate_cached_pair(audit, screens[1], screens[0], experiment["id"], champion_id, config)
    elif screens[0]["config"].get("operation_id") != screens[1]["config"].get("operation_id"):
        raise ValueError("Screening evidence must come from one matched evaluation operation")
    champion = (batch_results(screens[0]) if reuse else
                _batch(audit, experiment, request, "champion_screening", champion_id, None))
    candidate_screen = _batch(audit, experiment, request, "candidate_screening", candidate["id"], None)
    initial = _batch(audit, experiment, request, "initial_head_to_head", candidate["id"], champion_id)
    extended = (_batch(audit, experiment, request, "extended_head_to_head", candidate["id"], champion_id)
                if evidence["extended_head_to_head"] else None)
    wrap = lambda suites: shared.EvaluatedCheckpoint(None, None, suites, shared.combine_evaluation_results(suites))
    qualified = shared._passes_random_screen(wrap(candidate_screen), wrap(champion), config["screen_max_regression"])
    outcome, policy = shared.challenge_decision(initial, minimum_match_score=config["promotion_min_match_score"],
                                               minimum_suite_wins=config["promotion_min_suite_wins"])
    if extended is not None and (outcome != "extended" or not qualified):
        raise ValueError("Extended evidence is only applicable to a qualified borderline initial challenge")
    selected_id = evidence["initial_head_to_head"]
    if extended is not None:
        outcome, policy = shared.challenge_decision(extended, minimum_match_score=config["promotion_min_match_score"],
                                                   minimum_suite_wins=config["promotion_min_suite_wins"], extended=True)
        selected_id = evidence["extended_head_to_head"]
    policy = {**policy, "promotion_request": request}
    successful = any(d["stage"] == "promotion" and d["policy"] == policy
                     for d in audit.list_decisions(attempt["id"]))
    if audit.current_champion(experiment["id"])["id"] != assignment["id"] and not successful:
        raise PromotionConflict("Champion changed since this contest began")
    audit.record_decision(attempt["id"], candidate["id"], stage="screening",
                          result="qualified" if qualified else "rejected",
                          candidate_evaluation_id=evidence["candidate_screening"],
                          champion_evaluation_id=evidence["champion_screening"],
                          policy={"screen_max_regression": config["screen_max_regression"],
                                  **({"reuse_champion_screening": True} if reuse else {})},
                          operation_key=request["operation_id"] + ":screening")
    promoted = None
    if qualified:
        audit.record_decision(attempt["id"], candidate["id"], stage="challenge", result=outcome,
                              candidate_evaluation_id=selected_id,
                              champion_evaluation_id=evidence["champion_screening"], policy=policy,
                              operation_key=request["operation_id"] + ":challenge")
        if outcome == "passed":
            promoted = audit.promote(experiment["id"], candidate["id"], attempt_id=attempt["id"],
                                     expected_assignment_id=assignment["id"], candidate_evaluation_id=selected_id,
                                     champion_evaluation_id=evidence["champion_screening"], policy=policy)
    decision = "promoted" if promoted else "extended_required" if qualified and outcome == "extended" else "rejected"
    if decision == "rejected":
        audit.update_attempt(attempt["id"], status="completed", phase="finished",
                             outcome="no_challenger_passed" if qualified else "no_qualified_candidate")
    return {"version": 1, "operation_id": request["operation_id"], "kind": "promotion", "status": "completed",
            "decision": decision, "attempt_id": attempt["id"], "candidate_checkpoint_id": candidate["id"],
            "expected_assignment_id": assignment["id"], "assignment": promoted, "evidence": evidence,
            "config": config}


def run_request(request, database_url=None, *, output=None):
    validate_request(request)
    audit = AuditService(database_url)
    try:
        with operation_lock(audit, request["operation_id"]):
            experiment = audit.get_experiment(request["experiment"])
            operation = audit.ensure_operation(request["operation_id"], experiment["id"], "promotion", request)
            if operation["status"] == "completed":
                result = operation["operation_result"]
            else:
                with operation_lock(audit, "contest:" + request["attempt_id"]):
                    try:
                        result = _decide(audit, experiment, request)
                        audit.update_operation(request["operation_id"], status="completed", result=result)
                    except BaseException as exc:
                        audit.update_operation(request["operation_id"], status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                               result={"version": 1, "operation_id": request["operation_id"], "kind": "promotion",
                                                       "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                                       "error": str(exc) or type(exc).__name__})
                        raise
            # Export is a repairable acknowledgement; never roll back a committed decision.
            if output is not None and result["decision"] == "promoted":
                audit.export_checkpoint(result["candidate_checkpoint_id"], output)
            return result
    finally:
        close_audit(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--database-url")
    parser.add_argument("--output", type=Path, help="optional atomic export of a promoted checkpoint")
    args = parser.parse_args()
    request = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = {name: signal.signal(name, interrupted) for name in (signal.SIGINT, signal.SIGTERM)}
    try:
        request = json.loads(args.request.read_text())
        result = run_request(request, args.database_url, output=args.output)
        write_result(args.result, result)
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        print(f"promotion operation failed: {str(exc) or type(exc).__name__}", file=sys.stderr)
        if isinstance(request, dict) and _identifier(request.get("operation_id")):
            with AuditService(args.database_url) as audit:
                operation = audit.get_operation(request["operation_id"])
                if operation and operation["operation_request"] == request and operation["operation_result"]:
                    write_result(args.result, operation["operation_result"])
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
