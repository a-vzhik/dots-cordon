"""Run one durable policy-only evaluation stage without promotion or transitions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
from uuid import NAMESPACE_URL, uuid5

import torch

from .audit import AuditService
from .audit.integration import close_audit, resolve_checkpoint_reference
from .checkpoint import read_checkpoint
from .cached_screening import select_champion_baseline, validate_random_batch
from .operations import operation_lock, write_result
from . import promotion_evaluation as shared
from .self_play import EvaluationResult, MatchStats, combine_evaluation_results


CONFIG_FIELDS = frozenset({
    "server", "rows", "columns", "max_turns", "mode", "device", "games", "suites",
    "seed", "opening_random_moves", "paired_seats", "evaluation_workers", "rpc_timeout",
})
STAGES = {"screening", "head_to_head", "extended_head_to_head"}


def _identifier(value):
    return isinstance(value, str) and 1 <= len(value) <= 255


def request_arguments(request):
    if not isinstance(request, dict) or set(request) != {
        "version", "operation_id", "experiment", "contest", "stage", "config"
    }:
        raise ValueError("Evaluation request requires version, operation_id, experiment, contest, stage and config")
    if type(request["version"]) is not int or request["version"] != 1:
        raise ValueError("Unsupported evaluation request version")
    if not _identifier(request["operation_id"]) or not _identifier(request["experiment"]):
        raise ValueError("operation_id and experiment must be nonempty strings")
    if not isinstance(request["stage"], str) or request["stage"] not in STAGES:
        raise ValueError("Unsupported evaluation stage")
    contest = request["contest"]
    if not isinstance(contest, dict) or set(contest) != {
        "id", "candidate_checkpoint_id", "expected_assignment_id", "champion_checkpoint_id"
    } or not all(_identifier(value) for value in contest.values()):
        raise ValueError("contest requires id and immutable candidate, champion and assignment IDs")
    config = request["config"]
    if not isinstance(config, dict) or set(config) not in (CONFIG_FIELDS, CONFIG_FIELDS | {"reuse_champion_screening"}):
        raise ValueError("Complete explicit evaluation config required")
    if "reuse_champion_screening" in config and (request["stage"] != "screening"
            or type(config["reuse_champion_screening"]) is not bool):
        raise ValueError("reuse_champion_screening must be a boolean on screening jobs only")
    for key in CONFIG_FIELDS - {"server", "device", "mode", "paired_seats", "rpc_timeout"}:
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if any(not 2 <= config[key] <= 255 for key in ("rows", "columns")):
        raise ValueError("Board dimensions must be between 2 and 255")
    if config["games"] < 2 or config["games"] % 2:
        raise ValueError("games must be positive and even")
    if min(config["suites"], config["evaluation_workers"]) < 1:
        raise ValueError("suites and evaluation_workers must be positive")
    if min(config["max_turns"], config["opening_random_moves"], config["seed"]) < 0:
        raise ValueError("Turn limit, opening length and seed must be nonnegative")
    if config["opening_random_moves"] >= config["rows"] * config["columns"]:
        raise ValueError("opening_random_moves must be smaller than the board")
    if config["mode"] != "policy":
        raise ValueError("Only policy mode supports promotion evidence; use search-evaluate for MCTS diagnostics")
    if config["paired_seats"] is not True:
        raise ValueError("Promotion evaluation requires paired_seats=true")
    if request["stage"] == "screening" and config["opening_random_moves"] != 0:
        raise ValueError("Random screening requires opening_random_moves=0")
    if any(not isinstance(config[key], str) or not config[key] for key in ("server", "device")):
        raise ValueError("server and device must be explicit strings")
    if config["device"] == "auto":
        raise ValueError("device must be an explicit effective device")
    if (type(config["rpc_timeout"]) not in (float, int)
            or not math.isfinite(config["rpc_timeout"]) or config["rpc_timeout"] <= 0):
        raise ValueError("rpc_timeout must be finite and positive")
    return argparse.Namespace(**config)


def contest_attempt_id(experiment_id, contest_id):
    return str(uuid5(NAMESPACE_URL, f"dots-cordon-evaluation:{experiment_id}:{contest_id}"))


def _contest(audit, experiment, request):
    contest = request["contest"]
    config = request["config"]
    board = {key: config[key] for key in ("rows", "columns", "max_turns")}
    if any(experiment["game_config"].get(key, 0) != value for key, value in board.items()):
        raise ValueError("Evaluation board/rules differ from the experiment")
    assignment = next((a for a in audit.champion_history(experiment["id"])
                       if a["id"] == contest["expected_assignment_id"]), None)
    if assignment is None or assignment["checkpoint_id"] != contest["champion_checkpoint_id"]:
        raise ValueError("Contest champion does not match the captured assignment")
    participants = []
    for checkpoint_id in (contest["champion_checkpoint_id"], contest["candidate_checkpoint_id"]):
        saved = audit.get_checkpoint(checkpoint_id)
        if saved["experiment_id"] != experiment["id"]:
            raise ValueError("Evaluation participants must belong to the experiment")
        path = resolve_checkpoint_reference(f"checkpoint:{checkpoint_id}", audit)
        payload, metadata = read_checkpoint(path, map_location="cpu")
        if (metadata.board != (config["rows"], config["columns"])
                or payload.get("game_config", {}).get("max_turns", 0) != config["max_turns"]):
            raise ValueError("Checkpoint board/rules differ from the requested evaluation")
        if metadata.kind not in {"dqn", "policy_value"}:
            raise ValueError("Unsupported evaluation model kind")
        participants.append((saved, path, metadata))
    source, candidate_path, metadata = participants[1]
    attempt = audit.create_attempt(
        experiment["id"], starting_checkpoint_id=source["id"],
        champion_at_start_assignment_id=assignment["id"], target_episode=source["episode"],
        attempt_id=contest_attempt_id(experiment["id"], contest["id"]),
        config={"algorithm": "evaluation_contest_v1", "contest": contest, "mode": "policy", **board},
    )
    candidate = audit.import_checkpoint(
        experiment["id"], candidate_path, attempt_id=attempt["id"],
        checkpoint_id=str(uuid5(NAMESPACE_URL, f"{attempt['id']}:candidate")),
        parent_checkpoint_id=source["id"], is_screening_candidate=True,
        is_final_in_attempt=True, candidate_index=1,
    )
    return attempt, candidate, participants


def batch_results(batch):
    """Reconstruct exact complete suite results for the shared promotion gates."""
    if batch["status"] != "completed" or any(s["status"] != "completed" for s in batch["suites"]):
        raise ValueError("Evaluation evidence is incomplete")
    results = []
    for suite in batch["suites"]:
        seats = []
        for seat in (0, 1):
            wins, draws, losses, total = (suite[f"player_{seat}_{key}"] for key in (
                "wins", "draws", "losses", "score_difference_sum"))
            games = wins + draws + losses
            seats.append(MatchStats(games, wins, draws, losses, total / games, total))
        games = sum(s.games for s in seats)
        total = sum(s.score_difference_sum for s in seats)
        overall = MatchStats(games, sum(s.wins for s in seats), sum(s.draws for s in seats),
                             sum(s.losses for s in seats), total / games, total)
        results.append(EvaluationResult(overall, *seats))
    return tuple(results)


def _result(audit, request, attempt, candidate, progress, status, error=None):
    batches = [audit.get_evaluation(identifier) for identifier in progress.get("evaluation_ids", [])]
    evaluations = []
    for batch in batches:
        item = {key: batch[key] for key in (
            "id", "checkpoint_id", "opponent_checkpoint_id", "opponent_kind", "purpose", "status", "config")}
        item["model_kind"] = audit.get_checkpoint(batch["checkpoint_id"])["model"].get("kind", "dqn")
        item["opponent_model_kind"] = (audit.get_checkpoint(batch["opponent_checkpoint_id"])["model"].get("kind", "dqn")
                                       if batch["opponent_checkpoint_id"] else None)
        item["suites"] = [{"index": s["suite_index"], "definition": s["definition"],
                           "status": s["status"]} for s in batch["suites"]]
        if batch["status"] == "completed":
            suites = batch_results(batch)
            item["aggregate"] = shared._result_dict(combine_evaluation_results(suites))
            for suite, stats in zip(item["suites"], suites, strict=True):
                suite["result"] = shared._result_dict(stats)
        if request["config"].get("reuse_champion_screening"):
            item["reused"] = batch["id"] == progress.get("champion_screening_id")
        evaluations.append(item)
    result = {
        "version": 1, "operation_id": request["operation_id"], "kind": "evaluation", "status": status,
        "stage": request["stage"], "attempt_id": attempt["id"],
        "source_checkpoint_id": request["contest"]["candidate_checkpoint_id"],
        "candidate_checkpoint_id": candidate["id"],
        "champion_checkpoint_id": request["contest"]["champion_checkpoint_id"],
        "expected_assignment_id": request["contest"]["expected_assignment_id"],
        "config": request["config"], "generation": progress.get("generation", 0),
        "suite_seeds": progress.get("suite_seeds", []),
        "evaluation_ids": progress.get("evaluation_ids", []), "evaluations": evaluations,
    }
    if request["config"].get("reuse_champion_screening"):
        result["reused_evaluation_ids"] = [progress["champion_screening_id"]]
        result["executed_evaluation_ids"] = [i for i in result["evaluation_ids"] if i != progress["champion_screening_id"]]
    if error:
        result["error"] = error
    return result


def _run(audit, experiment, request, args, operation):
    attempt, candidate, participants = _contest(audit, experiment, request)
    progress = operation["progress"]
    reuse = request["config"].get("reuse_champion_screening", False)
    if reuse:
        assignment = next(a for a in audit.champion_history(experiment["id"])
                          if a["id"] == request["contest"]["expected_assignment_id"])
        if "champion_screening_id" not in progress:
            baseline = select_champion_baseline(audit, experiment["id"], assignment, request["config"])
            progress = {**progress, "champion_screening_id": baseline["id"]}
            audit.update_operation(request["operation_id"], status="running", progress=progress)
        else:
            baseline = validate_random_batch(audit, audit.get_evaluation(progress["champion_screening_id"]),
                                             experiment["id"], assignment["checkpoint_id"], request["config"])
        print(f"random-screen reusing champion evaluation={baseline['id']}", flush=True)
    all_batches = audit.list_evaluations(experiment["id"])
    previous = [b for b in all_batches
                if b["config"].get("operation_id") == request["operation_id"]]
    current = [b for b in previous if b["id"] in progress.get("evaluation_ids", [])]
    expected_count = 2 if request["stage"] == "screening" and not reuse else 1
    if len(current) == expected_count and all(b["status"] == "completed" for b in current):
        result = _result(audit, request, attempt, candidate, progress, "completed")
        audit.update_operation(request["operation_id"], status="completed", result=result)
        return result
    for batch in previous:
        if batch["status"] == "running":
            audit.fail_evaluation(batch["id"], "restarted incomplete evaluation", status="interrupted")
    used_seeds = {int(s["definition"]["seed"]) for b in all_batches
                  for s in audit.get_evaluation(b["id"])["suites"]
                  if "seed" in s["definition"]}
    # Stage namespaces retain existing defaults, while this check also protects
    # callers that choose overlapping custom base seeds for initial/extended jobs.
    while True:
        seeds = audit.reserve_suite_seeds(experiment["id"], request["stage"], args.suites, args.seed)
        if used_seeds.isdisjoint(seeds):
            break
    generation = progress.get("generation", 0) + 1
    identifiers = [str(uuid5(NAMESPACE_URL, f"{request['operation_id']}:{generation}:{index}"))
                   for index in range(expected_count)]
    progress = {**progress, "generation": generation, "suite_seeds": list(seeds),
                "evaluation_ids": ([progress["champion_screening_id"]] if reuse else []) + identifiers,
                "replaced_evaluation_ids": [b["id"] for b in previous]}
    audit.update_operation(request["operation_id"], status="running", progress=progress)
    champion_path, candidate_path = participants[0][1], participants[1][1]
    metadata = participants[1][2]
    screening = request["stage"] == "screening"
    definitions = shared.suite_definitions(args, metadata, seeds, args.games, head_to_head=not screening)
    subject_ids = (request["contest"]["champion_checkpoint_id"], candidate["id"]) if screening and not reuse else (candidate["id"],)
    # Persist all participant batches before execution. Retry preserves completed
    # siblings but replaces the entire partial screening pair with fresh suites.
    for identifier, subject in zip(identifiers, subject_ids, strict=True):
        audit.create_evaluation(
            experiment["id"], subject, purpose="screening" if screening else "head_to_head",
            attempt_id=attempt["id"], batch_id=identifier, suite_definitions=definitions,
            opponent_checkpoint_id=None if screening else request["contest"]["champion_checkpoint_id"],
            config={**request["config"], "operation_id": request["operation_id"], "stage": request["stage"],
                    "generation": generation, "replaces": progress["replaced_evaluation_ids"]},
        )
    try:
        torch.set_num_threads(1)
        device = shared._device(args.device)
        if screening:
            shared._evaluate_random_screen(args, (candidate_path,) if reuse else (champion_path, candidate_path), device, seeds, args.games,
                                          audit_service=audit, evaluation_ids=tuple(identifiers))
        else:
            contender = shared.EvaluatedCheckpoint(candidate_path, metadata, (), None)
            champion = shared.EvaluatedCheckpoint(champion_path, participants[0][2], (), None)
            shared._evaluate_head_to_head_suites(args, contender, champion, device, seeds, args.games,
                                                args.opening_random_moves, audit_service=audit,
                                                evaluation_id=identifiers[0])
        result = _result(audit, request, attempt, candidate, progress, "completed")
        if any(item["status"] != "completed" for item in result["evaluations"]):
            raise ValueError("Evaluator returned without completing every suite")
        audit.update_operation(request["operation_id"], status="completed", result=result)
        return result
    except BaseException as exc:
        shared._fail_evaluations(audit, tuple(identifiers), exc)
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        result = _result(audit, request, attempt, candidate, progress, status, str(exc) or type(exc).__name__)
        audit.update_operation(request["operation_id"], status=status, result=result)
        raise


def run_request(request, database_url=None):
    args = request_arguments(request)
    audit = AuditService(database_url)
    try:
        with operation_lock(audit, request["operation_id"]):
            experiment = audit.get_experiment(request["experiment"])
            operation = audit.ensure_operation(request["operation_id"], experiment["id"], "evaluation", request)
            if operation["status"] == "completed":
                return operation["operation_result"]
            # Separate operation IDs must not execute stages of one contest concurrently.
            with operation_lock(audit, "contest:" + contest_attempt_id(experiment["id"], request["contest"]["id"])):
                try:
                    return _run(audit, experiment, request, args, operation)
                except BaseException as exc:
                    # Failures during participant validation or batch creation
                    # precede the execution handler, but still need a durable status.
                    current = audit.get_operation(request["operation_id"])
                    if current["status"] == "running":
                        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                        audit.update_operation(request["operation_id"], status=status, result={
                            "version": 1, "kind": "evaluation", "operation_id": request["operation_id"],
                            "stage": request["stage"], "status": status,
                            "error": str(exc) or type(exc).__name__,
                        })
                    raise
    finally:
        close_audit(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--database-url")
    args = parser.parse_args()
    request = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = {name: signal.signal(name, interrupted) for name in (signal.SIGINT, signal.SIGTERM)}
    try:
        request = json.loads(args.request.read_text())
        result = run_request(request, args.database_url)
        write_result(args.result, result)
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        print(f"evaluation operation failed: {str(exc) or type(exc).__name__}", file=sys.stderr)
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
