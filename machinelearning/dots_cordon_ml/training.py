"""Run one idempotent, resumable search-training round from an explicit JSON request."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import grpc

from . import search_train
from .audit import AuditService
from .audit.integration import close_audit, resolve_checkpoint_reference
from .operations import operation_lock, write_result


# Every effective learner setting is required. No subsequent round inherits new
# CLI defaults accidentally. Diagnostics are disabled, not part of this contract.
TRAINING_CONFIG_FIELDS = frozenset({
    "server", "rows", "columns", "max_turns", "seed", "device", "channels", "blocks",
    "learning_rate", "batch_size", "replay_capacity", "learning_starts",
    "updates_per_episode", "simulations", "c_puct", "dirichlet_alpha",
    "noise_fraction", "temperature_moves", "log_every", "checkpoint_every",
    "checkpoint_dir", "rpc_timeout", "bootstrap_champion",
})


def request_arguments(request, database_url=None):
    if not isinstance(request, dict) or set(request) != {
        "version", "operation_id", "experiment", "source", "target_episode", "config"
    }:
        raise ValueError("Training request requires version, operation_id, experiment, source, target_episode and config")
    if type(request["version"]) is not int or request["version"] != 1:
        raise ValueError("Unsupported training request version")
    if not isinstance(request["operation_id"], str) or not 1 <= len(request["operation_id"]) <= 255:
        raise ValueError("operation_id must contain 1–255 characters")
    if not isinstance(request["experiment"], str) or not request["experiment"]:
        raise ValueError("experiment must be a nonempty name")
    if type(request["target_episode"]) is not int or request["target_episode"] <= 0:
        raise ValueError("target_episode must be a positive absolute episode")
    source = request["source"]
    if (not isinstance(source, dict) or set(source) != {"mode", "checkpoint_id"}
            or not isinstance(source["mode"], str) or source["mode"] not in {"resume", "initialize"}
            or not isinstance(source["checkpoint_id"], str) or not source["checkpoint_id"]):
        raise ValueError("source requires mode resume/initialize and an immutable checkpoint_id")
    config = request["config"]
    if not isinstance(config, dict):
        raise ValueError("config must be an object containing the complete training configuration")
    if set(config) != TRAINING_CONFIG_FIELDS:
        missing = sorted(TRAINING_CONFIG_FIELDS - set(config))
        extra = sorted(set(config) - TRAINING_CONFIG_FIELDS)
        raise ValueError(f"Complete training config required; missing={missing}, unknown={extra}")
    float_fields = {"learning_rate", "c_puct", "dirichlet_alpha", "noise_fraction", "rpc_timeout"}
    string_fields = {"server", "device", "checkpoint_dir"}
    for key, value in config.items():
        if key == "bootstrap_champion":
            continue
        if key in string_fields:
            valid = isinstance(value, str) and bool(value)
        elif key in float_fields:
            valid = type(value) in (float, int) and math.isfinite(value)
        else:
            valid = type(value) is int
        if not valid:
            raise ValueError(f"Invalid explicit training value for {key}")
    if type(config["bootstrap_champion"]) is not bool:
        raise ValueError("bootstrap_champion must be boolean")
    if config["channels"] is None or config["blocks"] is None or config["device"] == "auto":
        raise ValueError("channels, blocks and device must be explicit effective values")
    if not isinstance(config["checkpoint_dir"], str) or not Path(config["checkpoint_dir"]).is_absolute():
        raise ValueError("checkpoint_dir must be an absolute path")
    argv = ["--experiment", request["experiment"], "--episodes", str(request["target_episode"]),
            "--eval-every", "0", "--eval-simulations", "0", "--opening-random-moves", "0",
            "--resume" if source["mode"] == "resume" else "--initialize-from",
            f"checkpoint:{source['checkpoint_id']}"]
    if database_url:
        argv.extend(["--database-url", database_url])
    for key, value in sorted(config.items()):
        if key == "bootstrap_champion":
            if value:
                argv.append("--bootstrap-champion")
        else:
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    try:
        args = search_train.parse_args(argv)
    except SystemExit as exc:
        raise ValueError("Invalid effective training configuration") from exc
    args.operation_id = request["operation_id"]
    args._bounded_training = True
    return args


def _progress(audit, experiment_id, operation_id):
    attempts = [a for a in audit.list_attempts(experiment_id)
                if a["config"].get("operation_id") == operation_id]
    checkpoints = [c for a in attempts for c in audit.list_checkpoints(a["id"])]
    latest = max(checkpoints, key=lambda c: (c["episode"], c["created_at"], c["save_sequence"]), default=None)
    return attempts, latest


def _result(audit, request, checkpoint, status, *, error=None):
    result = {
        "version": 1, "operation_id": request["operation_id"], "kind": "training",
        "status": status, "target_episode": request["target_episode"],
        "attempt_id": checkpoint["attempt_id"] if checkpoint else None,
        "checkpoint_id": checkpoint["id"] if checkpoint else None,
        "checkpoint_sha256": None, "episode": checkpoint["episode"] if checkpoint else None,
        "config": request["config"],
    }
    if checkpoint:
        path = resolve_checkpoint_reference(f"checkpoint:{checkpoint['id']}", audit)
        result["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if error:
        result["error"] = error
    return result


def run_request(request, database_url=None):
    args = request_arguments(request, database_url)
    audit = AuditService(database_url)
    try:
        with operation_lock(audit, request["operation_id"]):
            experiment = audit.ensure_experiment(args.experiment, {
                "rows": args.rows, "columns": args.columns, "max_turns": args.max_turns,
            })
            audit.get_checkpoint(request["source"]["checkpoint_id"])
            operation = audit.ensure_operation(request["operation_id"], experiment["id"], "training", request)
            if operation["status"] == "completed":
                return operation["operation_result"]
            attempts, checkpoint = _progress(audit, experiment["id"], request["operation_id"])
            if checkpoint and checkpoint["episode"] >= request["target_episode"]:
                # The full checkpoint commit is authoritative even if the child
                # died before final flags, attempt status, or acknowledgement.
                path = resolve_checkpoint_reference(f"checkpoint:{checkpoint['id']}", audit)
                audit.import_checkpoint(
                    experiment["id"], path, attempt_id=checkpoint["attempt_id"],
                    checkpoint_id=checkpoint["id"], is_final_in_attempt=True,
                )
                audit.update_attempt(checkpoint["attempt_id"], status="completed", phase="finished",
                                     outcome="trained_only", latest_episode=checkpoint["episode"])
                result = _result(audit, request, checkpoint, "completed")
                audit.update_operation(request["operation_id"], status="completed", result=result)
                return result
            for attempt in attempts:
                if attempt["status"] == "running":
                    audit.update_attempt(attempt["id"], status="interrupted",
                                         stop_reason="operation recovery after process exit")
            if checkpoint:
                args.resume = Path(f"checkpoint:{checkpoint['id']}")
                args.initialize_from = None
                args.bootstrap_champion = bool(args.bootstrap_champion and checkpoint["episode"] == 0)
            audit.update_operation(request["operation_id"], status="running")
            try:
                code = search_train.run(args)
            except BaseException as exc:
                _, checkpoint = _progress(audit, experiment["id"], request["operation_id"])
                status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                result = _result(audit, request, checkpoint, status, error=str(exc) or type(exc).__name__)
                audit.update_operation(request["operation_id"], status=status, result=result)
                raise
            _, checkpoint = _progress(audit, experiment["id"], request["operation_id"])
            if checkpoint is None:
                raise ValueError("Training exited without a committed checkpoint")
            complete = checkpoint["episode"] == request["target_episode"]
            if not complete and code == 0:
                raise ValueError("Training exited before its absolute target")
            if complete:
                audit.update_attempt(checkpoint["attempt_id"], status="completed", phase="finished",
                                     outcome="trained_only", latest_episode=checkpoint["episode"])
            result = _result(audit, request, checkpoint, "completed" if complete else "interrupted")
            audit.update_operation(request["operation_id"], status=result["status"], result=result)
            return result
    finally:
        close_audit(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--database-url")
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text())
        result = run_request(request, args.database_url)
        write_result(args.result, result)
        return_code = 0 if result["status"] == "completed" else 130
    except (OSError, ValueError, grpc.RpcError) as exc:
        print(f"training operation failed: {exc}", file=sys.stderr)
        return_code = 1
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
