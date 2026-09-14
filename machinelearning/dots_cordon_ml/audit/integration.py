"""Shared CLI wiring; runners never depend on SQLAlchemy or database rows."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import math
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .service import AuditService


def add_audit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database-url",
        help="SQL database URL; defaults to DOTS_CORDON_DATABASE_URL or the local audit DB",
    )
    parser.add_argument(
        "--experiment",
        default="default",
        help="named board-compatible training lineage",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="explicitly use file-only training/evaluation",
    )


def open_audit(args: argparse.Namespace) -> AuditService | None:
    if args.no_audit:
        print("audit disabled (--no-audit): file-only operation", flush=True)
        return None
    from .service import AuditService

    return AuditService(args.database_url)


def close_audit(service: AuditService | None) -> None:
    if service is not None:
        temporary = getattr(service, "_reference_directory", None)
        try:
            service.close()
        finally:
            if temporary is not None:
                temporary.cleanup()


def resolve_checkpoint_reference(
    reference: str | Path,
    service: AuditService | None,
    *,
    experiment_name: str = "default",
) -> Path:
    value = str(reference)
    if not value.startswith(("checkpoint:", "champion:")):
        return Path(reference)
    if service is None:
        raise ValueError("database checkpoint references require auditing")
    references = getattr(service, "_checkpoint_references", None)
    if references is None:
        references = {}
        service._checkpoint_references = references
        service._reference_directory = tempfile.TemporaryDirectory(
            prefix="dots-cordon-checkpoints-"
        )
    if value not in references:
        if value.startswith("checkpoint:"):
            checkpoint_id = value.removeprefix("checkpoint:")
        else:
            experiment = service.get_experiment(
                value.removeprefix("champion:") or experiment_name
            )
            assignment = service.current_champion(experiment["id"])
            if assignment is None:
                raise ValueError(
                    "experiment does not have a champion; bootstrap one first"
                )
            checkpoint_id = assignment["checkpoint_id"]
        record = service.get_checkpoint(checkpoint_id)
        path = Path(service._reference_directory.name) / f"{checkpoint_id}.pt"
        service.export_checkpoint(checkpoint_id, path)
        references[value] = (path, record)
        references[str(path)] = (path, record)
    return references[value][0]


def checkpoint_record(
    service: AuditService,
    experiment_id: str,
    reference: str | Path,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    references = getattr(service, "_checkpoint_references", {})
    resolved = references.get(str(reference)) or references.get(str(path))
    if resolved is not None:
        record = resolved[1]
        if record["experiment_id"] != experiment_id:
            raise ValueError(
                "checkpoint belongs to another experiment; transfer initialization is not supported by --resume"
            )
        return record
    return service.import_checkpoint(experiment_id, path or Path(reference))


def ensure_experiment(
    service: AuditService, args: argparse.Namespace, rows: int, columns: int
) -> dict[str, Any]:
    return service.ensure_experiment(
        args.experiment, {"rows": rows, "columns": columns, "max_turns": args.max_turns}
    )


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return str(value) if abs(value) > 2**53 - 1 else value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return str(value)


def effective_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        key: json_value(value)
        for key, value in vars(args).items()
        if not key.startswith("_") and key != "database_url"
    }
    root = Path(__file__).resolve().parents[3]
    provenance: dict[str, Any] = {
        "python": platform.python_version(),
        "engine_revision": "unknown",
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--", "machinelearning"],
            cwd=root,
            capture_output=True,
            timeout=2,
            check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "machinelearning"],
            cwd=root,
            capture_output=True,
            timeout=2,
            check=True,
        )
        provenance.update(
            commit=commit.stdout.strip(),
            dirty=bool(status.stdout),
            diff_sha256=hashlib.sha256(diff.stdout).hexdigest(),
        )
    except (OSError, subprocess.SubprocessError):
        provenance["commit"] = None
    lock = root / "machinelearning" / "uv.lock"
    if lock.exists():
        provenance["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    config["provenance"] = provenance
    return config


def evaluation_definition(
    args: argparse.Namespace,
    *,
    rows: int,
    columns: int,
    seed: int,
    games: int,
    kind: str = "random",
    opening_random_moves: int = 0,
) -> dict[str, Any]:
    from ..self_play import random_game_seeds

    return {
        "definition_version": 1,
        "kind": kind,
        "rows": rows,
        "columns": columns,
        "max_turns": args.max_turns,
        "seed": str(seed),
        "expected_games": games,
        "game_seeds": [str(item) for item in random_game_seeds(games, seed)],
        "seat_schedule": [index % 2 for index in range(games)],
        "opening_random_moves": opening_random_moves,
        "evaluator_version": "paired-seeds-v1",
    }


def evaluation_result_dict(result: Any) -> dict[str, Any]:
    output = {}
    for name in ("as_player_0", "as_player_1"):
        stats = getattr(result, name)
        total = getattr(stats, "score_difference_sum", None)
        if total is None:
            raise ValueError("audited evaluations require exact integer score totals")
        output[name] = {
            "wins": stats.wins,
            "draws": stats.draws,
            "losses": stats.losses,
            "score_difference_sum": total,
        }
    return output
