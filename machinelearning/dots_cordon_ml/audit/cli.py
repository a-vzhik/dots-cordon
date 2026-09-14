"""Local administration commands; training never migrates schemas implicitly."""

import argparse
import json
from pathlib import Path

from ..checkpoint import read_checkpoint
from . import AuditError, AuditService, Database


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--database-url",
        help="SQLAlchemy URL; defaults to DOTS_CORDON_DATABASE_URL or the local SQLite database",
    )
    commands = root.add_subparsers(dest="command", required=True)
    db = commands.add_parser("db", help="Manage versioned schema migrations")
    db_commands = db.add_subparsers(dest="db_command", required=True)
    db_commands.add_parser("upgrade")
    db_commands.add_parser("status")
    for name in ("import", "bootstrap"):
        child = commands.add_parser(
            name,
            help="Import trusted checkpoint bytes"
            if name == "import"
            else "Import and set the first champion",
        )
        child.add_argument("--experiment", required=True)
        child.add_argument("--checkpoint", type=Path, required=True)
        child.add_argument(
            "--max-turns",
            type=int,
            default=0,
            help="Game turn limit; must match later training/evaluation",
        )
    export = commands.add_parser(
        "export", help="Export exact checkpoint bytes with checksum verification"
    )
    export.add_argument("checkpoint_id")
    export.add_argument("path", type=Path)
    status = commands.add_parser(
        "status", help="Show experiment champions and attempt counts"
    )
    status.add_argument("--experiment")
    serve = commands.add_parser(
        "serve", help="Serve the read-only training audit HTTP API"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--download-workers", type=int, default=2)
    return root


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        if args.command == "serve":
            if not 1 <= args.port <= 65535 or args.download_workers < 1:
                argument_parser.error(
                    "--port must be 1–65535 and --download-workers must be positive"
                )
            import uvicorn
            from .web.app import create_app

            uvicorn.run(
                create_app(args.database_url, download_workers=args.download_workers),
                host=args.host,
                port=args.port,
            )
            return
        if args.command == "db":
            database = Database(args.database_url)
            try:
                result = (
                    database.upgrade()
                    if args.db_command == "upgrade"
                    else database.status()
                )
            finally:
                database.close()
        else:
            with AuditService(args.database_url) as service:
                if args.command in ("import", "bootstrap"):
                    _, metadata = read_checkpoint(args.checkpoint, map_location="cpu")
                    experiment = service.ensure_experiment(
                        args.experiment,
                        {
                            "rows": metadata.rows,
                            "columns": metadata.columns,
                            "max_turns": args.max_turns,
                        },
                    )
                    checkpoint = service.import_checkpoint(
                        experiment["id"], args.checkpoint
                    )
                    result = {"experiment": experiment, "checkpoint": checkpoint}
                    if args.command == "bootstrap":
                        result["champion"] = service.bootstrap(
                            experiment["id"], checkpoint["id"]
                        )
                        result["experiment"] = service.get_experiment(experiment["id"])
                elif args.command == "export":
                    result = {
                        "checkpoint_id": args.checkpoint_id,
                        "path": str(
                            service.export_checkpoint(args.checkpoint_id, args.path)
                        ),
                    }
                else:
                    experiments = (
                        [service.get_experiment(args.experiment)]
                        if args.experiment
                        else service.list_experiments()
                    )
                    result = [
                        {
                            **experiment,
                            "champion": service.current_champion(experiment["id"]),
                            "attempt_count": len(
                                service.list_attempts(experiment["id"])
                            ),
                        }
                        for experiment in experiments
                    ]
        print(json.dumps(result, indent=2, sort_keys=True))
    except (AuditError, OSError) as exc:
        argument_parser.exit(1, f"Audit error: {exc}\n")


if __name__ == "__main__":
    main()
