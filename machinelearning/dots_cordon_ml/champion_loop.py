"""Train, screen, challenge, and promote DQN checkpoints in repeated rounds."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import grpc

from . import train
from .audit.integration import (
    add_audit_arguments,
    close_audit,
    effective_config,
    ensure_experiment,
    evaluation_definition,
    open_audit,
    resolve_checkpoint_reference,
)
from .checkpoint import CheckpointMetadata, read_checkpoint
from .self_play import combine_evaluation_results
from .promotion_evaluation import (
    add_evaluation_arguments,
    EvaluatedCheckpoint,
    _device,
    _evaluate_random_screen,
    _passes_random_screen,
    _screen_rank,
    _evaluate_head_to_head_suites,
    _passes_promotion,
    _requires_extended_validation,
    _extended_minimum_match_score,
    _result_dict,
    _evaluated_dict,
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_audit_arguments(parser)
    add_evaluation_arguments(parser)
    parser.add_argument("--server", default="127.0.0.1:50051")
    parser.add_argument(
        "--champion",
        type=Path,
        default=Path("checkpoints/dqn-champion.pt"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("checkpoints/champion-loop"),
        help="directory containing immutable round artifacts and JSON summaries",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="maximum promotions in one invocation; zero leaves this uncapped",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=20,
        help=(
            "stop after this many consecutive rounds without a promotion; "
            "the counter resets after every promotion"
        ),
    )
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--candidate-interval", type=int, default=250)
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help=("training RNG seed; defaults to the current Unix timestamp in seconds"),
    )
    parser.add_argument(
        "--fresh-training-rng",
        action="store_true",
        help=(
            "start each round's training random streams from --training-seed "
            "instead of restoring them from the champion"
        ),
    )
    parser.add_argument(
        "--training-opponent",
        choices=("frozen", "self-play"),
        default="frozen",
        help="opponent used by non-random training episodes",
    )
    parser.add_argument("--random-opponent-probability", type=float, default=0.20)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="optimizer learning rate used for each fine-tuning round",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=2_000,
        help=(
            "replay transitions collected before optimization starts in each "
            "fine-tuning round"
        ),
    )
    parser.add_argument(
        "--terminal-win-bonus",
        type=float,
        default=5.0,
        help="terminal reward added for a win and subtracted for a loss",
    )
    parser.add_argument(
        "--training-opening-random-moves",
        type=int,
        default=4,
        help="random opening moves in each frozen-champion training game",
    )
    parser.add_argument("--training-log-every", type=int, default=25)
    args = parser.parse_args(arguments)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (
        "candidate_count",
        "candidate_interval",
        "evaluation_workers",
        "max_consecutive_failures",
        "training_log_every",
        "screen_games",
        "screen_suites",
        "head_to_head_games",
        "head_to_head_suites",
        "extended_head_to_head_games",
        "extended_head_to_head_suites",
        "rpc_timeout",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_rounds < 0 or args.max_turns < 0 or args.learning_starts < 0:
        parser.error(
            "--max-rounds, --max-turns, and --learning-starts must be non-negative"
        )
    if (
        args.screen_seed < 0
        or args.head_to_head_seed < 0
        or args.extended_head_to_head_seed < 0
    ):
        parser.error("evaluation seeds must be non-negative")
    if args.opening_random_moves < 0 or args.training_opening_random_moves < 0:
        parser.error("opening random-move counts must be non-negative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.terminal_win_bonus < 0:
        parser.error("--terminal-win-bonus must be non-negative")
    if args.head_to_head_games % 2:
        parser.error("--head-to-head-games must be even for paired seats")
    if args.extended_head_to_head_games % 2:
        parser.error("--extended-head-to-head-games must be even for paired seats")
    if not 0 <= args.random_opponent_probability <= 1:
        parser.error("--random-opponent-probability must be between zero and one")
    if args.screen_max_regression < 0:
        parser.error("--screen-max-regression must be non-negative")
    if not 0.5 < args.promotion_min_match_score <= 1:
        parser.error("--promotion-min-match-score must be greater than 0.5")
    if not 1 <= args.promotion_min_suite_wins <= args.head_to_head_suites:
        parser.error(
            "--promotion-min-suite-wins must be between one and --head-to-head-suites"
        )


def _round_seeds(base: int, round_number: int, count: int) -> tuple[int, ...]:
    start = base + (round_number - 1) * count
    return tuple(start + offset for offset in range(count))


def _initial_training_seed(
    configured_seed: int | None,
    round_number: int,
) -> int:
    if configured_seed is None:
        return int(time.time())
    return configured_seed + round_number - 1


def _next_round_number(run_dir: Path) -> int:
    pattern = re.compile(r"^round-(\d+)-")
    existing: list[int] = []
    if run_dir.exists():
        for path in run_dir.iterdir():
            match = pattern.match(path.name)
            if match:
                existing.append(int(match.group(1)))
    return max(existing, default=0) + 1


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _run_training_round(
    args: argparse.Namespace,
    champion: Path,
    metadata: CheckpointMetadata,
    round_dir: Path,
    training_seed: int,
    *,
    audit_service=None,
    audit_attempt_id: str | None = None,
    champion_checkpoint_id: str | None = None,
) -> tuple[Path, ...]:
    checkpoint_dir = round_dir / "candidates"
    final_episode = metadata.episode + args.candidate_count * args.candidate_interval
    arguments = [
        "--server",
        args.server,
        "--rows",
        str(metadata.rows),
        "--columns",
        str(metadata.columns),
        "--channels",
        str(metadata.channels),
        "--blocks",
        str(metadata.blocks),
        "--episodes",
        str(final_episode),
        "--seed",
        str(training_seed),
        "--device",
        args.device,
        "--random-opponent-probability",
        str(args.random_opponent_probability),
        "--learning-rate",
        str(args.learning_rate),
        "--learning-starts",
        str(args.learning_starts),
        "--terminal-win-bonus",
        str(args.terminal_win_bonus),
        "--log-every",
        str(args.training_log_every),
        "--eval-every",
        "0",
        "--checkpoint-every",
        str(args.candidate_interval),
        "--checkpoint-relative-to-start",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--resume",
        str(champion),
        "--max-turns",
        str(args.max_turns),
        "--rpc-timeout",
        str(args.rpc_timeout),
    ]
    if args.training_opponent == "frozen":
        arguments.extend(
            (
                "--frozen-opponent",
                str(champion),
                "--frozen-opening-random-moves",
                str(args.training_opening_random_moves),
            )
        )
    if args.fresh_training_rng:
        arguments.append("--reset-rng-on-resume")

    arguments.extend(("--experiment", args.experiment))
    if audit_service is None:
        arguments.append("--no-audit")
    else:
        arguments.extend(("--resume", f"checkpoint:{champion_checkpoint_id}"))
        if args.training_opponent == "frozen":
            arguments.extend(
                ("--frozen-opponent", f"checkpoint:{champion_checkpoint_id}")
            )
    training_args = train.parse_args(arguments)
    training_args._screening_candidates = audit_service is not None
    if audit_service is None:
        exit_code = train.run(training_args)
    else:
        exit_code = train.run(
            training_args,
            audit_service=audit_service,
            audit_attempt_id=audit_attempt_id,
        )
    if exit_code == 130:
        raise KeyboardInterrupt("training interrupted")
    if exit_code:
        raise RuntimeError(f"training exited with status {exit_code}")

    candidates = tuple(
        checkpoint_dir
        / f"dqn-{metadata.episode + index * args.candidate_interval:07d}.pt"
        for index in range(1, args.candidate_count + 1)
    )
    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        raise ValueError(
            "training ended before every candidate was written: " + ", ".join(missing)
        )
    return candidates


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    args = copy.copy(args)
    args._audit_attempt_id = None
    service = open_audit(args)
    try:
        return _run_loop(args, service)
    except BaseException as exc:
        if service is not None and args._audit_attempt_id is not None:
            try:
                attempt = service.get_attempt(args._audit_attempt_id)
                # A failed compatibility export must not undo a committed promotion.
                if attempt["status"] != "completed":
                    service.update_attempt(
                        attempt["id"],
                        status="interrupted"
                        if isinstance(exc, KeyboardInterrupt)
                        else "failed",
                        error=str(exc) or type(exc).__name__,
                    )
            except Exception as audit_error:
                print(
                    f"could not record champion-loop failure: {audit_error}",
                    file=sys.stderr,
                )
        raise
    finally:
        close_audit(service)


def _run_loop(args: argparse.Namespace, service) -> int:
    experiment = None
    if service is not None:
        experiment = next(
            (
                item
                for item in service.list_experiments()
                if item["name"] == args.experiment
            ),
            None,
        )
        assignment = service.current_champion(experiment["id"]) if experiment else None
        if assignment is None:
            _, metadata = read_checkpoint(args.champion, map_location="cpu")
            experiment = ensure_experiment(
                service, args, metadata.rows, metadata.columns
            )
            checkpoint = service.import_checkpoint(experiment["id"], args.champion)
            assignment = service.bootstrap(experiment["id"], checkpoint["id"])
            print(
                f"audit bootstrapped champion={checkpoint['id']} from {args.champion}",
                flush=True,
            )
        checkpoint = service.get_checkpoint(assignment["checkpoint_id"])
        ensure_experiment(
            service, args, checkpoint["board"]["rows"], checkpoint["board"]["columns"]
        )
    elif not args.champion.is_file():
        raise FileNotFoundError(f"champion checkpoint not found: {args.champion}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    round_number = _next_round_number(args.run_dir)
    initial_training_seed = _initial_training_seed(
        args.training_seed,
        round_number,
    )
    promotions_completed = 0
    rounds_attempted = 0
    consecutive_failures = 0

    while args.max_rounds == 0 or promotions_completed < args.max_rounds:
        champion_source = args.champion
        if service is not None:
            assignment = service.current_champion(experiment["id"])
            # This file is a compatibility export; the assignment in SQL is authoritative.
            service.export_checkpoint(assignment["checkpoint_id"], args.champion)
            champion_source = resolve_checkpoint_reference(
                f"checkpoint:{assignment['checkpoint_id']}", service
            )
        checkpoint, champion_metadata = read_checkpoint(
            champion_source, map_location="cpu"
        )
        del checkpoint
        if (
            args.opening_random_moves
            >= champion_metadata.rows * champion_metadata.columns
        ):
            raise ValueError("--opening-random-moves must be smaller than the board")
        if (
            args.training_opening_random_moves
            >= champion_metadata.rows * champion_metadata.columns
        ):
            raise ValueError(
                "--training-opening-random-moves must be smaller than the board"
            )

        round_dir = (
            args.run_dir
            / f"round-{round_number:03d}-from-{champion_metadata.episode:07d}"
        )
        round_dir.mkdir(parents=True, exist_ok=False)
        round_champion = round_dir / "champion-before.pt"
        _atomic_copy(champion_source, round_champion)
        training_seed = initial_training_seed + rounds_attempted
        audit_kwargs = {}
        if service is not None:
            attempt = service.create_attempt(
                experiment["id"],
                starting_checkpoint_id=assignment["checkpoint_id"],
                champion_at_start_assignment_id=assignment["id"],
                config={**effective_config(args), "training_seed": training_seed},
                target_episode=champion_metadata.episode
                + args.candidate_count * args.candidate_interval,
            )
            args._audit_attempt_id = attempt["id"]
            audit_kwargs = dict(
                audit_service=service,
                audit_attempt_id=attempt["id"],
                champion_checkpoint_id=assignment["checkpoint_id"],
            )
            print(
                f"audit experiment={experiment['id']} attempt={attempt['id']}",
                flush=True,
            )
        print(
            f"\nround={round_number} champion={args.champion} "
            f"episode={champion_metadata.episode} training-seed={training_seed} "
            f"artifacts={round_dir}",
            flush=True,
        )

        candidates = _run_training_round(
            args,
            round_champion,
            champion_metadata,
            round_dir,
            training_seed,
            **audit_kwargs,
        )
        if service is not None:
            candidate_records = {}
            saved = {
                item["episode"]: item
                for item in service.list_checkpoints(attempt["id"])
            }
            for index, path in enumerate(candidates, start=1):
                episode = champion_metadata.episode + index * args.candidate_interval
                record = saved[episode]
                candidate_records[path] = service.import_checkpoint(
                    experiment["id"],
                    path,
                    attempt_id=attempt["id"],
                    checkpoint_id=record["id"],
                    is_screening_candidate=True,
                    candidate_index=index,
                )
            service.update_attempt(attempt["id"], phase="screening")
            screen_seeds = service.reserve_suite_seeds(
                experiment["id"], "screening", args.screen_suites, args.screen_seed
            )
            head_to_head_seeds = service.reserve_suite_seeds(
                experiment["id"],
                "head_to_head",
                args.head_to_head_suites,
                args.head_to_head_seed,
            )
        else:
            screen_seeds = _round_seeds(
                args.screen_seed, round_number, args.screen_suites
            )
            head_to_head_seeds = _round_seeds(
                args.head_to_head_seed, round_number, args.head_to_head_suites
            )
        summary: dict[str, object] = {
            "round": round_number,
            "champion_before": str(round_champion),
            "champion_episode": champion_metadata.episode,
            "training_opponent": args.training_opponent,
            "fresh_training_rng": args.fresh_training_rng,
            "training_seed": training_seed,
            "evaluation_workers": args.evaluation_workers,
            "candidate_count": args.candidate_count,
            "random_opponent_probability": args.random_opponent_probability,
            "learning_rate": args.learning_rate,
            "learning_starts": args.learning_starts,
            "terminal_win_bonus": args.terminal_win_bonus,
            "training_opening_random_moves": args.training_opening_random_moves,
            "candidate_interval": args.candidate_interval,
            "screen_games_per_suite": args.screen_games,
            "screen_seeds": list(screen_seeds),
            "screen_max_regression": args.screen_max_regression,
            "head_to_head_games_per_suite": args.head_to_head_games,
            "head_to_head_seeds": list(head_to_head_seeds),
            "extended_head_to_head_games_per_suite": (args.extended_head_to_head_games),
            "extended_head_to_head_suites": args.extended_head_to_head_suites,
            "extended_head_to_head_seed_base": args.extended_head_to_head_seed,
            "opening_random_moves": args.opening_random_moves,
            "promotion_min_match_score": args.promotion_min_match_score,
            "promotion_min_suite_wins": args.promotion_min_suite_wins,
            "max_consecutive_failures": args.max_consecutive_failures,
        }

        screen_kwargs = {}
        if service is not None:
            definitions = [
                evaluation_definition(
                    args,
                    rows=champion_metadata.rows,
                    columns=champion_metadata.columns,
                    seed=seed,
                    games=args.screen_games,
                )
                for seed in screen_seeds
            ]
            screen_batches = {}
            for path, checkpoint_id in [
                (round_champion, assignment["checkpoint_id"]),
                *[(path, candidate_records[path]["id"]) for path in candidates],
            ]:
                screen_batches[path] = service.create_evaluation(
                    experiment["id"],
                    checkpoint_id,
                    purpose="screening",
                    attempt_id=attempt["id"],
                    suite_definitions=definitions,
                    config={"screen_max_regression": args.screen_max_regression},
                )
            screen_kwargs = dict(
                audit_service=service,
                evaluation_ids=tuple(item["id"] for item in screen_batches.values()),
            )
            summary.update(
                experiment_id=experiment["id"],
                attempt_id=attempt["id"],
                champion_assignment_id=assignment["id"],
            )
        screened = _evaluate_random_screen(
            args,
            (round_champion, *candidates),
            device,
            screen_seeds,
            args.screen_games,
            **screen_kwargs,
        )
        champion_result = screened[0]
        candidate_results = screened[1:]
        summary["random_screen"] = {
            "champion": _evaluated_dict(champion_result),
            "candidates": [_evaluated_dict(item) for item in candidate_results],
        }

        contenders = sorted(
            (
                item
                for item in candidate_results
                if _passes_random_screen(
                    item, champion_result, args.screen_max_regression
                )
            ),
            key=_screen_rank,
            reverse=True,
        )
        if service is not None:
            service.select_best_screened_checkpoint(
                attempt["id"],
                [screen_batches[path]["id"] for path in candidates],
            )
            ranks = {item.path: index for index, item in enumerate(contenders, start=1)}
            for candidate in candidate_results:
                qualified = candidate.path in ranks
                service.record_decision(
                    attempt["id"],
                    candidate_records[candidate.path]["id"],
                    stage="screening",
                    result="qualified" if qualified else "rejected",
                    reason="within allowed random-score regression"
                    if qualified
                    else "random-score regression exceeds threshold",
                    candidate_evaluation_id=screen_batches[candidate.path]["id"],
                    champion_evaluation_id=screen_batches[round_champion]["id"],
                    policy={"screen_max_regression": args.screen_max_regression},
                    rank=ranks.get(candidate.path),
                )
            service.update_attempt(attempt["id"], phase="challenging")
        print(
            "random-screen contenders="
            + (
                ",".join(str(item.metadata.episode) for item in contenders)
                if contenders
                else "none"
            ),
            flush=True,
        )

        challenges: list[dict[str, object]] = []
        promoted: EvaluatedCheckpoint | None = None
        extended_head_to_head_seeds: tuple[int, ...] | None = None
        for contender in contenders:
            challenge_kwargs = {}
            promotion_policy = {
                "promotion_min_match_score": args.promotion_min_match_score,
                "promotion_min_suite_wins": args.promotion_min_suite_wins,
            }
            if service is not None:
                challenge_batch = service.create_evaluation(
                    experiment["id"],
                    candidate_records[contender.path]["id"],
                    purpose="head_to_head",
                    attempt_id=attempt["id"],
                    opponent_checkpoint_id=assignment["checkpoint_id"],
                    suite_definitions=[
                        evaluation_definition(
                            args,
                            rows=champion_metadata.rows,
                            columns=champion_metadata.columns,
                            seed=seed,
                            games=args.head_to_head_games,
                            kind="head_to_head",
                            opening_random_moves=args.opening_random_moves,
                        )
                        for seed in head_to_head_seeds
                    ],
                    config=promotion_policy,
                )
                challenge_kwargs = dict(
                    audit_service=service, evaluation_id=challenge_batch["id"]
                )
            suites = _evaluate_head_to_head_suites(
                args,
                contender,
                champion_result,
                device,
                head_to_head_seeds,
                args.head_to_head_games,
                args.opening_random_moves,
                **challenge_kwargs,
            )
            aggregate = combine_evaluation_results(suites)
            suite_wins = sum(result.overall.match_score > 0.5 for result in suites)
            passed = _passes_promotion(
                suites,
                args.promotion_min_match_score,
                args.promotion_min_suite_wins,
            )
            requires_extended = not passed and _requires_extended_validation(
                suites, args.promotion_min_match_score
            )
            challenge_record: dict[str, object] = {
                "path": str(contender.path),
                "episode": contender.metadata.episode,
                "suites": [_result_dict(result) for result in suites],
                "aggregate": _result_dict(aggregate),
                "suite_wins": suite_wins,
                "initial_passed": passed,
                "passed": passed,
                "extended": None,
            }
            print(
                f"promotion-gate episode={contender.metadata.episode} "
                f"suite-wins={suite_wins}/{args.head_to_head_suites} "
                f"passed={'yes' if passed else 'no'} "
                f"extended={'yes' if requires_extended else 'no'}",
                flush=True,
            )

            promotion_evaluation_id = (
                challenge_batch["id"] if service is not None else None
            )
            if requires_extended:
                if service is not None:
                    service.record_decision(
                        attempt["id"],
                        candidate_records[contender.path]["id"],
                        stage="challenge",
                        result="extended",
                        reason=(
                            "initial aggregate match score is above 0.5 but below "
                            "the normal promotion threshold"
                        ),
                        candidate_evaluation_id=challenge_batch["id"],
                        champion_evaluation_id=screen_batches[round_champion]["id"],
                        policy=promotion_policy,
                    )
                if extended_head_to_head_seeds is None:
                    if service is not None:
                        extended_head_to_head_seeds = service.reserve_suite_seeds(
                            experiment["id"],
                            "extended_head_to_head",
                            args.extended_head_to_head_suites,
                            args.extended_head_to_head_seed,
                        )
                    else:
                        extended_head_to_head_seeds = _round_seeds(
                            args.extended_head_to_head_seed,
                            round_number,
                            args.extended_head_to_head_suites,
                        )
                extended_total_games = (
                    args.extended_head_to_head_games * args.extended_head_to_head_suites
                )
                extended_policy = {
                    "promotion_min_match_score": _extended_minimum_match_score(
                        extended_total_games
                    ),
                    "promotion_min_suite_wins": 1,
                    "extended_validation": True,
                }
                extended_kwargs = {}
                if service is not None:
                    extended_batch = service.create_evaluation(
                        experiment["id"],
                        candidate_records[contender.path]["id"],
                        purpose="head_to_head",
                        attempt_id=attempt["id"],
                        opponent_checkpoint_id=assignment["checkpoint_id"],
                        suite_definitions=[
                            evaluation_definition(
                                args,
                                rows=champion_metadata.rows,
                                columns=champion_metadata.columns,
                                seed=seed,
                                games=args.extended_head_to_head_games,
                                kind="head_to_head",
                                opening_random_moves=args.opening_random_moves,
                            )
                            for seed in extended_head_to_head_seeds
                        ],
                        config=extended_policy,
                    )
                    promotion_evaluation_id = extended_batch["id"]
                    extended_kwargs = dict(
                        audit_service=service,
                        evaluation_id=extended_batch["id"],
                    )
                extended_suites = _evaluate_head_to_head_suites(
                    args,
                    contender,
                    champion_result,
                    device,
                    extended_head_to_head_seeds,
                    args.extended_head_to_head_games,
                    args.opening_random_moves,
                    **extended_kwargs,
                )
                extended_aggregate = combine_evaluation_results(extended_suites)
                passed = _passes_promotion(
                    extended_suites,
                    extended_policy["promotion_min_match_score"],
                    extended_policy["promotion_min_suite_wins"],
                )
                challenge_record["extended"] = {
                    "seeds": list(extended_head_to_head_seeds),
                    "games_per_suite": args.extended_head_to_head_games,
                    "total_games": extended_total_games,
                    "suites": [_result_dict(result) for result in extended_suites],
                    "aggregate": _result_dict(extended_aggregate),
                    "passed": passed,
                }
                challenge_record["passed"] = passed
                promotion_policy = extended_policy
                if service is not None:
                    service.record_decision(
                        attempt["id"],
                        candidate_records[contender.path]["id"],
                        stage="challenge",
                        result="passed" if passed else "rejected",
                        candidate_evaluation_id=extended_batch["id"],
                        champion_evaluation_id=screen_batches[round_champion]["id"],
                        policy=extended_policy,
                    )
                print(
                    f"extended-promotion-gate episode={contender.metadata.episode} "
                    f"suites={args.extended_head_to_head_suites} "
                    f"games={extended_total_games} "
                    f"match-score={extended_aggregate.overall.match_score:.4f} "
                    f"passed={'yes' if passed else 'no'}",
                    flush=True,
                )
            elif service is not None:
                service.record_decision(
                    attempt["id"],
                    candidate_records[contender.path]["id"],
                    stage="challenge",
                    result="passed" if passed else "rejected",
                    candidate_evaluation_id=challenge_batch["id"],
                    champion_evaluation_id=screen_batches[round_champion]["id"],
                    policy=promotion_policy,
                )

            challenges.append(challenge_record)
            if passed:
                promoted = contender
                if service is not None:
                    service.promote(
                        experiment["id"],
                        candidate_records[contender.path]["id"],
                        attempt_id=attempt["id"],
                        expected_assignment_id=assignment["id"],
                        candidate_evaluation_id=promotion_evaluation_id,
                        champion_evaluation_id=screen_batches[round_champion]["id"],
                        policy=promotion_policy,
                    )
                    for skipped in contenders[contenders.index(contender) + 1 :]:
                        service.record_decision(
                            attempt["id"],
                            candidate_records[skipped.path]["id"],
                            stage="challenge",
                            result="skipped",
                            reason="earlier ranked contender promoted",
                            policy=promotion_policy,
                        )
                break

        summary["head_to_head"] = challenges
        if promoted is None:
            consecutive_failures += 1
            summary["promoted"] = None
            summary["stop_reason"] = (
                "no_random_screen_improvement"
                if not contenders
                else "no_head_to_head_challenger_passed"
            )
            summary["consecutive_failures"] = consecutive_failures
            if service is not None:
                service.update_attempt(
                    attempt["id"],
                    status="completed",
                    phase="finished",
                    outcome="no_qualified_candidate"
                    if not contenders
                    else "no_challenger_passed",
                    stop_reason=summary["stop_reason"],
                )
            _write_summary(round_dir / "results.json", summary)
            rounds_attempted += 1
            round_number += 1
            if consecutive_failures >= args.max_consecutive_failures:
                print(
                    f"champion unchanged at episode={champion_metadata.episode}; "
                    f"stopping after {consecutive_failures} consecutive "
                    f"unsuccessful rounds ({summary['stop_reason']})",
                    flush=True,
                )
                return 0
            print(
                f"champion unchanged at episode={champion_metadata.episode}; "
                f"consecutive-failures={consecutive_failures}/"
                f"{args.max_consecutive_failures}; restarting with a new seed "
                f"({summary['stop_reason']})",
                flush=True,
            )
            continue

        if service is not None:
            service.export_checkpoint(
                candidate_records[promoted.path]["id"], args.champion
            )
        else:
            _atomic_copy(promoted.path, args.champion)
        summary["promoted"] = {
            "path": str(promoted.path),
            "episode": promoted.metadata.episode,
            "champion_path": str(args.champion),
        }
        summary["stop_reason"] = None
        summary["consecutive_failures"] = 0
        _write_summary(round_dir / "results.json", summary)
        print(
            f"promoted episode={promoted.metadata.episode} to {args.champion}",
            flush=True,
        )

        consecutive_failures = 0
        promotions_completed += 1
        rounds_attempted += 1
        round_number += 1

    print(f"reached --max-rounds={args.max_rounds}; stopping", flush=True)
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("champion loop interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError, grpc.RpcError) as exc:
        print(f"champion loop failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
