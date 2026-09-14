"""Train, screen, challenge, and promote DQN checkpoints in repeated rounds."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import (
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, replace
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import time

import grpc
import torch

from . import train
from .audit.integration import (
    add_audit_arguments,
    close_audit,
    effective_config,
    ensure_experiment,
    evaluation_definition,
    evaluation_result_dict,
    open_audit,
    resolve_checkpoint_reference,
)
from .checkpoint import CheckpointMetadata, read_checkpoint, restore_agent
from .dqn import DQNAgent
from .environment import GameEnvironment
from .self_play import (
    EvaluationResult,
    MatchStats,
    combine_evaluation_results,
    evaluate_against_random,
    evaluate_head_to_head,
    random_game_seeds,
)


@dataclass(frozen=True, slots=True)
class EvaluatedCheckpoint:
    path: Path
    metadata: CheckpointMetadata
    suites: tuple[EvaluationResult, ...]
    aggregate: EvaluationResult


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_audit_arguments(parser)
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
        help="maximum promoted rounds; zero continues until no challenger passes",
    )
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--candidate-interval", type=int, default=250)
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=5,
        help=("maximum parallel random-screen checkpoints or head-to-head suites"),
    )
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
    parser.add_argument("--random-opponent-probability", type=float, default=0.50)
    parser.add_argument("--training-log-every", type=int, default=25)
    parser.add_argument("--screen-games", type=int, default=1_000)
    parser.add_argument("--screen-suites", type=int, default=3)
    parser.add_argument("--screen-seed", type=int, default=30_000_001)
    parser.add_argument(
        "--screen-max-regression",
        type=float,
        default=0.003,
        help=(
            "maximum aggregate random match-score deficit allowed for a "
            "head-to-head challenge"
        ),
    )
    parser.add_argument("--head-to-head-games", type=int, default=1_000)
    parser.add_argument("--head-to-head-suites", type=int, default=3)
    parser.add_argument("--head-to-head-seed", type=int, default=40_000_001)
    parser.add_argument("--opening-random-moves", type=int, default=4)
    parser.add_argument(
        "--promotion-min-match-score",
        type=float,
        default=0.52,
        help="minimum combined challenger head-to-head match score",
    )
    parser.add_argument(
        "--promotion-min-suite-wins",
        type=int,
        default=2,
        help="number of head-to-head suites the challenger must score above 0.5",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="zero lets the server play until the board is full",
    )
    parser.add_argument("--rpc-timeout", type=float, default=10.0)
    args = parser.parse_args(arguments)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (
        "candidate_count",
        "candidate_interval",
        "evaluation_workers",
        "training_log_every",
        "screen_games",
        "screen_suites",
        "head_to_head_games",
        "head_to_head_suites",
        "rpc_timeout",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_rounds < 0 or args.max_turns < 0:
        parser.error("--max-rounds and --max-turns must be non-negative")
    if args.screen_seed < 0 or args.head_to_head_seed < 0:
        parser.error("evaluation seeds must be non-negative")
    if args.opening_random_moves < 0:
        parser.error("--opening-random-moves must be non-negative")
    if args.head_to_head_games % 2:
        parser.error("--head-to-head-games must be even for paired seats")
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


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_agent(
    path: Path,
    metadata: CheckpointMetadata,
    device: torch.device,
    seed: int,
) -> DQNAgent:
    agent = DQNAgent(
        device=device,
        learning_rate=3e-4,
        gamma=0.99,
        seed=seed,
        channels=metadata.channels,
        blocks=metadata.blocks,
    )
    checkpoint, _ = read_checkpoint(path, map_location=device)
    restore_agent(agent, checkpoint, restore_optimizer=False)
    return agent


def _game_environment(
    args: argparse.Namespace,
    metadata: CheckpointMetadata,
) -> GameEnvironment:
    return GameEnvironment(
        target=args.server,
        rows=metadata.rows,
        columns=metadata.columns,
        max_turns=args.max_turns,
        rpc_timeout=args.rpc_timeout,
    )


def _evaluation_executor(device: torch.device, max_workers: int) -> Executor:
    if device.type == "mps":
        return ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
    return ThreadPoolExecutor(max_workers=max_workers)


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
        arguments.extend(("--frozen-opponent", str(champion)))
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


def _evaluate_against_random_suites(
    args: argparse.Namespace,
    path: Path,
    device: torch.device,
    suite_seeds: tuple[int, ...],
    games: int,
) -> EvaluatedCheckpoint:
    checkpoint, metadata = read_checkpoint(path, map_location="cpu")
    del checkpoint
    agent = _load_agent(path, metadata, device, suite_seeds[0])
    results: list[EvaluationResult] = []
    with _game_environment(args, metadata) as environment:
        for suite_seed in suite_seeds:
            result = evaluate_against_random(
                environment,
                agent,
                random_game_seeds(games, suite_seed),
            )
            results.append(result)
            print(
                f"random-screen checkpoint={path} episode={metadata.episode} "
                f"seed={suite_seed} {_format_stats(result.overall)}",
                flush=True,
            )
    aggregate = combine_evaluation_results(results)
    print(
        f"random-screen checkpoint={path} episode={metadata.episode} "
        f"aggregate {_format_stats(aggregate.overall)}",
        flush=True,
    )
    return EvaluatedCheckpoint(path, metadata, tuple(results), aggregate)


def _evaluate_random_screen(
    args: argparse.Namespace,
    paths: tuple[Path, ...],
    device: torch.device,
    suite_seeds: tuple[int, ...],
    games: int,
    *,
    audit_service=None,
    evaluation_ids: tuple[str, ...] = (),
) -> tuple[EvaluatedCheckpoint, ...]:
    if audit_service is not None:
        # Only the parent writes SQL; worker processes receive plain inputs.
        results = [[None] * len(suite_seeds) for _ in paths]
        try:
            snapshots = [
                resolve_checkpoint_reference(
                    f"checkpoint:{audit_service.get_evaluation(batch_id)['checkpoint_id']}",
                    audit_service,
                )
                for batch_id in evaluation_ids
            ]
            with _evaluation_executor(
                device, min(args.evaluation_workers, len(paths) * len(suite_seeds))
            ) as executor:
                futures = {
                    executor.submit(
                        _evaluate_against_random_suites,
                        args,
                        path,
                        device,
                        (seed,),
                        games,
                    ): (path_index, suite_index)
                    for path_index, path in enumerate(snapshots)
                    for suite_index, seed in enumerate(suite_seeds)
                }
                failure = None
                for future in as_completed(futures):
                    path_index, suite_index = futures[future]
                    try:
                        evaluated = future.result()
                    except Exception as exc:
                        failure = failure or exc
                        continue
                    audit_service.complete_suite(
                        evaluation_ids[path_index],
                        suite_index,
                        evaluation_result_dict(evaluated.suites[0]),
                    )
                    results[path_index][suite_index] = evaluated
                if failure is not None:
                    raise failure
        except BaseException as exc:
            _fail_evaluations(audit_service, evaluation_ids, exc)
            raise
        return tuple(
            EvaluatedCheckpoint(
                path,
                items[0].metadata,
                tuple(item.suites[0] for item in items),
                combine_evaluation_results([item.suites[0] for item in items]),
            )
            for path, items in zip(paths, results, strict=True)
        )
    worker_count = min(args.evaluation_workers, len(paths))
    print(
        f"random-screen checkpoints={len(paths)} parallel-workers={worker_count}",
        flush=True,
    )
    with _evaluation_executor(device, worker_count) as executor:
        futures = tuple(
            executor.submit(
                _evaluate_against_random_suites,
                args,
                path,
                device,
                suite_seeds,
                games,
            )
            for path in paths
        )
        return tuple(future.result() for future in futures)


def _passes_random_screen(
    candidate: EvaluatedCheckpoint,
    champion: EvaluatedCheckpoint,
    maximum_regression: float,
) -> bool:
    score_delta = (
        candidate.aggregate.overall.match_score - champion.aggregate.overall.match_score
    )
    return score_delta + maximum_regression >= -1e-12


def _screen_rank(item: EvaluatedCheckpoint) -> tuple[float, float]:
    return (
        item.aggregate.overall.match_score,
        item.aggregate.overall.mean_score_difference,
    )


def _evaluate_head_to_head_suites(
    args: argparse.Namespace,
    candidate: EvaluatedCheckpoint,
    champion: EvaluatedCheckpoint,
    device: torch.device,
    suite_seeds: tuple[int, ...],
    games: int,
    opening_random_moves: int,
    *,
    audit_service=None,
    evaluation_id: str | None = None,
) -> tuple[EvaluationResult, ...]:
    worker_count = min(args.evaluation_workers, len(suite_seeds))
    print(
        f"head-to-head challenger={candidate.path} episode={candidate.metadata.episode} "
        f"champion-episode={champion.metadata.episode} "
        f"parallel-workers={worker_count}",
        flush=True,
    )
    completed = [None] * len(suite_seeds)
    try:
        if audit_service is not None:
            batch = audit_service.get_evaluation(evaluation_id)
            candidate = replace(
                candidate,
                path=resolve_checkpoint_reference(
                    f"checkpoint:{batch['checkpoint_id']}", audit_service
                ),
            )
            champion = replace(
                champion,
                path=resolve_checkpoint_reference(
                    f"checkpoint:{batch['opponent_checkpoint_id']}", audit_service
                ),
            )
        with _evaluation_executor(device, worker_count) as executor:
            futures = {
                executor.submit(
                    _evaluate_head_to_head_suite,
                    args,
                    candidate,
                    champion,
                    device,
                    suite_seed,
                    games,
                    opening_random_moves,
                ): index
                for index, suite_seed in enumerate(suite_seeds)
            }
            failure = None
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failure = failure or exc
                    continue
                if audit_service is not None:
                    audit_service.complete_suite(
                        evaluation_id, index, evaluation_result_dict(result)
                    )
                completed[index] = result
            if failure is not None:
                raise failure
        results = tuple(completed)
    except BaseException as exc:
        if audit_service is not None:
            _fail_evaluations(audit_service, (evaluation_id,), exc)
        raise

    for suite_seed, result in zip(suite_seeds, results, strict=True):
        print(f"  seed={suite_seed} {_format_stats(result.overall)}", flush=True)
    aggregate = combine_evaluation_results(results)
    print(f"  aggregate {_format_stats(aggregate.overall)}", flush=True)
    return results


def _fail_evaluations(service, evaluation_ids, error: BaseException) -> None:
    for evaluation_id in evaluation_ids:
        if service.get_evaluation(evaluation_id)["status"] == "running":
            service.fail_evaluation(
                evaluation_id,
                str(error) or type(error).__name__,
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
            )


def _evaluate_head_to_head_suite(
    args: argparse.Namespace,
    candidate: EvaluatedCheckpoint,
    champion: EvaluatedCheckpoint,
    device: torch.device,
    suite_seed: int,
    games: int,
    opening_random_moves: int,
) -> EvaluationResult:
    candidate_agent = _load_agent(
        candidate.path, candidate.metadata, device, suite_seed
    )
    champion_agent = _load_agent(
        champion.path, champion.metadata, device, suite_seed + 1
    )
    with _game_environment(args, candidate.metadata) as environment:
        return evaluate_head_to_head(
            environment,
            candidate_agent,
            champion_agent,
            random_game_seeds(games, suite_seed),
            opening_random_moves,
        )


def _passes_promotion(
    results: tuple[EvaluationResult, ...],
    minimum_match_score: float,
    minimum_suite_wins: int,
) -> bool:
    aggregate = combine_evaluation_results(results)
    suite_wins = sum(item.overall.match_score > 0.5 for item in results)
    return (
        aggregate.overall.match_score >= minimum_match_score
        and suite_wins >= minimum_suite_wins
    )


def _format_stats(stats: MatchStats) -> str:
    return (
        f"games={stats.games} W/D/L={stats.wins}/{stats.draws}/{stats.losses} "
        f"match_score={stats.match_score:.4f} "
        f"mean_score_diff={stats.mean_score_difference:+.3f}"
    )


def _stats_dict(stats: MatchStats) -> dict[str, int | float]:
    return {
        "games": stats.games,
        "wins": stats.wins,
        "draws": stats.draws,
        "losses": stats.losses,
        "match_score": stats.match_score,
        "mean_score_difference": stats.mean_score_difference,
    }


def _result_dict(result: EvaluationResult) -> dict[str, object]:
    return {
        "overall": _stats_dict(result.overall),
        "as_player_0": _stats_dict(result.as_player_0),
        "as_player_1": _stats_dict(result.as_player_1),
    }


def _evaluated_dict(item: EvaluatedCheckpoint) -> dict[str, object]:
    return {
        "path": str(item.path),
        "episode": item.metadata.episode,
        "suites": [_result_dict(result) for result in item.suites],
        "aggregate": _result_dict(item.aggregate),
    }


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
    rounds_completed = 0

    while args.max_rounds == 0 or rounds_completed < args.max_rounds:
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

        round_dir = (
            args.run_dir
            / f"round-{round_number:03d}-from-{champion_metadata.episode:07d}"
        )
        round_dir.mkdir(parents=True, exist_ok=False)
        round_champion = round_dir / "champion-before.pt"
        _atomic_copy(champion_source, round_champion)
        training_seed = initial_training_seed + rounds_completed
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
            "random_opponent_probability": args.random_opponent_probability,
            "candidate_interval": args.candidate_interval,
            "screen_games_per_suite": args.screen_games,
            "screen_seeds": list(screen_seeds),
            "screen_max_regression": args.screen_max_regression,
            "head_to_head_games_per_suite": args.head_to_head_games,
            "head_to_head_seeds": list(head_to_head_seeds),
            "opening_random_moves": args.opening_random_moves,
            "promotion_min_match_score": args.promotion_min_match_score,
            "promotion_min_suite_wins": args.promotion_min_suite_wins,
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
            if service is not None:
                service.record_decision(
                    attempt["id"],
                    candidate_records[contender.path]["id"],
                    stage="challenge",
                    result="passed" if passed else "rejected",
                    candidate_evaluation_id=challenge_batch["id"],
                    champion_evaluation_id=screen_batches[round_champion]["id"],
                    policy=promotion_policy,
                )
            challenges.append(
                {
                    "path": str(contender.path),
                    "episode": contender.metadata.episode,
                    "suites": [_result_dict(result) for result in suites],
                    "aggregate": _result_dict(aggregate),
                    "suite_wins": suite_wins,
                    "passed": passed,
                }
            )
            print(
                f"promotion-gate episode={contender.metadata.episode} "
                f"suite-wins={suite_wins}/{args.head_to_head_suites} "
                f"passed={'yes' if passed else 'no'}",
                flush=True,
            )
            if passed:
                promoted = contender
                if service is not None:
                    service.promote(
                        experiment["id"],
                        candidate_records[contender.path]["id"],
                        attempt_id=attempt["id"],
                        expected_assignment_id=assignment["id"],
                        candidate_evaluation_id=challenge_batch["id"],
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
            summary["promoted"] = None
            summary["stop_reason"] = (
                "no_random_screen_improvement"
                if not contenders
                else "no_head_to_head_challenger_passed"
            )
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
            print(
                f"champion unchanged at episode={champion_metadata.episode}; "
                f"stopping ({summary['stop_reason']})",
                flush=True,
            )
            return 0

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
        _write_summary(round_dir / "results.json", summary)
        print(
            f"promoted episode={promoted.metadata.episode} to {args.champion}",
            flush=True,
        )

        rounds_completed += 1
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
