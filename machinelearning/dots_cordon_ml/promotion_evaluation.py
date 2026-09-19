"""Shared parallel checkpoint evaluation and promotion gates for training and replay."""

from __future__ import annotations

import argparse
from concurrent.futures import (
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, replace
import multiprocessing
from pathlib import Path

import torch

from .audit.integration import resolve_checkpoint_reference, evaluation_result_dict
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
from .search import PolicyValueAgent


@dataclass(frozen=True, slots=True)
class EvaluatedCheckpoint:
    path: Path
    metadata: CheckpointMetadata
    suites: tuple[EvaluationResult, ...]
    aggregate: EvaluationResult


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
) -> DQNAgent | PolicyValueAgent:
    if metadata.kind == "policy_value":
        checkpoint, _ = read_checkpoint(path, map_location="cpu")
        agent = PolicyValueAgent(device, metadata.channels, metadata.blocks)
        agent.online.load_state_dict(checkpoint["online"])
        return agent
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


def _requires_extended_validation(
    results: tuple[EvaluationResult, ...],
    normal_minimum_match_score: float,
) -> bool:
    match_score = combine_evaluation_results(results).overall.match_score
    return 0.5 < match_score < normal_minimum_match_score


def _extended_minimum_match_score(games: int) -> float:
    # A draw is half a match point, so this is the smallest score strictly
    # greater than 0.5 that can be represented by a suite of this size.
    return 0.5 + 0.5 / games


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


def add_evaluation_arguments(parser):
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=4,
        help=("maximum parallel random-screen checkpoints or head-to-head suites"),
    )
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
    parser.add_argument(
        "--extended-head-to-head-games",
        type=int,
        default=1_000,
        help=(
            "paired-opening games per extended validation suite for a challenger "
            "whose initial aggregate score is above 0.5 but below the threshold"
        ),
    )
    parser.add_argument(
        "--extended-head-to-head-suites",
        type=int,
        default=10,
        help="independently seeded suites in extended head-to-head validation",
    )
    parser.add_argument(
        "--extended-head-to-head-seed",
        type=int,
        default=50_000_001,
        help="base seed for extended head-to-head validation",
    )
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
