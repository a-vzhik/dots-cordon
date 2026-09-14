"""Train, screen, challenge, and promote DQN checkpoints in repeated rounds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys

import grpc
import torch

from . import train
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
    parser.add_argument("--training-seed", type=int, default=7)
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
            "--promotion-min-suite-wins must be between one and "
            "--head-to-head-suites"
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


def _round_seeds(base: int, round_number: int, count: int) -> tuple[int, ...]:
    start = base + (round_number - 1) * count
    return tuple(start + offset for offset in range(count))


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
    round_number: int,
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
        str(args.training_seed + round_number - 1),
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

    exit_code = train.run(train.parse_args(arguments))
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
    environment: GameEnvironment,
    path: Path,
    device: torch.device,
    suite_seeds: tuple[int, ...],
    games: int,
) -> EvaluatedCheckpoint:
    checkpoint, metadata = read_checkpoint(path, map_location="cpu")
    del checkpoint
    agent = _load_agent(path, metadata, device, suite_seeds[0])
    results: list[EvaluationResult] = []
    print(f"random-screen checkpoint={path} episode={metadata.episode}", flush=True)
    for suite_seed in suite_seeds:
        result = evaluate_against_random(
            environment,
            agent,
            random_game_seeds(games, suite_seed),
        )
        results.append(result)
        print(f"  seed={suite_seed} {_format_stats(result.overall)}", flush=True)
    aggregate = combine_evaluation_results(results)
    print(f"  aggregate {_format_stats(aggregate.overall)}", flush=True)
    return EvaluatedCheckpoint(path, metadata, tuple(results), aggregate)


def _passes_random_screen(
    candidate: EvaluatedCheckpoint,
    champion: EvaluatedCheckpoint,
    maximum_regression: float,
) -> bool:
    score_delta = (
        candidate.aggregate.overall.match_score
        - champion.aggregate.overall.match_score
    )
    return score_delta + maximum_regression >= -1e-12


def _screen_rank(item: EvaluatedCheckpoint) -> tuple[float, float]:
    return (
        item.aggregate.overall.match_score,
        item.aggregate.overall.mean_score_difference,
    )


def _evaluate_head_to_head_suites(
    environment: GameEnvironment,
    candidate: EvaluatedCheckpoint,
    champion: EvaluatedCheckpoint,
    device: torch.device,
    suite_seeds: tuple[int, ...],
    games: int,
    opening_random_moves: int,
) -> tuple[EvaluationResult, ...]:
    candidate_agent = _load_agent(
        candidate.path, candidate.metadata, device, suite_seeds[0]
    )
    champion_agent = _load_agent(
        champion.path, champion.metadata, device, suite_seeds[0] + 1
    )
    results: list[EvaluationResult] = []
    print(
        f"head-to-head challenger={candidate.path} episode={candidate.metadata.episode} "
        f"champion-episode={champion.metadata.episode}",
        flush=True,
    )
    for suite_seed in suite_seeds:
        result = evaluate_head_to_head(
            environment,
            candidate_agent,
            champion_agent,
            random_game_seeds(games, suite_seed),
            opening_random_moves,
        )
        results.append(result)
        print(f"  seed={suite_seed} {_format_stats(result.overall)}", flush=True)
    aggregate = combine_evaluation_results(results)
    print(f"  aggregate {_format_stats(aggregate.overall)}", flush=True)
    return tuple(results)


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
    if not args.champion.is_file():
        raise FileNotFoundError(f"champion checkpoint not found: {args.champion}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    round_number = _next_round_number(args.run_dir)
    rounds_completed = 0

    while args.max_rounds == 0 or rounds_completed < args.max_rounds:
        checkpoint, champion_metadata = read_checkpoint(
            args.champion, map_location="cpu"
        )
        del checkpoint
        if args.opening_random_moves >= champion_metadata.rows * champion_metadata.columns:
            raise ValueError("--opening-random-moves must be smaller than the board")

        round_dir = (
            args.run_dir
            / f"round-{round_number:03d}-from-{champion_metadata.episode:07d}"
        )
        round_dir.mkdir(parents=True, exist_ok=False)
        round_champion = round_dir / "champion-before.pt"
        _atomic_copy(args.champion, round_champion)
        print(
            f"\nround={round_number} champion={args.champion} "
            f"episode={champion_metadata.episode} artifacts={round_dir}",
            flush=True,
        )

        candidates = _run_training_round(
            args,
            round_champion,
            champion_metadata,
            round_dir,
            round_number,
        )
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

        with GameEnvironment(
            target=args.server,
            rows=champion_metadata.rows,
            columns=champion_metadata.columns,
            max_turns=args.max_turns,
            rpc_timeout=args.rpc_timeout,
        ) as environment:
            champion_result = _evaluate_against_random_suites(
                environment,
                round_champion,
                device,
                screen_seeds,
                args.screen_games,
            )
            candidate_results = [
                _evaluate_against_random_suites(
                    environment,
                    path,
                    device,
                    screen_seeds,
                    args.screen_games,
                )
                for path in candidates
            ]
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
                suites = _evaluate_head_to_head_suites(
                    environment,
                    contender,
                    champion_result,
                    device,
                    head_to_head_seeds,
                    args.head_to_head_games,
                    args.opening_random_moves,
                )
                aggregate = combine_evaluation_results(suites)
                suite_wins = sum(
                    result.overall.match_score > 0.5 for result in suites
                )
                passed = _passes_promotion(
                    suites,
                    args.promotion_min_match_score,
                    args.promotion_min_suite_wins,
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
                    break

        summary["head_to_head"] = challenges
        if promoted is None:
            summary["promoted"] = None
            summary["stop_reason"] = (
                "no_random_screen_improvement"
                if not contenders
                else "no_head_to_head_challenger_passed"
            )
            _write_summary(round_dir / "results.json", summary)
            print(
                f"champion unchanged at episode={champion_metadata.episode}; "
                f"stopping ({summary['stop_reason']})",
                flush=True,
            )
            return 0

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
