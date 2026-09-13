"""Compare two DQN checkpoints directly with paired randomized openings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import grpc
import torch

from .checkpoint import CheckpointMetadata, read_checkpoint, restore_agent
from .dqn import DQNAgent
from .environment import GameEnvironment
from .self_play import (
    EvaluationResult,
    MatchStats,
    evaluate_head_to_head,
    random_game_seeds,
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_a", type=Path)
    parser.add_argument("candidate_b", type=Path)
    parser.add_argument("--server", default="127.0.0.1:50051")
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20_260_917)
    parser.add_argument(
        "--opening-random-moves",
        type=int,
        default=4,
        help="random moves at the start of each paired game; zero is deterministic",
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
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number for paired seats")
    if args.opening_random_moves < 0:
        parser.error("--opening-random-moves must be non-negative")
    if args.max_turns < 0:
        parser.error("--max-turns must be non-negative")
    if args.rpc_timeout <= 0:
        parser.error("--rpc-timeout must be positive")
    return args


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


def _opponent_stats(stats: MatchStats) -> MatchStats:
    return MatchStats(
        games=stats.games,
        wins=stats.losses,
        draws=stats.draws,
        losses=stats.wins,
        mean_score_difference=-stats.mean_score_difference,
    )


def _opponent_result(candidate_a: EvaluationResult) -> EvaluationResult:
    return EvaluationResult(
        overall=_opponent_stats(candidate_a.overall),
        as_player_0=_opponent_stats(candidate_a.as_player_1),
        as_player_1=_opponent_stats(candidate_a.as_player_0),
    )


def _format_stats(stats: MatchStats) -> str:
    return (
        f"games={stats.games} "
        f"W/D/L={stats.wins}/{stats.draws}/{stats.losses} "
        f"match_score={stats.match_score:.3f} "
        f"mean_score_diff={stats.mean_score_difference:+.3f}"
    )


def _print_candidate(label: str, result: EvaluationResult) -> None:
    print(f"{label}")
    print(f"  overall     {_format_stats(result.overall)}")
    print(f"  as-player-0 {_format_stats(result.as_player_0)}")
    print(f"  as-player-1 {_format_stats(result.as_player_1)}")


def run(args: argparse.Namespace) -> int:
    checkpoint_a, metadata_a = read_checkpoint(args.candidate_a, map_location="cpu")
    checkpoint_b, metadata_b = read_checkpoint(args.candidate_b, map_location="cpu")
    del checkpoint_a, checkpoint_b
    if metadata_a.board != metadata_b.board:
        raise ValueError(
            f"candidate boards differ: {metadata_a.board} and {metadata_b.board}"
        )

    rows, columns = metadata_a.board
    if args.opening_random_moves >= rows * columns:
        raise ValueError("--opening-random-moves must be smaller than the board")

    device = _device(args.device)
    candidate_a = _load_agent(args.candidate_a, metadata_a, device, args.seed)
    candidate_b = _load_agent(args.candidate_b, metadata_b, device, args.seed + 1)
    game_seeds = random_game_seeds(args.games, args.seed)

    print(
        f"head-to-head on {rows}x{columns} via {args.server} using {device}; "
        f"games={args.games} paired-opening seed={args.seed} "
        f"opening-random-moves={args.opening_random_moves}",
        flush=True,
    )
    print(f"candidate-a={args.candidate_a} episode={metadata_a.episode}")
    print(f"candidate-b={args.candidate_b} episode={metadata_b.episode}", flush=True)
    if args.opening_random_moves == 0 and args.games > 2:
        print(
            "warning: greedy play without randomized openings repeats one game per "
            "seat assignment",
            flush=True,
        )

    started_at = time.monotonic()
    with GameEnvironment(
        target=args.server,
        rows=rows,
        columns=columns,
        max_turns=args.max_turns,
        rpc_timeout=args.rpc_timeout,
    ) as environment:
        result_a = evaluate_head_to_head(
            environment,
            candidate_a,
            candidate_b,
            game_seeds,
            args.opening_random_moves,
        )
    result_b = _opponent_result(result_a)
    elapsed = time.monotonic() - started_at

    _print_candidate("candidate-a", result_a)
    _print_candidate("candidate-b", result_b)
    print(f"elapsed={elapsed:.1f}s games/s={args.games / elapsed:.2f}", flush=True)
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (ConnectionError, FileNotFoundError, ValueError, grpc.RpcError) as exc:
        print(f"head-to-head evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
