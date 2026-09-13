"""Evaluate one or more checkpoints against identical random-opponent suites."""

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
from .self_play import MatchStats, evaluate_against_random, random_game_seeds


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--server", default="127.0.0.1:50051")
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=10_007)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="zero lets the server play until the board is full",
    )
    parser.add_argument("--rpc-timeout", type=float, default=10.0)
    args = parser.parse_args(arguments)
    if args.games <= 0:
        parser.error("--games must be positive")
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


def _metadata(checkpoints: list[Path]) -> list[CheckpointMetadata]:
    metadata: list[CheckpointMetadata] = []
    for path in checkpoints:
        checkpoint, item = read_checkpoint(path, map_location="cpu")
        metadata.append(item)
        del checkpoint

    first_board = metadata[0].board
    for path, item in zip(checkpoints[1:], metadata[1:], strict=True):
        if item.board != first_board:
            raise ValueError(
                f"checkpoint {path} uses board {item.board}; expected {first_board}"
            )
    return metadata


def run(args: argparse.Namespace) -> int:
    device = _device(args.device)
    metadata = _metadata(args.checkpoints)
    rows, columns = metadata[0].board
    game_seeds = random_game_seeds(args.games, args.seed)

    print(
        f"evaluating {len(args.checkpoints)} checkpoint(s) on {rows}x{columns} "
        f"against {args.games} random games via {args.server} using {device}; "
        f"suite seed={args.seed}",
        flush=True,
    )
    with GameEnvironment(
        target=args.server,
        rows=rows,
        columns=columns,
        max_turns=args.max_turns,
        rpc_timeout=args.rpc_timeout,
    ) as environment:
        for path, item in zip(args.checkpoints, metadata, strict=True):
            agent = DQNAgent(
                device=device,
                learning_rate=3e-4,
                gamma=0.99,
                seed=args.seed,
                channels=item.channels,
                blocks=item.blocks,
            )
            checkpoint, _ = read_checkpoint(path, map_location=device)
            restore_agent(agent, checkpoint, restore_optimizer=False)

            started_at = time.monotonic()
            result = evaluate_against_random(environment, agent, game_seeds)
            elapsed = time.monotonic() - started_at
            print(f"checkpoint={path} episode={item.episode}")
            print(f"  overall     {_format_stats(result.overall)}")
            print(f"  as-player-0 {_format_stats(result.as_player_0)}")
            print(f"  as-player-1 {_format_stats(result.as_player_1)}")
            print(f"  elapsed={elapsed:.1f}s games/s={args.games / elapsed:.2f}", flush=True)

    return 0


def _format_stats(stats: MatchStats) -> str:
    return (
        f"games={stats.games} "
        f"W/D/L={stats.wins}/{stats.draws}/{stats.losses} "
        f"match_score={stats.match_score:.3f} "
        f"mean_score_diff={stats.mean_score_difference:+.3f}"
    )


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (ConnectionError, FileNotFoundError, ValueError, grpc.RpcError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
