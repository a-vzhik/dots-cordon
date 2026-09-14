"""Compare two DQN checkpoints directly with paired randomized openings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import grpc
import torch

from .audit.integration import (
    add_audit_arguments,
    checkpoint_record,
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
    add_audit_arguments(parser)
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
        score_difference_sum=(
            -stats.score_difference_sum
            if stats.score_difference_sum is not None
            else None
        ),
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
    audit = open_audit(args)
    batch_id: str | None = None
    try:
        path_a = resolve_checkpoint_reference(
            args.candidate_a, audit, experiment_name=args.experiment
        )
        path_b = resolve_checkpoint_reference(
            args.candidate_b, audit, experiment_name=args.experiment
        )
        checkpoint_a, metadata_a = read_checkpoint(path_a, map_location="cpu")
        checkpoint_b, metadata_b = read_checkpoint(path_b, map_location="cpu")
        del checkpoint_a, checkpoint_b
        if metadata_a.board != metadata_b.board:
            raise ValueError(
                f"candidate boards differ: {metadata_a.board} and {metadata_b.board}"
            )

        rows, columns = metadata_a.board
        if args.opening_random_moves >= rows * columns:
            raise ValueError("--opening-random-moves must be smaller than the board")

        if audit is not None:
            experiment = ensure_experiment(audit, args, rows, columns)
            record_a = checkpoint_record(
                audit, experiment["id"], args.candidate_a, path=path_a
            )
            record_b = checkpoint_record(
                audit, experiment["id"], args.candidate_b, path=path_b
            )
            path_a = resolve_checkpoint_reference(f"checkpoint:{record_a['id']}", audit)
            path_b = resolve_checkpoint_reference(f"checkpoint:{record_b['id']}", audit)
            checkpoint_a, metadata_a = read_checkpoint(path_a, map_location="cpu")
            checkpoint_b, metadata_b = read_checkpoint(path_b, map_location="cpu")
            del checkpoint_a, checkpoint_b
            definition = evaluation_definition(
                args,
                rows=rows,
                columns=columns,
                seed=args.seed,
                games=args.games,
                kind="head_to_head",
                opening_random_moves=args.opening_random_moves,
            )
            batch = audit.create_evaluation(
                experiment["id"],
                record_a["id"],
                purpose="standalone_head_to_head",
                suite_definitions=[definition],
                opponent_checkpoint_id=record_b["id"],
                config=effective_config(args),
            )
            batch_id = batch["id"]
        result = _run_match(args, path_a, path_b, metadata_a, metadata_b)
        if audit is not None and batch_id is not None:
            audit.complete_suite(batch_id, 0, evaluation_result_dict(result))
            batch_id = None
        return 0
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        if audit is not None and batch_id is not None:
            try:
                audit.fail_evaluation(
                    batch_id,
                    str(exc) or type(exc).__name__,
                    status=status,
                )
            except Exception as recording_error:
                exc.add_note(f"could not record evaluation failure: {recording_error}")
        raise
    finally:
        close_audit(audit)


def _run_match(
    args: argparse.Namespace,
    path_a: Path,
    path_b: Path,
    metadata_a: CheckpointMetadata,
    metadata_b: CheckpointMetadata,
) -> EvaluationResult:
    rows, columns = metadata_a.board
    device = _device(args.device)
    candidate_a = _load_agent(path_a, metadata_a, device, args.seed)
    candidate_b = _load_agent(path_b, metadata_b, device, args.seed + 1)
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
    return result_a


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (ConnectionError, FileNotFoundError, ValueError, grpc.RpcError) as exc:
        print(f"head-to-head evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
