"""Evaluate policy/value checkpoints, with and without search, against fixed opponents."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import grpc

from .audit.integration import (
    add_audit_arguments,
    close_audit,
    ensure_experiment,
    evaluation_definition,
    evaluation_result_dict,
    open_audit,
    resolve_checkpoint_reference,
)
from .checkpoint import read_checkpoint
from .environment import GameEnvironment
from .evaluate import _device, _format_stats
from .head_to_head import _load_agent
from .search import PolicyValueAgent, SearchPlayer
from .self_play import evaluate_against_random, evaluate_head_to_head, random_game_seeds


def evaluate(
    agent,
    environment,
    *,
    games,
    seed,
    simulations,
    c_puct=1.5,
    opponent=None,
    opening_random_moves=4,
    audit=None,
    experiment_id=None,
    checkpoint_id=None,
    opponent_checkpoint_id=None,
    attempt_id=None,
    purpose="training",
):
    """Keep policy-only and search-assisted results as separate, fully described suites."""
    results = {}
    for mode in ["policy", "search"] if simulations else ["policy"]:
        player = (
            agent
            if mode == "policy"
            else SearchPlayer(agent, environment, simulations, seed, c_puct)
        )
        for opponent_name in (
            ["random", "baseline"] if opponent is not None else ["random"]
        ):
            batch = None
            config = {
                "mode": mode,
                "simulations": simulations if mode == "search" else 0,
                "c_puct": c_puct,
                "root_noise": False,
                "temperature": 0,
            }
            if audit is not None:
                args = argparse.Namespace(max_turns=environment.max_turns)
                definition = evaluation_definition(
                    args,
                    rows=environment.rows,
                    columns=environment.columns,
                    seed=seed,
                    games=games,
                    kind="random" if opponent_name == "random" else "head_to_head",
                    opening_random_moves=0
                    if opponent_name == "random"
                    else opening_random_moves,
                )
                definition["evaluator_version"] = "policy-value-puct-v1"
                definition["subject_policy"] = config
                definition["opponent_policy"] = (
                    "uniform_random" if opponent_name == "random" else "greedy_dqn"
                )
                batch = audit.create_evaluation(
                    experiment_id,
                    checkpoint_id,
                    purpose=purpose,
                    attempt_id=attempt_id,
                    opponent_checkpoint_id=opponent_checkpoint_id
                    if opponent_name == "baseline"
                    else None,
                    suite_definitions=[definition],
                    config=config,
                )
            started = time.monotonic()
            try:
                seeds = random_game_seeds(games, seed)
                if opponent_name == "random":
                    result = evaluate_against_random(environment, player, seeds)
                else:
                    result = evaluate_head_to_head(
                        environment, player, opponent, seeds, opening_random_moves
                    )
                if batch is not None:
                    audit.complete_suite(batch["id"], 0, evaluation_result_dict(result))
            except BaseException as exc:
                if batch is not None:
                    audit.fail_evaluation(
                        batch["id"],
                        str(exc) or type(exc).__name__,
                        status="interrupted"
                        if isinstance(exc, KeyboardInterrupt)
                        else "failed",
                    )
                raise
            print(
                f"mode={mode} opponent={opponent_name} {_format_stats(result.overall)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
            print(f"  seat0 {_format_stats(result.as_player_0)}", flush=True)
            print(f"  seat1 {_format_stats(result.as_player_1)}", flush=True)
            results[f"{mode}_{opponent_name}"] = result
    return results


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_audit_arguments(parser)
    parser.set_defaults(experiment="search-self-play")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--opponent",
        type=Path,
        help="optional frozen DQN baseline, e.g. champion:default",
    )
    parser.add_argument("--server", default="127.0.0.1:50051")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--simulations",
        type=int,
        default=64,
        help="zero evaluates the neural policy only",
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--opening-random-moves", type=int, default=4)
    parser.add_argument("--rpc-timeout", type=float, default=10)
    args = parser.parse_args(arguments)
    if (
        args.games < 2
        or args.games % 2
        or args.simulations < 0
        or args.c_puct <= 0
        or args.max_turns < 0
        or args.opening_random_moves < 0
        or args.rpc_timeout <= 0
    ):
        parser.error(
            "use positive even games, nonnegative simulations/turns/opening, and positive c-puct/timeout"
        )
    return args


def run(args):
    audit = open_audit(args)
    try:
        path = resolve_checkpoint_reference(
            args.checkpoint, audit, experiment_name=args.experiment
        )
        payload, metadata = read_checkpoint(path, map_location="cpu")
        if metadata.kind != "policy_value":
            raise ValueError("expected a policy/value checkpoint")
        if payload.get("game_config", {}).get("max_turns", 0) != args.max_turns:
            raise ValueError("--max-turns must match the checkpoint's training rules")
        if (
            args.opponent
            and args.opening_random_moves >= metadata.rows * metadata.columns
        ):
            raise ValueError("opening must be shorter than the board")
        device = _device(args.device)
        agent = PolicyValueAgent(device, metadata.channels, metadata.blocks)
        agent.online.load_state_dict(payload["online"])
        opponent, opponent_id = None, None
        experiment = ensure_experiment(audit, args, *metadata.board) if audit else None
        subject = audit.import_checkpoint(experiment["id"], path) if audit else None
        if args.opponent:
            other_path = resolve_checkpoint_reference(args.opponent, audit)
            other_payload, other_metadata = read_checkpoint(
                other_path, map_location="cpu"
            )
            if other_metadata.board != metadata.board or other_metadata.kind != "dqn":
                raise ValueError(
                    "baseline must be a DQN with matching board dimensions"
                )
            opponent = _load_agent(other_path, other_metadata, device, args.seed)
            if audit:
                opponent_id = audit.import_checkpoint(experiment["id"], other_path)[
                    "id"
                ]
        with GameEnvironment(
            args.server, *metadata.board, args.max_turns, args.rpc_timeout
        ) as environment:
            evaluate(
                agent,
                environment,
                games=args.games,
                seed=args.seed,
                simulations=args.simulations,
                c_puct=args.c_puct,
                opponent=opponent,
                opening_random_moves=args.opening_random_moves,
                audit=audit,
                experiment_id=experiment["id"] if experiment else None,
                checkpoint_id=subject["id"] if subject else None,
                opponent_checkpoint_id=opponent_id,
                purpose="standalone_search",
            )
        return 0
    finally:
        close_audit(audit)


def main():
    try:
        raise SystemExit(run(parse_args()))
    except (ConnectionError, FileNotFoundError, ValueError, grpc.RpcError) as exc:
        print(f"search evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
