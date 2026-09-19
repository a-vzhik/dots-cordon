"""Continuous search-assisted self-play, with DQN feature transfer or fresh weights."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
import time

import grpc
import numpy as np
import torch

from .audit.integration import (
    add_audit_arguments,
    checkpoint_record,
    close_audit,
    effective_config,
    ensure_experiment,
    open_audit,
    resolve_checkpoint_reference,
)
from .checkpoint import read_checkpoint
from .environment import GameEnvironment
from .evaluate import _device
from .head_to_head import _load_agent
from .search import MCTS, PolicyValueAgent, SearchReplay, collect_episode
from .search_evaluate import evaluate
from .train import TrainingState


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_audit_arguments(parser)
    parser.set_defaults(experiment="search-self-play")
    parser.add_argument("--server", default="127.0.0.1:50051")
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="total search-training episodes, including resumed episodes",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--channels", type=int, help="infer from source checkpoint, otherwise 64"
    )
    parser.add_argument(
        "--blocks", type=int, help="infer from source checkpoint, otherwise 3"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--initialize-from",
        type=Path,
        help="copy only a DQN's feature extractor; reset counters and optimizer",
    )
    source.add_argument(
        "--resume",
        type=Path,
        help="restore a search checkpoint, including replay and RNG",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=20000)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--updates-per-episode", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    parser.add_argument("--temperature-moves", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/search")
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument(
        "--eval-simulations",
        type=int,
        default=64,
        help="zero skips search-assisted evaluation",
    )
    parser.add_argument("--evaluation-seed", type=int, default=10007)
    parser.add_argument(
        "--eval-opponent",
        type=Path,
        help="optional frozen DQN baseline, e.g. champion:default",
    )
    parser.add_argument("--opening-random-moves", type=int, default=4)
    parser.add_argument("--rpc-timeout", type=float, default=10)
    args = parser.parse_args(arguments)
    for key in (
        "episodes",
        "batch_size",
        "replay_capacity",
        "updates_per_episode",
        "simulations",
        "log_every",
        "learning_rate",
        "c_puct",
        "dirichlet_alpha",
        "rpc_timeout",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in (
        "max_turns",
        "learning_starts",
        "temperature_moves",
        "checkpoint_every",
        "eval_every",
        "eval_simulations",
        "opening_random_moves",
        "seed",
        "evaluation_seed",
    ):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be nonnegative")
    if not 2 <= args.rows <= 255 or not 2 <= args.columns <= 255:
        parser.error("board dimensions must be between 2 and 255")
    if (
        args.channels is not None
        and args.channels < 4
        or args.blocks is not None
        and args.blocks < 0
    ):
        parser.error("channels must be >= 4; blocks must be >= 0")
    if not 0 <= args.noise_fraction <= 1:
        parser.error("noise fraction must be between zero and one")
    if args.replay_capacity < max(args.batch_size, args.learning_starts):
        parser.error("replay capacity must cover batch size and learning starts")
    if args.eval_every and (args.eval_games < 2 or args.eval_games % 2):
        parser.error("evaluation requires a positive even game count")
    if args.eval_opponent and args.opening_random_moves >= args.rows * args.columns:
        parser.error("opening must be shorter than the board")
    return args


def save_payload(path, agent, replay, random, state, args, initialization):
    payload = {
        "format_version": 2,
        "online": agent.online.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "training_state": asdict(state),
        "board": {"rows": args.rows, "columns": args.columns},
        "model": {
            "kind": "policy_value",
            "channels": args.channels,
            "blocks": args.blocks,
        },
        "game_config": {"max_turns": args.max_turns},
        "replay": replay.state_dict(),
        "rng_state": {
            "numpy": random.bit_generator.state,
            "torch": torch.get_rng_state(),
        },
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "database_url" and not k.startswith("_")
        },
        "initialization": initialization,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(args):
    audit = open_audit(args)
    attempt = None
    try:
        torch.manual_seed(args.seed)
        random = np.random.default_rng(args.seed)
        device = _device(args.device)
        payload, metadata, source_path = None, None, None
        reference = args.resume or args.initialize_from
        initialization = {"kind": "random"}
        if reference:
            source_path = resolve_checkpoint_reference(
                reference, audit, experiment_name=args.experiment
            )
            payload, metadata = read_checkpoint(source_path, map_location="cpu")
            if metadata.board != (args.rows, args.columns):
                raise ValueError(
                    "source checkpoint board does not match --rows/--columns"
                )
            expected_kind = "policy_value" if args.resume else "dqn"
            if metadata.kind != expected_kind:
                raise ValueError(
                    f"expected {expected_kind} checkpoint for this initialization mode"
                )
            initialization = (
                payload.get("initialization", {})
                if args.resume
                else {
                    "kind": "dqn_features",
                    "reference": str(reference),
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "source_episode": metadata.episode,
                }
            )
        args.channels = (
            args.channels
            if args.channels is not None
            else (metadata.channels if metadata else 64)
        )
        args.blocks = (
            args.blocks
            if args.blocks is not None
            else (metadata.blocks if metadata else 3)
        )
        if metadata and metadata.model != (args.channels, args.blocks):
            raise ValueError(
                "source checkpoint architecture does not match channels/blocks"
            )
        agent = PolicyValueAgent(device, args.channels, args.blocks, args.learning_rate)
        replay = SearchReplay(args.replay_capacity)
        state = TrainingState()
        if args.resume:
            if (
                payload.get("format_version") != 2
                or payload.get("game_config", {}).get("max_turns") != args.max_turns
            ):
                raise ValueError(
                    "resume requires a version-2 search checkpoint with matching max-turns"
                )
            agent.online.load_state_dict(payload["online"])
            agent.optimizer.load_state_dict(payload["optimizer"])
            for group in agent.optimizer.param_groups:
                group["lr"] = args.learning_rate
            replay.restore(payload["replay"])
            random.bit_generator.state = payload["rng_state"]["numpy"]
            torch.set_rng_state(payload["rng_state"]["torch"].cpu())
            state = TrainingState(**payload["training_state"])
        elif payload:
            agent.online.initialize_from_dqn(payload["online"])
        if state.episode > args.episodes:
            raise ValueError("--episodes must not precede resumed episode")
        del payload
        experiment, parent = None, None
        if audit:
            experiment = ensure_experiment(audit, args, args.rows, args.columns)
            if args.resume:
                parent = checkpoint_record(
                    audit, experiment["id"], args.resume, path=source_path
                )
            config = {
                **effective_config(args),
                "algorithm": "policy_value_puct_v1",
                "effective_device": str(device),
                "initialization": initialization,
                "replay_restored": bool(args.resume),
            }
            attempt = audit.create_attempt(
                experiment["id"],
                starting_checkpoint_id=parent["id"] if parent else None,
                config=config,
                target_episode=args.episodes,
            )
            print(
                f"audit experiment={experiment['name']} attempt={attempt['id']}",
                flush=True,
            )
        opponent, opponent_id = None, None
        if args.eval_opponent:
            path = resolve_checkpoint_reference(args.eval_opponent, audit)
            _, other_metadata = read_checkpoint(path, map_location="cpu")
            if other_metadata.kind != "dqn" or other_metadata.board != (
                args.rows,
                args.columns,
            ):
                raise ValueError(
                    "evaluation opponent must be a DQN with matching board"
                )
            opponent = _load_agent(path, other_metadata, device, args.evaluation_seed)
            if audit:
                opponent_id = audit.import_checkpoint(experiment["id"], path)["id"]
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stop_requested = False

        def request_stop(*_):
            nonlocal stop_requested
            stop_requested = True
            print("stopping after the current episode/evaluation", flush=True)

        previous = signal.signal(signal.SIGINT, request_stop)
        try:
            return _train(
                args,
                agent,
                replay,
                random,
                state,
                initialization,
                audit,
                experiment,
                attempt,
                parent,
                opponent,
                opponent_id,
                lambda: stop_requested,
            )
        finally:
            signal.signal(signal.SIGINT, previous)
    except BaseException as exc:
        if audit and attempt:
            audit.update_attempt(
                attempt["id"],
                status="interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed",
                error=str(exc) or type(exc).__name__,
            )
        raise
    finally:
        close_audit(audit)


def _train(
    args,
    agent,
    replay,
    random,
    state,
    initialization,
    audit,
    experiment,
    attempt,
    parent,
    opponent,
    opponent_id,
    stopped,
):
    saved_state, checkpoint_id = None, parent["id"] if parent else None
    staging = args.checkpoint_dir / ".search-checkpoint.pt"
    started, starting_episode = time.monotonic(), state.episode
    recent, losses = [], {}

    def save(**flags):
        nonlocal saved_state, checkpoint_id
        if saved_state != state:
            save_payload(staging, agent, replay, random, state, args, initialization)
            if audit:
                record = audit.import_checkpoint(
                    experiment["id"], staging, attempt_id=attempt["id"], **flags
                )
                checkpoint_id = record["id"]
            saved_state = state
        elif audit:
            audit.import_checkpoint(
                experiment["id"],
                staging,
                attempt_id=attempt["id"],
                checkpoint_id=checkpoint_id,
                **flags,
            )
        for filename in (f"search-{state.episode:07d}.pt", "search-latest.pt"):
            target = args.checkpoint_dir / filename
            temporary = target.with_suffix(".pt.tmp")
            shutil.copyfile(staging, temporary)
            temporary.replace(target)
        return checkpoint_id

    def record_metrics():
        elapsed = time.monotonic() - started
        metrics = {
            **losses,
            "episode_start": state.episode - len(recent) + 1,
            "elapsed_seconds": elapsed,
            "games_per_second": (state.episode - starting_episode) / max(elapsed, 1e-9),
            "replay_size": len(replay),
            "environment_steps": state.environment_steps,
            "optimization_steps": state.optimization_steps,
            "simulations": args.simulations,
            "draw_fraction": float(
                np.mean([x["scores"][0] == x["scores"][1] for x in recent])
            ),
            "mean_total_score": float(np.mean([sum(x["scores"]) for x in recent])),
            "opponents": {"self_play": len(recent), "random": 0, "frozen": 0},
            "episodes": list(recent),
        }
        if audit:
            audit.record_metrics(attempt["id"], state.episode, metrics)
        print(
            f"episode={state.episode} replay={len(replay)} updates={state.optimization_steps} "
            f"loss={losses.get('loss', float('nan')):.4f} draws={metrics['draw_fraction']:.2f} "
            f"mean_points={metrics['mean_total_score']:.2f} games/s={metrics['games_per_second']:.3f}",
            flush=True,
        )
        recent.clear()

    with GameEnvironment(
        args.server, args.rows, args.columns, args.max_turns, args.rpc_timeout
    ) as environment:
        # Fail early if the server predates SimulateMove; no real move is made.
        environment.simulate(environment.game, 0)
        search = MCTS(
            agent.predict,
            environment.simulate,
            args.simulations,
            args.c_puct,
            args.dirichlet_alpha,
            args.noise_fraction,
        )
        save()

        def evaluate_current():
            cp = save()
            evaluate(
                agent,
                environment,
                games=args.eval_games,
                seed=args.evaluation_seed,
                simulations=args.eval_simulations,
                c_puct=args.c_puct,
                opponent=opponent,
                opening_random_moves=args.opening_random_moves,
                audit=audit,
                experiment_id=experiment["id"] if experiment else None,
                checkpoint_id=cp,
                opponent_checkpoint_id=opponent_id,
                attempt_id=attempt["id"] if attempt else None,
            )

        print(
            f"search self-play on {args.rows}x{args.columns}, device={agent.device}, "
            f"simulations={args.simulations}, initialization={initialization['kind']}",
            flush=True,
        )
        if args.eval_every:
            evaluate_current()
        while state.episode < args.episodes and not stopped():
            samples, scores, moves = collect_episode(
                environment, search, random, args.temperature_moves
            )
            replay.items.extend(samples)
            updates = state.optimization_steps
            if len(replay) >= max(args.learning_starts, args.batch_size):
                batch_losses = [
                    agent.optimize(replay, args.batch_size, random)
                    for _ in range(args.updates_per_episode)
                ]
                losses = {
                    key: float(np.mean([x[key] for x in batch_losses]))
                    for key in batch_losses[0]
                }
                updates += args.updates_per_episode
            state = TrainingState(
                state.episode + 1, state.environment_steps + len(moves), updates
            )
            recent.append(
                {
                    "scores": list(scores),
                    "opponent": "self-play",
                    "learner_player": None,
                }
            )
            with (args.checkpoint_dir / "games.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "attempt_id": attempt["id"] if attempt else None,
                            "episode": state.episode,
                            "rows": args.rows,
                            "columns": args.columns,
                            "max_turns": args.max_turns,
                            "moves": moves,
                            "scores": scores,
                        }
                    )
                    + "\n"
                )
            if state.episode % args.log_every == 0:
                record_metrics()
            if args.checkpoint_every and state.episode % args.checkpoint_every == 0:
                save(is_periodic_save=True)
            if (
                args.eval_every
                and state.episode % args.eval_every == 0
                and not stopped()
            ):
                evaluate_current()
        if recent:
            record_metrics()
        save(is_final_in_attempt=True)
    if audit:
        audit.update_attempt(
            attempt["id"],
            status="interrupted" if stopped() else "completed",
            phase="finished",
            outcome=None if stopped() else "trained_only",
            latest_episode=state.episode,
        )
    print(f"saved {args.checkpoint_dir / 'search-latest.pt'}", flush=True)
    return 130 if stopped() else 0


def main():
    try:
        raise SystemExit(run(parse_args()))
    except (ConnectionError, FileNotFoundError, ValueError, grpc.RpcError) as exc:
        print(f"search training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
