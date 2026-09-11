"""Command-line entry point for server-backed DQN self-play."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import signal
import sys
import time

import grpc
import numpy as np
import torch

from .dqn import DQNAgent, ReplayBuffer
from .environment import GameEnvironment
from .self_play import EpisodeResult, collect_self_play_episode, evaluate_against_random


@dataclass(frozen=True, slots=True)
class TrainingState:
    episode: int = 0
    environment_steps: int = 0
    optimization_steps: int = 0


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="127.0.0.1:50051")
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="zero lets the server play until the board is full",
    )
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--updates-per-transition", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    parser.add_argument("--terminal-win-bonus", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--rpc-timeout", type=float, default=10.0)
    args = parser.parse_args(arguments)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (
        "rows",
        "columns",
        "episodes",
        "channels",
        "batch_size",
        "replay_capacity",
        "target_update",
        "epsilon_decay_steps",
        "log_every",
        "rpc_timeout",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_turns < 0 or args.learning_starts < 0:
        parser.error("--max-turns and --learning-starts must be non-negative")
    if args.blocks < 0 or args.updates_per_transition < 0:
        parser.error("--blocks and --updates-per-transition must be non-negative")
    if args.eval_every < 0 or args.eval_games < 0 or args.checkpoint_every < 0:
        parser.error("evaluation and checkpoint intervals must be non-negative")
    if args.eval_every and not args.eval_games:
        parser.error("--eval-games must be positive when evaluation is enabled")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be between zero and one")


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _epsilon(args: argparse.Namespace, environment_steps: int) -> float:
    fraction = min(environment_steps / args.epsilon_decay_steps, 1.0)
    return args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start)


def _save_checkpoint(
    path: Path,
    agent: DQNAgent,
    state: TrainingState,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "online": agent.online.state_dict(),
            "target": agent.target.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "training_state": asdict(state),
            "board": {"rows": args.rows, "columns": args.columns},
            "model": {"channels": args.channels, "blocks": args.blocks},
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(path: Path, agent: DQNAgent, args: argparse.Namespace) -> TrainingState:
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    expected_model = {"channels": args.channels, "blocks": args.blocks}
    if checkpoint.get("model") != expected_model:
        raise ValueError(
            f"checkpoint model is {checkpoint.get('model')}, expected {expected_model}"
        )
    expected_board = {"rows": args.rows, "columns": args.columns}
    if checkpoint.get("board") != expected_board:
        raise ValueError(
            f"checkpoint board is {checkpoint.get('board')}, expected {expected_board}"
        )
    agent.online.load_state_dict(checkpoint["online"])
    agent.target.load_state_dict(checkpoint["target"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    return TrainingState(**checkpoint["training_state"])


def run(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    agent = DQNAgent(
        device=device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
        channels=args.channels,
        blocks=args.blocks,
    )
    replay = ReplayBuffer(args.replay_capacity, seed=args.seed)
    state = TrainingState()
    if args.resume is not None:
        state = _load_checkpoint(args.resume, agent, args)
    starting_episode = state.episode

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    recent: deque[EpisodeResult] = deque(maxlen=args.log_every)
    recent_losses: deque[float] = deque(maxlen=max(args.log_every * 50, 1))
    evaluation_random = np.random.default_rng(args.seed + 1)
    stop_requested = False

    def request_stop(_signal: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    started_at = time.monotonic()
    print(
        f"training on {args.rows}x{args.columns} via {args.server} "
        f"using {device} (starting episode {state.episode + 1})",
        flush=True,
    )

    try:
        with GameEnvironment(
            target=args.server,
            rows=args.rows,
            columns=args.columns,
            max_turns=args.max_turns,
            rpc_timeout=args.rpc_timeout,
        ) as environment:
            while state.episode < args.episodes and not stop_requested:
                epsilon = _epsilon(args, state.environment_steps)

                def optimize() -> None:
                    nonlocal state
                    if len(replay) < max(args.learning_starts, args.batch_size):
                        return
                    for _ in range(args.updates_per_transition):
                        loss = agent.optimize(replay, args.batch_size)
                        recent_losses.append(loss)
                        state = TrainingState(
                            episode=state.episode,
                            environment_steps=state.environment_steps,
                            optimization_steps=state.optimization_steps + 1,
                        )
                        if state.optimization_steps % args.target_update == 0:
                            agent.sync_target()

                result = collect_self_play_episode(
                    environment,
                    agent,
                    replay,
                    epsilon=epsilon,
                    terminal_win_bonus=args.terminal_win_bonus,
                    on_transition=optimize,
                )
                state = TrainingState(
                    episode=state.episode + 1,
                    environment_steps=state.environment_steps + result.moves,
                    optimization_steps=state.optimization_steps,
                )
                recent.append(result)

                if state.episode % args.log_every == 0:
                    score_differences = [item.scores[0] - item.scores[1] for item in recent]
                    mean_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
                    completed_this_run = state.episode - starting_episode
                    games_per_second = completed_this_run / max(
                        time.monotonic() - started_at, 1e-9
                    )
                    print(
                        f"episode={state.episode} steps={state.environment_steps} "
                        f"epsilon={epsilon:.3f} replay={len(replay)} "
                        f"mean_score_diff={np.mean(score_differences):+.3f} "
                        f"loss={mean_loss:.5f} games/s={games_per_second:.2f}",
                        flush=True,
                    )

                if args.eval_every and state.episode % args.eval_every == 0:
                    evaluation = evaluate_against_random(
                        environment, agent, args.eval_games, evaluation_random
                    )
                    print(
                        f"evaluation episode={state.episode} "
                        f"W/D/L={evaluation.wins}/{evaluation.draws}/{evaluation.losses} "
                        f"mean_score_diff={evaluation.mean_score_difference:+.3f}",
                        flush=True,
                    )

                if args.checkpoint_every and state.episode % args.checkpoint_every == 0:
                    _save_checkpoint(
                        args.checkpoint_dir / f"dqn-{state.episode:07d}.pt",
                        agent,
                        state,
                        args,
                    )

            final_path = args.checkpoint_dir / "dqn-latest.pt"
            _save_checkpoint(final_path, agent, state, args)
            print(f"saved {final_path}", flush=True)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (ConnectionError, grpc.RpcError) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
