"""Command-line entry point for server-backed DQN self-play."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import signal
import sys
import time
from typing import Any

import grpc
import numpy as np
import torch

from .checkpoint import read_checkpoint, restore_agent
from .dqn import DQNAgent, ReplayBuffer
from .environment import GameEnvironment
from .self_play import (
    EpisodeResult,
    EvaluationResult,
    MatchStats,
    collect_against_random_episode,
    collect_self_play_episode,
    evaluate_against_random,
    random_game_seeds,
)


@dataclass(frozen=True, slots=True)
class TrainingState:
    episode: int = 0
    environment_steps: int = 0
    optimization_steps: int = 0


@dataclass(slots=True)
class EvaluationMonitor:
    """Track the strongest policy and lack of meaningful improvement."""

    patience: int
    min_delta: float
    best_score: float | None = None
    best_mean_score_difference: float | None = None
    best_episode: int | None = None
    reference_score: float | None = None
    evaluations_without_improvement: int = 0

    def observe(self, episode: int, evaluation: EvaluationResult) -> bool:
        """Record an evaluation and return whether it is the new best."""
        score = evaluation.overall.match_score
        mean_difference = evaluation.overall.mean_score_difference
        best_rank = (
            float("-inf") if self.best_score is None else self.best_score,
            float("-inf")
            if self.best_mean_score_difference is None
            else self.best_mean_score_difference,
        )
        is_best = (score, mean_difference) > best_rank
        if is_best:
            self.best_score = score
            self.best_mean_score_difference = mean_difference
            self.best_episode = episode

        if self.reference_score is None:
            meaningful_improvement = True
        elif self.min_delta == 0:
            meaningful_improvement = score > self.reference_score
        else:
            meaningful_improvement = score >= self.reference_score + self.min_delta

        if meaningful_improvement:
            self.reference_score = score
            self.evaluations_without_improvement = 0
        else:
            self.evaluations_without_improvement += 1
        return is_best

    @property
    def should_stop(self) -> bool:
        return (
            self.patience > 0
            and self.evaluations_without_improvement >= self.patience
        )


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
    parser.add_argument(
        "--random-opponent-probability",
        type=float,
        default=0.0,
        help="fraction of training episodes played against a random opponent",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=10_007,
        help="seed for the fixed random-opponent evaluation suite",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            "stop after this many evaluations without a meaningful improvement; "
            "zero disables early stopping"
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.005,
        help="match-score increase required to reset early-stopping patience",
    )
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
    if args.early_stop_patience < 0 or args.early_stop_min_delta < 0:
        parser.error("early-stopping patience and minimum delta must be non-negative")
    if args.eval_every and not args.eval_games:
        parser.error("--eval-games must be positive when evaluation is enabled")
    if args.early_stop_patience and not args.eval_every:
        parser.error("--early-stop-patience requires evaluation to be enabled")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be between zero and one")
    if not 0 <= args.random_opponent_probability <= 1:
        parser.error("--random-opponent-probability must be between zero and one")


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
    rng_state: dict[str, Any],
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
            "rng_state": rng_state,
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    agent: DQNAgent,
    args: argparse.Namespace,
) -> tuple[TrainingState, dict[str, Any] | None]:
    checkpoint, metadata = read_checkpoint(path, map_location=agent.device)
    expected_model = (args.channels, args.blocks)
    if metadata.model != expected_model:
        raise ValueError(
            f"checkpoint model is {metadata.model}, expected {expected_model}"
        )
    expected_board = (args.rows, args.columns)
    if metadata.board != expected_board:
        raise ValueError(
            f"checkpoint board is {metadata.board}, expected {expected_board}"
        )
    restore_agent(agent, checkpoint, restore_optimizer=True)
    rng_state = checkpoint.get("rng_state")
    if rng_state is not None and not isinstance(rng_state, dict):
        raise ValueError("checkpoint contains invalid random state")
    return (
        TrainingState(
            episode=metadata.episode,
            environment_steps=metadata.environment_steps,
            optimization_steps=metadata.optimization_steps,
        ),
        rng_state,
    )


def _capture_rng_state(
    agent: DQNAgent,
    replay: ReplayBuffer,
    opponent_selection_random: np.random.Generator,
    training_opponent_random: np.random.Generator,
    random_opponent_episodes: int,
) -> dict[str, Any]:
    return {
        "agent": agent.random.bit_generator.state,
        "replay": replay.random_state(),
        "opponent_selection": opponent_selection_random.bit_generator.state,
        "training_opponent": training_opponent_random.bit_generator.state,
        "random_opponent_episodes": random_opponent_episodes,
    }


def _restore_rng_state(
    rng_state: dict[str, Any],
    agent: DQNAgent,
    replay: ReplayBuffer,
    opponent_selection_random: np.random.Generator,
    training_opponent_random: np.random.Generator,
) -> int:
    try:
        agent.random.bit_generator.state = rng_state["agent"]
        replay.restore_random_state(rng_state["replay"])
        opponent_selection_random.bit_generator.state = rng_state[
            "opponent_selection"
        ]
        training_opponent_random.bit_generator.state = rng_state["training_opponent"]
        return int(rng_state["random_opponent_episodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint contains invalid random state") from exc


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
    opponent_selection_random = np.random.default_rng(args.seed + 1)
    training_opponent_random = np.random.default_rng(args.seed + 2)
    random_opponent_episodes = 0
    state = TrainingState()
    if args.resume is not None:
        state, rng_state = _load_checkpoint(args.resume, agent, args)
        if rng_state is not None:
            random_opponent_episodes = _restore_rng_state(
                rng_state,
                agent,
                replay,
                opponent_selection_random,
                training_opponent_random,
            )
    starting_episode = state.episode

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    recent: deque[EpisodeResult] = deque(maxlen=args.log_every)
    recent_losses: deque[float] = deque(maxlen=max(args.log_every * 50, 1))
    evaluation_seeds = (
        random_game_seeds(args.eval_games, args.evaluation_seed)
        if args.eval_every
        else ()
    )
    stop_requested = False
    monitor = EvaluationMonitor(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
    )

    def request_stop(_signal: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    started_at = time.monotonic()
    print(
        f"training on {args.rows}x{args.columns} via {args.server} "
        f"using {device} (starting episode {state.episode + 1}, "
        f"random-opponent probability {args.random_opponent_probability:.2f})",
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
            def save_checkpoint(path: Path) -> None:
                _save_checkpoint(
                    path,
                    agent,
                    state,
                    args,
                    _capture_rng_state(
                        agent,
                        replay,
                        opponent_selection_random,
                        training_opponent_random,
                        random_opponent_episodes,
                    ),
                )

            def evaluate_and_track() -> bool:
                evaluation = evaluate_against_random(
                    environment, agent, evaluation_seeds
                )
                print(
                    f"evaluation episode={state.episode} "
                    f"W/D/L={evaluation.wins}/{evaluation.draws}/{evaluation.losses} "
                    f"match_score={evaluation.overall.match_score:.4f} "
                    f"mean_score_diff={evaluation.mean_score_difference:+.3f} "
                    f"P0={_format_stats(evaluation.as_player_0)} "
                    f"P1={_format_stats(evaluation.as_player_1)}",
                    flush=True,
                )
                if monitor.observe(state.episode, evaluation):
                    best_path = args.checkpoint_dir / "dqn-best.pt"
                    save_checkpoint(best_path)
                    print(
                        f"new best episode={state.episode} "
                        f"match_score={evaluation.overall.match_score:.4f} "
                        f"saved {best_path}",
                        flush=True,
                    )
                return monitor.should_stop

            if args.eval_every:
                evaluate_and_track()

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

                if (
                    opponent_selection_random.random()
                    < args.random_opponent_probability
                ):
                    learner_player = random_opponent_episodes % 2
                    result = collect_against_random_episode(
                        environment,
                        agent,
                        replay,
                        epsilon=epsilon,
                        random=training_opponent_random,
                        learner_player=learner_player,
                        terminal_win_bonus=args.terminal_win_bonus,
                        on_transition=optimize,
                    )
                    random_opponent_episodes += 1
                else:
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
                    self_play_results = [
                        item for item in recent if item.opponent == "self-play"
                    ]
                    random_results = [
                        item for item in recent if item.opponent == "random"
                    ]
                    mean_loss = (
                        float(np.mean(recent_losses))
                        if recent_losses
                        else float("nan")
                    )
                    completed_this_run = state.episode - starting_episode
                    games_per_second = completed_this_run / max(
                        time.monotonic() - started_at, 1e-9
                    )
                    print(
                        f"episode={state.episode} steps={state.environment_steps} "
                        f"epsilon={epsilon:.3f} replay={len(replay)} "
                        f"opponents=self:{len(self_play_results)}/random:{len(random_results)} "
                        f"{_score_summary(self_play_results, random_results)} "
                        f"loss={mean_loss:.5f} games/s={games_per_second:.2f}",
                        flush=True,
                    )

                early_stop = False
                if args.eval_every and state.episode % args.eval_every == 0:
                    early_stop = evaluate_and_track()

                if args.checkpoint_every and state.episode % args.checkpoint_every == 0:
                    save_checkpoint(
                        args.checkpoint_dir / f"dqn-{state.episode:07d}.pt"
                    )

                if early_stop:
                    print(
                        f"early stopping episode={state.episode}: no match-score "
                        f"improvement of at least {monitor.min_delta:.4f} for "
                        f"{monitor.evaluations_without_improvement} evaluations; "
                        f"best episode={monitor.best_episode} "
                        f"match_score={monitor.best_score:.4f}",
                        flush=True,
                    )
                    break

            final_path = args.checkpoint_dir / "dqn-latest.pt"
            save_checkpoint(final_path)
            print(f"saved {final_path}", flush=True)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    return 0


def _format_stats(stats: MatchStats) -> str:
    return (
        f"{stats.wins}/{stats.draws}/{stats.losses}"
        f"({stats.mean_score_difference:+.3f})"
    )


def _score_summary(
    self_play_results: list[EpisodeResult],
    random_results: list[EpisodeResult],
) -> str:
    parts: list[str] = []
    if self_play_results:
        differences = [item.scores[0] - item.scores[1] for item in self_play_results]
        parts.append(f"self_p0_diff={np.mean(differences):+.3f}")
    if random_results:
        differences = []
        for item in random_results:
            assert item.learner_player is not None
            player = item.learner_player
            differences.append(item.scores[player] - item.scores[1 - player])
        parts.append(f"random_learner_diff={np.mean(differences):+.3f}")
    return " ".join(parts)


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (ConnectionError, FileNotFoundError, ValueError, grpc.RpcError) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
