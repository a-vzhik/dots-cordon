"""Command-line entry point for server-backed DQN self-play."""

from __future__ import annotations

import argparse
import copy
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
from .checkpoint import read_checkpoint, restore_agent
from .dqn import DQNAgent, ReplayBuffer
from .environment import GameEnvironment
from .self_play import (
    EpisodeResult,
    EvaluationResult,
    MatchStats,
    collect_against_agent_episode,
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
            self.patience > 0 and self.evaluations_without_improvement >= self.patience
        )


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_audit_arguments(parser)
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
    parser.add_argument(
        "--frozen-opponent",
        type=Path,
        help=(
            "checkpoint used as a greedy frozen opponent for non-random episodes; "
            "without it those episodes use self-play"
        ),
    )
    parser.add_argument(
        "--frozen-opening-random-moves",
        type=int,
        default=0,
        help=(
            "uniform-random opening moves before each frozen-opponent episode; "
            "the opening does not enter replay"
        ),
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
    parser.add_argument(
        "--checkpoint-relative-to-start",
        action="store_true",
        help="measure checkpoint intervals from the resumed episode",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-rng-on-resume",
        action="store_true",
        help=(
            "load weights, optimizer, and counters from --resume but initialize "
            "training random streams from --seed"
        ),
    )
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
    if (
        args.max_turns < 0
        or args.learning_starts < 0
        or args.frozen_opening_random_moves < 0
    ):
        parser.error(
            "--max-turns, --learning-starts, and --frozen-opening-random-moves "
            "must be non-negative"
        )
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
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.terminal_win_bonus < 0:
        parser.error("--terminal-win-bonus must be non-negative")
    if not 0 <= args.random_opponent_probability <= 1:
        parser.error("--random-opponent-probability must be between zero and one")
    if args.frozen_opening_random_moves >= args.rows * args.columns:
        parser.error("--frozen-opening-random-moves must be smaller than the board")


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
    # Preserve Adam's accumulated state while honoring the learning rate chosen
    # for this run. load_state_dict also restores the checkpoint's old rate.
    for parameter_group in agent.optimizer.param_groups:
        parameter_group["lr"] = args.learning_rate
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
    frozen_opponent_episodes: int = 0,
) -> dict[str, Any]:
    return {
        "agent": agent.random.bit_generator.state,
        "replay": replay.random_state(),
        "opponent_selection": opponent_selection_random.bit_generator.state,
        "training_opponent": training_opponent_random.bit_generator.state,
        "random_opponent_episodes": random_opponent_episodes,
        "frozen_opponent_episodes": frozen_opponent_episodes,
    }


def _restore_rng_state(
    rng_state: dict[str, Any],
    agent: DQNAgent,
    replay: ReplayBuffer,
    opponent_selection_random: np.random.Generator,
    training_opponent_random: np.random.Generator,
) -> tuple[int, int]:
    try:
        agent.random.bit_generator.state = rng_state["agent"]
        replay.restore_random_state(rng_state["replay"])
        opponent_selection_random.bit_generator.state = rng_state["opponent_selection"]
        training_opponent_random.bit_generator.state = rng_state["training_opponent"]
        return (
            int(rng_state["random_opponent_episodes"]),
            int(rng_state.get("frozen_opponent_episodes", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint contains invalid random state") from exc


def run(
    args: argparse.Namespace, *, audit_service=None, audit_attempt_id: str | None = None
) -> int:
    """Record one training branch; a champion round can supply its existing attempt."""
    args = copy.copy(args)
    service = audit_service if audit_service is not None else open_audit(args)
    owns_service = audit_service is None
    attempt_id = audit_attempt_id
    try:
        if service is not None:
            experiment = ensure_experiment(service, args, args.rows, args.columns)
            original_resume = args.resume
            if args.resume is not None:
                args.resume = resolve_checkpoint_reference(
                    args.resume, service, experiment_name=args.experiment
                )
                starting = checkpoint_record(
                    service, experiment["id"], original_resume, path=args.resume
                )
                args.resume = resolve_checkpoint_reference(
                    f"checkpoint:{starting['id']}", service
                )
            else:
                starting = None
            config = effective_config(args)
            config["resume"] = (
                str(original_resume) if original_resume is not None else None
            )
            if args.frozen_opponent is not None:
                original_opponent = args.frozen_opponent
                args.frozen_opponent = resolve_checkpoint_reference(
                    original_opponent, service, experiment_name=args.experiment
                )
                opponent = checkpoint_record(
                    service,
                    experiment["id"],
                    original_opponent,
                    path=args.frozen_opponent,
                )
                config["frozen_opponent_checkpoint_id"] = opponent["id"]
                args.frozen_opponent = resolve_checkpoint_reference(
                    f"checkpoint:{opponent['id']}", service
                )
            if attempt_id is None:
                champion = service.current_champion(experiment["id"])
                attempt = service.create_attempt(
                    experiment["id"],
                    starting_checkpoint_id=starting["id"] if starting else None,
                    champion_at_start_assignment_id=champion["id"]
                    if champion
                    else None,
                    config=config,
                    target_episode=args.episodes,
                )
                attempt_id = attempt["id"]
            else:
                attempt = service.get_attempt(attempt_id)
                if attempt["experiment_id"] != experiment["id"] or attempt[
                    "starting_checkpoint_id"
                ] != (starting["id"] if starting else None):
                    raise ValueError(
                        "training inputs do not match the supplied audit attempt"
                    )
                service.update_attempt(
                    attempt_id, config={**attempt["config"], "training": config}
                )
            print(
                f"audit experiment={experiment['id']} attempt={attempt_id}", flush=True
            )
        result = _run_training(args, service, attempt_id)
        if service is not None and (audit_attempt_id is None or result):
            service.update_attempt(
                attempt_id,
                status="interrupted" if result == 130 else "completed",
                phase="finished",
                outcome=None if result else "trained_only",
                **({"error": "training interrupted"} if result else {}),
            )
        return result
    except BaseException as exc:
        if service is not None and attempt_id is not None:
            try:
                service.update_attempt(
                    attempt_id,
                    status="interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed",
                    error=str(exc) or type(exc).__name__,
                )
            except Exception as audit_error:
                print(
                    f"could not record training failure: {audit_error}", file=sys.stderr
                )
        raise
    finally:
        if owns_service:
            close_audit(service)


def _run_training(args: argparse.Namespace, service, attempt_id: str | None) -> int:
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
    frozen_opponent_episodes = 0
    state = TrainingState()
    if args.resume is not None:
        state, rng_state = _load_checkpoint(args.resume, agent, args)
        if rng_state is not None and not args.reset_rng_on_resume:
            random_opponent_episodes, frozen_opponent_episodes = _restore_rng_state(
                rng_state,
                agent,
                replay,
                opponent_selection_random,
                training_opponent_random,
            )
    if state.episode > args.episodes:
        raise ValueError("--episodes must not precede the resumed checkpoint episode")
    if service is not None:
        recorded = service.get_attempt(attempt_id)["config"]
        service.update_attempt(
            attempt_id,
            config={
                **recorded,
                "effective_device": str(device),
                "effective_optimizer": [
                    {key: value for key, value in group.items() if key != "params"}
                    for group in agent.optimizer.param_groups
                ],
                "rng_source": _rng_source(
                    args, rng_state if args.resume is not None else None
                ),
                "replay_restored": False,
            },
        )
    starting_episode = state.episode
    frozen_opponent: DQNAgent | None = None
    if args.frozen_opponent is not None:
        frozen_checkpoint, frozen_metadata = read_checkpoint(
            args.frozen_opponent, map_location=device
        )
        if frozen_metadata.board != (args.rows, args.columns):
            raise ValueError(
                f"frozen opponent board is {frozen_metadata.board}, "
                f"expected {(args.rows, args.columns)}"
            )
        frozen_opponent = DQNAgent(
            device=device,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            seed=args.seed + 3,
            channels=frozen_metadata.channels,
            blocks=frozen_metadata.blocks,
        )
        restore_agent(frozen_opponent, frozen_checkpoint, restore_optimizer=False)

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
        f"random-opponent probability {args.random_opponent_probability:.2f}, "
        f"other opponent={'frozen' if frozen_opponent is not None else 'self-play'}, "
        f"frozen-opening-random-moves={args.frozen_opening_random_moves}, "
        f"rng={_rng_source(args, rng_state if args.resume is not None else None)})",
        flush=True,
    )

    try:
        last_saved_state = None
        last_saved_path = None
        last_checkpoint_id = None
        if service is not None:
            previous = service.list_checkpoints(attempt_id)
            attempt = service.get_attempt(attempt_id)
            last_checkpoint_id = (
                previous[-1]["id"] if previous else attempt["starting_checkpoint_id"]
            )

        def save_checkpoint(path: Path, **flags):
            nonlocal last_saved_state, last_saved_path, last_checkpoint_id
            if flags.get("is_periodic_save") and getattr(
                args, "_screening_candidates", False
            ):
                flags.update(
                    is_screening_candidate=True,
                    candidate_index=(state.episode - starting_episode)
                    // args.checkpoint_every,
                )
            # All saves at one episode boundary share exact serialized bytes.
            if last_saved_state != state:
                staging = args.checkpoint_dir / ".audit-checkpoint.pt"
                _save_checkpoint(
                    staging,
                    agent,
                    state,
                    args,
                    _capture_rng_state(
                        agent,
                        replay,
                        opponent_selection_random,
                        training_opponent_random,
                        random_opponent_episodes,
                        frozen_opponent_episodes,
                    ),
                )
                last_saved_path = staging
                last_saved_state = state
                if service is not None:
                    record = service.import_checkpoint(
                        attempt["experiment_id"],
                        staging,
                        attempt_id=attempt_id,
                        parent_checkpoint_id=last_checkpoint_id,
                        **flags,
                    )
                    last_checkpoint_id = record["id"]
                    print(
                        f"audit checkpoint={record['id']} episode={state.episode}",
                        flush=True,
                    )
            elif service is not None:
                service.import_checkpoint(
                    attempt["experiment_id"],
                    last_saved_path,
                    attempt_id=attempt_id,
                    checkpoint_id=last_checkpoint_id,
                    **flags,
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            import shutil

            shutil.copyfile(last_saved_path, temporary)
            temporary.replace(path)
            return last_checkpoint_id

        if service is not None:
            save_checkpoint(args.checkpoint_dir / f"dqn-{state.episode:07d}.pt")
        with GameEnvironment(
            target=args.server,
            rows=args.rows,
            columns=args.columns,
            max_turns=args.max_turns,
            rpc_timeout=args.rpc_timeout,
        ) as environment:

            def evaluate_and_track() -> bool:
                batch = None
                if service is not None:
                    checkpoint_id = save_checkpoint(
                        args.checkpoint_dir / f"dqn-{state.episode:07d}.pt"
                    )
                    batch = service.create_evaluation(
                        attempt["experiment_id"],
                        checkpoint_id,
                        purpose="training",
                        attempt_id=attempt_id,
                        suite_definitions=[
                            evaluation_definition(
                                args,
                                rows=args.rows,
                                columns=args.columns,
                                seed=args.evaluation_seed,
                                games=args.eval_games,
                            )
                        ],
                    )
                try:
                    evaluation = evaluate_against_random(
                        environment, agent, evaluation_seeds
                    )
                    if batch is not None:
                        service.complete_suite(
                            batch["id"], 0, evaluation_result_dict(evaluation)
                        )
                except BaseException as exc:
                    if batch is not None:
                        service.fail_evaluation(
                            batch["id"],
                            str(exc) or type(exc).__name__,
                            status="interrupted"
                            if isinstance(exc, KeyboardInterrupt)
                            else "failed",
                        )
                    raise
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
                    save_checkpoint(best_path, is_best_in_attempt=True)
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
                elif frozen_opponent is not None:
                    learner_player = frozen_opponent_episodes % 2
                    result = collect_against_agent_episode(
                        environment,
                        agent,
                        frozen_opponent,
                        replay,
                        epsilon=epsilon,
                        learner_player=learner_player,
                        opening_random=training_opponent_random,
                        opening_random_moves=args.frozen_opening_random_moves,
                        terminal_win_bonus=args.terminal_win_bonus,
                        on_transition=optimize,
                    )
                    frozen_opponent_episodes += 1
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
                    frozen_results = [
                        item for item in recent if item.opponent == "frozen"
                    ]
                    mean_loss = (
                        float(np.mean(recent_losses)) if recent_losses else float("nan")
                    )
                    completed_this_run = state.episode - starting_episode
                    games_per_second = completed_this_run / max(
                        time.monotonic() - started_at, 1e-9
                    )
                    if service is not None:
                        service.record_metrics(
                            attempt_id,
                            state.episode,
                            {
                                "episode_start": state.episode - len(recent) + 1,
                                "elapsed_seconds": time.monotonic() - started_at,
                                "epsilon": epsilon,
                                "loss": mean_loss if np.isfinite(mean_loss) else None,
                                "games_per_second": games_per_second,
                                "replay_size": len(replay),
                                "environment_steps": state.environment_steps,
                                "optimization_steps": state.optimization_steps,
                                "opponents": {
                                    "self_play": len(self_play_results),
                                    "random": len(random_results),
                                    "frozen": len(frozen_results),
                                },
                                "episodes": [
                                    {
                                        "scores": list(item.scores),
                                        "opponent": item.opponent,
                                        "learner_player": item.learner_player,
                                    }
                                    for item in recent
                                ],
                            },
                        )
                    print(
                        f"episode={state.episode} steps={state.environment_steps} "
                        f"epsilon={epsilon:.3f} replay={len(replay)} "
                        f"opponents=self:{len(self_play_results)}/random:{len(random_results)}"
                        f"/frozen:{len(frozen_results)} "
                        f"{_score_summary(self_play_results, random_results, frozen_results)} "
                        f"loss={mean_loss:.5f} games/s={games_per_second:.2f}",
                        flush=True,
                    )

                early_stop = False
                if args.eval_every and state.episode % args.eval_every == 0:
                    early_stop = evaluate_and_track()

                checkpoint_episode = (
                    state.episode - starting_episode
                    if args.checkpoint_relative_to_start
                    else state.episode
                )
                if (
                    args.checkpoint_every
                    and checkpoint_episode % args.checkpoint_every == 0
                ):
                    save_checkpoint(
                        args.checkpoint_dir / f"dqn-{state.episode:07d}.pt",
                        is_periodic_save=True,
                    )

                if early_stop:
                    if service is not None:
                        service.update_attempt(attempt_id, stop_reason="early_stopping")
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
            save_checkpoint(final_path, is_final_in_attempt=True)
            if service is not None:
                service.update_attempt(attempt_id, latest_episode=state.episode)
            print(f"saved {final_path}", flush=True)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    return 130 if stop_requested else 0


def _format_stats(stats: MatchStats) -> str:
    return (
        f"{stats.wins}/{stats.draws}/{stats.losses}({stats.mean_score_difference:+.3f})"
    )


def _rng_source(
    args: argparse.Namespace,
    checkpoint_rng_state: dict[str, Any] | None,
) -> str:
    if args.resume is None:
        return f"seed:{args.seed}"
    if args.reset_rng_on_resume or checkpoint_rng_state is None:
        return f"fresh-seed:{args.seed}"
    return "checkpoint"


def _score_summary(
    self_play_results: list[EpisodeResult],
    random_results: list[EpisodeResult],
    frozen_results: list[EpisodeResult] | None = None,
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
    if frozen_results:
        differences = []
        for item in frozen_results:
            assert item.learner_player is not None
            player = item.learner_player
            differences.append(item.scores[player] - item.scores[1 - player])
        parts.append(f"frozen_learner_diff={np.mean(differences):+.3f}")
    return " ".join(parts)


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except (ConnectionError, FileNotFoundError, ValueError, grpc.RpcError) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
