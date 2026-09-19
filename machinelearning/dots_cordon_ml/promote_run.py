"""Evaluate saved checkpoints in episode order using the champion-loop gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

import torch

from .audit import AuditService
from .audit.database import PromotionConflict
from .audit.integration import (
    close_audit,
    evaluation_definition,
    resolve_checkpoint_reference,
)
from .checkpoint import read_checkpoint
from . import promotion_evaluation as evaluation


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-attempt", required=True)
    parser.add_argument(
        "--policy-from-attempt",
        help="copy evaluation settings from an earlier champion-loop attempt",
    )
    parser.add_argument("--database-url")
    parser.add_argument("--server", default="127.0.0.1:50051")
    parser.add_argument(
        "--baseline",
        help="initial champion reference; otherwise use the run's frozen evaluation opponent",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/promoted-policy.pt")
    )
    evaluation.add_evaluation_arguments(parser)
    return parser.parse_args(arguments)


EVALUATION_KEYS = (
    "evaluation_workers",
    "screen_games",
    "screen_suites",
    "screen_seed",
    "screen_max_regression",
    "head_to_head_games",
    "head_to_head_suites",
    "head_to_head_seed",
    "extended_head_to_head_games",
    "extended_head_to_head_suites",
    "extended_head_to_head_seed",
    "opening_random_moves",
    "promotion_min_match_score",
    "promotion_min_suite_wins",
    "device",
    "max_turns",
    "rpc_timeout",
)


def candidates_for_run(service, source_id):
    # Repeated saves of the same episode are one candidate; episode zero is untrained.
    candidates = {}
    for checkpoint in service.list_checkpoints(source_id):
        if checkpoint["episode"] > 0:
            candidates[checkpoint["episode"]] = checkpoint
    return [candidates[key] for key in sorted(candidates)]


def suite_definitions(args, metadata, seeds, games, *, head_to_head=False):
    return [
        {
            **evaluation_definition(
                args,
                rows=metadata.rows,
                columns=metadata.columns,
                seed=seed,
                games=games,
                kind="head_to_head" if head_to_head else "random",
                opening_random_moves=args.opening_random_moves if head_to_head else 0,
            ),
            "evaluator_version": "greedy-checkpoint-v2",
            # Same screening definition for DQN and policy checkpoints: each uses
            # its native greedy head, never tree search. Architecture lives on the checkpoint.
            "subject_policy": "greedy_checkpoint",
            "opponent_policy": "greedy_checkpoint"
            if head_to_head
            else "uniform_random",
        }
        for seed in seeds
    ]


def _run_candidate(args, service, experiment, attempt, candidate, assignment, device):
    candidate_path = resolve_checkpoint_reference(
        f"checkpoint:{candidate['id']}", service
    )
    champion_path = resolve_checkpoint_reference(
        f"checkpoint:{assignment['checkpoint_id']}", service
    )
    _, metadata = read_checkpoint(candidate_path, map_location="cpu")
    seeds = service.reserve_suite_seeds(
        experiment["id"], "screening", args.screen_suites, args.screen_seed
    )
    definitions = suite_definitions(args, metadata, seeds, args.screen_games)
    batches = [
        service.create_evaluation(
            experiment["id"],
            identifier,
            purpose="screening",
            attempt_id=attempt["id"],
            suite_definitions=definitions,
            config={
                "screen_max_regression": args.screen_max_regression,
                "mode": "policy",
            },
        )
        for identifier in (assignment["checkpoint_id"], candidate["id"])
    ]
    service.update_attempt(attempt["id"], phase="screening")
    champion_result, contender = evaluation._evaluate_random_screen(
        args,
        (champion_path, candidate_path),
        device,
        seeds,
        args.screen_games,
        audit_service=service,
        evaluation_ids=tuple(batch["id"] for batch in batches),
    )
    qualified = evaluation._passes_random_screen(
        contender, champion_result, args.screen_max_regression
    )
    service.select_best_screened_checkpoint(attempt["id"], [batches[1]["id"]])
    service.record_decision(
        attempt["id"],
        candidate["id"],
        stage="screening",
        result="qualified" if qualified else "rejected",
        reason="within allowed random-score regression"
        if qualified
        else "random-score regression exceeds threshold",
        candidate_evaluation_id=batches[1]["id"],
        champion_evaluation_id=batches[0]["id"],
        policy={"screen_max_regression": args.screen_max_regression},
        rank=1 if qualified else None,
    )
    if not qualified:
        service.update_attempt(
            attempt["id"],
            status="completed",
            phase="finished",
            outcome="no_qualified_candidate",
        )
        print(
            f"REJECTED episode={candidate['episode']} reason=random-screen", flush=True
        )
        return

    service.update_attempt(attempt["id"], phase="challenging")
    for extended in (False, True):
        games = (
            args.extended_head_to_head_games if extended else args.head_to_head_games
        )
        count = (
            args.extended_head_to_head_suites if extended else args.head_to_head_suites
        )
        kind = "extended_head_to_head" if extended else "head_to_head"
        base = args.extended_head_to_head_seed if extended else args.head_to_head_seed
        seeds = service.reserve_suite_seeds(experiment["id"], kind, count, base)
        policy = {
            "promotion_min_match_score": evaluation._extended_minimum_match_score(
                games * count
            )
            if extended
            else args.promotion_min_match_score,
            "promotion_min_suite_wins": 1
            if extended
            else args.promotion_min_suite_wins,
            **({"extended_validation": True} if extended else {}),
        }
        batch = service.create_evaluation(
            experiment["id"],
            candidate["id"],
            purpose="head_to_head",
            attempt_id=attempt["id"],
            opponent_checkpoint_id=assignment["checkpoint_id"],
            suite_definitions=suite_definitions(
                args, metadata, seeds, games, head_to_head=True
            ),
            config=policy,
        )
        suites = evaluation._evaluate_head_to_head_suites(
            args,
            contender,
            champion_result,
            device,
            seeds,
            games,
            args.opening_random_moves,
            audit_service=service,
            evaluation_id=batch["id"],
        )
        passed = evaluation._passes_promotion(
            suites,
            policy["promotion_min_match_score"],
            policy["promotion_min_suite_wins"],
        )
        extend = (
            not extended
            and not passed
            and evaluation._requires_extended_validation(
                suites, args.promotion_min_match_score
            )
        )
        service.record_decision(
            attempt["id"],
            candidate["id"],
            stage="challenge",
            result="passed" if passed else "extended" if extend else "rejected",
            candidate_evaluation_id=batch["id"],
            champion_evaluation_id=batches[0]["id"],
            policy=policy,
        )
        if passed:
            promoted = service.promote(
                experiment["id"],
                candidate["id"],
                attempt_id=attempt["id"],
                expected_assignment_id=assignment["id"],
                candidate_evaluation_id=batch["id"],
                champion_evaluation_id=batches[0]["id"],
                policy=policy,
            )
            service.export_checkpoint(candidate["id"], args.output)
            print(
                f"PROMOTED generation={promoted['generation']} episode={candidate['episode']} extended={extended}",
                flush=True,
            )
            return
        if not extend:
            service.update_attempt(
                attempt["id"],
                status="completed",
                phase="finished",
                outcome="no_challenger_passed",
            )
            print(
                f"REJECTED episode={candidate['episode']} reason=head-to-head extended={extended}",
                flush=True,
            )
            return
        print(
            f"EXTENDING episode={candidate['episode']} suites={args.extended_head_to_head_suites}",
            flush=True,
        )


def run(args):
    service = AuditService(args.database_url)
    active = None
    try:
        if args.policy_from_attempt:
            config = service.get_attempt(args.policy_from_attempt)["config"]
            for key in EVALUATION_KEYS:
                if key in config:
                    setattr(args, key, config[key])
        source = service.get_attempt(args.source_attempt)
        if source["status"] == "running":
            raise ValueError(
                "stop the source training run before promoting its checkpoints"
            )
        experiment = service.get_experiment(source["experiment_id"])
        if experiment["game_config"].get("max_turns", 0) != args.max_turns:
            raise ValueError("evaluation turn limit differs from training")
        candidates = candidates_for_run(service, source["id"])
        if not candidates or any(
            item["model"].get("kind") != "policy_value" for item in candidates
        ):
            raise ValueError("source must contain trained policy/value checkpoints")
        settings = {key: getattr(args, key) for key in EVALUATION_KEYS}
        for key in (
            "screen_games",
            "head_to_head_games",
            "extended_head_to_head_games",
        ):
            if settings[key] < 2 or settings[key] % 2:
                raise ValueError(f"{key} must be positive and even")
        for key in (
            "screen_suites",
            "head_to_head_suites",
            "extended_head_to_head_suites",
            "evaluation_workers",
        ):
            if settings[key] < 1:
                raise ValueError(f"{key} must be positive")
        if (
            not 0.5 < args.promotion_min_match_score <= 1
            or not 1 <= args.promotion_min_suite_wins <= args.head_to_head_suites
        ):
            raise ValueError("invalid promotion thresholds")
        if args.screen_max_regression < 0 or args.opening_random_moves < 0:
            raise ValueError("regression and opening length must be nonnegative")
        device = evaluation._device(args.device)
        torch.set_num_threads(1)
        # Check connectivity before creating any promotion records.
        path = resolve_checkpoint_reference(
            f"checkpoint:{candidates[0]['id']}", service
        )
        _, metadata = read_checkpoint(path, map_location="cpu")
        with evaluation._game_environment(args, metadata):
            pass
        assignment = service.current_champion(experiment["id"])
        if assignment is None:
            if args.baseline:
                path = resolve_checkpoint_reference(args.baseline, service)
                baseline = service.import_checkpoint(experiment["id"], path)
            else:
                opponents = {
                    batch["opponent_checkpoint_id"]
                    for batch in service.list_evaluations(experiment["id"])
                    if batch["attempt_id"] == source["id"]
                    and batch["opponent_checkpoint_id"]
                }
                if len(opponents) != 1:
                    raise ValueError(
                        "provide --baseline: source has no unique frozen opponent"
                    )
                baseline = service.get_checkpoint(opponents.pop())
            assignment = service.bootstrap(experiment["id"], baseline["id"])
        print(
            f"PROMOTION RUN experiment={experiment['name']} candidates={len(candidates)} settings={json.dumps(settings, sort_keys=True)}",
            flush=True,
        )
        previous_attempts = {
            item["id"]: item for item in service.list_attempts(experiment["id"])
        }
        for index, saved in enumerate(candidates, 1):
            config = {
                "algorithm": "saved_checkpoint_promotion_v1",
                "mode": "policy",
                "source_attempt_id": source["id"],
                "source_checkpoint_id": saved["id"],
                **settings,
            }
            identity = str(uuid5(NAMESPACE_URL, json.dumps(config, sort_keys=True)))
            attempt = previous_attempts.get(identity)
            if attempt and attempt["status"] == "completed":
                print(
                    f"SKIP already evaluated episode={saved['episode']} outcome={attempt['outcome']}",
                    flush=True,
                )
                continue
            assignment = service.current_champion(experiment["id"])
            if attempt is None:
                attempt = service.create_attempt(
                    experiment["id"],
                    starting_checkpoint_id=saved["id"],
                    champion_at_start_assignment_id=assignment["id"],
                    config=config,
                    target_episode=saved["episode"],
                    attempt_id=identity,
                )
            elif attempt["champion_at_start_assignment_id"] != assignment["id"]:
                raise PromotionConflict(
                    "champion changed while saved-checkpoint evaluation was stopped"
                )
            elif attempt["status"] != "running":
                service.update_attempt(attempt["id"], status="running", error=None)
            active = attempt["id"]
            # A retry retains failed evidence and uses new seed suites. Never promote
            # from a partial batch or silently retry an already rejected candidate.
            for batch in service.list_evaluations(experiment["id"]):
                if batch["attempt_id"] == active and batch["status"] == "running":
                    service.fail_evaluation(
                        batch["id"],
                        "restarted incomplete evaluation",
                        status="interrupted",
                    )
            path = resolve_checkpoint_reference(f"checkpoint:{saved['id']}", service)
            candidate = service.import_checkpoint(
                experiment["id"],
                path,
                attempt_id=active,
                parent_checkpoint_id=saved["id"],
                is_screening_candidate=True,
                is_final_in_attempt=True,
                candidate_index=1,
            )
            print(
                f"CANDIDATE {index}/{len(candidates)} episode={saved['episode']} attempt={active}",
                flush=True,
            )
            _run_candidate(
                args, service, experiment, attempt, candidate, assignment, device
            )
            active = None
        assignment = service.current_champion(experiment["id"])
        service.export_checkpoint(assignment["checkpoint_id"], args.output)
        champion = service.get_checkpoint(assignment["checkpoint_id"])
        print(
            f"FINISHED champion={assignment['checkpoint_id']} episode={champion['episode']} output={args.output}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        if active and service.get_attempt(active)["status"] != "completed":
            service.update_attempt(
                active,
                status="interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed",
                error=str(exc) or type(exc).__name__,
            )
        raise
    finally:
        close_audit(service)


def main():
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (ValueError, ConnectionError, FileNotFoundError) as exc:
        print(f"saved-checkpoint evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
