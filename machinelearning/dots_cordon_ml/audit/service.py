"""Application-facing training history operations shared by CLIs and a future API."""

from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
import math
from pathlib import Path
import tempfile
from uuid import uuid4

from ..checkpoint import read_checkpoint
from .database import AuditError, Database, PromotionConflict
from .repository import AuditRepository, CheckpointBlobRepository, fingerprint


_PARENT_UNSET = object()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_id():
    return str(uuid4())


def _same_experiment(record, experiment_id, label):
    if record["experiment_id"] != experiment_id:
        raise AuditError(f"{label} belongs to another experiment")


def _complete(batch):
    if (
        batch["status"] != "completed"
        or len(batch["suites"]) != batch["planned_suite_count"]
        or any(suite["status"] != "completed" for suite in batch["suites"])
    ):
        raise AuditError("Gate evidence requires all planned suites to be complete")


def _matched_definitions(candidate, champion):
    _complete(candidate)
    _complete(champion)
    left = [suite["definition"] for suite in candidate["suites"]]
    right = [suite["definition"] for suite in champion["suites"]]
    if left != right:
        raise AuditError(
            "Screening evidence must use identical ordered suite definitions"
        )


def _match_score(batch):
    games = sum(
        suite[f"player_{seat}_{field}"]
        for suite in batch["suites"]
        for seat in (0, 1)
        for field in ("wins", "draws", "losses")
    )
    points = sum(
        suite[f"player_{seat}_wins"] + 0.5 * suite[f"player_{seat}_draws"]
        for suite in batch["suites"]
        for seat in (0, 1)
    )
    return points / games


class AuditService:
    def __init__(
        self,
        database_url: str | None = None,
        *,
        read_only: bool = False,
        check_schema: bool = True,
    ):
        self.database = Database(database_url, read_only=read_only)
        try:
            if check_schema:
                self.database.check_schema()
        except BaseException:
            self.database.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.database.close()

    @contextmanager
    def reader(self):
        """Bounded application queries for HTTP and other read clients."""
        from .read_repository import AuditReadRepository
        from .reader import AuditReader

        self.database.check_schema()
        with self.database.read_snapshot() as connection:
            yield AuditReader(AuditReadRepository(connection))

    @contextmanager
    def _repositories(self):
        with self.database.transaction() as connection:
            yield AuditRepository(connection), CheckpointBlobRepository(connection)

    def ensure_experiment(self, name: str, game_config: dict) -> dict:
        game_config = {"max_turns": 0, **game_config}
        if not name or len(name) > 255:
            raise AuditError("Experiment name must contain 1–255 characters")
        for dimension in ("rows", "columns"):
            if (
                not isinstance(game_config.get(dimension), int)
                or game_config[dimension] < 2
            ):
                raise AuditError("Experiment requires positive board rows and columns")
        with self._repositories() as (repository, _):
            existing = repository.find_experiment(name)
            if existing:
                if existing["game_config"] != game_config:
                    raise AuditError(
                        f"Experiment {name!r} has a different game configuration"
                    )
                return existing
            timestamp = now()
            return repository.add_experiment(
                dict(
                    id=new_id(),
                    name=name,
                    game_config=game_config,
                    game_config_fingerprint=fingerprint(game_config),
                    current_champion_assignment_id=None,
                    seed_offsets={},
                    version=0,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )

    def get_experiment(self, id_or_name: str) -> dict:
        with self._repositories() as (repository, _):
            return repository.get_experiment(id_or_name)

    def list_experiments(self) -> list[dict]:
        with self._repositories() as (repository, _):
            return repository.list_experiments()

    def get_checkpoint(self, checkpoint_id: str) -> dict:
        with self._repositories() as (repository, _):
            return repository.get_checkpoint(checkpoint_id)

    def get_attempt(self, attempt_id: str) -> dict:
        with self._repositories() as (repository, _):
            return repository.get_attempt(attempt_id)

    def list_attempts(self, experiment_id: str) -> list[dict]:
        with self._repositories() as (repository, _):
            return repository.list_attempts(experiment_id)

    def list_checkpoints(self, attempt_id: str) -> list[dict]:
        with self._repositories() as (repository, _):
            return repository.list_checkpoints(attempt_id)

    def list_metrics(self, attempt_id: str) -> list[dict]:
        with self._repositories() as (repository, _):
            return repository.list_metrics(attempt_id)

    def list_evaluations(
        self, experiment_id: str, checkpoint_id: str | None = None
    ) -> list[dict]:
        with self._repositories() as (repository, _):
            return repository.list_evaluations(experiment_id, checkpoint_id)

    def get_evaluation(self, batch_id: str) -> dict:
        with self._repositories() as (repository, _):
            return repository.get_evaluation(batch_id)

    def list_decisions(self, attempt_id: str) -> list[dict]:
        with self._repositories() as (repository, _):
            return repository.list_decisions(attempt_id)

    def champion_history(self, experiment_id: str) -> list[dict]:
        with self._repositories() as (repository, _):
            return repository.list_champions(experiment_id)

    def current_champion(self, experiment_id: str) -> dict | None:
        with self._repositories() as (repository, _):
            assignment_id = repository.get_experiment(experiment_id)[
                "current_champion_assignment_id"
            ]
            return repository.get_assignment(assignment_id) if assignment_id else None

    def import_checkpoint(
        self,
        experiment_id,
        path,
        *,
        attempt_id=None,
        parent_checkpoint_id=_PARENT_UNSET,
        checkpoint_id=None,
        is_periodic_save=False,
        is_screening_candidate=False,
        is_best_in_attempt=False,
        is_final_in_attempt=False,
        candidate_index=None,
    ) -> dict:
        # Validate the exact bytes being stored, avoiding a file-replacement race.
        payload = Path(path).read_bytes()
        state, metadata = read_checkpoint(BytesIO(payload), map_location="cpu")
        if (
            min(
                metadata.episode,
                metadata.environment_steps,
                metadata.optimization_steps,
            )
            < 0
        ):
            raise AuditError("Checkpoint training counters must be nonnegative")
        flags = dict(
            is_periodic_save=is_periodic_save,
            is_screening_candidate=is_screening_candidate,
            is_best_in_attempt=is_best_in_attempt,
            is_final_in_attempt=is_final_in_attempt,
        )
        if attempt_id is None and any(flags.values()):
            raise AuditError("Checkpoint selection fields require an owning attempt")
        with self._repositories() as (repository, blobs):
            repository.lock_experiment(experiment_id)
            experiment = repository.get_experiment(experiment_id)
            if metadata.board != (
                experiment["game_config"]["rows"],
                experiment["game_config"]["columns"],
            ):
                raise AuditError("Checkpoint board does not match the experiment")
            attempt = repository.get_attempt(attempt_id) if attempt_id else None
            if attempt:
                _same_experiment(attempt, experiment_id, "Attempt")
            blob = blobs.put(
                payload,
                identifier=new_id(),
                created_at=now(),
                format_version=int(state.get("format_version", 1)),
            )
            if checkpoint_id:
                try:
                    existing = repository.get_checkpoint(checkpoint_id)
                except AuditError:
                    existing = None
                if existing and (
                    existing["checkpoint_blob_id"] != blob["id"]
                    or existing["attempt_id"] != attempt_id
                    or (
                        parent_checkpoint_id is not _PARENT_UNSET
                        and existing["parent_checkpoint_id"] != parent_checkpoint_id
                    )
                    or existing["experiment_id"] != experiment_id
                ):
                    raise AuditError(
                        "Checkpoint operation ID reused with different content or lineage"
                    )
            else:
                existing = repository.find_checkpoint_blob(
                    experiment_id, blob["id"], attempt_id
                )
            if existing:
                if attempt_id and parent_checkpoint_id not in (
                    _PARENT_UNSET,
                    existing["parent_checkpoint_id"],
                    existing["id"],
                ):
                    raise AuditError("Checkpoint ancestry cannot be changed")
                changes = {name: True for name, enabled in flags.items() if enabled}
                if candidate_index is not None:
                    if existing["candidate_index"] not in (None, candidate_index):
                        raise AuditError("Checkpoint candidate index cannot be changed")
                    changes["candidate_index"] = candidate_index
                for flag in ("is_best_in_attempt", "is_final_in_attempt"):
                    if flags[flag]:
                        repository.clear_checkpoint_selection(attempt_id, flag)
                return repository.update_checkpoint_flags(existing["id"], changes)
            checkpoints = repository.list_checkpoints(attempt_id) if attempt else []
            expected_parent = (
                checkpoints[-1]["id"]
                if checkpoints
                else (attempt["starting_checkpoint_id"] if attempt else None)
            )
            if parent_checkpoint_id is _PARENT_UNSET:
                parent_checkpoint_id = expected_parent
            if attempt and parent_checkpoint_id != expected_parent:
                raise AuditError(
                    "Checkpoint parent must be the latest saved state in its attempt"
                )
            if parent_checkpoint_id:
                parent = repository.get_checkpoint(parent_checkpoint_id)
                _same_experiment(parent, experiment_id, "Parent checkpoint")
                if parent["episode"] > metadata.episode:
                    raise AuditError("Checkpoint episode cannot precede its parent")
            if attempt and metadata.episode < attempt["start_episode"]:
                raise AuditError("Checkpoint episode cannot precede the attempt")
            for flag in ("is_best_in_attempt", "is_final_in_attempt"):
                if flags[flag]:
                    repository.clear_checkpoint_selection(attempt_id, flag)
            result = repository.add_checkpoint(
                dict(
                    id=checkpoint_id or new_id(),
                    experiment_id=experiment_id,
                    attempt_id=attempt_id,
                    checkpoint_blob_id=blob["id"],
                    parent_checkpoint_id=parent_checkpoint_id,
                    episode=metadata.episode,
                    environment_steps=metadata.environment_steps,
                    optimization_steps=metadata.optimization_steps,
                    board={"rows": metadata.rows, "columns": metadata.columns},
                    model={
                        "channels": metadata.channels,
                        "blocks": metadata.blocks,
                        **({"kind": metadata.kind} if metadata.kind != "dqn" else {}),
                    },
                    origin="training" if attempt else "imported",
                    save_sequence=len(checkpoints) + 1,
                    candidate_index=candidate_index,
                    created_at=now(),
                    **flags,
                )
            )
            if attempt:
                repository.update_attempt(
                    attempt_id,
                    {
                        "latest_episode": max(
                            metadata.episode, attempt["latest_episode"]
                        ),
                        "heartbeat_at": now(),
                        "updated_at": now(),
                    },
                )
            return result

    def export_checkpoint(self, checkpoint_id, path) -> Path:
        with self._repositories() as (repository, blobs):
            checkpoint = repository.get_checkpoint(checkpoint_id)
            payload = blobs.get(checkpoint["checkpoint_blob_id"])
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(payload)
                output.flush()
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target

    def create_attempt(
        self,
        experiment_id,
        *,
        starting_checkpoint_id=None,
        champion_at_start_assignment_id=None,
        config,
        target_episode,
        attempt_id=None,
    ) -> dict:
        with self._repositories() as (repository, _):
            repository.lock_experiment(experiment_id)
            start_episode = 0
            if starting_checkpoint_id:
                checkpoint = repository.get_checkpoint(starting_checkpoint_id)
                _same_experiment(checkpoint, experiment_id, "Starting checkpoint")
                start_episode = checkpoint["episode"]
            if champion_at_start_assignment_id:
                _same_experiment(
                    repository.get_assignment(champion_at_start_assignment_id),
                    experiment_id,
                    "Champion assignment",
                )
            if target_episode < start_episode:
                raise AuditError(
                    "Attempt target episode precedes its starting checkpoint"
                )
            timestamp = now()
            values = dict(
                experiment_id=experiment_id,
                starting_checkpoint_id=starting_checkpoint_id,
                champion_at_start_assignment_id=champion_at_start_assignment_id,
                config=config,
                start_episode=start_episode,
                target_episode=target_episode,
            )
            if attempt_id:
                try:
                    existing = repository.get_attempt(attempt_id)
                except AuditError:
                    existing = None
                if existing:
                    if any(existing[key] != value for key, value in values.items()):
                        raise AuditError(
                            "Attempt operation ID reused with different content"
                        )
                    return existing
            return repository.add_attempt(
                dict(
                    id=attempt_id or new_id(),
                    **values,
                    latest_episode=start_episode,
                    phase="training",
                    status="running",
                    outcome=None,
                    version=0,
                    created_at=timestamp,
                    updated_at=timestamp,
                    started_at=timestamp,
                    heartbeat_at=timestamp,
                )
            )

    def update_attempt(self, attempt_id, **fields) -> dict:
        allowed = {
            "config",
            "phase",
            "status",
            "outcome",
            "latest_episode",
            "error",
            "stop_reason",
            "heartbeat_at",
            "ended_at",
            "last_interrupted_at",
            "last_resumed_at",
            "error_at",
        }
        if set(fields) - allowed:
            raise AuditError(
                f"Attempt fields are immutable or unknown: {sorted(set(fields) - allowed)}"
            )
        if "status" in fields and fields["status"] not in {
            "running",
            "completed",
            "interrupted",
            "failed",
            "abandoned",
        }:
            raise AuditError("Unknown attempt status")
        if "phase" in fields and fields["phase"] not in {
            "training",
            "screening",
            "challenging",
            "finished",
        }:
            raise AuditError("Unknown attempt phase")
        with self._repositories() as (repository, _):
            attempt = repository.get_attempt(attempt_id)
            repository.lock_experiment(attempt["experiment_id"])
            attempt = repository.get_attempt(attempt_id)
            if "config" in fields and fields["config"] != attempt["config"]:
                if (
                    repository.list_checkpoints(attempt_id)
                    or repository.list_metrics(attempt_id)
                    or any(
                        batch["attempt_id"] == attempt_id
                        for batch in repository.list_evaluations(
                            attempt["experiment_id"]
                        )
                    )
                ):
                    raise AuditError(
                        "Attempt configuration is immutable once training evidence exists"
                    )
            if (
                fields.get("latest_episode", attempt["latest_episode"])
                < attempt["latest_episode"]
            ):
                raise AuditError("Attempt progress cannot move backwards")
            timestamp = now()
            if fields.get("status") == "running" and attempt["status"] != "running":
                if attempt["status"] not in {"interrupted", "failed"}:
                    raise AuditError("Only unfinished attempts can be continued")
                fields.update(last_resumed_at=timestamp, ended_at=None)
                repository.clear_checkpoint_selection(attempt_id, "is_final_in_attempt")
            elif fields.get("status") in {
                "completed",
                "interrupted",
                "failed",
                "abandoned",
            }:
                fields.setdefault("ended_at", timestamp)
                if fields["status"] == "interrupted":
                    fields.setdefault("last_interrupted_at", timestamp)
            if fields.get("error"):
                fields.setdefault("error_at", timestamp)
            fields.update(
                updated_at=timestamp,
                heartbeat_at=timestamp,
                version=attempt["version"] + 1,
            )
            return repository.update_attempt(attempt_id, fields)

    def record_metrics(
        self, attempt_id, episode, metrics, *, sample_sequence=None
    ) -> None:
        with self._repositories() as (repository, _):
            attempt = repository.get_attempt(attempt_id)
            repository.lock_experiment(attempt["experiment_id"])
            previous = repository.list_metrics(attempt_id)
            # Explicit sequence is preferred; identical retries of the last sample are harmless.
            if (
                sample_sequence is None
                and previous
                and previous[-1]["episode"] == episode
                and previous[-1]["metrics"] == metrics
            ):
                return
            sequence = (
                sample_sequence if sample_sequence is not None else len(previous) + 1
            )
            existing = next(
                (row for row in previous if row["sample_sequence"] == sequence), None
            )
            if existing:
                if existing["episode"] != episode or existing["metrics"] != metrics:
                    raise AuditError(
                        "Metric sample sequence reused with different content"
                    )
                return
            if (
                sequence != len(previous) + 1
                or episode < attempt["start_episode"]
                or (previous and episode < previous[-1]["episode"])
            ):
                raise AuditError(
                    "Metric samples must have increasing sequence and nondecreasing episodes"
                )
            repository.add_metrics(
                dict(
                    attempt_id=attempt_id,
                    sample_sequence=sequence,
                    episode=episode,
                    metrics=metrics,
                    created_at=now(),
                )
            )
            attempt = repository.get_attempt(attempt_id)
            timestamp = now()
            repository.update_attempt(
                attempt_id,
                {
                    "latest_episode": max(episode, attempt["latest_episode"]),
                    "heartbeat_at": timestamp,
                    "updated_at": timestamp,
                },
            )

    def bootstrap(self, experiment_id, checkpoint_id) -> dict:
        with self._repositories() as (repository, _):
            repository.lock_experiment(experiment_id)
            checkpoint = repository.get_checkpoint(checkpoint_id)
            _same_experiment(checkpoint, experiment_id, "Checkpoint")
            experiment = repository.get_experiment(experiment_id)
            current = experiment["current_champion_assignment_id"]
            if current:
                assignment = repository.get_assignment(current)
                if assignment["checkpoint_id"] == checkpoint_id:
                    return assignment
                raise AuditError(
                    "Experiment already has a champion; bootstrap cannot replace it"
                )
            timestamp = now()
            assignment = repository.add_assignment(
                dict(
                    id=new_id(),
                    experiment_id=experiment_id,
                    generation=1,
                    checkpoint_id=checkpoint_id,
                    reason="bootstrap",
                    created_at=timestamp,
                )
            )
            if not repository.compare_and_set_champion(
                experiment_id, None, assignment["id"], timestamp
            ):
                raise PromotionConflict("Champion changed while bootstrapping")
            return assignment

    def reserve_suite_seeds(self, experiment_id, kind, count, base) -> tuple[int, ...]:
        if count < 1:
            raise AuditError("Seed count must be positive")
        with self._repositories() as (repository, _):
            repository.lock_experiment(experiment_id)
            experiment = repository.get_experiment(experiment_id)
            offsets = experiment["seed_offsets"]
            offset = offsets.get(kind, 0)
            offsets[kind] = offset + count
            repository.update_experiment(
                experiment_id, {"seed_offsets": offsets, "updated_at": now()}
            )
            return tuple(base + offset + index for index in range(count))

    def create_evaluation(
        self,
        experiment_id,
        checkpoint_id,
        *,
        purpose,
        suite_definitions,
        attempt_id=None,
        opponent_checkpoint_id=None,
        config=None,
        batch_id=None,
    ) -> dict:
        if not suite_definitions:
            raise AuditError("An evaluation requires at least one suite")
        with self._repositories() as (repository, _):
            repository.lock_experiment(experiment_id)
            _same_experiment(
                repository.get_checkpoint(checkpoint_id),
                experiment_id,
                "Subject checkpoint",
            )
            if attempt_id:
                _same_experiment(
                    repository.get_attempt(attempt_id), experiment_id, "Attempt"
                )
            if opponent_checkpoint_id:
                _same_experiment(
                    repository.get_checkpoint(opponent_checkpoint_id),
                    experiment_id,
                    "Opponent checkpoint",
                )
            timestamp = now()
            batch_values = dict(
                experiment_id=experiment_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
                opponent_checkpoint_id=opponent_checkpoint_id,
                purpose=purpose,
                opponent_kind="checkpoint" if opponent_checkpoint_id else "random",
                config=config or {},
                planned_suite_count=len(suite_definitions),
            )
            if batch_id:
                try:
                    existing = repository.get_evaluation(batch_id)
                except AuditError:
                    existing = None
                if existing:
                    if (
                        any(
                            existing[key] != value
                            for key, value in batch_values.items()
                        )
                        or [suite["definition"] for suite in existing["suites"]]
                        != suite_definitions
                    ):
                        raise AuditError(
                            "Evaluation operation ID reused with different content"
                        )
                    return existing
            batch = repository.add_evaluation(
                dict(
                    id=batch_id or new_id(),
                    **batch_values,
                    status="running",
                    created_at=timestamp,
                    started_at=timestamp,
                    heartbeat_at=timestamp,
                )
            )
            for index, definition in enumerate(suite_definitions):
                games = definition.get("expected_games", definition.get("games"))
                seats = definition.get("seat_schedule")
                seeds = definition.get("game_seeds")
                if (
                    not isinstance(games, int)
                    or games < 1
                    or not isinstance(seats, list)
                    or len(seats) != games
                    or any(seat not in (0, 1) for seat in seats)
                ):
                    raise AuditError(
                        "Suite requires positive expected_games and an exact 0/1 seat_schedule"
                    )
                if not isinstance(seeds, list) or len(seeds) != games:
                    raise AuditError(
                        "Suite requires exact ordered game_seeds for every game"
                    )
                repository.add_suite(
                    dict(
                        id=new_id(),
                        batch_id=batch["id"],
                        suite_index=index,
                        definition=definition,
                        suite_fingerprint=fingerprint(definition),
                        expected_games=games,
                        expected_player_0_games=seats.count(0),
                        expected_player_1_games=seats.count(1),
                        status="running",
                        started_at=timestamp,
                    )
                )
            return repository.get_evaluation(batch["id"])

    def complete_suite(self, batch_id, suite_index, result) -> None:
        with self._repositories() as (repository, _):
            batch = repository.get_evaluation(batch_id)
            repository.lock_experiment(batch["experiment_id"])
            batch = repository.get_evaluation(batch_id)
            if not 0 <= suite_index < len(batch["suites"]):
                raise AuditError("Unknown suite index")
            suite = batch["suites"][suite_index]
            values = {}
            for seat in (0, 1):
                stats = result.get(f"as_player_{seat}", {})
                for field in ("wins", "draws", "losses", "score_difference_sum"):
                    value = stats.get(field)
                    if type(value) is not int or (
                        field != "score_difference_sum" and value < 0
                    ):
                        raise AuditError(
                            "Suite statistics require exact integer counts and score sums"
                        )
                    values[f"player_{seat}_{field}"] = value
                if (
                    sum(stats[field] for field in ("wins", "draws", "losses"))
                    != suite[f"expected_player_{seat}_games"]
                ):
                    raise AuditError(
                        "Completed suite game counts differ from the planned seat schedule"
                    )
            if suite["status"] == "completed":
                if any(suite[key] != value for key, value in values.items()):
                    raise AuditError("Completed suite results are immutable")
                return
            if batch["status"] != "running" or suite["status"] != "running":
                raise AuditError(
                    "Cannot complete a stopped evaluation; create a new batch"
                )
            timestamp = now()
            elapsed = (
                datetime.fromisoformat(timestamp)
                - datetime.fromisoformat(suite["started_at"])
            ).total_seconds()
            if "elapsed_seconds" in result:
                elapsed = float(result["elapsed_seconds"])
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise AuditError("Suite duration must be nonnegative")
            repository.update_suite(
                suite["id"],
                dict(
                    **values,
                    status="completed",
                    ended_at=timestamp,
                    elapsed_seconds=elapsed,
                ),
            )
            completed = all(
                item["status"] == "completed" or item["id"] == suite["id"]
                for item in batch["suites"]
            )
            repository.update_evaluation(
                batch_id,
                dict(
                    status="completed" if completed else "running",
                    heartbeat_at=timestamp,
                    **({"ended_at": timestamp} if completed else {}),
                ),
            )

    def fail_evaluation(self, batch_id, error, status="failed") -> None:
        if status not in {"failed", "interrupted", "cancelled"}:
            raise AuditError("Invalid evaluation failure status")
        with self._repositories() as (repository, _):
            batch = repository.get_evaluation(batch_id)
            repository.lock_experiment(batch["experiment_id"])
            batch = repository.get_evaluation(batch_id)
            if batch["status"] == "completed":
                return
            timestamp = now()
            for suite in batch["suites"]:
                if suite["status"] != "completed":
                    repository.update_suite(
                        suite["id"],
                        {"status": status, "error": error, "ended_at": timestamp},
                    )
            repository.update_evaluation(
                batch_id,
                {
                    "status": status,
                    "error": error,
                    "ended_at": timestamp,
                    "heartbeat_at": timestamp,
                },
            )

    def select_best_screened_checkpoint(self, attempt_id, evaluation_ids) -> dict:
        """Select the attempt's best candidate using complete, comparable screens.

        Match score ranks first, mean score difference breaks ties, and candidate
        order resolves exact ties, matching the champion loop's screening order.
        Qualification and champion promotion do not change this selection.
        """
        with self._repositories() as (repository, _):
            attempt = repository.get_attempt(attempt_id)
            repository.lock_experiment(attempt["experiment_id"])
            checkpoints = repository.list_checkpoints(attempt_id)
            candidates = sorted(
                (item for item in checkpoints if item["is_screening_candidate"]),
                key=lambda item: (
                    item["candidate_index"]
                    if item["candidate_index"] is not None
                    else item["save_sequence"],
                    item["save_sequence"],
                ),
            )
            batches = {}
            for evaluation_id in evaluation_ids:
                batch = repository.get_evaluation(evaluation_id)
                if (
                    batch["attempt_id"] != attempt_id
                    or batch["purpose"] != "screening"
                    or batch["opponent_kind"] != "random"
                ):
                    raise AuditError(
                        "Best selection requires this attempt's random screens"
                    )
                if batch["checkpoint_id"] in batches:
                    raise AuditError("Best selection requires one screen per candidate")
                _complete(batch)
                batches[batch["checkpoint_id"]] = batch
            if not candidates or set(batches) != {item["id"] for item in candidates}:
                raise AuditError(
                    "Best selection requires all of the attempt's candidates"
                )
            baseline = batches[candidates[0]["id"]]
            for batch in batches.values():
                _matched_definitions(batch, baseline)

            def rank(checkpoint):
                batch = batches[checkpoint["id"]]
                games = sum(suite["expected_games"] for suite in batch["suites"])
                difference = sum(
                    suite[f"player_{seat}_score_difference_sum"]
                    for suite in batch["suites"]
                    for seat in (0, 1)
                )
                return _match_score(batch), difference / games

            best = max(candidates, key=rank)
            if [item["id"] for item in checkpoints if item["is_best_in_attempt"]] == [
                best["id"]
            ]:
                return best
            repository.clear_checkpoint_selection(attempt_id, "is_best_in_attempt")
            best = repository.update_checkpoint_flags(
                best["id"], {"is_best_in_attempt": True}
            )
            repository.update_attempt(attempt_id, {"updated_at": now()})
            return best

    def record_decision(
        self,
        attempt_id,
        checkpoint_id,
        *,
        stage,
        result,
        reason=None,
        candidate_evaluation_id=None,
        champion_evaluation_id=None,
        policy=None,
        rank=None,
        operation_key=None,
    ) -> dict:
        if stage == "promotion" or result == "promoted":
            raise AuditError("Use promote() to atomically record champion promotion")
        with self._repositories() as (repository, _):
            attempt = repository.get_attempt(attempt_id)
            repository.lock_experiment(attempt["experiment_id"])
            return self._decision(
                repository,
                attempt,
                checkpoint_id,
                stage=stage,
                result=result,
                reason=reason,
                candidate_evaluation_id=candidate_evaluation_id,
                champion_evaluation_id=champion_evaluation_id,
                policy=policy or {},
                rank=rank,
                operation_key=operation_key,
            )

    @staticmethod
    def _decision(
        repository,
        attempt,
        checkpoint_id,
        *,
        stage,
        result,
        reason,
        candidate_evaluation_id,
        champion_evaluation_id,
        policy,
        rank,
        operation_key=None,
    ):
        checkpoint = repository.get_checkpoint(checkpoint_id)
        if checkpoint["attempt_id"] != attempt["id"]:
            raise AuditError("Decision candidate must belong to its attempt")
        candidate = None
        champion = None
        if candidate_evaluation_id:
            candidate = repository.get_evaluation(candidate_evaluation_id)
            if (
                candidate["checkpoint_id"] != checkpoint_id
                or candidate["attempt_id"] != attempt["id"]
            ):
                raise AuditError(
                    "Decision candidate evaluation references a different candidate or attempt"
                )
            _complete(candidate)
        if champion_evaluation_id:
            champion = repository.get_evaluation(champion_evaluation_id)
            _same_experiment(champion, attempt["experiment_id"], "Champion evaluation")
            _complete(champion)
            if attempt["champion_at_start_assignment_id"]:
                assignment = repository.get_assignment(
                    attempt["champion_at_start_assignment_id"]
                )
                if champion["checkpoint_id"] != assignment["checkpoint_id"]:
                    raise AuditError(
                        "Decision baseline differs from the attempt's champion"
                    )
        if stage == "screening" and candidate and champion:
            _matched_definitions(candidate, champion)
            if (
                candidate["opponent_kind"] != "random"
                or champion["opponent_kind"] != "random"
            ):
                raise AuditError("Random screening evidence must use random opponents")
            if result in {"qualified", "passed"}:
                maximum_regression = policy.get("screen_max_regression")
                if (
                    not isinstance(maximum_regression, (int, float))
                    or not math.isfinite(maximum_regression)
                    or maximum_regression < 0
                ):
                    raise AuditError(
                        "Screening policy requires a nonnegative screen_max_regression"
                    )
                if (
                    _match_score(candidate)
                    - _match_score(champion)
                    + maximum_regression
                    < -1e-12
                ):
                    raise AuditError(
                        "Screening evidence does not satisfy the qualification policy"
                    )
        elif stage == "screening" and result in {"qualified", "passed"}:
            raise AuditError(
                "Qualifying screening requires candidate and champion evidence"
            )
        if result in {"qualified", "passed", "promoted"} and not candidate:
            raise AuditError("A passing decision requires complete candidate evidence")
        values = dict(
            attempt_id=attempt["id"],
            checkpoint_id=checkpoint_id,
            stage=stage,
            result=result,
            reason=reason,
            candidate_evaluation_id=candidate_evaluation_id,
            champion_evaluation_id=champion_evaluation_id,
            policy=policy,
            rank=rank,
        )
        key = (
            operation_key
            or f"{attempt['id']}:{checkpoint_id}:{stage}:{candidate_evaluation_id or 'none'}"
        )
        existing = repository.find_decision(key)
        if existing:
            if any(existing[name] != value for name, value in values.items()):
                raise AuditError("Decision operation key reused with different content")
            return existing
        return repository.add_decision(
            dict(id=new_id(), operation_key=key, **values, created_at=now())
        )

    def promote(
        self,
        experiment_id,
        checkpoint_id,
        *,
        attempt_id,
        expected_assignment_id,
        candidate_evaluation_id,
        champion_evaluation_id,
        policy,
    ) -> dict:
        with self._repositories() as (repository, blobs):
            repository.lock_experiment(experiment_id)
            experiment = repository.get_experiment(experiment_id)
            attempt = repository.get_attempt(attempt_id)
            _same_experiment(attempt, experiment_id, "Attempt")
            operation_key = (
                f"promotion:{attempt_id}:{checkpoint_id}:{expected_assignment_id}"
            )
            existing = repository.find_decision(operation_key)
            if existing:
                if (
                    existing["candidate_evaluation_id"] != candidate_evaluation_id
                    or existing["champion_evaluation_id"] != champion_evaluation_id
                    or existing["policy"] != policy
                ):
                    raise AuditError(
                        "Promotion operation reused with different evidence"
                    )
                return next(
                    row
                    for row in repository.list_champions(experiment_id)
                    if row["decision_id"] == existing["id"]
                )
            if experiment["current_champion_assignment_id"] != expected_assignment_id:
                raise PromotionConflict("Champion changed since this attempt began")
            if attempt["champion_at_start_assignment_id"] != expected_assignment_id:
                raise AuditError(
                    "Promotion must challenge the attempt's recorded champion"
                )
            champion = repository.get_assignment(expected_assignment_id)
            challenge = repository.get_evaluation(candidate_evaluation_id)
            _complete(challenge)
            if (
                challenge["checkpoint_id"] != checkpoint_id
                or challenge["opponent_checkpoint_id"] != champion["checkpoint_id"]
                or challenge["attempt_id"] != attempt_id
            ):
                raise AuditError("Promotion challenge has incorrect participants")
            screening = [
                decision
                for decision in repository.list_decisions(attempt_id)
                if decision["checkpoint_id"] == checkpoint_id
                and decision["stage"] == "screening"
                and decision["result"] in {"qualified", "passed"}
                and decision["champion_evaluation_id"] == champion_evaluation_id
            ]
            if not screening:
                raise AuditError(
                    "Promotion requires a durable qualifying screening decision"
                )
            games = wins = draws = suite_wins = 0
            for suite in challenge["suites"]:
                count = sum(
                    suite[f"player_{seat}_{field}"]
                    for seat in (0, 1)
                    for field in ("wins", "draws", "losses")
                )
                won = sum(suite[f"player_{seat}_wins"] for seat in (0, 1))
                drawn = sum(suite[f"player_{seat}_draws"] for seat in (0, 1))
                games += count
                wins += won
                draws += drawn
                suite_wins += (won + 0.5 * drawn) / count > 0.5
            minimum_score = policy.get(
                "promotion_min_match_score", policy.get("minimum_match_score")
            )
            minimum_wins = policy.get(
                "promotion_min_suite_wins", policy.get("minimum_suite_wins")
            )
            if minimum_score is None or minimum_wins is None:
                raise AuditError(
                    "Promotion policy must specify match-score and suite-win thresholds"
                )
            if (
                not isinstance(minimum_score, (int, float))
                or not math.isfinite(minimum_score)
                or not 0.5 < minimum_score <= 1
                or type(minimum_wins) is not int
                or not 1 <= minimum_wins <= len(challenge["suites"])
            ):
                raise AuditError("Promotion policy contains invalid thresholds")
            if (
                wins + 0.5 * draws
            ) / games < minimum_score or suite_wins < minimum_wins:
                raise AuditError(
                    "Challenge evidence does not satisfy the promotion policy"
                )
            checkpoint = repository.get_checkpoint(checkpoint_id)
            blobs.get(checkpoint["checkpoint_blob_id"])
            decision = self._decision(
                repository,
                attempt,
                checkpoint_id,
                stage="promotion",
                result="promoted",
                reason=None,
                candidate_evaluation_id=candidate_evaluation_id,
                champion_evaluation_id=champion_evaluation_id,
                policy=policy,
                rank=None,
                operation_key=operation_key,
            )
            timestamp = now()
            assignment = repository.add_assignment(
                dict(
                    id=new_id(),
                    experiment_id=experiment_id,
                    generation=champion["generation"] + 1,
                    checkpoint_id=checkpoint_id,
                    previous_assignment_id=expected_assignment_id,
                    attempt_id=attempt_id,
                    decision_id=decision["id"],
                    reason="promotion",
                    created_at=timestamp,
                )
            )
            if not repository.compare_and_set_champion(
                experiment_id, expected_assignment_id, assignment["id"], timestamp
            ):
                raise PromotionConflict("Champion changed during promotion")
            repository.update_attempt(
                attempt_id,
                dict(
                    status="completed",
                    phase="finished",
                    outcome="promoted",
                    ended_at=timestamp,
                    updated_at=timestamp,
                    heartbeat_at=timestamp,
                ),
            )
            return assignment
