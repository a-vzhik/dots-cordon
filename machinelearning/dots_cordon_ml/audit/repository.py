"""SQL implementation details; repositories never commit independently."""

import hashlib
import json

import sqlalchemy as sa

from . import schema as s
from .database import AuditError


JSON_FIELDS = {
    "game_config",
    "seed_offsets",
    "config",
    "board",
    "model",
    "metrics",
    "definition",
    "policy",
}


def encode(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


def _values(values):
    return {
        key: encode(value) if key in JSON_FIELDS else value
        for key, value in values.items()
    }


def _row(row):
    if row is None:
        return None
    return {
        key: json.loads(value) if key in JSON_FIELDS and value is not None else value
        for key, value in row.items()
    }


class AuditRepository:
    def __init__(self, connection):
        self.connection = connection

    def _get(self, table, identifier):
        result = (
            self.connection.execute(sa.select(table).where(table.c.id == identifier))
            .mappings()
            .first()
        )
        if result is None:
            raise AuditError(f"Unknown {table.name} ID: {identifier}")
        return _row(result)

    def _insert(self, table, values):
        self.connection.execute(table.insert().values(**_values(values)))
        return self._get(table, values["id"])

    def _update(self, table, identifier, values):
        if not values:
            return self._get(table, identifier)
        self.connection.execute(
            table.update().where(table.c.id == identifier).values(**_values(values))
        )
        return self._get(table, identifier)

    def _list(self, table, *filters, order_by=None):
        query = sa.select(table).where(*filters)
        if order_by is not None:
            query = query.order_by(*order_by)
        return [_row(row) for row in self.connection.execute(query).mappings()]

    def lock_experiment(self, experiment_id):
        # Acquires the SQL row/write lock before subsequent reads, including on SQLite.
        result = self.connection.execute(
            s.experiments.update()
            .where(s.experiments.c.id == experiment_id)
            .values(version=s.experiments.c.version + 1)
        )
        if result.rowcount != 1:
            raise AuditError(f"Unknown experiment ID: {experiment_id}")

    def get_experiment(self, identifier):
        rows = self._list(
            s.experiments,
            sa.or_(
                s.experiments.c.id == identifier, s.experiments.c.name == identifier
            ),
        )
        if not rows:
            raise AuditError(f"Unknown experiment: {identifier}")
        return rows[0]

    def find_experiment(self, name):
        rows = self._list(s.experiments, s.experiments.c.name == name)
        return rows[0] if rows else None

    def list_experiments(self):
        return self._list(s.experiments, order_by=[s.experiments.c.created_at])

    def add_experiment(self, values):
        return self._insert(s.experiments, values)

    def update_experiment(self, identifier, values):
        return self._update(s.experiments, identifier, values)

    def compare_and_set_champion(
        self, experiment_id, expected, replacement, updated_at
    ):
        result = self.connection.execute(
            s.experiments.update()
            .where(
                s.experiments.c.id == experiment_id,
                s.experiments.c.current_champion_assignment_id == expected,
            )
            .values(current_champion_assignment_id=replacement, updated_at=updated_at)
        )
        return result.rowcount == 1

    def get_attempt(self, identifier):
        return self._get(s.attempts, identifier)

    def list_attempts(self, experiment_id):
        return self._list(
            s.attempts,
            s.attempts.c.experiment_id == experiment_id,
            order_by=[s.attempts.c.created_at, s.attempts.c.id],
        )

    def add_attempt(self, values):
        return self._insert(s.attempts, values)

    def update_attempt(self, identifier, values):
        return self._update(s.attempts, identifier, values)

    def get_checkpoint(self, identifier):
        return self._get(s.checkpoints, identifier)

    def find_checkpoint_blob(self, experiment_id, blob_id, attempt_id):
        filters = [
            s.checkpoints.c.experiment_id == experiment_id,
            s.checkpoints.c.checkpoint_blob_id == blob_id,
        ]
        if attempt_id is not None:
            filters.append(s.checkpoints.c.attempt_id == attempt_id)
        rows = self._list(
            s.checkpoints, *filters, order_by=[s.checkpoints.c.created_at]
        )
        return rows[0] if rows else None

    def list_checkpoints(self, attempt_id):
        return self._list(
            s.checkpoints,
            s.checkpoints.c.attempt_id == attempt_id,
            order_by=[s.checkpoints.c.save_sequence],
        )

    def add_checkpoint(self, values):
        return self._insert(s.checkpoints, values)

    def update_checkpoint_flags(self, identifier, values):
        return self._update(s.checkpoints, identifier, values)

    def clear_checkpoint_selection(self, attempt_id, flag):
        if flag not in {"is_best_in_attempt", "is_final_in_attempt"}:
            raise ValueError("Invalid selection flag")
        self.connection.execute(
            s.checkpoints.update()
            .where(s.checkpoints.c.attempt_id == attempt_id)
            .values(**{flag: False})
        )

    def list_metrics(self, attempt_id):
        return self._list(
            s.training_metrics,
            s.training_metrics.c.attempt_id == attempt_id,
            order_by=[s.training_metrics.c.sample_sequence],
        )

    def add_metrics(self, values):
        self.connection.execute(s.training_metrics.insert().values(**_values(values)))

    def get_evaluation(self, identifier):
        result = self._get(s.evaluation_batches, identifier)
        result["suites"] = self.list_suites(identifier)
        return result

    def list_evaluations(self, experiment_id, checkpoint_id=None):
        filters = [s.evaluation_batches.c.experiment_id == experiment_id]
        if checkpoint_id is not None:
            filters.append(s.evaluation_batches.c.checkpoint_id == checkpoint_id)
        return self._list(
            s.evaluation_batches, *filters, order_by=[s.evaluation_batches.c.created_at]
        )

    def add_evaluation(self, values):
        return self._insert(s.evaluation_batches, values)

    def update_evaluation(self, identifier, values):
        return self._update(s.evaluation_batches, identifier, values)

    def list_suites(self, batch_id):
        return self._list(
            s.evaluation_suites,
            s.evaluation_suites.c.batch_id == batch_id,
            order_by=[s.evaluation_suites.c.suite_index],
        )

    def add_suite(self, values):
        return self._insert(s.evaluation_suites, values)

    def update_suite(self, identifier, values):
        return self._update(s.evaluation_suites, identifier, values)

    def find_decision(self, operation_key):
        rows = self._list(s.decisions, s.decisions.c.operation_key == operation_key)
        return rows[0] if rows else None

    def add_decision(self, values):
        return self._insert(s.decisions, values)

    def list_decisions(self, attempt_id):
        return self._list(
            s.decisions,
            s.decisions.c.attempt_id == attempt_id,
            order_by=[s.decisions.c.created_at],
        )

    def get_assignment(self, identifier):
        return self._get(s.champion_history, identifier)

    def add_assignment(self, values):
        return self._insert(s.champion_history, values)

    def list_champions(self, experiment_id):
        return self._list(
            s.champion_history,
            s.champion_history.c.experiment_id == experiment_id,
            order_by=[s.champion_history.c.generation],
        )


class CheckpointBlobRepository:
    def __init__(self, connection):
        self.connection = connection

    def put(self, payload: bytes, *, identifier, created_at, format_version):
        digest = hashlib.sha256(payload).hexdigest()
        table = s.checkpoint_blobs
        columns = [column for column in table.c if column.name != "payload"]
        existing = (
            self.connection.execute(sa.select(*columns).where(table.c.sha256 == digest))
            .mappings()
            .first()
        )
        if existing:
            if existing["byte_length"] != len(payload):
                raise AuditError("Checkpoint blob hash collision or corrupt length")
            return dict(existing)
        values = dict(
            id=identifier,
            sha256=digest,
            byte_length=len(payload),
            format="pytorch",
            format_version=format_version,
            payload=payload,
            created_at=created_at,
        )
        self.connection.execute(table.insert().values(**values))
        values.pop("payload")
        return values

    def get(self, identifier):
        table = s.checkpoint_blobs
        row = (
            self.connection.execute(sa.select(table).where(table.c.id == identifier))
            .mappings()
            .first()
        )
        if row is None:
            raise AuditError(f"Unknown checkpoint blob: {identifier}")
        payload = bytes(row["payload"])
        if (
            len(payload) != row["byte_length"]
            or hashlib.sha256(payload).hexdigest() != row["sha256"]
        ):
            raise AuditError("Checkpoint blob checksum verification failed")
        return payload
