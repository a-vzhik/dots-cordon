"""Bounded SQL projections. Metadata queries never select checkpoint payloads."""

import base64
from datetime import datetime, timezone
import hashlib
import json
import math

import sqlalchemy as sa

from . import schema as s
from .database import InvalidCursor, RecordNotFound
from .repository import AuditRepository, CheckpointBlobRepository, _row


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def cursor_data(cursor):
    try:
        if len(cursor) > 4096:
            raise ValueError()
        value = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
        if not isinstance(value, dict) or value["version"] != 1:
            raise ValueError()
        timestamp = datetime.fromisoformat(value["snapshot_at"])
        if timestamp.tzinfo is None:
            raise ValueError()
        return value
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise InvalidCursor("Invalid pagination cursor") from exc


def encode_cursor(value):
    return base64.urlsafe_b64encode(
        json.dumps(value, separators=(",", ":")).encode()
    ).decode()


def scope_hash(scope):
    return hashlib.sha256(
        json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def lexicographic(columns, values, *, after):
    """Portable tuple comparison, including databases without row-value syntax."""
    terms = []
    for index, (column, value) in enumerate(zip(columns, values, strict=True)):
        comparison = column > value if after else column < value
        terms.append(
            sa.and_(*(columns[i] == values[i] for i in range(index)), comparison)
        )
    if not after:
        terms.append(
            sa.and_(
                *(
                    column == value
                    for column, value in zip(columns, values, strict=True)
                )
            )
        )
    return sa.or_(*terms)


class AuditReadRepository(AuditRepository):
    def __init__(self, connection):
        super().__init__(connection)
        self.snapshot_at = utc_now()

    def rows(self, statement):
        return [_row(row) for row in self.connection.execute(statement).mappings()]

    def required(self, table, identifier):
        if isinstance(table, str):
            table = getattr(s, table)
        row = (
            self.connection.execute(sa.select(table).where(table.c.id == identifier))
            .mappings()
            .first()
        )
        if row is None:
            raise RecordNotFound(f"Unknown {table.name} ID: {identifier}")
        return _row(row)

    def experiment(self, identifier):
        row = (
            self.connection.execute(
                sa.select(s.experiments).where(
                    sa.or_(
                        s.experiments.c.id == identifier,
                        s.experiments.c.name == identifier,
                    )
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            raise RecordNotFound(f"Unknown experiment: {identifier}")
        return _row(row)

    def set_snapshot(self, *cursors):
        snapshots = {cursor_data(cursor)["snapshot_at"] for cursor in cursors if cursor}
        if len(snapshots) > 1:
            raise InvalidCursor("Nested cursors must belong to the same snapshot")
        if snapshots:
            self.snapshot_at = snapshots.pop()

    def page(self, table, filters, *, limit, cursor, scope, order=None):
        order = order or [table.c.created_at, table.c.id]
        filters = list(filters)
        signature = scope_hash(scope)
        if cursor:
            state = cursor_data(cursor)
            if state.get("scope") != signature:
                raise InvalidCursor("Cursor does not belong to this query")
            for key in ("upper", "after"):
                values = state.get(key)
                if not isinstance(values, list) or len(values) != len(order):
                    raise InvalidCursor("Invalid pagination position")
                if any(
                    type(value) is not column.type.python_type
                    for column, value in zip(order, values, strict=True)
                ):
                    raise InvalidCursor("Invalid pagination position")
            self.snapshot_at = state["snapshot_at"]
        else:
            state = {"version": 1, "scope": signature, "snapshot_at": self.snapshot_at}
        if "created_at" in table.c:
            filters.append(table.c.created_at <= self.snapshot_at)
        if not cursor:
            upper = self.connection.execute(
                sa.select(*order)
                .where(*filters)
                .order_by(*(column.desc() for column in order))
                .limit(1)
            ).first()
            if upper is None:
                return dict(
                    items=[],
                    total=0,
                    remaining=0,
                    limit=limit,
                    next_cursor=None,
                    snapshot_at=self.snapshot_at,
                )
            state["upper"] = list(upper)
        filters.append(lexicographic(order, state["upper"], after=False))
        total = self.connection.execute(
            sa.select(sa.func.count()).select_from(table).where(*filters)
        ).scalar_one()
        if cursor:
            filters.append(lexicographic(order, state["after"], after=True))
        remaining = self.connection.execute(
            sa.select(sa.func.count()).select_from(table).where(*filters)
        ).scalar_one()
        items = self.rows(
            sa.select(table).where(*filters).order_by(*order).limit(limit + 1)
        )
        more = len(items) > limit
        items = items[:limit]
        next_cursor = None
        if more:
            state["after"] = [items[-1][column.name] for column in order]
            next_cursor = encode_cursor(state)
        return dict(
            items=items,
            total=total,
            remaining=max(0, remaining - len(items)),
            limit=limit,
            next_cursor=next_cursor,
            snapshot_at=self.snapshot_at,
        )

    def assignments(self, ids=(), checkpoint_ids=()):
        if not ids and not checkpoint_ids:
            return []
        return self.rows(
            sa.select(s.champion_history)
            .where(
                sa.or_(
                    s.champion_history.c.id.in_(ids),
                    s.champion_history.c.checkpoint_id.in_(checkpoint_ids),
                ),
            )
            .order_by(s.champion_history.c.generation)
        )

    def checkpoints(self, ids):
        if not ids:
            return []
        return self.rows(sa.select(s.checkpoints).where(s.checkpoints.c.id.in_(ids)))

    def blobs(self, ids):
        columns = [
            column for column in s.checkpoint_blobs.c if column.name != "payload"
        ]
        return (
            self.rows(sa.select(*columns).where(s.checkpoint_blobs.c.id.in_(ids)))
            if ids
            else []
        )

    def counts(self, experiment_id, starting_checkpoint_id=None):
        attempts = s.attempts
        attempt_filters = [
            attempts.c.experiment_id == experiment_id,
            attempts.c.created_at <= self.snapshot_at,
        ]
        if starting_checkpoint_id is not None:
            attempt_filters.append(
                attempts.c.starting_checkpoint_id == starting_checkpoint_id
            )
        attempt_ids = sa.select(attempts.c.id).where(*attempt_filters)
        qualified = (
            sa.select(sa.func.count(sa.distinct(s.decisions.c.checkpoint_id)))
            .where(
                s.decisions.c.attempt_id.in_(attempt_ids),
                s.decisions.c.stage == "screening",
                s.decisions.c.result.in_(["qualified", "passed"]),
                s.decisions.c.created_at <= self.snapshot_at,
            )
            .scalar_subquery()
        )
        challenges = (
            sa.select(sa.func.count())
            .select_from(s.evaluation_batches)
            .where(
                s.evaluation_batches.c.attempt_id.in_(attempt_ids),
                s.evaluation_batches.c.purpose == "head_to_head",
                s.evaluation_batches.c.created_at <= self.snapshot_at,
            )
            .scalar_subquery()
        )
        promotions = (
            sa.select(sa.func.count())
            .select_from(s.champion_history)
            .where(
                s.champion_history.c.attempt_id.in_(attempt_ids),
                s.champion_history.c.created_at <= self.snapshot_at,
            )
            .scalar_subquery()
        )
        return dict(
            self.connection.execute(
                sa.select(
                    sa.select(sa.func.count())
                    .select_from(attempts)
                    .where(*attempt_filters)
                    .scalar_subquery()
                    .label("attempts"),
                    qualified.label("qualified_checkpoints"),
                    challenges.label("challenge_batches"),
                    promotions.label("promotions"),
                )
            )
            .mappings()
            .one()
        )

    def experiment_counts(self, ids):
        output = {
            identifier: dict(attempts=0, checkpoints=0, evaluation_batches=0)
            for identifier in ids
        }
        for table, name in [
            (s.attempts, "attempts"),
            (s.checkpoints, "checkpoints"),
            (s.evaluation_batches, "evaluation_batches"),
        ]:
            for row in self.connection.execute(
                sa.select(table.c.experiment_id, sa.func.count().label("n"))
                .where(
                    table.c.experiment_id.in_(ids),
                    table.c.created_at <= self.snapshot_at,
                )
                .group_by(table.c.experiment_id)
            ).mappings():
                output[row["experiment_id"]][name] = row["n"]
        return output

    def attempt_counts(self, ids):
        output = {
            identifier: dict(
                checkpoints=0,
                qualified_checkpoints=0,
                challenge_batches=0,
                promotions=0,
            )
            for identifier in ids
        }
        queries = [
            (s.checkpoints, "checkpoints", [], sa.func.count()),
            (
                s.decisions,
                "qualified_checkpoints",
                [
                    s.decisions.c.stage == "screening",
                    s.decisions.c.result.in_(["qualified", "passed"]),
                ],
                sa.func.count(sa.distinct(s.decisions.c.checkpoint_id)),
            ),
            (
                s.evaluation_batches,
                "challenge_batches",
                [s.evaluation_batches.c.purpose == "head_to_head"],
                sa.func.count(),
            ),
            (s.champion_history, "promotions", [], sa.func.count()),
        ]
        for table, name, filters, count in queries:
            for row in self.connection.execute(
                sa.select(table.c.attempt_id, count.label("n"))
                .where(
                    table.c.attempt_id.in_(ids),
                    table.c.created_at <= self.snapshot_at,
                    *filters,
                )
                .group_by(table.c.attempt_id)
            ).mappings():
                output[row["attempt_id"]][name] = row["n"]
        return output

    def attempt_activity(self, ids):
        return dict(
            self.connection.execute(
                sa.select(
                    s.evaluation_batches.c.attempt_id,
                    sa.func.max(s.evaluation_batches.c.heartbeat_at),
                )
                .where(
                    s.evaluation_batches.c.attempt_id.in_(ids),
                    s.evaluation_batches.c.created_at <= self.snapshot_at,
                )
                .group_by(s.evaluation_batches.c.attempt_id)
            ).all()
        )

    def checkpoint_counts(self, ids):
        output = {
            identifier: dict(
                child_checkpoints=0,
                attempts_from_checkpoint=0,
                evaluation_batches=0,
                decisions=0,
            )
            for identifier in ids
        }
        for table, key, name in [
            (s.checkpoints, s.checkpoints.c.parent_checkpoint_id, "child_checkpoints"),
            (
                s.attempts,
                s.attempts.c.starting_checkpoint_id,
                "attempts_from_checkpoint",
            ),
            (
                s.evaluation_batches,
                s.evaluation_batches.c.checkpoint_id,
                "evaluation_batches",
            ),
            (s.decisions, s.decisions.c.checkpoint_id, "decisions"),
        ]:
            for row in self.connection.execute(
                sa.select(key.label("key"), sa.func.count().label("n"))
                .where(
                    key.in_(ids),
                    table.c.created_at <= self.snapshot_at,
                )
                .group_by(key)
            ).mappings():
                output[row["key"]][name] = row["n"]
        return output

    def recent_per_checkpoint(self, table, ids, count):
        if isinstance(table, str):
            table = getattr(s, table)
        ranked = (
            sa.select(
                table.c.id,
                sa.func.row_number()
                .over(
                    partition_by=table.c.checkpoint_id,
                    order_by=(table.c.created_at.desc(), table.c.id.desc()),
                )
                .label("position"),
            )
            .where(
                table.c.checkpoint_id.in_(ids), table.c.created_at <= self.snapshot_at
            )
            .subquery()
        )
        return self.rows(
            sa.select(table)
            .join(ranked, table.c.id == ranked.c.id)
            .where(ranked.c.position <= count)
            .order_by(table.c.created_at.desc(), table.c.id.desc())
        )

    def batch_statistics(self, ids):
        suite = s.evaluation_suites
        fields = [
            sa.func.count().label("recorded_suites"),
            sa.func.sum(suite.c.expected_games).label("expected_games"),
            sa.func.sum(sa.case((suite.c.status == "completed", 1), else_=0)).label(
                "completed_suites"
            ),
        ]
        for seat in (0, 1):
            for field in ("wins", "draws", "losses", "score_difference_sum"):
                name = f"player_{seat}_{field}"
                fields.append(
                    sa.func.sum(
                        sa.case(
                            (suite.c.status == "completed", suite.c[name]), else_=None
                        )
                    ).label(name)
                )
        return {
            row["batch_id"]: dict(row)
            for row in self.connection.execute(
                sa.select(suite.c.batch_id, *fields)
                .where(suite.c.batch_id.in_(ids))
                .group_by(suite.c.batch_id)
            ).mappings()
        }

    def evaluation_decisions(self, batch_id, limit=100, cursor=None):
        return self.page(
            s.decisions,
            [
                sa.or_(
                    s.decisions.c.candidate_evaluation_id == batch_id,
                    s.decisions.c.champion_evaluation_id == batch_id,
                )
            ],
            limit=limit,
            cursor=cursor,
            scope=["evaluation_decisions", batch_id],
        )

    def metrics(self, attempt_id, episode_min, episode_max, max_points):
        table = s.training_metrics
        filters = [
            table.c.attempt_id == attempt_id,
            table.c.created_at <= self.snapshot_at,
        ]
        if episode_min is not None:
            filters.append(table.c.episode >= episode_min)
        if episode_max is not None:
            filters.append(table.c.episode <= episode_max)
        count = self.connection.execute(
            sa.select(sa.func.count()).select_from(table).where(*filters)
        ).scalar_one()
        stride = max(1, math.ceil((count - 1) / (max_points - 1)))
        ranked = (
            sa.select(
                table,
                (sa.func.row_number().over(order_by=table.c.sample_sequence) - 1).label(
                    "sample_index"
                ),
            )
            .where(*filters)
            .subquery()
        )
        rows = self.rows(
            sa.select(ranked)
            .where(
                sa.or_(
                    ranked.c.sample_index % stride == 0,
                    ranked.c.sample_index == count - 1,
                )
            )
            .order_by(ranked.c.sample_sequence)
        )
        for row in rows:
            row.pop("sample_index")
        return dict(
            items=rows,
            total_samples=count,
            omitted_samples=count - len(rows),
            stride=stride,
            method="uniform_samples",
            snapshot_at=self.snapshot_at,
        )

    def descendants(self, root_id):
        descendants = (
            sa.select(s.checkpoints.c.id)
            .where(s.checkpoints.c.id == root_id)
            .cte("descendants", recursive=True)
        )
        return descendants.union_all(
            sa.select(s.checkpoints.c.id)
            .join(
                descendants,
                s.checkpoints.c.parent_checkpoint_id == descendants.c.id,
            )
            .where(s.checkpoints.c.created_at <= self.snapshot_at)
        )

    def payload(self, blob_id):
        return CheckpointBlobRepository(self.connection).get(blob_id)

    def experiments_page(self, limit, cursor):
        return self.page(
            s.experiments, [], limit=limit, cursor=cursor, scope=["experiments"]
        )

    def attempts_page(
        self,
        *,
        experiment_id=None,
        starting_checkpoint_id=None,
        status=None,
        phase=None,
        outcome=None,
        created_after=None,
        created_before=None,
        limit=50,
        cursor=None,
    ):
        scope = dict(
            experiment_id=experiment_id,
            starting_checkpoint_id=starting_checkpoint_id,
            status=status,
            phase=phase,
            outcome=outcome,
            created_after=created_after,
            created_before=created_before,
        )
        filters = [
            s.attempts.c[name] == value
            for name, value in scope.items()
            if value is not None and name not in {"created_after", "created_before"}
        ]
        if created_after:
            filters.append(s.attempts.c.created_at >= created_after)
        if created_before:
            filters.append(s.attempts.c.created_at <= created_before)
        return self.page(
            s.attempts, filters, limit=limit, cursor=cursor, scope=["attempts", scope]
        )

    def checkpoints_page(self, attempt_id, limit, cursor):
        return self.page(
            s.checkpoints,
            [s.checkpoints.c.attempt_id == attempt_id],
            limit=limit,
            cursor=cursor,
            scope=["attempt_checkpoints", attempt_id],
        )

    def evaluations_page(self, checkpoint_id, limit, cursor):
        return self.page(
            s.evaluation_batches,
            [s.evaluation_batches.c.checkpoint_id == checkpoint_id],
            limit=limit,
            cursor=cursor,
            scope=["checkpoint_evaluations", checkpoint_id],
        )

    def suites_page(self, batch_id, limit, cursor):
        return self.page(
            s.evaluation_suites,
            [s.evaluation_suites.c.batch_id == batch_id],
            limit=limit,
            cursor=cursor,
            scope=["evaluation_suites", batch_id],
            order=[s.evaluation_suites.c.suite_index, s.evaluation_suites.c.id],
        )

    def lineage_pages(
        self,
        experiment_id,
        root,
        *,
        episode_min,
        episode_max,
        attempt_limit,
        attempt_cursor,
        checkpoint_limit,
        checkpoint_cursor,
    ):
        self.set_snapshot(attempt_cursor, checkpoint_cursor)
        scope = dict(
            experiment_id=experiment_id,
            root_checkpoint_id=root["id"] if root else None,
            episode_min=episode_min,
            episode_max=episode_max,
        )
        filters = [s.attempts.c.experiment_id == experiment_id]
        descendants = self.descendants(root["id"]) if root else None
        if descendants is not None:
            filters.append(
                sa.or_(
                    s.attempts.c.starting_checkpoint_id.in_(
                        sa.select(descendants.c.id)
                    ),
                    s.attempts.c.id == root["attempt_id"]
                    if root["attempt_id"]
                    else sa.false(),
                )
            )
        if episode_min is not None:
            filters.append(s.attempts.c.latest_episode >= episode_min)
        if episode_max is not None:
            filters.append(s.attempts.c.start_episode <= episode_max)
        attempts = self.page(
            s.attempts,
            filters,
            limit=attempt_limit,
            cursor=attempt_cursor,
            scope=["lineage_attempts", scope],
        )
        attempt_ids = [item["id"] for item in attempts["items"]]
        nodes = s.checkpoints
        filters = [
            nodes.c.experiment_id == experiment_id,
            sa.or_(
                nodes.c.attempt_id.in_(attempt_ids),
                nodes.c.id == root["id"] if root else nodes.c.attempt_id.is_(None),
            ),
        ]
        if descendants is not None:
            filters.append(nodes.c.id.in_(sa.select(descendants.c.id)))
        if episode_min is not None:
            filters.append(nodes.c.episode >= episode_min)
        if episode_max is not None:
            filters.append(nodes.c.episode <= episode_max)
        checkpoints = self.page(
            nodes,
            filters,
            limit=checkpoint_limit,
            cursor=checkpoint_cursor,
            scope=["lineage_checkpoints", scope, attempt_ids],
        )
        return attempts, checkpoints
