"""Portable relational schema. Schema changes require an Alembic revision."""

import sqlalchemy as sa


metadata = sa.MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)


def identity():
    return sa.Column("id", sa.String(36), primary_key=True)


def reference(name, target, nullable=True):
    return sa.Column(name, sa.String(36), sa.ForeignKey(target), nullable=nullable)


def timestamp(name, nullable=False):
    # Explicit UTC ISO 8601 avoids SQLite's timezone-stripping datetime adapter.
    return sa.Column(name, sa.String(32), nullable=nullable)


experiments = sa.Table(
    "experiments",
    metadata,
    identity(),
    sa.Column("name", sa.String(255), nullable=False, unique=True),
    sa.Column("game_config", sa.Text, nullable=False),
    sa.Column("game_config_fingerprint", sa.String(64), nullable=False),
    reference("current_champion_assignment_id", "champion_history.id"),
    sa.Column("seed_offsets", sa.Text, nullable=False),
    sa.Column("version", sa.Integer, nullable=False),
    timestamp("created_at"),
    timestamp("updated_at"),
)

attempts = sa.Table(
    "attempts",
    metadata,
    identity(),
    reference("experiment_id", "experiments.id", False),
    reference("starting_checkpoint_id", "checkpoints.id"),
    reference("champion_at_start_assignment_id", "champion_history.id"),
    sa.Column("config", sa.Text, nullable=False),
    sa.Column("start_episode", sa.Integer, nullable=False),
    sa.Column("target_episode", sa.Integer, nullable=False),
    sa.Column("latest_episode", sa.Integer, nullable=False),
    sa.Column("phase", sa.String(32), nullable=False),
    sa.Column("status", sa.String(32), nullable=False),
    sa.Column("outcome", sa.String(64)),
    sa.Column("error", sa.Text),
    sa.Column("stop_reason", sa.Text),
    sa.Column("version", sa.Integer, nullable=False),
    timestamp("created_at"),
    timestamp("updated_at"),
    timestamp("started_at"),
    timestamp("ended_at", True),
    timestamp("heartbeat_at"),
    timestamp("last_interrupted_at", True),
    timestamp("last_resumed_at", True),
    timestamp("error_at", True),
    sa.CheckConstraint(
        "start_episode >= 0 AND target_episode >= start_episode AND latest_episode >= start_episode",
        name="episodes",
    ),
    sa.Index("ix_attempts_experiment_created", "experiment_id", "created_at"),
    sa.Index("ix_attempts_starting_checkpoint", "starting_checkpoint_id"),
)

checkpoint_blobs = sa.Table(
    "checkpoint_blobs",
    metadata,
    identity(),
    sa.Column("sha256", sa.String(64), nullable=False, unique=True),
    sa.Column("byte_length", sa.BigInteger, nullable=False),
    sa.Column("format", sa.String(64), nullable=False),
    sa.Column("format_version", sa.Integer, nullable=False),
    sa.Column("payload", sa.LargeBinary, nullable=False),
    timestamp("created_at"),
    sa.CheckConstraint("byte_length > 0", name="nonempty"),
)

checkpoints = sa.Table(
    "checkpoints",
    metadata,
    identity(),
    reference("experiment_id", "experiments.id", False),
    reference("attempt_id", "attempts.id"),
    reference("checkpoint_blob_id", "checkpoint_blobs.id", False),
    reference("parent_checkpoint_id", "checkpoints.id"),
    sa.Column("episode", sa.Integer, nullable=False),
    sa.Column("environment_steps", sa.BigInteger, nullable=False),
    sa.Column("optimization_steps", sa.BigInteger, nullable=False),
    sa.Column("board", sa.Text, nullable=False),
    sa.Column("model", sa.Text, nullable=False),
    sa.Column("origin", sa.String(32), nullable=False),
    sa.Column("save_sequence", sa.Integer, nullable=False),
    sa.Column("candidate_index", sa.Integer),
    sa.Column("is_periodic_save", sa.Boolean, nullable=False),
    sa.Column("is_screening_candidate", sa.Boolean, nullable=False),
    sa.Column("is_best_in_attempt", sa.Boolean, nullable=False),
    sa.Column("is_final_in_attempt", sa.Boolean, nullable=False),
    timestamp("created_at"),
    sa.CheckConstraint(
        "episode >= 0 AND environment_steps >= 0 AND optimization_steps >= 0",
        name="counters",
    ),
    sa.CheckConstraint(
        "parent_checkpoint_id IS NULL OR parent_checkpoint_id <> id",
        name="not_self_parent",
    ),
    sa.UniqueConstraint("attempt_id", "save_sequence"),
    sa.Index("ix_checkpoints_attempt_episode", "attempt_id", "episode"),
    sa.Index("ix_checkpoints_parent", "parent_checkpoint_id"),
    sa.Index("ix_checkpoints_experiment_created", "experiment_id", "created_at"),
)

training_metrics = sa.Table(
    "training_metrics",
    metadata,
    reference("attempt_id", "attempts.id", False),
    sa.Column("sample_sequence", sa.Integer, nullable=False),
    sa.Column("episode", sa.Integer, nullable=False),
    sa.Column("metrics", sa.Text, nullable=False),
    timestamp("created_at"),
    sa.PrimaryKeyConstraint("attempt_id", "sample_sequence"),
)

evaluation_batches = sa.Table(
    "evaluation_batches",
    metadata,
    identity(),
    reference("experiment_id", "experiments.id", False),
    reference("attempt_id", "attempts.id"),
    reference("checkpoint_id", "checkpoints.id", False),
    reference("opponent_checkpoint_id", "checkpoints.id"),
    sa.Column("purpose", sa.String(32), nullable=False),
    sa.Column("opponent_kind", sa.String(32), nullable=False),
    sa.Column("config", sa.Text, nullable=False),
    sa.Column("status", sa.String(32), nullable=False),
    sa.Column("planned_suite_count", sa.Integer, nullable=False),
    sa.Column("error", sa.Text),
    timestamp("created_at"),
    timestamp("started_at"),
    timestamp("heartbeat_at"),
    timestamp("ended_at", True),
    sa.CheckConstraint("planned_suite_count > 0", name="suite_count"),
    sa.Index("ix_evaluation_batches_checkpoint_purpose", "checkpoint_id", "purpose"),
)

evaluation_suites = sa.Table(
    "evaluation_suites",
    metadata,
    identity(),
    reference("batch_id", "evaluation_batches.id", False),
    sa.Column("suite_index", sa.Integer, nullable=False),
    sa.Column("definition", sa.Text, nullable=False),
    sa.Column("suite_fingerprint", sa.String(64), nullable=False, index=True),
    sa.Column("expected_games", sa.Integer, nullable=False),
    sa.Column("expected_player_0_games", sa.Integer, nullable=False),
    sa.Column("expected_player_1_games", sa.Integer, nullable=False),
    sa.Column("status", sa.String(32), nullable=False),
    *[
        sa.Column(f"player_{seat}_{field}", sa.BigInteger)
        for seat in (0, 1)
        for field in ("wins", "draws", "losses", "score_difference_sum")
    ],
    sa.Column("elapsed_seconds", sa.Float),
    sa.Column("error", sa.Text),
    timestamp("started_at"),
    timestamp("ended_at", True),
    sa.UniqueConstraint("batch_id", "suite_index"),
    sa.CheckConstraint(
        "expected_games > 0 AND expected_player_0_games >= 0 AND expected_player_1_games >= 0 AND expected_player_0_games + expected_player_1_games = expected_games",
        name="expected_counts",
    ),
    *[
        sa.CheckConstraint(
            f"player_{seat}_{field} IS NULL OR player_{seat}_{field} >= 0",
            name=f"nonnegative_{seat}_{field}",
        )
        for seat in (0, 1)
        for field in ("wins", "draws", "losses")
    ],
    *[
        sa.CheckConstraint(
            f"status <> 'completed' OR (player_{seat}_wins IS NOT NULL AND player_{seat}_draws IS NOT NULL AND player_{seat}_losses IS NOT NULL AND player_{seat}_score_difference_sum IS NOT NULL AND player_{seat}_wins + player_{seat}_draws + player_{seat}_losses = expected_player_{seat}_games)",
            name=f"completed_seat_{seat}",
        )
        for seat in (0, 1)
    ],
)

decisions = sa.Table(
    "decisions",
    metadata,
    identity(),
    sa.Column("operation_key", sa.String(255), nullable=False, unique=True),
    reference("attempt_id", "attempts.id", False),
    reference("checkpoint_id", "checkpoints.id", False),
    reference("candidate_evaluation_id", "evaluation_batches.id"),
    reference("champion_evaluation_id", "evaluation_batches.id"),
    sa.Column("stage", sa.String(32), nullable=False),
    sa.Column("result", sa.String(64), nullable=False),
    sa.Column("reason", sa.Text),
    sa.Column("policy", sa.Text, nullable=False),
    sa.Column("rank", sa.Integer),
    timestamp("created_at"),
)

champion_history = sa.Table(
    "champion_history",
    metadata,
    identity(),
    reference("experiment_id", "experiments.id", False),
    sa.Column("generation", sa.Integer, nullable=False),
    reference("checkpoint_id", "checkpoints.id", False),
    reference("previous_assignment_id", "champion_history.id"),
    reference("attempt_id", "attempts.id"),
    reference("decision_id", "decisions.id"),
    sa.Column("reason", sa.String(32), nullable=False),
    timestamp("created_at"),
    sa.UniqueConstraint("experiment_id", "generation"),
    sa.UniqueConstraint("decision_id"),
)
