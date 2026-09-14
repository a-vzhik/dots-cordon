"""Initial nine-table training audit schema.

Revision ID: 0001_training_audit
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_training_audit"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # SQLite supports forward foreign-key references. Other dialects need the
    # cyclic lineage/champion constraints attached after all tables exist.
    deferred = []

    def create_table(name, *elements):
        if op.get_bind().dialect.name != "sqlite":
            constraints = [
                item for item in elements if isinstance(item, sa.ForeignKeyConstraint)
            ]
            deferred.extend((name, item) for item in constraints)
            elements = tuple(
                item
                for item in elements
                if not isinstance(item, sa.ForeignKeyConstraint)
            )
        return op.create_table(name, *elements)

    create_table(
        "attempts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("starting_checkpoint_id", sa.String(length=36), nullable=True),
        sa.Column(
            "champion_at_start_assignment_id", sa.String(length=36), nullable=True
        ),
        sa.Column("config", sa.Text(), nullable=False),
        sa.Column("start_episode", sa.Integer(), nullable=False),
        sa.Column("target_episode", sa.Integer(), nullable=False),
        sa.Column("latest_episode", sa.Integer(), nullable=False),
        sa.Column("phase", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("outcome", sa.String(length=64), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("stop_reason", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.Column("updated_at", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.String(length=32), nullable=False),
        sa.Column("ended_at", sa.String(length=32), nullable=True),
        sa.Column("heartbeat_at", sa.String(length=32), nullable=False),
        sa.Column("last_interrupted_at", sa.String(length=32), nullable=True),
        sa.Column("last_resumed_at", sa.String(length=32), nullable=True),
        sa.Column("error_at", sa.String(length=32), nullable=True),
        sa.CheckConstraint(
            "start_episode >= 0 AND target_episode >= start_episode AND latest_episode >= start_episode",
            name=op.f("ck_attempts_episodes"),
        ),
        sa.ForeignKeyConstraint(
            ["champion_at_start_assignment_id"],
            ["champion_history.id"],
            name=op.f("fk_attempts_champion_at_start_assignment_id_champion_history"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name=op.f("fk_attempts_experiment_id_experiments"),
        ),
        sa.ForeignKeyConstraint(
            ["starting_checkpoint_id"],
            ["checkpoints.id"],
            name=op.f("fk_attempts_starting_checkpoint_id_checkpoints"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_attempts")),
    )
    op.create_index(
        "ix_attempts_experiment_created",
        "attempts",
        ["experiment_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_attempts_starting_checkpoint",
        "attempts",
        ["starting_checkpoint_id"],
        unique=False,
    )
    create_table(
        "champion_history",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=36), nullable=False),
        sa.Column("previous_assignment_id", sa.String(length=36), nullable=True),
        sa.Column("attempt_id", sa.String(length=36), nullable=True),
        sa.Column("decision_id", sa.String(length=36), nullable=True),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["attempts.id"],
            name=op.f("fk_champion_history_attempt_id_attempts"),
        ),
        sa.ForeignKeyConstraint(
            ["checkpoint_id"],
            ["checkpoints.id"],
            name=op.f("fk_champion_history_checkpoint_id_checkpoints"),
        ),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name=op.f("fk_champion_history_decision_id_decisions"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name=op.f("fk_champion_history_experiment_id_experiments"),
        ),
        sa.ForeignKeyConstraint(
            ["previous_assignment_id"],
            ["champion_history.id"],
            name=op.f("fk_champion_history_previous_assignment_id_champion_history"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_champion_history")),
        sa.UniqueConstraint(
            "decision_id", name=op.f("uq_champion_history_decision_id")
        ),
        sa.UniqueConstraint(
            "experiment_id",
            "generation",
            name=op.f("uq_champion_history_experiment_id"),
        ),
    )
    create_table(
        "checkpoint_blobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("byte_length", sa.BigInteger(), nullable=False),
        sa.Column("format", sa.String(length=64), nullable=False),
        sa.Column("format_version", sa.Integer(), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "byte_length > 0", name=op.f("ck_checkpoint_blobs_nonempty")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_checkpoint_blobs")),
        sa.UniqueConstraint("sha256", name=op.f("uq_checkpoint_blobs_sha256")),
    )
    create_table(
        "checkpoints",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("attempt_id", sa.String(length=36), nullable=True),
        sa.Column("checkpoint_blob_id", sa.String(length=36), nullable=False),
        sa.Column("parent_checkpoint_id", sa.String(length=36), nullable=True),
        sa.Column("episode", sa.Integer(), nullable=False),
        sa.Column("environment_steps", sa.BigInteger(), nullable=False),
        sa.Column("optimization_steps", sa.BigInteger(), nullable=False),
        sa.Column("board", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("origin", sa.String(length=32), nullable=False),
        sa.Column("save_sequence", sa.Integer(), nullable=False),
        sa.Column("candidate_index", sa.Integer(), nullable=True),
        sa.Column("is_periodic_save", sa.Boolean(), nullable=False),
        sa.Column("is_screening_candidate", sa.Boolean(), nullable=False),
        sa.Column("is_best_in_attempt", sa.Boolean(), nullable=False),
        sa.Column("is_final_in_attempt", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "episode >= 0 AND environment_steps >= 0 AND optimization_steps >= 0",
            name=op.f("ck_checkpoints_counters"),
        ),
        sa.CheckConstraint(
            "parent_checkpoint_id IS NULL OR parent_checkpoint_id <> id",
            name=op.f("ck_checkpoints_not_self_parent"),
        ),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["attempts.id"],
            name=op.f("fk_checkpoints_attempt_id_attempts"),
        ),
        sa.ForeignKeyConstraint(
            ["checkpoint_blob_id"],
            ["checkpoint_blobs.id"],
            name=op.f("fk_checkpoints_checkpoint_blob_id_checkpoint_blobs"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name=op.f("fk_checkpoints_experiment_id_experiments"),
        ),
        sa.ForeignKeyConstraint(
            ["parent_checkpoint_id"],
            ["checkpoints.id"],
            name=op.f("fk_checkpoints_parent_checkpoint_id_checkpoints"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_checkpoints")),
        sa.UniqueConstraint(
            "attempt_id", "save_sequence", name=op.f("uq_checkpoints_attempt_id")
        ),
    )
    op.create_index(
        "ix_checkpoints_attempt_episode",
        "checkpoints",
        ["attempt_id", "episode"],
        unique=False,
    )
    op.create_index(
        "ix_checkpoints_experiment_created",
        "checkpoints",
        ["experiment_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_checkpoints_parent", "checkpoints", ["parent_checkpoint_id"], unique=False
    )
    create_table(
        "decisions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("operation_key", sa.String(length=255), nullable=False),
        sa.Column("attempt_id", sa.String(length=36), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=36), nullable=False),
        sa.Column("candidate_evaluation_id", sa.String(length=36), nullable=True),
        sa.Column("champion_evaluation_id", sa.String(length=36), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("result", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("policy", sa.Text(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["attempts.id"],
            name=op.f("fk_decisions_attempt_id_attempts"),
        ),
        sa.ForeignKeyConstraint(
            ["candidate_evaluation_id"],
            ["evaluation_batches.id"],
            name=op.f("fk_decisions_candidate_evaluation_id_evaluation_batches"),
        ),
        sa.ForeignKeyConstraint(
            ["champion_evaluation_id"],
            ["evaluation_batches.id"],
            name=op.f("fk_decisions_champion_evaluation_id_evaluation_batches"),
        ),
        sa.ForeignKeyConstraint(
            ["checkpoint_id"],
            ["checkpoints.id"],
            name=op.f("fk_decisions_checkpoint_id_checkpoints"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_decisions")),
        sa.UniqueConstraint("operation_key", name=op.f("uq_decisions_operation_key")),
    )
    create_table(
        "evaluation_batches",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.String(length=36), nullable=False),
        sa.Column("attempt_id", sa.String(length=36), nullable=True),
        sa.Column("checkpoint_id", sa.String(length=36), nullable=False),
        sa.Column("opponent_checkpoint_id", sa.String(length=36), nullable=True),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("opponent_kind", sa.String(length=32), nullable=False),
        sa.Column("config", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("planned_suite_count", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.String(length=32), nullable=False),
        sa.Column("heartbeat_at", sa.String(length=32), nullable=False),
        sa.Column("ended_at", sa.String(length=32), nullable=True),
        sa.CheckConstraint(
            "planned_suite_count > 0", name=op.f("ck_evaluation_batches_suite_count")
        ),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["attempts.id"],
            name=op.f("fk_evaluation_batches_attempt_id_attempts"),
        ),
        sa.ForeignKeyConstraint(
            ["checkpoint_id"],
            ["checkpoints.id"],
            name=op.f("fk_evaluation_batches_checkpoint_id_checkpoints"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name=op.f("fk_evaluation_batches_experiment_id_experiments"),
        ),
        sa.ForeignKeyConstraint(
            ["opponent_checkpoint_id"],
            ["checkpoints.id"],
            name=op.f("fk_evaluation_batches_opponent_checkpoint_id_checkpoints"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_batches")),
    )
    op.create_index(
        "ix_evaluation_batches_checkpoint_purpose",
        "evaluation_batches",
        ["checkpoint_id", "purpose"],
        unique=False,
    )
    create_table(
        "experiments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("game_config", sa.Text(), nullable=False),
        sa.Column("game_config_fingerprint", sa.String(length=64), nullable=False),
        sa.Column(
            "current_champion_assignment_id", sa.String(length=36), nullable=True
        ),
        sa.Column("seed_offsets", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.Column("updated_at", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(
            ["current_champion_assignment_id"],
            ["champion_history.id"],
            name=op.f("fk_experiments_current_champion_assignment_id_champion_history"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_experiments")),
        sa.UniqueConstraint("name", name=op.f("uq_experiments_name")),
    )
    create_table(
        "evaluation_suites",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("batch_id", sa.String(length=36), nullable=False),
        sa.Column("suite_index", sa.Integer(), nullable=False),
        sa.Column("definition", sa.Text(), nullable=False),
        sa.Column("suite_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("expected_games", sa.Integer(), nullable=False),
        sa.Column("expected_player_0_games", sa.Integer(), nullable=False),
        sa.Column("expected_player_1_games", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("player_0_wins", sa.BigInteger(), nullable=True),
        sa.Column("player_0_draws", sa.BigInteger(), nullable=True),
        sa.Column("player_0_losses", sa.BigInteger(), nullable=True),
        sa.Column("player_0_score_difference_sum", sa.BigInteger(), nullable=True),
        sa.Column("player_1_wins", sa.BigInteger(), nullable=True),
        sa.Column("player_1_draws", sa.BigInteger(), nullable=True),
        sa.Column("player_1_losses", sa.BigInteger(), nullable=True),
        sa.Column("player_1_score_difference_sum", sa.BigInteger(), nullable=True),
        sa.Column("elapsed_seconds", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.String(length=32), nullable=False),
        sa.Column("ended_at", sa.String(length=32), nullable=True),
        sa.CheckConstraint(
            "status <> 'completed' OR (player_0_wins IS NOT NULL AND player_0_draws IS NOT NULL AND player_0_losses IS NOT NULL AND player_0_score_difference_sum IS NOT NULL AND player_0_wins + player_0_draws + player_0_losses = expected_player_0_games)",
            name=op.f("ck_evaluation_suites_completed_seat_0"),
        ),
        sa.CheckConstraint(
            "status <> 'completed' OR (player_1_wins IS NOT NULL AND player_1_draws IS NOT NULL AND player_1_losses IS NOT NULL AND player_1_score_difference_sum IS NOT NULL AND player_1_wins + player_1_draws + player_1_losses = expected_player_1_games)",
            name=op.f("ck_evaluation_suites_completed_seat_1"),
        ),
        sa.CheckConstraint(
            "expected_games > 0 AND expected_player_0_games >= 0 AND expected_player_1_games >= 0 AND expected_player_0_games + expected_player_1_games = expected_games",
            name=op.f("ck_evaluation_suites_expected_counts"),
        ),
        sa.CheckConstraint(
            "player_0_draws IS NULL OR player_0_draws >= 0",
            name=op.f("ck_evaluation_suites_nonnegative_0_draws"),
        ),
        sa.CheckConstraint(
            "player_0_losses IS NULL OR player_0_losses >= 0",
            name=op.f("ck_evaluation_suites_nonnegative_0_losses"),
        ),
        sa.CheckConstraint(
            "player_0_wins IS NULL OR player_0_wins >= 0",
            name=op.f("ck_evaluation_suites_nonnegative_0_wins"),
        ),
        sa.CheckConstraint(
            "player_1_draws IS NULL OR player_1_draws >= 0",
            name=op.f("ck_evaluation_suites_nonnegative_1_draws"),
        ),
        sa.CheckConstraint(
            "player_1_losses IS NULL OR player_1_losses >= 0",
            name=op.f("ck_evaluation_suites_nonnegative_1_losses"),
        ),
        sa.CheckConstraint(
            "player_1_wins IS NULL OR player_1_wins >= 0",
            name=op.f("ck_evaluation_suites_nonnegative_1_wins"),
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["evaluation_batches.id"],
            name=op.f("fk_evaluation_suites_batch_id_evaluation_batches"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_suites")),
        sa.UniqueConstraint(
            "batch_id", "suite_index", name=op.f("uq_evaluation_suites_batch_id")
        ),
    )
    op.create_index(
        op.f("ix_evaluation_suites_suite_fingerprint"),
        "evaluation_suites",
        ["suite_fingerprint"],
        unique=False,
    )
    create_table(
        "training_metrics",
        sa.Column("attempt_id", sa.String(length=36), nullable=False),
        sa.Column("sample_sequence", sa.Integer(), nullable=False),
        sa.Column("episode", sa.Integer(), nullable=False),
        sa.Column("metrics", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["attempts.id"],
            name=op.f("fk_training_metrics_attempt_id_attempts"),
        ),
        sa.PrimaryKeyConstraint(
            "attempt_id", "sample_sequence", name=op.f("pk_training_metrics")
        ),
    )
    for table_name, constraint in deferred:
        target_table = constraint.elements[0].target_fullname.rsplit(".", 1)[0]
        op.create_foreign_key(
            constraint.name,
            table_name,
            target_table,
            list(constraint.column_keys),
            [item.target_fullname.rsplit(".", 1)[1] for item in constraint.elements],
        )


def downgrade():
    raise RuntimeError(
        "Training history migrations are forward-only; restore a backup to roll back."
    )
