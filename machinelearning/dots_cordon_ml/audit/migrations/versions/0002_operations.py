"""Durable CLI operation requests and results."""

from alembic import op
import sqlalchemy as sa

revision = "0002_operations"
down_revision = "0001_training_audit"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "operations",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("experiment_id", sa.String(36), sa.ForeignKey("experiments.id"), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("operation_request", sa.Text, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("progress", sa.Text, nullable=False),
        sa.Column("operation_result", sa.Text),
        sa.Column("created_at", sa.String(32), nullable=False),
        sa.Column("updated_at", sa.String(32), nullable=False),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'interrupted', 'failed')", name="status"
        ),
    )


def downgrade():
    raise RuntimeError("Training history migrations are forward-only; restore a backup to roll back.")
