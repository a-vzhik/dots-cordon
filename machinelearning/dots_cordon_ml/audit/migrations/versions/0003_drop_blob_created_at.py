"""Remove the unused timestamp stored after checkpoint payloads.

Checkpoint creation times remain in checkpoints.created_at.
"""

from alembic import op

revision = "0003_drop_blob_created_at"
down_revision = "0002_operations"
branch_labels = None
depends_on = None


def upgrade():
    # Native DROP COLUMN preserves references to this table. Avoid SQLite's
    # batch table replacement, which would drop a referenced parent table.
    op.drop_column("checkpoint_blobs", "created_at")


def downgrade():
    raise RuntimeError(
        "Training history migrations are forward-only; restore a backup to roll back."
    )
