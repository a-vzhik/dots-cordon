"""Connection policy, transaction ownership, and migration lifecycle."""

from contextlib import contextmanager
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
import sqlalchemy as sa


class AuditError(ValueError):
    """Actionable application-level audit failure."""


class SchemaVersionError(AuditError):
    pass


class PromotionConflict(AuditError):
    pass


def default_database_url() -> str:
    path = Path(__file__).resolve().parents[2] / "audit" / "training.sqlite3"
    return f"sqlite:///{path}"


def database_url(value: str | None = None) -> str:
    return value or os.environ.get("DOTS_CORDON_DATABASE_URL") or default_database_url()


def migration_config() -> Config:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    return config


class Database:
    def __init__(self, url: str | None = None):
        selected = sa.engine.make_url(database_url(url))
        if selected.get_backend_name() == "sqlite" and selected.database not in (
            None,
            "",
            ":memory:",
        ):
            Path(selected.database).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self.engine = sa.create_engine(selected, pool_pre_ping=True)
        if selected.get_backend_name() == "sqlite":

            @sa.event.listens_for(self.engine, "connect")
            def sqlite_settings(connection, _):
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=10000")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

    def status(self) -> dict:
        with self.engine.connect() as connection:
            current = MigrationContext.configure(connection).get_current_revision()
        head = ScriptDirectory.from_config(migration_config()).get_current_head()
        return {
            "current_revision": current,
            "head_revision": head,
            "up_to_date": current == head,
        }

    def check_schema(self) -> None:
        if not self.status()["up_to_date"]:
            raise SchemaVersionError(
                "Audit schema is missing or outdated; run dots-cordon-audit db upgrade with the same database URL."
            )

    def upgrade(self) -> dict:
        config = migration_config()
        with self.engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        return self.status()

    @contextmanager
    def transaction(self):
        try:
            with self.engine.begin() as connection:
                yield connection
        except sa.exc.IntegrityError as exc:
            raise AuditError(
                "Audit write conflicts with existing records or a schema constraint."
            ) from exc
        except sa.exc.SQLAlchemyError as exc:
            raise AuditError(
                "Audit database operation failed; durable state was not confirmed."
            ) from exc

    def close(self):
        self.engine.dispose()
