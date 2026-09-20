from datetime import datetime
from pathlib import Path
import sqlite3

from alembic import command
import pytest

from dots_cordon_ml.audit import AuditError, Database
from dots_cordon_ml.audit import database as database_module
from dots_cordon_ml.audit.database import migration_config


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    class FixedTime:
        @staticmethod
        def now():
            return datetime(2026, 9, 20, 14, 30, 5)

    monkeypatch.setattr(database_module, "datetime", FixedTime)
    path = tmp_path / "training.sqlite3"
    database = Database(f"sqlite:///{path}")
    with database.engine.begin() as connection:
        config = migration_config()
        config.attributes["connection"] = connection
        command.upgrade(config, "0002_operations")
        connection.exec_driver_sql("CREATE TABLE backup_probe (payload BLOB NOT NULL)")
        connection.exec_driver_sql(
            "INSERT INTO backup_probe VALUES (?)", (b"committed WAL data" * 8192,)
        )
    yield database, path, tmp_path / "training-20262009-143005.sqlite3"
    database.close()


def assert_backup(path):
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA quick_check").fetchall() == [("ok",)]
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "0002_operations",
        )
        assert connection.execute("SELECT payload FROM backup_probe").fetchone() == (
            b"committed WAL data" * 8192,
        )
        assert "created_at" in {
            row[1] for row in connection.execute("PRAGMA table_info(checkpoint_blobs)")
        }


def test_backup_contains_wal_and_finishes_before_migration(legacy, monkeypatch):
    database, path, backup_path = legacy
    assert Path(f"{path}-wal").stat().st_size > 0
    original_upgrade = command.upgrade
    calls = []

    def checked_upgrade(config, revision):
        assert_backup(backup_path)
        # A writer cannot change data after the backup and before the migration.
        with sqlite3.connect(path, timeout=0) as writer:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer.execute("INSERT INTO backup_probe VALUES ('concurrent')")
        calls.append(revision)
        return original_upgrade(config, revision)

    monkeypatch.setattr(command, "upgrade", checked_upgrade)
    result = database.upgrade()
    assert result["up_to_date"]
    assert result["backup_path"] == str(backup_path)
    assert_backup(backup_path)
    with database.engine.connect() as connection:
        assert "created_at" not in {
            row[1]
            for row in connection.exec_driver_sql("PRAGMA table_info(checkpoint_blobs)")
        }
    # No pending migrations means no new backup, even in the same second.
    assert database.upgrade()["backup_path"] is None
    assert calls == ["head"]


def test_backup_failure_removes_partial_copy_and_prevents_migration(legacy, monkeypatch):
    database, _, backup_path = legacy
    connect = sqlite3.connect

    class FailedSource:
        def backup(self, _destination, **_kwargs):
            raise sqlite3.OperationalError("simulated backup failure")

        def close(self):
            pass

    def fail_backup_source(path, *args, **kwargs):
        if str(path).endswith("?mode=ro"):
            return FailedSource()
        return connect(path, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", fail_backup_source)
    monkeypatch.setattr(command, "upgrade", lambda *_: pytest.fail("Migration ran without backup"))
    with pytest.raises(AuditError, match="migration was not applied.*simulated backup failure"):
        database.upgrade()
    assert not backup_path.exists()
    assert database.status()["current_revision"] == "0002_operations"


def test_existing_backup_is_never_overwritten(legacy, monkeypatch):
    database, _, backup_path = legacy
    backup_path.write_bytes(b"previous backup")
    monkeypatch.setattr(command, "upgrade", lambda *_: pytest.fail("Migration ran without backup"))
    with pytest.raises(AuditError, match="Could not create pre-migration backup"):
        database.upgrade()
    assert backup_path.read_bytes() == b"previous backup"
    assert database.status()["current_revision"] == "0002_operations"


def test_migration_failure_keeps_backup_and_rolls_back(legacy, monkeypatch):
    database, _, backup_path = legacy

    def fail_migration(config, _revision):
        assert_backup(backup_path)
        config.attributes["connection"].exec_driver_sql("DROP TABLE backup_probe")
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(command, "upgrade", fail_migration)
    with pytest.raises(RuntimeError, match="simulated migration failure"):
        database.upgrade()
    assert_backup(backup_path)
    assert database.status()["current_revision"] == "0002_operations"
    with database.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT count(*) FROM backup_probe").scalar_one() == 1


def test_new_database_does_not_need_backup(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'new.sqlite3'}")
    try:
        result = database.upgrade()
        assert result["up_to_date"]
        assert result["backup_path"] is None
        assert list(tmp_path.glob("new-*.sqlite3")) == []
    finally:
        database.close()
