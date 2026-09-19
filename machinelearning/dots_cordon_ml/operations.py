"""Local execution ownership and JSON transport for bounded CLI operations."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


@contextmanager
def operation_lock(audit, operation_id):
    """One local process per database/operation; released automatically on death.

    This is deliberately not a distributed worker lease. All local invocations
    must use the same OS user and database URL (SQLite paths are normalized).
    """
    url = audit.database.engine.url
    if url.get_backend_name() == "sqlite" and url.database != ":memory:":
        url = url.set(database=str(Path(url.database).expanduser().resolve()))
    identity = url.render_as_string(hide_password=False) + "\n" + operation_id
    digest = hashlib.sha256(identity.encode()).hexdigest()
    directory = Path(tempfile.gettempdir()) / f"dots-cordon-operations-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / f"{digest}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Operation is already running in another local process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_result(path, result):
    """The database is authoritative; this atomic file is only an acknowledgement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
