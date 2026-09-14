"""Read-only FastAPI application; no trainer or game server is required."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from threading import BoundedSemaphore
from typing import Annotated

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException

from ..database import AuditError, InvalidCursor, RecordNotFound, SchemaVersionError
from ..service import AuditService
from . import schemas as dto


Limit = Annotated[int, Query(ge=1, le=200)]
Cursor = Annotated[str | None, Query(max_length=4096)]
Episode = Annotated[int | None, Query(ge=0)]


def error_response(status, code, message, details=None):
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details or []}},
    )


def episode_window(low, high):
    if low is not None and high is not None and low > high:
        raise AuditError("episode_min must not exceed episode_max")


def timestamp(value):
    if value is None:
        return None
    if value.tzinfo is None:
        raise AuditError("Time filters must include a timezone")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


class CheckpointFileResponse(FileResponse):
    """Release the download slot and temporary file even on disconnect."""

    def __init__(self, path, semaphore, **kwargs):
        self.download_semaphore = semaphore
        super().__init__(path, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                Path(self.path).unlink(missing_ok=True)
            finally:
                self.download_semaphore.release()


def create_app(
    database_url: str | None = None,
    *,
    download_workers: int = 2,
    frontend_dir: Path | None = None,
) -> FastAPI:
    if download_workers < 1:
        raise ValueError("download_workers must be positive")

    @asynccontextmanager
    async def lifespan(app):
        service = AuditService(database_url, read_only=True, check_schema=False)
        app.state.audit = service
        app.state.download_slots = BoundedSemaphore(download_workers)
        try:
            yield
        finally:
            service.close()

    app = FastAPI(
        title="Dots Cordon training audit",
        version="1.0.0",
        description="Read-only training lineage and evaluation evidence. Scores are always from the subject checkpoint's perspective.",
        lifespan=lifespan,
        responses={
            400: {"model": dto.ErrorResponse},
            404: {"model": dto.ErrorResponse},
            422: {"model": dto.ErrorResponse},
            503: {"model": dto.ErrorResponse},
        },
    )

    @app.middleware("http")
    async def cache_policy(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(RecordNotFound)
    def missing(_request, exc):
        return error_response(404, "not_found", str(exc))

    @app.exception_handler(InvalidCursor)
    def invalid_cursor(_request, exc):
        return error_response(400, "invalid_cursor", str(exc))

    @app.exception_handler(SchemaVersionError)
    def incompatible_schema(_request, _exc):
        return error_response(
            503,
            "schema_incompatible",
            "Apply audit migrations before using the API: dots-cordon-audit db upgrade",
        )

    @app.exception_handler(AuditError)
    def invalid_request(_request, exc):
        return error_response(400, "invalid_request", str(exc))

    @app.exception_handler(SQLAlchemyError)
    def database_unavailable(_request, _exc):
        return error_response(
            503, "database_unavailable", "The audit database is unavailable"
        )

    @app.exception_handler(RequestValidationError)
    def invalid_parameters(_request, exc):
        return error_response(
            422,
            "validation_error",
            "Invalid request parameters",
            [
                {
                    "location": list(item["loc"]),
                    "message": item["msg"],
                    "type": item["type"],
                }
                for item in exc.errors()
            ],
        )

    @app.exception_handler(HTTPException)
    def http_error(_request, exc):
        codes = {404: "not_found", 405: "method_not_allowed"}
        response = error_response(
            exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail)
        )
        response.headers.update(exc.headers or {})
        return response

    @app.exception_handler(OSError)
    def storage_error(_request, _exc):
        return error_response(
            503, "storage_unavailable", "Checkpoint storage is unavailable"
        )

    @app.exception_handler(Exception)
    def unexpected_error(_request, _exc):
        return error_response(
            500, "internal_error", "The audit request could not be completed"
        )

    @app.get(
        "/health",
        response_model=dto.Health,
        responses={503: {"model": dto.Health}},
        tags=["health"],
    )
    def health(request: Request, response: Response):
        try:
            status = request.app.state.audit.database.status()
            ready = status["up_to_date"]
            response.status_code = 200 if ready else 503
            return dict(
                status="ok" if ready else "unavailable",
                database="ok",
                schema="compatible" if ready else "incompatible",
                current_revision=status["current_revision"],
                head_revision=status["head_revision"],
            )
        except (SQLAlchemyError, AuditError):
            response.status_code = 503
            return dict(
                status="unavailable",
                database="unavailable",
                schema="unknown",
                current_revision=None,
                head_revision=None,
            )

    @app.get(
        "/api/v1/experiments",
        response_model=dto.Page[dto.Experiment],
        tags=["experiments"],
    )
    def experiments(request: Request, limit: Limit = 50, cursor: Cursor = None):
        with request.app.state.audit.reader() as reader:
            return reader.experiments(limit, cursor)

    @app.get(
        "/api/v1/experiments/{id}/lineage",
        response_model=dto.Lineage,
        tags=["experiments"],
    )
    def lineage(
        request: Request,
        id: str,
        root_checkpoint_id: str | None = None,
        episode_min: Episode = None,
        episode_max: Episode = None,
        attempt_limit: Annotated[int, Query(ge=1, le=100)] = 50,
        attempt_cursor: Cursor = None,
        checkpoint_limit: Annotated[int, Query(ge=1, le=500)] = 200,
        checkpoint_cursor: Cursor = None,
    ):
        episode_window(episode_min, episode_max)
        with request.app.state.audit.reader() as reader:
            return reader.lineage(
                id,
                root_checkpoint_id=root_checkpoint_id,
                episode_min=episode_min,
                episode_max=episode_max,
                attempt_limit=attempt_limit,
                attempt_cursor=attempt_cursor,
                checkpoint_limit=checkpoint_limit,
                checkpoint_cursor=checkpoint_cursor,
            )

    @app.get(
        "/api/v1/attempts",
        response_model=dto.Page[dto.AttemptSummary],
        tags=["attempts"],
    )
    def attempts(
        request: Request,
        experiment_id: str | None = None,
        starting_checkpoint_id: str | None = None,
        status: dto.WorkStatus | None = None,
        phase: dto.AttemptPhase | None = None,
        outcome: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        limit: Limit = 50,
        cursor: Cursor = None,
    ):
        after, before = timestamp(created_after), timestamp(created_before)
        if after and before and after > before:
            raise AuditError("created_after must not exceed created_before")
        with request.app.state.audit.reader() as reader:
            return reader.attempts(
                experiment_id=experiment_id,
                starting_checkpoint_id=starting_checkpoint_id,
                status=status,
                phase=phase,
                outcome=outcome,
                created_after=after,
                created_before=before,
                limit=limit,
                cursor=cursor,
            )

    @app.get(
        "/api/v1/attempts/{id}", response_model=dto.AttemptDetail, tags=["attempts"]
    )
    def attempt(
        request: Request,
        id: str,
        checkpoint_limit: Limit = 100,
        checkpoint_cursor: Cursor = None,
    ):
        with request.app.state.audit.reader() as reader:
            return reader.attempt(id, checkpoint_limit, checkpoint_cursor)

    @app.get(
        "/api/v1/attempts/{id}/metrics", response_model=dto.Metrics, tags=["attempts"]
    )
    def metrics(
        request: Request,
        id: str,
        episode_min: Episode = None,
        episode_max: Episode = None,
        max_points: Annotated[int, Query(ge=2, le=2000)] = 500,
    ):
        episode_window(episode_min, episode_max)
        with request.app.state.audit.reader() as reader:
            return reader.metrics(
                id,
                episode_min=episode_min,
                episode_max=episode_max,
                max_points=max_points,
            )

    @app.get(
        "/api/v1/checkpoints/{id}",
        response_model=dto.CheckpointDetail,
        tags=["checkpoints"],
    )
    def checkpoint(
        request: Request,
        id: str,
        evaluation_limit: Limit = 20,
        evaluation_cursor: Cursor = None,
    ):
        with request.app.state.audit.reader() as reader:
            return reader.checkpoint(id, evaluation_limit, evaluation_cursor)

    @app.get(
        "/api/v1/checkpoints/{id}/download",
        response_class=FileResponse,
        tags=["checkpoints"],
        responses={
            200: {"content": {"application/octet-stream": {}}},
            304: {"description": "Unchanged checkpoint"},
            409: {"model": dto.ErrorResponse},
            429: {"model": dto.ErrorResponse},
        },
    )
    def download(request: Request, id: str):
        service = request.app.state.audit
        with service.reader() as reader:
            metadata = reader.download_metadata(id)
        etag = f'"{metadata["sha256"]}"'
        headers = {
            "ETag": etag,
            "X-Checkpoint-SHA256": metadata["sha256"],
            "Cache-Control": "private, max-age=31536000, immutable",
        }
        supplied = [
            value.strip().removeprefix("W/")
            for value in request.headers.get("if-none-match", "").split(",")
        ]
        if "*" in supplied or etag in supplied:
            return Response(status_code=304, headers=headers)
        semaphore = request.app.state.download_slots
        if not semaphore.acquire(blocking=False):
            response = error_response(
                429, "download_limit", "All checkpoint download slots are occupied"
            )
            response.headers["Retry-After"] = "1"
            return response
        path = None
        try:
            with service.reader() as reader:
                payload = reader.download_bytes(metadata["id"])
            with tempfile.NamedTemporaryFile(
                prefix="dots-cordon-download-", suffix=".pt", delete=False
            ) as output:
                path = Path(output.name)
                output.write(payload)
            del payload
            return CheckpointFileResponse(
                path,
                semaphore,
                media_type="application/octet-stream",
                filename=f"dqn-{metadata['episode']:07d}-{id}.pt",
                headers={**headers, "Content-Length": str(metadata["byte_length"])},
            )
        except BaseException as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            semaphore.release()
            if isinstance(exc, AuditError):
                return error_response(
                    409,
                    "checkpoint_unavailable",
                    "Stored checkpoint bytes failed integrity verification",
                )
            raise

    @app.get(
        "/api/v1/evaluations/{id}",
        response_model=dto.EvaluationDetail,
        tags=["evaluations"],
    )
    def evaluation(
        request: Request,
        id: str,
        suite_limit: Limit = 20,
        suite_cursor: Cursor = None,
        decision_limit: Limit = 50,
        decision_cursor: Cursor = None,
    ):
        with request.app.state.audit.reader() as reader:
            return reader.evaluation(
                id, suite_limit, suite_cursor, decision_limit, decision_cursor
            )

    # The frontend calls the API on this same origin. An absent bundle does not
    # prevent API-only installations from starting or serving their read routes.
    static_dir = (
        frontend_dir if frontend_dir is not None else Path(__file__).parent / "static"
    )
    app.mount(
        "/assets",
        StaticFiles(directory=static_dir / "assets", check_dir=False),
        name="frontend-assets",
    )

    @app.get("/", include_in_schema=False)
    def dashboard():
        index = static_dir / "index.html"
        if not index.is_file():
            return error_response(
                503,
                "frontend_not_built",
                "Build the dashboard in machinelearning/web with npm ci and npm run build.",
            )
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
