"""Validated historical random evidence for candidate-only screening."""

from types import SimpleNamespace

from .audit.database import AuditError
from . import promotion_evaluation as shared


def _operation(store, identifier):
    getter = getattr(store, "get_operation", None) or store.find_operation
    return getter(identifier)


def validate_random_batch(store, batch, experiment_id, checkpoint_id, board):
    """Validate a completed batch against its own durable job, not today's budget."""
    from .evaluation import contest_attempt_id, request_arguments

    settings = batch["config"]
    if (batch["experiment_id"] != experiment_id or batch["checkpoint_id"] != checkpoint_id
            or batch["purpose"] != "screening" or batch["opponent_kind"] != "random"
            or batch["opponent_checkpoint_id"] is not None or batch["status"] != "completed"):
        raise AuditError("Cached random screening has incorrect participant, purpose or status")
    operation = _operation(store, settings.get("operation_id"))
    if (not operation or operation["kind"] != "evaluation" or operation["status"] != "completed"
            or operation["experiment_id"] != experiment_id
            or batch["id"] not in operation["progress"].get("evaluation_ids", [])):
        raise AuditError("Cached random screening requires a completed current evaluation operation")
    job = operation["operation_request"]
    request_arguments(job)
    attempt = store.get_attempt(batch["attempt_id"])
    if (job["stage"] != "screening" or settings.get("stage") != "screening"
            or job["contest"] != attempt["config"].get("contest")
            or contest_attempt_id(experiment_id, job["contest"]["id"]) != attempt["id"]
            or attempt["experiment_id"] != experiment_id
            or settings.get("generation") != operation["progress"].get("generation")
            or any(settings.get(k) != value for k, value in job["config"].items())
            or any(settings.get(k) != board[k] for k in ("rows", "columns", "max_turns"))):
        raise AuditError("Cached random screening provenance or board/rules differ")
    checkpoint = store.get_checkpoint(checkpoint_id)
    if checkpoint_id != job["contest"]["champion_checkpoint_id"]:
        source = store.get_checkpoint(job["contest"]["candidate_checkpoint_id"])
        if (checkpoint["attempt_id"] != attempt["id"]
                or checkpoint["parent_checkpoint_id"] != source["id"]
                or checkpoint["checkpoint_blob_id"] != source["checkpoint_blob_id"]):
            raise AuditError("Cached random screening subject differs from its immutable candidate")
    seeds = operation["progress"].get("suite_seeds", [])
    if len(seeds) != settings["suites"] or len(set(seeds)) != len(seeds):
        raise AuditError("Cached random screening suite seeds are incomplete")
    definitions = shared.suite_definitions(SimpleNamespace(**settings), SimpleNamespace(**settings),
                                           seeds, settings["games"])
    if len(batch["suites"]) != len(definitions) or batch["planned_suite_count"] != len(definitions):
        raise AuditError("Cached random screening suite count differs from its job")
    for index, (suite, definition) in enumerate(zip(batch["suites"], definitions, strict=True)):
        if suite["status"] != "completed" or suite["suite_index"] != index or suite["definition"] != definition:
            raise AuditError("Cached random screening suite definition or completion differs")
        for seat in (0, 1):
            counts = [suite[f"player_{seat}_{key}"] for key in ("wins", "draws", "losses")]
            if any(type(value) is not int or value < 0 for value in counts) or sum(counts) != settings["games"] // 2:
                raise AuditError("Cached random screening has incomplete or unpaired game counts")
    return batch


def select_champion_baseline(audit, experiment_id, assignment, board):
    """Prefer promotion-qualified evidence; bootstrap fallback is earliest valid batch."""
    champion_id = assignment["checkpoint_id"]
    if assignment.get("attempt_id"):
        decisions = audit.list_decisions(assignment["attempt_id"])
        preferred = [d["candidate_evaluation_id"] for d in decisions
                     if d["checkpoint_id"] == champion_id and d["stage"] == "screening"
                     and d["result"] in {"qualified", "passed"}]
        if preferred:
            return validate_random_batch(audit, audit.get_evaluation(preferred[0]), experiment_id, champion_id, board)
    for summary in sorted(audit.list_evaluations(experiment_id, champion_id), key=lambda b: (b["created_at"], b["id"])):
        try:
            return validate_random_batch(audit, audit.get_evaluation(summary["id"]), experiment_id, champion_id, board)
        except ValueError:
            continue
    raise AuditError("No compatible completed champion random screening exists; candidate-only screening cannot run")


def validate_cached_pair(store, candidate, champion, experiment_id, champion_id, board):
    """Require the exact historical baseline pinned before this candidate executed."""
    validate_random_batch(store, champion, experiment_id, champion_id, board)
    validate_random_batch(store, candidate, experiment_id, candidate["checkpoint_id"], board)
    operation = _operation(store, candidate["config"].get("operation_id"))
    if (operation["operation_request"]["config"].get("reuse_champion_screening") is not True
            or operation["progress"].get("champion_screening_id") != champion["id"]
            or operation["operation_request"]["contest"]["champion_checkpoint_id"] != champion_id):
        raise AuditError("Cached screening evidence differs from the pinned champion baseline")
