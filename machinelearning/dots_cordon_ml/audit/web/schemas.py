"""Versioned HTTP contracts. SQL rows and checkpoint bytes stay behind the API."""

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field


WorkStatus = Literal[
    "planned", "running", "completed", "interrupted", "failed", "abandoned", "cancelled"
]
AttemptPhase = Literal["training", "screening", "challenging", "finished"]
T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    database: Literal["ok", "unavailable"]
    schema_status: Literal["compatible", "incompatible", "unknown"] = Field(
        alias="schema"
    )
    current_revision: str | None
    head_revision: str | None


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    remaining: int
    limit: int
    next_cursor: str | None
    snapshot_at: datetime


class Diagnostics(BaseModel):
    last_activity_at: datetime | None
    heartbeat_age_seconds: float | None
    unresponsive: bool
    unresponsive_after_seconds: int


class MatchStats(BaseModel):
    games: int
    wins: int
    draws: int
    losses: int
    score_difference_sum: int
    match_score_numerator: int
    match_score_denominator: int
    match_score: float | None
    mean_score_difference: float | None


class Scores(BaseModel):
    overall: MatchStats
    as_player_0: MatchStats
    as_player_1: MatchStats


class EvaluationSummary(BaseModel):
    id: str
    experiment_id: str
    attempt_id: str | None
    checkpoint_id: str
    opponent_checkpoint_id: str | None
    opponent_kind: str
    purpose: str
    status: WorkStatus
    planned_suite_count: int
    recorded_suite_count: int
    completed_suite_count: int
    expected_games: int
    completed_games: int
    score_orientation: Literal["subject"]
    is_complete: bool
    aggregate_is_partial: bool
    aggregate: Scores | None
    error: str | None
    created_at: datetime
    started_at: datetime
    ended_at: datetime | None
    heartbeat_at: datetime
    diagnostics: Diagnostics


class DecisionSummary(BaseModel):
    id: str
    attempt_id: str
    checkpoint_id: str
    stage: str
    result: str
    reason: str | None
    candidate_evaluation_id: str | None
    champion_evaluation_id: str | None
    rank: int | None
    created_at: datetime


class Decision(DecisionSummary):
    policy: dict[str, Any]


class BlobMetadata(BaseModel):
    available: bool
    sha256: str | None
    byte_length: int | None
    format: str | None
    format_version: int | None


class Board(BaseModel):
    rows: int
    columns: int


class ModelArchitecture(BaseModel):
    channels: int
    blocks: int


class CheckpointCounts(BaseModel):
    child_checkpoints: int
    attempts_from_checkpoint: int
    evaluation_batches: int
    decisions: int


class CheckpointSummary(BaseModel):
    id: str
    experiment_id: str
    attempt_id: str | None
    checkpoint_blob_id: str
    parent_checkpoint_id: str | None
    episode: int
    environment_steps: int
    optimization_steps: int
    board: Board
    model: ModelArchitecture
    origin: str
    save_sequence: int
    candidate_index: int | None
    is_periodic_save: bool
    is_screening_candidate: bool
    is_best_in_attempt: bool
    is_final_in_attempt: bool
    created_at: datetime
    blob: BlobMetadata
    counts: CheckpointCounts
    recent_evaluations: list[EvaluationSummary]
    recent_decisions: list[DecisionSummary]


class CheckpointDetail(CheckpointSummary):
    evaluations: Page[EvaluationSummary]


class ChampionAssignment(BaseModel):
    id: str
    experiment_id: str
    generation: int
    checkpoint_id: str
    previous_assignment_id: str | None
    attempt_id: str | None
    decision_id: str | None
    reason: str
    created_at: datetime


class BranchCounts(BaseModel):
    attempts: int
    qualified_checkpoints: int
    challenge_batches: int
    promotions: int


class CurrentChampion(BaseModel):
    assignment: ChampionAssignment
    checkpoint: CheckpointSummary
    branches: BranchCounts | None = None


class ExperimentContext(BaseModel):
    id: str
    name: str
    game_config: dict[str, Any]
    game_config_fingerprint: str
    current_champion_assignment_id: str | None
    created_at: datetime
    updated_at: datetime


class ExperimentCounts(BaseModel):
    attempts: int
    checkpoints: int
    evaluation_batches: int


class Experiment(ExperimentContext):
    counts: ExperimentCounts
    current_champion: CurrentChampion | None


class AttemptCounts(BaseModel):
    checkpoints: int
    qualified_checkpoints: int
    challenge_batches: int
    promotions: int


class AttemptSummary(BaseModel):
    id: str
    experiment_id: str
    starting_checkpoint_id: str | None
    champion_at_start_assignment_id: str | None
    start_episode: int
    target_episode: int
    latest_episode: int
    phase: AttemptPhase
    status: WorkStatus
    outcome: str | None
    error: str | None
    stop_reason: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime
    ended_at: datetime | None
    heartbeat_at: datetime
    last_interrupted_at: datetime | None
    last_resumed_at: datetime | None
    error_at: datetime | None
    diagnostics: Diagnostics
    counts: AttemptCounts


class AttemptDetail(AttemptSummary):
    config: dict[str, Any]
    checkpoints: Page[CheckpointSummary]


class MetricSample(BaseModel):
    attempt_id: str
    sample_sequence: int
    episode: int
    metrics: dict[str, Any]
    created_at: datetime


class Metrics(BaseModel):
    attempt_id: str
    items: list[MetricSample]
    total_samples: int
    omitted_samples: int
    stride: int
    method: Literal["uniform_samples"]
    snapshot_at: datetime
    episode_min: int | None
    episode_max: int | None


class EvaluationSuite(BaseModel):
    id: str
    batch_id: str
    suite_index: int
    definition: dict[str, Any]
    suite_fingerprint: str
    expected_games: int
    expected_player_0_games: int
    expected_player_1_games: int
    status: WorkStatus
    elapsed_seconds: float | None
    error: str | None
    started_at: datetime
    ended_at: datetime | None
    result: Scores | None


class EvaluationDetail(EvaluationSummary):
    config: dict[str, Any]
    participants: list[CheckpointSummary]
    suites: Page[EvaluationSuite]
    decisions: Page[Decision]


class CheckpointNode(CheckpointSummary):
    is_boundary: bool
    children_outside_slice: int


class AncestryEdge(BaseModel):
    parent_checkpoint_id: str
    child_checkpoint_id: str


class Lineage(BaseModel):
    snapshot_at: datetime
    experiment: ExperimentContext
    counts: ExperimentCounts
    current_champion: CurrentChampion | None
    root_checkpoint_id: str | None
    episode_min: int | None
    episode_max: int | None
    attempts: Page[AttemptSummary]
    checkpoints: Page[CheckpointNode]
    boundary_checkpoints: list[CheckpointNode]
    edges: list[AncestryEdge]
    champion_history: list[ChampionAssignment]
    truncated: bool
