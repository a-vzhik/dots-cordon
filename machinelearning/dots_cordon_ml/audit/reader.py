"""Plain application read models, independent of HTTP and SQLAlchemy."""

from datetime import datetime, timezone
import math

from .database import AuditError, RecordNotFound


def json_safe(value):
    """Preserve large seeds for JavaScript; missing numeric evidence stays null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if type(value) is int and abs(value) > 2**53 - 1:
        return str(value)
    return value


def diagnostics(row):
    heartbeat = row.get("heartbeat_at")
    age = (
        max(
            0,
            (
                datetime.now(timezone.utc) - datetime.fromisoformat(heartbeat)
            ).total_seconds(),
        )
        if heartbeat
        else None
    )
    return {
        "last_activity_at": heartbeat,
        "heartbeat_age_seconds": age,
        "unresponsive": row["status"] == "running" and age is not None and age > 300,
        "unresponsive_after_seconds": 300,
    }


def stats(wins, draws, losses, score_difference_sum):
    if any(value is None for value in (wins, draws, losses, score_difference_sum)):
        return None
    games = wins + draws + losses
    return dict(
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        score_difference_sum=score_difference_sum,
        match_score_numerator=2 * wins + draws,
        match_score_denominator=2 * games,
        match_score=(wins + 0.5 * draws) / games if games else None,
        mean_score_difference=score_difference_sum / games if games else None,
    )


def result(row):
    seats = [
        stats(
            *(
                row.get(f"player_{seat}_{field}")
                for field in ("wins", "draws", "losses", "score_difference_sum")
            )
        )
        for seat in (0, 1)
    ]
    if any(seat is None for seat in seats):
        return None
    overall = stats(
        *(
            sum(seat[field] for seat in seats)
            for field in ("wins", "draws", "losses", "score_difference_sum")
        )
    )
    return dict(overall=overall, as_player_0=seats[0], as_player_1=seats[1])


def decision_summary(row):
    return {
        key: row[key]
        for key in (
            "id",
            "attempt_id",
            "checkpoint_id",
            "stage",
            "result",
            "reason",
            "candidate_evaluation_id",
            "champion_evaluation_id",
            "rank",
            "created_at",
        )
    }


class AuditReader:
    def __init__(self, repository):
        self.repository = repository

    def evaluation_summaries(self, rows):
        totals = self.repository.batch_statistics([row["id"] for row in rows])
        output = []
        for row in rows:
            total = totals.get(row["id"], {})
            complete = (
                row["status"] == "completed"
                and total.get("completed_suites", 0) == row["planned_suite_count"]
            )
            score = result(total)
            output.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "id",
                            "experiment_id",
                            "attempt_id",
                            "checkpoint_id",
                            "opponent_checkpoint_id",
                            "opponent_kind",
                            "purpose",
                            "status",
                            "planned_suite_count",
                            "error",
                            "created_at",
                            "started_at",
                            "ended_at",
                            "heartbeat_at",
                        )
                    },
                    "score_orientation": "subject",
                    "completed_suite_count": total.get("completed_suites", 0),
                    "recorded_suite_count": total.get("recorded_suites", 0),
                    "expected_games": total.get("expected_games", 0),
                    "completed_games": score["overall"]["games"] if score else 0,
                    "is_complete": complete,
                    "aggregate": score,
                    "aggregate_is_partial": not complete,
                    "diagnostics": diagnostics(row),
                }
            )
        return output

    def checkpoint_summaries(self, rows, *, evidence=True):
        ids = [row["id"] for row in rows]
        blobs = {
            row["id"]: row
            for row in self.repository.blobs(
                list({row["checkpoint_blob_id"] for row in rows})
            )
        }
        counts = self.repository.checkpoint_counts(ids)
        evaluations = (
            self.evaluation_summaries(
                self.repository.recent_per_checkpoint("evaluation_batches", ids, 3)
            )
            if evidence and ids
            else []
        )
        decisions = (
            self.repository.recent_per_checkpoint("decisions", ids, 4)
            if evidence and ids
            else []
        )
        evaluation_map, decision_map = {}, {}
        for item in evaluations:
            evaluation_map.setdefault(item["checkpoint_id"], []).append(item)
        for item in decisions:
            decision_map.setdefault(item["checkpoint_id"], []).append(
                decision_summary(item)
            )
        return [
            {
                **row,
                "blob": {
                    "available": row["checkpoint_blob_id"] in blobs,
                    **{
                        key: blobs.get(row["checkpoint_blob_id"], {}).get(key)
                        for key in ("sha256", "byte_length", "format", "format_version")
                    },
                },
                "counts": counts[row["id"]],
                "recent_evaluations": evaluation_map.get(row["id"], []),
                "recent_decisions": decision_map.get(row["id"], []),
            }
            for row in rows
        ]

    def attempt_summaries(self, rows):
        counts = self.repository.attempt_counts([row["id"] for row in rows])
        activity = self.repository.attempt_activity([row["id"] for row in rows])
        return [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"config", "version"}
                },
                "counts": counts[row["id"]],
                "diagnostics": diagnostics(
                    {
                        **row,
                        "heartbeat_at": max(
                            row["heartbeat_at"],
                            activity.get(row["id"]) or row["heartbeat_at"],
                        ),
                    }
                ),
            }
            for row in rows
        ]

    def experiments(self, limit=50, cursor=None):
        page = self.repository.experiments_page(limit, cursor)
        counts = self.repository.experiment_counts([row["id"] for row in page["items"]])
        assignments = {
            row["id"]: row
            for row in self.repository.assignments(
                [
                    row["current_champion_assignment_id"]
                    for row in page["items"]
                    if row["current_champion_assignment_id"]
                ]
            )
        }
        checkpoints = {
            row["id"]: row
            for row in self.checkpoint_summaries(
                self.repository.checkpoints(
                    [row["checkpoint_id"] for row in assignments.values()]
                ),
                evidence=False,
            )
        }
        page["items"] = [
            {
                **{
                    key: row[key]
                    for key in (
                        "id",
                        "name",
                        "game_config",
                        "game_config_fingerprint",
                        "current_champion_assignment_id",
                        "created_at",
                        "updated_at",
                    )
                },
                "counts": counts[row["id"]],
                "current_champion": {
                    "assignment": assignments[row["current_champion_assignment_id"]],
                    "checkpoint": checkpoints[
                        assignments[row["current_champion_assignment_id"]][
                            "checkpoint_id"
                        ]
                    ],
                }
                if row["current_champion_assignment_id"]
                else None,
            }
            for row in page["items"]
        ]
        return json_safe(page)

    def attempts(self, **params):
        if params.get("experiment_id"):
            params["experiment_id"] = self.repository.experiment(
                params["experiment_id"]
            )["id"]
        if params.get("starting_checkpoint_id"):
            self.repository.required("checkpoints", params["starting_checkpoint_id"])
        page = self.repository.attempts_page(**params)
        page["items"] = self.attempt_summaries(page["items"])
        return json_safe(page)

    def attempt(self, identifier, checkpoint_limit=100, checkpoint_cursor=None):
        row = self.repository.required("attempts", identifier)
        page = self.repository.checkpoints_page(
            identifier, checkpoint_limit, checkpoint_cursor
        )
        page["items"] = self.checkpoint_summaries(page["items"])
        return json_safe(
            {
                **self.attempt_summaries([row])[0],
                "config": row["config"],
                "checkpoints": page,
            }
        )

    def metrics(
        self, identifier, *, episode_min=None, episode_max=None, max_points=500
    ):
        self.repository.required("attempts", identifier)
        return json_safe(
            {
                "attempt_id": identifier,
                **self.repository.metrics(
                    identifier, episode_min, episode_max, max_points
                ),
                "episode_min": episode_min,
                "episode_max": episode_max,
            }
        )

    def checkpoint(self, identifier, evaluation_limit=20, evaluation_cursor=None):
        row = self.repository.required("checkpoints", identifier)
        page = self.repository.evaluations_page(
            identifier, evaluation_limit, evaluation_cursor
        )
        page["items"] = self.evaluation_summaries(page["items"])
        return json_safe(
            {**self.checkpoint_summaries([row], evidence=False)[0], "evaluations": page}
        )

    def download_metadata(self, identifier):
        row = self.repository.required("checkpoints", identifier)
        blobs = self.repository.blobs([row["checkpoint_blob_id"]])
        if not blobs:
            raise RecordNotFound("Checkpoint bytes are unavailable")
        return {"checkpoint_id": identifier, "episode": row["episode"], **blobs[0]}

    def download_bytes(self, blob_id):
        return self.repository.payload(blob_id)

    def evaluation(
        self,
        identifier,
        suite_limit=20,
        suite_cursor=None,
        decision_limit=50,
        decision_cursor=None,
    ):
        self.repository.set_snapshot(suite_cursor, decision_cursor)
        row = self.repository.required("evaluation_batches", identifier)
        suites = self.repository.suites_page(identifier, suite_limit, suite_cursor)
        suites["items"] = [
            {**item, "result": result(item) if item["status"] == "completed" else None}
            for item in suites["items"]
        ]
        decisions = self.repository.evaluation_decisions(
            identifier, decision_limit, decision_cursor
        )
        participants = self.checkpoint_summaries(
            self.repository.checkpoints(
                [
                    value
                    for value in (row["checkpoint_id"], row["opponent_checkpoint_id"])
                    if value
                ]
            ),
            evidence=False,
        )
        return json_safe(
            {
                **self.evaluation_summaries([row])[0],
                "config": row["config"],
                "participants": participants,
                "suites": suites,
                "decisions": decisions,
            }
        )

    def lineage(
        self,
        experiment_id,
        *,
        root_checkpoint_id=None,
        episode_min=None,
        episode_max=None,
        attempt_limit=50,
        attempt_cursor=None,
        checkpoint_limit=200,
        checkpoint_cursor=None,
    ):
        experiment = self.repository.experiment(experiment_id)
        root = (
            self.repository.required("checkpoints", root_checkpoint_id)
            if root_checkpoint_id
            else None
        )
        if root and root["experiment_id"] != experiment["id"]:
            raise AuditError("Root checkpoint belongs to another experiment")
        attempts, nodes = self.repository.lineage_pages(
            experiment["id"],
            root,
            episode_min=episode_min,
            episode_max=episode_max,
            attempt_limit=attempt_limit,
            attempt_cursor=attempt_cursor,
            checkpoint_limit=checkpoint_limit,
            checkpoint_cursor=checkpoint_cursor,
        )
        visible_ids = {row["id"] for row in nodes["items"]}
        assignment_ids = {
            row["champion_at_start_assignment_id"]
            for row in attempts["items"]
            if row["champion_at_start_assignment_id"]
        }
        if experiment["current_champion_assignment_id"]:
            assignment_ids.add(experiment["current_champion_assignment_id"])
        assignments = self.repository.assignments(
            list(assignment_ids), list(visible_ids)
        )
        boundary_ids = {
            row["parent_checkpoint_id"]
            for row in nodes["items"]
            if row["parent_checkpoint_id"]
        }
        boundary_ids.update(
            row["starting_checkpoint_id"]
            for row in attempts["items"]
            if row["starting_checkpoint_id"]
        )
        boundary_ids.update(row["checkpoint_id"] for row in assignments)
        if root:
            boundary_ids.add(root["id"])
        boundary_ids -= visible_ids
        boundaries = self.repository.checkpoints(list(boundary_ids))
        all_nodes = self.checkpoint_summaries([*nodes["items"], *boundaries])
        visible_children = {}
        edges = []
        for row in nodes["items"]:
            if row["parent_checkpoint_id"]:
                edges.append(
                    {
                        "parent_checkpoint_id": row["parent_checkpoint_id"],
                        "child_checkpoint_id": row["id"],
                    }
                )
                visible_children[row["parent_checkpoint_id"]] = (
                    visible_children.get(row["parent_checkpoint_id"], 0) + 1
                )
        for row in all_nodes:
            row["is_boundary"] = row["id"] not in visible_ids
            row["children_outside_slice"] = max(
                0,
                row["counts"]["child_checkpoints"] - visible_children.get(row["id"], 0),
            )
        nodes["items"] = [row for row in all_nodes if not row["is_boundary"]]
        attempts["items"] = self.attempt_summaries(attempts["items"])
        current = next(
            (
                row
                for row in assignments
                if row["id"] == experiment["current_champion_assignment_id"]
            ),
            None,
        )
        current_node = next(
            (
                row
                for row in all_nodes
                if current and row["id"] == current["checkpoint_id"]
            ),
            None,
        )
        return json_safe(
            {
                "snapshot_at": self.repository.snapshot_at,
                "experiment": {
                    key: experiment[key]
                    for key in (
                        "id",
                        "name",
                        "game_config",
                        "game_config_fingerprint",
                        "current_champion_assignment_id",
                        "created_at",
                        "updated_at",
                    )
                },
                "counts": self.repository.experiment_counts([experiment["id"]])[
                    experiment["id"]
                ],
                "current_champion": {
                    "assignment": current,
                    "checkpoint": current_node,
                    "branches": self.repository.counts(
                        experiment["id"], current["checkpoint_id"]
                    ),
                }
                if current
                else None,
                "root_checkpoint_id": root_checkpoint_id,
                "episode_min": episode_min,
                "episode_max": episode_max,
                "attempts": attempts,
                "checkpoints": nodes,
                "boundary_checkpoints": [
                    row for row in all_nodes if row["is_boundary"]
                ],
                "edges": edges,
                "champion_history": assignments,
                "truncated": bool(attempts["remaining"] or nodes["remaining"]),
            }
        )
