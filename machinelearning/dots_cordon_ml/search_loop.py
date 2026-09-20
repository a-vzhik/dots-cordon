"""Sequential search-training rounds, evaluated and promoted in child processes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
from uuid import NAMESPACE_URL, uuid5

from . import evaluation, promotion, promotion_evaluation as shared, training
from .audit import AuditService
from .operations import operation_lock, write_result


STAGES = ("screening", "head_to_head", "extended_head_to_head")
GATES = {"screen_max_regression", "promotion_min_match_score", "promotion_min_suite_wins"}


def identifier(run_id, suffix):
    return str(uuid5(NAMESPACE_URL, f"dots-cordon-search-loop:{run_id}:{suffix}"))


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {
        "version", "run_id", "experiment", "source", "total_episode", "round_episodes",
        "training_config", "evaluation_config", "gates", "work_dir",
    }:
        raise ValueError("Complete explicit search-loop request required")
    if type(request["version"]) is not int or request["version"] != 1:
        raise ValueError("Unsupported search-loop version")
    if not evaluation._identifier(request["run_id"]):
        raise ValueError("run_id must contain 1–255 characters")
    if any(type(request[k]) is not int or request[k] < 1 for k in ("total_episode", "round_episodes")):
        raise ValueError("total_episode and round_episodes must be positive integers")
    if not isinstance(request["work_dir"], str) or not Path(request["work_dir"]).is_absolute():
        raise ValueError("work_dir must be an absolute path")
    training.request_arguments({"version": 1, "operation_id": "validate", "experiment": request["experiment"],
                                "source": request["source"], "target_episode": request["total_episode"],
                                "config": request["training_config"]})
    configs = request["evaluation_config"]
    if not isinstance(configs, dict) or set(configs) != set(STAGES):
        raise ValueError("Explicit screening, head_to_head and extended_head_to_head configs required")
    contest = dict(id="validate", candidate_checkpoint_id="candidate", expected_assignment_id="assignment",
                   champion_checkpoint_id="champion")
    for stage, config in configs.items():
        evaluation.request_arguments(dict(version=1, operation_id="validate", experiment=request["experiment"],
                                          contest=contest, stage=stage, config=config))
        if any(config[k] != request["training_config"][k] for k in ("rows", "columns", "max_turns")):
            raise ValueError("Training and evaluation board/rules must match")
    if configs["head_to_head"]["opening_random_moves"] != configs["extended_head_to_head"]["opening_random_moves"]:
        raise ValueError("Initial and extended challenge opening lengths must match")
    if not isinstance(request["gates"], dict) or set(request["gates"]) != GATES:
        raise ValueError("All three promotion gates must be explicit")
    # Reuse promotion's complete validation, including threshold boundaries.
    config = promotion_config(request, {})
    offset = 0
    for prefix in ("screen", "head_to_head", "extended_head_to_head"):
        count = config[prefix + "_suites"]
        config[prefix + "_seeds"] = list(range(offset, offset + count))
        offset += count
    promotion.validate_request(dict(version=1, operation_id="validate", experiment=request["experiment"],
                                    attempt_id="attempt", candidate_checkpoint_id="candidate",
                                    expected_assignment_id="assignment", config=config,
                                    evidence={key: key for key in promotion.EVIDENCE_FIELDS}))


def promotion_config(request, results):
    configs = request["evaluation_config"]
    config = {key: request["training_config"][key] for key in ("rows", "columns", "max_turns")}
    if "reuse_champion_screening" in configs["screening"]:
        config["reuse_champion_screening"] = configs["screening"]["reuse_champion_screening"]
    config.update(request["gates"], mode="policy", opening_random_moves=configs["head_to_head"]["opening_random_moves"])
    for stage, prefix in zip(STAGES, ("screen", "head_to_head", "extended_head_to_head"), strict=True):
        for key in ("games", "suites"):
            config[f"{prefix}_{key}"] = configs[stage][key]
        config[f"{prefix}_seeds"] = results.get(stage, {}).get("suite_seeds", [])
    return config


class Paused(RuntimeError):
    pass


class ChildProcesses:
    """The production transport: argv arrays, inherited human logs, JSON acknowledgements."""

    def __init__(self):
        self.child = None
        self.stopped = None

    def signal(self, signum, frame):
        if self.stopped is None:
            self.stopped = signum
            if self.child is not None and self.child.poll() is None:
                try:
                    self.child.send_signal(signum)
                except ProcessLookupError:
                    pass  # child exited between poll and signal delivery

    def command(self, kind, request_path, result_path, database_url):
        return [sys.executable, "-m", "dots_cordon_ml." + kind, "--request", str(request_path),
                "--result", str(result_path), "--database-url", database_url]

    def invoke(self, kind, request_path, result_path, database_url):
        if self.stopped:
            raise Paused("Shutdown requested; pending stage preserved")
        # A separate session prevents terminal Ctrl-C from reaching the child
        # twice; the supervisor forwards SIGINT/SIGTERM and waits for graceful exit.
        self.child = subprocess.Popen(self.command(kind, request_path, result_path, database_url), start_new_session=True)
        try:
            if self.stopped:  # signal between the preflight check and Popen
                self.child.send_signal(self.stopped)
            return self.child.wait()
        finally:
            self.child = None


def _result(request, state, status, error=None):
    result = dict(version=1, kind="supervisor", run_id=request["run_id"], status=status,
                  episode=state["episode"], learner_checkpoint_id=state["learner_checkpoint_id"],
                  target_episode=request["total_episode"], stage=state["stage"], rounds=state["rounds"])
    if error:
        result["error"] = error
    return result


class Supervisor:
    def __init__(self, audit, request, state, children):
        self.audit, self.request, self.state, self.children = audit, request, state, children
        self.experiment = audit.get_experiment(request["experiment"])
        self.operation_id = identifier(request["run_id"], "supervisor")
        self.database_url = audit.database.engine.url.render_as_string(hide_password=False)

    def save(self):
        self.audit.update_operation(self.operation_id, status="running", progress=self.state)

    def child(self, kind, request):
        pending = self.state.get("pending")
        if pending is None:
            self.state["pending"] = dict(kind=kind, request=request)
            self.save()  # a retry always has exactly the same request and identity
        elif pending != dict(kind=kind, request=request):
            raise ValueError("Persisted pending operation differs from the next child request")
        operation = self.audit.get_operation(request["operation_id"])
        if operation and operation["operation_request"] != request:
            raise ValueError("Child operation ID reused with different request")
        if operation and operation["status"] == "completed":
            return operation["operation_result"]
        directory = Path(self.request["work_dir"]) / self.operation_id
        request_path = directory / f"{request['operation_id']}.request.json"
        result_path = directory / f"{request['operation_id']}.result.json"
        write_result(request_path, request)
        print(f"round={self.state['round_index']} stage={self.state['stage']} "
              f"operation={request['operation_id']}", flush=True)
        code = self.children.invoke(kind, request_path, result_path, self.database_url)
        operation = self.audit.get_operation(request["operation_id"])
        if self.children.stopped:
            raise Paused("Shutdown requested; child exited and pending stage is recoverable")
        if operation and operation["status"] == "completed":
            return operation["operation_result"]  # DB commit wins over missing ack/nonzero exit
        detail = (operation or {}).get("operation_result") or {}
        raise Paused(f"{kind} child exited {code}: {detail.get('error', 'operation not completed')}")

    def transition(self, stage):
        self.state["stage"] = stage
        self.state.pop("pending", None)
        self.save()

    def stale(self):
        incumbent = self.audit.current_champion(self.experiment["id"])
        return incumbent is None or incumbent["id"] != self.state["contest"]["expected_assignment_id"]

    def discard_contest(self):
        contest = self.state["contest"]
        attempt_id = evaluation.contest_attempt_id(self.experiment["id"], contest["id"])
        if any(a["id"] == attempt_id and a["status"] == "running"
               for a in self.audit.list_attempts(self.experiment["id"])):
            self.audit.update_attempt(attempt_id, status="interrupted", stop_reason="incumbent changed; new contest required")
        self.state.setdefault("discarded_contests", []).append(dict(contest=contest, results=self.state["results"]))
        self.transition("contest")

    def evaluate(self, stage):
        job = dict(version=1, operation_id=identifier(self.request["run_id"], self.state["contest"]["id"] + ":" + stage),
                   experiment=self.request["experiment"], contest=self.state["contest"], stage=stage,
                   config=self.request["evaluation_config"][stage])
        self.state["results"][stage] = self.child("evaluation", job)
        self.transition("after_" + stage)

    def decide(self, decision, reason):
        self.state["decision"] = decision
        self.state["reason"] = reason
        self.transition("finish")

    def run(self):
        state, request = self.state, self.request
        while state["stage"] != "completed":
            if self.children.stopped:
                raise Paused("Shutdown requested; pending stage preserved")
            stage = state["stage"]
            # Pending promotion MUST reconcile first: its own commit changes the
            # incumbent and may precede both operation completion and child ack.
            if stage in {*STAGES, *("after_" + s for s in STAGES)} and self.stale():
                self.discard_contest()
                continue
            if stage == "train":
                config = copy.deepcopy(request["training_config"])
                if state["round_index"] > 1:
                    config["bootstrap_champion"] = False
                source = (request["source"] if state["round_index"] == 1 else
                          dict(mode="resume", checkpoint_id=state["learner_checkpoint_id"]))
                target = min(state["episode"] + request["round_episodes"], request["total_episode"])
                job = dict(version=1, operation_id=identifier(request["run_id"], f"round:{state['round_index']}:training"),
                           experiment=request["experiment"], source=source, target_episode=target, config=config)
                result = self.child("training", job)
                if result["episode"] != target or not result["checkpoint_id"]:
                    raise ValueError("Training child did not reach its absolute target")
                state.update(learner_checkpoint_id=result["checkpoint_id"], episode=result["episode"], training=result)
                self.transition("contest")
            elif stage == "contest":
                incumbent = self.audit.current_champion(self.experiment["id"])
                if incumbent is None:
                    raise ValueError("Target experiment requires an initial champion; enable bootstrap_champion during search transfer")
                state["contest_index"] += 1
                state["contest"] = dict(id=identifier(request["run_id"], f"contest:{state['contest_index']}"),
                                        candidate_checkpoint_id=state["learner_checkpoint_id"],
                                        expected_assignment_id=incumbent["id"], champion_checkpoint_id=incumbent["checkpoint_id"])
                state["results"] = {}
                self.transition("screening")
            elif stage in STAGES:
                self.evaluate(stage)
            elif stage == "after_screening":
                screen = state["results"]["screening"]
                suites = [evaluation.batch_results(self.audit.get_evaluation(i)) for i in screen["evaluation_ids"]]
                wrapped = [shared.EvaluatedCheckpoint(None, None, s, shared.combine_evaluation_results(s)) for s in suites]
                if shared._passes_random_screen(wrapped[1], wrapped[0], request["gates"]["screen_max_regression"]):
                    self.transition("head_to_head")
                else:
                    self.decide("rejected", "screening")
            elif stage in {"after_head_to_head", "after_extended_head_to_head"}:
                challenge = stage.removeprefix("after_")
                suites = evaluation.batch_results(self.audit.get_evaluation(state["results"][challenge]["evaluation_ids"][0]))
                outcome, _ = shared.challenge_decision(suites,
                    minimum_match_score=request["gates"]["promotion_min_match_score"],
                    minimum_suite_wins=request["gates"]["promotion_min_suite_wins"],
                    extended=challenge == "extended_head_to_head")
                if outcome == "passed":
                    self.transition("promotion")
                elif outcome == "extended":
                    self.transition("extended_head_to_head")
                else:
                    self.decide("rejected", challenge)
            elif stage == "promotion":
                results = state["results"]
                screen, initial = results["screening"], results["head_to_head"]
                extended = results.get("extended_head_to_head")
                job = dict(version=1, operation_id=identifier(request["run_id"], state["contest"]["id"] + ":promotion"),
                           experiment=request["experiment"], attempt_id=screen["attempt_id"],
                           candidate_checkpoint_id=screen["candidate_checkpoint_id"],
                           expected_assignment_id=state["contest"]["expected_assignment_id"],
                           config=promotion_config(request, results),
                           evidence=dict(champion_screening=screen["evaluation_ids"][0], candidate_screening=screen["evaluation_ids"][1],
                                         initial_head_to_head=initial["evaluation_ids"][0],
                                         extended_head_to_head=extended["evaluation_ids"][0] if extended else None))
                try:
                    result = self.child("promotion", job)
                except Paused:
                    own_commit = any(d["stage"] == "promotion" and
                                     d["policy"].get("promotion_request", {}).get("operation_id") == job["operation_id"]
                                     for d in self.audit.list_decisions(screen["attempt_id"]))
                    if not self.children.stopped and self.stale() and not own_commit:
                        self.discard_contest()
                        continue
                    raise
                if result["decision"] != "promoted":
                    raise ValueError("Promotion child did not confirm the passing gate")
                results["promotion"] = result
                self.decide("promoted", "extended_head_to_head" if extended else "head_to_head")
            elif stage == "finish":
                screen = state["results"]["screening"]
                if state["decision"] == "rejected":
                    screening = state["reason"] == "screening"
                    self.audit.record_decision(screen["attempt_id"], screen["candidate_checkpoint_id"],
                        stage="screening" if screening else "challenge", result="rejected",
                        candidate_evaluation_id=screen["evaluation_ids"][1] if screening else state["results"][state["reason"]]["evaluation_ids"][0],
                        champion_evaluation_id=screen["evaluation_ids"][0],
                        policy={**request["gates"], **({"reuse_champion_screening": True}
                                if request["evaluation_config"]["screening"].get("reuse_champion_screening") else {})},
                        operation_key=identifier(request["run_id"], state["contest"]["id"] + ":rejection"))
                    self.audit.update_attempt(screen["attempt_id"], status="completed", phase="finished",
                                              outcome="no_qualified_candidate" if screening else "no_challenger_passed")
                state["rounds"].append({key: copy.deepcopy(state[key]) for key in (
                    "round_index", "episode", "learner_checkpoint_id", "training", "contest", "results", "decision", "reason")})
                state["round_index"] += 1
                self.transition("completed" if state["episode"] >= request["total_episode"] else "train")
            else:
                raise ValueError(f"Unknown persisted supervisor stage: {stage}")
        return _result(request, state, "completed")


def run_request(request, database_url=None, *, children=None):
    validate_request(request)
    children = children or ChildProcesses()
    operation_id = identifier(request["run_id"], "supervisor")
    with AuditService(database_url) as audit, operation_lock(audit, operation_id):
        board = {k: request["training_config"][k] for k in ("rows", "columns", "max_turns")}
        experiment = audit.ensure_experiment(request["experiment"], board)
        operation = audit.ensure_operation(operation_id, experiment["id"], "supervisor", request)
        if operation["status"] == "completed":
            return operation["operation_result"]
        source = audit.get_checkpoint(request["source"]["checkpoint_id"])
        episode = source["episode"] if request["source"]["mode"] == "resume" else 0
        if episode >= request["total_episode"]:
            raise ValueError("total_episode must exceed the source's resumed episode")
        state = operation["progress"] or dict(stage="train", round_index=1, contest_index=0, rounds=[],
                                               episode=episode, learner_checkpoint_id=source["id"])
        supervisor = Supervisor(audit, request, state, children)
        supervisor.save()
        previous = {name: signal.signal(name, children.signal) for name in (signal.SIGINT, signal.SIGTERM)}
        try:
            result = supervisor.run()
            audit.update_operation(operation_id, status="completed", progress=state, result=result)
            return result
        except (Exception, KeyboardInterrupt) as exc:
            status = "interrupted" if children.stopped or isinstance(exc, KeyboardInterrupt) else "failed"
            result = _result(request, state, "paused", str(exc) or type(exc).__name__)
            audit.update_operation(operation_id, status=status, progress=state, result=result)
            return result
        finally:
            for name, handler in previous.items():
                signal.signal(name, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--database-url")
    args = parser.parse_args()
    try:
        result = run_request(json.loads(args.request.read_text()), args.database_url)
        write_result(args.result, result)
        if result["status"] != "completed":
            print(f"Search loop paused: {result['error']}. Restart with the same request.", file=sys.stderr)
        raise SystemExit(0 if result["status"] == "completed" else 1)
    except (ValueError, OSError) as exc:
        print(f"search loop failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
