"""Test-only subprocess adapter running production units with deterministic games."""
import argparse
import json
import os
from pathlib import Path
import importlib
import sys
import time
from dots_cordon_ml import search_train, promotion_evaluation as shared
from dots_cordon_ml.audit import AuditService
from dots_cordon_ml.checkpoint import read_checkpoint
from dots_cordon_ml.search_loop import ChildProcesses
from test_promote_run import result
from test_search_training import SmallEnvironment

class FixtureChildren(ChildProcesses):
    def command(self, kind, request_path, result_path, database_url):
        return [sys.executable, str(Path(__file__).resolve()), "--kind", kind, "--request", str(request_path),
                "--result", str(result_path), "--database-url", database_url]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    args, remaining = parser.parse_known_args()
    request_path = Path(remaining[remaining.index("--request") + 1])
    request = json.loads(request_path.read_text())
    directory = request_path.parent
    branch = os.environ.get("SEARCH_LOOP_TEST_BRANCH", "normal")
    search_train.GameEnvironment = SmallEnvironment
    def random_suites(config, path, device, seeds, games):
        _, metadata = read_checkpoint(path, map_location="cpu")
        wins = 20 if branch == "screen_reject" and metadata.episode > 0 else 30
        suites = tuple(result(wins) for _ in seeds)
        return shared.EvaluatedCheckpoint(path, metadata, suites, shared.combine_evaluation_results(suites))
    def challenge(config, candidate, champion, device, seed, games, openings):
        if branch == "initial_reject":
            return result(25)
        if branch in {"extended", "extended_reject"}:
            return result(25, 0 if branch == "extended_reject" and request.get("stage") == "extended_head_to_head" else 1)
        return result(26)
    shared._evaluate_against_random_suites = random_suites
    shared._evaluate_head_to_head_suite = challenge
    if args.kind == "evaluation" and os.environ.get("SEARCH_LOOP_TEST_PARTIAL_EVALUATION"):
        screen = shared._evaluate_random_screen
        def partial_screen(config, paths, device, seeds, games, *, audit_service, evaluation_ids):
            marker = directory / "partial-screen"
            if not marker.exists():
                from dots_cordon_ml.audit.integration import evaluation_result_dict
                marker.touch()
                audit_service.complete_suite(evaluation_ids[0], 0, evaluation_result_dict(result(30)))
                raise KeyboardInterrupt
            return screen(config, paths, device, seeds, games,
                          audit_service=audit_service, evaluation_ids=evaluation_ids)
        shared._evaluate_random_screen = partial_screen
    if args.kind == "search_loop":
        import dots_cordon_ml.search_loop as loop
        loop.ChildProcesses = FixtureChildren
    update = AuditService.update_operation
    marker = directory / "crashed"
    def crashing_update(self, operation_id, **fields):
        if os.environ.get("SEARCH_LOOP_TEST_CRASH") == args.kind and fields["status"] == "completed" and not marker.exists():
            marker.touch()
            os._exit(91)
        return update(self, operation_id, **fields)
    AuditService.update_operation = crashing_update
    if args.kind == "training" and os.environ.get("SEARCH_LOOP_TEST_SLOW"):
        collect = search_train.collect_episode
        def slow_collect(*a, **kw):
            (directory / "ready").touch()
            time.sleep(.5)
            return collect(*a, **kw)
        search_train.collect_episode = slow_collect
    trace = os.environ.get("SEARCH_LOOP_TEST_TRACE")
    if trace:
        with open(trace, "a") as stream:
            stream.write(json.dumps({"kind": args.kind, "request": request, "pid": os.getpid()}) + "\n")
    sys.argv = [args.kind, *remaining]
    importlib.import_module("dots_cordon_ml." + args.kind).main()

if __name__ == "__main__":
    main()
