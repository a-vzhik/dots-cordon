import signal
import hashlib

import numpy as np
import pytest
import torch

from dots_cordon_ml import search_train, search_evaluate
from dots_cordon_ml.audit import AuditService
from dots_cordon_ml.audit.database import Database
from dots_cordon_ml.checkpoint import restore_agent
from dots_cordon_ml.dqn import DQNAgent
from dots_cordon_ml.encoding import move_count
from dots_cordon_ml.environment import StepResult
from dots_cordon_ml.proto import game_pb2 as pb
from dots_cordon_ml.search import PolicyValueNetwork


class SmallEnvironment:
    """Finite placement game for deterministic training/persistence integration tests."""

    def __init__(self, server, rows, columns, max_turns, rpc_timeout):
        self.rows, self.columns, self.max_turns = rows, columns, max_turns
        self.reset()

    def reset(self):
        self.game = pb.GameState(
            board=pb.Board(
                rows=self.rows,
                columns=self.columns,
                cells=bytes(self.rows * self.columns),
            ),
            scores=[0, 0],
        )
        return self.game

    def simulate(self, state, action):
        assert state.board.cells[action] == 0
        result = pb.GameState()
        result.CopyFrom(state)
        cells = bytearray(state.board.cells)
        cells[action] = state.next_turn_by + 1
        result.board.cells = bytes(cells)
        result.next_turn_by = 1 - state.next_turn_by
        result.terminal = 0 not in cells or bool(
            self.max_turns and move_count(result) >= self.max_turns
        )
        return result

    def step(self, action):
        player = self.game.next_turn_by
        self.game = self.simulate(self.game, action)
        return StepResult(self.game, player, 0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture(autouse=True)
def small_environment(monkeypatch):
    monkeypatch.setattr(search_train, "GameEnvironment", SmallEnvironment)
    monkeypatch.setattr(search_evaluate, "GameEnvironment", SmallEnvironment)


def arguments(tmp_path, *extra):
    return search_train.parse_args(
        [
            "--no-audit",
            "--device",
            "cpu",
            "--rows",
            "2",
            "--columns",
            "2",
            "--channels",
            "4",
            "--blocks",
            "0",
            "--episodes",
            "3",
            "--simulations",
            "3",
            "--batch-size",
            "2",
            "--learning-starts",
            "2",
            "--updates-per-episode",
            "1",
            "--log-every",
            "1",
            "--eval-every",
            "0",
            "--checkpoint-every",
            "1",
            "--checkpoint-dir",
            str(tmp_path),
            *extra,
        ]
    )


def load(path):
    return torch.load(path / "search-latest.pt", map_location="cpu", weights_only=True)


def test_resume_restores_replay_rng_optimizer_and_matches_uninterrupted(tmp_path):
    full, first, resumed = [tmp_path / name for name in ("full", "first", "resumed")]
    assert search_train.run(arguments(full)) == 0
    assert search_train.run(arguments(first, "--episodes", "1")) == 0
    assert (
        search_train.run(
            arguments(resumed, "--resume", str(first / "search-latest.pt"))
        )
        == 0
    )
    expected, actual = load(full), load(resumed)
    assert actual["training_state"] == expected["training_state"]
    assert actual["training_state"]["episode"] == 3
    assert actual["rng_state"]["numpy"] == expected["rng_state"]["numpy"]
    for key, value in expected["online"].items():
        torch.testing.assert_close(actual["online"][key], value, rtol=0, atol=0)
    for key, value in expected["replay"].items():
        torch.testing.assert_close(actual["replay"][key], value, rtol=0, atol=0)


def test_evaluation_does_not_change_training_stream(tmp_path):
    plain, evaluated = tmp_path / "plain", tmp_path / "evaluated"
    search_train.run(arguments(plain))
    search_train.run(
        arguments(
            evaluated,
            "--eval-every",
            "1",
            "--eval-games",
            "2",
            "--eval-simulations",
            "2",
        )
    )
    a, b = load(plain), load(evaluated)
    assert a["rng_state"]["numpy"] == b["rng_state"]["numpy"]
    for k in a["online"]:
        torch.testing.assert_close(a["online"][k], b["online"][k], rtol=0, atol=0)


def test_audit_tracks_new_algorithm_evaluations_and_resume_lineage(tmp_path):
    url = f"sqlite:///{tmp_path / 'audit.sqlite3'}"
    db = Database(url)
    db.upgrade()
    db.close()
    args = arguments(
        tmp_path / "first",
        "--database-url",
        url,
        "--eval-every",
        "1",
        "--eval-games",
        "2",
        "--eval-simulations",
        "2",
        "--episodes",
        "1",
    )
    args.no_audit = False
    assert search_train.run(args) == 0
    with AuditService(url) as audit:
        experiment = audit.get_experiment("search-self-play")
        attempts = audit.list_attempts(experiment["id"])
        assert attempts[0]["status"] == "completed"
        checkpoints = audit.list_checkpoints(attempts[0]["id"])
        final = next(x for x in checkpoints if x["is_final_in_attempt"])
        assert final["model"]["kind"] == "policy_value"
        assert final["episode"] == 1
        assert len(audit.list_metrics(attempts[0]["id"])) == 1
        evaluations = audit.list_evaluations(experiment["id"])
        assert len(evaluations) == 4  # initial/final x policy/search
        assert {e["config"]["mode"] for e in evaluations} == {"policy", "search"}
        assert all(e["status"] == "completed" for e in evaluations)
        assert audit.current_champion(experiment["id"]) is None
    resumed = arguments(
        tmp_path / "second",
        "--database-url",
        url,
        "--resume",
        f"checkpoint:{final['id']}",
        "--episodes",
        "2",
    )
    resumed.no_audit = False
    search_train.run(resumed)
    with AuditService(url) as audit:
        child = next(
            a
            for a in audit.list_attempts(experiment["id"])
            if a["starting_checkpoint_id"]
        )
        assert child["starting_checkpoint_id"] == final["id"]
        assert child["config"]["replay_restored"] is True


def test_warm_start_resets_counters_and_rejects_dqn_resume(tmp_path):
    dqn = DQNAgent(torch.device("cpu"), 1e-3, 0.99, 7, 4, 0)
    source = tmp_path / "dqn.pt"
    torch.save(
        {
            "online": dqn.online.state_dict(),
            "optimizer": dqn.optimizer.state_dict(),
            "board": {"rows": 2, "columns": 2},
            "model": {"channels": 4, "blocks": 0},
            "training_state": {
                "episode": 12345,
                "environment_steps": 500000,
                "optimization_steps": 200000,
            },
        },
        source,
    )
    target = tmp_path / "warm"
    search_train.run(
        arguments(target, "--initialize-from", str(source), "--episodes", "1")
    )
    result = load(target)
    assert result["training_state"]["episode"] == 1
    assert result["training_state"]["optimization_steps"] == 1
    assert result["initialization"]["source_episode"] == 12345
    with pytest.raises(ValueError, match="policy_value"):
        search_train.run(arguments(tmp_path / "bad", "--resume", str(source)))
    with pytest.raises(ValueError, match="policy/value"):
        restore_agent(dqn, result, restore_optimizer=False)


def test_interrupt_finishes_episode_and_saves_replay(tmp_path, monkeypatch):
    original = search_train.collect_episode

    def interrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        return result

    monkeypatch.setattr(search_train, "collect_episode", interrupt)
    assert search_train.run(arguments(tmp_path)) == 130
    result = load(tmp_path)
    assert result["training_state"]["episode"] == 1
    assert len(result["replay"]["states"]) == 4


def test_resume_rejects_changed_game_rules(tmp_path):
    search_train.run(arguments(tmp_path / "source", "--episodes", "1"))
    with pytest.raises(ValueError, match="max-turns"):
        search_train.run(
            arguments(
                tmp_path / "target",
                "--resume",
                str(tmp_path / "source/search-latest.pt"),
                "--max-turns",
                "2",
            )
        )


def test_standalone_evaluator_records_modes_and_cross_experiment_baseline(tmp_path):
    url = f"sqlite:///{tmp_path / 'audit.sqlite3'}"
    db = Database(url)
    db.upgrade()
    db.close()
    dqn = DQNAgent(torch.device("cpu"), 1e-3, 0.99, 7, 4, 0)
    source = tmp_path / "dqn.pt"
    torch.save(
        {
            "online": dqn.online.state_dict(),
            "board": {"rows": 2, "columns": 2},
            "model": {"channels": 4, "blocks": 0},
            "training_state": {
                "episode": 100,
                "environment_steps": 400,
                "optimization_steps": 100,
            },
        },
        source,
    )
    with AuditService(url) as audit:
        old = audit.ensure_experiment("default", {"rows": 2, "columns": 2})
        record = audit.import_checkpoint(old["id"], source)
        champion = audit.bootstrap(old["id"], record["id"])
    args = arguments(
        tmp_path / "warm",
        "--database-url",
        url,
        "--episodes",
        "1",
        "--initialize-from",
        "champion:default",
    )
    args.no_audit = False
    search_train.run(args)
    options = search_evaluate.parse_args(
        [
            str(tmp_path / "warm/search-latest.pt"),
            "--database-url",
            url,
            "--device",
            "cpu",
            "--games",
            "2",
            "--simulations",
            "2",
            "--opponent",
            "champion:default",
            "--opening-random-moves",
            "0",
        ]
    )
    assert search_evaluate.run(options) == 0
    with AuditService(url) as audit:
        experiment = audit.get_experiment("search-self-play")
        evaluations = audit.list_evaluations(experiment["id"])
        assert len(evaluations) == 4
        assert {e["opponent_kind"] for e in evaluations} == {"random", "checkpoint"}
        assert all(e["status"] == "completed" for e in evaluations)
        attempt = audit.list_attempts(experiment["id"])[0]
        assert attempt["start_episode"] == 0
        assert attempt["starting_checkpoint_id"] is None
        assert attempt["config"]["initialization"]["source_episode"] == 100
        assert audit.current_champion(old["id"])["id"] == champion["id"]


@pytest.mark.parametrize("rows,columns", [(10, 15), (15, 15)])
def test_search_transfer_preserves_weights_and_starts_fresh(
    tmp_path, monkeypatch, rows, columns
):
    source_dir, target = tmp_path / "source", tmp_path / "target"
    search_train.run(arguments(
        source_dir, "--rows", "7", "--columns", "7", "--max-turns", "2",
        "--episodes", "1", "--seed", "11",
    ))
    source_path = source_dir / "search-latest.pt"
    source_bytes = source_path.read_bytes()
    source = load(source_dir)
    assert source["optimizer"]["state"]
    original_collect = search_train.collect_episode

    def check_initial_checkpoint(*args, **kwargs):
        # The durable episode-zero artifact exists before the first game/update.
        initial = torch.load(target / "search-0000000.pt", weights_only=True)
        assert initial["training_state"] == {
            "episode": 0, "environment_steps": 0, "optimization_steps": 0,
        }
        assert not initial["optimizer"]["state"]
        assert initial["replay"] == {}
        assert initial["rng_state"]["numpy"] == np.random.default_rng(7).bit_generator.state
        for key, weight in source["online"].items():
            torch.testing.assert_close(initial["online"][key], weight, rtol=0, atol=0)
        # Use fork_rng so the verification itself cannot alter learner RNG.
        with torch.random.fork_rng():
            torch.manual_seed(7)
            before = PolicyValueNetwork(4, 0)
            torch.testing.assert_close(initial["rng_state"]["torch"], torch.get_rng_state())
            after = PolicyValueNetwork(4, 0)
            before.load_state_dict(source["online"])
            after.load_state_dict(initial["online"])
            with torch.inference_mode():
                for shape in [(7, 7), (10, 15), (15, 15)]:
                    inputs = torch.randn(2, before.stem[0].in_channels, *shape)
                    for expected, actual in zip(before(inputs), after(inputs)):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        provenance = initial["initialization"]
        assert provenance["kind"] == "policy_value_weights"
        assert provenance["source_board"] == {"rows": 7, "columns": 7}
        assert provenance["destination_board"] == {"rows": rows, "columns": columns}
        assert provenance["source_model"] == provenance["destination_model"]
        assert provenance["sha256"] == hashlib.sha256(source_bytes).hexdigest()
        assert provenance["source_episode"] == 1
        assert provenance["source_checkpoint_id"] is None
        return original_collect(*args, **kwargs)

    monkeypatch.setattr(search_train, "collect_episode", check_initial_checkpoint)
    assert search_train.run(arguments(
        target, "--initialize-from", str(source_path), "--rows", str(rows),
        "--columns", str(columns), "--max-turns", "2", "--episodes", "1",
    )) == 0
    assert source_path.read_bytes() == source_bytes
    assert load(target)["replay"]["states"].shape[-2:] == (rows, columns)
    monkeypatch.setattr(search_train, "collect_episode", original_collect)
    resumed = tmp_path / "resumed"
    assert search_train.run(arguments(
        resumed, "--resume", str(target / "search-latest.pt"), "--rows", str(rows),
        "--columns", str(columns), "--max-turns", "2", "--episodes", "2",
    )) == 0
    assert load(resumed)["training_state"]["episode"] == 2
    assert load(resumed)["initialization"] == load(target)["initialization"]


def test_transfer_audit_records_source_without_resume_ancestry(tmp_path):
    url = f"sqlite:///{tmp_path / 'audit.sqlite3'}"
    db = Database(url)
    db.upgrade()
    db.close()
    source_args = arguments(tmp_path / "source", "--database-url", url, "--episodes", "1")
    source_args.no_audit = False
    search_train.run(source_args)
    with AuditService(url) as audit:
        original = audit.get_experiment("search-self-play")
        attempt = audit.list_attempts(original["id"])[0]
        source = next(c for c in audit.list_checkpoints(attempt["id"]) if c["is_final_in_attempt"])
        champion = audit.bootstrap(original["id"], source["id"])
    target_args = arguments(
        tmp_path / "target", "--database-url", url, "--experiment", "larger",
        "--initialize-from", "champion:search-self-play", "--rows", "3",
        "--columns", "4", "--max-turns", "2", "--episodes", "1",
    )
    target_args.no_audit = False
    search_train.run(target_args)
    with AuditService(url) as audit:
        target = audit.get_experiment("larger")
        attempt = audit.list_attempts(target["id"])[0]
        assert attempt["start_episode"] == 0
        assert attempt["starting_checkpoint_id"] is None
        assert attempt["config"]["initialization"]["source_checkpoint_id"] == source["id"]
        checkpoints = audit.list_checkpoints(attempt["id"])
        initial = next(c for c in checkpoints if c["episode"] == 0)
        assert initial["parent_checkpoint_id"] is None
        assert audit.current_champion(original["id"])["id"] == champion["id"]
        assert audit.current_champion(target["id"]) is None


@pytest.mark.parametrize("overrides,error", [
    (["--rows", "3"], "board"),
    (["--blocks", "1"], "architecture"),
])
def test_resume_rejects_changed_board_or_architecture(tmp_path, overrides, error):
    search_train.run(arguments(tmp_path / "source", "--episodes", "1"))
    with pytest.raises(ValueError, match=error):
        search_train.run(arguments(
            tmp_path / "target", "--resume", str(tmp_path / "source/search-latest.pt"),
            *overrides,
        ))


def test_transfer_rejects_source_directory_overwrite(tmp_path):
    search_train.run(arguments(tmp_path, "--episodes", "1"))
    source = tmp_path / "search-latest.pt"
    original_bytes = source.read_bytes()
    with pytest.raises(ValueError, match="separate --checkpoint-dir"):
        search_train.run(arguments(tmp_path, "--initialize-from", str(source)))
    assert source.read_bytes() == original_bytes
