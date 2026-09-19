import pytest
import torch

from dots_cordon_ml import promote_run, promotion_evaluation as evaluation
from dots_cordon_ml.audit import AuditService, Database
from dots_cordon_ml.checkpoint import read_checkpoint
from dots_cordon_ml.dqn import DQNAgent
from dots_cordon_ml.search import PolicyValueAgent
from dots_cordon_ml.self_play import EvaluationResult, MatchStats


def checkpoint(path, episode, policy=True):
    agent = (
        PolicyValueAgent(torch.device("cpu"), 4, 0)
        if policy
        else DQNAgent(torch.device("cpu"), 3e-4, 0.99, 7, channels=4, blocks=0)
    )
    payload = {
        "online": agent.online.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "board": {"rows": 7, "columns": 7},
        "game_config": {"max_turns": 0},
        "model": {
            "channels": 4,
            "blocks": 0,
            "kind": "policy_value" if policy else "dqn",
        },
        "training_state": {
            "episode": episode,
            "environment_steps": episode * 49,
            "optimization_steps": episode,
        },
    }
    if not policy:
        payload["target"] = agent.target.state_dict()
    torch.save(payload, path)
    return path


def result(wins, draws=0):
    losses = 50 - wins - draws
    diff = wins - losses
    seat = MatchStats(50, wins, draws, losses, diff / 50, diff)
    return EvaluationResult(
        MatchStats(100, 2 * wins, 2 * draws, 2 * losses, diff / 50, 2 * diff),
        seat,
        seat,
    )


class Connected:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_saved_run_promotes_in_order_rejects_extends_and_resumes(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'audit.sqlite3'}"
    db = Database(url)
    db.upgrade()
    db.close()
    with AuditService(url) as service:
        exp = service.ensure_experiment(
            "saved", {"rows": 7, "columns": 7, "max_turns": 0}
        )
        baseline = service.import_checkpoint(
            exp["id"], checkpoint(tmp_path / "baseline.pt", 10000, False)
        )
        source = service.create_attempt(exp["id"], config={}, target_episode=5)
        originals = [
            service.import_checkpoint(
                exp["id"],
                checkpoint(tmp_path / f"{ep}.pt", ep),
                attempt_id=source["id"],
            )
            for ep in range(6)
        ]
        service.update_attempt(source["id"], status="interrupted")
        args = promote_run.parse_args(
            [
                "--database-url",
                url,
                "--source-attempt",
                source["id"],
                "--baseline",
                f"checkpoint:{baseline['id']}",
                "--device",
                "cpu",
                "--screen-games",
                "100",
                "--head-to-head-games",
                "100",
                "--extended-head-to-head-games",
                "100",
                "--evaluation-workers",
                "10",
                "--output",
                str(tmp_path / "champion.pt"),
            ]
        )
        monkeypatch.setattr(evaluation, "_game_environment", lambda *args: Connected())
        seen = []
        fail = [True]

        def screen(args, path, device, seeds, games):
            _, metadata = read_checkpoint(path, map_location="cpu")
            stats = result(0 if metadata.episode == 4 else 49)
            return evaluation.EvaluatedCheckpoint(path, metadata, (stats,), stats)

        def challenge(args, candidate, champion, device, seed, games, opening):
            episode = candidate.metadata.episode
            seen.append((episode, champion.metadata.episode, seed))
            if episode == 5 and fail[0]:
                raise RuntimeError("worker failed")
            if episode == 2:
                return result(25)
            if episode == 3 and seed < args.extended_head_to_head_seed:
                return result(25, 1)  # 50.5%, extended validation required.
            if episode == 3:
                return result(25, 1)  # Marginal advantage is sufficient in extension.
            return result(30)

        monkeypatch.setattr(evaluation, "_evaluate_against_random_suites", screen)
        monkeypatch.setattr(evaluation, "_evaluate_head_to_head_suite", challenge)
        with pytest.raises(RuntimeError, match="worker failed"):
            promote_run.run(args)
        assert [
            service.get_checkpoint(h["checkpoint_id"])["episode"]
            for h in service.champion_history(exp["id"])
        ] == [10000, 1, 3]
        fail[0] = False
        assert promote_run.run(args) == 0
        assert [
            service.get_checkpoint(h["checkpoint_id"])["episode"]
            for h in service.champion_history(exp["id"])
        ] == [10000, 1, 3, 5]
        assert {champion for ep, champion, seed in seen if ep in (2, 3)} == {1}
        assert {champion for ep, champion, seed in seen if ep == 5} == {3}
        branches = [
            a for a in service.list_attempts(exp["id"]) if a["id"] != source["id"]
        ]
        assert len(branches) == 5
        assert all(a["status"] == "completed" for a in branches)
        assert service.get_attempt(source["id"])["status"] == "interrupted"
        for branch in branches:
            candidate = service.list_checkpoints(branch["id"])[0]
            original = originals[candidate["episode"]]
            assert candidate["parent_checkpoint_id"] == original["id"]
            assert candidate["checkpoint_blob_id"] == original["checkpoint_blob_id"]
            assert service.list_metrics(branch["id"]) == []
        batches_before = service.list_evaluations(exp["id"])
        assert promote_run.run(args) == 0
        assert service.list_evaluations(exp["id"]) == batches_before
        _, metadata = read_checkpoint(args.output, map_location="cpu")
        assert metadata.episode == 5


@pytest.mark.parametrize("policy", [False, True])
def test_shared_loader_accepts_both_model_kinds(tmp_path, policy):
    path = checkpoint(tmp_path / "model.pt", 7, policy)
    payload, metadata = read_checkpoint(path, map_location="cpu")
    agent = evaluation._load_agent(path, metadata, torch.device("cpu"), 1)
    assert isinstance(agent, PolicyValueAgent if policy else DQNAgent)
    for key, tensor in agent.online.state_dict().items():
        torch.testing.assert_close(tensor, payload["online"][key])
