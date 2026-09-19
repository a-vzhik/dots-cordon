import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
from onnx.reference import ReferenceEvaluator

from dots_cordon_ml import export_policy
from dots_cordon_ml.audit import AuditService, Database
from dots_cordon_ml.search import PolicyValueNetwork
from test_promote_run import checkpoint


@pytest.mark.parametrize("policy", [False, True])
def test_export_matches_torch(tmp_path, policy):
    source = checkpoint(tmp_path / "model.pt", 12, policy)
    target = tmp_path / "model.onnx"
    info = export_policy.export_policy(str(source), target)
    network, _ = export_policy.load_policy(str(source))
    exported = onnx.load(target)
    dimensions = exported.graph.input[0].type.tensor_type.shape.dim
    assert [dim.dim_param for dim in dimensions[2:]] == ["rows", "columns"]
    evaluator = ReferenceEvaluator(exported)
    for rows, columns in [(7, 7), (10, 15), (12, 12), (5, 15), (15, 5), (1, 1), (7, 7)]:
        state = np.random.default_rng(7).normal(size=(1, 5, rows, columns)).astype(np.float32)
        with torch.inference_mode():
            expected = network(torch.from_numpy(state)).numpy()
        actual = evaluator.run(None, {"state": state})[0]
        assert actual.shape == (1, rows * columns)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert info["version"] == 2
    assert info["episode"] == 12
    assert info["kind"] == ("policy_value" if policy else "dqn")


def test_champion_export_is_read_only(tmp_path):
    url = f"sqlite:///{tmp_path / 'audit.sqlite3'}"
    database = Database(url)
    database.upgrade()
    database.close()
    path = checkpoint(tmp_path / "model.pt", 12)
    with AuditService(url) as audit:
        experiment = audit.ensure_experiment("play", {"rows": 7, "columns": 7})
        saved = audit.import_checkpoint(experiment["id"], path)
        assignment = audit.bootstrap(experiment["id"], saved["id"])
    path.unlink()
    info = export_policy.export_policy("champion:play", tmp_path / "model.onnx", url)
    assert info["episode"] == 12
    with AuditService(url) as audit:
        assert audit.current_champion(experiment["id"]) == assignment
        assert audit.list_attempts(experiment["id"]) == []
        assert audit.list_evaluations(experiment["id"]) == []


def test_export_rejects_turn_limit(tmp_path):
    path = checkpoint(tmp_path / "model.pt", 12)
    payload = torch.load(path, weights_only=True)
    payload["game_config"]["max_turns"] = 10
    torch.save(payload, path)
    with pytest.raises(ValueError, match="turn limit"):
        export_policy.export_policy(str(path), tmp_path / "model.onnx")


def test_expanded_policy_exports_with_prediction_parity(tmp_path):
    torch.manual_seed(12)
    original = PolicyValueNetwork(4, 3).eval()
    expanded = PolicyValueNetwork(4, 7).eval()
    expanded.initialize_from_policy_value(original.state_dict(), 3)
    source = checkpoint(tmp_path / "expanded.pt", 0)
    payload = torch.load(source, weights_only=True)
    payload["online"] = expanded.state_dict()
    payload["model"]["blocks"] = 7
    payload["board"] = {"rows": 15, "columns": 15}
    torch.save(payload, source)
    target = tmp_path / "expanded.onnx"
    export_policy.export_policy(str(source), target)
    restored, metadata = export_policy.load_policy(str(source))
    assert metadata.blocks == 7
    evaluator = ReferenceEvaluator(onnx.load(target))
    for rows, columns in [(7, 7), (10, 15), (15, 15)]:
        state = np.random.default_rng(7).normal(size=(1, 5, rows, columns)).astype(np.float32)
        with torch.inference_mode():
            expected, _ = original(torch.from_numpy(state))
            torch.testing.assert_close(restored(torch.from_numpy(state)), expected)
        actual = evaluator.run(None, {"state": state})[0]
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)
