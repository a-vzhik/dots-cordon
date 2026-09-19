import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
from onnx.reference import ReferenceEvaluator

from dots_cordon_ml import export_policy
from dots_cordon_ml.audit import AuditService, Database
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
