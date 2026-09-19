"""Regenerate Go/PyTorch parity fixtures with the training Python environment.

From the repository root:
  machinelearning/.venv/bin/python inference/testdata/generate.py
Optional: --checkpoint FILE_OR_REFERENCE --output-dir /tmp/champion-parity
"""

import argparse
import base64
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from dots_cordon_ml.dqn import QNetwork
from dots_cordon_ml.encoding import encode_state, legal_action_mask
from dots_cordon_ml.export_policy import export_policy, load_policy
from dots_cordon_ml.proto import game_pb2 as pb
from dots_cordon_ml.search import PolicyValueNetwork


def generate(reference, output, name, count):
    policy, meta = load_policy(reference)
    export_policy(reference, output / f"{name}.onnx")
    rng = np.random.default_rng(231)
    cases = []
    sizes = [meta.board, (7, 7), (10, 15), (12, 12), (5, 15), (15, 5), (1, 1), meta.board]
    for i in range(count):
        rows, columns = sizes[i % len(sizes)]
        n = rows * columns
        cells = rng.integers(0, 6, n, dtype=np.uint8)
        cells[rng.choice(n, max(1, n // 3), replace=False)] = 0
        if i < 2:
            cells[:] = 0
        game = pb.GameState(
            board=pb.Board(rows=rows, columns=columns, cells=cells.tobytes()),
            scores=[int(x) for x in rng.integers(0, 8, 2)],
            next_turn_by=i % 2,
        )
        encoded = encode_state(game)
        with torch.inference_mode():
            scores = policy(torch.from_numpy(encoded).unsqueeze(0)).numpy().ravel()
        action = int(np.where(legal_action_mask(game), scores, -np.inf).argmax())
        cases.append({
            "game": base64.b64encode(game.SerializeToString()).decode(),
            "encoded": encoded.ravel().tolist(),
            "scores": scores.tolist(),
            "action": action,
        })
    (output / f"{name}.json").write_text(json.dumps(cases) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.checkpoint:
        generate(args.checkpoint, args.output_dir, "champion", 200)
        return
    with tempfile.TemporaryDirectory() as tmp:
        for kind, cls in [("dqn", QNetwork), ("policy_value", PolicyValueNetwork)]:
            torch.manual_seed(157)
            path = Path(tmp) / "fixture.pt"
            torch.save({
                "online": cls(4, 1).state_dict(),
                "board": {"rows": 3, "columns": 4},
                "model": {"channels": 4, "blocks": 1, "kind": kind},
                "game_config": {"max_turns": 0},
                "training_state": {"episode": 12, "environment_steps": 0, "optimization_steps": 0},
            }, path)
            generate(str(path), args.output_dir, kind, 8)


if __name__ == "__main__":
    main()
