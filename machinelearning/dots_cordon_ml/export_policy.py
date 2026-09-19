"""Export trained weights for in-process Go inference (no Python during play)."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import onnx
import torch
from torch import nn

from .audit import AuditService
from .checkpoint import read_checkpoint
from .dqn import QNetwork
from .search import PolicyValueNetwork


def load_policy(reference: str, database_url: str | None = None):
    source = reference
    if reference.startswith(("champion:", "checkpoint:")):
        with AuditService(database_url, read_only=True) as audit:
            with audit.reader() as reader:
                source = BytesIO(reader.download_reference(reference))
    payload, metadata = read_checkpoint(source, "cpu")
    if payload.get("game_config", {}).get("max_turns", 0) != 0:
        raise ValueError("checkpoint was trained with a turn limit; CLI games use the full board")
    if not (1 <= metadata.rows <= 255 and 1 <= metadata.columns <= 255):
        raise ValueError("checkpoint board dimensions must be between 1 and 255")
    if metadata.kind == "policy_value":
        network = PolicyValueNetwork(metadata.channels, metadata.blocks)
    elif metadata.kind == "dqn":
        network = QNetwork(metadata.channels, metadata.blocks)
    else:
        raise ValueError(f"unsupported model kind: {metadata.kind}")
    network.load_state_dict(payload["online"])
    network.eval()
    # The value head is irrelevant to greedy policy play. Export only action scores.
    head = network.policy_head if metadata.kind == "policy_value" else network.head
    policy = nn.Sequential(network.stem, network.blocks, head, nn.Flatten(1)).eval()
    return policy, metadata


def export_policy(reference: str, output: Path, database_url: str | None = None):
    torch.set_num_threads(1)
    policy, metadata = load_policy(reference, database_url)
    buffer = BytesIO()
    with torch.inference_mode():
        torch.onnx.export(
            policy,
            torch.zeros(1, 5, metadata.rows, metadata.columns),
            buffer,
            input_names=["state"],
            output_names=["scores"],
            dynamic_axes={
                "state": {2: "rows", 3: "columns"},
                "scores": {1: "cells"},
            },
            opset_version=17,
            dynamo=False,
        )
    model = onnx.load_model_from_string(buffer.getvalue())
    info = {
        "version": 2,
        "encoding": "player-relative-5-v1",
        # Training dimensions are provenance, not inference constraints.
        "rows": metadata.rows,
        "columns": metadata.columns,
        "episode": metadata.episode,
        "kind": metadata.kind,
    }
    onnx.helper.set_model_props(model, {"dots_cordon": json.dumps(info)})
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(model.SerializeToString())
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help=".pt path, champion:EXPERIMENT or checkpoint:UUID")
    parser.add_argument("output", type=Path, help="output .onnx file")
    parser.add_argument("--database-url")
    args = parser.parse_args()
    try:
        info = export_policy(args.checkpoint, args.output, args.database_url)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Exported episode {info['episode']} ({info['kind']}) to {args.output}")


if __name__ == "__main__":
    main()
