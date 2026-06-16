#!/usr/bin/env python3
"""
fix_model.py — Patch the kashif/chronos-2-onnx model.onnx graph for ONNX
Runtime compatibility.

Problem:
  The exported model contains a ConstantOfShape node that produces a
  tensor(float) value which is then used as the *indices* input of a Gather
  node.  ONNX Runtime rejects this because Gather indices must be integer.

Fix:
  Insert a Cast(to=INT64) node between the ConstantOfShape output and any
  Gather node that consumes it as indices.

Usage:
  python3 examples/chronos/fix_model.py
  # reads  examples/chronos/model.onnx
  # writes examples/chronos/model_fixed.onnx

Requires: onnx  (pip install onnx)
"""

import sys
import pathlib

try:
    import onnx
    from onnx import TensorProto
except ImportError:
    sys.exit("Error: 'onnx' package not found.  Run:  pip3 install onnx")

SCRIPT_DIR = pathlib.Path(__file__).parent
SRC = SCRIPT_DIR / "model.onnx"
DST = SCRIPT_DIR / "model_fixed.onnx"


def fix(src: pathlib.Path, dst: pathlib.Path) -> None:
    if not src.exists():
        sys.exit(f"Error: source model not found: {src}\nRun download_model.sh first.")

    print(f"Loading {src} ...")
    model = onnx.load(str(src))
    graph = model.graph

    # Find ConstantOfShape nodes that output float and feed a Gather as indices
    output_remap: dict[str, str] = {}
    cast_nodes: list = []

    for node in graph.node:
        if node.op_type != "ConstantOfShape":
            continue
        val_attr = next((a for a in node.attribute if a.name == "value"), None)
        if val_attr is None or val_attr.t.data_type != TensorProto.FLOAT:
            continue
        out_name = node.output[0]
        for n2 in graph.node:
            if n2.op_type == "Gather" and out_name in list(n2.input)[1:]:
                cast_out = out_name + "_cast_int64"
                cast_node = onnx.helper.make_node(
                    "Cast",
                    inputs=[out_name],
                    outputs=[cast_out],
                    to=TensorProto.INT64,
                    name=node.name + "_fix_cast",
                )
                cast_nodes.append(cast_node)
                output_remap[out_name] = cast_out
                print(f"  Patching: {node.name} -> {n2.name}")

    if not output_remap:
        print("No patches needed — model appears healthy already.")
    else:
        # Remap Gather index inputs
        for node in graph.node:
            if node.op_type == "Gather":
                for i, inp in enumerate(node.input):
                    if inp in output_remap:
                        node.input[i] = output_remap[inp]

        for cn in cast_nodes:
            graph.node.append(cn)

        print(f"Applied {len(cast_nodes)} patch(es).")

    print(f"Saving {dst} ...")
    onnx.save(model, str(dst))
    print("Done.")


if __name__ == "__main__":
    fix(SRC, DST)
