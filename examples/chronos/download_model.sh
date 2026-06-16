#!/usr/bin/env bash
# download_model.sh — Download the chronos-2-onnx model from HuggingFace and
# apply a one-line graph fix required for ONNX Runtime compatibility.
#
# Usage:
#   bash examples/chronos/download_model.sh
#
# The script downloads model.onnx (~456 MB) into examples/chronos/ and then
# runs fix_model.py to produce model_fixed.onnx which is what the agent uses.
#
# Windows equivalent (PowerShell):
#   Invoke-WebRequest `
#     -Uri "https://huggingface.co/kashif/chronos-2-onnx/resolve/main/model.onnx" `
#     -OutFile "examples\chronos\model.onnx"
#   python examples\chronos\fix_model.py
#
# Dependencies: curl (macOS/Linux built-in), python3 with onnx package
#   pip3 install onnx
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_URL="https://huggingface.co/kashif/chronos-2-onnx/resolve/main/model.onnx"
MODEL_OUT="${SCRIPT_DIR}/model.onnx"
FIXED_OUT="${SCRIPT_DIR}/model_fixed.onnx"

echo "=== chronos-2-onnx download script ==="
echo "Target dir: ${SCRIPT_DIR}"
echo ""

# ── Download ──────────────────────────────────────────────────────────────────
if [ -f "${MODEL_OUT}" ]; then
  echo "model.onnx already present ($(du -sh "${MODEL_OUT}" | cut -f1)), skipping download."
else
  echo "Downloading model.onnx (~456 MB)..."
  curl -L --progress-bar -o "${MODEL_OUT}" "${MODEL_URL}"
  echo "Download complete: $(du -sh "${MODEL_OUT}" | cut -f1)"
fi

# ── Apply graph fix ───────────────────────────────────────────────────────────
if [ -f "${FIXED_OUT}" ]; then
  echo "model_fixed.onnx already present, skipping fix."
else
  echo "Applying graph compatibility fix..."
  python3 "${SCRIPT_DIR}/fix_model.py"
  echo "Fixed model written to: ${FIXED_OUT}"
fi

echo ""
echo "Done. Use model_fixed.onnx with the agent:"
echo "  ./build/src/main/mads-onnx-filter --inspect -m ${FIXED_OUT}"
echo "  python3 ${SCRIPT_DIR}/make_input.py"
echo "  ./build/src/main/mads-onnx-filter --test -m ${FIXED_OUT} --input ${SCRIPT_DIR}/chronos_input.json"
