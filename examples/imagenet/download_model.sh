#!/usr/bin/env bash
# download_model.sh — Download MobileNetV2-12 from the ONNX Model Zoo (GitHub)
# and an ImageNet class labels file into examples/imagenet/.
#
# Usage:
#   bash examples/imagenet/download_model.sh
#
# Then smoke-test with:
#   ./build/src/main/mads-onnx-source \
#     --inspect -m examples/imagenet/mobilenetv2-12.onnx
#   ./build/src/main/mads-onnx-source \
#     --test \
#     -m examples/imagenet/mobilenetv2-12.onnx \
#     --image examples/imagenet/sample_cat.jpg \
#     --labels examples/imagenet/imagenet_classes.txt
#
# Dependencies: curl (built-in on macOS / standard on Linux)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
MODEL_OUT="${SCRIPT_DIR}/mobilenetv2-12.onnx"

LABELS_URL="https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS_OUT="${SCRIPT_DIR}/imagenet_classes.txt"

# COCO validation image 000000039769 — the famous two-cats photo, well-recognised by
# ImageNet-trained models (MobileNetV2 predicts "tabby" at ~75 % confidence).
SAMPLE_URL="http://images.cocodataset.org/val2017/000000039769.jpg"
SAMPLE_OUT="${SCRIPT_DIR}/sample_cat.jpg"

echo "=== imagenet download script ==="
echo "Target dir: ${SCRIPT_DIR}"
echo ""

# ── Model ─────────────────────────────────────────────────────────────────────
if [ -f "${MODEL_OUT}" ] && [ "$(wc -c < "${MODEL_OUT}")" -gt 1000000 ]; then
  echo "mobilenetv2-12.onnx already present ($(du -sh "${MODEL_OUT}" | cut -f1)), skipping."
else
  echo "Downloading mobilenetv2-12.onnx (~13 MB)..."
  curl -L --progress-bar -o "${MODEL_OUT}" "${MODEL_URL}"
  echo "Model downloaded: $(du -sh "${MODEL_OUT}" | cut -f1)"
fi

# ── Labels ────────────────────────────────────────────────────────────────────
if [ -f "${LABELS_OUT}" ] && [ "$(wc -l < "${LABELS_OUT}")" -ge 1000 ]; then
  echo "imagenet_classes.txt already present, skipping."
else
  echo "Downloading imagenet_classes.txt (1000 ImageNet labels)..."
  curl -L --progress-bar -o "${LABELS_OUT}" "${LABELS_URL}"
  # Ensure last line ends with newline so wc -l reports 1000
  if [ "$(tail -c 1 "${LABELS_OUT}" | wc -c)" -eq 0 ] || [ "$(tail -c 1 "${LABELS_OUT}")" != $'\n' ]; then
    echo "" >> "${LABELS_OUT}"
  fi
  echo "Labels downloaded: $(wc -l < "${LABELS_OUT}") classes"
fi

# ── Sample image ──────────────────────────────────────────────────────────────
if [ -f "${SAMPLE_OUT}" ] && [ "$(wc -c < "${SAMPLE_OUT}")" -gt 10000 ]; then
  echo "sample_cat.jpg already present, skipping."
else
  echo "Downloading sample test image (COCO val 000000039769 — two cats)..."
  curl -L --progress-bar -o "${SAMPLE_OUT}" "${SAMPLE_URL}"
  echo "Sample image downloaded: $(du -sh "${SAMPLE_OUT}" | cut -f1)"
fi

echo ""
echo "=== Done. Next steps: ==="
echo ""
echo "Inspect the model:"
echo "  ./build/src/main/mads-onnx-source --inspect -m ${MODEL_OUT}"
echo ""
echo "Smoke-test (no MADS broker needed):"
echo "  ./build/src/main/mads-onnx-source \\"
echo "    --test \\"
echo "    -m ${MODEL_OUT} \\"
echo "    --image ${SAMPLE_OUT} \\"
echo "    --labels ${LABELS_OUT}"
echo ""
echo "Expected output: top prediction 'tabby' with ~75 % confidence."
