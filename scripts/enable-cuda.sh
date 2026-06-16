#!/usr/bin/env bash
# enable-cuda.sh — download and install the ONNX Runtime CUDA sidecar into the
# MADS lib directory so that `execution_provider = cuda` works.
#
# Prerequisites on the host:
#   • NVIDIA CUDA Toolkit (CUDA 12.x) must be installed and on PATH/LD_LIBRARY_PATH.
#   • cuDNN matching the CUDA major version (cuDNN 9.x for CUDA 12) is required
#     for some models; install from https://developer.nvidia.com/cudnn
#   • This script only covers Linux x86_64.  For aarch64 (Jetson / Grace-Hopper)
#     use the JetPack-specific ORT wheel instead.
#
# Usage:
#   chmod +x scripts/enable-cuda.sh
#   ./scripts/enable-cuda.sh           # installs into $(mads -p)/lib
#   ORT_DEST=/some/other/lib ./scripts/enable-cuda.sh
#
# Version coupling: this script is pinned to the ORT version that the package
# was built with.  Mixing sidecar and core versions is NOT supported.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
# This must match the ORT_VERSION used to build the onnx_agent package.
ORT_VERSION="1.26.0"
ORT_ARCHIVE="onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}"

# Destination: default to the MADS installation lib directory.
if [ -z "${ORT_DEST:-}" ]; then
  if command -v mads >/dev/null 2>&1; then
    MADS_PREFIX="$(mads -p)"
    ORT_DEST="${MADS_PREFIX}/lib"
  else
    echo "Error: 'mads' not found.  Set ORT_DEST=/path/to/lib or make sure" >&2
    echo "       the MADS installation is on PATH." >&2
    exit 1
  fi
fi

# ── Platform check ─────────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "${OS}" != "Linux" ] || [ "${ARCH}" != "x86_64" ]; then
  echo "Error: enable-cuda.sh only supports Linux x86_64." >&2
  echo "       Current platform: ${OS} ${ARCH}" >&2
  exit 1
fi

# ── Download ───────────────────────────────────────────────────────────────────
TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_WORK}"' EXIT

echo "Downloading ONNX Runtime ${ORT_VERSION} GPU archive (~215 MB)..."
curl -fL --progress-bar "${ORT_URL}" -o "${TMPDIR_WORK}/${ORT_ARCHIVE}"

echo "Extracting CUDA sidecar libraries..."
# Extract only the provider .so files — the core lib is already in ORT_DEST.
tar -xzf "${TMPDIR_WORK}/${ORT_ARCHIVE}" \
  -C "${TMPDIR_WORK}" \
  --wildcards \
  "*/lib/libonnxruntime_providers_cuda.so" \
  "*/lib/libonnxruntime_providers_shared.so" \
  2>/dev/null || true   # allow missing files (providers_shared may already exist)

# ── Install ────────────────────────────────────────────────────────────────────
echo "Installing into ${ORT_DEST} ..."
mkdir -p "${ORT_DEST}"

CUDA_SO="${TMPDIR_WORK}/onnxruntime-linux-x64-gpu-${ORT_VERSION}/lib/libonnxruntime_providers_cuda.so"
SHARED_SO="${TMPDIR_WORK}/onnxruntime-linux-x64-gpu-${ORT_VERSION}/lib/libonnxruntime_providers_shared.so"

if [ -f "${CUDA_SO}" ]; then
  cp -v "${CUDA_SO}" "${ORT_DEST}/"
else
  echo "Warning: libonnxruntime_providers_cuda.so not found in archive." >&2
fi

# Install providers_shared only if it is not already present (the base package
# installs it alongside the core lib).
if [ -f "${SHARED_SO}" ] && [ ! -f "${ORT_DEST}/libonnxruntime_providers_shared.so" ]; then
  cp -v "${SHARED_SO}" "${ORT_DEST}/"
fi

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "CUDA sidecar installed.  To enable CUDA inference:"
echo "  1. Ensure NVIDIA CUDA Toolkit 12.x is installed."
echo "  2. Ensure cuDNN 9.x (for CUDA 12) is installed."
echo "  3. Set  execution_provider = cuda  in your mads.ini [agents] section,"
echo "     or pass  --ep cuda  on the command line."
echo ""
echo "The onnx_agent runtime will dlopen libonnxruntime_providers_cuda.so from"
echo "${ORT_DEST} when the cuda EP is requested."
