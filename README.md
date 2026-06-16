[![Build and Release](https://github.com/MADS-NET/onnx_agent/actions/workflows/release.yml/badge.svg)](https://github.com/MADS-NET/onnx_agent/actions/workflows/release.yml)  ![MADS package](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FMADS-NET%2F.github%2Fmain%2Fprofile%2Fpackages.json&query=%24.packages.%5B'mads-onnx'%5D.type&label=mads%20package&logo=C%2B%2B)

# MADS ONNX Agent

Two [MADS](https://github.com/pbosetti/MADS) agents that load and run ONNX models:

- **`mads-onnx-filter`** — subscribes to a MADS topic, receives JSON-encoded
  tensors, runs an ONNX model, and publishes the output tensors as JSON to
  another topic. Works with any ONNX model without code changes.
- **`mads-onnx-source`** — internally timed agent that reads from a webcam
  (via OpenCV), runs an ONNX image classifier, and publishes top-k predictions.
  The source interface is extensible; the default implementation wraps a
  `CameraSource` backed by OpenCV VideoCapture.

A companion utility `mads-onnx-inspect` prints model I/O metadata and a ready-to-use
JSON schema without starting the MADS network.

Tested examples included:
- Chronos-2 time-series forecasting (`mads-onnx-filter`)
- MobileNetV2-12 ImageNet classification (`mads-onnx-source`, classifies a cat
  image as "tabby" with 74.9% confidence)

---

## Contents

```
onnx_agent/
  src/
    onnx/          OnnxModel, TensorJson — core library (no MADS dependency)
    source/        DataSource interface, CameraSource (OpenCV), Classify helper
    main/          onnx_filter.cpp, onnx_source.cpp, onnx_inspect.cpp
  examples/
    chronos/       Chronos-2 ONNX time-series forecast example
    imagenet/      MobileNetV2-12 ImageNet classification example
  mads.ini         Ready-made configuration for both agents
```

---

## Requirements

| Requirement | Notes |
|---|---|
| C++20 compiler | Clang recommended (LLVM style) |
| CMake >= 3.28 | Ninja backend preferred |
| MADS >= 2.2.0 | Discovered automatically via `mads -p` |
| ONNX Runtime 1.26.0 | Downloaded automatically by FetchContent |
| OpenCV 4.x (optional) | Required only for `mads-onnx-source` camera mode |

---

## Building

### Quick start (developer build)

```bash
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
```

MADS is discovered automatically: CMake runs `mads -p` to find the install
prefix. Override with `-DMADS_ROOT=/path/to/mads` or the `MADS_ROOT`
environment variable.

ONNX Runtime is downloaded automatically for your platform during configure.
No system-level install is needed.

OpenCV is found via `find_package(OpenCV)`. If absent, `mads-onnx-source` is
simply omitted from the build; the filter agent and inspect utility still build.

### Self-contained package (for distribution)

```bash
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DMADS_INSTALL_AGENT=ON \
      -DCMAKE_INSTALL_PREFIX="$PWD/package"
cmake --build build -j6
cmake --build build --target package
```

The resulting `build/onnx_agent-<version>-<os>-<arch>.zip` contains:

```
onnx_agent-<version>-<os>-<arch>/
  bin/
    mads-onnx-inspect
    mads-onnx-filter
    mads-onnx-source        (only when OpenCV was found)
  lib/
    libonnxruntime.1.26.0.dylib   (macOS)
    libonnxruntime.1.dylib        (symlink)
```

Executables are linked with `@executable_path/../lib` (macOS) or
`$ORIGIN/../lib` (Linux) so the package is fully self-contained: unzip
anywhere and run.

### CMake options

| Option | Default | Description |
|---|---|---|
| `MADS_ROOT` | auto | Path to MADS installation prefix |
| `MADS_INSTALL_AGENT` | `OFF` | Install agents + ORT lib (required for `package`) |
| `ORT_VERSION` | `1.26.0` | ONNX Runtime prebuilt version to download |
| `ONNX_CPU_ONLY` | `OFF` | Fetch the small CPU-only ORT archive (disables GPU EPs) |
| `VERSION` | from git tag | Package version (override for CI without tags) |

---

## GPU acceleration

Each pre-built package ships a **single, GPU-capable ONNX Runtime** — no
separate GPU variant is needed.  Inference runs on **CPU by default**; a GPU
execution provider (EP) is selected by setting `execution_provider` in
`mads.ini` or passing `--ep <name>`.

### Per-platform EP availability

| Platform | GPU EP | How to enable |
|---|---|---|
| macOS arm64 | CoreML (ANE / GPU) | `execution_provider = coreml` or `auto` |
| Windows x64 | DirectML (any DX12 GPU) | `execution_provider = directml` |
| Linux x64 | CUDA (NVIDIA) | see below — requires manual sidecar install |
| Linux aarch64 | CPU only (no gpu build) | — |

CPU is always the fallback if a requested EP is unavailable (a yellow warning
is printed at startup; inference continues).

### macOS — CoreML

The standard `osx-arm64` ONNX Runtime already includes CoreML and WebGPU EPs.
No extra steps are needed.

```bash
mads-onnx-source --ep coreml ...   # ANE / GPU via MLComputeUnitsAll
mads-onnx-source --ep auto   ...   # auto-picks best available EP
```

### Windows — DirectML

The Windows package ships `onnxruntime-win-x64-gpu` whose `onnxruntime.dll`
has DirectML EP compiled in.  DirectML works on any DirectX 12 capable GPU
(NVIDIA, AMD, Intel) without installing extra drivers.

```ini
[agents]
execution_provider = directml
```

The ~285 MB CUDA sidecar DLLs are **not** included in the package (to keep the
download small).  CPU and DirectML are always available.

### Linux x64 — CUDA (opt-in)

The Linux package ships the **GPU-capable core lib**
(`libonnxruntime.so.1.26.0` built against CUDA 12) plus
`libonnxruntime_providers_shared.so`.  This core lib runs on CPU out of the
box — it does **not** hard-require `libcudart` at load time; the CUDA provider
is dlopen'd lazily only when `execution_provider = cuda` is set.

> **Assumption (to validate on a real Linux box):** The ONNX Runtime GPU core
> lib (`onnxruntime-linux-x64-gpu`) links libcudart as a weak / lazy dependency
> and does NOT fail to load on hosts without CUDA installed.  This is the
> standard ORT design; verify with `ldd libonnxruntime.so.1.26.0` on a
> non-CUDA host if in doubt.

To enable CUDA inference:

1. Install **NVIDIA CUDA Toolkit 12.x** and **cuDNN 9.x** on the host.
2. Run the sidecar installer:
   ```bash
   chmod +x scripts/enable-cuda.sh
   ./scripts/enable-cuda.sh
   ```
   This downloads the `onnxruntime-linux-x64-gpu-1.26.0.tgz` (~215 MB) and
   extracts `libonnxruntime_providers_cuda.so` into `$(mads -p)/lib`.
3. Set `execution_provider = cuda` in `mads.ini [agents]`.

The `libonnxruntime_providers_cuda.so` and `libonnxruntime_providers_tensorrt.so`
sidecars are intentionally **not** included in the base package to keep its
size comparable to the CPU-only build.

---

## Execution providers and thread settings

All three executables (`mads-onnx-filter`, `mads-onnx-source`,
`mads-onnx-inspect`) support configurable execution providers (EPs) and thread
pool sizes.

### Execution providers

| Value | Description |
|---|---|
| `cpu` | CPU-only inference (always available, default) |
| `coreml` | Apple CoreML — uses ANE / GPU via `MLComputeUnitsAll` (macOS/iOS) |
| `webgpu` | WebGPU EP — experimental |
| `cuda` | NVIDIA CUDA — requires a GPU ONNX Runtime package (Linux/Windows) |
| `directml` | DirectX ML — requires a GPU ONNX Runtime package (Windows) |
| `auto` | Pick the best available EP on the current platform |

**Graceful fallback**: if the requested EP is not compiled into the loaded ONNX
Runtime library (or its `AppendExecutionProvider` call fails), a yellow warning
is printed at startup and inference falls back to CPU automatically. The binary
never hard-fails because a GPU EP is missing.

The active EP (after any fallback) is printed in every agent's startup banner
and in `mads-onnx-inspect` output so you can confirm which backend ran.

Platform availability (with the macOS arm64 prebuilt in ORT 1.26.0):

```
Available providers:
  CoreMLExecutionProvider
  WebGpuExecutionProvider
  CPUExecutionProvider
```

CUDA and DirectML are absent from this dylib — they will fall back to CPU with a
warning on macOS. A CUDA-capable package would be used on a Linux GPU build.

### Thread settings

| Setting | Default | Description |
|---|---|---|
| `intra_op_threads` | `0` | Parallelism within a single operator. 0 = ORT default (all cores). |
| `inter_op_threads` | `0` | Parallelism across independent graph nodes. 0 = ORT default. |

The old hardcoded value of `1` intra-op thread was a pessimisation. The new
default of `0` lets ORT use all available cores, which is almost always faster.

### CLI flags (all three executables)

```
--ep <name>             Execution provider (overrides INI execution_provider)
--threads <n>           Intra-op thread count (overrides INI intra_op_threads)
--inter-threads <n>     Inter-op thread count (overrides INI inter_op_threads)
--gpu-device <idx>      GPU device index for cuda/directml (overrides INI gpu_device_id)
```

`mads-onnx-inspect` also supports `-ep` / `--ep` as a short alias.

### INI settings (shared `[agents]` section)

```ini
[agents]
# Execution provider: cpu | coreml | webgpu | cuda | directml | auto
execution_provider = "cpu"

# 0 = let ORT choose (recommended); positive value caps the thread pool.
intra_op_threads = 0
inter_op_threads = 0

# GPU device index (cuda / directml only).
gpu_device_id = 0
```

These can also be overridden per-agent in the `[onnx-filter]` or `[onnx-source]`
sections.

### Examples

```bash
# Use CoreML on macOS (fast path via ANE/GPU):
mads-onnx-source --test -m model.onnx --image cat.jpg --ep coreml

# Let the runtime pick the best available EP:
mads-onnx-source --test -m model.onnx --image cat.jpg --ep auto

# Use 4 CPU threads explicitly:
mads-onnx-filter --test -m model.onnx --input data.json --threads 4

# On a machine without CUDA — falls back to CPU with a warning:
mads-onnx-filter --test -m model.onnx --ep cuda
# [OnnxModel] Warning: execution provider 'cuda' is not available in this
# ONNX Runtime build; falling back to CPU.
```

---

## Tensor ↔ JSON mapping

Every tensor is encoded as a JSON object:

```json
{
  "shape": [2, 3],
  "data":  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
  "dtype": "float32"
}
```

Rules:

- `shape` — array of integers; dynamic (symbolic) dimensions appear as `-1`.
- `data` — flat array in row-major (C) order.
- `dtype` — optional; when absent, inferred from JSON value types:
  - JSON integers → `int32`
  - JSON floats → `float32`
  - JSON strings → `string`
  - JSON booleans → `bool`

Explicit dtype strings: `float32`, `float64`, `int8`, `int16`, `int32`,
`int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bool`, `string`.

A message sent to the filter agent is a JSON object mapping **tensor names** to
tensor objects:

```json
{
  "context":         {"shape": [1, 192], "data": [...], "dtype": "float32"},
  "attention_mask":  {"shape": [1, 192], "data": [...], "dtype": "float32"}
}
```

Use `mads-onnx-inspect --model-path <model.onnx>` (or `mads-onnx-filter --inspect`)
to print the exact schema for any model.

---

## `mads-onnx-inspect` utility

Prints model I/O metadata without starting the MADS network.

```
Usage:
  mads-onnx-inspect [OPTION...]

  -m, --model-path arg     Path to the .onnx model file
  -j, --json               Emit output as JSON rather than human-readable text
      --ep arg             Execution provider (default: cpu)
      --threads arg        Intra-op thread count (0 = ORT default)
      --inter-threads arg  Inter-op thread count (0 = ORT default)
      --gpu-device arg     GPU device index
  -h, --help               Print usage
```

Example:

```bash
mads-onnx-inspect -m examples/imagenet/mobilenetv2-12.onnx
```

Output:

```
Inputs:
  input
    dtype : float32
    shape : [?, 3, 224, 224]

Outputs:
  output
    dtype : float32
    shape : [?, 1000]

Example input JSON schema:
{
  "input": {
    "shape": [1, 3, 224, 224],
    "data": [0.0],
    "dtype": "float32"
  }
}
```

---

## `mads-onnx-filter` — filter agent

### Overview

Subscribes to a topic, receives JSON tensor maps, runs an ONNX model, and
publishes the output tensor map to another topic. Timing is input-driven (each
received message triggers one inference).

No code changes are needed to switch models — only `--model-path` changes.

### CLI

```
Usage:
  mads-onnx-filter [MADS options] [OPTION...]

  -m, --model-path arg  Path to the ONNX model (overrides INI model_path)
      --inspect         Print model I/O metadata and example schema, then exit
      --test            Standalone self-test: run inference on dummy/file input
      --input arg       JSON input file for --test (tensor map matching schema)
      --ep arg          Execution provider (overrides INI execution_provider)
      --threads arg     Intra-op thread count (0 = ORT default)
      --inter-threads arg  Inter-op thread count (0 = ORT default)
      --gpu-device arg  GPU device index
  -n, --name arg        Agent name (overrides INI; sets topic names)
  -h, --help            Print usage
```

### INI configuration (`[onnx-filter]`)

```ini
[onnx-filter]
model_path  = "/path/to/your/model.onnx"   # required
pub_topic   = "onnx-filter"                 # default
sub_topic   = ["onnx-filter-input"]         # default; array
```

### Topic naming

Default sub topic: `onnx-filter-input`
Default pub topic: `onnx-filter`

Override either via the `[onnx-filter]` INI section or `--name` on the CLI
(MADS standard behaviour: `--name foo` sets sub topic to `foo-input` and pub
topic to `foo`).

### Self-test

```bash
# with dummy inputs (ramp data):
mads-onnx-filter --test -m /path/to/model.onnx

# with a JSON file:
mads-onnx-filter --test -m /path/to/model.onnx --input input.json
```

See `examples/chronos/README.md` for a worked Chronos-2 forecasting example.

---

## `mads-onnx-source` — source agent

### Overview

Internally timed agent that reads camera frames, preprocesses them into
float32 NCHW tensors, runs an ONNX classifier, and publishes top-k predictions.
Requires OpenCV to be installed at build time. Without OpenCV the binary is not
built (only `mads-onnx-filter` and `mads-onnx-inspect` are built).

### CLI

```
Usage:
  mads-onnx-source [MADS options] [OPTION...]

  -m, --model-path arg  Path to the ONNX model (overrides INI model_path)
      --inspect         Print model I/O metadata, then exit
      --test            Standalone self-test: classify --image (or dummy frame)
      --image arg       Image file to use in --test
      --camera arg      Camera device index (default: 0)
      --width arg       Camera capture width px (default: 640)
      --height arg       Camera capture height px (default: 480)
      --fps arg         Camera capture FPS hint (default: 30)
  -k, --top-k arg       Number of top predictions (default: 5)
      --period arg      Loop period in ms (default: 100)
      --labels arg      Path to labels file (overrides INI labels_path)
      --ep arg          Execution provider (overrides INI execution_provider)
      --threads arg     Intra-op thread count (0 = ORT default)
      --inter-threads arg  Inter-op thread count (0 = ORT default)
      --gpu-device arg  GPU device index
  -n, --name arg        Agent name
  -h, --help            Print usage
```

### INI configuration (`[onnx-source]`)

```ini
[onnx-source]
model_path      = "examples/imagenet/mobilenetv2-12.onnx"
labels_path     = "examples/imagenet/imagenet_classes.txt"
pub_topic       = "onnx-source"
period          = 100           # ms between frames (~10 fps)
top_k           = 5
camera          = 0
capture_width   = 640
capture_height  = 480
capture_fps     = 30.0

# Image pre-processing constants (match your model's expected normalisation)
input_width     = 224
input_height    = 224
nchw            = true          # NCHW layout (true) or NHWC (false)
rgb             = true          # convert BGR→RGB before normalising
scale           = 0.00392156862745098   # 1/255
mean            = [0.485, 0.456, 0.406]
std             = [0.229, 0.224, 0.225]
input_tensor_name = "input"
```

### Published message format

Each message on `pub_topic` contains:

```json
{
  "best":  { "class_id": 281, "label": "tabby", "confidence": 0.7495 },
  "top_k": [
    { "class_id": 281, "label": "tabby",       "confidence": 0.7495 },
    { "class_id": 285, "label": "Egyptian cat", "confidence": 0.1142 },
    ...
  ]
}
```

### Self-test (verified output)

```bash
mads-onnx-source --test \
  -m examples/imagenet/mobilenetv2-12.onnx \
  --image examples/imagenet/sample_cat.jpg \
  --labels examples/imagenet/imagenet_classes.txt
```

```
=== mads-onnx-source --test ===
Model: examples/imagenet/mobilenetv2-12.onnx
Labels: examples/imagenet/imagenet_classes.txt (1000 classes)
Image: examples/imagenet/sample_cat.jpg (640x480)

Top-5 predictions:
{
  "best": { "class_id": 281, "confidence": 0.7495, "label": "tabby" },
  "top_k": [
    { "class_id": 281, "confidence": 0.7495, "label": "tabby" },
    { "class_id": 285, "confidence": 0.1142, "label": "Egyptian cat" },
    { "class_id": 282, "confidence": 0.1081, "label": "tiger cat" },
    ...
  ]
}
```

See `examples/imagenet/README.md` for full download and setup instructions.

---

## Topic naming rules

Both agents follow the standard MADS convention:

- The INI section name is the executable name minus the `mads-` prefix,
  with dots replaced by dashes:
  - `mads-onnx-filter` → `[onnx-filter]`
  - `mads-onnx-source` → `[onnx-source]`
- The default pub topic equals the section name (`onnx-filter`, `onnx-source`).
- The default sub topic for the filter agent is `onnx-filter-input`.
- `--name foo` overrides the section lookup and sets custom topic names.

---

## Examples

### Chronos-2 time-series forecasting (`mads-onnx-filter`)

See [`examples/chronos/README.md`](examples/chronos/README.md) for:
- Model download and graph-bug patch (`fix_model.py`)
- CSV → input JSON conversion (`make_input.py`)
- Step-by-step `--test` and live-broker usage

Forecast output shape: `[1, 21, 16]` — 1 batch × 21 quantile levels × 16
future steps. Index 10 (q=0.50) is the median.

### MobileNetV2 ImageNet classification (`mads-onnx-source`)

See [`examples/imagenet/README.md`](examples/imagenet/README.md) for:
- Model and label download script
- Image pre-processing constants (resize to 224×224, BGR→RGB, ImageNet
  mean/std normalisation, NCHW layout)
- Live webcam usage with `mads.ini`

---

## Extending the source agent

Implement `Onnx::DataSource` (see `src/source/DataSource.hpp`) and pass an
instance to the agent loop. The interface is three virtual methods:

```cpp
bool open();                           // acquire hardware resources
void close();                          // release them
std::optional<nlohmann::json> next();  // produce one input tensor map
```

The default `CameraSource` can serve as a reference implementation.

---

## API documentation

Key public classes (all in namespace `Onnx`, headers under `src/`):

| Class / function | Header | Description |
|---|---|---|
| `OnnxModel` | `onnx/OnnxModel.hpp` | RAII model wrapper; load, inspect, run |
| `TensorInfo` | `onnx/OnnxModel.hpp` | Tensor name, shape, dtype metadata |
| `json_to_tensor()` | `onnx/TensorJson.hpp` | JSON tensor object → `Ort::Value` |
| `tensor_to_json()` | `onnx/TensorJson.hpp` | `Ort::Value` → JSON tensor object |
| `make_input_schema()` | `onnx/TensorJson.hpp` | Generate example input JSON from model |
| `DataSource` | `source/DataSource.hpp` | Pure-virtual data source interface |
| `CameraSource` | `source/CameraSource.hpp` | OpenCV webcam implementation |
| `PreprocessConfig` | `source/CameraSource.hpp` | Image preprocessing constants |

All public APIs have Doxygen documentation in the headers.

---

## License

See [LICENSE](LICENSE).
