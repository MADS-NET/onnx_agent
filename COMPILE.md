# Build, configure, and package instructions

All commands are run from the project root.

---

## macOS (arm64 or x86_64)

**Prerequisites:** Xcode command-line tools, CMake >= 3.28, Ninja, MADS installed.
OpenCV is optional (via Homebrew: `brew install opencv`).

```bash
# Developer build (no installation)
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu)

# Run tests
./build/src/main/mads-onnx-inspect -m examples/imagenet/mobilenetv2-12.onnx
./build/src/main/mads-onnx-source --test \
  -m examples/imagenet/mobilenetv2-12.onnx \
  --image examples/imagenet/sample_cat.jpg \
  --labels examples/imagenet/imagenet_classes.txt

# Self-contained distributable package
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DMADS_INSTALL_AGENT=ON \
      -DCMAKE_INSTALL_PREFIX="$PWD/package"
cmake --build build -j$(sysctl -n hw.logicalcpu)
cmake --build build --target package

# Package lands at:  build/onnx_agent-<version>-darwin-arm64.zip
# Test extraction:
unzip build/onnx_agent-*-darwin-arm64.zip -d /tmp/pkgtest
/tmp/pkgtest/onnx_agent-*/bin/mads-onnx-inspect          # must print usage and exit 0
otool -L /tmp/pkgtest/onnx_agent-*/bin/mads-onnx-inspect  # must show @rpath/libonnxruntime.1.dylib
```

### Install to MADS prefix

```bash
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release -DMADS_INSTALL_AGENT=ON
cmake --build build --target install
# Installs to $(mads -p)/bin/ and $(mads -p)/lib/
```

Remove quarantine on first run (macOS Gatekeeper):

```bash
xattr -d com.apple.quarantine "$(mads -p)/bin/mads-onnx-filter"
xattr -d com.apple.quarantine "$(mads -p)/bin/mads-onnx-source"
xattr -d com.apple.quarantine "$(mads -p)/lib/libonnxruntime.1.dylib"
```

---

## Linux (x86_64 or aarch64)

**Prerequisites:** Clang, CMake >= 3.28, Ninja, pkg-config, MADS installed.
OpenCV is optional (`sudo apt install libopencv-dev`).

```bash
export CC=clang CXX=clang++

# Developer build
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Self-contained distributable package
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DMADS_INSTALL_AGENT=ON \
      -DCMAKE_INSTALL_PREFIX="$PWD/package"
cmake --build build -j$(nproc)
cmake --build build --target package

# Package lands at:  build/onnx_agent-<version>-linux-x86_64.zip
#                or: build/onnx_agent-<version>-linux-aarch64.zip

# Test extraction:
unzip build/onnx_agent-*-linux-*.zip -d /tmp/pkgtest
/tmp/pkgtest/onnx_agent-*/bin/mads-onnx-inspect
ldd /tmp/pkgtest/onnx_agent-*/bin/mads-onnx-inspect  # libonnxruntime via $ORIGIN/../lib
```

### Install to MADS prefix

```bash
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release -DMADS_INSTALL_AGENT=ON
cmake --build build --target install
```

---

## Windows (x64, Visual Studio 2022 / MSVC)

Open a **Developer Command Prompt for VS 2022** (or run `vcvarsall.bat amd64`)
before running CMake.

**Prerequisites:** CMake >= 3.28, Ninja, MADS installed.
OpenCV is optional and must be installed such that `find_package(OpenCV)` works
(e.g. via vcpkg: `vcpkg install opencv4:x64-windows`).

```powershell
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Self-contained distributable package
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release `
      -DMADS_INSTALL_AGENT=ON `
      "-DCMAKE_INSTALL_PREFIX=$env:GITHUB_WORKSPACE\package"
cmake --build build --parallel
cmake --build build --target package

# Package lands at:  build\onnx_agent-<version>-windows-amd64.zip
# Contents: bin\mads-onnx-inspect.exe, bin\mads-onnx-filter.exe,
#           bin\onnxruntime.dll (and mads-onnx-source.exe if OpenCV found)
```

### MADS discovery on Windows

`mads -p` is used to discover the MADS prefix. Alternatively:

```powershell
cmake -Bbuild -GNinja -DMADS_ROOT="C:\path\to\mads" ...
```

---

## GPU-capable builds and the ONNX_CPU_ONLY knob

By default (`ONNX_CPU_ONLY=OFF`) CMake downloads a **GPU-capable** ONNX
Runtime prebuilt for each platform:

| Platform | Archive fetched | GPU EP available |
|---|---|---|
| macOS arm64 | `onnxruntime-osx-arm64-<ver>.tgz` | CoreML, WebGPU (already in standard tgz) |
| Linux x64 | `onnxruntime-linux-x64-gpu-<ver>.tgz` | CUDA (sidecar opt-in via `enable-cuda.sh`) |
| Linux aarch64 | `onnxruntime-linux-aarch64-<ver>.tgz` | CPU only (no gpu build for generic aarch64) |
| Windows x64 | `onnxruntime-win-x64-gpu-<ver>.zip` | DirectML (built into `onnxruntime.dll`) |

The large CUDA/TensorRT sidecar libs are **not** installed into the package
(they are excluded in the CMake install block).  On Linux x64, run
`scripts/enable-cuda.sh` to fetch and install the CUDA provider sidecar into
`$(mads -p)/lib` after deployment.

To opt out of the GPU archive (e.g. for a quick developer build where you do
not need GPU EPs and want a faster/smaller download):

```bash
cmake -Bbuild -DONNX_CPU_ONLY=ON ...
```

This falls back to the compact CPU-only archives on Linux x64 and Windows x64.
macOS and Linux aarch64 are unaffected (they always use the same archive).

## Overriding ONNX Runtime version

```bash
cmake -Bbuild ... -DORT_VERSION=1.26.0
```

The prebuilt archive is downloaded from the official ONNX Runtime GitHub
releases. Supported platforms: `osx-arm64`, `linux-x64`, `linux-x64-gpu`,
`linux-aarch64`, `win-x64`, `win-x64-gpu`.

---

## Version from git tag

The project version is read from the most recent `v*.*.*` git tag:

```bash
git tag v1.0.0
cmake -Bbuild ...   # version will be 1.0.0
```

Override manually (useful in CI without full git history):

```bash
cmake -Bbuild -DVERSION=1.0.0 ...
```

---

## Build directory layout (after build)

```
build/
  src/main/
    mads-onnx-inspect
    mads-onnx-filter
    mads-onnx-source          (if OpenCV found)
  _deps/
    onnxruntime_prebuilt-src/ (downloaded ORT prebuilt)
    cxxopts-src/              (downloaded cxxopts)
  onnx_agent-<ver>-<os>-<arch>.zip   (after 'package' target)
```
