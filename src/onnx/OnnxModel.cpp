/**
 * @file OnnxModel.cpp
 * @brief Implementation of OnnxModel.
 */

#include "OnnxModel.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

// rang for coloured warnings (header-only, available via the build system)
#include <rang.hpp>

// CoreML provider option keys (defines kCoremlProviderOption_* constants).
// This header ships only in the macOS ONNX Runtime package; on Linux/Windows
// it is absent, so include it conditionally and fall back to the literal key.
// The CoreML EP branch only executes when CoreMLExecutionProvider is actually
// available (macOS), so the literal is never used at runtime off-Apple.
#if __has_include(<coreml_provider_factory.h>)
#include <coreml_provider_factory.h>
#else
#define kCoremlProviderOption_MLComputeUnits "MLComputeUnits"
#endif

namespace Onnx {

// ─── TensorInfo helpers ───────────────────────────────────────────────────────

std::string TensorInfo::dtype_name() const {
  switch (dtype) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "float32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return "float64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return "int8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return "int16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return "int32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return "int64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return "uint8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return "uint16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return "uint32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return "uint64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return "bool";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    return "string";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return "float16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    return "complex64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    return "complex128";
  default:
    return "unknown";
  }
}

int64_t TensorInfo::element_count() const {
  if (shape.empty())
    return 0;
  int64_t n = 1;
  for (int64_t d : shape)
    n *= (d > 0 ? d : 1);
  return n;
}

// ─── EP selection helper ──────────────────────────────────────────────────────

/// Return true if name appears in the available-providers list (exact match).
static bool ep_available(const std::vector<std::string> &avail,
                         const std::string &name) {
  return std::find(avail.begin(), avail.end(), name) != avail.end();
}

/// Lowercase a copy of s.
static std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

void OnnxModel::_apply_ep(Ort::SessionOptions &opts,
                          const SessionConfig &config) {
  const auto avail = Ort::GetAvailableProviders();
  const std::string req = to_lower(config.execution_provider);

  // Helper: print a yellow warning and fall back to CPU.
  auto warn_fallback = [&](const std::string &requested,
                           const std::string &reason) {
    std::cerr << rang::fg::yellow
              << "[OnnxModel] Warning: execution provider '" << requested
              << "' " << reason << "; falling back to CPU.\n"
              << rang::fg::reset;
    _active_provider = "CPU";
  };

  if (req == "cpu") {
    _active_provider = "CPU";
    return; // CPU is always the implicit fallback — nothing to append.
  }

  if (req == "coreml") {
    const std::string ep_name = "CoreMLExecutionProvider";
    if (!ep_available(avail, ep_name)) {
      warn_fallback("coreml", "is not available in this ONNX Runtime build");
      return;
    }
    try {
      opts.AppendExecutionProvider(
          "CoreML",
          {{kCoremlProviderOption_MLComputeUnits, "ALL"}});
      _active_provider = "CoreML";
    } catch (const Ort::Exception &e) {
      warn_fallback("coreml", std::string("threw: ") + e.what());
    }
    return;
  }

  if (req == "webgpu") {
    const std::string ep_name = "WebGpuExecutionProvider";
    if (!ep_available(avail, ep_name)) {
      warn_fallback("webgpu", "is not available in this ONNX Runtime build");
      return;
    }
    try {
      opts.AppendExecutionProvider("WebGPU", {});
      _active_provider = "WebGPU";
    } catch (const Ort::Exception &e) {
      warn_fallback("webgpu", std::string("threw: ") + e.what());
    }
    return;
  }

  if (req == "cuda") {
    const std::string ep_name = "CUDAExecutionProvider";
    if (!ep_available(avail, ep_name)) {
      warn_fallback("cuda", "is not available in this ONNX Runtime build");
      return;
    }
    try {
      // Use the V2 API (does not require the legacy C symbol that is absent
      // from CPU-only / CoreML-only dylibs).
      OrtCUDAProviderOptionsV2 *cuda_opts = nullptr;
      Ort::ThrowOnError(
          Ort::GetApi().CreateCUDAProviderOptions(&cuda_opts));
      // device_id is a string option in the V2 API. Keep the string alive.
      const std::string device_id_str = std::to_string(config.gpu_device_id);
      const char *keys[]   = {"device_id"};
      const char *values[] = {device_id_str.c_str()};
      Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
          cuda_opts, keys, values, 1));
      opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
      Ort::GetApi().ReleaseCUDAProviderOptions(cuda_opts);
      _active_provider = "CUDA";
    } catch (const Ort::Exception &e) {
      warn_fallback("cuda", std::string("threw: ") + e.what());
    }
    return;
  }

  if (req == "directml") {
    const std::string ep_name = "DmlExecutionProvider";
    if (!ep_available(avail, ep_name)) {
      warn_fallback("directml",
                    "is not available in this ONNX Runtime build");
      return;
    }
    try {
      // DirectML via the generic string-key API (avoids the DML-specific C
      // symbol that is absent from non-Windows / non-DML dylibs).
      opts.AppendExecutionProvider(
          "DML",
          {{"device_id", std::to_string(config.gpu_device_id)}});
      _active_provider = "DirectML";
    } catch (const Ort::Exception &e) {
      warn_fallback("directml", std::string("threw: ") + e.what());
    }
    return;
  }

  if (req == "auto") {
    // Platform-aware priority order:
    //   macOS   : CoreML > WebGPU > CPU
    //   Linux   : CUDA   > WebGPU > CPU
    //   Windows : DirectML > CUDA > WebGPU > CPU
    const std::vector<std::pair<std::string, std::string>> priority = {
        {"CoreMLExecutionProvider",  "coreml"},
        {"CUDAExecutionProvider",    "cuda"},
        {"DmlExecutionProvider",     "directml"},
        {"WebGpuExecutionProvider",  "webgpu"},
    };

    for (const auto &[ep_key, ep_alias] : priority) {
      if (ep_available(avail, ep_key)) {
        // Re-enter via a sub-config so we reuse the same logic.
        SessionConfig sub = config;
        sub.execution_provider = ep_alias;
        _apply_ep(opts, sub);
        return;
      }
    }
    // Nothing available beyond CPU — that is fine.
    _active_provider = "CPU";
    return;
  }

  // Unknown provider name
  std::cerr << rang::fg::yellow << "[OnnxModel] Warning: unknown provider '"
            << config.execution_provider
            << "'; falling back to CPU. Valid values: cpu, coreml, cuda, "
               "directml, webgpu, auto.\n"
            << rang::fg::reset;
  _active_provider = "CPU";
}

// ─── OnnxModel ────────────────────────────────────────────────────────────────

OnnxModel::OnnxModel(const std::filesystem::path &path)
    : OnnxModel(path, SessionConfig{}) {}

OnnxModel::OnnxModel(const std::filesystem::path &path,
                     const SessionConfig &config)
    : _path(path),
      _env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel"),
      _session(nullptr) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Model file not found: " + path.string());
  }

  Ort::SessionOptions opts;

  // Thread settings — only call the setter when the user explicitly asks for
  // a specific count; otherwise leave ORT's default (all available cores).
  if (config.intra_op_threads > 0) {
    opts.SetIntraOpNumThreads(config.intra_op_threads);
  }
  if (config.inter_op_threads > 0) {
    opts.SetInterOpNumThreads(config.inter_op_threads);
  }

  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // Execution provider selection (sets _active_provider).
  _apply_ep(opts, config);

#if defined(_WIN32)
  // Windows: wstring path
  _session = Ort::Session(_env, path.wstring().c_str(), opts);
#else
  _session = Ort::Session(_env, path.string().c_str(), opts);
#endif

  _load_metadata();
}

void OnnxModel::_load_metadata() {
  const size_t num_inputs = _session.GetInputCount();
  const size_t num_outputs = _session.GetOutputCount();

  _inputs.reserve(num_inputs);
  _input_names.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    TensorInfo info;
    // Name
    auto name_ptr = _session.GetInputNameAllocated(i, _allocator);
    info.name = name_ptr.get();
    _input_names.push_back(info.name);

    // Type/shape
    auto type_info = _session.GetInputTypeInfo(i);
    if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      info.dtype = tensor_info.GetElementType();
      info.shape = tensor_info.GetShape();
    } else {
      info.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    _inputs.push_back(std::move(info));
  }

  _outputs.reserve(num_outputs);
  _output_names.reserve(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    TensorInfo info;
    auto name_ptr = _session.GetOutputNameAllocated(i, _allocator);
    info.name = name_ptr.get();
    _output_names.push_back(info.name);

    auto type_info = _session.GetOutputTypeInfo(i);
    if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      info.dtype = tensor_info.GetElementType();
      info.shape = tensor_info.GetShape();
    } else {
      info.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    _outputs.push_back(std::move(info));
  }
}

std::vector<Ort::Value> OnnxModel::run(std::vector<Ort::Value> &inputs) {
  // Build raw pointer arrays for the C API
  std::vector<const char *> in_names;
  std::vector<const char *> out_names;
  in_names.reserve(_input_names.size());
  out_names.reserve(_output_names.size());
  for (auto &n : _input_names)
    in_names.push_back(n.c_str());
  for (auto &n : _output_names)
    out_names.push_back(n.c_str());

  Ort::RunOptions run_opts;
  return _session.Run(run_opts, in_names.data(), inputs.data(), inputs.size(),
                      out_names.data(), out_names.size());
}

} // namespace Onnx
