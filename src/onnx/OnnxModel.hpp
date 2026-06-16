/**
 * @file OnnxModel.hpp
 * @brief RAII wrapper around the ONNX Runtime C++ API.
 *
 * Provides model loading, input/output metadata inspection, and single-call
 * inference over collections of named Ort::Value tensors.
 *
 * @author Paolo Bosetti
 * @date 2026
 */

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace Onnx {

/**
 * @brief Metadata describing a single input or output tensor of a model.
 */
struct TensorInfo {
  /** @brief Tensor name as stored in the ONNX graph. */
  std::string name;

  /**
   * @brief Tensor shape. Dynamic (symbolic) dimensions are represented as -1.
   *
   * An empty vector means the rank is unknown.
   */
  std::vector<int64_t> shape;

  /** @brief ONNX element type enum. */
  ONNXTensorElementDataType dtype;

  /**
   * @brief Return the dtype as a human-readable string (e.g. "float32").
   */
  std::string dtype_name() const;

  /**
   * @brief Return the total number of elements implied by shape.
   *
   * Dynamic dimensions (-1) are counted as 1 for the purpose of this
   * estimate. Returns 0 if shape is empty.
   */
  int64_t element_count() const;
};

/**
 * @brief Session configuration options for OnnxModel.
 *
 * Controls thread counts and execution provider selection. Pass a default-
 * constructed instance to get the same behaviour as the legacy path-only
 * constructor (ORT-chosen thread count, CPU execution provider).
 */
struct SessionConfig {
  /**
   * @brief Requested execution provider.
   *
   * Supported values (case-insensitive):
   *   - "cpu"      — CPU only (default, always available).
   *   - "coreml"   — Apple CoreML (macOS/iOS, requires ANE/GPU-capable device).
   *   - "cuda"     — NVIDIA CUDA (Linux/Windows GPU builds only).
   *   - "directml" — DirectX ML (Windows GPU builds only).
   *   - "webgpu"   — WebGPU EP (experimental).
   *   - "auto"     — pick the best available EP on the current platform.
   *
   * If the requested EP is unavailable a warning is printed and inference
   * falls back to CPU automatically.
   */
  std::string execution_provider = "cpu";

  /**
   * @brief Number of intra-op threads (parallelism within a single operator).
   *
   * 0 = let ORT choose (recommended; uses all available cores). Positive
   * values cap the thread count. The old hardcoded value of 1 was a
   * pessimisation; 0 is the new default.
   */
  int intra_op_threads = 0;

  /**
   * @brief Number of inter-op threads (parallelism across independent nodes).
   *
   * 0 = let ORT choose. Seldom needs tuning unless the graph has many
   * independent branches.
   */
  int inter_op_threads = 0;

  /**
   * @brief GPU device index used when execution_provider is "cuda".
   *
   * Ignored for all non-GPU providers.
   */
  int gpu_device_id = 0;
};

/**
 * @brief RAII model wrapper for ONNX Runtime.
 *
 * Loads a .onnx file, exposes input/output metadata, and runs inference.
 * Thread-hostile: create one instance per thread, or serialize access.
 *
 * @code
 * Onnx::OnnxModel model("my_model.onnx");
 * auto inputs  = model.input_info();
 * auto outputs = model.output_info();
 * std::vector<Ort::Value> in_tensors = ...;
 * auto out_tensors = model.run(in_tensors);
 * @endcode
 */
class OnnxModel {
public:
  /**
   * @brief Load a model from the given path using default CPU settings.
   *
   * Equivalent to OnnxModel(path, SessionConfig{}).
   *
   * @param path  Path to the .onnx model file.
   * @throws std::runtime_error if the file does not exist or cannot be loaded.
   */
  explicit OnnxModel(const std::filesystem::path &path);

  /**
   * @brief Load a model with explicit session configuration.
   *
   * @param path    Path to the .onnx model file.
   * @param config  Thread counts and EP selection.
   * @throws std::runtime_error if the file does not exist or cannot be loaded.
   */
  OnnxModel(const std::filesystem::path &path, const SessionConfig &config);

  OnnxModel(const OnnxModel &) = delete;
  OnnxModel &operator=(const OnnxModel &) = delete;
  OnnxModel(OnnxModel &&) = default;
  OnnxModel &operator=(OnnxModel &&) = default;
  ~OnnxModel() = default;

  /**
   * @brief Return metadata for all model inputs, in graph order.
   */
  const std::vector<TensorInfo> &input_info() const { return _inputs; }

  /**
   * @brief Return metadata for all model outputs, in graph order.
   */
  const std::vector<TensorInfo> &output_info() const { return _outputs; }

  /**
   * @brief Run inference.
   *
   * @param inputs  One Ort::Value per model input, in the same order as
   *                input_info().
   * @return        One Ort::Value per model output.
   * @throws Ort::Exception on inference error.
   */
  std::vector<Ort::Value> run(std::vector<Ort::Value> &inputs);

  /**
   * @brief Return the filesystem path the model was loaded from.
   */
  const std::filesystem::path &path() const { return _path; }

  /**
   * @brief Return the execution provider that is actually active.
   *
   * This reflects the real EP used after fallback. For example, if "coreml"
   * was requested but unavailable the session runs on CPU and this returns
   * "CPU". Possible values: "CPU", "CoreML", "CUDA", "DirectML", "WebGPU".
   */
  const std::string &active_provider() const { return _active_provider; }

private:
  void _load_metadata();

  /// Apply EP selection to opts; sets _active_provider.
  void _apply_ep(Ort::SessionOptions &opts, const SessionConfig &config);

  std::filesystem::path _path;
  Ort::Env _env;
  Ort::Session _session;
  Ort::AllocatorWithDefaultOptions _allocator;
  std::vector<TensorInfo> _inputs;
  std::vector<TensorInfo> _outputs;
  // Cached name C-strings for Run()
  std::vector<std::string> _input_names;
  std::vector<std::string> _output_names;

  std::string _active_provider{"CPU"};
};

} // namespace Onnx
