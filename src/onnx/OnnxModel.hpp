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
   * @brief Load a model from the given path.
   *
   * @param path  Path to the .onnx model file.
   * @throws std::runtime_error if the file does not exist or cannot be loaded.
   */
  explicit OnnxModel(const std::filesystem::path &path);

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

private:
  void _load_metadata();

  std::filesystem::path _path;
  Ort::Env _env;
  Ort::Session _session;
  Ort::AllocatorWithDefaultOptions _allocator;
  std::vector<TensorInfo> _inputs;
  std::vector<TensorInfo> _outputs;
  // Cached name C-strings for Run()
  std::vector<std::string> _input_names;
  std::vector<std::string> _output_names;
};

} // namespace Onnx
