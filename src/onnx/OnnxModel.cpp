/**
 * @file OnnxModel.cpp
 * @brief Implementation of OnnxModel.
 */

#include "OnnxModel.hpp"

#include <numeric>
#include <stdexcept>
#include <string>

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

// ─── OnnxModel ────────────────────────────────────────────────────────────────

OnnxModel::OnnxModel(const std::filesystem::path &path)
    : _path(path),
      _env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel"),
      _session(nullptr) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Model file not found: " + path.string());
  }

  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

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
