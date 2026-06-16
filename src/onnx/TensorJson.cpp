/**
 * @file TensorJson.cpp
 * @brief Implementation of JSON ↔ tensor conversion.
 */

#include "TensorJson.hpp"

#include <numeric>
#include <stdexcept>

namespace Onnx {

// ─── helpers ─────────────────────────────────────────────────────────────────

ONNXTensorElementDataType dtype_from_string(const std::string &s) {
  if (s == "float32" || s == "float")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  if (s == "float64" || s == "double")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  if (s == "int8")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  if (s == "int16")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  if (s == "int32" || s == "int")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  if (s == "int64")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  if (s == "uint8")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  if (s == "uint16")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  if (s == "uint32")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
  if (s == "uint64")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
  if (s == "bool")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  if (s == "string")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  if (s == "float16")
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  throw std::invalid_argument("Unknown dtype: " + s);
}

/// Infer dtype from the first element of a JSON array.
static ONNXTensorElementDataType infer_dtype(const nlohmann::json &data_arr) {
  if (data_arr.empty())
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  const auto &first = data_arr[0];
  if (first.is_boolean())
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  if (first.is_string())
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  if (first.is_number_integer())
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  if (first.is_number_float())
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  // Complex object {real, imag} – not supported for inference; caller must
  // supply dtype explicitly.
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

// ─── json_to_tensor ───────────────────────────────────────────────────────────

OwnedTensor json_to_tensor(const nlohmann::json &j,
                           ONNXTensorElementDataType hint) {
  if (!j.contains("shape") || !j.contains("data")) {
    throw std::invalid_argument(
        "Tensor JSON must have 'shape' and 'data' fields");
  }

  const auto &shape_j = j["shape"];
  const auto &data_j = j["data"];

  if (!shape_j.is_array() || !data_j.is_array()) {
    throw std::invalid_argument("'shape' and 'data' must be JSON arrays");
  }

  // Build shape vector
  std::vector<int64_t> shape;
  shape.reserve(shape_j.size());
  for (const auto &d : shape_j) {
    shape.push_back(d.get<int64_t>());
  }

  // Determine dtype
  ONNXTensorElementDataType dtype =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (j.contains("dtype") && j["dtype"].is_string()) {
    dtype = dtype_from_string(j["dtype"].get<std::string>());
  } else if (hint != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    dtype = hint;
  } else {
    dtype = infer_dtype(data_j);
  }

  const size_t n = data_j.size();

  // Validate shape vs data length: replace -1 dims with inferred value
  int64_t known_product = 1;
  int dyn_count = 0;
  for (int64_t d : shape) {
    if (d < 0)
      ++dyn_count;
    else
      known_product *= d;
  }
  if (dyn_count == 0 && known_product != static_cast<int64_t>(n)) {
    throw std::runtime_error(
        "Data length (" + std::to_string(n) +
        ") does not match shape product (" + std::to_string(known_product) +
        ")");
  }
  // Replace exactly one dynamic dim
  if (dyn_count == 1 && known_product > 0) {
    for (auto &d : shape) {
      if (d < 0) {
        d = static_cast<int64_t>(n) / known_product;
        break;
      }
    }
  }

  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  OwnedTensor ot;

  switch (dtype) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
    ot.f32_buf.reserve(n);
    for (auto &v : data_j)
      ot.f32_buf.push_back(v.get<float>());
    ot.tensor = Ort::Value::CreateTensor<float>(
        mem_info, ot.f32_buf.data(), n, shape.data(), shape.size());
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
    ot.f64_buf.reserve(n);
    for (auto &v : data_j)
      ot.f64_buf.push_back(v.get<double>());
    ot.tensor = Ort::Value::CreateTensor<double>(
        mem_info, ot.f64_buf.data(), n, shape.data(), shape.size());
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
    ot.i8_buf.reserve(n);
    for (auto &v : data_j)
      ot.i8_buf.push_back(static_cast<int8_t>(v.get<int>()));
    ot.tensor = Ort::Value::CreateTensor<int8_t>(
        mem_info, ot.i8_buf.data(), n, shape.data(), shape.size());
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
    ot.i16_buf.reserve(n);
    for (auto &v : data_j)
      ot.i16_buf.push_back(static_cast<int16_t>(v.get<int>()));
    ot.tensor = Ort::Value::CreateTensor<int16_t>(
        mem_info, ot.i16_buf.data(), n, shape.data(), shape.size());
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
    ot.i32_buf.reserve(n);
    for (auto &v : data_j)
      ot.i32_buf.push_back(v.get<int32_t>());
    ot.tensor = Ort::Value::CreateTensor<int32_t>(
        mem_info, ot.i32_buf.data(), n, shape.data(), shape.size());
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
    ot.i64_buf.reserve(n);
    for (auto &v : data_j)
      ot.i64_buf.push_back(v.get<int64_t>());
    ot.tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, ot.i64_buf.data(), n, shape.data(), shape.size());
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
    ot.u8_buf.reserve(n);
    for (auto &v : data_j)
      ot.u8_buf.push_back(v.get<uint8_t>());
    ot.tensor = Ort::Value::CreateTensor<uint8_t>(
        mem_info, ot.u8_buf.data(), n, shape.data(), shape.size());
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
    ot.u16_buf.reserve(n);
    for (auto &v : data_j)
      ot.u16_buf.push_back(v.get<uint16_t>());
    ot.tensor = Ort::Value::CreateTensor<uint16_t>(
        mem_info, ot.u16_buf.data(), n, shape.data(), shape.size());
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
    ot.u32_buf.reserve(n);
    for (auto &v : data_j)
      ot.u32_buf.push_back(v.get<uint32_t>());
    ot.tensor = Ort::Value::CreateTensor<uint32_t>(
        mem_info, ot.u32_buf.data(), n, shape.data(), shape.size());
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
    ot.u64_buf.reserve(n);
    for (auto &v : data_j)
      ot.u64_buf.push_back(v.get<uint64_t>());
    ot.tensor = Ort::Value::CreateTensor<uint64_t>(
        mem_info, ot.u64_buf.data(), n, shape.data(), shape.size());
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
    // bool specialization not always available; use uint8_t buffer
    ot.u8_buf.reserve(n);
    for (auto &v : data_j)
      ot.u8_buf.push_back(v.get<bool>() ? 1u : 0u);
    ot.tensor = Ort::Value::CreateTensor(mem_info, ot.u8_buf.data(),
                                         n * sizeof(uint8_t), shape.data(),
                                         shape.size(),
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
    ot.str_buf.reserve(n);
    for (auto &v : data_j)
      ot.str_buf.push_back(v.get<std::string>());
    // String tensors require a special creation path
    Ort::AllocatorWithDefaultOptions alloc;
    ot.tensor = Ort::Value::CreateTensor(alloc, shape.data(), shape.size(),
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    for (size_t i = 0; i < n; ++i) {
      ot.tensor.FillStringTensorElement(ot.str_buf[i].c_str(), i);
    }
    break;
  }
  default:
    throw std::invalid_argument("Unsupported dtype for JSON→tensor conversion");
  }

  return ot;
}

// ─── tensor_to_json ───────────────────────────────────────────────────────────

nlohmann::json tensor_to_json(const Ort::Value &tensor) {
  if (!tensor.IsTensor()) {
    throw std::runtime_error("Ort::Value is not a tensor");
  }

  auto type_info = tensor.GetTensorTypeAndShapeInfo();
  const auto shape = type_info.GetShape();
  const auto dtype = type_info.GetElementType();
  const size_t n = type_info.GetElementCount();

  nlohmann::json j;
  j["shape"] = shape;

  TensorInfo ti;
  ti.dtype = dtype;
  j["dtype"] = ti.dtype_name();

  switch (dtype) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
    const float *ptr = tensor.GetTensorData<float>();
    j["data"] = std::vector<float>(ptr, ptr + n);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
    const double *ptr = tensor.GetTensorData<double>();
    j["data"] = std::vector<double>(ptr, ptr + n);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
    const int8_t *ptr = tensor.GetTensorData<int8_t>();
    j["data"] = std::vector<int8_t>(ptr, ptr + n);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
    const int16_t *ptr = tensor.GetTensorData<int16_t>();
    j["data"] = std::vector<int16_t>(ptr, ptr + n);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
    const int32_t *ptr = tensor.GetTensorData<int32_t>();
    j["data"] = std::vector<int32_t>(ptr, ptr + n);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
    const int64_t *ptr = tensor.GetTensorData<int64_t>();
    j["data"] = std::vector<int64_t>(ptr, ptr + n);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
    const uint8_t *ptr = tensor.GetTensorData<uint8_t>();
    j["data"] = std::vector<uint8_t>(ptr, ptr + n);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
    const uint16_t *ptr = tensor.GetTensorData<uint16_t>();
    j["data"] = std::vector<uint16_t>(ptr, ptr + n);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
    const uint32_t *ptr = tensor.GetTensorData<uint32_t>();
    j["data"] = std::vector<uint32_t>(ptr, ptr + n);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
    const uint64_t *ptr = tensor.GetTensorData<uint64_t>();
    j["data"] = std::vector<uint64_t>(ptr, ptr + n);
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
    const bool *ptr = tensor.GetTensorData<bool>();
    nlohmann::json arr = nlohmann::json::array();
    for (size_t i = 0; i < n; ++i)
      arr.push_back(ptr[i]);
    j["data"] = arr;
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
    nlohmann::json arr = nlohmann::json::array();
    for (size_t i = 0; i < n; ++i) {
      size_t len = tensor.GetStringTensorElementLength(i);
      std::string s(len, '\0');
      tensor.GetStringTensorElement(len, i, s.data());
      arr.push_back(s);
    }
    j["data"] = arr;
    break;
  }
  default: {
    j["data"] = nlohmann::json::array();
    j["note"] = "unsupported dtype, data omitted";
    break;
  }
  }

  return j;
}

// ─── make_input_schema ────────────────────────────────────────────────────────

nlohmann::json make_input_schema(const std::vector<TensorInfo> &inputs) {
  nlohmann::json schema = nlohmann::json::object();
  for (const auto &info : inputs) {
    nlohmann::json t;
    t["shape"] = info.shape;
    t["dtype"] = info.dtype_name();

    // Emit one example element per dtype
    nlohmann::json example_elem;
    switch (info.dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      example_elem = 0.0;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      example_elem = 0;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      example_elem = false;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      example_elem = "";
      break;
    default:
      example_elem = nullptr;
      break;
    }

    // data array: one element placeholder (shape may contain -1)
    int64_t count = info.element_count();
    if (count <= 0)
      count = 1;
    t["data"] = nlohmann::json::array();
    for (int64_t i = 0; i < std::min(count, static_cast<int64_t>(8)); ++i)
      t["data"].push_back(example_elem);
    if (count > 8)
      t["data_note"] = "showing first 8 of " + std::to_string(count) +
                       " elements (shape may be dynamic)";

    schema[info.name] = t;
  }
  return schema;
}

} // namespace Onnx
