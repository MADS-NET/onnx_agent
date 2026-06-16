/**
 * @file TensorJson.hpp
 * @brief Bidirectional conversion between nlohmann::json and ONNX Runtime
 *        tensors (Ort::Value).
 *
 * JSON tensor format:
 * @code
 * {
 *   "shape": [2, 3],
 *   "data":  [1, 2, 3, 4, 5, 6],
 *   "dtype": "float32"          // optional; inferred when absent
 * }
 * @endcode
 *
 * Dtype inference rules (when "dtype" is absent):
 *  - JSON integers  → int32
 *  - JSON floats    → float32
 *  - JSON strings   → string
 *  - JSON booleans  → bool
 *
 * Supported dtype strings:
 *   float32, float64, int8, int16, int32, int64,
 *   uint8, uint16, uint32, uint64, bool, string
 *
 * @author Paolo Bosetti
 * @date 2026
 */

#pragma once

#include "OnnxModel.hpp"

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace Onnx {

/**
 * @brief Convert a JSON tensor object to an Ort::Value.
 *
 * The JSON object must contain at least "shape" and "data" arrays. "dtype" is
 * optional and overrides inference. Dynamic dimensions (-1) in the provided
 * shape must be compatible with the flat data length.
 *
 * @param j        JSON object with keys "shape", "data", and optional "dtype".
 * @param info     Optional TensorInfo used to resolve dtype when the JSON
 *                 "dtype" field is absent and inference is ambiguous. Pass
 *                 nullptr to rely purely on inference/JSON dtype.
 * @return         An Ort::Value backed by memory managed by the returned
 *                 vector pair. The caller owns the backing storage.
 *
 * @throws std::invalid_argument if the JSON is malformed or dtype is unknown.
 * @throws std::runtime_error    if data length is incompatible with shape.
 *
 * @note The backing memory (OrtMemoryInfo + data buffer) is embedded in the
 *       returned OwnedTensor struct to keep the Ort::Value alive.
 */

/**
 * @brief Container that keeps backing memory alive together with the tensor.
 *
 * ONNX Runtime tensors created via CreateTensor do NOT own the data buffer;
 * the caller must keep it alive for the duration of the tensor's lifetime.
 */
struct OwnedTensor {
  Ort::Value tensor{nullptr};
  // Numeric backing buffer (one of these is populated)
  std::vector<float>    f32_buf;
  std::vector<double>   f64_buf;
  std::vector<int8_t>   i8_buf;
  std::vector<int16_t>  i16_buf;
  std::vector<int32_t>  i32_buf;
  std::vector<int64_t>  i64_buf;
  std::vector<uint8_t>  u8_buf;
  std::vector<uint16_t> u16_buf;
  std::vector<uint32_t> u32_buf;
  std::vector<uint64_t> u64_buf;
  // String tensors need a separate path (Ort::Value::CreateTensor<std::string>)
  std::vector<std::string> str_buf;
};

/**
 * @brief Convert a JSON tensor object to an OwnedTensor.
 *
 * @param j     JSON object {"shape": [...], "data": [...], "dtype": "..."}.
 * @param hint  Optional dtype hint from model metadata (used when "dtype" key
 *              is absent).
 */
OwnedTensor json_to_tensor(const nlohmann::json &j,
                           ONNXTensorElementDataType hint =
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

/**
 * @brief Convert an Ort::Value tensor to a JSON object.
 *
 * Emits {"shape": [...], "data": [...], "dtype": "<name>"}.
 *
 * @param tensor  A tensor-type Ort::Value (must not be null).
 * @return        JSON object.
 * @throws std::runtime_error if the Ort::Value is not a tensor.
 */
nlohmann::json tensor_to_json(const Ort::Value &tensor);

/**
 * @brief Generate an example input JSON schema from model input metadata.
 *
 * Returns a JSON object mapping each input name to an example tensor JSON
 * with the correct shape (dynamic dims shown as -1) and dtype.
 *
 * Example output:
 * @code
 * {
 *   "input_ids": {
 *     "shape": [1, -1],
 *     "data": [0],
 *     "dtype": "int64"
 *   }
 * }
 * @endcode
 *
 * @param inputs  Vector of TensorInfo from OnnxModel::input_info().
 * @return        JSON object.
 */
nlohmann::json make_input_schema(const std::vector<TensorInfo> &inputs);

/**
 * @brief Parse a dtype string into the corresponding ONNX enum value.
 *
 * @throws std::invalid_argument for unknown dtype strings.
 */
ONNXTensorElementDataType dtype_from_string(const std::string &s);

} // namespace Onnx
