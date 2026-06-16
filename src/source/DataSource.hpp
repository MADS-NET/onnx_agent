/**
 * @file DataSource.hpp
 * @brief Pure-virtual interface for ONNX model data sources.
 *
 * A DataSource produces input tensors for an ONNX model on demand.  Concrete
 * implementations may read from a webcam, a file, a network stream, a sensor,
 * or any other origin.
 *
 * The tensor representation follows the project-wide JSON schema:
 * @code
 * {
 *   "input": {
 *     "shape": [1, 3, 224, 224],
 *     "data":  [...],          // flat row-major array
 *     "dtype": "float32"
 *   }
 * }
 * @endcode
 *
 * Implement this interface and pass an instance to the source agent to replace
 * the default camera source with a custom data provider.
 *
 * @author Paolo Bosetti
 * @date   2026
 */

#pragma once

#include <nlohmann/json.hpp>
#include <optional>

namespace Onnx {

/**
 * @brief Abstract base class for ONNX model input sources.
 *
 * The lifecycle is:
 *  1. Construct an instance.
 *  2. Call open() — returns false if the source cannot be initialised.
 *  3. Call next() repeatedly to obtain successive input tensors.
 *  4. Call close() when done.
 *
 * Implementations must be safe to call from a single thread only; no
 * synchronisation is provided by this interface.
 */
class DataSource {
public:
  virtual ~DataSource() = default;

  // Non-copyable, movable
  DataSource(const DataSource &) = delete;
  DataSource &operator=(const DataSource &) = delete;
  DataSource(DataSource &&) = default;
  DataSource &operator=(DataSource &&) = default;

  /**
   * @brief Open and initialise the data source.
   *
   * Must be called once before next().  Implementations should acquire any
   * hardware resources (camera handle, file descriptor, …) here.
   *
   * @return true on success, false if the source could not be opened.
   */
  virtual bool open() = 0;

  /**
   * @brief Release any resources acquired by open().
   *
   * Safe to call even if open() returned false or was never called.
   */
  virtual void close() = 0;

  /**
   * @brief Produce the next model input tensor map.
   *
   * Returns a JSON object mapping each model input tensor name to its
   * @code {"shape":[…], "data":[…], "dtype":"…"} @endcode descriptor,
   * ready to be fed directly to OnnxModel::run() via json_to_tensor().
   *
   * Returns std::nullopt when:
   * - No frame is yet available (e.g. camera buffer empty — caller should
   *   retry on the next tick).
   * - The source has reached its end (e.g. file exhausted).
   * - A recoverable error occurred; the implementation logs details
   *   internally.
   *
   * @return A JSON tensor map, or std::nullopt on no-data / error.
   */
  virtual std::optional<nlohmann::json> next() = 0;

protected:
  DataSource() = default;
};

} // namespace Onnx
