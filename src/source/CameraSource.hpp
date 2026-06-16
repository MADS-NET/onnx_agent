/**
 * @file CameraSource.hpp
 * @brief OpenCV VideoCapture-backed implementation of Onnx::DataSource.
 *
 * Grabs frames from a webcam (or any OpenCV-compatible source), converts
 * each frame to a float32 NCHW tensor with configurable pre-processing
 * constants (resize, channel order, scale, mean, std), and returns it in
 * the project JSON tensor-map format ready for inference.
 *
 * Compiled only when HAVE_OPENCV is defined (set by CMake when
 * find_package(OpenCV) succeeds).
 *
 * @author Paolo Bosetti
 * @date   2026
 */

#pragma once

#ifdef HAVE_OPENCV

#include "DataSource.hpp"

#include <array>
#include <string>

namespace Onnx {

/**
 * @brief Image pre-processing configuration for a classification model.
 *
 * All fields correspond directly to INI / CLI configurable values so that
 * the same binary can serve different backbone models without recompilation.
 */
struct PreprocessConfig {
  /** @brief Model input width in pixels. */
  int input_width = 224;

  /** @brief Model input height in pixels. */
  int input_height = 224;

  /**
   * @brief If true, produce NCHW layout (batch × C × H × W).
   *        If false, produce NHWC layout (batch × H × W × C).
   *        MobileNetV2 from the ONNX Model Zoo expects NCHW.
   */
  bool nchw = true;

  /**
   * @brief If true, convert BGR (OpenCV default) to RGB before normalising.
   *        Most ImageNet-trained models expect RGB.
   */
  bool rgb = true;

  /** @brief Pixel-value scale factor applied before mean/std. (1/255 for [0,1]) */
  float scale = 1.0f / 255.0f;

  /** @brief Per-channel mean subtracted after scaling. ImageNet defaults. */
  std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};

  /** @brief Per-channel std divisor applied after mean subtraction. */
  std::array<float, 3> std_dev = {0.229f, 0.224f, 0.225f};

  /** @brief Model input tensor name as reported by OnnxModel::input_info(). */
  std::string input_tensor_name = "input";
};

/**
 * @brief OpenCV-backed webcam data source.
 *
 * Each call to next() grabs one frame, resizes, converts channel order,
 * and normalises according to the PreprocessConfig, then wraps the result
 * in the project JSON tensor-map format.
 *
 * Example usage:
 * @code
 *   Onnx::CameraSource src(0,         // camera index
 *                           640, 480,  // capture resolution
 *                           30,        // fps (hint)
 *                           cfg);      // preprocessing
 *   if (!src.open()) { ... handle error ... }
 *   auto maybe = src.next();
 *   if (maybe) { ... feed *maybe to OnnxModel ... }
 *   src.close();
 * @endcode
 */
class CameraSource : public DataSource {
public:
  /**
   * @brief Construct a CameraSource.
   *
   * @param camera_index  OpenCV device index (0 = default camera).
   * @param capture_width  Requested capture width (hint; camera may differ).
   * @param capture_height Requested capture height (hint).
   * @param capture_fps    Requested frame rate (hint).
   * @param cfg            Pre-processing constants.
   */
  CameraSource(int camera_index, int capture_width, int capture_height,
               double capture_fps, const PreprocessConfig &cfg);

  ~CameraSource() override;

  /** @brief Open the capture device. Returns false on failure. */
  bool open() override;

  /** @brief Release the capture device. */
  void close() override;

  /**
   * @brief Grab and pre-process one frame.
   *
   * Returns std::nullopt if the camera is not open, if grab fails, or if
   * the frame is empty.
   */
  std::optional<nlohmann::json> next() override;

private:
  int _camera_index;
  int _capture_width;
  int _capture_height;
  double _capture_fps;
  PreprocessConfig _cfg;

  // pImpl-style: forward-declared to keep OpenCV headers out of this header.
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace Onnx

#endif // HAVE_OPENCV
