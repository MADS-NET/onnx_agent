/**
 * @file CameraSource.cpp
 * @brief Implementation of CameraSource using OpenCV VideoCapture.
 */

#ifdef HAVE_OPENCV

#include "CameraSource.hpp"

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace Onnx {

// ─── Impl (pImpl) ─────────────────────────────────────────────────────────────

struct CameraSource::Impl {
  cv::VideoCapture cap;
};

// ─── CameraSource ─────────────────────────────────────────────────────────────

CameraSource::CameraSource(int camera_index, int capture_width,
                           int capture_height, double capture_fps,
                           const PreprocessConfig &cfg)
    : _camera_index(camera_index), _capture_width(capture_width),
      _capture_height(capture_height), _capture_fps(capture_fps), _cfg(cfg),
      _impl(std::make_unique<Impl>()) {}

CameraSource::~CameraSource() { close(); }

bool CameraSource::open() {
  if (_impl->cap.isOpened()) {
    return true; // already open
  }

  _impl->cap.open(_camera_index);
  if (!_impl->cap.isOpened()) {
    std::cerr << "[CameraSource] Failed to open camera index "
              << _camera_index << "\n";
    return false;
  }

  if (_capture_width > 0)
    _impl->cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(_capture_width));
  if (_capture_height > 0)
    _impl->cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(_capture_height));
  if (_capture_fps > 0)
    _impl->cap.set(cv::CAP_PROP_FPS, _capture_fps);

  return true;
}

void CameraSource::close() {
  if (_impl && _impl->cap.isOpened()) {
    _impl->cap.release();
  }
}

std::optional<nlohmann::json> CameraSource::next() {
  if (!_impl || !_impl->cap.isOpened()) {
    return std::nullopt;
  }

  cv::Mat frame;
  if (!_impl->cap.read(frame) || frame.empty()) {
    return std::nullopt;
  }

  // ── Resize to model input size ─────────────────────────────────────────────
  cv::Mat resized;
  cv::resize(frame, resized,
             cv::Size(_cfg.input_width, _cfg.input_height),
             0, 0, cv::INTER_LINEAR);

  // ── Convert BGR → RGB if required ──────────────────────────────────────────
  if (_cfg.rgb) {
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  }

  // ── Convert to float32 in [0, 1] range ────────────────────────────────────
  cv::Mat float_img;
  resized.convertTo(float_img, CV_32FC3, static_cast<double>(_cfg.scale));

  const int H = _cfg.input_height;
  const int W = _cfg.input_width;
  const int C = 3;
  const int64_t N = 1LL * C * H * W;

  std::vector<float> data;
  data.reserve(static_cast<size_t>(N));

  if (_cfg.nchw) {
    // NCHW: iterate channels first, then spatial
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    for (int c = 0; c < C; ++c) {
      float mean_c = _cfg.mean[static_cast<size_t>(c)];
      float std_c  = _cfg.std_dev[static_cast<size_t>(c)];
      const float *row_ptr = nullptr;
      for (int h = 0; h < H; ++h) {
        row_ptr = channels[c].ptr<float>(h);
        for (int w = 0; w < W; ++w) {
          data.push_back((row_ptr[w] - mean_c) / std_c);
        }
      }
    }
  } else {
    // NHWC: iterate spatial first, then channels
    for (int h = 0; h < H; ++h) {
      const float *row_ptr = float_img.ptr<float>(h);
      for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) {
          float mean_c = _cfg.mean[static_cast<size_t>(c)];
          float std_c  = _cfg.std_dev[static_cast<size_t>(c)];
          data.push_back((row_ptr[w * C + c] - mean_c) / std_c);
        }
      }
    }
  }

  // ── Pack into JSON tensor map ──────────────────────────────────────────────
  nlohmann::json tensor;
  if (_cfg.nchw) {
    tensor["shape"] = {1, C, H, W};
  } else {
    tensor["shape"] = {1, H, W, C};
  }
  tensor["dtype"] = "float32";
  tensor["data"]  = data;

  nlohmann::json result;
  result[_cfg.input_tensor_name] = tensor;
  return result;
}

} // namespace Onnx

#endif // HAVE_OPENCV
