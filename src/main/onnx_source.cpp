/**
 * @file onnx_source.cpp
 * @brief MADS source agent that grabs webcam frames, runs an ONNX classifier,
 *        and publishes top-k ImageNet predictions.
 *
 * Internally timed (each tick = one frame from the camera). Period defaults to
 * 100 ms and can be overridden by --period on the CLI or `period` in the INI
 * file.
 *
 * CLI usage:
 * @code
 *   mads-onnx-source [common MADS options] \
 *     [-m <model.onnx>] [--inspect] \
 *     [--test [--image <path.jpg>]] \
 *     [--period <ms>] [--camera <idx>] [--width <px>] [--height <px>]
 * @endcode
 *
 * @author Paolo Bosetti
 * @date   2026
 */

#include <source/Classify.hpp>

#ifdef HAVE_OPENCV
#include <source/CameraSource.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <onnx/OnnxModel.hpp>
#include <onnx/TensorJson.hpp>

#include <agent_app.hpp>
#include <mads.hpp>
#include <rang.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std::chrono_literals;
using namespace rang;
using json = nlohmann::json;
namespace fs = std::filesystem;

// ─── helper: build session config ────────────────────────────────────────────

static Onnx::SessionConfig
build_session_config(const cxxopts::ParseResult &parsed, const json &settings) {
  Onnx::SessionConfig cfg;
  if (settings.contains("execution_provider"))
    cfg.execution_provider = settings["execution_provider"].get<std::string>();
  if (settings.contains("intra_op_threads"))
    cfg.intra_op_threads = settings["intra_op_threads"].get<int>();
  if (settings.contains("inter_op_threads"))
    cfg.inter_op_threads = settings["inter_op_threads"].get<int>();
  if (settings.contains("gpu_device_id"))
    cfg.gpu_device_id = settings["gpu_device_id"].get<int>();
  if (parsed.count("execution-provider"))
    cfg.execution_provider = parsed["execution-provider"].as<std::string>();
  if (parsed.count("threads"))
    cfg.intra_op_threads = parsed["threads"].as<int>();
  if (parsed.count("inter-threads"))
    cfg.inter_op_threads = parsed["inter-threads"].as<int>();
  if (parsed.count("gpu-device"))
    cfg.gpu_device_id = parsed["gpu-device"].as<int>();
  return cfg;
}

// ─── helper: print tensor metadata ───────────────────────────────────────────

static void print_tensor_list(const std::string &label,
                              const std::vector<Onnx::TensorInfo> &tensors) {
  std::cout << label << ":\n";
  if (tensors.empty()) {
    std::cout << "  (none)\n";
    return;
  }
  for (const auto &t : tensors) {
    std::string shape_str = "[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
      if (i > 0)
        shape_str += ", ";
      shape_str += (t.shape[i] < 0 ? "?" : std::to_string(t.shape[i]));
    }
    shape_str += "]";
    std::cout << "  " << t.name << "\n"
              << "    dtype : " << t.dtype_name() << "\n"
              << "    shape : " << shape_str << "\n";
  }
}

// ─── helper: run inference and return raw output JSON ───────────────────────

static json run_inference(Onnx::OnnxModel &model, const json &input_j) {
  const auto &infos = model.input_info();

  std::vector<Onnx::OwnedTensor> owned;
  owned.reserve(infos.size());
  std::vector<Ort::Value> input_vals;
  input_vals.reserve(infos.size());

  for (const auto &info : infos) {
    if (!input_j.contains(info.name)) {
      throw std::invalid_argument("Missing input tensor: " + info.name);
    }
    owned.push_back(Onnx::json_to_tensor(input_j.at(info.name), info.dtype));
    input_vals.push_back(std::move(owned.back().tensor));
  }

  auto output_vals = model.run(input_vals);

  json result = json::object();
  const auto &out_infos = model.output_info();
  for (size_t i = 0; i < output_vals.size(); ++i) {
    const std::string &name = (i < out_infos.size())
                                  ? out_infos[i].name
                                  : ("output_" + std::to_string(i));
    result[name] = Onnx::tensor_to_json(output_vals[i]);
  }
  return result;
}

// ─── helper: read PreprocessConfig from settings JSON ────────────────────────

#ifdef HAVE_OPENCV
static Onnx::PreprocessConfig preprocess_from_settings(const json &s) {
  Onnx::PreprocessConfig cfg;
  if (s.contains("input_width"))
    cfg.input_width = s["input_width"].get<int>();
  if (s.contains("input_height"))
    cfg.input_height = s["input_height"].get<int>();
  if (s.contains("nchw"))
    cfg.nchw = s["nchw"].get<bool>();
  if (s.contains("rgb"))
    cfg.rgb = s["rgb"].get<bool>();
  if (s.contains("scale"))
    cfg.scale = s["scale"].get<float>();
  if (s.contains("mean") && s["mean"].is_array() && s["mean"].size() == 3) {
    for (int i = 0; i < 3; ++i)
      cfg.mean[static_cast<size_t>(i)] = s["mean"][i].get<float>();
  }
  if (s.contains("std") && s["std"].is_array() && s["std"].size() == 3) {
    for (int i = 0; i < 3; ++i)
      cfg.std_dev[static_cast<size_t>(i)] = s["std"][i].get<float>();
  }
  if (s.contains("input_tensor_name"))
    cfg.input_tensor_name = s["input_tensor_name"].get<std::string>();
  return cfg;
}

/**
 * @brief Preprocess a BGR image (cv::Mat) into the model input JSON tensor.
 *
 * Shared between --test (imread) and the live camera path.
 */
static json preprocess_image(const cv::Mat &bgr_frame,
                             const Onnx::PreprocessConfig &cfg) {
  cv::Mat resized;
  cv::resize(bgr_frame, resized,
             cv::Size(cfg.input_width, cfg.input_height), 0, 0,
             cv::INTER_LINEAR);

  if (cfg.rgb) {
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  }

  cv::Mat float_img;
  resized.convertTo(float_img, CV_32FC3, static_cast<double>(cfg.scale));

  const int H = cfg.input_height;
  const int W = cfg.input_width;
  const int C = 3;
  const int64_t N = 1LL * C * H * W;

  std::vector<float> data;
  data.reserve(static_cast<size_t>(N));

  if (cfg.nchw) {
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    for (int c = 0; c < C; ++c) {
      float mean_c = cfg.mean[static_cast<size_t>(c)];
      float std_c  = cfg.std_dev[static_cast<size_t>(c)];
      for (int h = 0; h < H; ++h) {
        const float *row_ptr = channels[c].ptr<float>(h);
        for (int w = 0; w < W; ++w) {
          data.push_back((row_ptr[w] - mean_c) / std_c);
        }
      }
    }
  } else {
    for (int h = 0; h < H; ++h) {
      const float *row_ptr = float_img.ptr<float>(h);
      for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) {
          float mean_c = cfg.mean[static_cast<size_t>(c)];
          float std_c  = cfg.std_dev[static_cast<size_t>(c)];
          data.push_back((row_ptr[w * C + c] - mean_c) / std_c);
        }
      }
    }
  }

  json tensor;
  if (cfg.nchw) {
    tensor["shape"] = {1, C, H, W};
  } else {
    tensor["shape"] = {1, H, W, C};
  }
  tensor["dtype"] = "float32";
  tensor["data"]  = data;

  json result;
  result[cfg.input_tensor_name] = tensor;
  return result;
}
#endif // HAVE_OPENCV

// ─── main ────────────────────────────────────────────────────────────────────

/**
 * @brief Entry point for the mads-onnx-source agent.
 *
 * Parses CLI options, loads the ONNX model, and either performs an offline
 * action (--inspect, --test) or enters the internally-timed MADS pub loop.
 */
int main(int argc, char *argv[]) {
  // ── Default topics per PURPOSE.md ─────────────────────────────────────────
  const std::string default_pub_topic = "onnx-source";

  // ── AgentApp setup ────────────────────────────────────────────────────────
  Mads::AgentApp agent{argv[0], SETTINGS_URI};
  agent.add_common_options();
  agent.add_agent_identity_options();

  // clang-format off
  agent.options()
    ("m,model-path",
     "Path to the ONNX model file (overrides model_path INI setting)",
     cxxopts::value<fs::path>())
    ("inspect",
     "Print model I/O metadata and example input JSON schema, then exit")
    ("test",
     "Standalone self-test: load model, run inference on a sample image, "
     "print top-k classification JSON, then exit (no MADS network required)")
#ifdef HAVE_OPENCV
    ("image",
     "Path to an image file for --test (default: dummy grey frame)",
     cxxopts::value<fs::path>())
    ("camera",
     "Camera device index (default: 0)",
     cxxopts::value<int>()->default_value("0"))
    ("width",
     "Camera capture width in pixels (default: 640)",
     cxxopts::value<int>()->default_value("640"))
    ("height",
     "Camera capture height in pixels (default: 480)",
     cxxopts::value<int>()->default_value("480"))
    ("fps",
     "Camera capture frame rate hint (default: 30)",
     cxxopts::value<double>()->default_value("30.0"))
#endif
    ("k,top-k",
     "Number of top predictions to include in output (default: 5)",
     cxxopts::value<int>()->default_value("5"))
    ("period",
     "Loop period in milliseconds (default: 100)",
     cxxopts::value<int>()->default_value("100"))
    ("labels",
     "Path to ImageNet labels file (overrides labels_path INI setting)",
     cxxopts::value<fs::path>())
    ("ep,execution-provider",
     "Execution provider: cpu (default), coreml, cuda, directml, webgpu, auto "
     "(overrides execution_provider INI setting)",
     cxxopts::value<std::string>())
    ("threads",
     "Intra-op thread count (0 = ORT default; overrides intra_op_threads INI)",
     cxxopts::value<int>())
    ("inter-threads",
     "Inter-op thread count (0 = ORT default; overrides inter_op_threads INI)",
     cxxopts::value<int>())
    ("gpu-device",
     "GPU device index for CUDA/DirectML (overrides gpu_device_id INI setting)",
     cxxopts::value<int>());
  // clang-format on

  auto &parsed = agent.parse_options(argc, argv);

  // Handle --help, --version, --save-settings
  if (int rc = Mads::AgentApp::handle_standard_exit_options<Mads::AgentApp>(
          parsed, agent.raw_options(), argv);
      rc >= 0) {
    return rc;
  }

  // ── Resolve model path: CLI → INI fallback ────────────────────────────────
  fs::path model_path;
  const bool want_inspect = parsed.count("inspect") > 0;
  const bool want_test    = parsed.count("test") > 0;

  if (parsed.count("model-path") > 0) {
    model_path = parsed["model-path"].as<fs::path>();
  }

  // ── Offline modes (--inspect / --test): no broker connection needed ────────
  if (want_inspect || want_test) {
    if (model_path.empty()) {
      std::string settings_uri = SETTINGS_URI;
      if (parsed.count("settings") > 0) {
        settings_uri = parsed["settings"].as<std::string>();
      }
      if (!settings_uri.starts_with("tcp://")) {
        try {
          Mads::AgentApp tmp{argv[0], settings_uri};
          tmp.init(parsed, settings_uri, false);
          auto s = tmp.settings_json();
          if (s.contains("model_path")) {
            model_path = s["model_path"].get<std::string>();
          }
        } catch (...) {
        }
      }
    }

    if (model_path.empty()) {
      std::cerr << fg::red
                << "Error: no model path. Supply --model-path or set "
                   "model_path in the [onnx-source] INI section.\n"
                << fg::reset;
      return EXIT_FAILURE;
    }

    // Load model (offline: no settings JSON available)
    const Onnx::SessionConfig offline_cfg =
        build_session_config(parsed, json::object());
    std::unique_ptr<Onnx::OnnxModel> model;
    try {
      model = std::make_unique<Onnx::OnnxModel>(model_path, offline_cfg);
    } catch (const std::exception &e) {
      std::cerr << fg::red << "Error loading model: " << e.what() << "\n"
                << fg::reset;
      return EXIT_FAILURE;
    }

    // ── --inspect ─────────────────────────────────────────────────────────
    if (want_inspect) {
      std::cout << style::bold << "Model: " << style::reset
                << model_path.string() << "\n"
                << "Active provider:  " << model->active_provider() << "\n"
                << "Intra-op threads: "
                << (offline_cfg.intra_op_threads == 0
                        ? "ORT default"
                        : std::to_string(offline_cfg.intra_op_threads))
                << "\n\n";
      print_tensor_list("Inputs", model->input_info());
      std::cout << "\n";
      print_tensor_list("Outputs", model->output_info());
      std::cout << "\n"
                << style::bold << "Example input JSON schema:\n" << style::reset
                << Onnx::make_input_schema(model->input_info()).dump(2) << "\n";
      return EXIT_SUCCESS;
    }

    // ── --test ────────────────────────────────────────────────────────────
    if (want_test) {
      std::cout << style::bold << "=== mads-onnx-source --test ===\n"
                << style::reset
                << "Model:           " << model_path.string() << "\n"
                << "Active provider: " << model->active_provider() << "\n\n";

#ifndef HAVE_OPENCV
      std::cerr << fg::red
                << "Error: --test with image preprocessing requires OpenCV. "
                   "Rebuild with OpenCV installed.\n"
                << fg::reset;
      return EXIT_FAILURE;
#else
      // Determine preprocessing config (use defaults for --test)
      Onnx::PreprocessConfig cfg; // standard ImageNet defaults

      // Resolve labels path
      fs::path labels_path;
      if (parsed.count("labels") > 0) {
        labels_path = parsed["labels"].as<fs::path>();
      }

      // Determine top-k
      int top_k_count = parsed["top-k"].as<int>();

      // Load labels (may be empty if not provided)
      std::vector<std::string> labels;
      if (!labels_path.empty()) {
        labels = Onnx::load_labels(labels_path.string());
        if (labels.empty()) {
          std::cerr << fg::yellow << "Warning: labels file not found or empty: "
                    << labels_path.string() << "\n" << fg::reset;
        } else {
          std::cout << "Labels: " << labels_path.string() << " ("
                    << labels.size() << " classes)\n\n";
        }
      }

      // Determine input tensor name from model metadata
      if (!model->input_info().empty()) {
        cfg.input_tensor_name = model->input_info()[0].name;
      }

      // Determine output tensor name
      std::string output_name = "output";
      if (!model->output_info().empty()) {
        output_name = model->output_info()[0].name;
      }

      cv::Mat bgr_frame;
      if (parsed.count("image") > 0) {
        fs::path img_path = parsed["image"].as<fs::path>();
        bgr_frame = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (bgr_frame.empty()) {
          std::cerr << fg::red << "Error: cannot read image: "
                    << img_path.string() << "\n" << fg::reset;
          return EXIT_FAILURE;
        }
        std::cout << "Image: " << img_path.string()
                  << " (" << bgr_frame.cols << "x" << bgr_frame.rows << ")\n\n";
      } else {
        // Use a dummy grey frame when no image is supplied
        bgr_frame = cv::Mat(cfg.input_height, cfg.input_width, CV_8UC3,
                            cv::Scalar(128, 128, 128));
        std::cout << "No --image supplied; using dummy grey frame ("
                  << cfg.input_width << "x" << cfg.input_height << ")\n\n";
      }

      json input_j;
      try {
        input_j = preprocess_image(bgr_frame, cfg);
      } catch (const std::exception &e) {
        std::cerr << fg::red << "Preprocessing error: " << e.what() << "\n"
                  << fg::reset;
        return EXIT_FAILURE;
      }

      json raw_output;
      try {
        raw_output = run_inference(*model, input_j);
      } catch (const std::exception &e) {
        std::cerr << fg::red << "Inference error: " << e.what() << "\n"
                  << fg::reset;
        return EXIT_FAILURE;
      }

      json result = Onnx::classify(raw_output, output_name, labels, top_k_count);
      std::cout << style::bold << "Top-" << top_k_count << " predictions:\n"
                << style::reset << result.dump(2) << "\n";
      return EXIT_SUCCESS;
#endif // HAVE_OPENCV
    }
  }

  // ── Online (MADS broker) mode ─────────────────────────────────────────────
#ifndef HAVE_OPENCV
  std::cerr << fg::red
            << "Error: mads-onnx-source requires OpenCV for live camera "
               "operation. Rebuild with OpenCV installed.\n"
            << fg::reset;
  return EXIT_FAILURE;
#else
  try {
    agent.init(parsed);
  } catch (const std::exception &e) {
    std::cerr << fg::red << "Error initializing agent: " << e.what() << "\n"
              << fg::reset;
    return EXIT_FAILURE;
  }

  const json &settings = agent.settings_json();

  // Resolve model path from settings if not given on CLI
  if (model_path.empty()) {
    if (settings.contains("model_path")) {
      model_path = settings["model_path"].get<std::string>();
    } else {
      std::cerr << fg::red
                << "Error: no model path. Supply --model-path or set "
                   "model_path in the [onnx-source] INI section.\n"
                << fg::reset;
      return EXIT_FAILURE;
    }
  }

  // Load model
  const Onnx::SessionConfig online_cfg = build_session_config(parsed, settings);
  std::unique_ptr<Onnx::OnnxModel> model;
  try {
    model = std::make_unique<Onnx::OnnxModel>(model_path, online_cfg);
  } catch (const std::exception &e) {
    std::cerr << fg::red << "Error loading model: " << e.what() << "\n"
              << fg::reset;
    return EXIT_FAILURE;
  }

  // Resolve labels path
  fs::path labels_path;
  if (parsed.count("labels") > 0) {
    labels_path = parsed["labels"].as<fs::path>();
  } else if (settings.contains("labels_path")) {
    labels_path = settings["labels_path"].get<std::string>();
  }

  std::vector<std::string> labels;
  if (!labels_path.empty()) {
    labels = Onnx::load_labels(labels_path.string());
    if (labels.empty()) {
      std::cerr << fg::yellow << "Warning: labels file empty or not found: "
                << labels_path.string() << "\n" << fg::reset;
    }
  }

  // Top-k count
  int top_k_count = parsed["top-k"].as<int>();
  if (settings.contains("top_k"))
    top_k_count = settings["top_k"].get<int>();

  // Period (ms)
  int period_ms = parsed["period"].as<int>();
  if (settings.contains("period"))
    period_ms = settings["period"].get<int>();

  // Camera parameters
  int cam_idx    = parsed["camera"].as<int>();
  int cam_width  = parsed["width"].as<int>();
  int cam_height = parsed["height"].as<int>();
  double cam_fps = parsed["fps"].as<double>();
  if (settings.contains("camera"))
    cam_idx = settings["camera"].get<int>();
  if (settings.contains("capture_width"))
    cam_width = settings["capture_width"].get<int>();
  if (settings.contains("capture_height"))
    cam_height = settings["capture_height"].get<int>();
  if (settings.contains("capture_fps"))
    cam_fps = settings["capture_fps"].get<double>();

  // Preprocessing config
  Onnx::PreprocessConfig cfg = preprocess_from_settings(settings);
  if (!model->input_info().empty()) {
    cfg.input_tensor_name = model->input_info()[0].name;
  }

  // Determine output tensor name
  std::string output_name = "output";
  if (!model->output_info().empty()) {
    output_name = model->output_info()[0].name;
  }

  // Build camera source
  auto cam_source = std::make_unique<Onnx::CameraSource>(
      cam_idx, cam_width, cam_height, cam_fps, cfg);

  if (!cam_source->open()) {
    std::cerr << fg::red
              << "Error: failed to open camera index " << cam_idx << "\n"
              << fg::reset;
    return EXIT_FAILURE;
  }

  // Topics
  std::string pub_topic = settings.value("pub_topic", default_pub_topic);
  agent.set_pub_topic(pub_topic);

  agent.enable_events();
  agent.connect();
  agent.info();

  std::cout << "  model path:     " << style::bold << model_path.string()
            << style::reset << "\n"
            << "  EP (active):    " << style::bold << model->active_provider()
            << style::reset << "\n"
            << "  intra threads:  " << style::bold
            << (online_cfg.intra_op_threads == 0
                    ? "ORT default"
                    : std::to_string(online_cfg.intra_op_threads))
            << style::reset << "\n"
            << "  pub topic:      " << style::bold << pub_topic << style::reset
            << "\n"
            << "  camera index:   " << style::bold << cam_idx << style::reset
            << "\n"
            << "  period (ms):    " << style::bold << period_ms << style::reset
            << "\n"
            << "  top-k:          " << style::bold << top_k_count << style::reset
            << "\n"
            << "  labels:         " << style::bold
            << (labels_path.empty() ? "(none)" : labels_path.string())
            << style::reset << "\n\n";

  // ── Internally-timed loop ─────────────────────────────────────────────────
  agent.loop([&]() -> std::chrono::milliseconds {
    auto maybe_input = cam_source->next();
    if (!maybe_input) {
      // No frame available; retry on next tick
      return std::chrono::milliseconds(period_ms);
    }

    json raw_output;
    try {
      raw_output = run_inference(*model, *maybe_input);
    } catch (const std::exception &e) {
      std::cerr << fg::yellow << "Inference error: " << e.what() << "\n"
                << fg::reset;
      return std::chrono::milliseconds(period_ms);
    }

    try {
      json result = Onnx::classify(raw_output, output_name, labels, top_k_count);
      agent.publish(result, pub_topic);
    } catch (const std::exception &e) {
      std::cerr << fg::yellow << "Post-processing error: " << e.what() << "\n"
                << fg::reset;
    }

    return std::chrono::milliseconds(period_ms);
  });

  cam_source->close();
  agent.disconnect();
  agent.restart_if_requested(argv);
  return 0;
#endif // HAVE_OPENCV
}
