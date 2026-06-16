/**
 * @file onnx_filter.cpp
 * @brief MADS filter agent that runs inference on an ONNX model.
 *
 * Subscribes to a topic, receives JSON messages that encode input tensors,
 * runs the loaded ONNX model, and publishes the output tensors as JSON to
 * another topic.
 *
 * CLI usage:
 * @code
 *   mads-onnx-filter [common MADS options] \
 *     [-m <model.onnx>] [--inspect] [--test]
 * @endcode
 *
 * @author Paolo Bosetti
 * @date   2026
 */

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

// ─── helpers ─────────────────────────────────────────────────────────────────

/** @brief Print tensor metadata in human-readable form. */
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

/**
 * @brief Build a deterministic dummy input for --test.
 *
 * Each tensor is filled with a ramp [0, 1, 2, …] cast to the correct dtype.
 * Dynamic dimensions are resolved as length 1.
 */
static json make_test_input(const std::vector<Onnx::TensorInfo> &inputs) {
  json result = json::object();
  for (const auto &info : inputs) {
    // Resolve dynamic dims to 1
    std::vector<int64_t> shape;
    shape.reserve(info.shape.size());
    for (auto d : info.shape)
      shape.push_back(d < 0 ? 1 : d);

    int64_t n = 1;
    for (auto d : shape)
      n *= d;
    if (n <= 0)
      n = 1;

    json tensor;
    tensor["shape"] = shape;
    tensor["dtype"] = info.dtype_name();

    // build data array: ramp of correct numeric type
    const auto dtype = info.dtype;
    json data = json::array();
    for (int64_t i = 0; i < n; ++i) {
      switch (dtype) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        data.push_back(static_cast<double>(i));
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        data.push_back(static_cast<int64_t>(i));
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        data.push_back(i % 2 == 0);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        data.push_back(std::to_string(i));
        break;
      default:
        data.push_back(0);
        break;
      }
    }
    tensor["data"] = data;
    result[info.name] = tensor;
  }
  return result;
}

/**
 * @brief Convert a JSON input map to Ort::Value tensors, run the model,
 *        and return the output as a JSON object.
 *
 * @param model   Loaded OnnxModel.
 * @param input_j JSON object mapping tensor names to tensor objects.
 * @return JSON object mapping output tensor names to tensor objects.
 * @throws std::invalid_argument / std::runtime_error on shape/dtype errors.
 */
static json run_inference(Onnx::OnnxModel &model, const json &input_j) {
  const auto &infos = model.input_info();

  // Build owned tensors in input order
  std::vector<Onnx::OwnedTensor> owned;
  owned.reserve(infos.size());
  std::vector<Ort::Value> input_vals;
  input_vals.reserve(infos.size());

  for (const auto &info : infos) {
    if (!input_j.contains(info.name)) {
      throw std::invalid_argument("Missing input tensor: " + info.name);
    }
    owned.push_back(
        Onnx::json_to_tensor(input_j.at(info.name), info.dtype));
    input_vals.push_back(std::move(owned.back().tensor));
  }

  auto output_vals = model.run(input_vals);

  json result = json::object();
  const auto &out_infos = model.output_info();
  for (size_t i = 0; i < output_vals.size(); ++i) {
    const std::string &name =
        (i < out_infos.size()) ? out_infos[i].name : ("output_" + std::to_string(i));
    result[name] = Onnx::tensor_to_json(output_vals[i]);
  }
  return result;
}

// ─── main ────────────────────────────────────────────────────────────────────

/**
 * @brief Entry point for the mads-onnx-filter agent.
 *
 * Parses CLI options, loads the ONNX model, and either performs an offline
 * action (--inspect, --test) or enters the MADS pub/sub loop.
 */
int main(int argc, char *argv[]) {
  // ── Default topics per PURPOSE.md ─────────────────────────────────────────
  const std::string default_sub_topic = "onnx-filter-input";
  const std::string default_pub_topic = "onnx-filter";

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
     "Standalone self-test: load model, run inference on a sample input, "
     "print output JSON, then exit (no MADS network required)")
    ("input",
     "JSON file to use as model input for --test instead of dummy data. "
     "The file must map tensor names to {shape, data, dtype} objects "
     "matching the project tensor<->JSON schema.",
     cxxopts::value<fs::path>());
  // clang-format on

  auto &parsed = agent.parse_options(argc, argv);

  // Handle --help, --version, --save-settings
  if (int rc = Mads::AgentApp::handle_standard_exit_options<Mads::AgentApp>(
          parsed, agent.raw_options(), argv);
      rc >= 0) {
    return rc;
  }

  // ── Determine model path: CLI → INI fallback ───────────────────────────
  // For --inspect / --test we must resolve the model path *before* init(),
  // because init() connects to the broker (not needed for offline modes).
  // We do a lightweight settings read only when needed.
  fs::path model_path;
  const bool want_inspect = parsed.count("inspect") > 0;
  const bool want_test    = parsed.count("test") > 0;

  if (parsed.count("model-path") > 0) {
    model_path = parsed["model-path"].as<fs::path>();
  }

  // ── Offline modes: no broker connection needed ──────────────────────────
  if (want_inspect || want_test) {
    // If model_path not given on CLI, try reading the INI file directly
    if (model_path.empty()) {
      // Try settings file path: common option "--settings" or the default path
      std::string settings_uri = SETTINGS_URI;
      if (parsed.count("settings") > 0) {
        settings_uri = parsed["settings"].as<std::string>();
      }
      // Only try local-file settings (URI that is a path)
      if (!settings_uri.starts_with("tcp://")) {
        try {
          // Use a temporary agent to read settings without connecting
          Mads::AgentApp tmp{argv[0], settings_uri};
          tmp.init(parsed, settings_uri, false);
          auto s = tmp.settings_json();
          if (s.contains("model_path")) {
            model_path = s["model_path"].get<std::string>();
          }
        } catch (...) {
          // silently ignore settings read errors in offline mode
        }
      }
    }

    if (model_path.empty()) {
      std::cerr << fg::red
                << "Error: no model path. Supply --model-path or set "
                   "model_path in the INI file.\n"
                << fg::reset;
      return EXIT_FAILURE;
    }

    // Load model
    std::unique_ptr<Onnx::OnnxModel> model;
    try {
      model = std::make_unique<Onnx::OnnxModel>(model_path);
    } catch (const std::exception &e) {
      std::cerr << fg::red << "Error loading model: " << e.what() << "\n"
                << fg::reset;
      return EXIT_FAILURE;
    }

    if (want_inspect) {
      std::cout << style::bold << "Model: " << style::reset
                << model_path.string() << "\n\n";
      print_tensor_list("Inputs", model->input_info());
      std::cout << "\n";
      print_tensor_list("Outputs", model->output_info());
      std::cout << "\n" << style::bold << "Example input JSON schema:\n"
                << style::reset
                << Onnx::make_input_schema(model->input_info()).dump(2) << "\n";
      return EXIT_SUCCESS;
    }

    if (want_test) {
      std::cout << style::bold << "=== mads-onnx-filter --test ===\n"
                << style::reset << "Model: " << model_path.string() << "\n\n";

      // ── Determine input JSON: --input file or generated dummy ──────────────
      json test_input;
      if (parsed.count("input") > 0) {
        fs::path input_path = parsed["input"].as<fs::path>();
        if (!fs::exists(input_path)) {
          std::cerr << fg::red << "Error: --input file not found: "
                    << input_path.string() << "\n" << fg::reset;
          return EXIT_FAILURE;
        }
        try {
          std::ifstream ifs(input_path);
          if (!ifs)
            throw std::runtime_error("cannot open file");
          test_input = json::parse(ifs);
        } catch (const std::exception &e) {
          std::cerr << fg::red << "Error reading --input JSON: " << e.what()
                    << "\n" << fg::reset;
          return EXIT_FAILURE;
        }
        std::cout << "Input file: " << input_path.string() << "\n\n";
      } else {
        test_input = make_test_input(model->input_info());
        std::cout << "Dummy input:\n" << test_input.dump(2) << "\n\n";
      }

      json output;
      try {
        output = run_inference(*model, test_input);
      } catch (const std::exception &e) {
        std::cerr << fg::red << "Inference error: " << e.what() << "\n"
                  << fg::reset;
        return EXIT_FAILURE;
      }

      std::cout << style::bold << "Output:\n" << style::reset
                << output.dump(2) << "\n";
      return EXIT_SUCCESS;
    }
  }

  // ── Online (MADS broker) mode ─────────────────────────────────────────────
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
                   "model_path in the INI [onnx-filter] section.\n"
                << fg::reset;
      return EXIT_FAILURE;
    }
  }

  // Load model
  std::unique_ptr<Onnx::OnnxModel> model;
  try {
    model = std::make_unique<Onnx::OnnxModel>(model_path);
  } catch (const std::exception &e) {
    std::cerr << fg::red << "Error loading model: " << e.what() << "\n"
              << fg::reset;
    return EXIT_FAILURE;
  }

  // Topics: INI → defaults
  std::string pub_topic = settings.value("pub_topic", default_pub_topic);
  std::vector<std::string> sub_topics = {default_sub_topic};
  if (settings.contains("sub_topic") && settings["sub_topic"].is_array()) {
    sub_topics = settings["sub_topic"].get<std::vector<std::string>>();
  }

  agent.set_pub_topic(pub_topic);
  agent.set_sub_topic(sub_topics);

  agent.enable_events();
  agent.connect();
  agent.info();

  std::cout << "  model path:     " << style::bold << model_path.string()
            << style::reset << "\n"
            << "  sub topic(s):   " << style::bold;
  for (const auto &t : sub_topics)
    std::cout << t << " ";
  std::cout << style::reset << "\n"
            << "  pub topic:      " << style::bold << pub_topic << style::reset
            << "\n\n";

  // ── Input-triggered loop ─────────────────────────────────────────────────
  agent.loop([&]() -> std::chrono::milliseconds {
    if (agent.receive(false) == Mads::message_type::json) {
      const std::string topic = agent.last_topic();
      if (topic == "control" || topic.empty()) {
        return 0ms;
      }

      auto [recv_topic, input_j] = agent.last_json();

      if (!input_j.is_object() || input_j.empty()) {
        std::cerr << fg::yellow
                  << "Warning: received empty or non-object JSON on topic '"
                  << topic << "', skipping\n"
                  << fg::reset;
        return 0ms;
      }

      try {
        json output = run_inference(*model, input_j);
        agent.publish(output, pub_topic);
      } catch (const std::exception &e) {
        std::cerr << fg::red << "Inference error (topic '" << topic
                  << "'): " << e.what() << "\n"
                  << fg::reset;
        // keep running — do not crash on bad input
      }
    }
    return 0ms;
  });

  agent.disconnect();
  agent.restart_if_requested(argv);
  return 0;
}
