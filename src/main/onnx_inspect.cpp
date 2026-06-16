/**
 * @file onnx_inspect.cpp
 * @brief Standalone utility that loads an ONNX model and prints its metadata.
 *
 * This executable is NOT a MADS agent. It exists purely to verify that the
 * OnnxEngine library (OnnxModel + TensorJson) compiles and links correctly
 * against ONNX Runtime.
 *
 * Usage:
 *   mads-onnx-inspect --model-path <path/to/model.onnx>
 *   mads-onnx-inspect -m model.onnx
 *
 * Without --model-path it prints a usage hint and exits.
 *
 * @author Paolo Bosetti
 * @date 2026
 */

#include <onnx/OnnxModel.hpp>
#include <onnx/TensorJson.hpp>

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ─── pretty-print helpers ────────────────────────────────────────────────────

static std::string shape_str(const std::vector<int64_t> &shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      s += ", ";
    s += (shape[i] < 0 ? "?" : std::to_string(shape[i]));
  }
  s += "]";
  return s;
}

static void print_tensor_info(const std::string &label,
                              const std::vector<Onnx::TensorInfo> &tensors) {
  std::cout << label << ":\n";
  if (tensors.empty()) {
    std::cout << "  (none)\n";
    return;
  }
  for (const auto &t : tensors) {
    std::cout << "  " << t.name << "\n"
              << "    dtype : " << t.dtype_name() << "\n"
              << "    shape : " << shape_str(t.shape) << "\n";
  }
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  cxxopts::Options opts("mads-onnx-inspect",
                        "Inspect an ONNX model: print I/O metadata and "
                        "example input JSON schema");
  // clang-format off
  opts.add_options()
    ("m,model-path", "Path to the .onnx model file",
     cxxopts::value<fs::path>())
    ("j,json",       "Emit output as JSON rather than human-readable text")
    ("ep,execution-provider",
     "Execution provider: cpu (default), coreml, cuda, directml, webgpu, auto",
     cxxopts::value<std::string>()->default_value("cpu"))
    ("threads",
     "Intra-op thread count (0 = ORT default, uses all available cores)",
     cxxopts::value<int>()->default_value("0"))
    ("inter-threads",
     "Inter-op thread count (0 = ORT default)",
     cxxopts::value<int>()->default_value("0"))
    ("gpu-device",
     "GPU device index (used with --ep cuda or --ep directml)",
     cxxopts::value<int>()->default_value("0"))
    ("h,help",       "Print usage");
  // clang-format on

  cxxopts::ParseResult parsed;
  try {
    parsed = opts.parse(argc, argv);
  } catch (const cxxopts::exceptions::exception &e) {
    std::cerr << "Error: " << e.what() << "\n\n" << opts.help() << "\n";
    return EXIT_FAILURE;
  }

  if (parsed.count("help") || !parsed.count("model-path")) {
    std::cout << opts.help() << "\n";
    if (!parsed.count("model-path")) {
      std::cout << "No --model-path supplied. Provide a .onnx file to inspect.\n";
    }
    return EXIT_SUCCESS;
  }

  const fs::path model_path = parsed["model-path"].as<fs::path>();

  // ── Session config ─────────────────────────────────────────────────────────
  Onnx::SessionConfig cfg;
  cfg.execution_provider = parsed["execution-provider"].as<std::string>();
  cfg.intra_op_threads   = parsed["threads"].as<int>();
  cfg.inter_op_threads   = parsed["inter-threads"].as<int>();
  cfg.gpu_device_id      = parsed["gpu-device"].as<int>();

  // ── Load model ─────────────────────────────────────────────────────────────
  Onnx::OnnxModel model{model_path, cfg};

  const bool as_json = parsed.count("json") > 0;

  if (as_json) {
    json out;
    out["model"]            = model_path.string();
    out["active_provider"]  = model.active_provider();
    out["intra_op_threads"] = cfg.intra_op_threads;
    out["inter_op_threads"] = cfg.inter_op_threads;
    out["inputs"] = json::array();
    for (const auto &t : model.input_info()) {
      out["inputs"].push_back({{"name", t.name},
                                {"dtype", t.dtype_name()},
                                {"shape", t.shape}});
    }
    out["outputs"] = json::array();
    for (const auto &t : model.output_info()) {
      out["outputs"].push_back({{"name", t.name},
                                 {"dtype", t.dtype_name()},
                                 {"shape", t.shape}});
    }
    out["example_input"] = Onnx::make_input_schema(model.input_info());
    std::cout << out.dump(2) << "\n";
  } else {
    std::cout << "Model:            " << model_path.string() << "\n"
              << "Active provider:  " << model.active_provider() << "\n"
              << "Intra-op threads: "
              << (cfg.intra_op_threads == 0 ? "ORT default"
                                            : std::to_string(cfg.intra_op_threads))
              << "\n"
              << "Inter-op threads: "
              << (cfg.inter_op_threads == 0 ? "ORT default"
                                            : std::to_string(cfg.inter_op_threads))
              << "\n\n";
    print_tensor_info("Inputs", model.input_info());
    std::cout << "\n";
    print_tensor_info("Outputs", model.output_info());
    std::cout << "\nExample input JSON schema:\n"
              << Onnx::make_input_schema(model.input_info()).dump(2) << "\n";
  }

  return EXIT_SUCCESS;
}
