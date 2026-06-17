/**
 * @file onnx_bench.cpp
 * @brief Dev-only micro-benchmark comparing execution providers (CPU / CoreML /
 *        WebGPU / …) for a single ONNX model on this machine.
 *
 * Not installed and not part of the shipped package — it exists to produce
 * apples-to-apples latency numbers for a fixed input across providers.
 *
 * Usage:
 *   onnx-bench -m <model.onnx> [--iters N] [--warmup N] [--eps cpu,coreml,webgpu]
 */

#include "onnx/OnnxModel.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cxxopts.hpp>
#include <onnxruntime_cxx_api.h>
#include <rang.hpp>

using namespace std::chrono;
using Onnx::OnnxModel;
using Onnx::SessionConfig;

namespace {

/// Resolve a model shape (which may contain dynamic dims as -1) to a concrete
/// shape by replacing every dynamic/symbolic dimension with 1.
std::vector<int64_t> concrete_shape(const std::vector<int64_t> &shape) {
  std::vector<int64_t> out = shape;
  for (auto &d : out) {
    if (d < 0) d = 1;
  }
  if (out.empty()) out.push_back(1); // scalar → 1 element
  return out;
}

int64_t element_count(const std::vector<int64_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

struct Stats {
  double mean_ms = 0, median_ms = 0, min_ms = 0, p95_ms = 0;
  double throughput = 0; // inferences / second (from mean)
};

Stats summarize(std::vector<double> &samples_ms) {
  Stats s;
  if (samples_ms.empty()) return s;
  std::sort(samples_ms.begin(), samples_ms.end());
  s.min_ms = samples_ms.front();
  s.median_ms = samples_ms[samples_ms.size() / 2];
  s.p95_ms = samples_ms[std::min(samples_ms.size() - 1,
                                 (size_t)(samples_ms.size() * 0.95))];
  s.mean_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) /
              samples_ms.size();
  s.throughput = s.mean_ms > 0 ? 1000.0 / s.mean_ms : 0;
  return s;
}

} // namespace

int main(int argc, char *argv[]) {
  cxxopts::Options opts("onnx-bench",
                        "Benchmark ONNX execution providers on this machine");
  // clang-format off
  opts.add_options()
    ("m,model-path", "ONNX model file", cxxopts::value<std::string>())
    ("iters", "Timed iterations per provider", cxxopts::value<int>()->default_value("200"))
    ("warmup", "Warmup iterations per provider", cxxopts::value<int>()->default_value("25"))
    ("eps", "Comma-separated providers to test",
        cxxopts::value<std::string>()->default_value("cpu,coreml,webgpu"))
    ("h,help", "Print help");
  // clang-format on
  auto args = opts.parse(argc, argv);

  if (args.count("help") || !args.count("model-path")) {
    std::cout << opts.help() << std::endl;
    return args.count("model-path") ? 0 : 1;
  }

  const std::string model_path = args["model-path"].as<std::string>();
  const int iters = args["iters"].as<int>();
  const int warmup = args["warmup"].as<int>();

  // Parse provider list.
  std::vector<std::string> eps;
  {
    std::string list = args["eps"].as<std::string>();
    size_t pos = 0;
    while (pos < list.size()) {
      size_t comma = list.find(',', pos);
      if (comma == std::string::npos) comma = list.size();
      std::string tok = list.substr(pos, comma - pos);
      if (!tok.empty()) eps.push_back(tok);
      pos = comma + 1;
    }
  }

  std::cout << rang::style::bold << "ONNX provider benchmark" << rang::style::reset
            << "\n  model:  " << model_path << "\n  iters:  " << iters
            << " (warmup " << warmup << ")"
            << "\n  ORT providers available: ";
  for (const auto &p : Ort::GetAvailableProviders()) std::cout << p << " ";
  std::cout << "\n" << std::endl;

  // Build a fixed float32 input from the model's first input shape (these
  // classification/regression models take a single float tensor). We read the
  // shape once with a throwaway CPU session.
  std::vector<int64_t> in_shape;
  std::string in_dtype;
  {
    OnnxModel probe(model_path, SessionConfig{});
    if (probe.input_info().empty()) {
      std::cerr << "Model has no inputs." << std::endl;
      return 1;
    }
    in_shape = concrete_shape(probe.input_info().front().shape);
    in_dtype = probe.input_info().front().dtype_name();
  }
  if (in_dtype != "float32") {
    std::cerr << rang::fg::red
              << "onnx-bench currently supports float32-input models only; this "
                 "model's first input is "
              << in_dtype << "." << rang::fg::reset << std::endl;
    return 1;
  }

  const int64_t n = element_count(in_shape);
  std::vector<float> input_buf(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    input_buf[i] = static_cast<float>((i % 255) / 255.0); // deterministic ramp

  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::cout << "  input shape: [";
  for (size_t i = 0; i < in_shape.size(); ++i)
    std::cout << in_shape[i] << (i + 1 < in_shape.size() ? "," : "");
  std::cout << "] float32 (" << n << " elements)\n" << std::endl;

  std::printf("%-10s %-10s %10s %10s %10s %10s %12s %14s\n", "requested",
              "active", "mean_ms", "median_ms", "min_ms", "p95_ms", "inf/s",
              "out_checksum");
  std::printf(
      "------------------------------------------------------------------------"
      "------------------------\n");

  for (const auto &ep : eps) {
    SessionConfig cfg;
    cfg.execution_provider = ep;
    try {
      OnnxModel model(model_path, cfg);

      auto make_inputs = [&]() {
        std::vector<Ort::Value> v;
        v.push_back(Ort::Value::CreateTensor<float>(
            mem, input_buf.data(), input_buf.size(), in_shape.data(),
            in_shape.size()));
        return v;
      };

      double checksum = 0;
      for (int i = 0; i < warmup; ++i) {
        auto in = make_inputs();
        auto out = model.run(in);
        if (i == 0 && !out.empty()) {
          const float *p = out.front().GetTensorData<float>();
          auto cnt = out.front().GetTensorTypeAndShapeInfo().GetElementCount();
          for (size_t k = 0; k < cnt; ++k) checksum += p[k];
        }
      }

      std::vector<double> samples;
      samples.reserve(iters);
      for (int i = 0; i < iters; ++i) {
        auto in = make_inputs();
        auto t0 = steady_clock::now();
        auto out = model.run(in);
        auto t1 = steady_clock::now();
        samples.push_back(duration<double, std::milli>(t1 - t0).count());
      }
      Stats s = summarize(samples);
      std::printf("%-10s %-10s %10.3f %10.3f %10.3f %10.3f %12.1f %14.2f\n",
                  ep.c_str(), model.active_provider().c_str(), s.mean_ms,
                  s.median_ms, s.min_ms, s.p95_ms, s.throughput, checksum);
    } catch (const std::exception &e) {
      std::printf("%-10s %-10s  ERROR: %s\n", ep.c_str(), "-", e.what());
    }
  }

  std::cout << "\nNote: 'active' may differ from 'requested' when an EP is "
               "unavailable and falls back to CPU. Matching out_checksum across "
               "rows confirms identical computation.\n";
  return 0;
}
