/**
 * @file Classify.cpp
 * @brief Implementation of image-classification post-processing helpers.
 */

#include "Classify.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace Onnx {

// ─── load_labels ─────────────────────────────────────────────────────────────

std::vector<std::string> load_labels(const std::string &path) {
  std::vector<std::string> labels;
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return labels; // empty on error; caller decides how to handle
  }
  std::string line;
  while (std::getline(ifs, line)) {
    // Trim trailing whitespace / CR
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                             line.back() == ' ')) {
      line.pop_back();
    }
    labels.push_back(line);
  }
  return labels;
}

// ─── softmax_inplace ─────────────────────────────────────────────────────────

void softmax_inplace(std::vector<float> &v) {
  if (v.empty())
    return;
  const float max_val = *std::max_element(v.begin(), v.end());
  float sum = 0.0f;
  for (auto &x : v) {
    x = std::exp(x - max_val);
    sum += x;
  }
  if (sum > 0.0f) {
    for (auto &x : v)
      x /= sum;
  }
}

// ─── top_k_predictions ───────────────────────────────────────────────────────

nlohmann::json top_k_predictions(const std::vector<float> &scores,
                                 const std::vector<std::string> &labels,
                                 int k, bool do_softmax) {
  if (scores.empty()) {
    nlohmann::json err;
    err["error"] = "empty scores vector";
    return err;
  }

  // Copy and optionally apply softmax
  std::vector<float> probs = scores;
  if (do_softmax) {
    softmax_inplace(probs);
  }

  // Build indices sorted by descending probability
  std::vector<int> indices(probs.size());
  std::iota(indices.begin(), indices.end(), 0);
  const int kk = std::min(k, static_cast<int>(probs.size()));
  std::partial_sort(
      indices.begin(), indices.begin() + kk, indices.end(),
      [&probs](int a, int b) { return probs[a] > probs[b]; });

  auto label_for = [&](int id) -> std::string {
    if (id >= 0 && static_cast<size_t>(id) < labels.size()) {
      return labels[static_cast<size_t>(id)];
    }
    return "class_" + std::to_string(id);
  };

  nlohmann::json top_arr = nlohmann::json::array();
  for (int i = 0; i < kk; ++i) {
    const int id = indices[i];
    nlohmann::json entry;
    entry["class_id"]   = id;
    entry["label"]      = label_for(id);
    entry["confidence"] = probs[static_cast<size_t>(id)];
    top_arr.push_back(entry);
  }

  nlohmann::json result;
  result["top_k"] = top_arr;
  result["best"]  = top_arr.empty() ? nlohmann::json(nullptr) : top_arr[0];
  return result;
}

// ─── classify ────────────────────────────────────────────────────────────────

nlohmann::json classify(const nlohmann::json &output_j,
                        const std::string &output_name,
                        const std::vector<std::string> &labels, int k) {
  if (!output_j.contains(output_name)) {
    nlohmann::json err;
    err["error"] = "output tensor '" + output_name + "' not found";
    return err;
  }

  const auto &tensor_j = output_j.at(output_name);
  if (!tensor_j.contains("data") || !tensor_j["data"].is_array()) {
    nlohmann::json err;
    err["error"] = "output tensor '" + output_name + "' has no data array";
    return err;
  }

  const auto &data_j = tensor_j["data"];
  std::vector<float> scores;
  scores.reserve(data_j.size());
  for (const auto &v : data_j) {
    scores.push_back(v.get<float>());
  }

  return top_k_predictions(scores, labels, k, /*do_softmax=*/true);
}

} // namespace Onnx
