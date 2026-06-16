/**
 * @file Classify.hpp
 * @brief Post-processing helpers for image classification ONNX models.
 *
 * Takes raw logit/score output tensors from a classification model (e.g.
 * MobileNetV2), applies softmax, ranks by confidence, and returns the top-k
 * predictions enriched with human-readable labels from an ImageNet class file.
 *
 * Output JSON format:
 * @code
 * {
 *   "best": { "class_id": 281, "label": "tabby cat", "confidence": 0.9143 },
 *   "top_k": [
 *     { "class_id": 281, "label": "tabby cat",  "confidence": 0.9143 },
 *     { "class_id": 282, "label": "tiger cat",  "confidence": 0.0421 },
 *     ...
 *   ]
 * }
 * @endcode
 *
 * @author Paolo Bosetti
 * @date   2026
 */

#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace Onnx {

/**
 * @brief Prediction record for a single class.
 */
struct Prediction {
  int class_id;        ///< Zero-based ImageNet class index.
  std::string label;   ///< Human-readable class name.
  float confidence;    ///< Softmax probability in [0, 1].
};

/**
 * @brief Load ImageNet class labels from a text file.
 *
 * The file must contain exactly one label per line (1000 lines for standard
 * ImageNet).  Lines are stripped of leading/trailing whitespace.  Returns an
 * empty vector on error (file not found, I/O error, …).
 *
 * @param path  Filesystem path to the labels file.
 * @return      Vector of label strings, indexed by class ID.
 */
std::vector<std::string> load_labels(const std::string &path);

/**
 * @brief Apply softmax to a flat logit/score vector in-place.
 *
 * Uses the numerically stable max-subtraction trick.
 *
 * @param logits  Input logit vector; modified in-place to probabilities.
 */
void softmax_inplace(std::vector<float> &logits);

/**
 * @brief Rank model output scores and return the top-k predictions.
 *
 * @param scores   Flat float32 score/logit array (one value per class).
 * @param labels   Label strings from load_labels().  May be empty or shorter
 *                 than @p scores; missing labels are reported as "class_<id>".
 * @param k        Number of top predictions to return (clamped to class count).
 * @param do_softmax  If true, apply softmax before ranking.
 * @return JSON object {"best":…, "top_k":[…]}.
 */
nlohmann::json top_k_predictions(const std::vector<float> &scores,
                                 const std::vector<std::string> &labels,
                                 int k = 5,
                                 bool do_softmax = true);

/**
 * @brief Convenience: extract scores from a raw model output JSON tensor and
 *        return top-k predictions.
 *
 * @param output_j   JSON object returned by run_inference() — maps output
 *                   tensor names to tensor JSON objects.
 * @param output_name  Name of the tensor containing class logits/scores
 *                     (e.g. "output" for MobileNetV2).
 * @param labels     Label strings from load_labels().
 * @param k          Top-k count.
 * @return JSON object {"best":…, "top_k":[…]}, or an error JSON if the
 *         named tensor is absent or the data array is empty.
 */
nlohmann::json classify(const nlohmann::json &output_j,
                        const std::string &output_name,
                        const std::vector<std::string> &labels,
                        int k = 5);

} // namespace Onnx
