#include "qwen_onnx/sampling.h"

// C++ 采样逻辑与 scripts/onnx_runtime/sampling.py 对齐。
// 主 talker 和 code_predictor 都会走这里，只是 logits 来源不同。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace qwen::onnx {
namespace {

std::vector<double> Softmax(const std::vector<double>& logits) {
  // 先减最大值防止 exp 溢出；被过滤成 -inf 的 token 概率自然为 0。
  const double max_v = *std::max_element(logits.begin(), logits.end());
  std::vector<double> probs(logits.size());
  double sum = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = std::isfinite(logits[i]) ? std::exp(logits[i] - max_v) : 0.0;
    sum += probs[i];
  }
  if (sum <= 0.0) throw std::runtime_error("Softmax sum is zero");
  for (double& p : probs) p /= sum;
  return probs;
}

}  // 匿名命名空间

int64_t Sampler::Sample(const std::vector<double>& logits_in,
                        const SamplingOptions& options,
                        const std::vector<int64_t>& generated) {
  if (logits_in.empty()) throw std::invalid_argument("Cannot sample from empty logits");
  std::vector<double> logits = logits_in;

  if (options.repetition_penalty != 1.0f) {
    // repetition penalty 使用 Transformers 常见语义：
    // 正 logit 除以 penalty，负 logit 乘以 penalty，降低已生成 token 再出现概率。
    std::unordered_set<int64_t> unique_tokens(generated.begin(), generated.end());
    for (int64_t token : unique_tokens) {
      if (token < 0 || static_cast<size_t>(token) >= logits.size()) continue;
      double& v = logits[static_cast<size_t>(token)];
      if (v < 0.0) v *= options.repetition_penalty;
      else v /= options.repetition_penalty;
    }
  }

  if (!options.do_sample) {
    // greedy 对齐 Python --greedy 和 PyTorch 对比脚本，用于排查数值差异。
    return static_cast<int64_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
  }

  const double temperature = options.temperature > 0.0f ? options.temperature : 1.0f;
  for (double& v : logits) v /= temperature;

  if (options.top_k > 0 && static_cast<size_t>(options.top_k) < logits.size()) {
    // top-k：只保留 logit 最大的 k 个候选。
    std::vector<double> tmp = logits;
    std::nth_element(tmp.begin(), tmp.end() - options.top_k, tmp.end());
    const double kth = *(tmp.end() - options.top_k);
    for (double& v : logits) {
      if (v < kth) v = -std::numeric_limits<double>::infinity();
    }
  }

  if (options.top_p < 1.0f) {
    // nucleus/top-p：按概率从高到低累加，超过阈值后的 token 置为 -inf。
    std::vector<size_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return logits[a] > logits[b]; });
    std::vector<double> sorted_logits;
    sorted_logits.reserve(order.size());
    for (size_t idx : order) sorted_logits.push_back(logits[idx]);
    auto probs = Softmax(sorted_logits);
    double cumulative = 0.0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
      cumulative += probs[rank];
      if (rank > 0 && cumulative > options.top_p) {
        logits[order[rank]] = -std::numeric_limits<double>::infinity();
      }
    }
  }

  auto probs = Softmax(logits);
  std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
  return dist(rng_);
}

int64_t Sampler::Sample(const std::vector<float>& logits_f,
                        const SamplingOptions& options,
                        const std::vector<int64_t>& generated) {
  std::vector<double> logits(logits_f.begin(), logits_f.end());
  return Sample(logits, options, generated);
}

}  // 命名空间 qwen::onnx
