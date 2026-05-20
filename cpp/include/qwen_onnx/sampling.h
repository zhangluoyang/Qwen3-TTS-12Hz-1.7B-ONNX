#pragma once

// talker/code_predictor 共用的采样器。
// 支持 greedy、temperature、top-k、top-p 和 repetition penalty。

#include <cstdint>
#include <random>
#include <vector>

namespace qwen::onnx {

struct SamplingOptions {
  // false 表示 greedy argmax；true 表示按过滤后的概率分布随机采样。
  bool do_sample = true;
  int top_k = 50;
  float top_p = 1.0f;
  float temperature = 0.9f;
  float repetition_penalty = 1.05f;
};

class Sampler {
 public:
  explicit Sampler(uint64_t seed = 1234) : rng_(seed) {}

  // generated 只在 talker 第一 token 采样中用于 repetition penalty。
  int64_t Sample(const std::vector<double>& logits,
                 const SamplingOptions& options,
                 const std::vector<int64_t>& generated = {});

  int64_t Sample(const std::vector<float>& logits,
                 const SamplingOptions& options,
                 const std::vector<int64_t>& generated = {});

 private:
  std::mt19937_64 rng_;
};

}  // 命名空间 qwen::onnx
