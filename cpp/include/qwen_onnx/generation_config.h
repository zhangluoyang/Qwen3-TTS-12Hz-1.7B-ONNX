#pragma once

// generation_config.json 的 C++ 轻量映射。
// 它决定 talker 第一 codebook token 和 code_predictor residual tokens
// 分别如何采样，例如 greedy、top-k、top-p、temperature、重复惩罚等。

#include <filesystem>

#include "qwen_onnx/sampling.h"

namespace qwen::onnx {

struct GenerationConfig {
  // talker 采样第 0 个 codec codebook token。
  SamplingOptions main_sampling;
  // code_predictor 采样剩余 15 个 residual codebook token。
  SamplingOptions code_sampling;
  int max_new_tokens = 8192;
};

GenerationConfig LoadGenerationConfig(const std::filesystem::path& model_dir);

}  // 命名空间 qwen::onnx
