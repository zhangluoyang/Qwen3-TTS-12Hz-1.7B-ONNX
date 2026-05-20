#include "qwen_onnx/generation_config.h"

// generation_config.json 只需要读取少量标量字段，因此这里用轻量正则解析。
// 生产级 JSON 解析可以换成 nlohmann/json；当前写法让 runtime 依赖更少。

#include <fstream>
#include <regex>
#include <sstream>

namespace qwen::onnx {
namespace {

std::string ReadTextIfExists(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int FindInt(const std::string& text, const std::string& key, int fallback) {
  // 找不到字段时用结构体默认值兜底，方便老模型目录缺少某些采样字段。
  std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)");
  std::smatch m;
  return std::regex_search(text, m, re) ? std::stoi(m[1].str()) : fallback;
}

float FindFloat(const std::string& text, const std::string& key, float fallback) {
  std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
  std::smatch m;
  return std::regex_search(text, m, re) ? std::stof(m[1].str()) : fallback;
}

bool FindBool(const std::string& text, const std::string& key, bool fallback) {
  std::regex re("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
  std::smatch m;
  if (!std::regex_search(text, m, re)) return fallback;
  return m[1].str() == "true";
}

}  // 匿名命名空间

GenerationConfig LoadGenerationConfig(const std::filesystem::path& model_dir) {
  GenerationConfig cfg;
  const auto text = ReadTextIfExists(model_dir / "generation_config.json");
  if (text.empty()) return cfg;
  // main_sampling 对应 talker 的第 0 个 codec token。
  cfg.main_sampling.do_sample = FindBool(text, "do_sample", cfg.main_sampling.do_sample);
  cfg.main_sampling.top_k = FindInt(text, "top_k", cfg.main_sampling.top_k);
  cfg.main_sampling.top_p = FindFloat(text, "top_p", cfg.main_sampling.top_p);
  cfg.main_sampling.temperature = FindFloat(text, "temperature", cfg.main_sampling.temperature);
  cfg.main_sampling.repetition_penalty = FindFloat(text, "repetition_penalty", cfg.main_sampling.repetition_penalty);
  // code_sampling 对应 subtalker/code_predictor，负责每帧剩余 15 个 codebook token。
  cfg.code_sampling.do_sample = FindBool(text, "subtalker_dosample", cfg.code_sampling.do_sample);
  cfg.code_sampling.top_k = FindInt(text, "subtalker_top_k", cfg.code_sampling.top_k);
  cfg.code_sampling.top_p = FindFloat(text, "subtalker_top_p", cfg.code_sampling.top_p);
  cfg.code_sampling.temperature = FindFloat(text, "subtalker_temperature", cfg.code_sampling.temperature);
  cfg.code_sampling.repetition_penalty = 1.0f;
  cfg.max_new_tokens = FindInt(text, "max_new_tokens", cfg.max_new_tokens);
  return cfg;
}

}  // 命名空间 qwen::onnx
