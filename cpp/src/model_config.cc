#include "qwen_onnx/model_config.h"

// config.json 的轻量读取器。
//
// C++ 侧只关心推理时必须知道的 token id 和维度，不尝试完整解析所有
// Transformers 配置字段。Find* 系列函数都提供 fallback，保证字段缺失时
// 仍能使用和当前 Qwen3-TTS Base 相符的默认值。

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace qwen::onnx {
namespace {

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int64_t FindInt(const std::string& text, const std::string& key, int64_t fallback) {
  std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)");
  std::smatch m;
  return std::regex_search(text, m, re) ? std::stoll(m[1].str()) : fallback;
}

std::string FindString(const std::string& text, const std::string& key, std::string fallback) {
  std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch m;
  return std::regex_search(text, m, re) ? m[1].str() : fallback;
}


std::string ExtractObject(const std::string& text, const std::string& key) {
  // talker_config 是嵌套对象，先从完整 config 中取出它再查内部字段。
  const auto key_pos = text.find("\"" + key + "\"");
  if (key_pos == std::string::npos) return text;
  const auto open = text.find('{', key_pos);
  if (open == std::string::npos) return text;
  int depth = 0;
  for (size_t i = open; i < text.size(); ++i) {
    if (text[i] == '{') ++depth;
    else if (text[i] == '}') {
      --depth;
      if (depth == 0) return text.substr(open, i - open + 1);
    }
  }
  return text;
}


std::string RemoveObject(const std::string& text, const std::string& key) {
  // 有些字段名在外层对象和内层对象都可能出现。
  // 例如 talker_config 内部还有 code_predictor_config，解析 talker 字段时先移除。
  const auto key_pos = text.find("\"" + key + "\"");
  if (key_pos == std::string::npos) return text;
  const auto open = text.find('{', key_pos);
  if (open == std::string::npos) return text;
  int depth = 0;
  for (size_t i = open; i < text.size(); ++i) {
    if (text[i] == '{') ++depth;
    else if (text[i] == '}') {
      --depth;
      if (depth == 0) {
        std::string out = text;
        out.erase(key_pos, i - key_pos + 1);
        return out;
      }
    }
  }
  return text;
}

std::unordered_map<std::string, int64_t> FindIntMap(const std::string& text, const std::string& key) {
  // codec_language_id 是 {"zh": id, "en": id, ...} 这种简单 map。
  std::unordered_map<std::string, int64_t> out;
  const auto pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) return out;
  const auto begin = text.find('{', pos);
  const auto end = text.find('}', begin);
  if (begin == std::string::npos || end == std::string::npos) return out;
  const std::string body = text.substr(begin + 1, end - begin - 1);
  std::regex item("\\\"([^\\\"]+)\\\"\\s*:\\s*(-?[0-9]+)");
  for (auto it = std::sregex_iterator(body.begin(), body.end(), item); it != std::sregex_iterator(); ++it) {
    out[(*it)[1].str()] = std::stoll((*it)[2].str());
  }
  return out;
}

}  // 匿名命名空间

ModelConfig LoadModelConfig(const std::filesystem::path& model_dir) {
  const std::string text = ReadText(model_dir / "config.json");
  ModelConfig cfg;
  // 文本侧 special tokens 来自主模型 config；它们会进入 text_project.onnx。
  cfg.tts_bos_token_id = FindInt(text, "tts_bos_token_id", cfg.tts_bos_token_id);
  cfg.tts_eos_token_id = FindInt(text, "tts_eos_token_id", cfg.tts_eos_token_id);
  cfg.tts_pad_token_id = FindInt(text, "tts_pad_token_id", cfg.tts_pad_token_id);
  cfg.tokenizer_type = FindString(text, "tokenizer_type", cfg.tokenizer_type);
  cfg.tts_model_type = FindString(text, "tts_model_type", cfg.tts_model_type);
  cfg.codec_frame_samples = FindInt(text, "decode_upsample_rate", cfg.codec_frame_samples);

  const std::string talker_text = RemoveObject(ExtractObject(text, "talker_config"), "code_predictor_config");
  auto& t = cfg.talker;
  // talker 相关字段决定 KV cache 层数、hidden size、codec token 过滤范围等。
  t.hidden_size = FindInt(talker_text, "hidden_size", t.hidden_size);
  t.vocab_size = FindInt(talker_text, "vocab_size", t.vocab_size);
  t.num_hidden_layers = FindInt(talker_text, "num_hidden_layers", t.num_hidden_layers);
  t.num_code_groups = FindInt(talker_text, "num_code_groups", t.num_code_groups);
  t.first_codebook_mask_tail = FindInt(talker_text, "first_codebook_mask_tail", t.first_codebook_mask_tail);
  t.codec_eos_token_id = FindInt(talker_text, "codec_eos_token_id", t.codec_eos_token_id);
  t.codec_bos_id = FindInt(talker_text, "codec_bos_id", t.codec_bos_id);
  t.codec_pad_id = FindInt(talker_text, "codec_pad_id", t.codec_pad_id);
  t.codec_nothink_id = FindInt(talker_text, "codec_nothink_id", t.codec_nothink_id);
  t.codec_think_id = FindInt(talker_text, "codec_think_id", t.codec_think_id);
  t.codec_think_bos_id = FindInt(talker_text, "codec_think_bos_id", t.codec_think_bos_id);
  t.codec_think_eos_id = FindInt(talker_text, "codec_think_eos_id", t.codec_think_eos_id);
  t.codec_language_id = FindIntMap(talker_text, "codec_language_id");
  return cfg;
}

}  // 命名空间 qwen::onnx
