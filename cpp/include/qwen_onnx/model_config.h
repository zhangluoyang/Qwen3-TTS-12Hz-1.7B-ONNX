#pragma once

// 从 HuggingFace/Qwen 模型目录的 config.json 中抽取 C++ 推理所需配置。
//
// 这里不是完整 JSON schema，只保留 ONNX runtime 会直接用到的字段：
// special token id、talker 维度、codec 控制 token、语言 token 映射等。

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace qwen::onnx {

// talker 是 Qwen3-TTS 里负责生成 codec token 的自回归模型。
// num_code_groups=16 表示每个 12Hz 帧由 16 个 RVQ codebook token 组成。
struct TalkerConfig {
  int64_t hidden_size = 2048;
  int64_t vocab_size = 3072;
  int64_t num_hidden_layers = 28;
  int64_t num_code_groups = 16;
  // The public Qwen3-TTS generation code masks the last 1024 talker vocab
  // entries during first-codebook sampling, except codec_eos_token_id.
  int64_t first_codebook_mask_tail = 1024;
  int64_t codec_eos_token_id = 2150;
  int64_t codec_bos_id = 2149;
  int64_t codec_pad_id = 2148;
  int64_t codec_nothink_id = 2155;
  int64_t codec_think_id = 2154;
  int64_t codec_think_bos_id = 2156;
  int64_t codec_think_eos_id = 2157;
  std::unordered_map<std::string, int64_t> codec_language_id;
};

struct ModelConfig {
  // 文本侧 TTS special tokens，进入 text_project.onnx 得到对应 embedding。
  int64_t tts_bos_token_id = 151672;
  int64_t tts_eos_token_id = 151673;
  int64_t tts_pad_token_id = 151671;
  std::string tokenizer_type = "qwen3_tts_tokenizer_12hz";
  std::string tts_model_type = "base";
  // 12Hz tokenizer decoder emits 1920 PCM samples per codec frame at 24 kHz.
  int64_t codec_frame_samples = 1920;
  TalkerConfig talker;
};

// 加载 model_dir/config.json，并用上面的默认值兜底。
ModelConfig LoadModelConfig(const std::filesystem::path& model_dir);

}  // 命名空间 qwen::onnx
