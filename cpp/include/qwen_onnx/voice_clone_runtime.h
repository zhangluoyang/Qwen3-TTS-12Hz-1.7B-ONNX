#pragma once

// Qwen3-TTS ONNX 声音克隆 C++ runtime 的主入口。
//
// 这个类把拆开的 ONNX 子模型重新串成完整 pipeline：
//   reference audio -> codec codes + speaker embedding
//   target/ref text -> text embedding
//   prompt prefill -> 自回归生成 codec codes
//   codec codes -> tokenizer decoder -> waveform
//
// Python 版本在 scripts/onnx_runtime/voice_clone_ort.py；这里的实现尽量保持
// 张量命名和调试 dump 文件一致，方便逐步对齐和学习。

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "qwen_onnx/bpe_tokenizer.h"
#include "qwen_onnx/model_config.h"
#include "qwen_onnx/ort_session.h"
#include "qwen_onnx/sampling.h"
#include "qwen_onnx/tensor.h"

namespace qwen::onnx {

struct RuntimeOptions {
  // HuggingFace/Qwen 原模型目录：读取 config、generation_config、tokenizer 文件。
  std::filesystem::path model_dir;
  // 导出的 ONNX 子模型根目录，例如 ./onnx_isolated_fp16。
  std::filesystem::path onnx_root;
  // Heavy generation/decoder sessions. Use CUDA by default.
  std::vector<std::string> providers{"CUDAExecutionProvider"};
  // Optional tokenizer decoder provider override. Empty means use providers.
  std::vector<std::string> decode_providers;
  // Lightweight preprocessing/embedding sessions. Keep on CPU by default for
  // batch=1 latency and lower GPU memory pressure.
  std::vector<std::string> prep_providers{"CPUExecutionProvider"};
  uint64_t seed = 1234;
  int cuda_device_id = 0;
  // Base voice clone needs tokenizer_encode + speaker_encoder. CustomVoice does
  // not have/use those ONNX files, so it can skip loading them.
  bool load_reference_frontend = true;
};

struct VoiceCloneInputs {
  // 文本分词结果刻意由外部注入。生产应用可以接入
  // HuggingFace 分词器、sentencepiece/BPE 或服务端 tokenizer。
  std::vector<int64_t> assistant_text_ids;
  std::vector<int64_t> reference_text_ids;

  // 可选的预计算前端结果。后续即使补全 C++ 音频前端，
  // 也不需要改动 ONNX 生成核心。
  Int64Tensor reference_codes;       // [参考帧数, 16]
  FloatTensor speaker_embedding;     // [1, 1, hidden_size]

  // language="auto" 时使用 codec_nothink 控制 token；指定语言时会插入语言 id。
  std::string language = "auto";
  // 与 Transformers generate() 对齐：实际最多生成 max_new_tokens - 1 个 codec 帧。
  int max_new_tokens = 300;
  // main_sampling 控制 talker 第 0 个 codebook；code_sampling 控制 residual codebooks。
  SamplingOptions main_sampling;
  SamplingOptions code_sampling;

  // 非空时写出 prompt/logits/KV/cache/codes/waveform 等 .npy，供 Python/C++ 对齐。
  std::filesystem::path debug_dump_dir;
};

struct VoiceCloneResult {
  Int64Tensor generated_codes;  // [帧数, 16]
  FloatTensor waveform;         // [采样点数]
  int sample_rate = 24000;
};

struct VoiceCloneChunkOptions {
  // 每生成多少 codec 帧就触发一次 tokenizer_decode_chunk。
  int chunk_frames = 50;
  // chunk 解码时带多少左侧上下文帧，降低块边界不连续。
  int left_context_frames = 25;
  // true 时用后台线程解码 chunk，让生成线程继续往前跑。
  bool async_chunk_decode = false;
  int decode_workers = 1;
  // async 模式下最多缓存多少个未解码/待发出的 chunk。
  int max_decode_queue = 2;
};

struct VoiceCloneChunk {
  // 这一段 chunk 对应的 PCM 音频，已经裁掉左上下文。
  FloatTensor audio;
  int sample_rate = 24000;
  // start/end 是 full_codes 坐标：reference_codes + generated_codes。
  int64_t start_frame = 0;
  int64_t end_frame = 0;
  int64_t generated_frames = 0;
  bool is_final = false;
};

// 和 Python iter_voice_clone_chunked() 的 yield 语义对应：
// 每个 chunk 解码完成后立即回调一次，调用方可以在这里播放、推流或写分片。
using VoiceCloneChunkCallback = std::function<void(const VoiceCloneChunk&)>;

struct VoiceCloneChunkedResult {
  Int64Tensor generated_codes;  // [帧数, 16]
  std::vector<VoiceCloneChunk> chunks;
  FloatTensor waveform;         // 拼接后的 chunk 音频
  int sample_rate = 24000;
};

struct VoiceCloneRequest {
  // 高层接口输入，runtime 内部会负责文本分词和参考音频前端。
  std::string text;
  std::string reference_text;
  std::filesystem::path reference_audio;
  std::string language = "auto";
  int max_new_tokens = 300;
  bool x_vector_only_mode = false;
  SamplingOptions main_sampling;
  SamplingOptions code_sampling;
};

struct CustomVoiceRequest {
  std::string text;
  std::string speaker = "Vivian";
  std::string language = "auto";
  // 0.6B CustomVoice ignores instruct in the official wrapper; keep the field
  // here so higher-level callers can share one API with larger variants later.
  std::string instruct;
  bool non_streaming_mode = true;
  int max_new_tokens = 300;
  SamplingOptions main_sampling;
  SamplingOptions code_sampling;
  std::filesystem::path debug_dump_dir;
};

struct VoiceDesignRequest {
  std::string text;
  std::string instruct = "一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。";
  std::string language = "auto";
  bool non_streaming_mode = true;
  int max_new_tokens = 300;
  SamplingOptions main_sampling;
  SamplingOptions code_sampling;
  std::filesystem::path debug_dump_dir;
};

struct TimingRecord {
  std::string name;
  int64_t count = 0;
  double total_ms = 0.0;
};

struct ReferenceAudioFeatures {
  // 参考音频缓存项。Gradio/服务端场景里同一参考音频常被多次复用。
  Int64Tensor reference_codes;
  FloatTensor speaker_embedding;
};

// VoiceCloneRuntime 是“把所有子模型接线”的地方。
//
// 阅读建议：
// 1. 先看 GenerateVoiceClone()/GenerateVoiceCloneChunked()，了解高层入口；
// 2. 再看 BuildTalkerPrompt()，这是最容易困惑但最关键的 prompt 拼装；
// 3. 最后看 GenerateFromPrepared() 主循环：talker 生成首 token，
//    RunCodePredictor() 补齐剩余 15 个 token，DecodeCodes() 合成音频。
class VoiceCloneRuntime {
 public:
  explicit VoiceCloneRuntime(RuntimeOptions options);

  const ModelConfig& Config() const { return config_; }
  void PrintTimingSummary(std::ostream& os, const std::string& title = "[Timing] Detail") const;

  FloatTensor TextProject(const Int64Tensor& input_ids) const;
  FloatTensor CodecEmbed(const Int64Tensor& token_ids) const;
  FloatTensor CodePredictorEmbed(const Int64Tensor& token_ids, int64_t layer_idx) const;
  // 完整 decoder：一次性解码全部 codes，非流式路径使用。
  FloatTensor DecodeCodes(const Int64Tensor& codes, int64_t* output_length = nullptr) const;
  // chunk decoder：只输出 [start_frame, end_frame) 这段新音频。
  FloatTensor DecodeCodesChunk(const Int64Tensor& full_codes,
                               int64_t start_frame,
                               int64_t end_frame,
                               int64_t left_context_frames,
                               int64_t* output_length = nullptr);
  Int64Tensor EncodeReferenceCodes(const FloatTensor& audio) const;
  FloatTensor ExtractSpeakerEmbedding(const FloatTensor& mel) const;
  std::vector<int64_t> EncodeAssistantText(const std::string& text) const;
  std::vector<int64_t> EncodeReferenceText(const std::string& text) const;

  // 文本 id、参考 codes、说话人 embedding 准备好之后的核心生成路径。
  VoiceCloneResult GenerateFromPrepared(const VoiceCloneInputs& inputs);
  VoiceCloneResult GenerateVoiceClone(const VoiceCloneRequest& request);
  VoiceCloneResult GenerateCustomVoice(const CustomVoiceRequest& request);
  VoiceCloneResult GenerateVoiceDesign(const VoiceDesignRequest& request);
  VoiceCloneChunkedResult GenerateFromPreparedChunked(const VoiceCloneInputs& inputs,
                                                      const VoiceCloneChunkOptions& chunk_options,
                                                      const VoiceCloneChunkCallback& on_chunk = {});
  VoiceCloneChunkedResult GenerateVoiceCloneChunked(const VoiceCloneRequest& request,
                                                    const VoiceCloneChunkOptions& chunk_options,
                                                    const VoiceCloneChunkCallback& on_chunk = {});

 private:
  OrtSession& ChunkDecoder();
  // 下面这些私有函数对应 pipeline 中间步骤，保持粒度小是为了方便逐步对齐。
  std::vector<int64_t> LanguagePrefillIds(const std::string& language) const;
  std::vector<int64_t> CustomVoiceLanguagePrefillIds(const std::string& language, const std::string& speaker) const;
  FloatTensor CustomVoiceSpeakerEmbedding(const std::string& speaker) const;
  FloatTensor ReferenceCodeEmbedding(const Int64Tensor& ref_codes) const;
  FloatTensor BuildTalkerPrompt(const VoiceCloneInputs& inputs, FloatTensor* trailing_text, FloatTensor* tts_pad) const;
  FloatTensor BuildCustomVoicePrompt(const CustomVoiceRequest& request, FloatTensor* trailing_text, FloatTensor* tts_pad) const;
  FloatTensor BuildVoiceDesignPrompt(const VoiceDesignRequest& request, FloatTensor* trailing_text, FloatTensor* tts_pad) const;
  VoiceCloneResult GenerateFromPrompt(const FloatTensor& prompt,
                                      const FloatTensor& trailing_text,
                                      const FloatTensor& tts_pad,
                                      const Int64Tensor& reference_codes,
                                      int max_new_tokens,
                                      const SamplingOptions& main_sampling,
                                      const SamplingOptions& code_sampling,
                                      const std::filesystem::path& debug_dump_dir);
  std::pair<std::vector<int64_t>, FloatTensor> RunCodePredictor(const FloatTensor& past_hidden,
                                                                 int64_t first_token,
                                                                 const SamplingOptions& options,
                                                                 int frame_index,
                                                                 const std::filesystem::path& debug_dump_dir);
  ReferenceAudioFeatures GetReferenceAudioFeatures(const std::filesystem::path& audio_path, bool x_vector_only_mode);
  void AddTiming(const std::string& name, double milliseconds) const;

  RuntimeOptions options_;
  Ort::Env env_;
  ModelConfig config_;
  std::unique_ptr<Qwen2BpeTokenizer> tokenizer_;
  Sampler sampler_;
  // key 包含绝对路径、文件大小、mtime 和 x_vector_only_mode，避免同名文件变化后误命中。
  std::unordered_map<std::string, ReferenceAudioFeatures> reference_audio_cache_;
  mutable std::vector<TimingRecord> timing_;
  mutable std::mutex timing_mutex_;

  OrtSession text_project_;
  OrtSession codec_embed_;
  OrtSession code_predictor_embed_;
  OrtSession speaker_encoder_;
  OrtSession tokenizer_encode_;
  OrtSession tokenizer_decode_;
  std::unique_ptr<OrtSession> tokenizer_decode_chunk_;
  OrtSession code_predictor_;
  OrtSession talker_prefill_;
  OrtSession talker_decode_;
  // code_predictor residual token embedding 在生成循环里重复率很高，缓存单 token 查询。
  mutable std::unordered_map<uint64_t, FloatTensor> code_predictor_embed_cache_;
};

}  // 命名空间 qwen::onnx
