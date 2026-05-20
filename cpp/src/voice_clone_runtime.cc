#include "qwen_onnx/voice_clone_runtime.h"

// C++ 版完整声音克隆 runtime。
//
// 建议阅读顺序：
//   1. VoiceCloneRuntime::GenerateVoiceClone()        高层入口；
//   2. GetReferenceAudioFeatures()                    参考音频 -> codes + speaker embedding；
//   3. BuildTalkerPrompt()                            文本/音色/参考 codec prompt 拼接；
//   4. GenerateFromPrepared()                         非流式生成主循环；
//   5. RunCodePredictor()                             每帧 residual codebook 生成；
//   6. GenerateFromPreparedChunked()                  chunk/pipeline 解码实验路径。
//
// 这里刻意和 Python 版本的张量 dump 文件名保持一致，方便用
// scripts/onnx_runtime/compare_cpp_voice_clone.py 做逐张量对齐。

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <dlfcn.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "qwen_onnx/audio_frontend.h"
#include "qwen_onnx/bpe_tokenizer.h"

namespace qwen::onnx {
namespace {

OrtSessionConfig SessionCfg(const std::filesystem::path& path, const std::vector<std::string>& providers) {
  // 给每个子模型构建 session 配置。真正的 CUDA device id 在重载版本中补上。
  OrtSessionConfig cfg;
  cfg.model_path = path.string();
  cfg.providers = providers;
  return cfg;
}

OrtSessionConfig SessionCfg(const std::filesystem::path& path, const std::vector<std::string>& providers, int cuda_device_id) {
  auto cfg = SessionCfg(path, providers);
  cfg.cuda_device_id = cuda_device_id;
  return cfg;
}

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string ReferenceAudioCacheKey(const std::filesystem::path& path, bool x_vector_only_mode) {
  // 缓存 key 包含路径、文件大小、mtime 和模式。这样同一路径的参考音频被替换后，
  // 不会误用旧的 reference_codes / speaker_embedding。
  const auto abs_path = std::filesystem::absolute(path).lexically_normal();
  const auto size = std::filesystem::file_size(path);
  const auto mtime = std::filesystem::last_write_time(path).time_since_epoch().count();
  std::ostringstream os;
  os << abs_path.string() << "|" << size << "|" << mtime << "|" << (x_vector_only_mode ? 1 : 0);
  return os.str();
}

Int64Tensor RowTensor(const std::vector<int64_t>& ids) {
  return Int64Tensor({1, static_cast<int64_t>(ids.size())}, ids);
}

Int64Tensor Scalar(int64_t value) {
  return Int64Tensor({}, std::vector<int64_t>{value});
}

bool UsesPrepProviders(const std::string& session_name) {
  // 小模型/前处理默认走 prep_providers。batch=1 时把这些放 CPU 往往更省显存，
  // 也避免频繁启动小 CUDA kernel。
  return session_name == "text_project" ||
         session_name == "codec_embed" ||
         session_name == "code_predictor_embed" ||
         session_name == "speaker_encoder" ||
         session_name == "tokenizer_encode";
}

const std::vector<std::string>& ProvidersForSession(const RuntimeOptions& options,
                                                    const std::string& session_name) {
  return UsesPrepProviders(session_name) ? options.prep_providers : options.providers;
}

uint64_t CodePredictorEmbedCacheKey(int64_t token_id, int64_t layer_idx) {
  // code_predictor_embed 是 residual codebook 的 token embedding。
  // 生成时单 token 查询很多，缓存 (layer_idx, token_id) 能少跑不少小 session。
  return (static_cast<uint64_t>(static_cast<uint32_t>(layer_idx)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(token_id));
}

bool TryGetSingleToken(const Int64Tensor& token_ids, int64_t* token_id) {
  if (token_ids.size() != 1) return false;
  const auto& shape = token_ids.shape();
  if (shape.size() != 2 || shape[0] != 1 || shape[1] != 1) return false;
  *token_id = token_ids.values()[0];
  return true;
}

struct ChunkDecodeTask {
  // 一个待解码 chunk。full_codes 是 reference_codes + 当前已生成 codes 的快照。
  size_t index = 0;
  Int64Tensor full_codes;
  int64_t start_frame = 0;
  int64_t end_frame = 0;
  int64_t left_context_frames = 0;
  int64_t generated_frames = 0;
  bool is_final = false;
};

struct ChunkDecodeResult {
  size_t index = 0;
  VoiceCloneChunk chunk;
};

class AsyncChunkDecoder {
 public:
  using DecodeFn = std::function<ChunkDecodeResult(ChunkDecodeTask)>;

  AsyncChunkDecoder(DecodeFn decode_fn, size_t max_queue)
      : decode_fn_(std::move(decode_fn)),
        max_queue_(std::max<size_t>(1, max_queue)),
        worker_([this] { WorkerLoop(); }) {}

  ~AsyncChunkDecoder() {
    Cancel();
  }

  void Enqueue(ChunkDecodeTask task) {
    // 有界队列提供背压：解码太慢时生成线程会在这里等待，避免内存无限涨。
    std::unique_lock<std::mutex> lock(mutex_);
    task_cv_.wait(lock, [&] {
      return cancelled_ || exception_ != nullptr || tasks_.size() < max_queue_;
    });
    CheckErrorLocked();
    if (cancelled_ || input_closed_) {
      throw std::runtime_error("Async chunk decoder is closed");
    }
    tasks_.push_back(std::move(task));
    lock.unlock();
    task_cv_.notify_one();
  }

  bool TryPop(size_t index, VoiceCloneChunk* chunk) {
    std::lock_guard<std::mutex> lock(mutex_);
    CheckErrorLocked();
    auto it = results_.find(index);
    if (it == results_.end()) return false;
    *chunk = std::move(it->second);
    results_.erase(it);
    return true;
  }

  VoiceCloneChunk WaitPop(size_t index) {
    // 按 index 顺序取结果，即使后台线程先完成了后面的 chunk，也不会乱序播放。
    std::unique_lock<std::mutex> lock(mutex_);
    result_cv_.wait(lock, [&] {
      return cancelled_ || exception_ != nullptr || results_.find(index) != results_.end() || worker_done_;
    });
    CheckErrorLocked();
    auto it = results_.find(index);
    if (it == results_.end()) {
      throw std::runtime_error("Async chunk decoder stopped before producing chunk " + std::to_string(index));
    }
    VoiceCloneChunk chunk = std::move(it->second);
    results_.erase(it);
    return chunk;
  }

  void CloseInput() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      input_closed_ = true;
    }
    task_cv_.notify_all();
  }

  void Join() {
    if (worker_.joinable()) worker_.join();
  }

 private:
  void CheckErrorLocked() const {
    if (exception_ != nullptr) std::rethrow_exception(exception_);
  }

  void Cancel() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = true;
      tasks_.clear();
    }
    task_cv_.notify_all();
    result_cv_.notify_all();
    Join();
  }

  void WorkerLoop() {
    try {
      while (true) {
        ChunkDecodeTask task;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          task_cv_.wait(lock, [&] {
            return cancelled_ || input_closed_ || !tasks_.empty();
          });
          if (cancelled_) break;
          if (tasks_.empty()) {
            if (input_closed_) break;
            continue;
          }
          task = std::move(tasks_.front());
          tasks_.pop_front();
        }
        task_cv_.notify_all();

        auto decoded = decode_fn_(std::move(task));
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (cancelled_) break;
          results_.emplace(decoded.index, std::move(decoded.chunk));
        }
        result_cv_.notify_all();
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        exception_ = std::current_exception();
      }
      task_cv_.notify_all();
      result_cv_.notify_all();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      worker_done_ = true;
    }
    result_cv_.notify_all();
  }

  DecodeFn decode_fn_;
  size_t max_queue_ = 1;
  std::thread worker_;
  mutable std::mutex mutex_;
  std::condition_variable task_cv_;
  std::condition_variable result_cv_;
  std::deque<ChunkDecodeTask> tasks_;
  std::map<size_t, VoiceCloneChunk> results_;
  std::exception_ptr exception_;
  bool input_closed_ = false;
  bool cancelled_ = false;
  bool worker_done_ = false;
};

std::vector<float> LastLogits(const FloatTensor& logits) {
  // logits shape [B,T,V]，采样只需要最后一个时间步 [V]。
  const auto& s = logits.shape();
  if (s.size() != 3) throw std::runtime_error("Expected logits rank 3");
  const int64_t vocab = s[2];
  const size_t begin = static_cast<size_t>((s[1] - 1) * vocab);
  return std::vector<float>(logits.values().begin() + static_cast<long>(begin),
                            logits.values().begin() + static_cast<long>(begin + vocab));
}

bool ReadNpyInt64Scalar(const std::filesystem::path& path, int64_t* value) {
  // 调试对齐用：如果 dump_dir 里存在 Python 采样出来的 token，就强制 C++ 用同一个。
  // 这样能把“采样随机性”从数值对比中剥离出来。
  if (!std::filesystem::exists(path)) return false;
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  char magic[6] = {0};
  in.read(magic, 6);
  if (in.gcount() != 6 || std::string(magic, 6) != "\x93NUMPY") return false;
  char ver[2] = {0};
  in.read(ver, 2);
  if (in.gcount() != 2) return false;

  uint32_t header_len = 0;
  if (ver[0] == 1) {
    unsigned char len[2] = {0, 0};
    in.read(reinterpret_cast<char*>(len), 2);
    if (in.gcount() != 2) return false;
    header_len = static_cast<uint32_t>(len[0]) | (static_cast<uint32_t>(len[1]) << 8);
  } else {
    unsigned char len[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(len), 4);
    if (in.gcount() != 4) return false;
    header_len = static_cast<uint32_t>(len[0]) | (static_cast<uint32_t>(len[1]) << 8) |
                 (static_cast<uint32_t>(len[2]) << 16) | (static_cast<uint32_t>(len[3]) << 24);
  }

  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!in) return false;
  if (header.find("'descr': '<i8'") == std::string::npos && header.find("\"descr\": \"<i8\"") == std::string::npos) {
    return false;
  }

  int64_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (in.gcount() != static_cast<std::streamsize>(sizeof(v))) return false;
  *value = v;
  return true;
}

class CudaRuntime {
 public:
  using CudaMemcpyFn = int (*)(void*, const void*, size_t, int);

  static CudaRuntime& Instance() {
    static CudaRuntime runtime;
    return runtime;
  }

  void CopyDeviceToHost(void* dst, const void* src, size_t bytes) const {
    // ORT 的 C++ API 没有直接提供任意 device pointer 的拷贝工具，
    // 这里动态加载 libcudart，只在确实需要从 CUDA 输出读 logits/hidden 时使用。
    if (bytes == 0) return;
    if (cuda_memcpy_ == nullptr) {
      throw std::runtime_error("CUDA tensor copy requires libcudart, but it was not found");
    }
    constexpr int kCudaMemcpyDeviceToHost = 2;
    const int status = cuda_memcpy_(dst, src, bytes, kCudaMemcpyDeviceToHost);
    if (status != 0) {
      throw std::runtime_error("cudaMemcpyDeviceToHost failed with status " + std::to_string(status));
    }
  }

 private:
  CudaRuntime() {
    const char* names[] = {"libcudart.so", "libcudart.so.13", "libcudart.so.12"};
    for (const char* name : names) {
      handle_ = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
      if (handle_ != nullptr) break;
    }
    if (handle_ != nullptr) {
      cuda_memcpy_ = reinterpret_cast<CudaMemcpyFn>(dlsym(handle_, "cudaMemcpy"));
    }
  }

  void* handle_ = nullptr;
  CudaMemcpyFn cuda_memcpy_ = nullptr;
};

bool IsCudaTensor(const Ort::Value& value) {
  return value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU;
}

std::vector<float> LastLogits(const Ort::Value& logits) {
  // I/O Binding 模式下 logits 可能在 CPU，也可能在 CUDA；并且 FP16/FP32 都要处理。
  // 输出统一转换为 float32 vector，交给 Sampler。
  auto info = logits.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  if (shape.size() != 3) throw std::runtime_error("Expected logits rank 3");
  const int64_t vocab = shape[2];
  const int64_t t = shape[1];
  size_t begin = static_cast<size_t>((t - 1) * vocab);
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const auto* data = logits.GetTensorData<Ort::Float16_t>();
    std::vector<Ort::Float16_t> tmp;
    if (IsCudaTensor(logits)) {
      tmp.resize(static_cast<size_t>(vocab));
      CudaRuntime::Instance().CopyDeviceToHost(tmp.data(), data + begin, tmp.size() * sizeof(Ort::Float16_t));
      data = tmp.data();
      begin = 0;
    }
    std::vector<float> out(static_cast<size_t>(vocab));
    for (int64_t i = 0; i < vocab; ++i) out[static_cast<size_t>(i)] = data[begin + static_cast<size_t>(i)].ToFloat();
    return out;
  }
  const float* data = logits.GetTensorData<float>();
  if (IsCudaTensor(logits)) {
    std::vector<float> out(static_cast<size_t>(vocab));
    CudaRuntime::Instance().CopyDeviceToHost(out.data(), data + begin, out.size() * sizeof(float));
    return out;
  }
  return std::vector<float>(data + begin, data + begin + vocab);
}

FloatTensor CopyFloatTensor(const Ort::Value& value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  const size_t count = info.GetElementCount();
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const auto* src = value.GetTensorData<Ort::Float16_t>();
    std::vector<Ort::Float16_t> tmp;
    if (IsCudaTensor(value)) {
      tmp.resize(count);
      CudaRuntime::Instance().CopyDeviceToHost(tmp.data(), src, tmp.size() * sizeof(Ort::Float16_t));
      src = tmp.data();
    }
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = src[i].ToFloat();
    return FloatTensor(std::move(shape), std::move(values));
  }
  const float* src = value.GetTensorData<float>();
  if (IsCudaTensor(value)) {
    std::vector<float> values(count);
    CudaRuntime::Instance().CopyDeviceToHost(values.data(), src, values.size() * sizeof(float));
    return FloatTensor(std::move(shape), std::move(values));
  }
  return FloatTensor(std::move(shape), std::vector<float>(src, src + count));
}

FloatTensor LastHidden(const Ort::Value& hidden) {
  // code_predictor 需要上一轮 talker 的最后 hidden state 作为上下文起点。
  auto info = hidden.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  if (shape.size() != 3) throw std::runtime_error("Expected hidden rank 3");
  const int64_t width = shape[2];
  size_t begin = static_cast<size_t>((shape[1] - 1) * width);
  std::vector<float> values(static_cast<size_t>(width));
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const auto* data = hidden.GetTensorData<Ort::Float16_t>();
    std::vector<Ort::Float16_t> tmp;
    if (IsCudaTensor(hidden)) {
      tmp.resize(static_cast<size_t>(width));
      CudaRuntime::Instance().CopyDeviceToHost(tmp.data(), data + begin, tmp.size() * sizeof(Ort::Float16_t));
      data = tmp.data();
      begin = 0;
    }
    for (int64_t i = 0; i < width; ++i) values[static_cast<size_t>(i)] = data[begin + static_cast<size_t>(i)].ToFloat();
  } else {
    const float* data = hidden.GetTensorData<float>();
    if (IsCudaTensor(hidden)) {
      CudaRuntime::Instance().CopyDeviceToHost(values.data(), data + begin, values.size() * sizeof(float));
    } else {
      std::copy(data + begin, data + begin + width, values.begin());
    }
  }
  return FloatTensor({1, 1, width}, std::move(values));
}

float RoundToFloat16(float value) {
  return Ort::Float16_t(value).ToFloat();
}

void AddIntoOnnxTypedAccum(std::vector<float>& accum,
                           const std::vector<float>& values,
                           ONNXTensorElementDataType onnx_type) {
  // 参考 codec embedding 和 frame embedding 是多个 codebook embedding 的和。
  // FP16 模型下 Python/ONNX 的中间累加会有半精度舍入；这里显式模拟，减少差异。
  if (accum.size() != values.size()) throw std::invalid_argument("Accumulation shape mismatch");
  if (onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    for (size_t i = 0; i < accum.size(); ++i) {
      accum[i] = RoundToFloat16(accum[i] + RoundToFloat16(values[i]));
    }
    return;
  }
  if (onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    for (size_t i = 0; i < accum.size(); ++i) {
      accum[i] += values[i];
    }
    return;
  }
  throw std::runtime_error("Unsupported embedding output type for accumulation: " + std::to_string(onnx_type));
}

int64_t CopyInt64Scalar(const Ort::Value& value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || info.GetElementCount() < 1) {
    throw std::runtime_error("Expected non-empty int64 tensor");
  }
  const int64_t* data = value.GetTensorData<int64_t>();
  int64_t out = 0;
  if (IsCudaTensor(value)) {
    CudaRuntime::Instance().CopyDeviceToHost(&out, data, sizeof(out));
  } else {
    out = data[0];
  }
  return out;
}

template <typename T>
void WriteNpy(const std::filesystem::path& path, const std::vector<T>& values, const std::vector<int64_t>& shape) {
  if (path.empty()) return;
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Failed to open npy for writing: " + path.string());
  const char* descr = nullptr;
  if constexpr (std::is_same_v<T, float>) descr = "<f4";
  else if constexpr (std::is_same_v<T, int64_t>) descr = "<i8";
  else static_assert(sizeof(T) == 0, "Unsupported npy type");
  std::string shape_text = "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) shape_text += ", ";
    shape_text += std::to_string(shape[i]);
  }
  if (shape.size() == 1) shape_text += ",";
  shape_text += ")";
  std::string header = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': " + shape_text + ", }";
  const size_t padding = 16 - ((10 + header.size() + 1) % 16);
  header.append(padding, ' ');
  header.push_back('\n');
  out.write("\x93NUMPY", 6);
  char version[2] = {1, 0};
  out.write(version, 2);
  const uint16_t header_len = static_cast<uint16_t>(header.size());
  char len[2] = {static_cast<char>(header_len & 0xff), static_cast<char>((header_len >> 8) & 0xff)};
  out.write(len, 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void DumpChunkDebug(const std::filesystem::path& dump_dir,
                    size_t chunk_index,
                    const Int64Tensor& full_codes,
                    const FloatTensor& audio,
                    int64_t start_frame,
                    int64_t end_frame,
                    int64_t left_context_frames,
                    int64_t generated_frames,
                    bool is_final,
                    int64_t code_groups) {
  if (dump_dir.empty()) return;
  // chunk debug 文件记录 full/input/audio/meta，便于单独复现某一段 decoder 输入。
  const int64_t context = std::min<int64_t>(left_context_frames, start_frame);
  const int64_t input_start = start_frame - context;
  const int64_t input_frames = end_frame - input_start;
  Int64Tensor chunk_codes({input_frames, code_groups});
  std::copy(full_codes.values().begin() + input_start * code_groups,
            full_codes.values().begin() + end_frame * code_groups,
            chunk_codes.values().begin());
  const auto prefix = dump_dir / ("chunk_" + std::to_string(chunk_index));
  WriteNpy(prefix.string() + "_full_codes.npy", full_codes.values(), full_codes.shape());
  WriteNpy(prefix.string() + "_input_codes.npy", chunk_codes.values(), chunk_codes.shape());
  WriteNpy(prefix.string() + "_audio.npy", audio.values(), audio.shape());
  WriteNpy(prefix.string() + "_meta.npy",
           std::vector<int64_t>{start_frame, end_frame, context, input_start, generated_frames, is_final ? 1 : 0},
           {6});
}

}  // 匿名命名空间

VoiceCloneRuntime::VoiceCloneRuntime(RuntimeOptions options)
    : options_(std::move(options)),
      env_(ORT_LOGGING_LEVEL_WARNING, "qwen3_tts_onnx"),
      config_(LoadModelConfig(options_.model_dir)),
      sampler_(options_.seed) {
  // tokenizer 来自原始模型目录；ONNX 目录只保存推理子图，不含完整文本 tokenizer 逻辑。
  const auto tokenizer_start = Clock::now();
  tokenizer_ = std::make_unique<Qwen2BpeTokenizer>(options_.model_dir);
  AddTiming("init.load_tokenizer", ElapsedMs(tokenizer_start));

  auto load_session = [&](const std::string& name, const std::filesystem::path& path) {
    // 每个子模型独立 session，方便把轻量前处理和重型生成放到不同 provider。
    const auto start = Clock::now();
    OrtSession session(env_, SessionCfg(path, ProvidersForSession(options_, name), options_.cuda_device_id));
    AddTiming("session_load." + name, ElapsedMs(start));
    return session;
  };

  text_project_ = load_session("text_project", options_.onnx_root / "text_project" / "text_project.onnx");
  codec_embed_ = load_session("codec_embed", options_.onnx_root / "codec_embed" / "codec_embed.onnx");
  code_predictor_embed_ = load_session("code_predictor_embed", options_.onnx_root / "code_predictor_embed" / "code_predictor_embed.onnx");
  speaker_encoder_ = load_session("speaker_encoder", options_.onnx_root / "speaker_encoder" / "speaker_encoder.onnx");
  tokenizer_encode_ = load_session("tokenizer_encode", options_.onnx_root / "tokenizer12hz" / "tokenizer12hz_encode.onnx");
  tokenizer_decode_ = load_session("tokenizer_decode", options_.onnx_root / "tokenizer12hz" / "tokenizer12hz_decode.onnx");
  code_predictor_ = load_session("code_predictor", options_.onnx_root / "code_predictor" / "code_predictor.onnx");
  talker_prefill_ = load_session("talker_prefill", options_.onnx_root / "talker_prefill" / "talker_prefill.onnx");
  talker_decode_ = load_session("talker_decode", options_.onnx_root / "talker_decode" / "talker_decode.onnx");
}

void VoiceCloneRuntime::AddTiming(const std::string& name, double milliseconds) const {
  std::lock_guard<std::mutex> lock(timing_mutex_);
  auto it = std::find_if(timing_.begin(), timing_.end(), [&](const TimingRecord& record) {
    return record.name == name;
  });
  if (it == timing_.end()) {
    timing_.push_back(TimingRecord{name, 1, milliseconds});
    return;
  }
  it->count += 1;
  it->total_ms += milliseconds;
}

void VoiceCloneRuntime::PrintTimingSummary(std::ostream& os, const std::string& title) const {
  std::lock_guard<std::mutex> lock(timing_mutex_);
  if (timing_.empty()) return;
  const auto old_flags = os.flags();
  const auto old_precision = os.precision();
  os << "\n" << title << "\n";
  os << std::fixed << std::setprecision(2);
  for (const auto& record : timing_) {
    os << "  " << record.name << ": ";
    if (record.count == 1) {
      os << record.total_ms << " ms\n";
    } else {
      os << "total=" << record.total_ms << " ms, count=" << record.count
         << ", avg=" << (record.total_ms / static_cast<double>(record.count)) << " ms\n";
    }
  }
  os.flags(old_flags);
  os.precision(old_precision);
}

FloatTensor VoiceCloneRuntime::TextProject(const Int64Tensor& input_ids) const {
  // text_project.onnx: token ids -> talker hidden size 的文本 embedding。
  std::unordered_map<std::string, Ort::Value> inputs;
  inputs.emplace("input_ids", text_project_.MakeTensor(input_ids));
  const auto start = Clock::now();
  auto out = text_project_.RunFloat(inputs, "text_embed");
  AddTiming("onnx.text_project", ElapsedMs(start));
  return out;
}

FloatTensor VoiceCloneRuntime::CodecEmbed(const Int64Tensor& token_ids) const {
  // codec_embed.onnx: 第 0 个 codebook token -> hidden size embedding。
  std::unordered_map<std::string, Ort::Value> inputs;
  inputs.emplace("token_ids", codec_embed_.MakeTensor(token_ids));
  const auto start = Clock::now();
  auto out = codec_embed_.RunFloat(inputs, "embed");
  AddTiming("onnx.codec_embed", ElapsedMs(start));
  return out;
}

FloatTensor VoiceCloneRuntime::CodePredictorEmbed(const Int64Tensor& token_ids, int64_t layer_idx) const {
  int64_t scalar_token = 0;
  const bool cacheable = TryGetSingleToken(token_ids, &scalar_token);
  if (cacheable) {
    // 生成循环里 residual token 通常是 [1,1] 单 token 查询，适合缓存。
    const uint64_t key = CodePredictorEmbedCacheKey(scalar_token, layer_idx);
    auto it = code_predictor_embed_cache_.find(key);
    if (it != code_predictor_embed_cache_.end()) {
      AddTiming("cache.code_predictor_embed_hit", 0.0);
      return it->second;
    }
  }

  std::unordered_map<std::string, Ort::Value> inputs;
  auto idx = Scalar(layer_idx);
  inputs.emplace("token_id", code_predictor_embed_.MakeTensor(token_ids));
  inputs.emplace("layer_idx", code_predictor_embed_.MakeTensor(idx));
  const auto start = Clock::now();
  auto out = code_predictor_embed_.RunFloat(inputs, "embed");
  AddTiming("onnx.code_predictor_embed", ElapsedMs(start));
  if (cacheable) {
    const uint64_t key = CodePredictorEmbedCacheKey(scalar_token, layer_idx);
    auto [it, inserted] = code_predictor_embed_cache_.emplace(key, std::move(out));
    AddTiming("cache.code_predictor_embed_miss", 0.0);
    return it->second;
  }
  return out;
}

FloatTensor VoiceCloneRuntime::DecodeCodes(const Int64Tensor& codes, int64_t* output_length) const {
  // tokenizer_decode.onnx 期望 batch 维度：[1, frames, 16]。
  Int64Tensor batched({1, codes.shape()[0], codes.shape()[1]});
  batched.values() = codes.values();
  std::unordered_map<std::string, Ort::Value> inputs;
  inputs.emplace("audio_codes", tokenizer_decode_.MakeTensor(batched));
  const auto start = Clock::now();
  auto outputs =
      tokenizer_decode_.RunRawIoBinding(inputs, {"audio_values", "lengths"}, {"audio_values", "lengths"});
  AddTiming("onnx.tokenizer_decode", ElapsedMs(start));
  auto info = outputs[0].GetTensorTypeAndShapeInfo();
  auto audio = CopyFloatTensor(outputs[0]);
  int64_t len = static_cast<int64_t>(audio.size());
  if (outputs.size() > 1) {
    len = CopyInt64Scalar(outputs[1]);
  }
  if (output_length) *output_length = len;
  if (audio.shape().size() == 2 && audio.shape()[0] == 1) {
    audio.shape() = {audio.shape()[1]};
  }
  if (len >= 0 && static_cast<size_t>(len) < audio.values().size()) {
    // 部分导出图会返回 padding 后的 audio_values 和真实 lengths；这里按真实长度裁剪。
    audio.values().resize(static_cast<size_t>(len));
    audio.shape() = {static_cast<int64_t>(audio.values().size())};
  }
  return audio;
}

OrtSession& VoiceCloneRuntime::ChunkDecoder() {
  // chunk decoder 不是默认必需文件，只有流式/chunk 路径第一次调用时才加载。
  if (tokenizer_decode_chunk_) return *tokenizer_decode_chunk_;
  const auto path = options_.onnx_root / "tokenizer12hz" / "tokenizer12hz_decode_chunk.onnx";
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("Chunk decoder not found: " + path.string());
  }
  const auto start = Clock::now();
  tokenizer_decode_chunk_ = std::make_unique<OrtSession>(
      env_,
      SessionCfg(path, ProvidersForSession(options_, "tokenizer_decode_chunk"), options_.cuda_device_id));
  AddTiming("session_load.tokenizer_decode_chunk", ElapsedMs(start));
  return *tokenizer_decode_chunk_;
}

FloatTensor VoiceCloneRuntime::DecodeCodesChunk(const Int64Tensor& full_codes,
                                                int64_t start_frame,
                                                int64_t end_frame,
                                                int64_t left_context_frames,
                                                int64_t* output_length) {
  if (full_codes.shape().size() != 2 || full_codes.shape()[1] != config_.talker.num_code_groups) {
    throw std::invalid_argument("full_codes must have shape [frames, 16]");
  }
  if (end_frame <= start_frame) {
    if (output_length) *output_length = 0;
    return FloatTensor({0});
  }
  const int64_t context = std::min<int64_t>(std::max<int64_t>(left_context_frames, 0), start_frame);
  const int64_t input_start = start_frame - context;
  const int64_t input_frames = end_frame - input_start;
  const int64_t groups = config_.talker.num_code_groups;
  Int64Tensor codes_chunk({1, input_frames, groups});
  // 输入包含左上下文，但 decoder 输出只保留 [start_frame,end_frame) 的新音频。
  const auto src_begin = full_codes.values().begin() + static_cast<long>(input_start * groups);
  const auto src_end = full_codes.values().begin() + static_cast<long>(end_frame * groups);
  std::copy(src_begin, src_end, codes_chunk.values().begin());

  auto& decoder = ChunkDecoder();
  std::unordered_map<std::string, Ort::Value> inputs;
  auto context_tensor = Scalar(context);
  inputs.emplace("audio_codes", decoder.MakeTensor(codes_chunk));
  inputs.emplace("context_frames", decoder.MakeTensor(context_tensor));
  const auto decode_start = Clock::now();
  auto outputs =
      decoder.RunRawIoBinding(inputs, {"audio_values", "lengths"}, {"audio_values", "lengths"});
  AddTiming("onnx.tokenizer_decode_chunk", ElapsedMs(decode_start));

  auto audio = CopyFloatTensor(outputs[0]);
  int64_t reported = static_cast<int64_t>(audio.size());
  if (outputs.size() > 1) {
    reported = CopyInt64Scalar(outputs[1]);
  }
  const int64_t expected = (end_frame - start_frame) * config_.codec_frame_samples;
  // Some ORT/CUDA builds have returned an invalid scalar for this auxiliary
  // lengths output while audio_values itself is correct. Treat audio_values as
  // authoritative for chunk slicing, matching the Python runtime's expected
  // sample count check.
  if (reported > 0 && reported < expected) {
    throw std::runtime_error("tokenizer_decode_chunk reported fewer samples than expected: reported=" +
                             std::to_string(reported) + ", expected=" + std::to_string(expected));
  }
  if (static_cast<int64_t>(audio.values().size()) < expected) {
    throw std::runtime_error("tokenizer_decode_chunk output is shorter than expected: audio_samples=" +
                             std::to_string(audio.values().size()) + ", expected=" + std::to_string(expected));
  }
  audio.values().resize(static_cast<size_t>(expected));
  audio.shape() = {expected};
  if (output_length) *output_length = expected;
  return audio;
}

Int64Tensor VoiceCloneRuntime::EncodeReferenceCodes(const FloatTensor& audio) const {
  // tokenizer12hz_encode.onnx: 24k waveform -> [T,16] RVQ codec codes。
  std::unordered_map<std::string, Ort::Value> inputs;
  inputs.emplace("audio", tokenizer_encode_.MakeTensor(audio, "audio"));
  const auto start = Clock::now();
  auto codes = tokenizer_encode_.RunInt64(inputs, "codes");
  AddTiming("onnx.tokenizer_encode", ElapsedMs(start));
  if (codes.shape().size() == 3 && codes.shape()[0] == 1) {
    Int64Tensor squeezed({codes.shape()[1], codes.shape()[2]});
    squeezed.values() = std::move(codes.values());
    return squeezed;
  }
  return codes;
}

FloatTensor VoiceCloneRuntime::ExtractSpeakerEmbedding(const FloatTensor& mel) const {
  // speaker_encoder.onnx 输出 reshape 成 [1,1,hidden]，便于直接拼进 codec prompt。
  std::unordered_map<std::string, Ort::Value> inputs;
  inputs.emplace("mel", speaker_encoder_.MakeTensor(mel, "mel"));
  const auto start = Clock::now();
  auto out = speaker_encoder_.RunFloat(inputs, "speaker_embedding");
  AddTiming("onnx.speaker_encoder", ElapsedMs(start));
  if (out.shape().size() == 2) out.shape() = {out.shape()[0], 1, out.shape()[1]};
  return out;
}

std::vector<int64_t> VoiceCloneRuntime::EncodeAssistantText(const std::string& text) const {
  // 目标文本使用 assistant 对话模板，并在末尾再开一个 assistant turn，
  // 与 Python Qwen3-TTS wrapper 的 prompt 格式对齐。
  return tokenizer_->Encode("<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n");
}

std::vector<int64_t> VoiceCloneRuntime::EncodeReferenceText(const std::string& text) const {
  // 参考文本只需要一个完整 assistant turn，用于 ICL 参考段。
  return tokenizer_->Encode("<|im_start|>assistant\n" + text + "<|im_end|>\n");
}

ReferenceAudioFeatures VoiceCloneRuntime::GetReferenceAudioFeatures(const std::filesystem::path& audio_path,
                                                                     bool x_vector_only_mode) {
  // x_vector_only_mode=true 时只使用 speaker embedding，不把参考 codec 放入 ICL。
  const auto cache_key = ReferenceAudioCacheKey(audio_path, x_vector_only_mode);
  auto it = reference_audio_cache_.find(cache_key);
  if (it != reference_audio_cache_.end()) {
    AddTiming("prep.reference_audio_cache_hit", 0.0);
    return it->second;
  }

  const auto cache_start = Clock::now();
  AddTiming("prep.reference_audio_cache_miss", 0.0);
  const auto audio_start = Clock::now();
  auto audio = LoadAudioMono(audio_path, 24000);
  FloatTensor audio_tensor({1, static_cast<int64_t>(audio.samples.size())}, audio.samples);
  AddTiming("prep.load_audio", ElapsedMs(audio_start));

  const auto mel_start = Clock::now();
  // speaker encoder 接收 mel，不直接接 waveform。
  auto mel = MelSpectrogram(audio.samples);
  AddTiming("prep.mel_spectrogram", ElapsedMs(mel_start));

  ReferenceAudioFeatures features;
  if (!x_vector_only_mode) {
    const auto ref_codes_start = Clock::now();
    features.reference_codes = EncodeReferenceCodes(audio_tensor);
    AddTiming("prep.encode_ref_codes", ElapsedMs(ref_codes_start));
  }

  const auto speaker_start = Clock::now();
  features.speaker_embedding = ExtractSpeakerEmbedding(mel);
  AddTiming("prep.extract_speaker_embedding", ElapsedMs(speaker_start));
  AddTiming("prep.reference_audio_cache_build", ElapsedMs(cache_start));

  auto [inserted, _] = reference_audio_cache_.emplace(cache_key, std::move(features));
  return inserted->second;
}

std::vector<int64_t> VoiceCloneRuntime::LanguagePrefillIds(const std::string& language) const {
  // codec 侧的“思考/语言”控制 token：
  // auto = nothink + think_bos + think_eos；指定语言时插入 codec_language_id。
  std::string lang = language;
  std::transform(lang.begin(), lang.end(), lang.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lang.empty() || lang == "auto") {
    return {config_.talker.codec_nothink_id, config_.talker.codec_think_bos_id, config_.talker.codec_think_eos_id};
  }
  auto it = config_.talker.codec_language_id.find(lang);
  if (it == config_.talker.codec_language_id.end()) throw std::runtime_error("Unsupported language: " + language);
  return {config_.talker.codec_think_id, config_.talker.codec_think_bos_id, it->second, config_.talker.codec_think_eos_id};
}

FloatTensor VoiceCloneRuntime::ReferenceCodeEmbedding(const Int64Tensor& ref_codes) const {
  const auto timing_start = Clock::now();
  if (ref_codes.shape().size() != 2 || ref_codes.shape()[1] != config_.talker.num_code_groups) {
    throw std::invalid_argument("reference_codes must have shape [ref_len, 16]");
  }
  const int64_t frames = ref_codes.shape()[0];
  FloatTensor summed;
  for (int64_t group = 0; group < config_.talker.num_code_groups; ++group) {
    // 一帧 16 个 codebook：第 0 路走 codec_embed，其余 15 路走 code_predictor_embed。
    // 16 路 embedding 求和后表示这一帧 codec。
    Int64Tensor ids({1, frames});
    for (int64_t t = 0; t < frames; ++t) ids.values()[t] = ref_codes.values()[t * config_.talker.num_code_groups + group];
    auto piece = group == 0 ? CodecEmbed(ids) : CodePredictorEmbed(ids, group - 1);
    if (group == 0) summed = FloatTensor(piece.shape());
    const auto output_type = group == 0 ? codec_embed_.OutputType("embed") : code_predictor_embed_.OutputType("embed");
    AddIntoOnnxTypedAccum(summed.values(), piece.values(), output_type);
  }
  auto out = ConcatAxis1({CodecEmbed(RowTensor({config_.talker.codec_bos_id})), summed});
  AddTiming("prep.reference_code_embedding", ElapsedMs(timing_start));
  return out;
}

FloatTensor VoiceCloneRuntime::BuildTalkerPrompt(const VoiceCloneInputs& inputs, FloatTensor* trailing_text, FloatTensor* tts_pad) const {
  const auto timing_start = Clock::now();
  if (inputs.assistant_text_ids.size() < 8) throw std::invalid_argument("assistant_text_ids is too short");
  if (inputs.speaker_embedding.shape() != std::vector<int64_t>({1, 1, config_.talker.hidden_size})) {
    throw std::invalid_argument("speaker_embedding must be [1,1,hidden_size]");
  }
  auto special = TextProject(RowTensor({config_.tts_bos_token_id, config_.tts_eos_token_id, config_.tts_pad_token_id}));
  // 文本侧 special token embedding，后面用于和 codec embedding 对齐相加。
  auto tts_bos = SliceAxis1(special, 0, 1);
  auto tts_eos = SliceAxis1(special, 1, 2);
  *tts_pad = SliceAxis1(special, 2, 3);

  auto codec_prefill = CodecEmbed(RowTensor(LanguagePrefillIds(inputs.language)));
  auto codec_tail = CodecEmbed(RowTensor({config_.talker.codec_pad_id, config_.talker.codec_bos_id}));
  // codec_input = 语言控制 + speaker embedding + codec_pad/codec_bos。
  // speaker_embedding 放在 codec 流里，是声音克隆音色条件的核心。
  auto codec_input = ConcatAxis1({codec_prefill, inputs.speaker_embedding, codec_tail});

  auto role = TextProject(Int64Tensor({1, 3}, {inputs.assistant_text_ids[0], inputs.assistant_text_ids[1], inputs.assistant_text_ids[2]}));
  auto left_pad = RepeatAxis1(*tts_pad, codec_input.shape()[1] - 2);
  // codec_part 把文本 pad/BOS embedding 与 codec 控制 embedding 相加，形成同一条 hidden 序列。
  auto codec_part = Add(ConcatAxis1({left_pad, tts_bos}), SliceAxis1(codec_input, 0, codec_input.shape()[1] - 1));
  auto talker_input = ConcatAxis1({role, codec_part});

  if (!inputs.reference_codes.empty() && !inputs.reference_text_ids.empty()) {
    // ICL 模式：参考文本 + 目标文本前缀，与参考 codec embedding 按时间步相加。
    // 如果文本比参考 codec 更长，剩下的文本 embedding 放到 trailing_text，
    // 后续每生成一帧 codec 消耗一个 trailing token。
    std::vector<int64_t> ref_text(inputs.reference_text_ids.begin() + 3, inputs.reference_text_ids.end() - 2);
    std::vector<int64_t> target_text(inputs.assistant_text_ids.begin() + 3, inputs.assistant_text_ids.end() - 5);
    ref_text.insert(ref_text.end(), target_text.begin(), target_text.end());
    auto text_embed = ConcatAxis1({TextProject(RowTensor(ref_text)), tts_eos});
    auto ref_codec_embed = ReferenceCodeEmbedding(inputs.reference_codes);
    if (text_embed.shape()[1] > ref_codec_embed.shape()[1]) {
      auto icl = Add(SliceAxis1(text_embed, 0, ref_codec_embed.shape()[1]), ref_codec_embed);
      *trailing_text = SliceAxis1(text_embed, ref_codec_embed.shape()[1], text_embed.shape()[1]);
      auto out = ConcatAxis1({talker_input, icl});
      AddTiming("prep.build_talker_prompt", ElapsedMs(timing_start));
      return out;
    }
    auto padded_text = text_embed;
    if (text_embed.shape()[1] < ref_codec_embed.shape()[1]) {
      padded_text = ConcatAxis1({text_embed, RepeatAxis1(*tts_pad, ref_codec_embed.shape()[1] - text_embed.shape()[1])});
    }
    *trailing_text = *tts_pad;
    auto out = ConcatAxis1({talker_input, Add(padded_text, ref_codec_embed)});
    AddTiming("prep.build_talker_prompt", ElapsedMs(timing_start));
    return out;
  }

  auto first_text = Add(TextProject(RowTensor({inputs.assistant_text_ids[3]})), SliceAxis1(codec_input, codec_input.shape()[1] - 1, codec_input.shape()[1]));
  // 没有参考 codec 时，先把目标文本第一个 token 和 codec_bos 条件相加；
  // 剩余目标文本进入 trailing_text，生成循环里逐帧喂给 talker。
  std::vector<int64_t> trailing_ids(inputs.assistant_text_ids.begin() + 4, inputs.assistant_text_ids.end() - 5);
  *trailing_text = ConcatAxis1({TextProject(RowTensor(trailing_ids)), tts_eos});
  auto out = ConcatAxis1({talker_input, first_text});
  AddTiming("prep.build_talker_prompt", ElapsedMs(timing_start));
  return out;
}

std::pair<std::vector<int64_t>, FloatTensor> VoiceCloneRuntime::RunCodePredictor(const FloatTensor& past_hidden,
                                                                                  int64_t first_token,
                                                                                  const SamplingOptions& options,
                                                                                  int frame_index,
                                                                                  const std::filesystem::path& debug_dump_dir) {
  const auto frame_start = Clock::now();
  std::vector<int64_t> tokens{first_token};
  auto main_embed = CodecEmbed(RowTensor({first_token}));
  const int64_t hidden_dim = main_embed.shape()[2];
  const auto context_init_start = Clock::now();
  std::vector<float> context_values(static_cast<size_t>((config_.talker.num_code_groups + 1) * hidden_dim));
  // context 初始为 [talker_last_hidden, first_codebook_embedding]。
  // 后续每预测一个 residual token，就把它的 embedding 追加到 context。
  std::copy(past_hidden.values().begin(), past_hidden.values().end(), context_values.begin());
  std::copy(main_embed.values().begin(), main_embed.values().end(),
            context_values.begin() + static_cast<long>(hidden_dim));
  int64_t context_len = 2;
  std::vector<float> frame_sum(static_cast<size_t>(hidden_dim), 0.0f);
  AddIntoOnnxTypedAccum(frame_sum, main_embed.values(), codec_embed_.OutputType("embed"));
  AddTiming("generation.code_predictor.context_init", ElapsedMs(context_init_start));
  for (int64_t step = 0; step < config_.talker.num_code_groups - 1; ++step) {
    const auto feed_start = Clock::now();
    std::unordered_map<std::string, Ort::Value> inputs;
    auto step_t = Scalar(step);
    std::vector<int64_t> context_shape{1, context_len, hidden_dim};
    inputs.emplace("context", code_predictor_.MakeTensorFromData(
                                  context_values.data(),
                                  static_cast<size_t>(context_len * hidden_dim),
                                  context_shape,
                                  "context"));
    inputs.emplace("gen_step", code_predictor_.MakeTensor(step_t));
    AddTiming("generation.code_predictor.prepare_feed", ElapsedMs(feed_start));
    const auto onnx_start = Clock::now();
    auto outputs = code_predictor_.RunRawIoBinding(inputs, {"logits"}, {"logits"});
    auto last_logits = LastLogits(outputs[0]);
    AddTiming("onnx.code_predictor", ElapsedMs(onnx_start));
    const auto sample_start = Clock::now();
    int64_t token = sampler_.Sample(last_logits, options);
    if (!debug_dump_dir.empty()) {
      int64_t forced = 0;
      const auto forced_path = debug_dump_dir / ("code_predictor_pick_f" + std::to_string(frame_index) + "_s" + std::to_string(step) + ".npy");
      if (ReadNpyInt64Scalar(forced_path, &forced)) {
        token = forced;
      }
    }
    AddTiming("generation.code_predictor.sample", ElapsedMs(sample_start));
    if (!debug_dump_dir.empty() && frame_index < 8) {
      WriteNpy(debug_dump_dir / ("code_predictor_logits_f" + std::to_string(frame_index) + "_s" + std::to_string(step) + ".npy"),
               last_logits, {static_cast<int64_t>(last_logits.size())});
      std::vector<float> context_dump(context_values.begin(),
                                      context_values.begin() + static_cast<long>(context_len * hidden_dim));
      WriteNpy(debug_dump_dir / ("code_predictor_context_f" + std::to_string(frame_index) + "_s" + std::to_string(step) + ".npy"),
               context_dump, context_shape);
    }
    if (!debug_dump_dir.empty()) {
      WriteNpy(debug_dump_dir / ("code_predictor_pick_f" + std::to_string(frame_index) + "_s" + std::to_string(step) + ".npy"),
               std::vector<int64_t>{token}, {1});
    }
    tokens.push_back(token);
    const auto embed_start = Clock::now();
    auto emb = CodePredictorEmbed(RowTensor({token}), step);
    AddTiming("generation.code_predictor.residual_embed", ElapsedMs(embed_start));
    const auto concat_start = Clock::now();
    std::copy(emb.values().begin(), emb.values().end(),
              context_values.begin() + static_cast<long>(context_len * hidden_dim));
    // frame_sum 累加 16 路 codebook embedding，作为下一次 talker_decode 的 codec 条件。
    AddIntoOnnxTypedAccum(frame_sum, emb.values(), code_predictor_embed_.OutputType("embed"));
    ++context_len;
    AddTiming("generation.code_predictor.context_append", ElapsedMs(concat_start));
  }
  const auto frame_embed_start = Clock::now();
  FloatTensor frame_embed({1, 1, hidden_dim}, std::move(frame_sum));
  AddTiming("generation.code_predictor.frame_embed_sum", ElapsedMs(frame_embed_start));
  AddTiming("generation.code_predictor_frame", ElapsedMs(frame_start));
  return {tokens, frame_embed};
}

VoiceCloneResult VoiceCloneRuntime::GenerateFromPrepared(const VoiceCloneInputs& inputs) {
  const auto total_start = Clock::now();
  const bool trace_tokens = std::getenv("QWEN_TRACE_TOKENS") != nullptr;
  FloatTensor trailing_text;
  FloatTensor tts_pad;
  // prompt 是 prefill 阶段一次性喂给 talker 的上下文；
  // trailing_text 是生成循环中逐帧追加的目标文本 embedding。
  auto prompt = BuildTalkerPrompt(inputs, &trailing_text, &tts_pad);

  auto attention_mask = Int64Tensor({1, prompt.shape()[1]}, std::vector<int64_t>(static_cast<size_t>(prompt.shape()[1]), 1));
  std::unordered_map<std::string, Ort::Value> prefill_kv_inputs;
  prefill_kv_inputs.emplace("inputs_embeds", talker_prefill_.MakeTensor(prompt, "inputs_embeds"));
  prefill_kv_inputs.emplace("attention_mask", talker_prefill_.MakeTensor(attention_mask));
  std::vector<std::string> prefill_output_names{"logits", "last_hidden"};
  for (int64_t layer = 0; layer < config_.talker.num_hidden_layers; ++layer) {
    // prefill 一次性建立整段 prompt 的 KV cache，后续 decode 每步只喂 1 个 embedding。
    prefill_output_names.push_back("past_key_" + std::to_string(layer));
    prefill_output_names.push_back("past_value_" + std::to_string(layer));
  }
  std::unordered_set<std::string> prefill_device_output_names;
  prefill_device_output_names.reserve(static_cast<size_t>(config_.talker.num_hidden_layers * 2));
  for (int64_t layer = 0; layer < config_.talker.num_hidden_layers; ++layer) {
    // KV cache 很大，CUDA 路径下让它留在 device memory，下一步直接复用。
    prefill_device_output_names.insert("past_key_" + std::to_string(layer));
    prefill_device_output_names.insert("past_value_" + std::to_string(layer));
  }
  const auto prefill_start = Clock::now();
  auto prefill_outputs =
      talker_prefill_.RunRawIoBinding(prefill_kv_inputs, prefill_output_names, prefill_device_output_names);
  AddTiming("onnx.talker_prefill", ElapsedMs(prefill_start));
  std::vector<float> next_logits = LastLogits(prefill_outputs[0]);
  auto last_hidden = LastHidden(prefill_outputs[1]);
  if (!inputs.debug_dump_dir.empty()) {
    WriteNpy(inputs.debug_dump_dir / "prompt.npy", prompt.values(), prompt.shape());
    WriteNpy(inputs.debug_dump_dir / "prefill_logits_last.npy", next_logits, {config_.talker.vocab_size});
    auto last_hidden_full = CopyFloatTensor(prefill_outputs[1]);
    WriteNpy(inputs.debug_dump_dir / "prefill_last_hidden_full.npy", last_hidden_full.values(), last_hidden_full.shape());
  }
  std::vector<Ort::Value> past;
  past.reserve(static_cast<size_t>(config_.talker.num_hidden_layers * 2));
  for (size_t i = 2; i < prefill_outputs.size(); ++i) {
    past.emplace_back(std::move(prefill_outputs[i]));
  }
  if (!inputs.debug_dump_dir.empty() && !past.empty()) {
    auto k0 = CopyFloatTensor(past[0]);
    auto v0 = CopyFloatTensor(past[1]);
    WriteNpy(inputs.debug_dump_dir / "prefill_past_key_0.npy", k0.values(), k0.shape());
    WriteNpy(inputs.debug_dump_dir / "prefill_past_value_0.npy", v0.values(), v0.shape());
  }

  std::vector<int64_t> first_tokens;
  std::vector<int64_t> generated_flat;
  int64_t past_len = prompt.shape()[1];
  const int max_frames = std::max(inputs.max_new_tokens - 1, 1);
  std::vector<std::string> decode_output_names{"logits", "last_hidden"};
  decode_output_names.reserve(static_cast<size_t>(2 + config_.talker.num_hidden_layers * 2));
  std::unordered_set<std::string> decode_device_output_names{"logits", "last_hidden"};
  decode_device_output_names.reserve(static_cast<size_t>(config_.talker.num_hidden_layers * 2));
  for (int64_t layer = 0; layer < config_.talker.num_hidden_layers; ++layer) {
    // decode 子图的输出名使用 new_past_key/value，避免和输入 past_key/value 混淆。
    auto key_name = "new_past_key_" + std::to_string(layer);
    auto value_name = "new_past_value_" + std::to_string(layer);
    decode_device_output_names.insert(key_name);
    decode_device_output_names.insert(value_name);
    decode_output_names.push_back(std::move(key_name));
    decode_output_names.push_back(std::move(value_name));
  }

  Int64Tensor decode_attention_mask({1, past_len + 2},
                                    std::vector<int64_t>(static_cast<size_t>(past_len + 2), 1));
  decode_attention_mask.values().reserve(static_cast<size_t>(prompt.shape()[1] + max_frames + 2));
  Int64Tensor decode_cache_position({1}, {past_len});
  if (!inputs.debug_dump_dir.empty()) {
    WriteNpy(inputs.debug_dump_dir / "sampling_flags.npy",
             std::vector<int64_t>{inputs.main_sampling.do_sample ? 1 : 0,
                                  inputs.code_sampling.do_sample ? 1 : 0},
             {2});
  }

  const auto loop_start = Clock::now();
  for (int frame = 0; frame < max_frames; ++frame) {
    const auto frame_start = Clock::now();
    const auto logits_filter_start = Clock::now();
    auto filtered_logits = next_logits;
    // 对齐官方 Qwen3-TTS generate()：屏蔽 talker vocab 尾部的一段控制/保留区间，
    // 但保留 codec_eos_token_id 作为停止条件。
    // 这里把它们屏蔽掉，只允许采样合法的第 0 codebook token 和 codec_eos。
    const int64_t mask_begin = std::max<int64_t>(0, config_.talker.vocab_size - config_.talker.first_codebook_mask_tail);
    for (int64_t id = mask_begin; id < config_.talker.vocab_size; ++id) {
      if (id != config_.talker.codec_eos_token_id) filtered_logits[static_cast<size_t>(id)] = -std::numeric_limits<float>::infinity();
    }
    AddTiming("generation.last_logits_filter", ElapsedMs(logits_filter_start));
    const auto sample_start = Clock::now();
    int64_t first = sampler_.Sample(filtered_logits, inputs.main_sampling, first_tokens);
    if (!inputs.debug_dump_dir.empty()) {
      int64_t forced = 0;
      const auto forced_path = inputs.debug_dump_dir / ("first_token_pick_f" + std::to_string(frame) + ".npy");
      if (ReadNpyInt64Scalar(forced_path, &forced)) {
        first = forced;
      }
    }
    AddTiming("generation.sample_first_token", ElapsedMs(sample_start));
    if (trace_tokens) {
      const float eos_logit = next_logits[static_cast<size_t>(config_.talker.codec_eos_token_id)];
      std::cerr << "[trace] frame=" << frame
                << " first=" << first
                << " eos_id=" << config_.talker.codec_eos_token_id
                << " eos_logit=" << eos_logit
                << (first == config_.talker.codec_eos_token_id ? " hit_eos=1" : " hit_eos=0")
                << "\n";
    }
    if (!inputs.debug_dump_dir.empty()) {
      WriteNpy(inputs.debug_dump_dir / ("first_token_logits_f" + std::to_string(frame) + ".npy"),
               filtered_logits, {static_cast<int64_t>(filtered_logits.size())});
      WriteNpy(inputs.debug_dump_dir / ("first_token_pick_f" + std::to_string(frame) + ".npy"),
               std::vector<int64_t>{first}, {1});
    }
    if (first == config_.talker.codec_eos_token_id) {
      // EOS 表示 codec 序列结束，不再进入 code_predictor。
      break;
    }
    first_tokens.push_back(first);
    // talker 只给出每帧第 0 个 codebook token；剩余 15 个由 code_predictor 补齐。
    auto [row, frame_embed] = RunCodePredictor(last_hidden, first, inputs.code_sampling, frame, inputs.debug_dump_dir);
    const auto append_codes_start = Clock::now();
    generated_flat.insert(generated_flat.end(), row.begin(), row.end());
    AddTiming("generation.append_generated_codes", ElapsedMs(append_codes_start));

    const auto decode_embed_start = Clock::now();
    FloatTensor decode_embed = frame < trailing_text.shape()[1]
                                    ? Add(frame_embed, SliceAxis1(trailing_text, frame, frame + 1))
                                    : Add(frame_embed, tts_pad);
    // decode_embed 是“上一帧 codec embedding + 当前文本 embedding”。
    // 文本耗尽后用 tts_pad，让模型继续依靠 codec 自回归直到 EOS。
    AddTiming("generation.build_decode_embed", ElapsedMs(decode_embed_start));
    if (!inputs.debug_dump_dir.empty() && frame == 0) {
      WriteNpy(inputs.debug_dump_dir / "decode0_inputs_embeds.npy", decode_embed.values(), decode_embed.shape());
      WriteNpy(inputs.debug_dump_dir / "decode0_attention_mask.npy",
               std::vector<int64_t>(static_cast<size_t>(past_len + 2), 1), {1, past_len + 2});
      WriteNpy(inputs.debug_dump_dir / "decode0_cache_position.npy", std::vector<int64_t>{past_len}, {1});
      auto k0 = CopyFloatTensor(past[0]);
      auto v0 = CopyFloatTensor(past[1]);
      WriteNpy(inputs.debug_dump_dir / "decode0_past_key_0.npy", k0.values(), k0.shape());
      WriteNpy(inputs.debug_dump_dir / "decode0_past_value_0.npy", v0.values(), v0.shape());
    }
    const auto feed_start = Clock::now();
    std::unordered_map<std::string, Ort::Value> feeds;
    feeds.reserve(static_cast<size_t>(3 + config_.talker.num_hidden_layers * 2));
    if (decode_attention_mask.shape()[1] < past_len + 2) {
      // attention_mask 长度随 KV cache 增长。这里复用 vector，只追加一个 1。
      decode_attention_mask.values().push_back(1);
      decode_attention_mask.shape()[1] = past_len + 2;
    }
    decode_cache_position.values()[0] = past_len;
    feeds.emplace("inputs_embeds", talker_decode_.MakeTensor(decode_embed, "inputs_embeds"));
    feeds.emplace("attention_mask", talker_decode_.MakeTensor(decode_attention_mask));
    feeds.emplace("cache_position", talker_decode_.MakeTensor(decode_cache_position));
    for (int64_t layer = 0; layer < config_.talker.num_hidden_layers; ++layer) {
      // past 被 move 进 ORT 输入，Run 之后会用输出 new_past 重新填充。
      feeds.emplace("past_key_" + std::to_string(layer), std::move(past[static_cast<size_t>(2 * layer)]));
      feeds.emplace("past_value_" + std::to_string(layer), std::move(past[static_cast<size_t>(2 * layer + 1)]));
    }
    AddTiming("generation.prepare_decode_feed", ElapsedMs(feed_start));
    const auto decode_start = Clock::now();
    auto out = talker_decode_.RunRawIoBinding(feeds, decode_output_names, decode_device_output_names);
    AddTiming("onnx.talker_decode", ElapsedMs(decode_start));
    next_logits = LastLogits(out[0]);
    last_hidden = LastHidden(out[1]);
    if (!inputs.debug_dump_dir.empty() && frame == 0) {
      WriteNpy(inputs.debug_dump_dir / "decode0_logits_last.npy", next_logits, {config_.talker.vocab_size});
      auto hidden_full = CopyFloatTensor(out[1]);
      WriteNpy(inputs.debug_dump_dir / "decode0_last_hidden.npy", hidden_full.values(), hidden_full.shape());
    }
    past.clear();
    past.reserve(static_cast<size_t>(config_.talker.num_hidden_layers * 2));
    for (size_t i = 2; i < out.size(); ++i) {
      past.emplace_back(std::move(out[i]));
    }
    ++past_len;
    AddTiming("generation.frame_total", ElapsedMs(frame_start));
  }
  AddTiming("generation.decode_loop_total", ElapsedMs(loop_start));

  VoiceCloneResult result;
  const auto stack_start = Clock::now();
  // generated_flat 是 [frame0 16 tokens][frame1 16 tokens]...，这里 reshape 成 [T,16]。
  const int64_t generated_frames = static_cast<int64_t>(generated_flat.size() / static_cast<size_t>(config_.talker.num_code_groups));
  result.generated_codes = Int64Tensor({generated_frames, config_.talker.num_code_groups}, std::move(generated_flat));
  AddTiming("post.stack_codes", ElapsedMs(stack_start));
  if (!result.generated_codes.empty()) {
    const auto post_start = Clock::now();
    Int64Tensor decode_codes = result.generated_codes;
    int64_t ref_frames = 0;
    if (!inputs.reference_codes.empty()) {
      // ICL 模式下完整 decoder 需要 reference + generated 一起解码，声音衔接更自然。
      ref_frames = inputs.reference_codes.shape()[0];
      decode_codes = Int64Tensor({ref_frames + result.generated_codes.shape()[0], config_.talker.num_code_groups});
      std::copy(inputs.reference_codes.values().begin(), inputs.reference_codes.values().end(), decode_codes.values().begin());
      std::copy(result.generated_codes.values().begin(), result.generated_codes.values().end(),
                decode_codes.values().begin() + static_cast<long>(inputs.reference_codes.size()));
    }
    AddTiming("post.prepare_vocoder_codes", ElapsedMs(post_start));
    const auto vocoder_start = Clock::now();
    FloatTensor audio = DecodeCodes(decode_codes);
    AddTiming("post.decode_codes_to_audio", ElapsedMs(vocoder_start));
    const auto trim_start = Clock::now();
    // Full decode includes reference audio and must be trimmed.
    if (ref_frames > 0 && !audio.empty()) {
      // tokenizer_decode 的输出可能受 trace/padding 影响，不直接假设每帧固定采样数；
      // 按参考帧数占比裁掉前缀，和 Python runtime 保持一致。
      const int64_t cut = static_cast<int64_t>(static_cast<double>(ref_frames) /
                                               std::max<int64_t>(decode_codes.shape()[0], 1) * audio.shape()[0]);
      if (cut > 0 && cut < audio.shape()[0]) {
        audio.values().erase(audio.values().begin(), audio.values().begin() + cut);
        audio.shape() = {static_cast<int64_t>(audio.values().size())};
      }
    }
    AddTiming("post.trim_reference_audio", ElapsedMs(trim_start));
    result.waveform = std::move(audio);
  }
  AddTiming("total.generate_from_prepared", ElapsedMs(total_start));
  return result;
}

VoiceCloneChunkedResult VoiceCloneRuntime::GenerateFromPreparedChunked(
    const VoiceCloneInputs& inputs,
    const VoiceCloneChunkOptions& chunk_options,
    const VoiceCloneChunkCallback& on_chunk) {
  // chunked 路径复用和 GenerateFromPrepared() 相同的生成逻辑，只是每攒够
  // chunk_frames 帧就立刻调用 tokenizer_decode_chunk 并回调一段音频。
  if (chunk_options.chunk_frames <= 0) throw std::invalid_argument("chunk_frames must be positive");
  if (chunk_options.left_context_frames < 0) throw std::invalid_argument("left_context_frames must be non-negative");
  if (chunk_options.async_chunk_decode && chunk_options.decode_workers != 1) {
    throw std::invalid_argument("async chunk decode currently supports exactly one decoder worker");
  }
  if (chunk_options.max_decode_queue <= 0) throw std::invalid_argument("max_decode_queue must be positive");
  (void)ChunkDecoder();  // fail early if the optional chunk decoder is missing.

  const auto total_start = Clock::now();
  const bool trace_tokens = std::getenv("QWEN_TRACE_TOKENS") != nullptr;
  FloatTensor trailing_text;
  FloatTensor tts_pad;
  auto prompt = BuildTalkerPrompt(inputs, &trailing_text, &tts_pad);

  auto attention_mask = Int64Tensor({1, prompt.shape()[1]}, std::vector<int64_t>(static_cast<size_t>(prompt.shape()[1]), 1));
  std::unordered_map<std::string, Ort::Value> prefill_kv_inputs;
  prefill_kv_inputs.emplace("inputs_embeds", talker_prefill_.MakeTensor(prompt, "inputs_embeds"));
  prefill_kv_inputs.emplace("attention_mask", talker_prefill_.MakeTensor(attention_mask));
  std::vector<std::string> prefill_output_names{"logits", "last_hidden"};
  for (int64_t layer = 0; layer < config_.talker.num_hidden_layers; ++layer) {
    prefill_output_names.push_back("past_key_" + std::to_string(layer));
    prefill_output_names.push_back("past_value_" + std::to_string(layer));
  }
  std::unordered_set<std::string> prefill_device_output_names;
  prefill_device_output_names.reserve(static_cast<size_t>(config_.talker.num_hidden_layers * 2));
  for (int64_t layer = 0; layer < config_.talker.num_hidden_layers; ++layer) {
    prefill_device_output_names.insert("past_key_" + std::to_string(layer));
    prefill_device_output_names.insert("past_value_" + std::to_string(layer));
  }
  const auto prefill_start = Clock::now();
  auto prefill_outputs =
      talker_prefill_.RunRawIoBinding(prefill_kv_inputs, prefill_output_names, prefill_device_output_names);
  AddTiming("onnx.talker_prefill", ElapsedMs(prefill_start));

  std::vector<float> next_logits = LastLogits(prefill_outputs[0]);
  auto last_hidden = LastHidden(prefill_outputs[1]);
  if (!inputs.debug_dump_dir.empty()) {
    WriteNpy(inputs.debug_dump_dir / "prompt.npy", prompt.values(), prompt.shape());
    WriteNpy(inputs.debug_dump_dir / "prefill_logits_last.npy", next_logits, {config_.talker.vocab_size});
    auto last_hidden_full = CopyFloatTensor(prefill_outputs[1]);
    WriteNpy(inputs.debug_dump_dir / "prefill_last_hidden_full.npy", last_hidden_full.values(), last_hidden_full.shape());
  }
  std::vector<Ort::Value> past;
  past.reserve(static_cast<size_t>(config_.talker.num_hidden_layers * 2));
  for (size_t i = 2; i < prefill_outputs.size(); ++i) past.emplace_back(std::move(prefill_outputs[i]));
  if (!inputs.debug_dump_dir.empty() && !past.empty()) {
    auto k0 = CopyFloatTensor(past[0]);
    auto v0 = CopyFloatTensor(past[1]);
    WriteNpy(inputs.debug_dump_dir / "prefill_past_key_0.npy", k0.values(), k0.shape());
    WriteNpy(inputs.debug_dump_dir / "prefill_past_value_0.npy", v0.values(), v0.shape());
  }

  VoiceCloneChunkedResult result;
  result.sample_rate = 24000;
  std::vector<int64_t> first_tokens;
  std::vector<int64_t> generated_flat;
  std::vector<float> waveform;
  const int64_t ref_frames = inputs.reference_codes.empty() ? 0 : inputs.reference_codes.shape()[0];
  // full_codes 坐标中 [0, ref_frames) 是参考音频，不能发给用户播放。
  int64_t next_decode_start = ref_frames;
  size_t next_chunk_index = 0;
  size_t next_emit_index = 0;
  int64_t past_len = prompt.shape()[1];
  const int max_frames = std::max(inputs.max_new_tokens - 1, 1);

  auto decode_task = [&](ChunkDecodeTask task) -> ChunkDecodeResult {
    // 这个 lambda 同时服务同步和异步 chunk decode，保证两条路径行为一致。
    const auto chunk_start = Clock::now();
    auto audio = DecodeCodesChunk(task.full_codes, task.start_frame, task.end_frame, task.left_context_frames);
    AddTiming(task.is_final ? "pipeline.decode_final_chunk" : "pipeline.decode_chunk", ElapsedMs(chunk_start));
    DumpChunkDebug(inputs.debug_dump_dir,
                   task.index,
                   task.full_codes,
                   audio,
                   task.start_frame,
                   task.end_frame,
                   task.left_context_frames,
                   task.generated_frames,
                   task.is_final,
                   config_.talker.num_code_groups);
    VoiceCloneChunk chunk{std::move(audio),
                          24000,
                          task.start_frame,
                          task.end_frame,
                          task.generated_frames,
                          task.is_final};
    return ChunkDecodeResult{task.index, std::move(chunk)};
  };

  std::unique_ptr<AsyncChunkDecoder> async_decoder;
  if (chunk_options.async_chunk_decode) {
    async_decoder = std::make_unique<AsyncChunkDecoder>(
        decode_task,
        static_cast<size_t>(chunk_options.max_decode_queue));
  }

  auto emit_chunk = [&](VoiceCloneChunk chunk) {
    // emit 顺序即用户听到的顺序。waveform 用于最终返回完整拼接音频。
    waveform.insert(waveform.end(), chunk.audio.values().begin(), chunk.audio.values().end());
    if (on_chunk) on_chunk(chunk);
    result.chunks.push_back(std::move(chunk));
  };

  auto drain_ready_chunks = [&]() {
    // 异步模式下尽量及时取出已完成 chunk，减少 result map 占用。
    if (!async_decoder) return;
    VoiceCloneChunk chunk;
    while (async_decoder->TryPop(next_emit_index, &chunk)) {
      emit_chunk(std::move(chunk));
      ++next_emit_index;
    }
  };

  auto submit_chunk_decode = [&](Int64Tensor full_codes,
                                 int64_t start_frame,
                                 int64_t end_frame,
                                 int64_t generated_frames,
                                 bool is_final) {
    ChunkDecodeTask task{next_chunk_index++,
                         std::move(full_codes),
                         start_frame,
                         end_frame,
                         chunk_options.left_context_frames,
                         generated_frames,
                         is_final};
    if (async_decoder) {
      // 异步时生成线程只提交任务；后台线程负责 DecodeCodesChunk。
      const auto enqueue_start = Clock::now();
      async_decoder->Enqueue(std::move(task));
      AddTiming("pipeline.enqueue_chunk_decode", ElapsedMs(enqueue_start));
      drain_ready_chunks();
      return;
    }
    auto decoded = decode_task(std::move(task));
    emit_chunk(std::move(decoded.chunk));
    ++next_emit_index;
  };

  auto finish_async_decode = [&]() {
    // 生成循环结束后必须收完所有后台 chunk，才能返回完整 waveform。
    if (!async_decoder) return;
    async_decoder->CloseInput();
    while (next_emit_index < next_chunk_index) {
      auto chunk = async_decoder->WaitPop(next_emit_index);
      emit_chunk(std::move(chunk));
      ++next_emit_index;
    }
    async_decoder->Join();
    async_decoder.reset();
  };

  std::vector<std::string> decode_output_names{"logits", "last_hidden"};
  decode_output_names.reserve(static_cast<size_t>(2 + config_.talker.num_hidden_layers * 2));
  std::unordered_set<std::string> decode_device_output_names{"logits", "last_hidden"};
  decode_device_output_names.reserve(static_cast<size_t>(config_.talker.num_hidden_layers * 2));
  for (int64_t layer = 0; layer < config_.talker.num_hidden_layers; ++layer) {
    auto key_name = "new_past_key_" + std::to_string(layer);
    auto value_name = "new_past_value_" + std::to_string(layer);
    decode_device_output_names.insert(key_name);
    decode_device_output_names.insert(value_name);
    decode_output_names.push_back(std::move(key_name));
    decode_output_names.push_back(std::move(value_name));
  }

  Int64Tensor decode_attention_mask({1, past_len + 2},
                                    std::vector<int64_t>(static_cast<size_t>(past_len + 2), 1));
  decode_attention_mask.values().reserve(static_cast<size_t>(prompt.shape()[1] + max_frames + 2));
  Int64Tensor decode_cache_position({1}, {past_len});

  const auto loop_start = Clock::now();
  for (int frame = 0; frame < max_frames; ++frame) {
    const auto frame_start = Clock::now();
    const auto logits_filter_start = Clock::now();
    auto filtered_logits = next_logits;
    const int64_t mask_begin = std::max<int64_t>(0, config_.talker.vocab_size - config_.talker.first_codebook_mask_tail);
    for (int64_t id = mask_begin; id < config_.talker.vocab_size; ++id) {
      if (id != config_.talker.codec_eos_token_id) filtered_logits[static_cast<size_t>(id)] = -std::numeric_limits<float>::infinity();
    }
    AddTiming("generation.last_logits_filter", ElapsedMs(logits_filter_start));
    const auto sample_start = Clock::now();
    int64_t first = sampler_.Sample(filtered_logits, inputs.main_sampling, first_tokens);
    if (!inputs.debug_dump_dir.empty()) {
      int64_t forced = 0;
      const auto forced_path = inputs.debug_dump_dir / ("first_token_pick_f" + std::to_string(frame) + ".npy");
      if (ReadNpyInt64Scalar(forced_path, &forced)) {
        first = forced;
      }
    }
    AddTiming("generation.sample_first_token", ElapsedMs(sample_start));
    if (trace_tokens) {
      const float eos_logit = next_logits[static_cast<size_t>(config_.talker.codec_eos_token_id)];
      std::cerr << "[trace] frame=" << frame
                << " first=" << first
                << " eos_id=" << config_.talker.codec_eos_token_id
                << " eos_logit=" << eos_logit
                << (first == config_.talker.codec_eos_token_id ? " hit_eos=1" : " hit_eos=0")
                << "\n";
    }
    if (!inputs.debug_dump_dir.empty()) {
      WriteNpy(inputs.debug_dump_dir / ("first_token_logits_f" + std::to_string(frame) + ".npy"),
               filtered_logits, {static_cast<int64_t>(filtered_logits.size())});
      WriteNpy(inputs.debug_dump_dir / ("first_token_pick_f" + std::to_string(frame) + ".npy"),
               std::vector<int64_t>{first}, {1});
    }
    if (first == config_.talker.codec_eos_token_id) {
      break;
    }
    first_tokens.push_back(first);
    auto [row, frame_embed] = RunCodePredictor(last_hidden, first, inputs.code_sampling, frame, inputs.debug_dump_dir);
    const auto append_codes_start = Clock::now();
    generated_flat.insert(generated_flat.end(), row.begin(), row.end());
    AddTiming("generation.append_generated_codes", ElapsedMs(append_codes_start));

    const auto decode_embed_start = Clock::now();
    FloatTensor decode_embed = frame < trailing_text.shape()[1]
                                    ? Add(frame_embed, SliceAxis1(trailing_text, frame, frame + 1))
                                    : Add(frame_embed, tts_pad);
    AddTiming("generation.build_decode_embed", ElapsedMs(decode_embed_start));
    if (!inputs.debug_dump_dir.empty() && frame == 0) {
      WriteNpy(inputs.debug_dump_dir / "decode0_inputs_embeds.npy", decode_embed.values(), decode_embed.shape());
      WriteNpy(inputs.debug_dump_dir / "decode0_attention_mask.npy",
               std::vector<int64_t>(static_cast<size_t>(past_len + 2), 1), {1, past_len + 2});
      WriteNpy(inputs.debug_dump_dir / "decode0_cache_position.npy", std::vector<int64_t>{past_len}, {1});
      auto k0 = CopyFloatTensor(past[0]);
      auto v0 = CopyFloatTensor(past[1]);
      WriteNpy(inputs.debug_dump_dir / "decode0_past_key_0.npy", k0.values(), k0.shape());
      WriteNpy(inputs.debug_dump_dir / "decode0_past_value_0.npy", v0.values(), v0.shape());
    }
    const auto feed_start = Clock::now();
    std::unordered_map<std::string, Ort::Value> feeds;
    feeds.reserve(static_cast<size_t>(3 + config_.talker.num_hidden_layers * 2));
    if (decode_attention_mask.shape()[1] < past_len + 2) {
      decode_attention_mask.values().push_back(1);
      decode_attention_mask.shape()[1] = past_len + 2;
    }
    decode_cache_position.values()[0] = past_len;
    feeds.emplace("inputs_embeds", talker_decode_.MakeTensor(decode_embed, "inputs_embeds"));
    feeds.emplace("attention_mask", talker_decode_.MakeTensor(decode_attention_mask));
    feeds.emplace("cache_position", talker_decode_.MakeTensor(decode_cache_position));
    for (int64_t layer = 0; layer < config_.talker.num_hidden_layers; ++layer) {
      feeds.emplace("past_key_" + std::to_string(layer), std::move(past[static_cast<size_t>(2 * layer)]));
      feeds.emplace("past_value_" + std::to_string(layer), std::move(past[static_cast<size_t>(2 * layer + 1)]));
    }
    AddTiming("generation.prepare_decode_feed", ElapsedMs(feed_start));

    const auto decode_start = Clock::now();
    auto out = talker_decode_.RunRawIoBinding(feeds, decode_output_names, decode_device_output_names);
    AddTiming("onnx.talker_decode", ElapsedMs(decode_start));
    const auto decode_copy_start = Clock::now();
    next_logits = LastLogits(out[0]);
    last_hidden = LastHidden(out[1]);
    if (!inputs.debug_dump_dir.empty() && frame == 0) {
      WriteNpy(inputs.debug_dump_dir / "decode0_logits_last.npy", next_logits, {config_.talker.vocab_size});
      auto hidden_full = CopyFloatTensor(out[1]);
      WriteNpy(inputs.debug_dump_dir / "decode0_last_hidden.npy", hidden_full.values(), hidden_full.shape());
    }
    AddTiming("generation.talker_decode_copy_outputs", ElapsedMs(decode_copy_start));
    const auto past_move_start = Clock::now();
    past.clear();
    past.reserve(static_cast<size_t>(config_.talker.num_hidden_layers * 2));
    for (size_t i = 2; i < out.size(); ++i) past.emplace_back(std::move(out[i]));
    AddTiming("generation.talker_decode_move_past", ElapsedMs(past_move_start));
    ++past_len;

    const int64_t generated_frames = static_cast<int64_t>(generated_flat.size() / static_cast<size_t>(config_.talker.num_code_groups));
    const int64_t available_end = ref_frames + generated_frames;
    while (available_end - next_decode_start >= chunk_options.chunk_frames) {
      // 注意 full_codes 是当前快照：参考帧 + 已生成帧。
      // 每个 chunk 解码都需要能看到它左侧的上下文。
      const int64_t end_frame = next_decode_start + chunk_options.chunk_frames;
      const auto prepare_start = Clock::now();
      Int64Tensor full_codes({ref_frames + generated_frames, config_.talker.num_code_groups});
      if (!inputs.reference_codes.empty()) {
        std::copy(inputs.reference_codes.values().begin(), inputs.reference_codes.values().end(), full_codes.values().begin());
      }
      std::copy(generated_flat.begin(), generated_flat.end(),
                full_codes.values().begin() + static_cast<long>(inputs.reference_codes.size()));
      AddTiming("pipeline.prepare_chunk_codes", ElapsedMs(prepare_start));
      submit_chunk_decode(std::move(full_codes), next_decode_start, end_frame, generated_frames, false);
      next_decode_start = end_frame;
    }
    drain_ready_chunks();
    AddTiming("generation.frame_total", ElapsedMs(frame_start));
  }
  AddTiming("generation.decode_loop_total", ElapsedMs(loop_start));

  const int64_t generated_frames = static_cast<int64_t>(generated_flat.size() / static_cast<size_t>(config_.talker.num_code_groups));
  result.generated_codes = Int64Tensor({generated_frames, config_.talker.num_code_groups}, generated_flat);
  const int64_t final_end = ref_frames + generated_frames;
  if (final_end > next_decode_start) {
    // 最后一段通常不足 chunk_frames，也要解码并标记 is_final=true。
    const auto prepare_start = Clock::now();
    Int64Tensor full_codes({ref_frames + generated_frames, config_.talker.num_code_groups});
    if (!inputs.reference_codes.empty()) {
      std::copy(inputs.reference_codes.values().begin(), inputs.reference_codes.values().end(), full_codes.values().begin());
    }
    std::copy(generated_flat.begin(), generated_flat.end(),
              full_codes.values().begin() + static_cast<long>(inputs.reference_codes.size()));
    AddTiming("pipeline.prepare_final_chunk_codes", ElapsedMs(prepare_start));
    submit_chunk_decode(std::move(full_codes), next_decode_start, final_end, generated_frames, true);
  }
  finish_async_decode();
  const int64_t waveform_samples = static_cast<int64_t>(waveform.size());
  result.waveform = FloatTensor({waveform_samples}, std::move(waveform));
  AddTiming("total.generate_from_prepared_chunked", ElapsedMs(total_start));
  return result;
}

VoiceCloneResult VoiceCloneRuntime::GenerateVoiceClone(const VoiceCloneRequest& request) {
  const auto total_start = Clock::now();
  VoiceCloneInputs inputs;
  // 高层接口负责把“用户输入”转成 GenerateFromPrepared 需要的三类条件：
  // 文本 token ids、参考 codec codes、speaker embedding。
  const auto encode_text_start = Clock::now();
  inputs.assistant_text_ids = EncodeAssistantText(request.text);
  if (!request.reference_text.empty()) {
    inputs.reference_text_ids = EncodeReferenceText(request.reference_text);
  }
  AddTiming("prep.text_ids", ElapsedMs(encode_text_start));
  auto reference_features = GetReferenceAudioFeatures(request.reference_audio, request.x_vector_only_mode);
  inputs.reference_codes = std::move(reference_features.reference_codes);
  inputs.speaker_embedding = std::move(reference_features.speaker_embedding);
  inputs.language = request.language;
  inputs.max_new_tokens = request.max_new_tokens;
  inputs.main_sampling = request.main_sampling;
  inputs.code_sampling = request.code_sampling;
  auto result = GenerateFromPrepared(inputs);
  AddTiming("total.generate_voice_clone", ElapsedMs(total_start));
  return result;
}

VoiceCloneChunkedResult VoiceCloneRuntime::GenerateVoiceCloneChunked(
    const VoiceCloneRequest& request,
    const VoiceCloneChunkOptions& chunk_options,
    const VoiceCloneChunkCallback& on_chunk) {
  const auto total_start = Clock::now();
  VoiceCloneInputs inputs;
  // chunk 高层接口和非流式接口共享同一套准备逻辑，差异只在后面的 decoder 输出方式。
  const auto encode_text_start = Clock::now();
  inputs.assistant_text_ids = EncodeAssistantText(request.text);
  if (!request.reference_text.empty()) {
    inputs.reference_text_ids = EncodeReferenceText(request.reference_text);
  }
  AddTiming("prep.text_ids", ElapsedMs(encode_text_start));
  auto reference_features = GetReferenceAudioFeatures(request.reference_audio, request.x_vector_only_mode);
  inputs.reference_codes = std::move(reference_features.reference_codes);
  inputs.speaker_embedding = std::move(reference_features.speaker_embedding);
  inputs.language = request.language;
  inputs.max_new_tokens = request.max_new_tokens;
  inputs.main_sampling = request.main_sampling;
  inputs.code_sampling = request.code_sampling;
  auto result = GenerateFromPreparedChunked(inputs, chunk_options, on_chunk);
  AddTiming("total.generate_voice_clone_chunked", ElapsedMs(total_start));
  return result;
}

}  // 命名空间 qwen::onnx
