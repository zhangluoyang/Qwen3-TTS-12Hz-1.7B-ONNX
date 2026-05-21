// chunk/pipeline C++ 声音克隆命令行入口。
//
// 和 qwen_voice_clone.cc 的区别：这里调用 GenerateVoiceCloneChunked()，
// 生成过程中每攒够一段 codec 帧就解码 chunk，适合观察准流式输出和延迟。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "qwen_onnx/audio_frontend.h"
#include "qwen_onnx/generation_config.h"
#include "qwen_onnx/voice_clone_runtime.h"
#include "qwen_onnx/wav_writer.h"

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void Usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --text TEXT --ref-audio WAV --ref-text TEXT --output out.wav"
            << " [--model DIR] [--onnx-root DIR]"
            << " [--provider CPUExecutionProvider|CUDAExecutionProvider]"
            << " [--prep-provider CPUExecutionProvider|CUDAExecutionProvider]"
            << " [--decode-provider CPUExecutionProvider|CUDAExecutionProvider]"
            << " [--language auto] [--max-new-tokens N] [--chunk-frames N]"
            << " [--left-context-frames N] [--crossfade-ms N] [--seed N] [--cuda-device N]"
            << " [--greedy] [--async-chunk-decode] [--decode-workers N] [--max-decode-queue N]"
            << " [--dump-dir DIR] [--chunk-dir DIR] [--no-timing]\n";
}

template <typename T>
void WriteNpy(const std::filesystem::path& path, const std::vector<T>& values, const std::vector<int64_t>& shape) {
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
  const size_t prefix = 10;
  const size_t padding = 16 - ((prefix + header.size() + 1) % 16);
  header.append(padding, ' ');
  header.push_back('\n');

  out.write("\x93NUMPY", 6);
  char version[2] = {1, 0};
  out.write(version, 2);
  uint16_t header_len = static_cast<uint16_t>(header.size());
  char len[2] = {static_cast<char>(header_len & 0xff), static_cast<char>((header_len >> 8) & 0xff)};
  out.write(len, 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void DumpTensor(const std::filesystem::path& dir, const std::string& name, const qwen::onnx::FloatTensor& tensor) {
  WriteNpy(dir / (name + ".npy"), tensor.values(), tensor.shape());
}

void DumpTensor(const std::filesystem::path& dir, const std::string& name, const qwen::onnx::Int64Tensor& tensor) {
  WriteNpy(dir / (name + ".npy"), tensor.values(), tensor.shape());
}

void DumpIds(const std::filesystem::path& dir, const std::string& name, const std::vector<int64_t>& ids) {
  WriteNpy(dir / (name + ".npy"), ids, {static_cast<int64_t>(ids.size())});
}

int MillisecondsToSamples(double milliseconds, int sample_rate) {
  return static_cast<int>(std::llround(std::max(0.0, milliseconds) * static_cast<double>(sample_rate) / 1000.0));
}

std::vector<float> ConcatWithCrossfade(const std::vector<qwen::onnx::VoiceCloneChunk>& chunks,
                                       int crossfade_samples) {
  // chunk decoder 已经裁掉左上下文；这里可选再做短交叉淡入淡出，平滑块边界。
  std::vector<float> output;
  for (const auto& chunk : chunks) {
    const auto& audio = chunk.audio.values();
    if (audio.empty()) continue;
    const int fade = std::min<int>({crossfade_samples,
                                    static_cast<int>(output.size()),
                                    static_cast<int>(audio.size())});
    if (fade <= 0) {
      output.insert(output.end(), audio.begin(), audio.end());
      continue;
    }

    const size_t base = output.size() - static_cast<size_t>(fade);
    for (int i = 0; i < fade; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(fade);
      const float fade_out = std::cos(t * static_cast<float>(M_PI) * 0.5f);
      const float fade_in = std::sin(t * static_cast<float>(M_PI) * 0.5f);
      output[base + static_cast<size_t>(i)] =
          output[base + static_cast<size_t>(i)] * fade_out + audio[static_cast<size_t>(i)] * fade_in;
    }
    output.insert(output.end(), audio.begin() + fade, audio.end());
  }
  return output;
}

}  // namespace

int main(int argc, char** argv) {
  // chunk 参数的核心是 chunk_frames 和 left_context_frames：
  // 前者控制多久产出一段，后者控制解码时参考多少历史帧。
  qwen::onnx::RuntimeOptions options;
  options.model_dir = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base";
  options.onnx_root = "./onnx_isolated_fp16";
  options.providers = {"CUDAExecutionProvider"};

  qwen::onnx::VoiceCloneRequest request;
  request.text = "你好，这是 C++ chunk 流水线声音克隆测试。";
  request.reference_audio = "./data/ref_from_mp3_24k_mono.wav";
  request.reference_text = "告诉自己，不要怕";
  request.max_new_tokens = 160;

  qwen::onnx::VoiceCloneChunkOptions chunk_options;
  std::filesystem::path output = "output_voice_clone_cpp_chunk.wav";
  std::filesystem::path dump_dir;
  std::filesystem::path chunk_dir;
  double crossfade_ms = 0.0;
  bool greedy = false;
  bool print_timing = true;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--model") options.model_dir = next();
    else if (arg == "--onnx-root") options.onnx_root = next();
    else if (arg == "--provider") options.providers = {next()};
    else if (arg == "--prep-provider") options.prep_providers = {next()};
    else if (arg == "--decode-provider") options.decode_providers = {next()};
    else if (arg == "--text") request.text = next();
    else if (arg == "--ref-audio") request.reference_audio = next();
    else if (arg == "--ref-text") request.reference_text = next();
    else if (arg == "--language") request.language = next();
    else if (arg == "--output") output = next();
    else if (arg == "--dump-dir") dump_dir = next();
    else if (arg == "--chunk-dir") chunk_dir = next();
    else if (arg == "--max-new-tokens") request.max_new_tokens = std::stoi(next());
    else if (arg == "--chunk-frames") chunk_options.chunk_frames = std::stoi(next());
    else if (arg == "--left-context-frames") chunk_options.left_context_frames = std::stoi(next());
    else if (arg == "--async-chunk-decode") chunk_options.async_chunk_decode = true;
    else if (arg == "--decode-workers") chunk_options.decode_workers = std::stoi(next());
    else if (arg == "--max-decode-queue") chunk_options.max_decode_queue = std::stoi(next());
    else if (arg == "--crossfade-ms") crossfade_ms = std::stod(next());
    else if (arg == "--seed") options.seed = static_cast<uint64_t>(std::stoull(next()));
    else if (arg == "--cuda-device") options.cuda_device_id = std::stoi(next());
    else if (arg == "--greedy") greedy = true;
    else if (arg == "--no-timing") print_timing = false;
    else if (arg == "--x-vector-only") request.x_vector_only_mode = true;
    else if (arg == "--help" || arg == "-h") {
      Usage(argv[0]);
      return 0;
    } else {
      Usage(argv[0]);
      std::cerr << "ERROR: Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  try {
    auto gen = qwen::onnx::LoadGenerationConfig(options.model_dir);
    request.main_sampling = gen.main_sampling;
    request.code_sampling = gen.code_sampling;
    if (greedy) {
      request.main_sampling.do_sample = false;
      request.code_sampling.do_sample = false;
    }
    if (chunk_options.chunk_frames <= 0) throw std::runtime_error("--chunk-frames must be positive");
    if (chunk_options.left_context_frames < 0) throw std::runtime_error("--left-context-frames must be non-negative");
    if (chunk_options.decode_workers <= 0) throw std::runtime_error("--decode-workers must be positive");
    if (chunk_options.max_decode_queue <= 0) throw std::runtime_error("--max-decode-queue must be positive");
    if (chunk_options.async_chunk_decode && chunk_options.decode_workers != 1) {
      throw std::runtime_error("--async-chunk-decode currently supports exactly one --decode-workers");
    }
    if (!options.providers.empty() || !options.prep_providers.empty()) {
      std::cerr << "Using run provider=" << (options.providers.empty() ? "<none>" : options.providers[0])
                << ", prep provider=" << (options.prep_providers.empty() ? "<none>" : options.prep_providers[0])
                << ", decode provider=" << (options.decode_providers.empty() ? "<run>" : options.decode_providers[0])
                << ", cuda_device_id=" << options.cuda_device_id << "\n";
    }

    const auto init_start = Clock::now();
    qwen::onnx::VoiceCloneRuntime runtime(options);
    const double init_ms = ElapsedMs(init_start);

    const auto gen_start = Clock::now();
    qwen::onnx::VoiceCloneChunkedResult result;
    size_t streamed_chunks = 0;
    const auto on_chunk = [&](const qwen::onnx::VoiceCloneChunk& chunk) {
      // 每个 chunk 完成后立即打印/可选写出单独 wav；真实流式服务可在这里推送网络包。
      const auto chunk_start = Clock::now();
      if (!chunk_dir.empty()) {
        std::filesystem::create_directories(chunk_dir);
        std::ostringstream name;
        name << "chunk_" << std::setw(3) << std::setfill('0') << streamed_chunks << ".wav";
        qwen::onnx::WriteWav(chunk_dir / name.str(), chunk.audio.values(), chunk.sample_rate);
      }
      std::cout << "stream chunk index=" << streamed_chunks
                << " frames=" << chunk.start_frame << ":" << chunk.end_frame
                << " samples=" << chunk.audio.size()
                << " generated_frames=" << chunk.generated_frames
                << " final=" << (chunk.is_final ? 1 : 0);
      if (!chunk_dir.empty()) {
        std::cout << " write_ms=" << ElapsedMs(chunk_start);
      }
      std::cout << "\n" << std::flush;
      ++streamed_chunks;
    };
    if (!dump_dir.empty()) {
      auto audio = qwen::onnx::LoadAudioMono(request.reference_audio, 24000);
      qwen::onnx::FloatTensor audio_tensor({1, static_cast<int64_t>(audio.samples.size())}, audio.samples);
      auto mel = qwen::onnx::MelSpectrogram(audio.samples);

      qwen::onnx::VoiceCloneInputs inputs;
      inputs.assistant_text_ids = runtime.EncodeAssistantText(request.text);
      inputs.reference_text_ids = runtime.EncodeReferenceText(request.reference_text);
      if (!request.x_vector_only_mode) inputs.reference_codes = runtime.EncodeReferenceCodes(audio_tensor);
      inputs.speaker_embedding = runtime.ExtractSpeakerEmbedding(mel);
      inputs.language = request.language;
      inputs.max_new_tokens = request.max_new_tokens;
      inputs.main_sampling = request.main_sampling;
      inputs.code_sampling = request.code_sampling;
      inputs.debug_dump_dir = dump_dir;

      DumpIds(dump_dir, "assistant_text_ids", inputs.assistant_text_ids);
      DumpIds(dump_dir, "reference_text_ids", inputs.reference_text_ids);
      DumpTensor(dump_dir, "audio_24k", audio_tensor);
      DumpTensor(dump_dir, "mel", mel);
      DumpTensor(dump_dir, "reference_codes", inputs.reference_codes);
      DumpTensor(dump_dir, "speaker_embedding", inputs.speaker_embedding);

      result = runtime.GenerateFromPreparedChunked(inputs, chunk_options, on_chunk);
      DumpTensor(dump_dir, "generated_codes", result.generated_codes);
      DumpTensor(dump_dir, "waveform", result.waveform);
      for (size_t i = 0; i < result.chunks.size(); ++i) {
        DumpTensor(dump_dir, "chunk_" + std::to_string(i) + "_audio", result.chunks[i].audio);
      }
    } else {
      result = runtime.GenerateVoiceCloneChunked(request, chunk_options, on_chunk);
    }
    const double generate_ms = ElapsedMs(gen_start);

    const auto write_start = Clock::now();
    auto wav = crossfade_ms > 0.0
                   ? ConcatWithCrossfade(result.chunks, MillisecondsToSamples(crossfade_ms, result.sample_rate))
                   : result.waveform.values();
    qwen::onnx::WriteWav(output, wav, result.sample_rate);
    const double write_ms = ElapsedMs(write_start);

    std::cout << "wrote " << output.string()
              << ": samples=" << wav.size()
              << " sr=" << result.sample_rate
              << " generated_frames=" << result.generated_codes.shape()[0]
              << " chunks=" << result.chunks.size() << "\n";
    for (const auto& chunk : result.chunks) {
      std::cout << "  chunk frames=" << chunk.start_frame << ":" << chunk.end_frame
                << " samples=" << chunk.audio.size()
                << " final=" << (chunk.is_final ? 1 : 0) << "\n";
    }

    if (print_timing) {
      const auto old_flags = std::cout.flags();
      const auto old_precision = std::cout.precision();
      std::cout << "\n[Timing] Overall\n" << std::fixed << std::setprecision(2)
                << "  total.init_runtime: " << init_ms << " ms\n"
                << "  total.generate_voice_clone_chunked: " << generate_ms << " ms\n"
                << "  total.write_wav: " << write_ms << " ms\n";
      std::cout.flags(old_flags);
      std::cout.precision(old_precision);
      runtime.PrintTimingSummary(std::cout);
    }
  } catch (const std::exception& e) {
    Usage(argv[0]);
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
