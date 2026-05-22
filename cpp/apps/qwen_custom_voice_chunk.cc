// chunk/pipeline C++ CustomVoice command line entry.
//
// This is the streaming-flavored sibling of qwen_custom_voice.cc: it calls
// GenerateCustomVoiceChunked() with an on_chunk callback so each decoded audio
// chunk can be observed, written, played, or forwarded immediately.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
            << " --text TEXT --speaker Vivian --output out.wav"
            << " [--model DIR] [--onnx-root DIR]"
            << " [--provider CPUExecutionProvider|CUDAExecutionProvider]"
            << " [--prep-provider CPUExecutionProvider|CUDAExecutionProvider]"
            << " [--decode-provider CPUExecutionProvider|CUDAExecutionProvider]"
            << " [--language Chinese] [--max-new-tokens N] [--chunk-frames N]"
            << " [--left-context-frames N] [--crossfade-ms N] [--seed N] [--cuda-device N]"
            << " [--greedy] [--async-chunk-decode] [--decode-workers N] [--max-decode-queue N]"
            << " [--dump-dir DIR] [--chunk-dir DIR] [--no-timing]\n";
}

int MillisecondsToSamples(double milliseconds, int sample_rate) {
  return static_cast<int>(std::llround(std::max(0.0, milliseconds) * static_cast<double>(sample_rate) / 1000.0));
}

std::vector<float> ConcatWithCrossfade(const std::vector<qwen::onnx::VoiceCloneChunk>& chunks,
                                       int crossfade_samples) {
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

void ValidateChunkOptions(const qwen::onnx::VoiceCloneChunkOptions& chunk_options) {
  if (chunk_options.chunk_frames <= 0) throw std::runtime_error("--chunk-frames must be positive");
  if (chunk_options.left_context_frames < 0) throw std::runtime_error("--left-context-frames must be non-negative");
  if (chunk_options.decode_workers <= 0) throw std::runtime_error("--decode-workers must be positive");
  if (chunk_options.max_decode_queue <= 0) throw std::runtime_error("--max-decode-queue must be positive");
  if (chunk_options.async_chunk_decode && chunk_options.decode_workers != 1) {
    throw std::runtime_error("--async-chunk-decode currently supports exactly one --decode-workers");
  }
}

}  // namespace

int main(int argc, char** argv) {
  qwen::onnx::RuntimeOptions options;
  options.model_dir = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
  options.onnx_root = "./onnx_custom_voice_0p6b_fp16";
  options.providers = {"CUDAExecutionProvider"};
  options.prep_providers = {"CPUExecutionProvider"};
  options.load_reference_frontend = false;

  qwen::onnx::CustomVoiceRequest request;
  request.text = "你好，这是 Qwen 三自定义音色的 C++ chunk 流式测试。";
  request.language = "Chinese";
  request.speaker = "Vivian";
  request.max_new_tokens = 160;

  qwen::onnx::VoiceCloneChunkOptions chunk_options;
  std::filesystem::path output = "output_custom_voice_cpp_chunk.wav";
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
    else if (arg == "--speaker") request.speaker = next();
    else if (arg == "--language") request.language = next();
    else if (arg == "--instruct") request.instruct = next();
    else if (arg == "--output") output = next();
    else if (arg == "--dump-dir") request.debug_dump_dir = next();
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
    if (request.max_new_tokens <= 0) request.max_new_tokens = gen.max_new_tokens;
    if (greedy) {
      request.main_sampling.do_sample = false;
      request.code_sampling.do_sample = false;
    }
    ValidateChunkOptions(chunk_options);

    std::cerr << "Using run provider=" << (options.providers.empty() ? "<none>" : options.providers[0])
              << ", prep provider=" << (options.prep_providers.empty() ? "<none>" : options.prep_providers[0])
              << ", decode provider=" << (options.decode_providers.empty() ? "<run>" : options.decode_providers[0])
              << ", cuda_device_id=" << options.cuda_device_id << "\n";

    const auto init_start = Clock::now();
    qwen::onnx::VoiceCloneRuntime runtime(options);
    const double init_ms = ElapsedMs(init_start);

    size_t streamed_chunks = 0;
    const auto on_chunk = [&](const qwen::onnx::VoiceCloneChunk& chunk) {
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
      if (!chunk_dir.empty()) std::cout << " write_ms=" << ElapsedMs(chunk_start);
      std::cout << "\n" << std::flush;
      ++streamed_chunks;
    };

    const auto gen_start = Clock::now();
    auto result = runtime.GenerateCustomVoiceChunked(request, chunk_options, on_chunk);
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

    if (print_timing) {
      const auto old_flags = std::cout.flags();
      const auto old_precision = std::cout.precision();
      std::cout << "\n[Timing] Overall\n" << std::fixed << std::setprecision(2)
                << "  total.init_runtime: " << init_ms << " ms\n"
                << "  total.generate_custom_voice_chunked: " << generate_ms << " ms\n"
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
