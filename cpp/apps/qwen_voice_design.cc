// C++ VoiceDesign command line entry.
//
// Mirrors scripts/onnx_runtime/voice_design_ort.py: user instruction text
// conditions the voice/style, then the shared talker / code_predictor /
// tokenizer decoder pipeline generates codec codes and waveform.

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
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
            << " --text TEXT --instruct INSTRUCT --output out.wav"
            << " [--model DIR] [--onnx-root DIR]"
            << " [--provider CPUExecutionProvider|CUDAExecutionProvider]"
            << " [--prep-provider CPUExecutionProvider|CUDAExecutionProvider]"
            << " [--decode-provider CPUExecutionProvider|CUDAExecutionProvider]"
            << " [--language Chinese] [--max-new-tokens N] [--seed N] [--cuda-device N]"
            << " [--greedy] [--codes-output codes.npy] [--dump-dir DIR] [--no-timing]\n";
}

template <typename T>
void WriteNpy(const std::filesystem::path& path, const std::vector<T>& values, const std::vector<int64_t>& shape) {
  if (path.empty()) return;
  std::filesystem::create_directories(path.parent_path().empty() ? "." : path.parent_path());
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
  uint16_t header_len = static_cast<uint16_t>(header.size());
  char len[2] = {static_cast<char>(header_len & 0xff), static_cast<char>((header_len >> 8) & 0xff)};
  out.write(len, 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}

}  // namespace

int main(int argc, char** argv) {
  qwen::onnx::RuntimeOptions options;
  options.model_dir = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign";
  options.onnx_root = "./onnx_voice_design_1p7b_fp16";
  options.providers = {"CUDAExecutionProvider"};
  options.prep_providers = {"CPUExecutionProvider"};
  options.load_reference_frontend = false;

  qwen::onnx::VoiceDesignRequest request;
  request.text = "你好，这是 Qwen 三 VoiceDesign 的 C++ ONNX Runtime 测试。";
  request.language = "Chinese";
  request.instruct = "一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。";
  request.max_new_tokens = 120;

  std::filesystem::path output = "output_voice_design_cpp.wav";
  std::filesystem::path codes_output;
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
    else if (arg == "--instruct") request.instruct = next();
    else if (arg == "--language") request.language = next();
    else if (arg == "--output") output = next();
    else if (arg == "--codes-output") codes_output = next();
    else if (arg == "--dump-dir") request.debug_dump_dir = next();
    else if (arg == "--max-new-tokens") request.max_new_tokens = std::stoi(next());
    else if (arg == "--seed") options.seed = static_cast<uint64_t>(std::stoull(next()));
    else if (arg == "--cuda-device") options.cuda_device_id = std::stoi(next());
    else if (arg == "--greedy") greedy = true;
    else if (arg == "--no-timing") print_timing = false;
    else if (arg == "--help" || arg == "-h") {
      Usage(argv[0]);
      return 0;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
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

    std::cerr << "Using run provider=" << (options.providers.empty() ? "<none>" : options.providers[0])
              << ", prep provider=" << (options.prep_providers.empty() ? "<none>" : options.prep_providers[0])
              << ", decode provider=" << (options.decode_providers.empty() ? "<run>" : options.decode_providers[0])
              << ", cuda_device_id=" << options.cuda_device_id << "\n";

    const auto init_start = Clock::now();
    qwen::onnx::VoiceCloneRuntime runtime(options);
    const double init_ms = ElapsedMs(init_start);

    const auto gen_start = Clock::now();
    auto result = runtime.GenerateVoiceDesign(request);
    const double generate_ms = ElapsedMs(gen_start);

    const auto write_start = Clock::now();
    qwen::onnx::WriteWav(output, result.waveform.values(), result.sample_rate);
    if (!codes_output.empty()) {
      WriteNpy(codes_output, result.generated_codes.values(), result.generated_codes.shape());
    }
    const double write_ms = ElapsedMs(write_start);

    std::cout << "wrote " << output.string()
              << ": samples=" << result.waveform.size()
              << " sr=" << result.sample_rate
              << " generated_frames=" << result.generated_codes.shape()[0] << "\n";
    if (!codes_output.empty()) {
      std::cout << "wrote " << codes_output.string() << "\n";
    }

    if (print_timing) {
      const auto old_flags = std::cout.flags();
      const auto old_precision = std::cout.precision();
      std::cout << "\n[Timing] Overall\n" << std::fixed << std::setprecision(2)
                << "  total.init_runtime: " << init_ms << " ms\n"
                << "  total.generate_voice_design: " << generate_ms << " ms\n"
                << "  total.write_outputs: " << write_ms << " ms\n";
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
