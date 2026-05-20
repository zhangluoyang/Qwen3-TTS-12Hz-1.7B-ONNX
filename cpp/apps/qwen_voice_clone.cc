// 非流式 C++ 声音克隆命令行入口。
//
// 这个程序负责解析命令行参数、构造 RuntimeOptions/VoiceCloneRequest，
// 调用 VoiceCloneRuntime::GenerateVoiceClone()，最后把完整 waveform 写成 wav。
// 如果要学习“C++ 怎么跑完整模型”，从这里进入再跳到 voice_clone_runtime.cc。

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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
            << " [--language auto] [--max-new-tokens N] [--seed N] [--cuda-device N] [--greedy]"
            << " [--dump-dir DIR] [--repeat N] [--no-timing]\n";
}

template <typename T>
void WriteNpy(const std::filesystem::path& path, const std::vector<T>& values, const std::vector<int64_t>& shape) {
  // 命令行调试 dump 使用 numpy .npy，方便 Python 脚本直接 np.load 对比。
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

std::filesystem::path RepeatOutputPath(const std::filesystem::path& output, int index, int repeat) {
  if (repeat <= 1) return output;
  auto stem = output.stem().string();
  auto ext = output.extension().string();
  if (stem.empty()) stem = "output";
  return output.parent_path() / (stem + "_" + std::to_string(index + 1) + ext);
}

}  // 匿名命名空间

int main(int argc, char** argv) {
  // 默认值指向本机缓存模型和导出的 ONNX 目录，可被命令行覆盖。
  qwen::onnx::RuntimeOptions options;
  options.model_dir = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base";
  options.onnx_root = "./onnx_isolated";
  options.providers = {"CUDAExecutionProvider"};

  qwen::onnx::VoiceCloneRequest request;
  request.text = "我和我的祖国，一刻也不能分割，无论你走到哪里";
  request.reference_audio = "./data/tokenizer_demo_1.wav";
  request.reference_text = "告诉自己，不要怕";
  std::filesystem::path output = "output_voice_clone_cpp.wav";
  std::filesystem::path dump_dir;
  bool greedy = false;
  bool print_timing = true;
  int repeat = 1;

  for (int i = 1; i < argc; ++i) {
    // 手写参数解析保持依赖最小；新增参数时注意同步 Usage()。
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--model") options.model_dir = next();
    else if (arg == "--onnx-root") options.onnx_root = next();
    else if (arg == "--provider") options.providers = {next()};
    else if (arg == "--prep-provider") options.prep_providers = {next()};
    else if (arg == "--text") request.text = next();
    else if (arg == "--ref-audio") request.reference_audio = next();
    else if (arg == "--ref-text") request.reference_text = next();
    else if (arg == "--language") request.language = next();
    else if (arg == "--output") output = next();
    else if (arg == "--dump-dir") dump_dir = next();
    else if (arg == "--max-new-tokens") request.max_new_tokens = std::stoi(next());
    else if (arg == "--repeat") repeat = std::stoi(next());
    else if (arg == "--seed") options.seed = static_cast<uint64_t>(std::stoull(next()));
    else if (arg == "--cuda-device") options.cuda_device_id = std::stoi(next());
    else if (arg == "--greedy") greedy = true;
    else if (arg == "--no-timing") print_timing = false;
    else if (arg == "--x-vector-only") request.x_vector_only_mode = true;
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
      // --greedy 用于 Python/C++/PyTorch 数值对齐，禁用两级采样随机性。
      request.main_sampling.do_sample = false;
      request.code_sampling.do_sample = false;
    }
    if (repeat <= 0) throw std::runtime_error("--repeat must be positive");
    if (!dump_dir.empty() && repeat != 1) throw std::runtime_error("--repeat is only supported without --dump-dir");
    if (!options.providers.empty() || !options.prep_providers.empty()) {
      std::cerr << "Using run provider=" << (options.providers.empty() ? "<none>" : options.providers[0])
                << ", prep provider=" << (options.prep_providers.empty() ? "<none>" : options.prep_providers[0])
                << ", cuda_device_id=" << options.cuda_device_id << "\n";
    }

    std::vector<qwen::onnx::TimingRecord> overall_timing;
    auto add_overall = [&](const std::string& name, double milliseconds) {
      overall_timing.push_back(qwen::onnx::TimingRecord{name, 1, milliseconds});
    };

    auto start = Clock::now();
    qwen::onnx::VoiceCloneRuntime runtime(options);
    add_overall("total.init_runtime", ElapsedMs(start));

    start = Clock::now();
    if (!dump_dir.empty()) {
      // dump 模式把“准备输入”和“生成核心”拆开，输出所有关键中间张量。
      qwen::onnx::VoiceCloneResult result;
      auto prep_start = Clock::now();
      auto audio = qwen::onnx::LoadAudioMono(request.reference_audio, 24000);
      qwen::onnx::FloatTensor audio_tensor({1, static_cast<int64_t>(audio.samples.size())}, audio.samples);
      auto mel = qwen::onnx::MelSpectrogram(audio.samples);
      add_overall("dump_path.prepare_inputs", ElapsedMs(prep_start));

      qwen::onnx::VoiceCloneInputs inputs;
      prep_start = Clock::now();
      inputs.assistant_text_ids = runtime.EncodeAssistantText(request.text);
      inputs.reference_text_ids = runtime.EncodeReferenceText(request.reference_text);
      if (!request.x_vector_only_mode) inputs.reference_codes = runtime.EncodeReferenceCodes(audio_tensor);
      inputs.speaker_embedding = runtime.ExtractSpeakerEmbedding(mel);
      add_overall("dump_path.frontend_onnx", ElapsedMs(prep_start));
      inputs.language = request.language;
      inputs.max_new_tokens = request.max_new_tokens;
      inputs.main_sampling = request.main_sampling;
      inputs.code_sampling = request.code_sampling;
      inputs.debug_dump_dir = dump_dir;

      prep_start = Clock::now();
      DumpIds(dump_dir, "assistant_text_ids", inputs.assistant_text_ids);
      DumpIds(dump_dir, "reference_text_ids", inputs.reference_text_ids);
      DumpTensor(dump_dir, "audio_24k", audio_tensor);
      DumpTensor(dump_dir, "mel", mel);
      DumpTensor(dump_dir, "reference_codes", inputs.reference_codes);
      DumpTensor(dump_dir, "speaker_embedding", inputs.speaker_embedding);
      add_overall("dump_path.dump_inputs", ElapsedMs(prep_start));

      result = runtime.GenerateFromPrepared(inputs);
      prep_start = Clock::now();
      DumpTensor(dump_dir, "generated_codes", result.generated_codes);
      DumpTensor(dump_dir, "waveform", result.waveform);
      add_overall("dump_path.dump_outputs", ElapsedMs(prep_start));
      add_overall("total.generate_voice_clone", ElapsedMs(start));

      start = Clock::now();
      qwen::onnx::WriteWav(output, result.waveform.values(), result.sample_rate);
      add_overall("total.write_wav", ElapsedMs(start));
      std::cout << "wrote " << output.string()
                << ": samples=" << result.waveform.size()
                << " sr=" << result.sample_rate
                << " generated_frames=" << result.generated_codes.shape()[0] << "\n";
    } else {
      for (int iter = 0; iter < repeat; ++iter) {
        // 多次 repeat 会复用同一个 runtime；参考音频特征会命中内部缓存。
        const auto iter_start = Clock::now();
        auto result = runtime.GenerateVoiceClone(request);
        add_overall("total.generate_voice_clone", ElapsedMs(iter_start));

        const auto write_start = Clock::now();
        const auto iter_output = RepeatOutputPath(output, iter, repeat);
        qwen::onnx::WriteWav(iter_output, result.waveform.values(), result.sample_rate);
        add_overall("total.write_wav", ElapsedMs(write_start));
        std::cout << "wrote " << iter_output.string()
                  << ": samples=" << result.waveform.size()
                  << " sr=" << result.sample_rate
                  << " generated_frames=" << result.generated_codes.shape()[0];
        std::cout << "\n";
      }
    }
    if (print_timing) {
      const auto old_flags = std::cout.flags();
      const auto old_precision = std::cout.precision();
      std::cout << "\n[Timing] Overall\n" << std::fixed << std::setprecision(2);
      for (const auto& record : overall_timing) {
        std::cout << "  " << record.name << ": " << record.total_ms << " ms\n";
      }
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
