// Minimal tokenizer decoder debug tool.
//
// Reads int64 codec codes from a .npy file and runs tokenizer12hz_decode.onnx
// without loading the full TTS pipeline. This isolates CUDA decoder failures
// from talker/code_predictor state.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "qwen_onnx/ort_session.h"

namespace {

struct NpyRaw {
  std::string descr;
  std::vector<int64_t> shape;
  std::vector<char> data;
};

NpyRaw ReadNpyRaw(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Failed to open npy: " + path.string());
  in.seekg(0, std::ios::end);
  const auto bytes = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<char> buf(bytes);
  in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
  if (buf.size() < 10 || std::string(buf.data(), 6) != "\x93NUMPY") {
    throw std::runtime_error("Not a numpy .npy file: " + path.string());
  }
  const uint16_t header_len = static_cast<unsigned char>(buf[8]) |
                              (static_cast<unsigned char>(buf[9]) << 8);
  std::string header(buf.data() + 10, header_len);
  std::regex descr_re("'descr': '([^']+)'");
  std::regex shape_re("'shape': \\(([^\\)]*)\\)");
  std::smatch m;
  NpyRaw out;
  if (std::regex_search(header, m, descr_re)) out.descr = m[1].str();
  if (std::regex_search(header, m, shape_re)) {
    const std::string s = m[1].str();
    std::regex num("([0-9]+)");
    for (auto it = std::sregex_iterator(s.begin(), s.end(), num); it != std::sregex_iterator(); ++it) {
      out.shape.push_back(std::stoll((*it)[1].str()));
    }
  }
  const size_t offset = 10 + header_len;
  out.data.assign(buf.begin() + static_cast<long>(offset), buf.end());
  return out;
}

qwen::onnx::Int64Tensor ReadCodes(const std::filesystem::path& path) {
  auto raw = ReadNpyRaw(path);
  if (raw.descr != "<i8") throw std::runtime_error("Expected int64 npy codes");
  std::vector<int64_t> values(raw.data.size() / sizeof(int64_t));
  std::memcpy(values.data(), raw.data.data(), raw.data.size());
  if (raw.shape.size() == 2) {
    return qwen::onnx::Int64Tensor({1, raw.shape[0], raw.shape[1]}, std::move(values));
  }
  if (raw.shape.size() == 3) {
    return qwen::onnx::Int64Tensor(raw.shape, std::move(values));
  }
  throw std::runtime_error("Expected codes shape [frames, groups] or [1, frames, groups]");
}

int64_t ReadLength(const Ort::Value& value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) return -1;
  return value.GetTensorData<int64_t>()[0];
}

void Usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --onnx-root DIR --codes codes.npy"
            << " [--provider CPUExecutionProvider|CUDAExecutionProvider]"
            << " [--cuda-device N] [--iobinding]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path onnx_root = "./onnx_custom_voice_0p6b_fp16";
  std::filesystem::path codes_path = "compare_custom_voice_gpu_fp16_smoke/cpp_custom_voice_codes.npy";
  std::string provider = "CUDAExecutionProvider";
  int cuda_device = 0;
  bool iobinding = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--onnx-root") onnx_root = next();
    else if (arg == "--codes") codes_path = next();
    else if (arg == "--provider") provider = next();
    else if (arg == "--cuda-device") cuda_device = std::stoi(next());
    else if (arg == "--iobinding") iobinding = true;
    else if (arg == "--help" || arg == "-h") {
      Usage(argv[0]);
      return 0;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tokenizer_decode_debug");
    qwen::onnx::OrtSessionConfig cfg;
    cfg.model_path = (onnx_root / "tokenizer12hz" / "tokenizer12hz_decode.onnx").string();
    cfg.providers = {provider};
    cfg.cuda_device_id = cuda_device;
    qwen::onnx::OrtSession session(env, cfg);

    auto codes = ReadCodes(codes_path);
    std::unordered_map<std::string, Ort::Value> feeds;
    feeds.emplace("audio_codes", session.MakeTensor(codes));
    std::vector<Ort::Value> outputs;
    if (iobinding) {
      outputs = session.RunRawIoBinding(feeds, {"audio_values", "lengths"}, {"audio_values", "lengths"});
    } else {
      outputs = session.RunRaw(feeds, {"audio_values", "lengths"});
    }
    const auto audio_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto audio_shape = audio_info.GetShape();
    std::cout << "ok provider=" << provider << " iobinding=" << (iobinding ? 1 : 0)
              << " codes_shape=[";
    for (size_t i = 0; i < codes.shape().size(); ++i) {
      if (i) std::cout << ",";
      std::cout << codes.shape()[i];
    }
    std::cout << "] audio_shape=[";
    for (size_t i = 0; i < audio_shape.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << audio_shape[i];
    }
    std::cout << "] length=";
    if (iobinding && provider == "CUDAExecutionProvider") {
      std::cout << "<device>";
    } else {
      std::cout << ReadLength(outputs[1]);
    }
    std::cout << "\n";
  } catch (const std::exception& e) {
    Usage(argv[0]);
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
