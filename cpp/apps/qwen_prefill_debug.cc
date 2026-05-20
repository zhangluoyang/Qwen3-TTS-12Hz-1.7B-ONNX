// talker_prefill 调试工具。
//
// 输入一个 prompt.npy，直接跑 talker_prefill.onnx 并把 logits/last_hidden dump 出来，
// 用来定位 prompt 构造或 prefill 子图和 Python 是否一致。

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "qwen_onnx/ort_session.h"

namespace {

template <typename T>
void WriteNpy(const std::filesystem::path& path, const std::vector<T>& values, const std::vector<int64_t>& shape) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  const char* descr = std::is_same_v<T, float> ? "<f4" : "<i8";
  std::string shape_text = "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) shape_text += ", ";
    shape_text += std::to_string(shape[i]);
  }
  if (shape.size() == 1) shape_text += ",";
  shape_text += ")";
  std::string header = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': " + shape_text + ", }";
  header.append(16 - ((10 + header.size() + 1) % 16), ' ');
  header.push_back('\n');
  out.write("\x93NUMPY", 6);
  char version[2] = {1, 0};
  out.write(version, 2);
  uint16_t n = static_cast<uint16_t>(header.size());
  char len[2] = {static_cast<char>(n & 0xff), static_cast<char>((n >> 8) & 0xff)};
  out.write(len, 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}

qwen::onnx::FloatTensor ReadPromptRaw(const std::filesystem::path& path) {
  // 支持本项目 dump 出来的简单 .npy v1.0 float32 文件。
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("open failed");
  in.seekg(0, std::ios::end);
  auto bytes = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<char> buf(bytes);
  in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
  const char* magic = "\x93NUMPY";
  if (std::string(buf.data(), 6) != std::string(magic, 6)) throw std::runtime_error("not npy");
  uint16_t header_len = static_cast<unsigned char>(buf[8]) | (static_cast<unsigned char>(buf[9]) << 8);
  std::string header(buf.data() + 10, header_len);
  std::vector<int64_t> shape;
  std::regex shape_re("'shape': \\(([^\\)]*)\\)");
  std::smatch m;
  if (std::regex_search(header, m, shape_re)) {
    std::string s = m[1].str();
    std::regex num("([0-9]+)");
    for (auto it = std::sregex_iterator(s.begin(), s.end(), num); it != std::sregex_iterator(); ++it) {
      shape.push_back(std::stoll((*it)[1].str()));
    }
  }
  if (shape.empty()) throw std::runtime_error("prompt.npy has no shape");
  size_t off = 10 + header_len;
  size_t count = (bytes - off) / sizeof(float);
  std::vector<float> values(count);
  std::memcpy(values.data(), buf.data() + off, count * sizeof(float));
  return qwen::onnx::FloatTensor(std::move(shape), std::move(values));
}

}  // 匿名命名空间

int main(int argc, char** argv) {
  std::string prompt_path = argc > 1 ? argv[1] : "compare_outputs/cpp_parity_debug/cpp/prompt.npy";
  std::string out_dir = argc > 2 ? argv[2] : "compare_outputs/prefill_debug_cpp";
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "prefill_debug");
  qwen::onnx::OrtSessionConfig cfg;
  cfg.model_path = "onnx_isolated/talker_prefill/talker_prefill.onnx";
  cfg.providers = {"CPUExecutionProvider"};
  qwen::onnx::OrtSession session(env, cfg);
  auto prompt = ReadPromptRaw(prompt_path);
  if (prompt.shape().size() != 3 || prompt.shape()[0] != 1) {
    throw std::runtime_error("prompt.npy must have shape [1, seq_len, hidden_size]");
  }
  const int64_t seq_len = prompt.shape()[1];
  qwen::onnx::Int64Tensor mask({1, seq_len}, std::vector<int64_t>(static_cast<size_t>(seq_len), 1));
  std::unordered_map<std::string, Ort::Value> inputs;
  inputs.emplace("inputs_embeds", session.MakeTensor(prompt, "inputs_embeds"));
  inputs.emplace("attention_mask", session.MakeTensor(mask));
  auto out = session.RunRaw(inputs, {"logits", "last_hidden"});
  for (size_t i = 0; i < out.size(); ++i) {
    auto tensor = qwen::onnx::OrtSession::CopyFloatTensor(out[i]);
    auto shape = tensor.shape();
    auto v = tensor.values();
    WriteNpy(std::filesystem::path(out_dir) / (i == 0 ? "logits.npy" : "last_hidden.npy"), v, shape);
    std::cout << i << " shape";
    for (auto d : shape) std::cout << " " << d;
    std::cout << " first=" << v.front() << "\n";
  }
}
