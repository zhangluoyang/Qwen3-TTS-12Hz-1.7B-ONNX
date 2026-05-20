// talker_decode 调试工具。
//
// 从 .npy 读取 decode0 输入和 prefill KV cache，单独跑一轮 talker_decode.onnx，
// 方便和 Python dump 的 decode0_logits_last / decode0_last_hidden 做逐项比较。

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
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

struct NpyRaw {
  std::string descr;
  std::vector<int64_t> shape;
  std::vector<char> data;
};

NpyRaw ReadNpyRaw(const std::filesystem::path& path) {
  // 支持本项目 dump 出来的简单 .npy v1.0 文件，读取 descr、shape 和原始 data。
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("open failed " + path.string());
  in.seekg(0, std::ios::end);
  auto bytes = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<char> buf(bytes);
  in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
  uint16_t header_len = static_cast<unsigned char>(buf[8]) | (static_cast<unsigned char>(buf[9]) << 8);
  std::string header(buf.data() + 10, header_len);
  std::regex descr_re("'descr': '([^']+)'");
  std::regex shape_re("'shape': \\(([^\\)]*)\\)");
  std::smatch m;
  NpyRaw out;
  if (std::regex_search(header, m, descr_re)) out.descr = m[1].str();
  if (std::regex_search(header, m, shape_re)) {
    std::string s = m[1].str();
    std::regex num("([0-9]+)");
    for (auto it = std::sregex_iterator(s.begin(), s.end(), num); it != std::sregex_iterator(); ++it) {
      out.shape.push_back(std::stoll((*it)[1].str()));
    }
  }
  size_t off = 10 + header_len;
  out.data.assign(buf.begin() + static_cast<long>(off), buf.end());
  return out;
}

qwen::onnx::FloatTensor ReadFloat(const std::filesystem::path& p) {
  auto r = ReadNpyRaw(p);
  std::vector<float> v(r.data.size() / sizeof(float));
  std::memcpy(v.data(), r.data.data(), r.data.size());
  return qwen::onnx::FloatTensor(r.shape, std::move(v));
}

qwen::onnx::Int64Tensor ReadInt64(const std::filesystem::path& p) {
  auto r = ReadNpyRaw(p);
  std::vector<int64_t> v(r.data.size() / sizeof(int64_t));
  std::memcpy(v.data(), r.data.data(), r.data.size());
  return qwen::onnx::Int64Tensor(r.shape, std::move(v));
}

}  // 匿名命名空间

int main() {
  const std::filesystem::path dir = "compare_outputs/cpp_parity_debug/python";
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "decode_debug");
  qwen::onnx::OrtSessionConfig cfg;
  cfg.model_path = "onnx_isolated/talker_decode/talker_decode.onnx";
  cfg.providers = {"CPUExecutionProvider"};
  qwen::onnx::OrtSession s(env, cfg);
  std::unordered_map<std::string, Ort::Value> feeds;
  feeds.emplace("inputs_embeds", s.MakeTensor(ReadFloat(dir / "decode0_inputs_embeds.npy"), "inputs_embeds"));
  feeds.emplace("attention_mask", s.MakeTensor(ReadInt64(dir / "decode0_attention_mask.npy")));
  feeds.emplace("cache_position", s.MakeTensor(ReadInt64(dir / "decode0_cache_position.npy")));
  for (int i = 0; i < 28; ++i) {
    const auto key_name = "past_key_" + std::to_string(i);
    const auto value_name = "past_value_" + std::to_string(i);
    feeds.emplace(key_name, s.MakeTensor(ReadFloat(dir / ("prefill_past_key_" + std::to_string(i) + ".npy")), key_name));
    feeds.emplace(value_name,
                  s.MakeTensor(ReadFloat(dir / ("prefill_past_value_" + std::to_string(i) + ".npy")), value_name));
  }
  std::vector<std::string> outs{"logits", "last_hidden"};
  auto out = s.RunRaw(feeds, outs);
  for (int i = 0; i < 2; ++i) {
    auto tensor = qwen::onnx::OrtSession::CopyFloatTensor(out[i]);
    auto shape = tensor.shape();
    auto v = tensor.values();
    WriteNpy(std::filesystem::path("compare_outputs/decode_debug_cpp") / (i == 0 ? "logits.npy" : "last_hidden.npy"), v, shape);
    std::cout << i << " first=" << v.front() << "\n";
  }
}
