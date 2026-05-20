#include "qwen_onnx/wav_writer.h"

// 不依赖 libsndfile 的最小 WAV 写出实现。
// 输入为 float32 PCM，写出时裁剪到 [-1,1] 并转成 little-endian int16。

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace qwen::onnx {
namespace {

void WriteU16(std::ostream& out, uint16_t v) {
  char b[2] = {static_cast<char>(v & 0xff), static_cast<char>((v >> 8) & 0xff)};
  out.write(b, 2);
}

void WriteU32(std::ostream& out, uint32_t v) {
  char b[4] = {static_cast<char>(v & 0xff), static_cast<char>((v >> 8) & 0xff),
               static_cast<char>((v >> 16) & 0xff), static_cast<char>((v >> 24) & 0xff)};
  out.write(b, 4);
}

}  // 匿名命名空间

void WriteWav(const std::filesystem::path& path, const std::vector<float>& audio, int sample_rate) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Failed to open wav for writing: " + path.string());
  const uint16_t channels = 1;
  const uint16_t bits = 16;
  const uint32_t data_bytes = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
  out.write("RIFF", 4);
  WriteU32(out, 36 + data_bytes);
  out.write("WAVE", 4);
  out.write("fmt ", 4);
  WriteU32(out, 16);
  WriteU16(out, 1);
  WriteU16(out, channels);
  WriteU32(out, static_cast<uint32_t>(sample_rate));
  WriteU32(out, static_cast<uint32_t>(sample_rate * channels * bits / 8));
  WriteU16(out, channels * bits / 8);
  WriteU16(out, bits);
  out.write("data", 4);
  WriteU32(out, data_bytes);
  for (float x : audio) {
    x = std::max(-1.0f, std::min(1.0f, x));
    int32_t sample = static_cast<int32_t>(x * 32768.0f);
    sample = std::max<int32_t>(-32768, std::min<int32_t>(32767, sample));
    const int16_t v = static_cast<int16_t>(sample);
    WriteU16(out, static_cast<uint16_t>(v));
  }
}

}  // 命名空间 qwen::onnx
