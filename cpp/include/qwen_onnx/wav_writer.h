#pragma once

// 最小 WAV 写出工具：把 float32 PCM [-1, 1] 写成 16-bit PCM wav。

#include <filesystem>
#include <vector>

namespace qwen::onnx {

void WriteWav(const std::filesystem::path& path, const std::vector<float>& audio, int sample_rate);

}  // 命名空间 qwen::onnx
