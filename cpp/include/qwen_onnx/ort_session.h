#pragma once

// ONNX Runtime 的轻量封装。
//
// C++ runtime 需要同时跑很多拆开的子模型：text_project、codec_embed、
// talker_prefill、talker_decode、tokenizer decoder 等。这个类把 session
// 初始化、输入 dtype 对齐、FP16 输入转换、普通 run 和 I/O Binding 收在一起，
// 让上层 VoiceCloneRuntime 只关心“给某个子模型喂什么张量”。

#include <memory>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "qwen_onnx/tensor.h"

namespace qwen::onnx {

struct OrtSessionConfig {
  // .onnx 文件路径，通常位于 onnx_isolated/<submodel>/<name>.onnx。
  std::string model_path;
  // Execution provider 列表，例如 CPUExecutionProvider 或 CUDAExecutionProvider。
  std::vector<std::string> providers{"CPUExecutionProvider"};
  // Keep aligned with Python onnxruntime.InferenceSession defaults unless overridden.
  int intra_op_num_threads = 0;
  int inter_op_num_threads = 0;
  bool enable_mem_pattern = true;
  int cuda_device_id = 0;
};

class OrtSession {
 public:
  OrtSession() = default;
  OrtSession(Ort::Env& env, OrtSessionConfig config);

  const std::vector<std::string>& InputNames() const { return input_names_; }
  const std::vector<std::string>& OutputNames() const { return output_names_; }
  const std::string& ModelPath() const { return config_.model_path; }
  ONNXTensorElementDataType InputType(const std::string& name) const;
  ONNXTensorElementDataType OutputType(const std::string& name) const;

  std::vector<Ort::Value> RunRaw(std::unordered_map<std::string, Ort::Value>& inputs,
                                const std::vector<std::string>& output_names = {}) const;
  // 使用 ORT I/O Binding 运行 session。device_output_names 里的输出会绑定到 CUDA
  // device memory，适合 talker KV cache 这种下一步还要继续喂回 GPU 的张量。
  std::vector<Ort::Value> RunRawIoBinding(std::unordered_map<std::string, Ort::Value>& inputs,
                                          const std::vector<std::string>& output_names,
                                          const std::unordered_set<std::string>& device_output_names) const;

  // 常用便捷接口：运行一个输出并拷贝成项目自己的 Tensor 容器。
  FloatTensor RunFloat(std::unordered_map<std::string, Ort::Value>& inputs,
                       const std::string& output_name) const;
  Int64Tensor RunInt64(std::unordered_map<std::string, Ort::Value>& inputs,
                       const std::string& output_name) const;
  static FloatTensor CopyFloatTensor(Ort::Value& value);

  // MakeTensor 会根据 ONNX 输入声明自动处理 float32/float16。
  // 这样上层可以统一用 FloatTensor(float32) 表示中间结果，具体 FP16 转换
  // 留在 session 边界完成。
  Ort::Value MakeTensor(const FloatTensor& tensor) const;
  Ort::Value MakeTensor(const FloatTensor& tensor, const std::string& input_name) const;
  Ort::Value MakeTensorFromData(const float* data,
                                size_t count,
                                const std::vector<int64_t>& shape,
                                const std::string& input_name) const;
  Ort::Value MakeTensor(const Int64Tensor& tensor) const;
  bool UsesCuda() const { return uses_cuda_; }

 private:
  void InitNames();
  static Ort::SessionOptions BuildOptions(const OrtSessionConfig& config);

  OrtSessionConfig config_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_{nullptr};
  Ort::MemoryInfo cuda_memory_info_{nullptr};
  bool uses_cuda_ = false;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, ONNXTensorElementDataType> input_types_;
  std::unordered_map<std::string, ONNXTensorElementDataType> output_types_;
  // FP16 输入需要临时拥有一份转换后的 Ort::Float16_t buffer；
  // Ort::Value 只引用外部内存，所以 buffer 必须活到 session.Run 返回。
  mutable std::vector<std::vector<Ort::Float16_t>> fp16_input_storage_;
  mutable size_t fp16_input_storage_index_ = 0;
};

}  // 命名空间 qwen::onnx
