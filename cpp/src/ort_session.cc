#include "qwen_onnx/ort_session.h"

// ONNX Runtime C++ API 的项目级封装。
//
// 这个文件解决三个重复问题：
// 1. 每个 session 的 provider/optimization 初始化；
// 2. FloatTensor(float32) 到 ONNX 输入声明 dtype(float32/float16) 的自动适配；
// 3. CUDA I/O Binding，让 talker KV cache 这类大张量可以留在 device 上。

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace qwen::onnx {
namespace {

template <typename T>
Tensor<T> CopyOrtTensor(Ort::Value& value) {
  // 普通 CPU 输出拷贝成项目自己的 Tensor<T>。CUDA/FP16 的特殊拷贝在
  // voice_clone_runtime.cc 里另有热路径处理。
  auto info = value.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  size_t count = info.GetElementCount();
  const T* src = value.GetTensorData<T>();
  return Tensor<T>(shape, std::vector<T>(src, src + count));
}

template <typename OrtElemT>
struct OrtFloatConvert;

template <>
struct OrtFloatConvert<float> {
  static float FromFloat(float value) { return value; }
  static float ToFloat(float value) { return value; }
};

template <>
struct OrtFloatConvert<Ort::Float16_t> {
  static Ort::Float16_t FromFloat(float value) { return Ort::Float16_t(value); }
  static float ToFloat(Ort::Float16_t value) { return value.ToFloat(); }
};

template <typename OrtElemT>
FloatTensor CopyOrtFloatTensorTyped(Ort::Value& value) {
  // 不管 ONNX 输出是 float 还是 float16，上层统一拿 float32，简化采样和调试 dump。
  auto info = value.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  const size_t count = info.GetElementCount();
  const auto* src = value.GetTensorData<OrtElemT>();
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) values[i] = OrtFloatConvert<OrtElemT>::ToFloat(src[i]);
  return FloatTensor(std::move(shape), std::move(values));
}

bool HasProvider(const std::vector<std::string>& providers, const std::string& provider) {
  for (const auto& p : providers) {
    if (p == provider) return true;
  }
  return false;
}

void ConvertFloatToFloat16(const float* src, Ort::Float16_t* dst, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dst[i] = Ort::Float16_t(src[i]);
  }
}

}  // 匿名命名空间

OrtSession::OrtSession(Ort::Env& env, OrtSessionConfig config)
    : config_(std::move(config)),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      cuda_memory_info_("Cuda", OrtDeviceAllocator, config_.cuda_device_id, OrtMemTypeDefault),
      uses_cuda_(HasProvider(config_.providers, "CUDAExecutionProvider")) {
  auto options = BuildOptions(config_);
  session_ = std::make_unique<Ort::Session>(env, config_.model_path.c_str(), options);
  InitNames();
}

Ort::SessionOptions OrtSession::BuildOptions(const OrtSessionConfig& config) {
  Ort::SessionOptions options;
  if (config.intra_op_num_threads > 0) options.SetIntraOpNumThreads(config.intra_op_num_threads);
  if (config.inter_op_num_threads > 0) options.SetInterOpNumThreads(config.inter_op_num_threads);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // Keep runtime behavior aligned with Python ORT defaults for latency comparison.
  options.SetDeterministicCompute(false);
  if (!config.enable_mem_pattern) options.DisableMemPattern();

  if (HasProvider(config.providers, "CUDAExecutionProvider")) {
    // provider 追加失败时直接报错，避免用户以为跑在 GPU 实际回落到 CPU。
    try {
      Ort::CUDAProviderOptions cuda_options;
      cuda_options.Update({{"device_id", std::to_string(config.cuda_device_id)}});
      options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    } catch (const Ort::Exception& e) {
      throw std::runtime_error(std::string("Failed to append CUDAExecutionProvider: ") + e.what());
    }
  }

  return options;
}

void OrtSession::InitNames() {
  // 记录输入输出名和 dtype，后面 MakeTensor/RunRaw 能做校验和自动转换。
  Ort::AllocatorWithDefaultOptions allocator;
  input_names_.clear();
  output_names_.clear();
  input_types_.clear();
  output_types_.clear();
  for (size_t i = 0; i < session_->GetInputCount(); ++i) {
    auto allocated_name = session_->GetInputNameAllocated(i, allocator);
    std::string name = allocated_name.get();
    input_names_.push_back(name);
    input_types_[name] = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    auto allocated_name = session_->GetOutputNameAllocated(i, allocator);
    std::string name = allocated_name.get();
    output_names_.push_back(name);
    output_types_[name] = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
  }
}

ONNXTensorElementDataType OrtSession::InputType(const std::string& name) const {
  auto it = input_types_.find(name);
  if (it == input_types_.end()) throw std::runtime_error("Unknown input name: " + name);
  return it->second;
}

ONNXTensorElementDataType OrtSession::OutputType(const std::string& name) const {
  auto it = output_types_.find(name);
  if (it == output_types_.end()) throw std::runtime_error("Unknown output name: " + name);
  return it->second;
}

FloatTensor OrtSession::CopyFloatTensor(Ort::Value& value) {
  const auto type = value.GetTensorTypeAndShapeInfo().GetElementType();
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return CopyOrtFloatTensorTyped<float>(value);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return CopyOrtFloatTensorTyped<Ort::Float16_t>(value);
    default:
      throw std::runtime_error("Expected float or float16 tensor output");
  }
}

Ort::Value OrtSession::MakeTensor(const FloatTensor& tensor) const {
  return Ort::Value::CreateTensor<float>(memory_info_, const_cast<float*>(tensor.data()), tensor.size(),
                                         tensor.shape().data(), tensor.shape().size());
}

Ort::Value OrtSession::MakeTensor(const FloatTensor& tensor, const std::string& input_name) const {
  return MakeTensorFromData(tensor.data(), tensor.size(), tensor.shape(), input_name);
}

Ort::Value OrtSession::MakeTensorFromData(const float* data,
                                          size_t count,
                                          const std::vector<int64_t>& shape,
                                          const std::string& input_name) const {
  const auto type = InputType(input_name);
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::Value::CreateTensor<float>(memory_info_, const_cast<float*>(data), count,
                                           shape.data(), shape.size());
  }
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    // 上层一直用 float32 维护中间张量，只有喂 FP16 ONNX 模型时才转换。
    // storage 是 mutable 成员，保证 Ort::Value 引用的内存活到 RunRaw 结束。
    if (fp16_input_storage_index_ >= fp16_input_storage_.size()) {
      fp16_input_storage_.emplace_back();
    }
    auto& storage = fp16_input_storage_[fp16_input_storage_index_++];
    storage.resize(count);
    ConvertFloatToFloat16(data, storage.data(), count);
    return Ort::Value::CreateTensor<Ort::Float16_t>(memory_info_, storage.data(), storage.size(),
                                                    shape.data(), shape.size());
  }
  throw std::runtime_error("FloatTensor input does not match model input type for " + input_name);
}

Ort::Value OrtSession::MakeTensor(const Int64Tensor& tensor) const {
  return Ort::Value::CreateTensor<int64_t>(memory_info_, const_cast<int64_t*>(tensor.data()), tensor.size(),
                                           tensor.shape().data(), tensor.shape().size());
}

std::vector<Ort::Value> OrtSession::RunRaw(std::unordered_map<std::string, Ort::Value>& inputs,
                                           const std::vector<std::string>& requested_outputs) const {
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  std::vector<Ort::Value> input_values;
  input_names.reserve(inputs.size());
  input_values.reserve(inputs.size());
  for (const auto& name : input_names_) {
    // 按模型声明顺序组装输入，避免 unordered_map 的遍历顺序影响 ORT 调用。
    auto it = inputs.find(name);
    if (it != inputs.end()) {
      input_names.push_back(it->first.c_str());
      input_values.push_back(std::move(it->second));
    }
  }
  if (input_names.size() != inputs.size()) {
    throw std::runtime_error("Input map contains names that are not in model " + config_.model_path);
  }

  const auto& outs = requested_outputs.empty() ? output_names_ : requested_outputs;
  output_names.reserve(outs.size());
  for (const auto& name : outs) output_names.push_back(name.c_str());

  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(), input_values.size(),
                               output_names.data(), output_names.size());
  // 重置 FP16 scratch buffer 下标，下次 Run 可以复用内存。
  fp16_input_storage_index_ = 0;
  return outputs;
}

std::vector<Ort::Value> OrtSession::RunRawIoBinding(
    std::unordered_map<std::string, Ort::Value>& inputs,
    const std::vector<std::string>& requested_outputs,
    const std::unordered_set<std::string>& device_output_names) const {
  if (!uses_cuda_) {
    // CPU provider 没必要走 I/O Binding，普通 Run 更简单。
    return RunRaw(inputs, requested_outputs);
  }

  Ort::IoBinding binding(*session_);
  for (const auto& name : input_names_) {
    auto it = inputs.find(name);
    if (it != inputs.end()) {
      binding.BindInput(it->first.c_str(), it->second);
    }
  }
  if (inputs.size() > input_names_.size()) {
    throw std::runtime_error("Input map contains names that are not in model " + config_.model_path);
  }

  const auto& outs = requested_outputs.empty() ? output_names_ : requested_outputs;
  for (const auto& name : outs) {
    if (device_output_names.find(name) != device_output_names.end()) {
      // 这些输出下一轮还会作为输入，例如 past_key/value，直接留在 CUDA 上。
      binding.BindOutput(name.c_str(), cuda_memory_info_);
    } else {
      // logits/last_hidden 需要 CPU 侧采样或做 debug dump，就绑定到 CPU。
      binding.BindOutput(name.c_str(), memory_info_);
    }
  }

  session_->Run(Ort::RunOptions{nullptr}, binding);
  binding.SynchronizeOutputs();
  auto outputs = binding.GetOutputValues();
  fp16_input_storage_index_ = 0;
  return outputs;
}

FloatTensor OrtSession::RunFloat(std::unordered_map<std::string, Ort::Value>& inputs,
                                 const std::string& output_name) const {
  auto outputs = RunRaw(inputs, {output_name});
  return CopyFloatTensor(outputs[0]);
}

Int64Tensor OrtSession::RunInt64(std::unordered_map<std::string, Ort::Value>& inputs,
                                 const std::string& output_name) const {
  auto outputs = RunRaw(inputs, {output_name});
  return CopyOrtTensor<int64_t>(outputs[0]);
}

}  // 命名空间 qwen::onnx
