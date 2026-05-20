// ONNX Runtime smoke test。
// 只加载一个小的 code_predictor_embed.onnx 并跑一次，快速确认 ORT/provider 可用。

#include <iostream>
#include <string>
#include <unordered_map>

#include "qwen_onnx/model_config.h"
#include "qwen_onnx/ort_session.h"

int main(int argc, char** argv) {
  // 如果这个程序能跑通，说明 CMake 链接、ONNX Runtime 动态库和 provider 基本正常。
  std::string model_dir = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base";
  std::string onnx_root = "./onnx_isolated";
  std::string provider = "CUDAExecutionProvider";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) model_dir = argv[++i];
    else if (arg == "--onnx-root" && i + 1 < argc) onnx_root = argv[++i];
    else if (arg == "--provider" && i + 1 < argc) provider = argv[++i];
  }

  try {
    auto cfg = qwen::onnx::LoadModelConfig(model_dir);
    std::cout << "model_type=" << cfg.tts_model_type << " tokenizer=" << cfg.tokenizer_type
              << " hidden=" << cfg.talker.hidden_size << " layers=" << cfg.talker.num_hidden_layers << "\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "qwen_ort_smoke");
    qwen::onnx::OrtSessionConfig session_cfg;
    session_cfg.model_path = onnx_root + "/code_predictor_embed/code_predictor_embed.onnx";
    session_cfg.providers = {provider};
    qwen::onnx::OrtSession session(env, session_cfg);

    qwen::onnx::Int64Tensor token_ids({1, 5}, {100, 101, 102, 103, 104});
    qwen::onnx::Int64Tensor layer_idx({}, {0});
    std::unordered_map<std::string, Ort::Value> inputs;
    inputs.emplace("token_id", session.MakeTensor(token_ids));
    inputs.emplace("layer_idx", session.MakeTensor(layer_idx));
    auto out = session.RunFloat(inputs, "embed");
    std::cout << "code_predictor_embed output shape=[";
    for (size_t i = 0; i < out.shape().size(); ++i) {
      if (i) std::cout << ",";
      std::cout << out.shape()[i];
    }
    std::cout << "] first=" << out.values().front() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
