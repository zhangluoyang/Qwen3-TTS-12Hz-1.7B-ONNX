// 最小 runtime 初始化示例。
// 它不真正合成语音，只验证模型目录、ONNX 目录和 provider 是否能成功加载。

#include <iostream>
#include <string>

#include "qwen_onnx/voice_clone_runtime.h"

int main(int argc, char** argv) {
  qwen::onnx::RuntimeOptions options;
  options.model_dir = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base";
  options.onnx_root = "./onnx_isolated";
  options.providers = {"CUDAExecutionProvider"};

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) options.model_dir = argv[++i];
    else if (arg == "--onnx-root" && i + 1 < argc) options.onnx_root = argv[++i];
    else if (arg == "--provider" && i + 1 < argc) options.providers = {argv[++i]};
  }

  try {
    qwen::onnx::VoiceCloneRuntime runtime(options);
    const auto& cfg = runtime.Config();
    std::cout << "Qwen3-TTS C++ runtime initialized\n";
    std::cout << "  model_type: " << cfg.tts_model_type << "\n";
    std::cout << "  tokenizer:  " << cfg.tokenizer_type << "\n";
    std::cout << "  hidden:     " << cfg.talker.hidden_size << "\n";
    std::cout << "  layers:     " << cfg.talker.num_hidden_layers << "\n";
    std::cout << "\nThis executable is the structured voice-clone runtime entry point.\n";
    std::cout << "Feed prepared assistant/reference token ids, reference codec codes, and speaker embedding\n";
    std::cout << "into VoiceCloneRuntime::GenerateFromPrepared() to run the ONNX generation core.\n";
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
