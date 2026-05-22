# Qwen3-TTS ONNX Runtime C++

这个目录包含 Qwen3-TTS 声音克隆的 C++ ONNX Runtime 实现。

代码按层拆分，方便后续继续接入更多模型类型，而不是把所有逻辑堆在一个 demo 文件里。

## 目录结构

```text
cpp/
  CMakeLists.txt
  cmake/OnnxRuntime.cmake
  include/qwen_onnx/
    audio_frontend.h
    bpe_tokenizer.h
    generation_config.h
    model_config.h
    ort_session.h
    sampling.h
    tensor.h
    voice_clone_runtime.h
    wav_writer.h
  src/
    audio_frontend.cc
    bpe_tokenizer.cc
    generation_config.cc
    model_config.cc
    ort_session.cc
    sampling.cc
    tensor.cc
    voice_clone_runtime.cc
    wav_writer.cc
  apps/
    qwen_ort_smoke.cc
    qwen_custom_voice.cc
    qwen_voice_design.cc
    qwen_voice_clone.cc
    qwen_voice_clone_chunk.cc
    qwen_voice_clone_skeleton.cc
```

## 构建

```bash
cmake -S cpp -B cpp/build -DQWEN_ORT_USE_CUDA=ON
cmake --build cpp/build -j2
```

CMake 辅助脚本会使用当前 Python 环境中的 ONNX Runtime GPU 动态库。如果缺少 C/C++ 头文件，会把匹配版本的头文件放到 `cpp/third_party/onnxruntime/include`。

## 声音克隆 CLI

```bash
./cpp/build/qwen_voice_clone \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated \
  --provider CUDAExecutionProvider \
  --text "我和我的祖国，一刻也不能分割，无论你走到哪里" \
  --ref-audio ./data/林志玲.mp3 \
  --ref-text "告诉自己，不要怕" \
  --output output_voice_clone_cpp.wav
```

当前 C++ runtime 会从 ONNX Runtime 的输入/输出元信息里自动识别 FP32/FP16 浮点 IO，因此普通
`onnx_isolated` 和 `onnx_isolated_fp16` 都可以用于验证。导出 FP32 模型示例：

```bash
python scripts/onnx_export/export.py all \
  --clean \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output-root ./onnx_isolated \
  --device cuda \
  --dtype float32 \
  --with-chunk-decoder \
  --chunk-size 300 \
  --left-context-size 25
```

如果要验证 FLOAT16 模型，把运行参数里的 `--onnx-root` 改成 `./onnx_isolated_fp16` 即可。**如果验证的是 0.6B-Base，`tokenizer12hz_decode.onnx` 的 CUDA FP16 路径也可能遇到 CUDNN `ReduceSum` / `Conv` kernel 问题，建议先生成对应的 `*_fp32_islands` ONNX root。**

```bash
python scripts/onnx_export/export.py patch-decoder \
  --onnx-root ./onnx_qwen3_tts_0p6b_base_fp16 \
  --output-root ./onnx_qwen3_tts_0p6b_base_fp16_fp32_islands \
  --overwrite
```

CLI 执行的完整路径：

```text
文本 -> Qwen2 byte-level BPE tokenizer -> text_project
参考音频 -> tokenizer12hz_encode -> reference_codes
参考音频 -> Slaney mel spectrogram -> speaker_encoder -> speaker_embedding
talker_prefill/talker_decode/code_predictor -> 生成 codec codes
参考 codes + 生成 codes -> tokenizer12hz_decode -> wav
```

## CustomVoice CLI

`qwen_custom_voice` 支持 `Qwen3-TTS-12Hz-0.6B-CustomVoice`。它不读取参考音频，也不加载
`speaker_encoder` / `tokenizer12hz_encode`，音色条件来自模型配置里的预置 speaker id。

先导出 CustomVoice ONNX：

```bash
python scripts/onnx_export/export.py all \
  --clean \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --output-root ./onnx_custom_voice_0p6b_fp16 \
  --device cuda \
  --dtype float16 \
  --skip-speaker-encoder \
  --with-chunk-decoder \
  --chunk-size 300 \
  --left-context-size 25
```

**0.6B-CustomVoice 和 0.6B-Base 一样，FP16 decoder CUDA 路径可能触发 CUDNN `ReduceSum` / `Conv` kernel 问题；如果要让 decoder 也走 CUDA，建议先修补 decoder：**

```bash
python scripts/onnx_export/export.py patch-decoder \
  --onnx-root ./onnx_custom_voice_0p6b_fp16 \
  --output-root ./onnx_custom_voice_0p6b_fp16_fp32_islands \
  --overwrite
```

运行 C++ CustomVoice：

```bash
./cpp/build/qwen_custom_voice \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --onnx-root ./onnx_custom_voice_0p6b_fp16 \
  --text "你好，这是 C++ ONNX Runtime 自定义音色测试。" \
  --language Chinese \
  --speaker Vivian \
  --max-new-tokens 120 \
  --output output_custom_voice_cpp.wav
```

默认配置是主生成 CUDA、prep CPU、decode 跟随主 provider，也就是 CUDA。`--provider` 控制 talker/code predictor 等主生成图；
`--prep-provider` 控制 text/codec embedding 等小图；`--decode-provider` 控制 `tokenizer12hz_decode.onnx`。
当前环境里 decoder 全 CUDA 如果在 CUDNN Conv/ReduceSum kernel 上失败，再显式加 `--decode-provider CPUExecutionProvider`。
**如果要测试 0.6B decoder 全 CUDA，把 `--onnx-root` 换成对应的 `*_fp32_islands` 目录；这个修补对 0.6B-Base 和 0.6B-CustomVoice 都适用。**

常用 speaker 名称来自模型配置，当前 0.6B CustomVoice 包含：

```text
Vivian Serena Uncle_Fu Ryan Aiden Ono_Anna Sohee Eric Dylan
```

其中 `Eric` 和 `Dylan` 是方言音色；当 `--language Chinese` 或 `auto` 时，runtime 会按模型配置自动切到对应方言 language id。

### CustomVoice 对齐验证

Python ORT 和 PyTorch 的贪心对齐：

```bash
python scripts/verify_onnx.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --onnx-root ./onnx_custom_voice_0p6b_fp16 \
  --provider CUDAExecutionProvider \
  --device cuda \
  --torch-dtype float16 \
  --text "你好，这是 Qwen 三自定义音色的 GPU 贪心对齐测试。" \
  --language Chinese \
  --speaker Vivian \
  --max-new-tokens 40 \
  --output-dir compare_custom_voice_gpu_fp16_smoke
```

**如果对齐验证的是修补后的 0.6B decoder，把上面的 `--onnx-root` 换成 `./onnx_custom_voice_0p6b_fp16_fp32_islands`。**

随后运行 C++，导出 codes：

```bash
./cpp/build/qwen_custom_voice \
  --onnx-root ./onnx_custom_voice_0p6b_fp16 \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --text "你好，这是 Qwen 三自定义音色的 GPU 贪心对齐测试。" \
  --language Chinese \
  --speaker Vivian \
  --max-new-tokens 40 \
  --greedy \
  --output compare_custom_voice_gpu_fp16_smoke/cpp_custom_voice_greedy.wav \
  --codes-output compare_custom_voice_gpu_fp16_smoke/cpp_custom_voice_codes.npy
```

对比 codes：

```bash
python - <<'PY'
import numpy as np
py = np.load("compare_custom_voice_gpu_fp16_smoke/ort_custom_voice_codes.npy")
cpp = np.load("compare_custom_voice_gpu_fp16_smoke/cpp_custom_voice_codes.npy")
print(py.shape, cpp.shape)
print("exact:", np.array_equal(py, cpp))
print("mismatch:", int(np.count_nonzero(py != cpp)))
PY
```

已验证的结果：`(39, 16)` 对 `(39, 16)`，`exact: True`，`mismatch: 0`。C++ decoder 走 CPU 时，波形和 Python ORT 不是 bitwise 一样，但相关性约为 `0.999672`。

## VoiceDesign CLI

`qwen_voice_design` 支持 `Qwen3-TTS-12Hz-1.7B-VoiceDesign`。它不读取参考音频，也不加载
`speaker_encoder` / `tokenizer12hz_encode`，音色条件来自 `--instruct` 文本。

先导出 VoiceDesign ONNX：

```bash
python scripts/onnx_export/export.py all \
  --clean \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --output-root ./onnx_voice_design_1p7b_fp16 \
  --device cuda \
  --dtype float16 \
  --skip-speaker-encoder \
  --with-chunk-decoder \
  --chunk-size 300 \
  --left-context-size 25
```

运行 C++ VoiceDesign：

```bash
./cpp/build/qwen_voice_design \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --onnx-root ./onnx_voice_design_1p7b_fp16 \
  --text "你好，这是 C++ ONNX Runtime 音色设计测试。" \
  --language Chinese \
  --instruct "一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。" \
  --max-new-tokens 120 \
  --output output_voice_design_cpp.wav
```

默认 provider 组合同 CustomVoice：主生成 CUDA、prep CPU、decode CUDA。已验证的短贪心结果：
Python ORT 与 C++ 生成 codes 同为 `(39, 16)`，`exact: True`，`mismatch: 0`。

### 完整 decode

CLI 使用完整 decoder：

```text
参考 codes + 生成 codes -> tokenizer12hz_decode.onnx -> 裁掉参考音频 -> wav
```

完整 decode 路径保持为 C++ 的稳定基线，不受 chunk/pipeline 实验入口影响。
如果 `tokenizer12hz_decode.onnx` 报告的有效长度大于实际输出长度，runtime 会直接报错，
避免静默返回被 trace 长度截断的音频。
**0.6B FP16 模型如果完整 decoder 走 CUDA 报 CUDNN 错误，优先使用 `--decode-provider CPUExecutionProvider`，或者把 `--onnx-root` 指向对应的 `*_fp32_islands` 目录再测试。**

推荐验证命令：

```bash
./cpp/build/qwen_voice_clone \
  --onnx-root ./onnx_isolated_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是完整解码测试。" \
  --ref-audio ./data/tokenizer_demo_1.wav \
  --ref-text "告诉自己，不要怕" \
  --output full_decode.wav \
  --max-new-tokens 80 \
  --seed 1234 \
  --greedy
```

音频加载现在尽量少依赖第三方库：WAV 16-bit PCM 和 32-bit float 会直接读取；mp3/m4a 会通过 ffmpeg 解码到 24 kHz mono float 音频。

### Chunk/pipeline decode

chunk CLI 使用 `tokenizer12hz_decode_chunk.onnx`。talker 仍逐帧生成 codec，
每攒够 `--chunk-frames` 帧就解码一段音频，最后再解码不足一段的尾巴。
这些入口是独立可执行程序，不会改变原来的完整 decode 行为。

```bash
./cpp/build/qwen_voice_clone_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 C++ chunk 解码测试。" \
  --ref-audio ./data/林志玲.mp3 \
  --ref-text "告诉自己，不要怕" \
  --max-new-tokens 80 \
  --chunk-frames 300 \
  --left-context-frames 25 \
  --crossfade-ms 20 \
  --greedy \
  --output output_voice_clone_cpp_chunk.wav
```

```bash
./cpp/build/qwen_custom_voice_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --onnx-root ./onnx_custom_voice_1p7b_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 C++ CustomVoice chunk 解码测试。" \
  --speaker Vivian \
  --max-new-tokens 80 \
  --chunk-frames 30 \
  --left-context-frames 25 \
  --chunk-dir ./custom_voice_chunks \
  --output output_custom_voice_cpp_chunk.wav
```

```bash
./cpp/build/qwen_voice_design_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --onnx-root ./onnx_voice_design_1p7b_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 C++ VoiceDesign chunk 解码测试。" \
  --instruct "一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。" \
  --max-new-tokens 80 \
  --chunk-frames 30 \
  --left-context-frames 25 \
  --chunk-dir ./voice_design_chunks \
  --output output_voice_design_cpp_chunk.wav
```

`--crossfade-ms` 默认为 0，表示直接拼接每个 chunk；如果边界听起来一段一段的，
可以先从 20 ms 试起。

## GPU Smoke Test

```bash
./cpp/build/qwen_ort_smoke \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated \
  --provider CUDAExecutionProvider
```

期望输出包含：

```text
model_type=base tokenizer=qwen3_tts_tokenizer_12hz hidden=2048 layers=28
code_predictor_embed output shape=[1,5,2048]
```

## Prepared Runtime API

```bash
./cpp/build/qwen_voice_clone_skeleton \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated \
  --provider CUDAExecutionProvider
```

如果外部已经准备好这些前端结果，可以直接复用 `VoiceCloneRuntime::GenerateFromPrepared()`：

```text
assistant_text_ids
reference_text_ids
reference_codes: [ref_len, 16]
speaker_embedding: [1, 1, 2048]
```

端到端 C++ 声音克隆完整 decode 请使用 `VoiceCloneRuntime::GenerateVoiceClone()` 或
`qwen_voice_clone` 可执行程序；chunk/pipeline decode 请使用
`VoiceCloneRuntime::GenerateVoiceCloneChunked()` 或 `qwen_voice_clone_chunk`。
