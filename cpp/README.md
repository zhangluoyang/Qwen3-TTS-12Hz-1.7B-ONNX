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
python scripts/onnx_export/export_all_isolated_onnx.py \
  --clean \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output-root ./onnx_isolated \
  --device cuda \
  --dtype float32
```

如果要验证 FLOAT16 模型，把运行参数里的 `--onnx-root` 改成 `./onnx_isolated_fp16` 即可。

CLI 执行的完整路径：

```text
文本 -> Qwen2 byte-level BPE tokenizer -> text_project
参考音频 -> tokenizer12hz_encode -> reference_codes
参考音频 -> Slaney mel spectrogram -> speaker_encoder -> speaker_embedding
talker_prefill/talker_decode/code_predictor -> 生成 codec codes
参考 codes + 生成 codes -> tokenizer12hz_decode -> wav
```

### 完整 decode

CLI 使用完整 decoder：

```text
参考 codes + 生成 codes -> tokenizer12hz_decode.onnx -> 裁掉参考音频 -> wav
```

完整 decode 路径保持为 C++ 的稳定基线，不受 chunk/pipeline 实验入口影响。
如果 `tokenizer12hz_decode.onnx` 报告的有效长度大于实际输出长度，runtime 会直接报错，
避免静默返回被 trace 长度截断的音频。

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

新增的 chunk CLI 使用 `tokenizer12hz_decode_chunk.onnx`。talker 仍逐帧生成 codec，
每攒够 `--chunk-frames` 帧就解码一段音频，最后再解码不足一段的尾巴。
这个入口是独立可执行程序，不会改变原来的 `qwen_voice_clone` 完整 decode 行为。

```bash
./cpp/build/qwen_voice_clone_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 C++ chunk 解码测试。" \
  --ref-audio ./data/林志玲.mp3 \
  --ref-text "告诉自己，不要怕" \
  --max-new-tokens 80 \
  --chunk-frames 20 \
  --left-context-frames 25 \
  --crossfade-ms 20 \
  --greedy \
  --output output_voice_clone_cpp_chunk.wav
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
