# Qwen3-TTS ONNX Runtime

这个项目用于把 Qwen3-TTS 12Hz 模型拆分导出为 ONNX 子模型，并提供 Python / C++ ONNX Runtime 推理代码。

当前支持三条主要流程：

```text
Base voice clone:
参考音频 + 参考文本 + 目标文本
  -> 生成 codec tokens
  -> tokenizer decoder 还原 wav

CustomVoice:
预置 speaker + 目标文本
  -> 生成 codec tokens
  -> tokenizer decoder 还原 wav

VoiceDesign:
音色描述 instruct + 目标文本
  -> 生成 codec tokens
  -> tokenizer decoder 还原 wav
```

主要内容：

```text
scripts/export_onnx.py    统一 ONNX 导出入口
scripts/verify_onnx.py    统一 ONNX 验证入口
scripts/onnx_export/      内部导出/校验合并入口，日常不用直接调用
scripts/onnx_runtime/     推理 runtime 和统一样例
examples/                 PyTorch 参考样例
cpp/                      C++ ONNX Runtime 推理
```

## 一键导出 ONNX

日常只用 `scripts/export_onnx.py`。它会读取模型 `config.json` 里的
`tts_model_type`，自动决定 Base / CustomVoice / VoiceDesign 流程：

- 默认 `--dtype fp16`，自动用 `cuda`
- 默认导出 `tokenizer12hz_decode_chunk.onnx`
- 默认把分散的 external data 合并成每个 ONNX 一个 `.onnx.data`
- CustomVoice / VoiceDesign 自动跳过 `speaker_encoder`

Base voice clone：

```bash
python scripts/export_onnx.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --dtype fp16 \
  --clean
```

CustomVoice：

```bash
python scripts/export_onnx.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --dtype fp16 \
  --clean
```

VoiceDesign：

```bash
python scripts/export_onnx.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --dtype fp16 \
  --clean
```

默认输出目录会按模型类型、尺寸和 dtype 自动命名，例如：

```text
onnx_base_1p7b_fp16
onnx_custom_voice_1p7b_fp16
onnx_voice_design_1p7b_fp16
```

如果不想导出 chunk decoder，加 `--no-chunk-decoder`；如果想指定目录，加
`--output-root ./your_onnx_dir`。README 默认只写 1.7B FP16 路线。

导出后目录大致如下：

```text
onnx_base_1p7b_fp16/
  tokenizer12hz/
  text_project/
  codec_embed/
  code_predictor_embed/
  code_predictor/
  talker_prefill/
  talker_decode/
  speaker_encoder/
```

## 一键验证 ONNX

日常验证也只用 `scripts/verify_onnx.py`。它同样会读取 `tts_model_type`，
自动选择 Base / CustomVoice / VoiceDesign 的 PyTorch vs ONNX greedy 对齐流程：

```bash
python scripts/verify_onnx.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

如果 ONNX 目录不是默认命名，加 `--onnx-root`：

```bash
python scripts/verify_onnx.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --onnx-root ./onnx_custom_voice_1p7b_fp16
```

Base 模型可额外传 `--ref-audio` / `--ref-text`；CustomVoice 可传
`--speaker`；VoiceDesign 可传 `--instruct`。其它参数保持默认即可。

## Python ONNX Runtime 运行

统一入口会读取 `config.json` 里的 `tts_model_type`，自动选择 Base /
CustomVoice / VoiceDesign runtime。

Base voice clone：

```bash
python scripts/onnx_runtime/example.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --provider CUDAExecutionProvider \
  --text "你好，这是 Python ONNX Runtime 声音克隆测试。" \
  --ref-audio ./data/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --output output_voice_clone_example.wav
```

CustomVoice：

```bash
python scripts/onnx_runtime/example.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --provider CUDAExecutionProvider \
  --text "你好，这是 Qwen 三自定义音色的 ONNX Runtime 测试。" \
  --language Chinese \
  --speaker Vivian \
  --max-new-tokens 600 \
  --output output_custom_voice_example.wav
```

VoiceDesign：

```bash
python scripts/onnx_runtime/example.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --provider CUDAExecutionProvider \
  --text "你好，这是 Qwen 三音色设计的 ONNX Runtime 测试。" \
  --language Chinese \
  --instruct "一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。" \
  --max-new-tokens 600 \
  --output output_voice_design_example.wav
```

CustomVoice 贪心对齐验证示例：

```bash
python scripts/verify_onnx.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --onnx-root ./onnx_custom_voice_1p7b_fp16 \
  --provider CUDAExecutionProvider \
  --device cuda \
  --torch-dtype float16 \
  --text "你好，这是 Qwen 三自定义音色的 GPU 贪心对齐测试。" \
  --language Chinese \
  --speaker Vivian \
  --max-new-tokens 40 \
  --output-dir compare_custom_voice_gpu_fp16_smoke
```

## C++ 环境依赖

需要 C++17、CMake 和一个可用的 ONNX Runtime C++ 库。Ubuntu 上常用依赖：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ffmpeg
```

CUDA 推理需要 Python 环境里安装 GPU 版 ONNX Runtime：

```bash
python -m pip install onnxruntime-gpu==1.26.0
```

当前 C++ 构建会使用这两个本地 third-party 目录：

```text
cpp/third_party/onnxruntime-local/onnxruntime-linux-x64-gpu-1.26.0
cpp/third_party/fftw3-local
```

其中：

```text
onnxruntime-local  提供 ONNX Runtime C++ include/lib
fftw3-local        提供 speaker encoder 前处理用的 libfftw3f.a
```

如果这两个目录不存在，需要先准备 ONNX Runtime release 包和 FFTW3f 本地构建；当前仓库里的 `cpp/third_party/downloads/*.tgz` 和 `cpp/third_party/fftw-3.3.10.tar.gz` 只是下载/源码缓存，不是运行时直接依赖。

## 构建 C++

```bash
cmake -S cpp -B cpp/build -DQWEN_ORT_USE_CUDA=ON
cmake --build cpp/build -j2
```

快速检查 ONNX Runtime / CUDA provider 是否可用：

```bash
./cpp/build/qwen_ort_smoke \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_base_1p7b_fp16 \
  --provider CUDAExecutionProvider
```

## C++ 非流式运行

非流式入口会先生成完整 codec 序列，再一次性调用 `tokenizer12hz_decode.onnx` 输出 wav。

```bash
./cpp/build/qwen_voice_clone \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_base_1p7b_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 C++ ONNX Runtime 非流式声音克隆测试。" \
  --ref-audio ./data/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --max-new-tokens 80 \
  --output output_voice_clone_cpp.wav
```

调试时可以加 `--greedy` 关闭采样随机性：

```bash
./cpp/build/qwen_voice_clone \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_base_1p7b_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 greedy 测试。" \
  --ref-audio ./data/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --max-new-tokens 80 \
  --greedy \
  --output output_voice_clone_cpp_greedy.wav
```

## C++ CustomVoice 运行

CustomVoice 入口复用 Base 的 talker / code predictor / tokenizer decoder 流水线，只是 prompt 中的音色条件来自预置 speaker id，不需要 `ref-audio` 和 `ref-text`。

```bash
./cpp/build/qwen_custom_voice \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --onnx-root ./onnx_custom_voice_1p7b_fp16 \
  --text "你好，这是 C++ ONNX Runtime 自定义音色测试。" \
  --language Chinese \
  --speaker Vivian \
  --max-new-tokens 120 \
  --output output_custom_voice_cpp.wav
```

C++ CLI 默认就是主生成 CUDA、prep CPU、decode 跟随主 provider，也就是 CUDA。贪心测试中 C++ 生成 codes 已和 Python ORT 完全一致。

## C++ VoiceDesign 运行

VoiceDesign 入口复用同一条生成流水线，音色条件来自 `--instruct`，不需要 `ref-audio`、`ref-text` 或 `speaker`。

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

C++ VoiceDesign 贪心测试中，生成 codes 已和 Python ORT 完全一致。

## C++ 流式 / 分块运行

流式入口会边生成 codec，边按 chunk 调用 `tokenizer12hz_decode_chunk.onnx` 解码音频片段。
Base / CustomVoice / VoiceDesign 都有对应的 chunk CLI。

```bash
./cpp/build/qwen_voice_clone_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_base_1p7b_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 C++ ONNX Runtime 流式声音克隆测试。" \
  --ref-audio ./data/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --max-new-tokens 80 \
  --chunk-frames 300 \
  --left-context-frames 25 \
  --crossfade-ms 20 \
  --output output_voice_clone_cpp_chunk.wav
```

CustomVoice 流式：

```bash
./cpp/build/qwen_custom_voice_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --onnx-root ./onnx_custom_voice_1p7b_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 C++ ONNX Runtime 自定义音色流式测试。" \
  --language Chinese \
  --speaker Vivian \
  --max-new-tokens 120 \
  --chunk-frames 30 \
  --left-context-frames 25 \
  --output output_custom_voice_cpp_chunk.wav
```

VoiceDesign 流式：

```bash
./cpp/build/qwen_voice_design_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --onnx-root ./onnx_voice_design_1p7b_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 C++ ONNX Runtime 音色设计流式测试。" \
  --language Chinese \
  --instruct "一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。" \
  --max-new-tokens 120 \
  --chunk-frames 30 \
  --left-context-frames 25 \
  --output output_voice_design_cpp_chunk.wav
```

如果想把每个 chunk 单独写出来：

```bash
./cpp/build/qwen_voice_clone_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_base_1p7b_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 chunk 文件输出测试。" \
  --ref-audio ./data/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --max-new-tokens 80 \
  --chunk-frames 300 \
  --left-context-frames 25 \
  --chunk-dir ./chunk_outputs \
  --output output_voice_clone_cpp_chunk.wav
```

## 常用参数

```text
--model              Qwen3-TTS 原始模型目录
--onnx-root          ONNX 子模型目录，例如 ./onnx_base_1p7b_fp16
--provider           CPUExecutionProvider 或 CUDAExecutionProvider
--prep-provider      轻量前处理/embedding 子图 provider
--decode-provider    tokenizer decoder 子图 provider
--text               要合成的文本
--ref-audio          参考音频
--ref-text           参考音频对应文本
--speaker            CustomVoice 预置音色，例如 Vivian
--instruct           VoiceDesign 音色描述
--language           语言控制，例如 Chinese / English / auto
--max-new-tokens     最大生成 token 数
--greedy             关闭采样，方便复现和调试
```

流式入口额外常用：

```text
--chunk-frames        每攒多少 codec 帧解码一个音频块
--left-context-frames chunk decoder 的左上下文帧数
--crossfade-ms        chunk 拼接时的交叉淡入淡出毫秒数
--chunk-dir           单独保存每个 chunk wav
--async-chunk-decode  后台线程解码 chunk，生成线程继续前进
```

更多实现细节见 [cpp/README.md](cpp/README.md) 和 [知乎文章](https://zhuanlan.zhihu.com/p/2040565149275252714)。
