# Qwen3-TTS ONNX Runtime

这个项目用于把 Qwen3-TTS 12Hz 模型拆分导出为 ONNX 子模型，并提供 Python / C++ ONNX Runtime 推理代码。

当前重点支持 `Base` voice clone 流程：

```text
参考音频 + 参考文本 + 目标文本
  -> 生成 codec tokens
  -> tokenizer decoder 还原 wav
```

主要内容：

```text
scripts/onnx_export/      ONNX 导出脚本
scripts/onnx_runtime/     Python ONNX Runtime 示例
cpp/                      C++ ONNX Runtime 推理
```

## 一键导出 ONNX

默认模型路径示例：

```bash
/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

### FP32

```bash
python scripts/export_onnx.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --dtype fp32 \
  --output-root ./onnx_isolated \
  --clean
```

### FP16

```bash
python scripts/export_onnx.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --dtype fp16 \
  --output-root ./onnx_isolated_fp16 \
  --clean \
  --with-chunk-decoder
```

导出后目录大致如下：

```text
onnx_isolated_fp16/
  tokenizer12hz/
  text_project/
  codec_embed/
  code_predictor_embed/
  code_predictor/
  talker_prefill/
  talker_decode/
  speaker_encoder/
```

如果导出 `0.6B-Base`，注意 `code_predictor` 不建议全 FP16；更稳的方式是其它子图 FP16，`code_predictor` 保持 FP32。详细原因见 [qwen3-tts-onnx-runtime-zhihu-v2.md](qwen3-tts-onnx-runtime-zhihu-v2.md) 的精度策略部分。

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
  --onnx-root ./onnx_isolated_fp16 \
  --provider CUDAExecutionProvider
```

## C++ 非流式运行

非流式入口会先生成完整 codec 序列，再一次性调用 `tokenizer12hz_decode.onnx` 输出 wav。

```bash
./cpp/build/qwen_voice_clone \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated_fp16 \
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
  --onnx-root ./onnx_isolated_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 greedy 测试。" \
  --ref-audio ./data/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --max-new-tokens 80 \
  --greedy \
  --output output_voice_clone_cpp_greedy.wav
```

## C++ 流式 / 分块运行

流式入口会边生成 codec，边按 chunk 调用 `tokenizer12hz_decode_chunk.onnx` 解码音频片段。

```bash
./cpp/build/qwen_voice_clone_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 C++ ONNX Runtime 流式声音克隆测试。" \
  --ref-audio ./data/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --max-new-tokens 80 \
  --chunk-frames 20 \
  --left-context-frames 25 \
  --crossfade-ms 20 \
  --output output_voice_clone_cpp_chunk.wav
```

如果想把每个 chunk 单独写出来：

```bash
./cpp/build/qwen_voice_clone_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 chunk 文件输出测试。" \
  --ref-audio ./data/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --max-new-tokens 80 \
  --chunk-frames 20 \
  --left-context-frames 25 \
  --chunk-dir ./chunk_outputs \
  --output output_voice_clone_cpp_chunk.wav
```

## 常用参数

```text
--model              Qwen3-TTS 原始模型目录
--onnx-root          ONNX 子模型目录
--provider           CPUExecutionProvider 或 CUDAExecutionProvider
--text               要合成的文本
--ref-audio          参考音频
--ref-text           参考音频对应文本
--max-new-tokens     最大生成 token 数
--greedy             关闭采样，方便复现和调试
```

流式入口额外常用：

```text
--chunk-frames        每攒多少 codec 帧解码一个音频块
--left-context-frames chunk decoder 的左上下文帧数
--crossfade-ms        chunk 拼接时的交叉淡入淡出毫秒数
--chunk-dir           单独保存每个 chunk wav
```

更多实现细节见 [cpp/README.md](cpp/README.md) 和 [知乎文章](https://zhuanlan.zhihu.com/p/2040565149275252714)。
