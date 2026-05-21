# Qwen3-TTS ONNX Runtime

这个项目用于把 Qwen3-TTS 12Hz 模型拆分导出为 ONNX 子模型，并提供 Python / C++ ONNX Runtime 推理代码。

当前支持三条主要流程：

```text
Base voice clone:
参考音频 + 参考文本 + 目标文本
  -> 生成 codec tokens
  -> tokenizer decoder 还原 wav

0.6B CustomVoice:
预置 speaker + 目标文本
  -> 生成 codec tokens
  -> tokenizer decoder 还原 wav

1.7B VoiceDesign:
音色描述 instruct + 目标文本
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

### Base FP32

```bash
python scripts/export_onnx.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --dtype fp32 \
  --output-root ./onnx_isolated \
  --clean
```

### Base FP16

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

如果导出 `0.6B-Base`，注意 `code_predictor` 不建议全 FP16；更稳的方式是其它子图 FP16，`code_predictor` 保持 FP32。**0.6B-Base 的 `tokenizer12hz_decode.onnx` 也可能遇到下面的 FP16 decoder CUDA 问题，导出后同样建议跑一次 FP32 islands 修补脚本。**

### CustomVoice 0.6B

CustomVoice 不需要参考音频前端，所以导出时可以跳过 `speaker_encoder`：

```bash
python scripts/onnx_export/export_all_isolated_onnx.py \
  --clean \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --output-root ./onnx_custom_voice_0p6b_fp16 \
  --device cuda \
  --dtype float16 \
  --skip-speaker-encoder
```

CPU/FP32 调试也可以导出到 `./onnx_custom_voice_0p6b_fp32`，把 `--device` 改成 `cpu`、`--dtype` 改成 `float32` 即可。

### VoiceDesign 1.7B

VoiceDesign 的音色条件来自 `instruct` 文本，也不需要参考音频前端：

```bash
python scripts/onnx_export/export_all_isolated_onnx.py \
  --clean \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --output-root ./onnx_voice_design_1p7b_fp16 \
  --device cuda \
  --dtype float16 \
  --skip-speaker-encoder
```

### 0.6B FP16 decoder CUDA 修补

**0.6B-Base 和 0.6B-CustomVoice 都可能在 `tokenizer12hz_decode.onnx` 的 CUDA FP16 路径出现 CUDNN `ReduceSum` / `Conv` 相关报错；可以先用下面的脚本给 decoder 插入局部 FP32 计算岛，再用修补后的 ONNX root 跑 Python 或 C++：**

```bash
python scripts/onnx_export/patch_decoder_fp32_islands.py \
  --onnx-root ./onnx_custom_voice_0p6b_fp16 \
  --output-root ./onnx_custom_voice_0p6b_fp16_fp32_islands \
  --overwrite
```

默认会修补 `tokenizer12hz/tokenizer12hz_decode.onnx` 里的 lengths `ReduceSum` 和 8 个 SwiGLU block；主生成模型仍保持原来的 FP16 图不变。

`0.6B-Base` 也使用同一个脚本，只需要把 `--onnx-root` 和 `--output-root` 换成对应目录，例如：

```bash
python scripts/onnx_export/patch_decoder_fp32_islands.py \
  --onnx-root ./onnx_qwen3_tts_0p6b_base_fp16 \
  --output-root ./onnx_qwen3_tts_0p6b_base_fp16_fp32_islands \
  --overwrite
```

## Python ONNX Runtime 运行

Base voice clone：

```bash
python scripts/onnx_runtime/voice_clone_ort.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 Python ONNX Runtime 声音克隆测试。" \
  --ref-audio ./data/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --output output_voice_clone_ort.wav
```

CustomVoice：

```bash
python scripts/onnx_runtime/custom_voice_ort.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --onnx-root ./onnx_custom_voice_0p6b_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 Qwen 三自定义音色的 ONNX Runtime 测试。" \
  --language Chinese \
  --speaker Vivian \
  --max-new-tokens 120 \
  --output output_custom_voice_ort.wav
```

VoiceDesign：

```bash
python scripts/onnx_runtime/voice_design_ort.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --onnx-root ./onnx_voice_design_1p7b_fp16 \
  --provider CUDAExecutionProvider \
  --text "你好，这是 Qwen 三音色设计的 ONNX Runtime 测试。" \
  --language Chinese \
  --instruct "一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。" \
  --max-new-tokens 120 \
  --output output_voice_design_ort.wav
```

**如果使用 0.6B FP16 并且 decoder 走 CUDA，`--onnx-root` 可以换成前面生成的 `./onnx_custom_voice_0p6b_fp16_fp32_islands`。0.6B-Base 同理，换成自己的 `*_fp32_islands` 目录。**

CustomVoice 已验证可和原始 PyTorch 贪心生成对齐：

```bash
python scripts/onnx_runtime/compare_custom_voice_greedy.py \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --onnx-root ./onnx_custom_voice_0p6b_fp16 \
  --provider CUDAExecutionProvider \
  --device cuda \
  --dtype float16 \
  --text "你好，这是 Qwen 三自定义音色的 GPU 贪心对齐测试。" \
  --language Chinese \
  --speaker Vivian \
  --max-new-tokens 40 \
  --output-dir compare_custom_voice_gpu_fp16_smoke
```

**对齐测试如果要验证修补后的 0.6B decoder，把 `--onnx-root ./onnx_custom_voice_0p6b_fp16` 换成 `--onnx-root ./onnx_custom_voice_0p6b_fp16_fp32_islands`。**

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

## C++ CustomVoice 运行

CustomVoice 入口复用 Base 的 talker / code predictor / tokenizer decoder 流水线，只是 prompt 中的音色条件来自预置 speaker id，不需要 `ref-audio` 和 `ref-text`。

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

C++ CLI 默认就是主生成 CUDA、prep CPU、decode 跟随主 provider，也就是 CUDA。当前测试环境里，`tokenizer12hz_decode.onnx` 的 CUDA 路径如果触发 CUDNN kernel 兼容问题，再显式加 `--decode-provider CPUExecutionProvider`。贪心测试中 C++ 生成 codes 已和 Python ORT 完全一致。

**如果要尝试 0.6B decoder 全 CUDA，可以先把 `--onnx-root` 换成对应的 `*_fp32_islands` 目录；这个修补对 0.6B-Base 和 0.6B-CustomVoice 都适用。**

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
--onnx-root          ONNX 子模型目录；0.6B FP16 decoder CUDA 出错时可使用 *_fp32_islands 目录
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
```

更多实现细节见 [cpp/README.md](cpp/README.md) 和 [知乎文章](https://zhuanlan.zhihu.com/p/2040565149275252714)。
