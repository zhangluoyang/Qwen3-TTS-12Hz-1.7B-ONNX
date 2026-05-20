# Qwen3-TTS ONNX C++ Runtime 的实现思路

这篇文章整理 Qwen3-TTS ONNX C++ runtime 的设计思路。

它不是一篇源码逐行解析，也不展开内部调试细节。重点是说明：当一个语音生成模型被拆成多个 ONNX 子图之后，C++ runtime 应该如何组织推理流程、如何分配 CPU/GPU、如何支持流式输出，以及如何让整个生成过程稳定可控。

## 1. 目标不是“能跑”，而是“稳定、可控、可调优”

TTS 模型的自回归生成很敏感。某一帧 token 只要出现偏移，后续结果就可能一路变化，最终表现成：

- 声音重复
- 长时间空白
- EOS 没有命中
- 输出长度异常
- 听感不稳定

所以 C++ runtime 的目标不只是把 ONNX 跑起来，而是：

- 推理流程清晰
- 停止条件明确
- 非流式和流式都能稳定运行
- CPU/GPU 分配可控
- 后续可以继续做性能优化

## 2. 整体流程

整个 runtime 可以理解成四个阶段。

第一阶段是输入准备：

```text
文本输入
参考音频
参考文本
```

第二阶段是前处理：

```text
文本 tokenizer
参考音频读取和重采样
参考音频 tokenizer encode
说话人 embedding 提取
```

第三阶段是自回归生成：

```text
构造 prompt
talker prefill
逐帧生成 codec tokens
每帧再用 code predictor 补齐 residual codebook
遇到 EOS 后停止
```

第四阶段是音频解码：

```text
非流式：所有 codec tokens 生成完后一次性 decode
流式：每生成够一个 chunk 就 decode 一段音频
```

核心链路可以概括成：

```text
文本 + 参考音频
  -> prompt
  -> codec token 自回归生成
  -> tokenizer decoder
  -> waveform
```

## 3. EOS 的处理原则

EOS 的判断只发生在每一帧的第一个 codec token 上。

一帧里有多个 codebook token，但真正决定“是否结束”的，是 talker 生成出来的第一个 codec token。后面的 residual codebook token 是由 code predictor 补齐的，它们不是停止条件。

所以逻辑是：

```text
生成当前帧第一个 codec token
如果它是 codec_eos_token_id，则停止
否则继续用 code predictor 补齐这一帧
```

这个原则很重要。否则很容易出现模型已经应该结束，但 runtime 仍继续生成，最终表现成重复声音、长静音，或者一直生成到 `max_new_tokens` 才停。

## 4. 精度策略：尊重 ONNX 自己的类型

在自回归生成里，精度策略不是简单地“越高越好”。

ONNX 模型本身可能是 FP16，也可能是 FP32。runtime 如果无条件把中间结果转换成更高精度，看似更精确，但不一定符合模型导出时的数值语义。

自回归模型里，logits 很接近时，极小的数值变化就可能让 token 选择翻转：

```text
候选 token A 略高一点
候选 token B 略低一点
```

如果某一次舍入导致最高分 token 发生变化，后面整条生成路径就会分叉。

所以这里的原则是：

```text
ONNX 输出是什么精度，C++ 后处理就尊重对应的精度语义
```

C++ runtime 会根据 ONNX 输出类型自动决定累加策略：

- embedding 输出是 FP16，就按 FP16 语义累加
- embedding 输出是 FP32，就按 FP32 语义累加

这样既能适配当前 FP16 模型，也能兼容未来可能出现的 FP32 ONNX。

## 5. CPU/GPU 分配：不是所有 ONNX 都适合上 CUDA

很多人第一反应是：既然有 GPU，那所有 ONNX 都放 CUDA 会不会最快？

实际不一定。

这个 runtime 把 ONNX 子图分成两类：

```text
准备阶段 / 小图：更适合 CPU
生成主干 / 大图：更适合 GPU
```

### 5.1 适合 CPU 的部分

这些子图默认放 CPU：

```text
text_project
codec_embed
code_predictor_embed
speaker_encoder
tokenizer_encode
```

原因是它们通常具备这些特点：

- batch=1
- 输入规模小
- 大多只在请求开始时运行
- 或者属于 embedding / 小投影这类轻量操作
- 放 GPU 后容易被 kernel launch、同步、数据搬运抵消收益
- 放 CPU 可以减少显存占用

特别是 batch=1 的情况下，小图上 CUDA 不一定更快。GPU 不是没有成本的，每次 kernel launch、CPU/GPU 同步、显存分配都有开销。

### 5.2 适合 GPU 的部分

这些子图默认放 GPU：

```text
talker_prefill
talker_decode
code_predictor
tokenizer_decode
tokenizer_decode_chunk
```

原因是它们属于真正的重计算路径：

- `talker_prefill` 处理完整 prompt
- `talker_decode` 每生成一帧都要运行
- `code_predictor` 每帧要补齐多个 residual codebook token，调用次数非常多
- `tokenizer_decode` / `tokenizer_decode_chunk` 负责把 codec tokens 转成 waveform，计算量也更大

特别是 `code_predictor`，它通常是长文本生成里的主要瓶颈。放 CPU 会非常慢。

### 5.3 tokenizer_encode 为什么放 CPU，decoder 为什么放 GPU

这个点很容易误解。

`tokenizer_encode` 是把参考音频编码成 codec codes。它一般只在开头跑一次，而且参考音频通常较短，也更容易缓存。

`tokenizer_decode` / `tokenizer_decode_chunk` 是把生成出来的 codec codes 解码成音频。它处理的是最终输出语音，chunk 模式下还会运行多次。

所以策略是：

```text
tokenizer_encode: CPU
tokenizer_decode/tokenizer_decode_chunk: GPU
```

这不是因为 encoder 天生适合 CPU、decoder 天生适合 GPU，而是因为它们在这个 runtime 里的工作量不同。

## 6. 参考音频最好先统一成 WAV

语音克隆对参考音频很敏感。为了减少输入侧的不确定性，建议先把参考音频统一成：

```text
24kHz
mono
wav
```

这样可以减少格式解码、声道、采样率等因素带来的不稳定性，也更方便复现实验结果。

## 7. 流式 chunk 解码

非流式模式是：

```text
所有 codec tokens 生成完
一次性 decode 成完整 waveform
```

chunk 模式则是：

```text
一边生成 codec tokens
一边按 chunk 解码音频
```

比如每 50 帧 codec tokens 解码一次，同时带上一定的左上下文。左上下文用于改善块边界的连续性，但最终输出时只输出当前 chunk 的新音频，不重复输出上下文。

这个模式的好处是：

- 可以更早拿到第一段音频
- 更适合边生成边播放
- 更适合网络流式返回
- 不需要等整句话全部生成完才开始合成声音

## 8. producer-consumer 是否值得做

同步 chunk 流程是：

```text
生成一段 codec tokens
decode 当前 chunk
继续生成下一段 codec tokens
decode 下一个 chunk
```

producer-consumer 模式则是：

```text
生成线程继续生成后面的 codec tokens
解码线程同时处理已经完成的 chunk
```

理论上，它可以隐藏一部分 chunk decoder 的耗时。

但在单 GPU、单请求场景下，收益不一定明显。原因是 producer 和 consumer 仍然抢同一块 GPU：

```text
producer: talker_decode / code_predictor
consumer: tokenizer_decode_chunk
```

GPU 不会因为多线程就凭空多出算力。所以在这种场景下，同步和异步的总耗时可能接近。

那为什么还要保留这个结构？

因为它对后续扩展有价值：

- 推流时可以让 decoder / writer / network sender 独立工作
- 多请求并发时更容易调度
- decoder 如果放另一块 GPU，会更容易重叠
- 如果输出端很慢，生成端不必完全被阻塞

当前实现里，异步 chunk decode 是可选开关，默认不启用。这样既保留稳定路径，也方便后续继续扩展。

## 9. 当前主要性能瓶颈

从耗时看，真正的大头不是文本处理，也不是 wav 写文件，而是生成主干：

```text
code_predictor
talker_decode
```

其中 `code_predictor` 尤其明显。因为每一帧 codec token 里，主 token 由 talker 生成，剩余 residual codebook token 要由 code predictor 一个个补齐。

所以长文本时，code predictor 会被调用很多次。

这意味着后续如果继续优化，重点应该放在：

- 减少 code predictor 调用次数
- 尝试融合 residual codebook 生成过程
- 减少 CPU/GPU 同步
- 复用固定 shape 的 buffer
- 多请求场景下做调度

相比之下，把所有 embedding 小图都搬到 GPU，收益可能很有限，甚至可能让显存和同步开销变大。

## 10. 设计原则总结

这个 C++ runtime 的设计原则可以总结为：

```text
小图留 CPU
大图上 GPU
精度跟随 ONNX
EOS 只看主 codec token
chunk 支持流式
异步作为可选
关键路径保持可观测
```

最重要的不是某个局部优化，而是让整条推理链路稳定、清晰、可解释。

## 11. 使用方法

下面给出几个常用命令。

### 11.1 推荐的 chunk CUDA 运行方式

```bash
./cpp/build/qwen_voice_clone_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated_fp16 \
  --provider CUDAExecutionProvider \
  --prep-provider CPUExecutionProvider \
  --text "你好，这是 C++ chunk 流水线声音克隆测试。" \
  --ref-audio /tmp/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --output ./cpp_chunk_out.wav \
  --max-new-tokens 512 \
  --seed 1234 \
  --greedy
```

这里的意思是：

```text
生成主干和 decoder 使用 GPU
前处理和 embedding 小图使用 CPU
```

这是 batch=1 场景下推荐的配置。

### 11.2 开启异步 chunk decode

```bash
./cpp/build/qwen_voice_clone_chunk \
  --model /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-root ./onnx_isolated_fp16 \
  --provider CUDAExecutionProvider \
  --prep-provider CPUExecutionProvider \
  --text "你好，这是 C++ chunk 异步解码测试。" \
  --ref-audio /tmp/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --output ./cpp_chunk_async_out.wav \
  --max-new-tokens 512 \
  --seed 1234 \
  --greedy \
  --async-chunk-decode
```

异步模式适合实验流式推送、多请求调度或者后续多 GPU 解码。单 GPU 单请求下，它不一定显著缩短总耗时。

### 11.3 测试全 CUDA

如果想测试所有 ONNX 都放 GPU：

```bash
--provider CUDAExecutionProvider \
--prep-provider CUDAExecutionProvider
```

这个配置不一定更快，但可以用来观察不同子图放置策略对耗时和显存的影响。

### 11.4 测试全 CPU

如果想确认 CPU 路径：

```bash
--provider CPUExecutionProvider \
--prep-provider CPUExecutionProvider
```

全 CPU 通常会明显更慢，但适合在没有 CUDA 环境时运行。

## 12. 最后

这个 C++ 实现的核心不是“把模型跑起来”，而是把一个拆分后的 ONNX TTS 系统组织成稳定的 runtime：

- 输入可控
- 精度可控
- 停止策略可控
- CPU/GPU 分配可控
- 流式输出可控
- 同步和异步路径都可以按需选择

在这个基础上，后续无论是做低延迟推流、多请求并发，还是继续做 GPU 性能优化，都会更清晰。
