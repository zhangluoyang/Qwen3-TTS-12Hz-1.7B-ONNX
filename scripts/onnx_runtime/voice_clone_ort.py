#!/usr/bin/env python3
"""使用导出的 ONNX Runtime 子模型执行单样本 Qwen3-TTS Base 声音克隆。

这个文件复刻的是官方 Qwen3-TTS Base voice clone 的主推理路径：

1. 参考音频 -> 12Hz tokenizer encoder，得到参考 codec codes。
2. 参考音频 -> speaker encoder，得到说话人 embedding。
3. 目标文本/参考文本 -> processor/tokenizer，再通过 text_project 得到文本 embedding。
4. 参考 codec codes -> codec/code_predictor embedding，和文本 embedding 拼成 talker prompt。
5. talker 自回归生成每一帧的第 1 个 codebook token。
6. code_predictor 继续补齐该帧剩余 15 个 residual codebook token。
7. tokenizer decoder 把 codec codes 转成 24 kHz waveform。

默认 `generate_voice_clone()` 是完整非流式基线；`iter_voice_clone_chunked()`
是额外的 chunk/pipeline 实验入口，不影响默认路径。
"""

import argparse
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoConfig, AutoProcessor

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from sampling import apply_repetition_penalty, sample_token

DEFAULT_MODEL = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_ONNX_ROOT = "./onnx_isolated"
SESSION_IO_BINDING = {}


@dataclass(frozen=True)
class PipelineAudioChunk:
    """一段已经完成 chunk decoder 的流水线音频输出。

    start_frame/end_frame 使用的是 full_codes 坐标，也就是
    `ref_codes + generated_codes` 拼接后的 codec 帧下标。
    """

    audio: np.ndarray
    sample_rate: int
    start_frame: int
    end_frame: int
    generated_frames: int
    is_final: bool = False


def load_audio(path, target_sr=24000):
    audio, sr = librosa.load(path, sr=None, mono=True)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
    return audio


class Timer:
    """线程安全的分项计时器。

    推理链路被拆成很多 ONNX 子模型，单看总耗时很难优化。
    Timer 会按名字累计 count/total，最后能看到 talker_decode、
    code_predictor、tokenizer_decode 等具体瓶颈。
    """

    def __init__(self):
        self.items = OrderedDict()
        self._lock = threading.Lock()

    def add(self, name, seconds):
        with self._lock:
            stat = self.items.setdefault(name, {"count": 0, "total": 0.0})
            stat["count"] += 1
            stat["total"] += seconds

    def measure(self, name):
        return TimerScope(self, name)

    def print_summary(self, title="[Timing]"):
        with self._lock:
            items = list(self.items.items())
        if not items:
            return
        print(f"\n{title}")
        for name, stat in items:
            count = stat["count"]
            total = stat["total"]
            avg = total / count if count else 0.0
            if count == 1:
                print(f"  {name}: {total * 1000:.2f} ms")
            else:
                print(f"  {name}: total={total * 1000:.2f} ms, count={count}, avg={avg * 1000:.2f} ms")

    def snapshot(self):
        with self._lock:
            items = list(self.items.items())
        rows = []
        for name, stat in items:
            count = int(stat["count"])
            total = float(stat["total"])
            rows.append(
                {
                    "name": name,
                    "count": count,
                    "total_ms": total * 1000.0,
                    "avg_ms": total * 1000.0 / count if count else 0.0,
                }
            )
        return rows

    def write_json(self, path, extra=None):
        payload = {
            "extra": extra or {},
            "items": self.snapshot(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


class TimerScope:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timer.add(self.name, time.perf_counter() - self.start)


def provider_uses_cuda(providers):
    return any(provider == "CUDAExecutionProvider" for provider in providers)


def is_ortvalue(value):
    return isinstance(value, ort.OrtValue)


def to_numpy_for_debug(value):
    return value.numpy() if is_ortvalue(value) else value


def make_session(path, providers, timer=None, name=None, use_iobinding=None):
    """创建 ONNX Runtime session，并记录是否对这个 session 启用 I/O Binding。"""
    label = f"session_load.{name or Path(path).stem}"
    start = time.perf_counter()
    session = ort.InferenceSession(str(path), providers=providers)
    if use_iobinding is None:
        use_iobinding = provider_uses_cuda(providers)
    SESSION_IO_BINDING[id(session)] = bool(use_iobinding)
    if timer is not None:
        timer.add(label, time.perf_counter() - start)
    return session


# ONNX Runtime 的 Python API 不会自动把 numpy 输入转成模型声明的类型。
# FP16 模型尤其容易踩坑：Python 侧很多中间张量默认是 float32，
# 如果直接喂给 tensor(float16) 输入，会报类型不匹配；如果手写每个输入
# 的 astype，又很容易遗漏。这里集中维护 ORT 类型字符串到 numpy dtype
# 的映射，run_session() 会在真正 session.run() 前统一处理。
ORT_INPUT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def cast_feed_for_session(session, feed):
    """按 ONNX 模型真实输入类型自动 cast feed。

    这个函数的目的不是“强行把所有输入变成 FP16”，而是读取
       session.get_inputs() 里每个输入的声明类型：

    - 模型输入是 tensor(float16)，就把对应 numpy 数组转成 np.float16；
    - 模型输入是 tensor(float)，就保持/转成 np.float32；
    - int64/int32 也同理。

    这样同一套 Python 推理代码可以同时跑 FP32 ONNX 和 FP16 ONNX，
    不需要在业务逻辑里到处写 dtype 判断。
    """
    casted = {}
    input_types = {item.name: item.type for item in session.get_inputs()}
    for name, value in feed.items():
        array = np.asarray(value)
        dtype = ORT_INPUT_DTYPES.get(input_types.get(name))
        casted[name] = array.astype(dtype, copy=False) if dtype is not None else array
    return casted


def run_session(session, output_names, feed, timer=None, name=None):
    """统一的 ORT session.run 包装。

    这里做两件事：
    1. 调用 cast_feed_for_session()，保证输入 dtype 和 ONNX 声明一致；
    2. CUDA session 默认走 Python I/O Binding，减少普通 run() 的隐式输出分配路径；
    3. 记录每个子模型的耗时，方便后面定位瓶颈，比如 talker_decode、
       tokenizer_decode、code_predictor 等。

    当前这一版 I/O Binding 仍会把输出 copy 回 numpy，因为采样、拼接 prompt、
    Gradio/写 wav 都还在 Python 侧处理。它是低风险的第一步；如果要进一步
    提速，需要把 KV cache / hidden state 保持为 CUDA OrtValue 并重写生成循环。
    """
    feed = cast_feed_for_session(session, feed)
    output_names_for_binding = output_names
    if output_names_for_binding is None:
        output_names_for_binding = [item.name for item in session.get_outputs()]
    elif isinstance(output_names_for_binding, str):
        output_names_for_binding = [output_names_for_binding]

    start = time.perf_counter()
    if SESSION_IO_BINDING.get(id(session), False):
        binding = session.io_binding()
        for input_name, input_value in feed.items():
            binding.bind_cpu_input(input_name, input_value)
        for output_name in output_names_for_binding:
            binding.bind_output(output_name, device_type="cuda", device_id=0)
        session.run_with_iobinding(binding)
        outputs = binding.copy_outputs_to_cpu()
    else:
        outputs = session.run(output_names, feed)
    if timer is not None:
        timer.add(f"onnx.{name or 'session'}", time.perf_counter() - start)
    return outputs


class Qwen3TTSVoiceCloneORT:
    """Python ONNX Runtime 版声音克隆主类。

    它和 C++ `VoiceCloneRuntime` 的职责一致：加载所有 ONNX 子模型，
    准备参考音频/文本条件，执行 talker/code_predictor 自回归生成，
    最后调用 tokenizer decoder 合成音频。
    """

    def __init__(
        self,
        model_dir,
        onnx_root,
        providers=None,
        seed=1234,
        print_timing=True,
        use_iobinding=True,
        load_reference_frontend=True,
    ):
        self.model_dir = Path(model_dir)
        self.onnx_root = Path(onnx_root)
        self.providers = providers or ["CPUExecutionProvider"]
        self.prep_providers = (
            ["CPUExecutionProvider"] if "CUDAExecutionProvider" in self.providers else self.providers
        )
        self.rng = np.random.default_rng(seed)
        self.print_timing = print_timing
        self.use_iobinding = bool(use_iobinding)
        self.timer = Timer()
        self.reference_cache = {}
        self.reference_text_cache = {}
        self.reference_code_embedding_cache = {}

        with self.timer.measure("init.load_configs"):
            with open(self.model_dir / "config.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
            with open(self.model_dir / "generation_config.json", "r", encoding="utf-8") as f:
                self.generation_config = json.load(f)
            with open(self.model_dir / "speech_tokenizer" / "config.json", "r", encoding="utf-8") as f:
                self.speech_tokenizer_config = json.load(f)

        with self.timer.measure("init.load_processor"):
            AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
            AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
            self.processor = AutoProcessor.from_pretrained(str(self.model_dir), fix_mistral_regex=True)

        # Qwen3-TTS 的 ONNX 被拆成多个子图：准备阶段的小 embedding/encoder
        # 可以放 CPU，真正高频调用的 talker/code_predictor/tokenizer decoder
        # 放用户指定 provider。这样能减少 GPU 初始化和显存压力。
        prep_iobinding = self.use_iobinding and provider_uses_cuda(self.prep_providers)
        run_iobinding = self.use_iobinding and provider_uses_cuda(self.providers)
        self.text_project = make_session(self.onnx_root / "text_project" / "text_project.onnx", self.prep_providers, self.timer, "text_project", prep_iobinding)
        self.codec_embed = make_session(self.onnx_root / "codec_embed" / "codec_embed.onnx", self.prep_providers, self.timer, "codec_embed", prep_iobinding)
        self.code_predictor_embed = make_session(
            self.onnx_root / "code_predictor_embed" / "code_predictor_embed.onnx",
            self.prep_providers,
            self.timer,
            "code_predictor_embed",
            prep_iobinding,
        )
        self.speaker_encoder = None
        self.tokenizer_encode = None
        if load_reference_frontend:
            self.speaker_encoder = make_session(self.onnx_root / "speaker_encoder" / "speaker_encoder.onnx", self.prep_providers, self.timer, "speaker_encoder", prep_iobinding)
            self.tokenizer_encode = make_session(self.onnx_root / "tokenizer12hz" / "tokenizer12hz_encode.onnx", self.prep_providers, self.timer, "tokenizer_encode", prep_iobinding)
        self.tokenizer_decode = make_session(self.onnx_root / "tokenizer12hz" / "tokenizer12hz_decode.onnx", self.providers, self.timer, "tokenizer_decode", run_iobinding)
        # chunk decoder 是可选实验能力，默认非流式路径不加载它。
        self.tokenizer_decode_chunk = None
        self.code_predictor = make_session(self.onnx_root / "code_predictor" / "code_predictor.onnx", self.providers, self.timer, "code_predictor", run_iobinding)
        self.code_predictor_input_types = {item.name: item.type for item in self.code_predictor.get_inputs()}
        self.talker_prefill = make_session(self.onnx_root / "talker_prefill" / "talker_prefill.onnx", self.providers, self.timer, "talker_prefill", run_iobinding)
        self.talker_decode = make_session(self.onnx_root / "talker_decode" / "talker_decode.onnx", self.providers, self.timer, "talker_decode", run_iobinding)
        self.talker_decode_input_types = {item.name: item.type for item in self.talker_decode.get_inputs()}
        self.talker_decode_output_names = [item.name for item in self.talker_decode.get_outputs()]

        self.talker_cfg = self.config["talker_config"]
        self.num_layers = int(self.talker_cfg["num_hidden_layers"])
        self.num_code_groups = int(self.talker_cfg["num_code_groups"])
        self.hidden_size = int(self.talker_cfg["hidden_size"])
        self.vocab_size = int(self.talker_cfg["vocab_size"])
        self.first_codebook_mask_tail = int(self.talker_cfg.get("first_codebook_mask_tail", 1024))
        self.codec_eos = int(self.talker_cfg["codec_eos_token_id"])

    def use_cuda_kv_cache(self):
        return self.use_iobinding and provider_uses_cuda(self.providers)

    def to_cuda_ortvalue(self, array):
        """把 numpy tensor 上传为 CUDA OrtValue，用于 talker KV cache 复用。"""
        if is_ortvalue(array):
            return array
        return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(array), "cuda", 0)

    def prepare_talker_past(self, past):
        """把 prefill 产出的 past cache 转成 CUDA OrtValue。

        prefill 输出目前仍回到 numpy；这里只做一次上传。之后每个
        talker_decode step 会直接复用上一轮输出的 CUDA OrtValue，避免
        KV cache 每步 GPU -> CPU -> GPU 来回搬。
        """
        if not self.use_cuda_kv_cache():
            return list(past)
        return [self.to_cuda_ortvalue(item) for item in past]

    def run_talker_decode_step(self, feeds):
        """运行一个 talker_decode step，并尽量让 KV cache 留在 GPU。

        返回值保持和普通 `session.run(None, feeds)` 兼容：
        outputs[0] logits 和 outputs[1] last_hidden 是 numpy，后续采样逻辑继续
        使用它们；outputs[2:] 在 CUDA I/O Binding 模式下是 CUDA OrtValue，
        下一步会直接作为 past_key/value 绑定回 talker_decode。
        """
        if not self.use_cuda_kv_cache():
            return run_session(self.talker_decode, None, feeds, self.timer, "talker_decode")

        start = time.perf_counter()
        binding = self.talker_decode.io_binding()
        for input_name, input_value in feeds.items():
            if is_ortvalue(input_value):
                binding.bind_ortvalue_input(input_name, input_value)
                continue
            array = np.asarray(input_value)
            dtype = ORT_INPUT_DTYPES.get(self.talker_decode_input_types.get(input_name))
            if dtype is not None:
                array = array.astype(dtype, copy=False)
            binding.bind_cpu_input(input_name, np.ascontiguousarray(array))

        for output_name in self.talker_decode_output_names:
            binding.bind_output(output_name, device_type="cuda", device_id=0)

        self.talker_decode.run_with_iobinding(binding)
        output_values = binding.get_outputs()
        outputs = [output_values[0].numpy(), output_values[1].numpy()]
        outputs.extend(output_values[2:])
        self.timer.add("onnx.talker_decode", time.perf_counter() - start)
        return outputs

    def run_code_predictor_logits(self, context, gen_step):
        """运行 code_predictor 的单步 logits。

        这个函数是 code predictor 热路径的专用包装。它避免每个 residual
        codebook step 都重新查询 session 元数据，并把 I/O Binding 的细节
        收在一个地方。当前 logits 仍返回 numpy，因为采样逻辑在 Python/NumPy。
        """
        feeds = {
            "context": context,
            "gen_step": np.asarray(gen_step),
        }
        if not (self.use_iobinding and provider_uses_cuda(self.providers)):
            return run_session(self.code_predictor, ["logits"], feeds, self.timer, "code_predictor")[0]

        start = time.perf_counter()
        binding = self.code_predictor.io_binding()
        for input_name, input_value in feeds.items():
            array = np.asarray(input_value)
            dtype = ORT_INPUT_DTYPES.get(self.code_predictor_input_types.get(input_name))
            if dtype is not None:
                array = array.astype(dtype, copy=False)
            binding.bind_cpu_input(input_name, np.ascontiguousarray(array))
        binding.bind_output("logits", device_type="cuda", device_id=0)
        self.code_predictor.run_with_iobinding(binding)
        logits = binding.get_outputs()[0].numpy()
        self.timer.add("onnx.code_predictor", time.perf_counter() - start)
        return logits

    def load_chunk_decoder(self):
        """懒加载 chunk decoder，避免影响默认非流式路径。"""
        if self.tokenizer_decode_chunk is not None:
            return self.tokenizer_decode_chunk
        path = self.onnx_root / "tokenizer12hz" / "tokenizer12hz_decode_chunk.onnx"
        if not path.exists():
            raise FileNotFoundError(
                f"Chunk decoder not found: {path}. "
                "Export it explicitly with export_tokenizer12hz_onnx.py --export-chunk-decoder."
            )
        self.tokenizer_decode_chunk = make_session(
            path,
            self.providers,
            self.timer,
            "tokenizer_decode_chunk",
            self.use_iobinding and provider_uses_cuda(self.providers),
        )
        return self.tokenizer_decode_chunk

    def get_reference_features(self, ref_audio, x_vector_only_mode=False):
        """提取并缓存参考音频特征。

        同一个参考音频在 Gradio 里会被反复使用，缓存 ref_code 和
        speaker_embed 可以避免每次生成都重新跑 tokenizer encoder 和
        speaker encoder。
        """
        path = Path(ref_audio)
        try:
            stat = path.stat()
            cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, bool(x_vector_only_mode))
        except OSError:
            cache_key = (str(ref_audio), bool(x_vector_only_mode))
        cached = self.reference_cache.get(cache_key)
        if cached is not None:
            self.timer.add("prep.reference_audio_cache_hit", 0.0)
            return cached

        with self.timer.measure("prep.reference_audio_cache_build"):
            self.timer.add("prep.reference_audio_cache_miss", 0.0)
            with self.timer.measure("prep.load_audio"):
                audio = load_audio(ref_audio, target_sr=24000)
            with self.timer.measure("prep.encode_ref_codes"):
                ref_code = None if x_vector_only_mode else self.encode_ref_codes(audio)
            with self.timer.measure("prep.extract_speaker_embedding"):
                speaker_embed = self.extract_speaker_embedding(audio)
        cached = (audio, ref_code, speaker_embed)
        self.reference_cache[cache_key] = cached
        return cached

    def text_ids(self, text):
        with self.timer.measure("prep.text_ids"):
            item = self.processor(text=text, return_tensors="pt", padding=True)
            ids = item["input_ids"]
            ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
            return ids.cpu().numpy().astype(np.int64)

    def text_embed(self, ids):
        return run_session(self.text_project, ["text_embed"], {"input_ids": ids}, self.timer, "text_project")[0]

    def reference_text_ids(self, ref_text):
        """缓存参考文本 token ids。

        参考文本通常和参考音频一起反复复用，缓存它可以避免每次声音克隆
        都重新走 processor。特殊 token embedding 不在这里缓存。
        """
        if not ref_text:
            return None
        cached = self.reference_text_cache.get(ref_text)
        if cached is not None:
            self.timer.add("prep.reference_text_cache_hit", 0.0)
            return cached
        with self.timer.measure("prep.reference_text_cache_build"):
            self.timer.add("prep.reference_text_cache_miss", 0.0)
            cached = self.text_ids(self.build_ref_text(ref_text))
        self.reference_text_cache[ref_text] = cached
        return cached

    def codec_embedding(self, token_ids):
        return run_session(self.codec_embed, ["embed"], {"token_ids": token_ids}, self.timer, "codec_embed")[0]

    def code_predictor_embedding(self, token_ids, layer_idx):
        return run_session(
            self.code_predictor_embed,
            ["embed"],
            {"token_id": token_ids, "layer_idx": np.asarray(layer_idx)},
            self.timer,
            "code_predictor_embed",
        )[0]

    def encode_ref_codes(self, audio_24k):
        codes = run_session(self.tokenizer_encode, ["codes"], {"audio": audio_24k[None, :]}, self.timer, "tokenizer_encode")[0]
        if codes.ndim == 3:
            codes = codes[0]
        return codes.astype(np.int64)

    def extract_speaker_embedding(self, audio_24k):
        with self.timer.measure("prep.mel_spectrogram"):
            wav = torch.from_numpy(audio_24k.astype(np.float32)).unsqueeze(0)
            mel = mel_spectrogram(
                wav,
                n_fft=1024,
                num_mels=128,
                sampling_rate=24000,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12000,
            ).transpose(1, 2)
        spk = run_session(self.speaker_encoder, ["speaker_embedding"], {"mel": mel.numpy()}, self.timer, "speaker_encoder")[0]
        return spk.reshape(1, 1, self.hidden_size).astype(np.float32)

    @staticmethod
    def build_assistant_text(text):
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def build_ref_text(text):
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def language_prefill_ids(self, language):
        """生成 codec 侧的语言/think 控制 token。

        Qwen3-TTS 的 talker prompt 不只有文本 embedding，还会在 codec 侧
        放入语言控制 token、speaker embedding 和 codec BOS/PAD token。
        `auto` 使用 nothink 路径，让模型自己处理语言。
        """
        language = (language or "auto").lower()
        if language == "auto":
            return [
                int(self.talker_cfg["codec_nothink_id"]),
                int(self.talker_cfg["codec_think_bos_id"]),
                int(self.talker_cfg["codec_think_eos_id"]),
            ]
        lang_map = self.talker_cfg["codec_language_id"]
        if language not in lang_map:
            raise ValueError(f"Unsupported language={language!r}; supported={sorted(lang_map)} plus auto")
        return [
            int(self.talker_cfg["codec_think_id"]),
            int(self.talker_cfg["codec_think_bos_id"]),
            int(lang_map[language]),
            int(self.talker_cfg["codec_think_eos_id"]),
        ]

    def ref_code_embedding(self, ref_code):
        """把参考音频 codec codes 转成 talker 可消费的 embedding。

        一帧 codec 有 16 个 codebook：第 0 个走 codec_embedding，后面
        15 个 residual codebook 走 code_predictor_embedding，最后把
        16 路 embedding 相加，得到和 talker hidden size 对齐的一帧表示。
        """
        cache_key = None
        if ref_code is not None:
            contiguous = np.ascontiguousarray(ref_code.astype(np.int64, copy=False))
            cache_key = (contiguous.shape, contiguous.tobytes())
            cached = self.reference_code_embedding_cache.get(cache_key)
            if cached is not None:
                self.timer.add("prep.ref_code_embedding_cache_hit", 0.0)
                return cached
            ref_code = contiguous
        pieces = []
        for i in range(self.num_code_groups):
            ids = ref_code[:, i][None, :].astype(np.int64)
            if i == 0:
                pieces.append(self.codec_embedding(ids))
            else:
                pieces.append(self.code_predictor_embedding(ids, i - 1))
        summed = np.sum(np.stack(pieces, axis=0), axis=0).astype(np.float32)
        bos = self.codec_embedding(np.asarray([[self.talker_cfg["codec_bos_id"]]], dtype=np.int64))
        embedding = np.concatenate([bos, summed], axis=1)
        if cache_key is not None:
            self.reference_code_embedding_cache[cache_key] = embedding
        return embedding

    def build_icl_prompt(self, input_ids, ref_ids, ref_code, tts_pad_embed, tts_eos_embed):
        """构造 in-context learning prompt 的参考音频/文本部分。

        Base voice clone 使用“参考文本 + 参考 codec”的 ICL 方式让模型学到
        目标音色。文本和 codec 按时间步相加；如果文本比 codec 短，用
        tts_pad_embed 补齐，剩下的目标文本会作为 trailing 在生成循环里逐步注入。
        """
        text_cat = np.concatenate([ref_ids[:, 3:-2], input_ids[:, 3:-5]], axis=1).astype(np.int64)
        text_embed = self.text_embed(text_cat)
        text_embed = np.concatenate([text_embed, tts_eos_embed], axis=1)
        codec_embed = self.ref_code_embedding(ref_code)
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if text_lens > codec_lens:
            return (text_embed[:, :codec_lens] + codec_embed).astype(np.float32), text_embed[:, codec_lens:].astype(np.float32)
        pads = np.repeat(tts_pad_embed, codec_lens - text_lens, axis=1)
        text_embed = np.concatenate([text_embed, pads], axis=1)
        return (text_embed + codec_embed).astype(np.float32), tts_pad_embed.astype(np.float32)

    def build_talker_prompt(self, text, ref_text, ref_code, speaker_embed, language="auto"):
        """组装 talker_prefill 的 inputs_embeds。

        prompt 的核心结构是：
        role tokens + codec 控制/speaker 前缀 + 可选 ICL 参考段。
        Qwen3-TTS 生成时不是一次性把全部目标文本喂进去，而是把 prompt
        之后剩余的目标文本 embedding 保存为 trailing，在每个 codec step
        与上一帧 codec embedding 相加后再喂给 talker_decode。
        """
        input_ids = self.text_ids(self.build_assistant_text(text))
        ref_ids = self.reference_text_ids(ref_text) if ref_text else None

        special = self.text_embed(np.asarray([[self.config["tts_bos_token_id"], self.config["tts_eos_token_id"], self.config["tts_pad_token_id"]]], dtype=np.int64))
        tts_bos_embed = special[:, 0:1]
        tts_eos_embed = special[:, 1:2]
        tts_pad_embed = special[:, 2:3]

        codec_prefill = self.codec_embedding(np.asarray([self.language_prefill_ids(language)], dtype=np.int64))
        codec_tail = self.codec_embedding(np.asarray([[self.talker_cfg["codec_pad_id"], self.talker_cfg["codec_bos_id"]]], dtype=np.int64))
        codec_input = np.concatenate([codec_prefill, speaker_embed, codec_tail], axis=1).astype(np.float32)

        role = self.text_embed(input_ids[:, :3])
        left_pad = np.repeat(tts_pad_embed, codec_input.shape[1] - 2, axis=1)
        codec_part = np.concatenate([left_pad, tts_bos_embed], axis=1) + codec_input[:, :-1]
        talker_input = np.concatenate([role, codec_part], axis=1)

        if ref_code is not None and ref_ids is not None:
            icl_input, trailing = self.build_icl_prompt(input_ids, ref_ids, ref_code, tts_pad_embed, tts_eos_embed)
            talker_input = np.concatenate([talker_input, icl_input], axis=1)
        else:
            first_text = self.text_embed(input_ids[:, 3:4]) + codec_input[:, -1:]
            talker_input = np.concatenate([talker_input, first_text], axis=1)
            trailing = np.concatenate([self.text_embed(input_ids[:, 4:-5]), tts_eos_embed], axis=1)

        return talker_input.astype(np.float32), trailing.astype(np.float32), tts_pad_embed.astype(np.float32)

    def run_code_predictor(self, past_hidden, first_token, do_sample=True, top_k=50, top_p=1.0, temperature=0.9, frame_index=None, dump_dir=None):
        """根据 talker 生成的第 0 个 codebook token 补齐一整帧 codec。

        talker 只直接预测每帧的第一个 codec token。剩余 15 个 residual
        codebook token 由 code_predictor 逐个生成；每生成一个 token，就把
        它的 embedding 追加到 context 里，作为下一个 residual token 的条件。
        """
        with self.timer.measure("generation.code_predictor_frame"):
            tokens = [int(first_token)]
            main_embed = self.codec_embedding(np.asarray([[first_token]], dtype=np.int64))
            context = np.concatenate([past_hidden.astype(np.float32), main_embed.astype(np.float32)], axis=1)
            residual_embeds = []
            for gen_step in range(self.num_code_groups - 1):
                logits = self.run_code_predictor_logits(context, gen_step)
                if dump_dir is not None and frame_index is not None and frame_index < 8:
                    np.save(Path(dump_dir) / f"code_predictor_logits_f{frame_index}_s{gen_step}.npy", logits[0, -1].astype(np.float32))
                    np.save(Path(dump_dir) / f"code_predictor_context_f{frame_index}_s{gen_step}.npy", context.astype(np.float32))
                token = sample_token(logits[0, -1], self.rng, do_sample, top_k, top_p, temperature)
                if dump_dir is not None and frame_index is not None:
                    np.save(Path(dump_dir) / f"code_predictor_pick_f{frame_index}_s{gen_step}.npy", np.asarray([token], dtype=np.int64))
                tokens.append(token)
                emb = self.code_predictor_embedding(np.asarray([[token]], dtype=np.int64), gen_step)
                residual_embeds.append(emb)
                context = np.concatenate([context, emb.astype(np.float32)], axis=1)
            all_embeds = np.concatenate([main_embed] + residual_embeds, axis=1)
            frame_embed = np.sum(all_embeds, axis=1, keepdims=True).astype(np.float32)
            return np.asarray(tokens, dtype=np.int64), frame_embed

    def decode_codes_to_audio(self, codes):
        """一次性把完整 codec 序列解码成音频。

        这个路径用于普通非流式输出。注意：如果前面拼上了参考音频的
        ref_code，那么完整 decode 出来的音频里也会包含参考音频对应的
        前缀，调用方需要在后面 trim 掉。
        """
        audio, lengths = run_session(
            self.tokenizer_decode,
            ["audio_values", "lengths"],
            {"audio_codes": codes[None, :, :]},
            self.timer,
            "tokenizer_decode",
        )
        length = int(np.asarray(lengths).reshape(-1)[0]) if lengths is not None else audio.shape[-1]
        if audio.shape[-1] < length:
            raise RuntimeError(
                "tokenizer_decode output is shorter than its reported length: "
                f"audio_samples={audio.shape[-1]}, expected_samples={length}. "
                "The exported tokenizer12hz_decode.onnx is likely capped by its trace length."
            )
        return audio.reshape(-1)[:length].astype(np.float32), 24000

    def decode_codes_chunk_to_audio(
        self,
        full_codes,
        start_frame,
        end_frame,
        left_context_frames=25,
    ):
        """按 full_codes 的帧区间解码一个 chunk，并裁掉左上下文音频。

        Args:
            full_codes: shape [T, 16]，通常是 ref_codes + generated_codes。
            start_frame: 本次真正要输出的起始帧，基于 full_codes 坐标。
            end_frame: 本次真正要输出的结束帧，基于 full_codes 坐标。
            left_context_frames: 解码时额外带上的左上下文帧数。

        Returns:
            shape [samples] 的 float32 PCM，长度为
            (end_frame - start_frame) * decode_upsample_rate。

        注意：chunk decoder 输入会包含 start_frame 前面的 left context，
        但输出只保留 [start_frame, end_frame) 这段新音频。这样可以降低
        块边界不连续，同时避免重复播放上下文音频。
        """
        if end_frame <= start_frame:
            return np.zeros((0,), dtype=np.float32), 24000
        session = self.load_chunk_decoder()
        context = min(int(left_context_frames), int(start_frame))
        input_start = int(start_frame) - context
        codes_chunk = full_codes[input_start:int(end_frame)]
        expected_samples = (int(end_frame) - int(start_frame)) * int(
            self.speech_tokenizer_config.get("decode_upsample_rate", 1920)
        )
        audio, lengths = run_session(
            session,
            ["audio_values", "lengths"],
            {
                "audio_codes": codes_chunk[None, :, :],
                "context_frames": np.asarray(context, dtype=np.int64),
            },
            self.timer,
            "tokenizer_decode_chunk",
        )
        reported = int(np.asarray(lengths).reshape(-1)[0]) if lengths is not None else audio.shape[-1]
        if reported < expected_samples:
            raise RuntimeError(
                "tokenizer_decode_chunk reported fewer samples than expected: "
                f"reported={reported}, expected={expected_samples}, "
                f"frames={start_frame}:{end_frame}, context={context}"
            )
        if audio.shape[-1] < expected_samples:
            raise RuntimeError(
                "tokenizer_decode_chunk output is shorter than expected: "
                f"audio_samples={audio.shape[-1]}, expected_samples={expected_samples}. "
                "The exported tokenizer12hz_decode_chunk.onnx may be capped by its trace length."
            )
        return audio.reshape(-1)[:expected_samples].astype(np.float32), 24000

    def generate_voice_clone(
        self,
        text,
        ref_audio,
        ref_text,
        language="auto",
        max_new_tokens=300,
        x_vector_only_mode=False,
        do_sample=None,
        top_k=None,
        top_p=None,
        temperature=None,
        repetition_penalty=None,
        subtalker_dosample=None,
        subtalker_top_k=None,
        subtalker_top_p=None,
        subtalker_temperature=None,
        cancel_event=None,
        dump_dir=None,
        verbose=True,
    ):
        """完整非流式声音克隆。

        这是当前最稳定的基线：先生成全部 codec 帧，再一次性调用
        tokenizer12hz_decode.onnx。chunk/pipeline 逻辑不要改动这条路径，
        方便随时回归对比。
        """
        gen = self.generation_config
        do_sample = gen.get("do_sample", True) if do_sample is None else do_sample
        top_k = gen.get("top_k", 50) if top_k is None else top_k
        top_p = gen.get("top_p", 1.0) if top_p is None else top_p
        temperature = gen.get("temperature", 0.9) if temperature is None else temperature
        repetition_penalty = gen.get("repetition_penalty", 1.05) if repetition_penalty is None else repetition_penalty
        subtalker_dosample = gen.get("subtalker_dosample", True) if subtalker_dosample is None else subtalker_dosample
        subtalker_top_k = gen.get("subtalker_top_k", 50) if subtalker_top_k is None else subtalker_top_k
        subtalker_top_p = gen.get("subtalker_top_p", 1.0) if subtalker_top_p is None else subtalker_top_p
        subtalker_temperature = gen.get("subtalker_temperature", 0.9) if subtalker_temperature is None else subtalker_temperature

        with self.timer.measure("prep.initial_inputs"):
            audio, ref_code, speaker_embed = self.get_reference_features(ref_audio, x_vector_only_mode)
            input_ids = self.text_ids(self.build_assistant_text(text))
            ref_ids = self.text_ids(self.build_ref_text(ref_text)) if ref_text else None
        if dump_dir:
            with self.timer.measure("io.dump_inputs"):
                dump_dir = Path(dump_dir)
                dump_dir.mkdir(parents=True, exist_ok=True)
                np.save(dump_dir / "audio_24k.npy", audio[None, :].astype(np.float32))
                np.save(dump_dir / "assistant_text_ids.npy", input_ids.reshape(-1).astype(np.int64))
                if ref_ids is not None:
                    np.save(dump_dir / "reference_text_ids.npy", ref_ids.reshape(-1).astype(np.int64))
                if ref_code is not None:
                    np.save(dump_dir / "reference_codes.npy", ref_code.astype(np.int64))
                np.save(dump_dir / "speaker_embedding.npy", speaker_embed.astype(np.float32))
                wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
                mel = mel_spectrogram(
                    wav,
                    n_fft=1024,
                    num_mels=128,
                    sampling_rate=24000,
                    hop_size=256,
                    win_size=1024,
                    fmin=0,
                    fmax=12000,
                ).transpose(1, 2)
                np.save(dump_dir / "mel.npy", mel.numpy().astype(np.float32))
        with self.timer.measure("prep.build_talker_prompt"):
            prompt, trailing, tts_pad = self.build_talker_prompt(
                text=text,
                ref_text=ref_text,
                ref_code=ref_code,
                speaker_embed=speaker_embed,
                language=language,
            )

        if verbose:
            print(f"prompt_len={prompt.shape[1]}, trailing_len={trailing.shape[1]}, ref_code_len={0 if ref_code is None else ref_code.shape[0]}")

        prefill_outputs = run_session(
            self.talker_prefill,
            None,
            {"inputs_embeds": prompt, "attention_mask": np.ones((1, prompt.shape[1]), dtype=np.int64)},
            self.timer,
            "talker_prefill",
        )
        logits = prefill_outputs[0]
        last_hidden = prefill_outputs[1][:, -1:, :].astype(np.float32)
        if dump_dir:
            np.save(dump_dir / "prompt.npy", prompt.astype(np.float32))
            np.save(dump_dir / "prefill_logits_last.npy", logits[0, -1].astype(np.float32))
            np.save(dump_dir / "prefill_last_hidden_full.npy", prefill_outputs[1].astype(np.float32))
        past = self.prepare_talker_past(prefill_outputs[2:])
        past_len = prompt.shape[1]
        generated_first = []
        generated_codes = []

        # 对齐原始 Qwen3-TTS wrapper 使用的 Transformers generate() 语义：
        # 一个生成状态不会产出 codec 行，所以公开的 max_new_tokens 上限
        # 在 talker_codes_list 中最多对应 max_new_tokens - 1 个 codec 帧。
        max_codec_frames = max(int(max_new_tokens) - 1, 1)
        trace_tokens = os.getenv("QWEN_TRACE_TOKENS") is not None
        with self.timer.measure("generation.decode_loop_total"):
            hit_eos = False
            for step in range(max_codec_frames):
                with self.timer.measure("generation.frame_total"):
                    if cancel_event is not None and cancel_event.is_set():
                        if verbose:
                            print(f"generation cancelled at step={step}")
                        break
                    next_logits = logits[0, -1].astype(np.float64)
                    # 对齐官方 Qwen3-TTS generate()：屏蔽 talker vocab 尾部控制/保留区间，
                    # 非当前 codec 采样范围，只保留有效 codec token 和 EOS。
                    next_logits[self.vocab_size - self.first_codebook_mask_tail :] = -np.inf
                    next_logits[self.codec_eos] = logits[0, -1, self.codec_eos]
                    next_logits = apply_repetition_penalty(next_logits, generated_first, repetition_penalty)
                    if dump_dir:
                        np.save(dump_dir / f"first_token_logits_f{step}.npy", next_logits.astype(np.float32))
                    with self.timer.measure("generation.sample_first_token"):
                        first = sample_token(next_logits, self.rng, do_sample, top_k, top_p, temperature)
                    if trace_tokens:
                        eos_logit = float(next_logits[self.codec_eos])
                        print(
                            f"[trace] frame={step} first={int(first)} eos_id={self.codec_eos} "
                            f"eos_logit={eos_logit:.6f} hit_eos={1 if first == self.codec_eos else 0}"
                        )
                    if dump_dir:
                        np.save(dump_dir / f"first_token_pick_f{step}.npy", np.asarray([first], dtype=np.int64))
                    if first == self.codec_eos:
                        hit_eos = True
                        if verbose:
                            print(f"hit eos at step={step}")
                        break

                    code_row, frame_embed = self.run_code_predictor(
                        last_hidden,
                        first,
                        do_sample=subtalker_dosample,
                        top_k=subtalker_top_k,
                        top_p=subtalker_top_p,
                        temperature=subtalker_temperature,
                        frame_index=step,
                        dump_dir=dump_dir,
                    )
                    generated_first.append(first)
                    generated_codes.append(code_row)

                    if step < trailing.shape[1]:
                        # 每生成一帧 codec，就消耗一个 trailing 文本 embedding。
                        # 文本耗尽后用 tts_pad 维持 codec 自回归继续。
                        decode_embed = frame_embed + trailing[:, step : step + 1]
                    else:
                        decode_embed = frame_embed + tts_pad
                    if dump_dir and step == 0:
                        with self.timer.measure("io.dump_decode0"):
                            np.save(dump_dir / "decode0_inputs_embeds.npy", decode_embed.astype(np.float32))
                            np.save(dump_dir / "decode0_attention_mask.npy", np.ones((1, past_len + 2), dtype=np.int64))
                            np.save(dump_dir / "decode0_cache_position.npy", np.asarray([past_len], dtype=np.int64))
                            for i in range(self.num_layers):
                                np.save(dump_dir / f"prefill_past_key_{i}.npy", to_numpy_for_debug(past[2 * i]).astype(np.float32))
                                np.save(dump_dir / f"prefill_past_value_{i}.npy", to_numpy_for_debug(past[2 * i + 1]).astype(np.float32))
                            np.save(dump_dir / "decode0_past_key_0.npy", to_numpy_for_debug(past[0]).astype(np.float32))
                            np.save(dump_dir / "decode0_past_value_0.npy", to_numpy_for_debug(past[1]).astype(np.float32))

                    with self.timer.measure("generation.prepare_decode_feed"):
                        feeds = {
                            "inputs_embeds": decode_embed.astype(np.float32),
                            "attention_mask": np.ones((1, past_len + 2), dtype=np.int64),
                            "cache_position": np.asarray([past_len], dtype=np.int64),
                        }
                        for i in range(self.num_layers):
                            feeds[f"past_key_{i}"] = past[2 * i] if is_ortvalue(past[2 * i]) else past[2 * i].astype(np.float32)
                            feeds[f"past_value_{i}"] = past[2 * i + 1] if is_ortvalue(past[2 * i + 1]) else past[2 * i + 1].astype(np.float32)
                    dec_outputs = self.run_talker_decode_step(feeds)
                    logits = dec_outputs[0]
                    last_hidden = dec_outputs[1][:, -1:, :].astype(np.float32)
                    if dump_dir and step == 0:
                        with self.timer.measure("io.dump_decode0_outputs"):
                            np.save(dump_dir / "decode0_logits_last.npy", logits[0, -1].astype(np.float32))
                            np.save(dump_dir / "decode0_last_hidden.npy", dec_outputs[1].astype(np.float32))
                    past = list(dec_outputs[2:])
                    past_len += 1

                    if verbose and (step + 1) % 20 == 0:
                        print(f"generated_frames={step + 1}")
            if not hit_eos and len(generated_codes) >= max_codec_frames and verbose:
                print(
                    "[WARN] generation reached max_new_tokens before EOS; "
                    f"text may be truncated. max_new_tokens={max_new_tokens}, "
                    f"generated_codec_frames={len(generated_codes)}"
                )

        if not generated_codes:
            raise RuntimeError("No codec frames were generated before EOS")
        with self.timer.measure("post.stack_codes"):
            codes = np.stack(generated_codes, axis=0).astype(np.int64)
        if dump_dir:
            with self.timer.measure("io.dump_generated_codes"):
                np.save(dump_dir / "generated_codes.npy", codes)
        with self.timer.measure("post.prepare_vocoder_codes"):
            if ref_code is not None:
                # ICL 模式需要把参考 codec 拼在生成 codec 前面一起 decode，
                # 后面再把参考音频对应的前缀裁掉。
                decode_codes = np.concatenate([ref_code, codes], axis=0)
            else:
                decode_codes = codes
        with self.timer.measure("post.decode_codes_to_audio"):
            wav, sr = self.decode_codes_to_audio(decode_codes)
        with self.timer.measure("post.trim_reference_audio"):
            if ref_code is not None:
                # 完整 decoder 的输出长度由 ONNX 实际输出决定，按帧数比例
                # 裁掉参考段，避免 trace/padding 细节导致 off-by-one。
                cut = int(ref_code.shape[0] / max(decode_codes.shape[0], 1) * wav.shape[0])
                wav = wav[cut:]
        if dump_dir:
            with self.timer.measure("io.dump_waveform"):
                np.save(dump_dir / "waveform.npy", wav.astype(np.float32))
        return wav.astype(np.float32), sr, codes

    def iter_voice_clone_chunked(
        self,
        text,
        ref_audio,
        ref_text,
        language="auto",
        max_new_tokens=300,
        chunk_frames=50,
        left_context_frames=25,
        x_vector_only_mode=False,
        do_sample=None,
        top_k=None,
        top_p=None,
        temperature=None,
        repetition_penalty=None,
        subtalker_dosample=None,
        subtalker_top_k=None,
        subtalker_top_p=None,
        subtalker_temperature=None,
        cancel_event=None,
        dump_dir=None,
        verbose=True,
    ):
        """同步流水线版本：生成一批 codec 后立刻 chunk decode 并 yield 音频。

        这个 API 保留“边产边消费”的形状，后续可以直接在外层接队列、
        Gradio generator 或 C++ 双线程实现。默认非流式 generate_voice_clone()
        不调用这里。

        当前实现仍在同一个线程里执行：生成 chunk_frames 帧 codec 后马上
        decode 一段音频。它不是最终的真并行流水线，但接口已经按生产者/
        消费者模型设计，后面可以把 codec 生成和 chunk decode 分到两个线程。
        """
        chunk_frames = int(chunk_frames)
        left_context_frames = int(left_context_frames)
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if left_context_frames < 0:
            raise ValueError("left_context_frames must be non-negative")

        gen = self.generation_config
        do_sample = gen.get("do_sample", True) if do_sample is None else do_sample
        top_k = gen.get("top_k", 50) if top_k is None else top_k
        top_p = gen.get("top_p", 1.0) if top_p is None else top_p
        temperature = gen.get("temperature", 0.9) if temperature is None else temperature
        repetition_penalty = gen.get("repetition_penalty", 1.05) if repetition_penalty is None else repetition_penalty
        subtalker_dosample = gen.get("subtalker_dosample", True) if subtalker_dosample is None else subtalker_dosample
        subtalker_top_k = gen.get("subtalker_top_k", 50) if subtalker_top_k is None else subtalker_top_k
        subtalker_top_p = gen.get("subtalker_top_p", 1.0) if subtalker_top_p is None else subtalker_top_p
        subtalker_temperature = gen.get("subtalker_temperature", 0.9) if subtalker_temperature is None else subtalker_temperature

        # Fail early if the optional ONNX file is missing.
        self.load_chunk_decoder()

        with self.timer.measure("prep.initial_inputs"):
            audio, ref_code, speaker_embed = self.get_reference_features(ref_audio, x_vector_only_mode)
            input_ids = self.text_ids(self.build_assistant_text(text))
            ref_ids = self.text_ids(self.build_ref_text(ref_text)) if ref_text else None
        if dump_dir:
            dump_dir = Path(dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            np.save(dump_dir / "audio_24k.npy", audio[None, :].astype(np.float32))
            np.save(dump_dir / "assistant_text_ids.npy", input_ids.reshape(-1).astype(np.int64))
            if ref_ids is not None:
                np.save(dump_dir / "reference_text_ids.npy", ref_ids.reshape(-1).astype(np.int64))
            if ref_code is not None:
                np.save(dump_dir / "reference_codes.npy", ref_code.astype(np.int64))
            np.save(dump_dir / "speaker_embedding.npy", speaker_embed.astype(np.float32))
            wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            mel = mel_spectrogram(
                wav,
                n_fft=1024,
                num_mels=128,
                sampling_rate=24000,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12000,
            ).transpose(1, 2)
            np.save(dump_dir / "mel.npy", mel.numpy().astype(np.float32))
        with self.timer.measure("prep.build_talker_prompt"):
            prompt, trailing, tts_pad = self.build_talker_prompt(
                text=text,
                ref_text=ref_text,
                ref_code=ref_code,
                speaker_embed=speaker_embed,
                language=language,
            )

        ref_len = 0 if ref_code is None else int(ref_code.shape[0])
        # full_codes 坐标中，参考音频部分不应该输出给用户，所以第一段
        # 新音频从 ref_len 开始 decode/yield。
        next_decode_start = ref_len
        if verbose:
            print(
                f"prompt_len={prompt.shape[1]}, trailing_len={trailing.shape[1]}, "
                f"ref_code_len={ref_len}, chunk_frames={chunk_frames}, "
                f"left_context_frames={left_context_frames}"
            )

        prefill_outputs = run_session(
            self.talker_prefill,
            None,
            {"inputs_embeds": prompt, "attention_mask": np.ones((1, prompt.shape[1]), dtype=np.int64)},
            self.timer,
            "talker_prefill",
        )
        logits = prefill_outputs[0]
        last_hidden = prefill_outputs[1][:, -1:, :].astype(np.float32)
        if dump_dir:
            np.save(dump_dir / "prompt.npy", prompt.astype(np.float32))
            np.save(dump_dir / "prefill_logits_last.npy", logits[0, -1].astype(np.float32))
            np.save(dump_dir / "prefill_last_hidden_full.npy", prefill_outputs[1].astype(np.float32))
        past = self.prepare_talker_past(prefill_outputs[2:])
        if dump_dir:
            np.save(dump_dir / "prefill_past_key_0.npy", to_numpy_for_debug(past[0]).astype(np.float32))
            np.save(dump_dir / "prefill_past_value_0.npy", to_numpy_for_debug(past[1]).astype(np.float32))
        past_len = prompt.shape[1]
        generated_first = []
        generated_codes = []

        max_codec_frames = max(int(max_new_tokens) - 1, 1)
        trace_tokens = os.getenv("QWEN_TRACE_TOKENS") is not None
        hit_eos = False
        with self.timer.measure("generation.decode_loop_total"):
            for step in range(max_codec_frames):
                with self.timer.measure("generation.frame_total"):
                    if cancel_event is not None and cancel_event.is_set():
                        if verbose:
                            print(f"generation cancelled at step={step}")
                        break
                    next_logits = logits[0, -1].astype(np.float64)
                    next_logits[self.vocab_size - self.first_codebook_mask_tail :] = -np.inf
                    next_logits[self.codec_eos] = logits[0, -1, self.codec_eos]
                    next_logits = apply_repetition_penalty(next_logits, generated_first, repetition_penalty)
                    if dump_dir:
                        np.save(dump_dir / f"first_token_logits_f{step}.npy", next_logits.astype(np.float32))
                    with self.timer.measure("generation.sample_first_token"):
                        first = sample_token(next_logits, self.rng, do_sample, top_k, top_p, temperature)
                    if trace_tokens:
                        eos_logit = float(next_logits[self.codec_eos])
                        print(
                            f"[trace] frame={step} first={int(first)} eos_id={self.codec_eos} "
                            f"eos_logit={eos_logit:.6f} hit_eos={1 if first == self.codec_eos else 0}"
                        )
                    if dump_dir:
                        np.save(dump_dir / f"first_token_pick_f{step}.npy", np.asarray([first], dtype=np.int64))
                    if first == self.codec_eos:
                        hit_eos = True
                        if verbose:
                            print(f"hit eos at step={step}")
                        break

                    code_row, frame_embed = self.run_code_predictor(
                        last_hidden,
                        first,
                        do_sample=subtalker_dosample,
                        top_k=subtalker_top_k,
                        top_p=subtalker_top_p,
                        temperature=subtalker_temperature,
                        frame_index=step,
                        dump_dir=dump_dir,
                    )
                    generated_first.append(first)
                    generated_codes.append(code_row)

                    if step < trailing.shape[1]:
                        decode_embed = frame_embed + trailing[:, step : step + 1]
                    else:
                        decode_embed = frame_embed + tts_pad
                    if dump_dir and step == 0:
                        np.save(dump_dir / "decode0_inputs_embeds.npy", decode_embed.astype(np.float32))
                        np.save(dump_dir / "decode0_attention_mask.npy", np.ones((1, past_len + 2), dtype=np.int64))
                        np.save(dump_dir / "decode0_cache_position.npy", np.asarray([past_len], dtype=np.int64))
                        np.save(dump_dir / "decode0_past_key_0.npy", to_numpy_for_debug(past[0]).astype(np.float32))
                        np.save(dump_dir / "decode0_past_value_0.npy", to_numpy_for_debug(past[1]).astype(np.float32))

                    with self.timer.measure("generation.prepare_decode_feed"):
                        feeds = {
                            "inputs_embeds": decode_embed.astype(np.float32),
                            "attention_mask": np.ones((1, past_len + 2), dtype=np.int64),
                            "cache_position": np.asarray([past_len], dtype=np.int64),
                        }
                        for i in range(self.num_layers):
                            feeds[f"past_key_{i}"] = past[2 * i] if is_ortvalue(past[2 * i]) else past[2 * i].astype(np.float32)
                            feeds[f"past_value_{i}"] = past[2 * i + 1] if is_ortvalue(past[2 * i + 1]) else past[2 * i + 1].astype(np.float32)
                    dec_outputs = self.run_talker_decode_step(feeds)
                    logits = dec_outputs[0]
                    last_hidden = dec_outputs[1][:, -1:, :].astype(np.float32)
                    if dump_dir and step == 0:
                        np.save(dump_dir / "decode0_logits_last.npy", logits[0, -1].astype(np.float32))
                        np.save(dump_dir / "decode0_last_hidden.npy", dec_outputs[1].astype(np.float32))
                    past = list(dec_outputs[2:])
                    past_len += 1

                    available_end = ref_len + len(generated_codes)
                    while available_end - next_decode_start >= chunk_frames:
                        end_frame = next_decode_start + chunk_frames
                        with self.timer.measure("pipeline.prepare_chunk_codes"):
                            codes = np.stack(generated_codes, axis=0).astype(np.int64)
                            full_codes = np.concatenate([ref_code, codes], axis=0) if ref_code is not None else codes
                        with self.timer.measure("pipeline.decode_chunk"):
                            wav, sr = self.decode_codes_chunk_to_audio(
                                full_codes,
                                next_decode_start,
                                end_frame,
                                left_context_frames=left_context_frames,
                            )
                        if dump_dir:
                            chunk_index = len(list(dump_dir.glob("chunk_*_meta.npy")))
                            context = min(int(left_context_frames), int(next_decode_start))
                            input_start = int(next_decode_start) - context
                            np.save(dump_dir / f"chunk_{chunk_index}_full_codes.npy", full_codes.astype(np.int64))
                            np.save(dump_dir / f"chunk_{chunk_index}_input_codes.npy", full_codes[input_start:end_frame].astype(np.int64))
                            np.save(dump_dir / f"chunk_{chunk_index}_audio.npy", wav.astype(np.float32))
                            np.save(
                                dump_dir / f"chunk_{chunk_index}_meta.npy",
                                np.asarray([next_decode_start, end_frame, context, input_start, len(generated_codes), 0], dtype=np.int64),
                            )
                        if verbose:
                            print(
                                f"yield chunk frames={next_decode_start}:{end_frame} "
                                f"samples={wav.shape[0]}"
                            )
                        yield PipelineAudioChunk(
                            audio=wav,
                            sample_rate=sr,
                            start_frame=next_decode_start,
                            end_frame=end_frame,
                            generated_frames=len(generated_codes),
                            is_final=False,
                        )
                        next_decode_start = end_frame

                    if verbose and (step + 1) % 20 == 0:
                        print(f"generated_frames={step + 1}")

        if not hit_eos and len(generated_codes) >= max_codec_frames and verbose:
            print(
                "[WARN] generation reached max_new_tokens before EOS; "
                f"text may be truncated. max_new_tokens={max_new_tokens}, "
                f"generated_codec_frames={len(generated_codes)}"
            )
        if not generated_codes:
            raise RuntimeError("No codec frames were generated before EOS")
        codes = np.stack(generated_codes, axis=0).astype(np.int64)
        if dump_dir:
            np.save(dump_dir / "generated_codes.npy", codes)

        final_end = ref_len + len(generated_codes)
        if final_end > next_decode_start:
            with self.timer.measure("pipeline.prepare_final_chunk_codes"):
                full_codes = np.concatenate([ref_code, codes], axis=0) if ref_code is not None else codes
            with self.timer.measure("pipeline.decode_final_chunk"):
                wav, sr = self.decode_codes_chunk_to_audio(
                    full_codes,
                    next_decode_start,
                    final_end,
                    left_context_frames=left_context_frames,
                )
            if dump_dir:
                chunk_index = len(list(dump_dir.glob("chunk_*_meta.npy")))
                context = min(int(left_context_frames), int(next_decode_start))
                input_start = int(next_decode_start) - context
                np.save(dump_dir / f"chunk_{chunk_index}_full_codes.npy", full_codes.astype(np.int64))
                np.save(dump_dir / f"chunk_{chunk_index}_input_codes.npy", full_codes[input_start:final_end].astype(np.int64))
                np.save(dump_dir / f"chunk_{chunk_index}_audio.npy", wav.astype(np.float32))
                np.save(
                    dump_dir / f"chunk_{chunk_index}_meta.npy",
                    np.asarray([next_decode_start, final_end, context, input_start, len(generated_codes), 1], dtype=np.int64),
                )
            if verbose:
                print(
                    f"yield final chunk frames={next_decode_start}:{final_end} "
                    f"samples={wav.shape[0]}"
                )
            yield PipelineAudioChunk(
                audio=wav,
                sample_rate=sr,
                start_frame=next_decode_start,
                end_frame=final_end,
                generated_frames=len(generated_codes),
                is_final=True,
            )


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-TTS Base voice clone using exported ONNX Runtime models")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--onnx-root", default=DEFAULT_ONNX_ROOT)
    parser.add_argument("--text", default="我和我的祖国，一刻也不能分割，无论你走到哪里")
    parser.add_argument("--ref-audio", default="./data/ref_from_mp3_24k_mono.wav")
    parser.add_argument("--ref-text", default="告诉自己，不要怕")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--output", default="output_voice_clone_ort.wav")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling for the main talker and code predictor")
    parser.add_argument("--dump-dir", default=None, help="Write intermediate tensors as .npy for Python/C++ parity checks")
    parser.add_argument("--no-timing", action="store_true", help="Disable timing summary output")
    parser.add_argument("--timing-json", default=None, help="Write detailed timing records to a JSON file")
    parser.add_argument("--no-iobinding", action="store_true", help="Disable Python ONNX Runtime I/O Binding")
    args = parser.parse_args()

    providers = [args.provider]
    total_timer = Timer()
    with total_timer.measure("total.init_runner"):
        runner = Qwen3TTSVoiceCloneORT(
            args.model,
            args.onnx_root,
            providers=providers,
            seed=args.seed,
            print_timing=not args.no_timing,
            use_iobinding=not args.no_iobinding,
        )
    with total_timer.measure("total.generate_voice_clone"):
        wav, sr, codes = runner.generate_voice_clone(
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            subtalker_dosample=not args.greedy,
            dump_dir=args.dump_dir,
        )
    generated_frames = codes.shape[0]
    with total_timer.measure("total.write_wav"):
        sf.write(args.output, wav, sr)
    print(f"wrote {args.output}: samples={wav.shape[0]}, sr={sr}, generated_frames={generated_frames}")
    if not args.no_timing:
        total_timer.print_summary("[Timing] Overall")
        runner.timer.print_summary("[Timing] Detail")
    if args.timing_json:
        detail_path = Path(args.timing_json)
        total_timer.write_json(
            detail_path,
            extra={
                "kind": "voice_clone_full",
                "output": args.output,
                "generated_frames": int(generated_frames),
                "samples": int(wav.shape[0]),
                "sample_rate": int(sr),
            },
        )
        runner.timer.write_json(
            detail_path.with_suffix(".detail.json"),
            extra={
                "kind": "voice_clone_full_detail",
                "output": args.output,
                "generated_frames": int(generated_frames),
            },
        )


if __name__ == "__main__":
    main()
