"""Python ONNX Runtime demos 共用的采样工具。

Qwen3-TTS 的生成有两级采样：
1. talker 采样每个 codec 帧的第 0 个 codebook token；
2. code_predictor 采样剩余 15 个 residual codebook token。
这里的逻辑需要和 C++ `cpp/src/sampling.cc` 保持一致。
"""

from __future__ import annotations

import numpy as np


def softmax(x):
    """数值稳定 softmax：先减最大值，避免 exp 溢出。"""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def top_k_top_p_filter(logits, top_k=50, top_p=1.0):
    """按 top-k/top-p 把不允许采样的 token logit 置为 -inf。"""
    logits = logits.copy()
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    if top_p is not None and top_p < 1.0:
        # nucleus sampling：按概率从高到低累加，超过 top_p 的尾部候选被屏蔽。
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        probs = softmax(sorted_logits)
        cum = np.cumsum(probs)
        remove = cum > top_p
        if remove.size:
            remove[1:] = remove[:-1]
            remove[0] = False
            logits[order[remove]] = -np.inf
    return logits


def sample_token(logits, rng, do_sample=True, top_k=50, top_p=1.0, temperature=0.9):
    """从一维 logits 中采样一个 token id；do_sample=False 时为 greedy argmax。"""
    logits = logits.astype(np.float64).reshape(-1)
    if temperature is not None and temperature > 0:
        logits = logits / float(temperature)
    if not do_sample:
        return int(np.argmax(logits))
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = softmax(logits)
    return int(rng.choice(np.arange(logits.shape[0]), p=probs))


def apply_repetition_penalty(logits, generated, penalty):
    """对已经生成过的 token 应用 repetition penalty。"""
    if not generated or penalty is None or penalty == 1.0:
        return logits
    logits = logits.copy()
    for token in set(generated):
        if token < 0 or token >= logits.shape[-1]:
            continue
        if logits[token] < 0:
            logits[token] *= penalty
        else:
            logits[token] /= penalty
    return logits
