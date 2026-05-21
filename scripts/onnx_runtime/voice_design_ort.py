#!/usr/bin/env python3
"""Qwen3-TTS VoiceDesign ONNX Runtime inference.

VoiceDesign reuses the same talker/code-predictor/tokenizer decoder pipeline as
the Base and CustomVoice runtimes. The difference is the prompt: voice/style
conditioning comes from a user instruction text prepended as a user turn.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from sampling import apply_repetition_penalty, sample_token
from voice_clone_ort import Qwen3TTSVoiceCloneORT, run_session


DEFAULT_MODEL = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_ONNX_ROOT = "./onnx_voice_design_1p7b_fp16"


class Qwen3TTSVoiceDesignORT(Qwen3TTSVoiceCloneORT):
    """Single-sample VoiceDesign ONNX runtime."""

    def __init__(self, model_dir, onnx_root, providers=None, seed=1234, print_timing=True, use_iobinding=True):
        super().__init__(
            model_dir=model_dir,
            onnx_root=onnx_root,
            providers=providers,
            seed=seed,
            print_timing=print_timing,
            use_iobinding=use_iobinding,
            load_reference_frontend=False,
        )
        if self.config.get("tts_model_type") != "voice_design":
            raise ValueError(f"Expected voice_design model, got {self.config.get('tts_model_type')!r}")

    @staticmethod
    def build_instruct_text(instruct):
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def build_voice_design_prompt(self, text, instruct, language="auto", non_streaming_mode=True):
        input_ids = self.text_ids(self.build_assistant_text(text))

        special = self.text_embed(
            np.asarray(
                [[self.config["tts_bos_token_id"], self.config["tts_eos_token_id"], self.config["tts_pad_token_id"]]],
                dtype=np.int64,
            )
        )
        tts_bos_embed = special[:, 0:1]
        tts_eos_embed = special[:, 1:2]
        tts_pad_embed = special[:, 2:3]

        talker_prefix = []
        if instruct:
            instruct_ids = self.text_ids(self.build_instruct_text(instruct))
            talker_prefix.append(self.text_embed(instruct_ids))

        codec_prefill = self.codec_embedding(np.asarray([self.language_prefill_ids(language)], dtype=np.int64))
        codec_tail = self.codec_embedding(
            np.asarray([[self.talker_cfg["codec_pad_id"], self.talker_cfg["codec_bos_id"]]], dtype=np.int64)
        )
        codec_input = np.concatenate([codec_prefill, codec_tail], axis=1).astype(np.float32)

        role = self.text_embed(input_ids[:, :3])
        left_pad = np.repeat(tts_pad_embed, codec_input.shape[1] - 2, axis=1)
        codec_part = np.concatenate([left_pad, tts_bos_embed], axis=1) + codec_input[:, :-1]
        talker_input = np.concatenate([role, codec_part], axis=1)

        first_text = self.text_embed(input_ids[:, 3:4]) + codec_input[:, -1:]
        talker_input = np.concatenate([talker_input, first_text], axis=1)
        if non_streaming_mode:
            text_body = self.text_embed(input_ids[:, 3:-5])
            text_with_eos = np.concatenate([text_body, tts_eos_embed], axis=1)
            codec_pad = self.codec_embedding(
                np.asarray([[self.talker_cfg["codec_pad_id"]] * text_with_eos.shape[1]], dtype=np.int64)
            )
            codec_bos = self.codec_embedding(np.asarray([[self.talker_cfg["codec_bos_id"]]], dtype=np.int64))
            talker_input = np.concatenate(
                [
                    talker_input[:, :-1],
                    text_with_eos + codec_pad,
                    tts_pad_embed + codec_bos,
                ],
                axis=1,
            )
            trailing = tts_pad_embed
        else:
            trailing = np.concatenate([self.text_embed(input_ids[:, 4:-5]), tts_eos_embed], axis=1)

        if talker_prefix:
            talker_input = np.concatenate([*talker_prefix, talker_input], axis=1)
        return talker_input.astype(np.float32), trailing.astype(np.float32), tts_pad_embed.astype(np.float32)

    def generate_voice_design(
        self,
        text,
        instruct,
        language="auto",
        non_streaming_mode=True,
        max_new_tokens=None,
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
        gen = self.generation_config
        max_new_tokens = gen.get("max_new_tokens", 2048) if max_new_tokens is None else max_new_tokens
        do_sample = gen.get("do_sample", True) if do_sample is None else do_sample
        top_k = gen.get("top_k", 50) if top_k is None else top_k
        top_p = gen.get("top_p", 1.0) if top_p is None else top_p
        temperature = gen.get("temperature", 0.9) if temperature is None else temperature
        repetition_penalty = gen.get("repetition_penalty", 1.05) if repetition_penalty is None else repetition_penalty
        subtalker_dosample = gen.get("subtalker_dosample", True) if subtalker_dosample is None else subtalker_dosample
        subtalker_top_k = gen.get("subtalker_top_k", 50) if subtalker_top_k is None else subtalker_top_k
        subtalker_top_p = gen.get("subtalker_top_p", 1.0) if subtalker_top_p is None else subtalker_top_p
        subtalker_temperature = gen.get("subtalker_temperature", 0.9) if subtalker_temperature is None else subtalker_temperature

        with self.timer.measure("prep.build_voice_design_prompt"):
            prompt, trailing, tts_pad = self.build_voice_design_prompt(
                text=text,
                instruct=instruct,
                language=language,
                non_streaming_mode=non_streaming_mode,
            )
        if verbose:
            print(f"prompt_len={prompt.shape[1]}, trailing_len={trailing.shape[1]}, language={language}")
        if dump_dir:
            dump_dir = Path(dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            np.save(dump_dir / "prompt.npy", prompt.astype(np.float32))

        prefill_outputs = run_session(
            self.talker_prefill,
            None,
            {"inputs_embeds": prompt, "attention_mask": np.ones((1, prompt.shape[1]), dtype=np.int64)},
            self.timer,
            "talker_prefill",
        )
        logits = prefill_outputs[0]
        last_hidden = prefill_outputs[1][:, -1:, :].astype(np.float32)
        past = self.prepare_talker_past(prefill_outputs[2:])
        past_len = prompt.shape[1]
        generated_first = []
        generated_codes = []
        max_codec_frames = max(int(max_new_tokens) - 1, 1)

        for step in range(max_codec_frames):
            if cancel_event is not None and cancel_event.is_set():
                break
            next_logits = logits[0, -1].astype(np.float64)
            next_logits[self.vocab_size - self.first_codebook_mask_tail :] = -np.inf
            next_logits[self.codec_eos] = logits[0, -1, self.codec_eos]
            next_logits = apply_repetition_penalty(next_logits, generated_first, repetition_penalty)
            first = sample_token(next_logits, self.rng, do_sample, top_k, top_p, temperature)
            if first == self.codec_eos:
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
            decode_embed = frame_embed + (trailing[:, step : step + 1] if step < trailing.shape[1] else tts_pad)

            feeds = {
                "inputs_embeds": decode_embed.astype(np.float32),
                "attention_mask": np.ones((1, past_len + 2), dtype=np.int64),
                "cache_position": np.asarray([past_len], dtype=np.int64),
            }
            for i in range(self.num_layers):
                feeds[f"past_key_{i}"] = past[2 * i]
                feeds[f"past_value_{i}"] = past[2 * i + 1]
            dec_outputs = self.run_talker_decode_step(feeds)
            logits = dec_outputs[0]
            last_hidden = dec_outputs[1][:, -1:, :].astype(np.float32)
            past = list(dec_outputs[2:])
            past_len += 1
            if verbose and (step + 1) % 20 == 0:
                print(f"generated_frames={step + 1}")

        if not generated_codes:
            raise RuntimeError("No codec frames were generated before EOS")
        codes = np.stack(generated_codes, axis=0).astype(np.int64)
        wav, sr = self.decode_codes_to_audio(codes)
        if dump_dir:
            np.save(dump_dir / "generated_codes.npy", codes)
            np.save(dump_dir / "waveform.npy", wav.astype(np.float32))
        return wav.astype(np.float32), sr, codes


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-TTS VoiceDesign with ONNX Runtime")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--onnx-root", default=DEFAULT_ONNX_ROOT)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--text", default="其实我真的有发现，我是一个特别善于观察别人情绪的人。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default="一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--output", default="output_voice_design_ort.wav")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--dump-dir", default="")
    parser.add_argument("--no-iobinding", action="store_true")
    args = parser.parse_args()

    runner = Qwen3TTSVoiceDesignORT(
        model_dir=args.model,
        onnx_root=args.onnx_root,
        providers=[args.provider],
        seed=args.seed,
        use_iobinding=not args.no_iobinding,
    )
    wav, sr, codes = runner.generate_voice_design(
        text=args.text,
        instruct=args.instruct,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        do_sample=False if args.greedy else None,
        subtalker_dosample=False if args.greedy else None,
        dump_dir=args.dump_dir or None,
        verbose=True,
    )
    sf.write(args.output, wav, sr)
    print(f"wrote {args.output}: samples={wav.shape[0]}, sr={sr}, generated_frames={codes.shape[0]}")
    if runner.print_timing:
        runner.timer.print_summary()


if __name__ == "__main__":
    main()
