"""Device pipeline configs matching the validated Qwen3/Qwen3.5 optimization setup
(see benchmark/tests/Qwen3_Qwen3.5_优化说明与测试数据_统一版_20260711.md).

  - iGPU/GPU : ATTENTION_BACKEND=PA  (PagedAttention; Qwen3.5 hybrid Gated-DeltaNet
               path. opt-3/5/6 on top are byte-exact lossless.)
  - NPU      : NPU_TURBO=YES (byte-exact lossless, -4~7% TTFT) + the NPUW baseline
               (MAX_PROMPT_LEN / MIN_RESPONSE_LEN / prefill chunk). We deliberately do
               NOT set NPUW_LLM_GENERATE_HINT=BEST_PERF: it changes outputs on 0.6B/1.7B.

These are performance/back-end switches proven md5-lossless, so they should not change
accuracy; the NPU length params also make the static NPU pipeline behave correctly.
"""
from __future__ import annotations


def _coerce(v: str):
    s = str(v)
    if s.lstrip("-").isdigit():
        return int(s)
    return s


def build_pipe_config(device: str, extra=None, auto: bool = True) -> dict:
    cfg: dict = {}
    dev = (device or "").upper()
    if auto:
        if dev.startswith("NPU"):
            cfg.update({
                "NPU_TURBO": "YES",
                "MAX_PROMPT_LEN": 1024,
                "MIN_RESPONSE_LEN": 512,
                "NPUW_LLM_PREFILL_CHUNK_SIZE": 2048,
            })
        elif dev.startswith("GPU"):
            cfg.update({"ATTENTION_BACKEND": "PA"})
    for kv in (extra or []):
        k, v = kv.split("=", 1)
        cfg[k] = _coerce(v)
    return cfg
