"""Device pipeline configs for the Qwen3/Qwen3.5 evaluation setup
(see benchmark/tests/Qwen3_Qwen3.5_优化说明与测试数据_统一版_20260711.md).

  - GPU : ATTENTION_BACKEND=SDPA  (correctness-first default; see WARNING below)
  - NPU : NPU_TURBO=YES + the NPUW baseline (MAX_PROMPT_LEN / MIN_RESPONSE_LEN /
          prefill chunk). We deliberately do NOT set NPUW_LLM_GENERATE_HINT=BEST_PERF
          for accuracy runs: it changes outputs on 0.6B/1.7B.

  ⚠️  WARNING — ATTENTION_BACKEND=PA is NOT lossless (2026-07-20).
  PagedAttention (PA) is faster (~5x lower TTFT on the 2B VLM at short prompts) and
  is accuracy-EQUAL to SDPA for the LLMPipeline (Qwen3-1.7B: PA==SDPA on all tasks)
  and for the VLMPipeline on CLINC/BANKING. BUT it SILENTLY corrupts the
  Qwen3.5-2B VLMPipeline on the cpu_routing task: FP16 cpu_routing drops 93.1%
  (SDPA, correct) -> 60.3% (PA). The error scales with precision (FP16 worst, INT4
  negligible). Root cause: a correctness issue in the PA path of the VLM pipeline on
  this OV build. Therefore we default GPU to SDPA. PA remains a valid opt-in latency
  optimization for the LLM (`--pipe-config ATTENTION_BACKEND=PA`), or for the VLM only
  after validating your specific task. NPU_TURBO / NPUW length params ARE lossless.
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
            cfg.update({"ATTENTION_BACKEND": "SDPA"})
    for kv in (extra or []):
        k, v = kv.split("=", 1)
        cfg[k] = _coerce(v)
    return cfg
