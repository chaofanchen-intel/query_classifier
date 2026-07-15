"""Unified OpenVINO GenAI evaluator for the CPU A/B/C/D classifier.

Uses the SAME toolchain (OV 2026.2 + openvino_genai + transformers tokenizer)
for both Qwen3 (text-only, LLMPipeline) and Qwen3.5 (multimodal, VLMPipeline),
so the two model families are compared on an identical inference path.

Key detail: we pre-render the chat template with the HF tokenizer
(enable_thinking=False) and set apply_chat_template=False so the pipeline does
NOT re-apply a template (which would re-enable thinking and break max_tokens=1).

Example (see README for environment + how to plug in your own model):
    python eval_ov_genai.py \
        --ov-model /path/to/your/OV-IR-FP16 \
        --hf-model /path/to/your/HF-model \
        --pipeline vlm --device GPU.1 --prompt cpu_routing_qwen3.5
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

EXP_DIR = Path(__file__).parent
PROMPTS_DIR = EXP_DIR / "prompts"
RESULTS_DIR = EXP_DIR / "results"
EVAL_SET_PATH = EXP_DIR / "data" / "eval_set.jsonl"
CATEGORIES = ("A", "B", "C", "D")


def parse_category(raw: str) -> str:
    for ch in (raw or "").upper():
        if ch in CATEGORIES:
            return ch
    return "A"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ov-model", required=True, help="OV IR dir")
    parser.add_argument("--hf-model", required=True, help="HF dir for the tokenizer/chat template")
    parser.add_argument("--pipeline", choices=["llm", "vlm"], required=True)
    parser.add_argument("--device", default="GPU.1")
    parser.add_argument("--prompt", default="cpu_routing_qwen3.5")
    parser.add_argument("--tag", default=None, help="label for output file")
    args = parser.parse_args()

    import openvino_genai as ov_genai
    from transformers import AutoTokenizer

    system_prompt = (PROMPTS_DIR / f"{args.prompt}.txt").read_text(encoding="utf-8").strip()
    records = [json.loads(l) for l in EVAL_SET_PATH.open(encoding="utf-8") if l.strip()]
    tag = args.tag or Path(args.ov_model).name

    print(f"OV model: {args.ov_model}")
    print(f"Pipeline: {args.pipeline}  Device: {args.device}  Prompt: {args.prompt}  N: {len(records)}")
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    t0 = time.time()
    if args.pipeline == "llm":
        pipe = ov_genai.LLMPipeline(args.ov_model, args.device)
    else:
        pipe = ov_genai.VLMPipeline(args.ov_model, args.device)
    print(f"  loaded in {time.time()-t0:.1f}s")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 1
    cfg.do_sample = False
    cfg.apply_chat_template = False  # we pre-render below

    def generate(prompt: str) -> str:
        if args.pipeline == "vlm":
            return str(pipe.generate(prompt, images=[], generation_config=cfg))
        return str(pipe.generate(prompt, cfg))

    gold, pred, latencies, per_query = [], [], [], []
    for i, rec in enumerate(records):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"查询：{rec['query']}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        t1 = time.perf_counter()
        raw = generate(prompt)
        lat = (time.perf_counter() - t1) * 1000
        cat = parse_category(raw)
        gold.append(rec["category"]); pred.append(cat); latencies.append(lat)
        per_query.append({"query": rec["query"], "gold": rec["category"], "pred": cat,
                          "raw_output": raw, "latency_ms": lat,
                          "routing_label": rec.get("routing_label")})
        if (i + 1) % 15 == 0:
            acc = sum(1 for g, p in zip(gold, pred) if g == p) / (i + 1)
            print(f"  [{i+1}/{len(records)}] acc {acc:.1%}")

    n = len(gold)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    accuracy = correct / n
    warm = sorted(latencies[1:]) if len(latencies) > 1 else sorted(latencies)
    p50 = warm[len(warm) // 2]; p95 = warm[int(len(warm) * 0.95)]
    cat_to_route = {"A": "local", "B": "local", "C": "cloud", "D": "cloud"}
    gold_cloud = sum(1 for r in per_query if r["routing_label"] == "cloud")
    under = sum(1 for r in per_query
                if r["routing_label"] == "cloud" and cat_to_route[r["pred"]] == "local")

    print(f"\n=== {tag} OV-genai @ {args.device} + {args.prompt} ===")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{n})")
    print(f"Under-routing: {under}/{gold_cloud}")
    print(f"Latency (warm): p50={p50:.1f}ms p95={p95:.1f}ms (cold={latencies[0]:.0f}ms)")
    print(f"\n{'class':<8}{'recall':<10}{'support'}")
    for c in CATEGORIES:
        sup = [g for g in gold if g == c]
        rc = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        print(f"{c:<8}{(rc/len(sup) if sup else 0):<10.1%}{len(sup)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"ovg_{tag}_{args.device.replace('.','')}_{args.prompt}_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"model": tag, "backend": f"openvino-genai-{args.pipeline}", "device": args.device,
                   "prompt_version": args.prompt, "timestamp": datetime.now().isoformat(timespec="seconds"),
                   "samples": n, "accuracy": accuracy, "under_routing": f"{under}/{gold_cloud}",
                   "latency_ms": {"p50": p50, "p95": p95, "cold": latencies[0]},
                   "per_query": per_query}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path.name}")


if __name__ == "__main__":
    main()
