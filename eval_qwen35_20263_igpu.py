"""Qwen3.5-2B iGPU evaluation against the OpenVINO 2026.3 runtime.

Records prompt/token statistics and per-query generate latency so accuracy and
TTFT can be analyzed with the same exact rendered prompt.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import openvino_genai as ov_genai
from transformers import AutoTokenizer

ROOT = Path(__file__).parent
MODEL_ROOT = Path("/home/chaofan/model_benchmark/benchmark/models")
OUTPUT_ROOT = ROOT / "results_20263_qwen35_igpu"
PROMPTS = {
    "cpu_routing": "cpu_routing_qwen3.5.txt",
    "clinc": "clinc_domain_qwen3.5.txt",
    "banking_cluster10": "banking_cluster10_qwen3.5.txt",
}
DATASETS = {
    "cpu_routing": (ROOT / "data/eval_set.jsonl", "category", "query", "查询："),
    "clinc": (ROOT / "data/clinc150_domain_full.jsonl", "domain", "query", "Query:"),
    "banking_cluster10": (ROOT / "data/banking77_cluster10_full.jsonl", "cluster", "query", "Query:"),
}
MODELS = {
    "FP16": "Qwen3.5-2B-FP16",
    "INT8": "Qwen3.5-2B-INT8",
    "INT4": "Qwen3.5-2B-INT4",
    "INT4-g128": "Qwen3.5-2B-INT4-g128",
    "INT4-g128-awqmix": "Qwen3.5-2B-INT4-g128-awqmix",
    "INT4-g128-awqmix-r08": "Qwen3.5-2B-INT4-g128-awqmix-r08",
    "INT4-g128-awqzh": "Qwen3.5-2B-INT4-g128-awqzh",
    "INT4-g128-r08": "Qwen3.5-2B-INT4-g128-r08",
}
LABELS = {
    "cpu_routing": ("A", "B", "C", "D"),
    "clinc": ("banking", "credit_cards", "kitchen_dining", "home", "auto_commute", "travel", "utility", "work", "small_talk", "meta"),
    "banking_cluster10": ("card_payment_issues", "transfer_and_refund", "exchange_and_support", "top_up", "physical_card_setup", "cash_and_balance", "account_security", "fees_and_charges", "lost_or_stolen", "identity_verification"),
}


def parse_output(dataset: str, raw: str) -> str:
    labels = LABELS[dataset]
    if dataset == "cpu_routing":
        for char in (raw or "").upper():
            if char in labels:
                return char
        return "A"
    for char in (raw or "").strip():
        if char in "ABCDEFGHIJ":
            return labels[ord(char) - ord("A")]
    for char in (raw or "").upper().strip():
        if char in "ABCDEFGHIJ":
            return labels[ord(char) - ord("A")]
    return "unknown"


def prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * fraction))]


def evaluate(
    model_tag: str,
    dataset: str,
    backend: str,
    prefix_caching: bool,
    cache_interval_multiplier: int | None,
    limit: int | None,
) -> Path:
    model_dir = MODEL_ROOT / MODELS[model_tag]
    data_path, label_key, query_key, user_prefix = DATASETS[dataset]
    prompt_path = ROOT / "prompts" / PROMPTS[dataset]
    records = [json.loads(line) for line in data_path.open(encoding="utf-8") if line.strip()]
    if limit is not None:
        records = records[:limit]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    rendered = []
    for record in records:
        query = record[query_key]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prefix}{query}"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        rendered.append({
            "query": query,
            "gold": record[label_key],
            "text": text,
            "ids": ids,
            "query_tokens": len(tokenizer(query, add_special_tokens=False)["input_ids"]),
            "prompt_tokens": len(ids),
            "system_prompt_tokens": len(tokenizer(system_prompt, add_special_tokens=False)["input_ids"]),
        })
    for index, item in enumerate(rendered):
        item["shared_prefix_tokens"] = prefix_length(item["ids"], rendered[index - 1]["ids"]) if index else 0

    scheduler = ov_genai.SchedulerConfig()
    scheduler.enable_prefix_caching = prefix_caching
    if cache_interval_multiplier is not None:
        scheduler.cache_interval_multiplier = cache_interval_multiplier
    scheduler.max_num_batched_tokens = 100000
    props = {"ATTENTION_BACKEND": backend, "scheduler_config": scheduler}
    print(f"[{model_tag}/{dataset}/{backend}] loading {model_dir}", flush=True)
    started = time.time()
    pipe = ov_genai.VLMPipeline(str(model_dir), "GPU", **props)
    load_seconds = time.time() - started
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 1
    config.do_sample = False
    config.apply_chat_template = False

    latencies = []
    predictions = []
    for index, item in enumerate(rendered, 1):
        started = time.perf_counter()
        raw = str(pipe.generate(item["text"], images=[], generation_config=config))
        latency = (time.perf_counter() - started) * 1000
        item["ttft_ms"] = latency
        item["raw"] = raw
        item["pred"] = parse_output(dataset, raw)
        item.pop("text")
        item.pop("ids")
        latencies.append(latency)
        predictions.append(item["pred"])
        if index % 250 == 0:
            print(f"[{model_tag}/{dataset}] {index}/{len(rendered)}", flush=True)
    del pipe

    correct = sum(item["gold"] == item["pred"] for item in rendered)
    warm = latencies[1:] or latencies
    output = {
        "runtime": "OpenVINO 2026.3",
        "runtime_version": getattr(__import__("openvino"), "__version__", "unknown"),
        "model": model_tag,
        "model_dir": str(model_dir),
        "dataset": dataset,
        "device": "GPU.0",
        "attention_backend": backend,
        "prefix_caching": prefix_caching,
        "cache_interval_multiplier": cache_interval_multiplier,
        "samples": len(rendered),
        "correct": correct,
        "accuracy": correct / len(rendered),
        "load_seconds": load_seconds,
        "token_stats": {
            "query_tokens": {"min": min(i["query_tokens"] for i in rendered), "p50": statistics.median(i["query_tokens"] for i in rendered), "mean": statistics.mean(i["query_tokens"] for i in rendered), "max": max(i["query_tokens"] for i in rendered)},
            "prompt_tokens": {"min": min(i["prompt_tokens"] for i in rendered), "p50": statistics.median(i["prompt_tokens"] for i in rendered), "mean": statistics.mean(i["prompt_tokens"] for i in rendered), "max": max(i["prompt_tokens"] for i in rendered)},
            "shared_prefix_tokens": {"min": min(i["shared_prefix_tokens"] for i in rendered[1:] or rendered), "p50": statistics.median(i["shared_prefix_tokens"] for i in rendered[1:] or rendered), "mean": statistics.mean(i["shared_prefix_tokens"] for i in rendered[1:] or rendered), "max": max(i["shared_prefix_tokens"] for i in rendered[1:] or rendered)},
        },
        "ttft_ms": {"cold": latencies[0], "p50": percentile(warm, 0.50), "p95": percentile(warm, 0.95), "mean": statistics.mean(warm)},
        "per_query": [{key: item[key] for key in ("query", "gold", "pred", "raw", "query_tokens", "prompt_tokens", "system_prompt_tokens", "shared_prefix_tokens", "ttft_ms")} for item in rendered],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache_tag = "prefix_on" if prefix_caching else "prefix_off"
    interval_tag = f"_interval{cache_interval_multiplier}" if cache_interval_multiplier is not None else ""
    out = OUTPUT_ROOT / f"{model_tag}_{dataset}_{backend}_{cache_tag}{interval_tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{model_tag}/{dataset}] accuracy={output['accuracy']:.1%} ttft_p50={output['ttft_ms']['p50']:.1f}ms -> {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--backend", choices=("PA", "SDPA"), default="PA")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prefix-caching", choices=("on", "off"), default="on")
    parser.add_argument("--cache-interval-multiplier", type=int)
    args = parser.parse_args()
    prefix_caching = args.prefix_caching == "on"
    for model in args.models:
        for dataset in args.datasets:
            evaluate(model, dataset, args.backend, prefix_caching, args.cache_interval_multiplier, args.limit)


if __name__ == "__main__":
    main()
