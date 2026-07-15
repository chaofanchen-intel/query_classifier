"""Evaluate OV (genai) classifiers on public intent datasets (CLINC150, BANKING-cluster10).

Same clean inference path as eval_ov_genai.py:
  tokenizer pre-render (enable_thinking=False) + apply_chat_template=False + max_tokens=N.
Generalization test: does the LLM-prompt classification method hold on other
intent-classification tasks (label is intrinsic to the query)?
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

EXP_DIR = Path(__file__).parent
PROMPTS_DIR = EXP_DIR / "prompts"
RESULTS_DIR = EXP_DIR / "results"
DATA_DIR = EXP_DIR / "data"  # datasets live in-repo here (NOT /tmp — /tmp gets wiped on reboot)

DATASETS = {
    "clinc": {
        "path": f"{DATA_DIR}/clinc150_domain_full.jsonl",
        "label_key": "domain",
        "labels": ["banking", "credit_cards", "kitchen_dining", "home", "auto_commute",
                   "travel", "utility", "work", "small_talk", "meta"],
        "query_key": "query",
        "user_prefix": "Query: ",
        "max_new_tokens": 1,    # letter mode (A-J)
        "letter_mode": True,
    },
    "banking_cluster10": {
        # 77 intents clustered into 10 semantic groups (Ward), labels A-J
        "path": f"{DATA_DIR}/banking77_cluster10_full.jsonl",
        "label_key": "cluster",        # gold = cluster name
        "labels": ["card_payment_issues", "transfer_and_refund", "exchange_and_support",
                   "top_up", "physical_card_setup", "cash_and_balance", "account_security",
                   "fees_and_charges", "lost_or_stolen", "identity_verification"],
        "query_key": "query",
        "user_prefix": "Query: ",
        "max_new_tokens": 1,
        "letter_mode": True,
        # char A-J -> cluster name
        "letter_map_inline": True,
    },
}


def build_letter_map(labels):
    return {chr(ord("A") + i): lab for i, lab in enumerate(labels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ov-model", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--pipeline", choices=["llm", "vlm"], required=True)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--prompt", required=True, help="prompt file stem in prompts/")
    parser.add_argument("--device", default="GPU.1")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate on a random stratified-ish subset of N samples (fast iteration)")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed for --limit")
    args = parser.parse_args()

    import openvino_genai as ov_genai
    from transformers import AutoTokenizer

    ds = DATASETS[args.dataset]
    labels = ds["labels"]
    label_set = set(labels)
    if ds["letter_mode"]:
        if ds.get("letter_map_inline"):
            letter_map = {chr(ord("A") + i): lab for i, lab in enumerate(labels)}
        else:
            letter_map = build_letter_map(labels)
    else:
        letter_map = None

    system_prompt = (PROMPTS_DIR / f"{args.prompt}.txt").read_text(encoding="utf-8").strip()
    records = [json.loads(l) for l in open(ds["path"], encoding="utf-8") if l.strip()]
    if args.limit and args.limit < len(records):
        import random
        random.seed(args.seed)
        records = random.sample(records, args.limit)
    tag = args.tag or Path(args.ov_model).name

    print(f"Dataset: {args.dataset} ({len(records)} samples, {len(labels)} classes, "
          f"random={1/len(labels):.1%})")
    print(f"Model: {tag}  Pipeline: {args.pipeline}  Prompt: {args.prompt}")
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    t0 = time.time()
    pipe = (ov_genai.LLMPipeline if args.pipeline == "llm" else ov_genai.VLMPipeline)(
        args.ov_model, args.device)
    print(f"  loaded in {time.time()-t0:.1f}s")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = ds["max_new_tokens"]
    cfg.do_sample = False
    cfg.apply_chat_template = False

    def generate(prompt):
        if args.pipeline == "vlm":
            return str(pipe.generate(prompt, images=[], generation_config=cfg))
        return str(pipe.generate(prompt, cfg))

    def parse(raw):
        if ds["letter_mode"]:
            # case-sensitive: map may contain upper, lower, digits, CJK
            for ch in raw.strip():
                if ch in letter_map:
                    return letter_map[ch]
            # fallback: try upper for A-J style maps
            for ch in raw.strip().upper():
                if ch in letter_map:
                    return letter_map[ch]
            return "unknown"
        r = raw.strip().lower().replace(" ", "_")
        for lab in labels:
            if lab.lower() == r:
                return lab
        for lab in labels:
            if lab.lower() in r:
                return lab
        return "unknown"

    gold, pred, latencies, unparsed = [], [], [], 0
    for i, rec in enumerate(records):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{ds['user_prefix']}{rec[ds['query_key']]}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        t1 = time.perf_counter()
        raw = generate(prompt)
        latencies.append((time.perf_counter() - t1) * 1000)
        p = parse(raw)
        if p == "unknown":
            unparsed += 1
        gold.append(rec[ds["label_key"]]); pred.append(p)
        if (i + 1) % 50 == 0:
            acc = sum(1 for g, q in zip(gold, pred) if g == q) / (i + 1)
            print(f"  [{i+1}/{len(records)}] acc {acc:.1%}")

    n = len(gold)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    accuracy = correct / n
    warm = sorted(latencies[1:]) if len(latencies) > 1 else sorted(latencies)
    p50 = warm[len(warm) // 2]; p95 = warm[int(len(warm) * 0.95)]

    print(f"\n=== {tag} / {args.dataset} / {args.prompt} ===")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{n})  random={1/len(labels):.1%}  "
          f"x{accuracy*len(labels):.1f} over random")
    print(f"Unparseable: {unparsed}")
    print(f"Latency (warm): p50={p50:.1f}ms p95={p95:.1f}ms")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"pub_{args.dataset}_{tag}_{args.prompt}_{ts}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "model": tag, "prompt": args.prompt,
                   "samples": n, "accuracy": accuracy, "unparseable": unparsed,
                   "random_baseline": 1/len(labels),
                   "latency_ms": {"p50": p50, "p95": p95},
                   "per_query": [{"q": records[i][ds["query_key"]][:120], "gold": gold[i],
                                  "pred": pred[i]} for i in range(n)]},
                  f, ensure_ascii=False, indent=2)
    print(f"Saved {out.name}")


if __name__ == "__main__":
    main()
