import json, time, sys
from pathlib import Path

import openvino_genai as ov_genai
from transformers import AutoTokenizer

MODEL = "/home/chaofan/model_benchmark/benchmark/models/Qwen3.5-2B-FP16"
EXP_DIR = Path("/home/chaofan/model_benchmark/query_classifier_handoff")
PROMPT_FILE = EXP_DIR / "prompts" / "cpu_routing_qwen3.5.txt"
EVAL_SET = EXP_DIR / "data" / "eval_set.jsonl"
CATEGORIES = ("A", "B", "C", "D")


def parse_category(raw):
    for ch in (raw or "").upper():
        if ch in CATEGORIES:
            return ch
    return "A"


def run(tag, device, extra_props, scheduler_cfg=None):
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    records = [json.loads(l) for l in EVAL_SET.open(encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(MODEL)

    props = dict(extra_props)
    if scheduler_cfg is not None:
        props["scheduler_config"] = scheduler_cfg

    t0 = time.time()
    pipe = ov_genai.VLMPipeline(MODEL, device, **props)
    print(f"[{tag}] loaded in {time.time()-t0:.1f}s", flush=True)

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 1
    cfg.do_sample = False
    cfg.apply_chat_template = False

    gold, pred = [], []
    for i, rec in enumerate(records):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"查询：{rec['query']}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        raw = str(pipe.generate(prompt, images=[], generation_config=cfg))
        cat = parse_category(raw)
        gold.append(rec["category"]); pred.append(cat)

    n = len(gold)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    acc = correct / n
    print(f"[{tag}] accuracy = {acc:.1%} ({correct}/{n})", flush=True)
    del pipe
    return acc, list(zip(gold, pred))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}

    if which in ("all", "pa_prefix_on"):
        sc = ov_genai.SchedulerConfig()
        sc.enable_prefix_caching = True
        sc.max_num_batched_tokens = 100000
        results["PA_prefix_ON"] = run("PA_prefix_ON", "GPU", {"ATTENTION_BACKEND": "PA"}, sc)

    if which in ("all", "pa_prefix_off"):
        sc = ov_genai.SchedulerConfig()
        sc.enable_prefix_caching = False
        sc.max_num_batched_tokens = 100000
        results["PA_prefix_OFF"] = run("PA_prefix_OFF", "GPU", {"ATTENTION_BACKEND": "PA"}, sc)

    print("\n=== SUMMARY ===")
    for k, (acc, _) in results.items():
        print(k, f"{acc:.1%}")
