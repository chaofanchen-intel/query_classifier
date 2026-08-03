import json, time
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


def run(tag, props):
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    records = [json.loads(l) for l in EVAL_SET.open(encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(MODEL)

    t0 = time.time()
    pipe = ov_genai.VLMPipeline(MODEL, "GPU", **props)
    load_s = time.time() - t0

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 1
    cfg.do_sample = False
    cfg.apply_chat_template = False

    gold, pred, lat = [], [], []
    for i, rec in enumerate(records):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"查询：{rec['query']}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        t1 = time.perf_counter()
        raw = str(pipe.generate(prompt, images=[], generation_config=cfg))
        lat.append((time.perf_counter() - t1) * 1000)
        cat = parse_category(raw)
        gold.append(rec["category"]); pred.append(cat)

    acc = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)
    warm = sorted(lat[1:])
    p50 = warm[len(warm)//2]
    p95 = warm[int(len(warm)*0.95)]
    print(f"[{tag}] load={load_s:.1f}s  acc={acc:.1%}  p50={p50:.1f}ms  p95={p95:.1f}ms  cold={lat[0]:.1f}ms")
    del pipe
    return acc, p50, p95


if __name__ == "__main__":
    sc_off = ov_genai.SchedulerConfig()
    sc_off.enable_prefix_caching = False
    sc_off.max_num_batched_tokens = 100000
    run("PA_prefix_OFF", {"ATTENTION_BACKEND": "PA", "scheduler_config": sc_off})

    sc_on = ov_genai.SchedulerConfig()
    sc_on.enable_prefix_caching = True
    sc_on.max_num_batched_tokens = 100000
    run("PA_prefix_ON", {"ATTENTION_BACKEND": "PA", "scheduler_config": sc_on})

    run("SDPA", {"ATTENTION_BACKEND": "SDPA"})
