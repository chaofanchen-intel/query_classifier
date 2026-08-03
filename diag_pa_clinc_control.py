"""Control experiment: does the PA prefix-caching bug (see diag_pa_prefix_caching.py)
also corrupt CLINC150, or is it specific to cpu_routing?

Runs the SAME 60-sample stratified-ish random subset (seed=42) through PA and SDPA
and diffs the per-record predictions. Used to confirm PA==SDPA on CLINC (byte-identical
predictions) while cpu_routing diverges massively (see report).
"""
import json, random, sys
from pathlib import Path
import openvino_genai as ov_genai
from transformers import AutoTokenizer

MODEL = "/home/chaofan/model_benchmark/benchmark/models/Qwen3.5-2B-FP16"
EXP_DIR = Path(__file__).parent
PROMPT_FILE = EXP_DIR / "prompts" / "clinc_domain_qwen3.5.txt"
DATA = EXP_DIR / "data" / "clinc150_domain_full.jsonl"
LABELS = ["banking", "credit_cards", "kitchen_dining", "home", "auto_commute",
          "travel", "utility", "work", "small_talk", "meta"]
LETTER_MAP = {chr(ord("A") + i): lab for i, lab in enumerate(LABELS)}


def parse(raw):
    for ch in raw.strip():
        if ch in LETTER_MAP:
            return LETTER_MAP[ch]
    for ch in raw.strip().upper():
        if ch in LETTER_MAP:
            return LETTER_MAP[ch]
    return "unknown"


def run(backend, extra_props=None):
    records = [json.loads(l) for l in DATA.open(encoding="utf-8") if l.strip()]
    random.seed(42)
    records = random.sample(records, 60)

    system_prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    tok = AutoTokenizer.from_pretrained(MODEL)

    props = {"ATTENTION_BACKEND": backend}
    if extra_props:
        props.update(extra_props)
    pipe = ov_genai.VLMPipeline(MODEL, "GPU", **props)

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 1
    cfg.do_sample = False
    cfg.apply_chat_template = False

    preds = []
    correct = 0
    for rec in records:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {rec['query']}"},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        raw = str(pipe.generate(prompt, images=[], generation_config=cfg))
        p = parse(raw)
        preds.append(p)
        correct += (p == rec["domain"])
    print(f"[{backend}] accuracy = {correct/len(records):.1%}")
    del pipe
    return preds


if __name__ == "__main__":
    pa_preds = run("PA")
    sdpa_preds = run("SDPA")
    diffs = [i for i, (a, b) in enumerate(zip(pa_preds, sdpa_preds)) if a != b]
    print(f"Diffs between PA and SDPA predictions: {len(diffs)} / {len(pa_preds)}")
    print("Diff indices:", diffs)
