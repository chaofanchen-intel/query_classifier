"""Diagnostic: do the C/D routing classes recover if we let the model *think*?
Compares thinking-off/1-token (baseline) vs thinking-on/N-token (parse final letter)
on the gold C and D subset. Run on a free device to not disturb the main campaigns.
"""
from __future__ import annotations
import argparse, json, re, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
EXP = Path(__file__).parent


def final_letter(raw: str) -> str:
    # take text after </think> if present, else whole string; last A-D letter wins
    tail = raw.split("</think>")[-1]
    m = re.findall(r"[ABCD]", tail.upper())
    if m:
        return m[-1]
    m = re.findall(r"[ABCD]", raw.upper())
    return m[-1] if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ov-model", required=True)
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--pipeline", choices=["llm", "vlm"], required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--prompt", default="cpu_routing_qwen3.5")
    ap.add_argument("--think-tokens", type=int, default=512)
    args = ap.parse_args()

    import openvino_genai as ov_genai
    from transformers import AutoTokenizer

    sysp = (EXP / "prompts" / f"{args.prompt}.txt").read_text(encoding="utf-8").strip()
    recs = [json.loads(l) for l in (EXP / "data" / "eval_set.jsonl").open(encoding="utf-8") if l.strip()]
    cd = [r for r in recs if r["category"] in ("C", "D")]
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    pipe = (ov_genai.LLMPipeline if args.pipeline == "llm" else ov_genai.VLMPipeline)(args.ov_model, args.device)

    def gen(prompt, max_tok):
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_tok
        cfg.do_sample = False
        cfg.apply_chat_template = False
        if args.pipeline == "vlm":
            return str(pipe.generate(prompt, images=[], generation_config=cfg))
        return str(pipe.generate(prompt, cfg))

    def render(query, thinking):
        msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": f"查询：{query}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

    base_ok = think_ok = 0
    for r in cd:
        raw1 = gen(render(r["query"], False), 1)
        p1 = final_letter(raw1)
        t0 = time.perf_counter()
        rawN = gen(render(r["query"], True), args.think_tokens)
        dt = (time.perf_counter() - t0) * 1000
        pN = final_letter(rawN)
        base_ok += p1 == r["category"]
        think_ok += pN == r["category"]
        print(f"  gold={r['category']} base={p1} think={pN} ({dt:.0f}ms, {len(rawN)}ch) :: {r['query'][:34]}")
    n = len(cd)
    print(f"\nC/D subset (n={n}):  thinking-OFF/1tok = {base_ok}/{n} ({base_ok/n:.0%})   "
          f"thinking-ON/{args.think_tokens}tok = {think_ok}/{n} ({think_ok/n:.0%})")


if __name__ == "__main__":
    main()
