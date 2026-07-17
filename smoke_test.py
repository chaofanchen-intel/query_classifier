"""Quick smoke test: load a model on a device and run a few classification queries.
Verifies the inference path works (esp. NPU static-shape / iGPU VLM) before long runs.

    python smoke_test.py --ov-model DIR --hf-model DIR --pipeline llm --device NPU
"""
from __future__ import annotations
import argparse, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

EXP_DIR = Path(__file__).parent
EVAL_SET = EXP_DIR / "data" / "eval_set.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ov-model", required=True)
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--pipeline", choices=["llm", "vlm"], required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--prompt", default="cpu_routing_qwen3.5")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    import openvino_genai as ov_genai
    from transformers import AutoTokenizer

    sys_prompt = (EXP_DIR / "prompts" / f"{args.prompt}.txt").read_text(encoding="utf-8").strip()
    recs = [json.loads(l) for l in EVAL_SET.open(encoding="utf-8") if l.strip()][: args.n]
    tok = AutoTokenizer.from_pretrained(args.hf_model)

    print(f"Loading {args.pipeline} on {args.device} ...", flush=True)
    t0 = time.time()
    pipe = (ov_genai.LLMPipeline if args.pipeline == "llm" else ov_genai.VLMPipeline)(
        args.ov_model, args.device)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 1
    cfg.do_sample = False
    cfg.apply_chat_template = False

    for rec in recs:
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"查询：{rec['query']}"}]
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        t1 = time.perf_counter()
        if args.pipeline == "vlm":
            raw = str(pipe.generate(prompt, images=[], generation_config=cfg))
        else:
            raw = str(pipe.generate(prompt, cfg))
        dt = (time.perf_counter() - t1) * 1000
        print(f"  gold={rec['category']} raw={raw!r} ({dt:.0f}ms) :: {rec['query'][:40]}", flush=True)

    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
