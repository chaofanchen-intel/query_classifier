"""Data-aware INT4 weight quantization (AWQ + scale-estimation) with a GENERAL
calibration set (wikitext-2), so the quantization generalizes rather than being
tuned to this task's own prompts. wikitext-2 was fetched via hf-mirror (see
data/wikitext2_calib.txt); huggingface.co is proxy-blocked and the `datasets`
library conflicts with this env's huggingface_hub, so we read the parquet-derived
text file directly.

Produces an OV IR usable by openvino_genai LLMPipeline (copies the OV tokenizer/
detokenizer + chat template from the reference FP16 dir).

    python requant_aware.py --model-id Qwen/Qwen3-1.7B --ref-ir <FP16-dir> \
        --sym --out <OUT-dir> [--ratio 1.0] [--num-samples 128]
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

EXP = Path(__file__).parent
CALIB_FILE = EXP / "data" / "wikitext2_calib.txt"


def build_calibration(n):
    """Load N general-text (wikitext-2) calibration strings."""
    texts = [json.loads(l)["text"] for l in open(CALIB_FILE, encoding="utf-8") if l.strip()]
    return texts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="HF model id or path (source for export)")
    ap.add_argument("--ref-ir", required=True, help="reference OV IR dir to copy tokenizer/chat template from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sym", action="store_true", help="symmetric (required for NPU)")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--num-samples", type=int, default=128)
    args = ap.parse_args()

    from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    calib = build_calibration(args.num_samples)
    print(f"calibration texts: {len(calib)} (general wikitext-2, sample len chars={len(calib[0])})")

    qcfg = OVWeightQuantizationConfig(
        bits=4, sym=args.sym, group_size=args.group_size, ratio=args.ratio,
        dataset=calib, num_samples=len(calib),
        scale_estimation=True, awq=True,
        # NPU requires symmetric everywhere; force int8_sym for any non-int4 (backup) layers.
        backup_precision=("int8_sym" if args.sym else None),
        tokenizer=args.model_id,
    )
    print(f"quant config: bits=4 sym={args.sym} g={args.group_size} ratio={args.ratio} awq+scale_estimation, in-domain calib")
    model = OVModelForCausalLM.from_pretrained(args.model_id, export=True, quantization_config=qcfg)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    # copy OV tokenizer/detokenizer + chat template from the reference IR (genai needs these)
    for fn in ["openvino_tokenizer.xml", "openvino_tokenizer.bin",
               "openvino_detokenizer.xml", "openvino_detokenizer.bin", "chat_template.jinja"]:
        src = Path(args.ref_ir) / fn
        if src.exists():
            shutil.copy2(src, Path(args.out) / fn)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
