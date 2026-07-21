"""Data-aware INT4 re-quantizer (generalized: LLM or VLM), with a selectable
GENERAL calibration corpus (Chinese / English / mixed — never the task's own prompts).

Motivation: cpu_routing is Chinese while CLINC/BANKING are English. AWQ/scale-estimation
protect the channels most important for the *calibration* distribution, so an
English-only calib (wikitext2) may under-serve the Chinese task. This script lets us
calibrate on Chinese or mixed general text.

Examples:
    # 2B VLM, AWQ + scale-estimation, mixed zh+en calibration, asym g128
    python requant2.py --model-id benchmark/models/Qwen3.5-2B --ref-ir <FP16-VLM-dir> \
        --pipeline vlm --calib-file data/mixed_calib.txt --awq --scale-estimation \
        --group-size 128 --ratio 1.0 --out <OUT>

    # "other method": data-free mixed precision (20% layers -> higher precision), no AWQ
    python requant2.py --model-id benchmark/models/Qwen3.5-2B --ref-ir <FP16-VLM-dir> \
        --pipeline vlm --ratio 0.8 --group-size 128 --out <OUT>
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

EXP = Path(__file__).parent


def load_calib(path, n):
    texts = [json.loads(l)["text"] for l in open(path, encoding="utf-8") if l.strip()]
    return texts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="HF model id or local HF dir (export source)")
    ap.add_argument("--ref-ir", required=True, help="reference OV IR dir to copy tokenizer/chat template from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipeline", choices=["llm", "vlm"], default="llm")
    ap.add_argument("--calib-file", default=str(EXP / "data" / "wikitext2_calib.txt"))
    ap.add_argument("--awq", action="store_true")
    ap.add_argument("--scale-estimation", action="store_true")
    ap.add_argument("--sym", action="store_true", help="symmetric (required for NPU; 2B is GPU-only so leave off)")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--num-samples", type=int, default=128)
    args = ap.parse_args()

    from optimum.intel import OVWeightQuantizationConfig
    if args.pipeline == "vlm":
        from optimum.intel import OVModelForVisualCausalLM as OVModel
    else:
        from optimum.intel import OVModelForCausalLM as OVModel
    from transformers import AutoTokenizer

    data_aware = args.awq or args.scale_estimation
    calib = load_calib(args.calib_file, args.num_samples) if data_aware else None
    if data_aware:
        print(f"calibration: {len(calib)} samples from {args.calib_file} (sample head: {calib[0][:50]!r})")

    qkwargs = dict(
        bits=4, sym=args.sym, group_size=args.group_size, ratio=args.ratio,
        tokenizer=args.model_id,
    )
    if data_aware:
        qkwargs.update(dataset=calib, num_samples=len(calib),
                       awq=args.awq, scale_estimation=args.scale_estimation)
    if args.sym:
        qkwargs["backup_precision"] = "int8_sym"
    qcfg = OVWeightQuantizationConfig(**qkwargs)
    print(f"quant: bits=4 sym={args.sym} g={args.group_size} ratio={args.ratio} "
          f"awq={args.awq} scale_est={args.scale_estimation} pipeline={args.pipeline}")

    model = OVModel.from_pretrained(args.model_id, export=True, quantization_config=qcfg)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model_id).save_pretrained(args.out)

    # genai needs the OV tokenizer/detokenizer + chat template — copy from the reference IR
    for fn in ["openvino_tokenizer.xml", "openvino_tokenizer.bin",
               "openvino_detokenizer.xml", "openvino_detokenizer.bin", "chat_template.jinja"]:
        src = Path(args.ref_ir) / fn
        if src.exists():
            shutil.copy2(src, Path(args.out) / fn)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
