"""Data-aware INT4 (AWQ + scale-estimation) for the Qwen3.5-2B VLM, driven directly
through NNCF — because optimum's OVModelForVisualCausalLM only accepts *predefined*
image-text calibration datasets, not a custom text list. Here we calibrate the LM on a
GENERAL text corpus of our choice (Chinese / mixed), never the task's own prompts.

The 2B VLM's language_model takes `inputs_embeds` (not input_ids), so we precompute
embeds via the text_embeddings OV model, then AWQ-compress only the language_model.
All other IR files (vision, embeddings, tokenizer, chat template) are copied through.

    python requant_nncf_vlm.py --ref-ir <FP16-VLM-dir> --hf <HF-dir-for-tokenizer> \
        --calib-file data/mixed_calib.txt --group-size 128 --ratio 1.0 \
        --num-samples 128 --max-len 512 --out <OUT>
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-ir", required=True, help="FP16 VLM IR dir (source of LM + all other files)")
    ap.add_argument("--hf", required=True, help="HF dir/id for the tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-file", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--no-awq", action="store_true")
    ap.add_argument("--no-scale-estimation", action="store_true")
    args = ap.parse_args()

    import openvino as ov
    import nncf
    from transformers import AutoTokenizer

    ref = Path(args.ref_ir)
    core = ov.Core()
    tok = AutoTokenizer.from_pretrained(args.hf)

    texts = [json.loads(l)["text"] for l in open(args.calib_file, encoding="utf-8") if l.strip()]
    texts = texts[:args.num_samples]
    print(f"calibration: {len(texts)} samples from {args.calib_file} (head: {texts[0][:40]!r})")

    # text_embeddings: input_ids -> inputs_embeds  (run on CPU)
    te = core.compile_model(core.read_model(ref / "openvino_text_embeddings_model.xml"), "CPU")

    def make_inputs(text):
        ids = tok(text, return_tensors="np", truncation=True, max_length=args.max_len)["input_ids"].astype(np.int64)
        seq = ids.shape[1]
        embeds = te(ids)[0]                                    # [1, seq, 2048] float32
        attn = np.ones((1, seq), dtype=np.int64)
        pos = np.tile(np.arange(seq, dtype=np.int64), (4, 1, 1))  # [4,1,seq] mrope (text: all rows = index)
        beam = np.zeros((1,), dtype=np.int32)
        return {"inputs_embeds": embeds, "attention_mask": attn, "position_ids": pos, "beam_idx": beam}

    calib = [make_inputs(t) for t in texts]
    print(f"built {len(calib)} LM calibration inputs (embeds precomputed)")

    lm = core.read_model(ref / "openvino_language_model.xml")
    dataset = nncf.Dataset(calib, lambda x: x)

    comp = nncf.compress_weights(
        lm,
        mode=nncf.CompressWeightsMode.INT4_ASYM,     # 2B is GPU-only; asym is best & matches existing g128
        group_size=args.group_size,
        ratio=args.ratio,
        dataset=dataset,
        awq=not args.no_awq,
        scale_estimation=not args.no_scale_estimation,
        sensitivity_metric=nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
    )
    print(f"compressed: g={args.group_size} ratio={args.ratio} awq={not args.no_awq} "
          f"scale_est={not args.no_scale_estimation}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # copy every file from the FP16 dir, then overwrite the language_model with the compressed one
    for f in ref.iterdir():
        if f.is_file():
            shutil.copy2(f, out / f.name)
    ov.save_model(comp, out / "openvino_language_model.xml")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
