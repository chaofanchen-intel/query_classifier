"""Generate per-class accuracy tables and confusion matrices from results.

Covers three result types:
  - top3_* : logits-based top-k (uses top3[0] as pred for confusion matrix)
  - ovg_*  : OV genai eval on CPU dataset (gold/pred in per_query)
  - pub_*  : OV genai eval on public datasets (gold/pred in per_query)

For each model×dataset, picks the LATEST result file.
Outputs: PNG confusion matrices + summary table to stdout.
"""

import json
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = Path(__file__).parent / "confusion_matrices"
OUT_DIR.mkdir(exist_ok=True)


def load_latest(pattern):
    """Find latest file matching glob pattern, return parsed JSON."""
    files = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    if not files:
        return None
    return json.load(open(files[-1]))


def collect_top3_results():
    """Collect top3 results (top-1 prediction = top3[0])."""
    results = {}
    files = sorted(glob.glob(str(RESULTS_DIR / "top3_*.json")))
    # group by (dataset, tag) and keep latest
    grouped = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        if "per_query" not in d or not d["per_query"]:
            continue
        # infer dataset from key or filename
        ds = d.get("dataset")
        if not ds:
            stem = Path(f).stem  # top3_cpu_Qwen3-1.7B_... or top3_Qwen3-1.7B_...
            for candidate in ["cpu", "clinc", "banking"]:
                if f"_{candidate}_" in stem:
                    ds = candidate
                    break
            if not ds:
                ds = "cpu"  # old files without dataset field were cpu
        tag = d.get("tag", Path(f).stem)
        key = (ds, tag)
        grouped[key].append((f, d))
    for key, items in grouped.items():
        # latest by filename (timestamp in name)
        _, d = sorted(items, key=lambda x: x[0])[-1]
        results[key] = d
    return results


def collect_ovg_results():
    """Collect OV genai CPU dataset results."""
    results = {}
    files = sorted(glob.glob(str(RESULTS_DIR / "ovg_*.json")))
    grouped = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        if "per_query" not in d or not d["per_query"]:
            continue
        # prefer JSON fields (robust) over filename parsing
        model = d.get("model", Path(f).stem)
        device = d.get("device", "?")
        prompt = d.get("prompt_version", "?")
        key = ("cpu_ovg", f"{model}_{device}_{prompt}")
        grouped[key].append((f, d))
    for key, items in grouped.items():
        _, d = sorted(items, key=lambda x: x[0])[-1]
        results[key] = d
    return results


def collect_pub_results():
    """Collect public dataset results."""
    results = {}
    files = sorted(glob.glob(str(RESULTS_DIR / "pub_*.json")))
    grouped = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        if "per_query" not in d or not d["per_query"]:
            continue
        ds = d.get("dataset", "?")
        model = d.get("model", "?")
        prompt = d.get("prompt", "?")
        device = d.get("device", "?")
        key = (f"pub_{ds}", f"{model}_{device}_{prompt}")
        grouped[key].append((f, d))
    for key, items in grouped.items():
        _, d = sorted(items, key=lambda x: x[0])[-1]
        results[key] = d
    return results


def make_confusion_matrix(gold, pred, labels, title, out_path):
    """Build and save confusion matrix heatmap."""
    n = len(labels)
    label2idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gold, pred):
        gi = label2idx.get(g)
        pi = label2idx.get(p)
        if gi is not None and pi is not None:
            cm[gi][pi] += 1

    # per-class accuracy (recall)
    per_class_acc = {}
    for i, lab in enumerate(labels):
        total = cm[i].sum()
        if total > 0:
            per_class_acc[lab] = cm[i][i] / total
        else:
            per_class_acc[lab] = 0.0

    # plot - always use row-normalized (recall %)
    fig_size = max(6, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    if n <= 15:
        # small matrix: show percentage values as annotations
        annot_arr = np.array([[f"{v:.0%}" for v in row] for row in cm_norm])
        sns.heatmap(cm_norm, annot=annot_arr, fmt="", cmap="Blues", xticklabels=labels,
                    yticklabels=labels, ax=ax, vmin=0, vmax=1,
                    cbar_kws={"shrink": 0.8, "label": "Recall"})
    else:
        sns.heatmap(cm_norm, annot=False, cmap="Blues", xticklabels=labels,
                    yticklabels=labels, ax=ax, vmin=0, vmax=1,
                    cbar_kws={"shrink": 0.6, "label": "Recall"})

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title, fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=max(4, 8 - n // 10))
    plt.yticks(rotation=0, fontsize=max(4, 8 - n // 10))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return per_class_acc, cm


def extract_gold_pred_top3(data):
    """Extract gold and pred from top3 results (pred = top3[0])."""
    gold = [r["gold"] for r in data["per_query"]]
    pred = [r["top3"][0] if r["top3"] else "unknown" for r in data["per_query"]]
    return gold, pred


def extract_gold_pred_ovg(data):
    """Extract gold and pred from ovg results."""
    gold = [r["gold"] for r in data["per_query"]]
    pred = [r["pred"] for r in data["per_query"]]
    return gold, pred


def extract_gold_pred_pub(data):
    """Extract gold and pred from pub results."""
    gold = [r["gold"] for r in data["per_query"]]
    pred = [r["pred"] for r in data["per_query"]]
    return gold, pred


def main():
    print("=" * 70)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 70)

    all_results = []

    # --- Top-3 results (logits-based) ---
    top3 = collect_top3_results()
    print(f"\n[top3] Found {len(top3)} model×dataset combinations")
    for (ds, tag), data in sorted(top3.items()):
        gold, pred = extract_gold_pred_top3(data)
        labels = sorted(set(gold) | set(pred) - {"unknown"})
        title = f"[top3] {tag} / {ds}"
        fname = f"cm_top3_{ds}_{tag}.png"
        per_class, cm = make_confusion_matrix(gold, pred, labels, title,
                                              OUT_DIR / fname)
        acc = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)
        all_results.append((ds, tag, "top3", acc, per_class, len(gold)))

    # --- OVG results (CPU dataset, genai) ---
    ovg = collect_ovg_results()
    print(f"\n[ovg] Found {len(ovg)} model×prompt combinations")
    for (ds, tag), data in sorted(ovg.items()):
        gold, pred = extract_gold_pred_ovg(data)
        labels = sorted(set(gold) | set(pred) - {"unknown"})
        title = f"[ovg] {tag}"
        fname = f"cm_ovg_{tag}.png"
        per_class, cm = make_confusion_matrix(gold, pred, labels, title,
                                              OUT_DIR / fname)
        acc = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)
        all_results.append((ds, tag, "ovg", acc, per_class, len(gold)))

    # --- Pub results (public datasets, genai) ---
    pub = collect_pub_results()
    print(f"\n[pub] Found {len(pub)} model×dataset combinations")
    for (ds, tag), data in sorted(pub.items()):
        gold, pred = extract_gold_pred_pub(data)
        labels = sorted(set(gold) | set(pred) - {"unknown"})
        title = f"[pub] {tag} / {ds}"
        safe_tag = re.sub(r"[/\\:]", "_", tag)
        safe_ds = re.sub(r"[/\\:]", "_", ds)
        fname = f"cm_{safe_ds}_{safe_tag}.png"
        per_class, cm = make_confusion_matrix(gold, pred, labels, title,
                                              OUT_DIR / fname)
        acc = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)
        all_results.append((ds, tag, "pub", acc, per_class, len(gold)))

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Per-class accuracy (recall)")
    print("=" * 70)

    # Group by dataset
    by_ds = defaultdict(list)
    for ds, tag, src, acc, per_class, n in all_results:
        by_ds[ds].append((tag, src, acc, per_class, n))

    for ds in sorted(by_ds.keys()):
        items = by_ds[ds]
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds} (N={items[0][4]})")
        print(f"{'─' * 60}")

        # get all class labels
        all_labels = sorted(set(k for _, _, _, pc, _ in items for k in pc.keys()))

        # header
        hdr = f"{'Model':<35} {'Acc':>5}"
        for lab in all_labels:
            short = lab[:8] if len(lab) > 8 else lab
            hdr += f" {short:>8}"
        print(hdr)
        print("-" * len(hdr))

        for tag, src, acc, per_class, n in sorted(items, key=lambda x: -x[2]):
            row = f"{tag:<35} {acc:>5.1%}"
            for lab in all_labels:
                v = per_class.get(lab, 0)
                row += f" {v:>7.0%}"
            print(row)

    print(f"\n\nConfusion matrix PNGs saved to: {OUT_DIR}/")
    print(f"Total plots generated: {len(all_results)}")


if __name__ == "__main__":
    main()
