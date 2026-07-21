"""Generate accuracy & latency comparison charts from results/*.json.

Outputs to charts/:
  - acc_<family>.png   : grouped bars, accuracy by quant x task x device (+FP16 line)
  - lat_<family>.png   : p50 latency by quant x device
  - summary_table.md   : machine-readable summary of all (model,device,task) cells
"""
from __future__ import annotations
import json, glob, re
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).parent
RES = EXP / "results"
OUT = EXP / "charts"; OUT.mkdir(exist_ok=True)

TASKS = ["cpu_routing", "clinc", "banking_cluster10"]
TASK_LABEL = {"cpu_routing": "cpu_routing(58)", "clinc": "CLINC(4500)", "banking_cluster10": "BANKING(3080)"}

# canonical quant label + plot order from the model dir name
def quant_of(model: str):
    m = model
    if m.endswith("-FP16"): return "FP16"
    if m.endswith("-INT8"): return "INT8"
    if "INT4-g128sym-aware" in m: return "INT4 g128sym+AWQ"
    if "INT4-g128sym-r08" in m:   return "INT4 g128sym r0.8"
    if "INT4-g128sym" in m:       return "INT4 g128sym"
    if "INT4-g128-aware" in m:    return "INT4 g128+AWQ"
    if "INT4-g128" in m:          return "INT4 g128"
    if m.endswith("-INT4"):       return "INT4 aggressive"
    return None

QORDER = ["INT4 aggressive", "INT4 g128sym", "INT4 g128sym r0.8", "INT4 g128sym+AWQ",
          "INT4 g128", "INT4 g128+AWQ", "INT8", "FP16"]
QCOLOR = {"INT4 aggressive": "#d62728", "INT4 g128sym": "#ff9896", "INT4 g128sym r0.8": "#ff7f0e",
          "INT4 g128sym+AWQ": "#e377c2", "INT4 g128": "#2ca02c", "INT4 g128+AWQ": "#17becf",
          "INT8": "#1f77b4", "FP16": "#7f7f7f"}
DEVS = ["NPU", "GPU.0", "GPU.1"]
DEV_LABEL = {"NPU": "NPU", "GPU.0": "iGPU", "GPU.1": "dGPU"}


def family_of(model):
    if "Qwen3.5-2B" in model or "Qwen3_5-2B" in model: return "Qwen3.5-2B"
    if "Qwen3-1.7B" in model: return "Qwen3-1.7B"
    return None


def load():
    # cell[(family, quant, device, task)] = {acc, p50, n}
    # Iterate files in filename (timestamp) order so that on equal-N ties the
    # NEWEST run wins — important for GPU cpu_routing where later SDPA re-runs
    # supersede earlier PA runs (PA has an accuracy bug on the 2B VLM, see report §8).
    cell = {}
    for f in sorted(glob.glob(str(RES / "*.json"))):
        try: d = json.load(open(f))
        except Exception: continue
        model = d.get("model", ""); dev = d.get("device", "")
        fam = family_of(model); q = quant_of(model)
        if not fam or not q or dev not in DEVS: continue
        task = d.get("dataset", "cpu_routing")
        n = d.get("samples", 0)
        p50 = (d.get("latency_ms") or {}).get("p50")
        k = (fam, q, dev, task)
        # keep the largest-N run per cell; on ties keep the newest (>= + sorted order)
        if k not in cell or n >= cell[k]["n"]:
            cell[k] = {"acc": d.get("accuracy"), "p50": p50, "n": n}
    return cell


def acc_chart(cell, fam):
    quants = [q for q in QORDER if any((fam, q, dv, t) in cell for dv in DEVS for t in TASKS)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    fp16 = {t: cell.get((fam, "FP16", "GPU.1", t)) or cell.get((fam, "FP16", "GPU.0", t)) for t in TASKS}
    for ax, task in zip(axes, TASKS):
        x = np.arange(len(quants)); w = 0.26
        for i, dv in enumerate(DEVS):
            vals = [(cell.get((fam, q, dv, task)) or {}).get("acc") for q in quants]
            xs = [xx + (i - 1) * w for xx, v in zip(x, vals) if v is not None]
            ys = [v * 100 for v in vals if v is not None]
            if ys: ax.bar(xs, ys, w, label=DEV_LABEL[dv], edgecolor="black", linewidth=0.4,
                          color=plt.cm.Set2(i))
        ref = fp16.get(task)
        if ref and ref.get("acc"): ax.axhline(ref["acc"] * 100, ls="--", c="gray", lw=1,
                                               label=f"FP16={ref['acc']*100:.0f}%")
        ax.set_title(TASK_LABEL[task]); ax.set_xticks(x)
        ax.set_xticklabels(quants, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(f"{fam} — accuracy by quantization × task × device", fontsize=13)
    fig.tight_layout(); p = OUT / f"acc_{fam}.png"; fig.savefig(p, dpi=140); plt.close(fig)
    return p


def lat_chart(cell, fam):
    quants = [q for q in QORDER if any((fam, q, dv, "clinc") in cell for dv in DEVS)]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(quants)); w = 0.26
    for i, dv in enumerate(DEVS):
        vals = [(cell.get((fam, q, dv, "clinc")) or {}).get("p50") for q in quants]
        xs = [xx + (i - 1) * w for xx, v in zip(x, vals) if v]
        ys = [v for v in vals if v]
        if ys: ax.bar(xs, ys, w, label=DEV_LABEL[dv], edgecolor="black", linewidth=0.4,
                      color=plt.cm.Set2(i))
    ax.set_xticks(x); ax.set_xticklabels(quants, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("p50 per-query latency (ms, CLINC)"); ax.grid(axis="y", alpha=0.3); ax.legend()
    ax.set_title(f"{fam} — latency (1-token, CLINC prompt) by quant × device")
    fig.tight_layout(); p = OUT / f"lat_{fam}.png"; fig.savefig(p, dpi=140); plt.close(fig)
    return p


def main():
    cell = load()
    made = []
    lines = ["| family | quant | device | task | acc | p50 ms | N |", "|---|---|---|---|---:|---:|---:|"]
    for k in sorted(cell):
        fam, q, dv, t = k; c = cell[k]
        lines.append(f"| {fam} | {q} | {DEV_LABEL[dv]} | {t} | "
                     f"{(c['acc'] or 0)*100:.1f}% | {c['p50'] or '-'} | {c['n']} |")
    (OUT / "summary_table.md").write_text("\n".join(lines), encoding="utf-8")
    for fam in ["Qwen3-1.7B", "Qwen3.5-2B"]:
        if any(f == fam for (f, *_ ) in cell):
            made.append(acc_chart(cell, fam)); made.append(lat_chart(cell, fam))
    print("charts:", *[p.name for p in made], sep="\n  ")
    print(f"cells: {len(cell)} -> charts/summary_table.md")


if __name__ == "__main__":
    main()
