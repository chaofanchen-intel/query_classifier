"""Perf charts from perf_results/perf.json (+ accuracy from results/*.json).

Outputs to charts/:
  - perf_ttft_<family>.png      : TTFT + TPOT + TPS grouped bars by quant x device
  - perf_tradeoff_<family>.png  : TTFT (1k-in) vs CLINC accuracy scatter — the decision chart
  - perf_summary.md             : machine-readable perf+acc table
"""
from __future__ import annotations
import json, glob
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).parent
OUT = EXP / "charts"; OUT.mkdir(exist_ok=True)


def qof(m):
    if m.endswith("-FP16"): return "FP16"
    if m.endswith("-INT8"): return "INT8"
    if "INT4-g128sym-aware" in m: return "INT4 g128sym+AWQ"
    if "INT4-g128sym-r08" in m:   return "INT4 g128sym r0.8"
    if "INT4-g128sym" in m:       return "INT4 g128sym"
    if "INT4-g128-aware" in m:    return "INT4 g128+AWQ"
    if "INT4-g128" in m:          return "INT4 g128"
    if m.endswith("-INT4"):       return "INT4 aggr(g=-1)"
    return m
def fam(m):
    if "Qwen3.5-2B" in m: return "Qwen3.5-2B"
    if "Qwen3-1.7B" in m: return "Qwen3-1.7B"
    return m

QORD = ["FP16","INT8","INT4 aggr(g=-1)","INT4 g128","INT4 g128+AWQ",
        "INT4 g128sym","INT4 g128sym r0.8","INT4 g128sym+AWQ"]
QCOLOR = {"FP16":"#7f7f7f","INT8":"#1f77b4","INT4 aggr(g=-1)":"#d62728",
          "INT4 g128":"#2ca02c","INT4 g128+AWQ":"#17becf","INT4 g128sym":"#ff9896",
          "INT4 g128sym r0.8":"#ff7f0e","INT4 g128sym+AWQ":"#e377c2"}
DEVL = {"GPU.0":"iGPU","GPU.1":"dGPU","NPU":"NPU"}


def load_perf():
    P = {}
    for r in json.load(open(EXP/"perf_results"/"perf.json")):
        P[(fam(r["model_name"]), qof(r["model_name"]), r["device"])] = r
    return P

def load_acc():
    # sorted() + >= so newest run wins on equal-N ties (SDPA re-runs supersede PA; see report §8)
    acc = defaultdict(dict); best = {}
    for f in sorted(glob.glob(str(EXP/"results"/"*.json"))):
        try: d = json.load(open(f))
        except: continue
        m = d.get("model","").rstrip("/").split("/")[-1]; dev = d.get("device","")
        if dev not in ("GPU.0","GPU.1","NPU"): continue
        t = d.get("dataset","cpu_routing"); n = d.get("samples",0)
        k = (fam(m), qof(m), dev, t)
        if k not in best or n >= best[k]:
            best[k] = n; acc[(fam(m), qof(m), dev)][t] = d.get("accuracy")
    return acc


def bar_chart(P, fm):
    devs = sorted({d for (f,q,d) in P if f==fm}, key=lambda x:{"GPU.0":0,"GPU.1":1,"NPU":2}[x])
    quants = [q for q in QORD if any((fm,q,d) in P for d in devs)]
    metrics = [("ttft_mean_ms","TTFT (ms)  ↓lower better"),
               ("tpot_mean_ms","TPOT (ms/token)  ↓"),
               ("throughput_mean_tps","Throughput (tok/s)  ↑higher better")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    for ax,(key,lab) in zip(axes, metrics):
        x = np.arange(len(quants)); w = 0.8/max(1,len(devs))
        for i,dv in enumerate(devs):
            vals = [P[(fm,q,dv)][key] if (fm,q,dv) in P else None for q in quants]
            xs = [xx+(i-(len(devs)-1)/2)*w for xx,v in zip(x,vals) if v is not None]
            ys = [v for v in vals if v is not None]
            bars = ax.bar(xs, ys, w, label=DEVL[dv], edgecolor="black", linewidth=0.4,
                          color=plt.cm.Set2(i))
            for bx,by in zip(xs,ys):
                ax.text(bx, by, f"{by:.0f}" if by>=10 else f"{by:.1f}", ha="center",
                        va="bottom", fontsize=6.5)
        ax.set_title(lab, fontsize=11); ax.set_xticks(x)
        ax.set_xticklabels(quants, rotation=40, ha="right", fontsize=7.5)
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"{fm} — TTFT / TPOT / Throughput (1k in → 512 out, n=10; GPU=PA, NPU=best-perf turbo)", fontsize=12)
    fig.tight_layout(); p = OUT/f"perf_ttft_{fm}.png"; fig.savefig(p, dpi=140); plt.close(fig)
    return p


def tradeoff_chart(P, A, fm):
    devs = sorted({d for (f,q,d) in P if f==fm}, key=lambda x:{"GPU.0":0,"GPU.1":1,"NPU":2}[x])
    marker = {"GPU.0":"o","GPU.1":"s","NPU":"^"}
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for dv in devs:
        for q in QORD:
            if (fm,q,dv) not in P: continue
            a = A.get((fm,q,dv),{}).get("clinc")
            if a is None: continue
            ttft = P[(fm,q,dv)]["ttft_mean_ms"]
            ax.scatter(ttft, a*100, s=130, marker=marker[dv], color=QCOLOR.get(q,"#333"),
                       edgecolor="black", linewidth=0.6, zorder=3)
            ax.annotate(f"{q}\n{DEVL[dv]}", (ttft, a*100), fontsize=6.8,
                        xytext=(5,4), textcoords="offset points")
    ax.set_xlabel("TTFT (ms) @ 1k input  —  ← faster")
    ax.set_ylabel("CLINC accuracy (%)  —  higher better ↑")
    ax.grid(alpha=0.3)
    ax.set_title(f"{fm} — TTFT vs accuracy tradeoff (top-left = fast & accurate)\n"
                 f"○ iGPU  □ dGPU  △ NPU", fontsize=11)
    fig.tight_layout(); p = OUT/f"perf_tradeoff_{fm}.png"; fig.savefig(p, dpi=140); plt.close(fig)
    return p


def main():
    P = load_perf(); A = load_acc()
    made = []
    lines = ["| family | quant | device | TTFT ms | TTFT p90 | TPOT ms | TPS | Decode TPS | CLINC | BANK | cpu |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (fm,q,dv) in sorted(P, key=lambda k:(k[0],{"GPU.0":0,"GPU.1":1,"NPU":2}[k[2]],QORD.index(k[1]) if k[1] in QORD else 9)):
        r = P[(fm,q,dv)]; a = A.get((fm,q,dv),{})
        def pc(t):
            v=a.get(t); return f"{v*100:.1f}" if v is not None else "—"
        lines.append(f"| {fm} | {q} | {DEVL[dv]} | {r['ttft_mean_ms']:.0f} | {r['ttft_p90_ms']:.0f} | "
                     f"{r['tpot_mean_ms']:.1f} | {r['throughput_mean_tps']:.1f} | {r['decode_throughput_mean_tps']:.1f} | "
                     f"{pc('clinc')} | {pc('banking_cluster10')} | {pc('cpu_routing')} |")
    (OUT/"perf_summary.md").write_text("\n".join(lines), encoding="utf-8")
    for fm in ["Qwen3-1.7B","Qwen3.5-2B"]:
        if any(f==fm for (f,_,_) in P):
            made.append(bar_chart(P,fm)); made.append(tradeoff_chart(P,A,fm))
    print("perf charts:", *[p.name for p in made], sep="\n  ")
    print("table -> charts/perf_summary.md")


if __name__ == "__main__":
    main()
