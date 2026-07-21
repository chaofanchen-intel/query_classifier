"""Dedicated chart for the 2B AWQ calibration experiment (cpu_routing is Chinese,
CLINC/BANKING English): accuracy by variant × task, with FP16 reference lines."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "charts"; OUT.mkdir(exist_ok=True)

# (label, cpu_routing(SDPA), CLINC, BANKING)
VARIANTS = [
    ("INT4 g128\n(plain)",        51.7, 75.7, 70.2),
    ("+AWQ zh",                   67.2, 75.8, 63.3),
    ("+AWQ mixed",                65.5, 72.4, 66.6),
    ("r0.8\n(no AWQ)",            55.2, 77.4, 73.8),
    ("+AWQ mixed\n+r0.8 ★",       82.8, 74.3, 69.2),
    ("INT8\n(ref)",               75.9, 76.2, 72.1),
]
FP16 = {"cpu_routing(zh)": 93.1, "CLINC(en)": 78.0, "BANKING(en)": 72.9}
TASKS = ["cpu_routing(zh)", "CLINC(en)", "BANKING(en)"]
COLORS = ["#d62728", "#2ca02c", "#1f77b4"]

fig, ax = plt.subplots(figsize=(12, 5.6))
x = np.arange(len(VARIANTS)); w = 0.26
for i, t in enumerate(TASKS):
    vals = [v[i + 1] for v in VARIANTS]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=t, color=COLORS[i],
                  edgecolor="black", linewidth=0.4)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.5, f"{val:.0f}",
                ha="center", va="bottom", fontsize=7)
    ax.axhline(list(FP16.values())[i], ls="--", lw=1, color=COLORS[i], alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([v[0] for v in VARIANTS], fontsize=8)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
ax.legend(title="task (dashed = FP16 ref)", fontsize=8)
ax.set_title("Qwen3.5-2B INT4 — Chinese/mixed AWQ calibration recovers cpu_routing(zh)\n"
             "(cpu_routing=SDPA; CLINC/BANKING=PA==SDPA)  ★=best balance", fontsize=11)
fig.tight_layout()
p = OUT / "awq_2b_calibration.png"; fig.savefig(p, dpi=140); plt.close(fig)
print("wrote", p)
