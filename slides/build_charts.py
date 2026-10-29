"""Build the result charts for the Phase 1 slides directly from the benchmark CSVs (no hand-typed numbers).

Usage (repo root):  /tmp/fkvenv/bin/python slides/build_charts.py
Reads  outputs/m5_main/report/{main_all_days,main_non_censored,cold_start,origin_stability}.csv
Writes slides/figures/{main_wape,main_pinball,by_setting}.svg and slides/figures/values.json (for validation)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "outputs" / "m5_main" / "report"
OUT = ROOT / "slides" / "figures"

# one name and one colour per model, used everywhere in the deck
MODELS = [("snaive4", "Seasonal naive (4 wk)", "#9aa0a6"),
          ("kalman", "Kalman", "#4f8fbf"),
          ("lgbm", "LightGBM", "#7b6cc4"),
          ("kalman_lgbm", "Kalman–LightGBM", "#1a8a6a"),
          ("deepar", "DeepAR", "#d9822b"),
          ("ens_global_local", "Ensemble (equal weight)", "#b5577a")]

plt.rcParams.update({"font.family": "Helvetica", "font.size": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#c9c4b8", "axes.labelcolor": "#33302a",
                     "xtick.color": "#33302a", "ytick.color": "#33302a", "svg.fonttype": "none"})


def load():
    rd = lambda f: pd.read_csv(REP / f).set_index("model")
    return rd("main_all_days.csv"), rd("main_non_censored.csv"), rd("cold_start.csv"), rd("origin_stability.csv")


def hbar(ax, vals, labels, colors, fmt, xlim, xlabel):
    y = range(len(vals))[::-1]
    ax.barh(list(y), vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + (xlim[1] - xlim[0]) * 0.01, yi, fmt.format(v), va="center", fontsize=15, color="#1f1d1a")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.set_xlim(*xlim); ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#ebe7de"); ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m, nc, cs, st = load()
    keys = [k for k, _, _ in MODELS]; names = [n for _, n, _ in MODELS]; cols = [c for _, _, c in MODELS]

    # 1) all-days WAPE, axis from 0 so bar lengths are honest
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    hbar(ax, [m.loc[k, "WAPE"] for k in keys], names, cols, "{:.3f}", (0, 0.86), "WAPE (lower is better)")
    fig.tight_layout(); fig.savefig(OUT / "main_wape.svg", transparent=True); plt.close(fig)

    # 2) pinball: dot + ±1 std across the 4 origins, so the size of the gaps is shown against noise
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y = list(range(len(keys)))[::-1]
    for yi, k, c in zip(y, keys, cols):
        mu, sd = st.loc[k, "pinball_mean"], st.loc[k, "pinball_std"]
        ax.errorbar(mu, yi, xerr=sd, fmt="o", color=c, ecolor=c, elinewidth=3, capsize=5, markersize=10)
        ax.text(mu + sd + 0.0012, yi, f"{m.loc[k, 'pinball']:.5f}" if k in ("lgbm", "kalman_lgbm", "deepar", "ens_global_local")
                else f"{m.loc[k, 'pinball']:.3f}", va="center", fontsize=14)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0.250, 0.305)
    ax.set_xlabel("Pinball loss (lower is better), ±1 sd over origins")
    ax.grid(axis="x", color="#ebe7de"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "main_pinball.svg", transparent=True); plt.close(fig)

    # 3) WAPE in the three evaluation sets
    sel = ["kalman", "lgbm", "kalman_lgbm", "deepar", "ens_global_local"]
    sets = [("All days\n(124,992 cells)", m), ("Non-censored days\n(114,127 cells)", nc), ("Simulated cold start\n(9,408 cells)", cs)]
    fig, ax = plt.subplots(figsize=(12.6, 3.7))
    w = 0.16
    for j, k in enumerate(sel):
        name, col = [(n, c) for kk, n, c in MODELS if kk == k][0]
        xs = [i + (j - 2) * w for i in range(len(sets))]
        vs = [df.loc[k, "WAPE"] for _, df in sets]
        ax.bar(xs, vs, width=w * 0.92, color=col, label=name)
        for x, v in zip(xs, vs):
            ax.text(x, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=12, rotation=90)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s for s, _ in sets])
    ax.set_ylim(0.55, 0.91); ax.set_ylabel("WAPE")
    ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=13)
    ax.grid(axis="y", color="#ebe7de"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "by_setting.svg", transparent=True); plt.close(fig)

    vals = {"all": m.loc[keys, ["MAE", "RMSE", "WAPE", "pinball", "cov80", "cov90"]].to_dict("index"),
            "noncens": nc.loc[keys, ["WAPE", "pinball"]].to_dict("index"),
            "cold": cs.loc[keys, ["WAPE", "pinball"]].to_dict("index"),
            "pinball_std": st.loc[keys, "pinball_std"].to_dict()}
    (OUT / "values.json").write_text(json.dumps(vals, indent=1))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
