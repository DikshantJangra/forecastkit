"""Draws the Phase 2 hybrid architecture diagram -> results/phase2/figures/hybrid_architecture.png (no data used)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("results/phase2/figures/hybrid_architecture.png")


def box(ax, x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=10.5, weight="bold", color="#1f1d1a")
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.085 - i * 0.045, s, ha="center", va="top", fontsize=8.3, color="#3a362f")


def arrow(ax, a, b, text=None, off=(0, 0)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#55514a"))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, fontsize=7.6, color="#55514a", ha="center")


def main():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.26, 0.80, 0.48, 0.17, "Leak-safe context at origin T",
        ["recorded sales y(1:T), availability, history-only proxy c(T)", "known covariates x(1:T+H), static codes"],
        "#f3f1ec", "#9a958a")
    box(ax, 0.02, 0.36, 0.42, 0.36, "DL: DeepAR v2 (masked NB-LSTM)",
        ["inputs: y(t-1), y(t-7), y(t-14), y(t-28), flags,", "log scale, covariates, embeddings",
         "loss masked on proxy-flagged / unavailable days", "validation block = last 28 d of history",
         "-> S = 200 paths, per-path NB(mu, alpha)", "-> exact p_D(0) and p_D(y | y > 0)"],
        "#fdf0e3", "#d9822b")
    box(ax, 0.56, 0.36, 0.42, 0.36, "ML: zero-run persistence (LightGBM)",
        ["target: 1[recorded sales at T+h = 0]", "features: current zero-run length, ADI,",
         "completed-run stats, rolling non-zero rates,", "lags, calendar, price, hierarchy, h",
         "binary log loss, early stopping on history", "-> pi_h = P(y(T+h) = 0 | F_T)"],
        "#e6f3ee", "#1a8a6a")
    box(ax, 0.22, 0.03, 0.56, 0.24, "Hurdle fusion (one valid count distribution)",
        ["q(0) = pi_h      q(y) = (1 - pi_h) p_D(y) / (1 - p_D(0)),  y >= 1",
         "quantiles: 0 if tau <= pi_h, else F+^-1((tau - pi_h)/(1 - pi_h))",
         "mean = (1 - pi_h) E_D[y | y > 0];   ablation: naive ZI counts zeros twice"],
        "#ffffff", "#1f1d1a")
    arrow(ax, (0.40, 0.80), (0.23, 0.72))
    arrow(ax, (0.60, 0.80), (0.77, 0.72))
    arrow(ax, (0.23, 0.36), (0.40, 0.27), "size of positive sales", (-0.10, 0.0))
    arrow(ax, (0.77, 0.36), (0.60, 0.27), "probability of a zero", (0.10, 0.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
