"""Draws the two explanatory figures used in README / report / slides:
   docs/architecture.png  and  docs/rolling_origin.png      (python scripts/make_diagrams.py)"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs"
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fc, ec="#333", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.3, color="#444"))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.6); ax.axis("off")
    # adapters
    for i, t in enumerate(["M5 adapter\n(Walmart)", "CSV adapter\n(any table + YAML map)", "your adapter\n@register_dataset"]):
        box(ax, 0.1, 3.95 - i * 1.25, 2.0, 0.95, t, "#DCEBFA", fs=8)
        arrow(ax, 2.1, 4.42 - i * 1.25, 2.7, 3.0)
    ax.text(1.1, 5.1, "ADAPTERS", ha="center", fontsize=9, fontweight="bold", color="#2b5d9b")
    box(ax, 2.7, 2.2, 2.0, 1.6, "Canonical Dataset\nY [N,T]  avail [N,T]\nK [N,T,k] covariates\nstatic table", "#FFF2CC", bold=True, fs=8)
    box(ax, 5.1, 2.2, 1.9, 1.6, "Masks\nmark, never impute\nleading zeros\nstockout proxy", "#FDE0DD")
    arrow(ax, 4.7, 3.0, 5.1, 3.0)
    box(ax, 7.4, 2.2, 2.0, 1.6, "ForecastContext\nhistory <= origin ONLY\nleak-proof by design", "#E5F5E0", bold=True, fs=8)
    arrow(ax, 7.0, 3.0, 7.4, 3.0)
    # models
    ax.text(10.8, 5.25, "MODELS (plug-ins)", ha="center", fontsize=9, fontweight="bold", color="#2b6a2b")
    for i, (t, c) in enumerate([("Naive / seasonal naive", "#EDEDED"), ("Kalman (local)", "#D9F0D3"), ("LightGBM (tabular)", "#D9F0D3"),
                                ("Kalman->LGBM hybrid", "#D9F0D3"), ("DeepAR (global DL)", "#D9F0D3"), ("your model: fit / predict", "#EDEDED")]):
        box(ax, 9.9, 4.55 - i * 0.75, 2.0, 0.6, t, c, fs=8)
    arrow(ax, 9.4, 3.0, 9.9, 3.0)
    # runner and evaluation
    box(ax, 7.4, 0.3, 2.0, 1.1, "Rolling-origin\nbacktester", "#E5F5E0")
    arrow(ax, 8.4, 2.2, 8.4, 1.4)
    box(ax, 4.6, 0.3, 2.4, 1.1, "ONE long forecast file\nmodel|series|origin|h|stat|value", "#FFF2CC", fs=8, bold=True)
    arrow(ax, 7.4, 0.85, 7.0, 0.85)
    box(ax, 1.6, 0.3, 2.4, 1.1, "Evaluator\nMAE RMSE WAPE pinball\ncoverage / width / slices", "#FDE0DD")
    arrow(ax, 4.6, 0.85, 4.0, 0.85)
    box(ax, 0.1, 0.3, 1.2, 1.1, "report\n+ figures", "#DCEBFA", fs=8)
    arrow(ax, 1.6, 0.85, 1.3, 0.85)
    ax.set_title("forecastkit: nothing downstream of the adapter knows which dataset it is", fontsize=11)
    fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def rolling():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 6); ax.axis("off")
    rows = [("validation (tuning only)", 52, 24), ("test origin 1", 60, 28), ("test origin 2", 68, 28), ("test origin 3", 76, 28), ("test origin 4", 84, 28)]
    rows = [("validation origin", 44, 8, "#BAB0AC"), ("test origin 1", 52, 8, "#4C78A8"), ("test origin 2", 60, 8, "#4C78A8"),
            ("test origin 3", 68, 8, "#4C78A8"), ("test origin 4", 76, 8, "#4C78A8")]
    for i, (lab, o, h, c) in enumerate(rows):
        y = 5 - i
        ax.add_patch(plt.Rectangle((0, y - .3), o, .6, fc="#E5F5E0", ec="#555"))
        ax.add_patch(plt.Rectangle((o, y - .3), h, .6, fc=c, ec="#555", alpha=.85))
        ax.text(-1, y, lab, ha="right", va="center", fontsize=8)
        ax.text(o / 2, y, "train: data <= origin", ha="center", va="center", fontsize=7)
        ax.text(o + h / 2, y, "H=28", ha="center", va="center", fontsize=7, color="white")
        ax.plot([o, o], [y - .4, y + .4], "k-", lw=1.5)
    ax.text(84.5, 0.0, "last origin's window\nends on the last day", fontsize=7, ha="left", va="center")
    ax.set_title("Rolling-origin backtest: every model, same origins, same series, same horizon; tuning never touches test windows", fontsize=10)
    fig.savefig(OUT / "rolling_origin.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture(); rolling()
    print("wrote", OUT / "architecture.png", OUT / "rolling_origin.png")
