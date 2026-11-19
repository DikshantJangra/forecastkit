"""Probability-forecast scores for binary events (here: recorded zero sales).

Brier = mean (p - z)^2 (proper; lower is better); log loss = -mean [z log p + (1-z) log(1-p)] (proper);
ECE = sum_b (n_b / n) |mean z_b - mean p_b| over equal-width probability bins (calibration only);
ROC-AUC / PR-AUC measure ranking (discrimination) only and ignore calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def brier(p, z):
    return float(np.mean((p - z) ** 2))


def log_loss(p, z, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(z * np.log(p) + (1 - z) * np.log(1 - p)))


def reliability_table(p, z, n_bins: int = 10) -> pd.DataFrame:
    p, z = np.asarray(p, float), np.asarray(z, float)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        if m.any():
            rows.append({"bin": k, "lo": k / n_bins, "hi": (k + 1) / n_bins, "n": int(m.sum()),
                         "mean_pred": float(p[m].mean()), "obs_rate": float(z[m].mean())})
    return pd.DataFrame(rows)


def ece(p, z, n_bins: int = 10) -> float:
    t = reliability_table(p, z, n_bins)
    if not len(t):
        return float("nan")
    return float((t.n * (t.obs_rate - t.mean_pred).abs()).sum() / t.n.sum())


def binary_scores(p, z) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    p, z = np.asarray(p, float), np.asarray(z, float)
    out = {"n": int(len(z)), "base_rate": float(z.mean()) if len(z) else np.nan, "brier": brier(p, z),
           "log_loss": log_loss(p, z), "ece": ece(p, z)}
    clim = float(z.mean()) if len(z) else np.nan
    out["brier_skill_vs_base_rate"] = 1 - out["brier"] / brier(np.full_like(z, clim), z) if len(z) else np.nan
    both = len(np.unique(z)) == 2
    out["roc_auc"] = float(roc_auc_score(z, p)) if both else np.nan
    out["pr_auc_zero"] = float(average_precision_score(z, p)) if both else np.nan
    return out
