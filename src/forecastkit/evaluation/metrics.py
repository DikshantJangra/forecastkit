"""Pure-numpy metric definitions (kept tiny and unit-tested on toy data).

WAPE is AGGREGATED (sum|error| / sum actual), never a mean of per-series ratios.
Point metrics: MAE & WAPE use the MEDIAN forecast (q0.5); RMSE uses the MEAN forecast.
"""
from __future__ import annotations

import numpy as np


def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


def wape(y, f):
    d = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - f)) / d) if d > 0 else float("nan")


def pinball(y, f, q):
    """mean pinball (quantile) loss of forecast f at level q."""
    d = y - f
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def mean_pinball(y, F, qs):
    """F [..., Q] forecasts for quantile levels qs; average over observations and levels."""
    return float(np.mean([pinball(y, F[..., j], q) for j, q in enumerate(qs)]))


def wql(y, F, qs):
    """weighted quantile loss: mean over q of 2 * sum(pinball) / sum(|y|)."""
    d = np.sum(np.abs(y))
    if d == 0:
        return float("nan")
    out = []
    for j, q in enumerate(qs):
        e = y - F[..., j]
        out.append(2 * np.sum(np.maximum(q * e, (q - 1) * e)) / d)
    return float(np.mean(out))


def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def width(lo, hi):
    return float(np.mean(hi - lo))
