"""Availability and stockout handling. Rule: MARK, never impute.

* `leading_zero_mask`  - zeros before a series' first sale are "not yet launched", not demand.
* `stockout_mask`      - PROXY (not ground truth): a zero run far longer than the series' normal
                         inter-demand interval, in a series that otherwise sells regularly.

The stockout rule is a plain, swappable function: other businesses define stockouts differently
(register your own via `STOCKOUT_RULES`).  All rules take only data up to the end of the array they
are given, so applying them to a *history slice* is leak-free (the runner does exactly that).
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------------


def leading_zero_mask(Y: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """True for available days before the first positive sale of each series."""
    pos = (np.nan_to_num(Y) > 0) & avail
    first = np.where(pos.any(1), pos.argmax(1), Y.shape[1])  # no sale at all -> everything is leading
    idx = np.arange(Y.shape[1])[None, :]
    return (idx < first[:, None]) & avail


def _flag_runs_1d(y, a, min_run, adi_mult, min_nz):
    out = np.zeros(len(y), dtype=bool)
    nz = (y > 0) & a
    n_nz = int(nz.sum())
    if n_nz < min_nz:
        return out
    adi = a.sum() / n_nz
    thr = max(min_run, adi_mult * adi)
    zero = (a & (y == 0)).astype(np.int8)
    d = np.diff(np.concatenate(([0], zero, [0])))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        if e - s >= thr:
            out[s:e] = True
    return out


def zero_run_stockout(Y: np.ndarray, avail: np.ndarray, min_run: int = 28, adi_multiplier: float = 3.0,
                      min_nonzero_events: int = 10, **_) -> np.ndarray:
    """Flag zero runs with length >= max(min_run, adi_multiplier * ADI) in series with enough sales events."""
    Y = np.nan_to_num(np.asarray(Y, dtype=np.float64))
    out = np.zeros(Y.shape, dtype=bool)
    for i in range(Y.shape[0]):
        out[i] = _flag_runs_1d(Y[i], avail[i], min_run, adi_multiplier, min_nonzero_events)
    return out


STOCKOUT_RULES = {"zero_run": zero_run_stockout}


def stockout_mask(Y: np.ndarray, avail: np.ndarray, cfg: dict | None) -> np.ndarray:
    """cfg: {enabled: bool, rule: 'zero_run' (or 'pkg.mod:fn'), min_run, adi_multiplier, ...}"""
    if not cfg or not cfg.get("enabled", True):
        return np.zeros(Y.shape, dtype=bool)
    rule = cfg.get("rule", "zero_run")
    if rule in STOCKOUT_RULES:
        fn = STOCKOUT_RULES[rule]
    else:
        import importlib
        mod, _, attr = rule.partition(":")
        fn = getattr(importlib.import_module(mod), attr)
    params = {k: v for k, v in cfg.items() if k not in ("enabled", "rule")}
    return fn(Y, avail, **params)
