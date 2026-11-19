"""Causal zero-run features for the zero-run persistence model.

Input is the RECORDED sales history up to the forecast origin (NaN = series not available, e.g. before launch).
Every array returned here has shape [N, T] and its value on day t depends only on days <= t, so evaluating it at
an anchor/origin never uses later data (tested in tests/test_zerorun.py by perturbing the future).

Terminology: a *zero run* is a maximal block of consecutive recorded-zero days. Long runs MAY be stockouts,
delistings or genuinely absent demand; M5 has no inventory data, so nothing here claims to detect stockouts.
"""
from __future__ import annotations

import numpy as np

# features that describe the zero-run structure explicitly (dropped together in the "no persistence features"
# ablation); `days_since_sale` comes from features.tabular.recency_features and equals the current run length.
PERSISTENCE_FEATURES = ("zr_run", "zr_run_over_adi", "zr_flag", "zr_runs_done", "zr_mean_run", "zr_max_run",
                        "zr_long_runs", "zr_run_vs_max", "days_since_sale")


def current_run_length(Y: np.ndarray) -> np.ndarray:
    """[N,T] days since the last positive sale (0 on a selling day); NaN before the first sale."""
    _N, T = Y.shape
    pos = np.nan_to_num(Y) > 0
    idx = np.arange(T)[None, :]
    last = np.maximum.accumulate(np.where(pos, idx, -1), axis=1)
    return np.where(last >= 0, idx - last, np.nan)


def completed_run_stats(Y: np.ndarray, long_run: int = 28) -> dict:
    """Statistics of zero runs that have ENDED (a positive sale observed) on or before day t.

    Returns [N,T] arrays: number of completed runs, their mean and max length and the number with length >=
    long_run. A run that is still open on day t is not included (its final length is unknown at t)."""
    N, T = Y.shape
    pos = np.nan_to_num(Y) > 0
    out = {k: np.zeros((N, T), dtype=np.float32) for k in ("zr_runs_done", "zr_mean_run", "zr_max_run", "zr_long_runs")}
    days = np.arange(T)
    for i in range(N):
        p = np.flatnonzero(pos[i])
        if len(p) < 2:
            continue
        gaps = (p[1:] - p[:-1] - 1).astype(np.float64)   # run length closed by the sale at p[1:]
        is_run = gaps > 0
        cnt = np.cumsum(is_run)
        sm = np.cumsum(gaps)
        mx = np.maximum.accumulate(gaps)
        lg = np.cumsum(gaps >= long_run)
        j = np.searchsorted(p[1:], days, side="right") - 1  # last run-closing sale <= t
        ok = j >= 0
        jj = np.clip(j, 0, None)
        c = np.where(ok, cnt[jj], 0)
        out["zr_runs_done"][i] = c
        out["zr_mean_run"][i] = np.where(ok & (c > 0), sm[jj] / np.maximum(c, 1), 0)
        out["zr_max_run"][i] = np.where(ok, mx[jj], 0)
        out["zr_long_runs"][i] = np.where(ok, lg[jj], 0)
    return out


def expanding_intermittency(Y: np.ndarray) -> dict:
    """[N,T] expanding ADI (available days per positive sale) and count of positive sales up to day t."""
    obs = ~np.isnan(Y)
    pos = (np.nan_to_num(Y) > 0) & obs
    n_obs = np.cumsum(obs, 1)
    n_nz = np.cumsum(pos, 1)
    with np.errstate(all="ignore"):
        adi = np.where(n_nz > 0, n_obs / np.maximum(n_nz, 1), np.nan)
    return {"zr_adi": adi, "zr_nnz": np.log1p(n_nz), "_n_nz": n_nz}


def zero_run_features(Y: np.ndarray, flag_min_run: int = 28, flag_adi_mult: float = 3.0, flag_min_nz: int = 10,
                      long_run: int = 28) -> dict:
    """All causal zero-run day features [N,T] for recorded sales Y (NaN = unavailable).

    `zr_flag` is the causal analogue of the Phase 1 stockout proxy: the run open on day t is already at least
    max(flag_min_run, flag_adi_mult * ADI_t) days long in a series with >= flag_min_nz sales so far."""
    run = current_run_length(Y)
    it = expanding_intermittency(Y)
    adi = it["zr_adi"]
    thr = np.maximum(flag_min_run, flag_adi_mult * np.nan_to_num(adi, nan=np.inf))
    f = {"zr_run": run, "zr_adi": adi, "zr_nnz": it["zr_nnz"]}
    with np.errstate(all="ignore"):
        f["zr_run_over_adi"] = run / adi
        f["zr_flag"] = ((np.nan_to_num(run) >= thr) & (it["_n_nz"] >= flag_min_nz)).astype(np.float32)
    f.update(completed_run_stats(Y, long_run))
    with np.errstate(all="ignore"):
        f["zr_run_vs_max"] = run / np.maximum(f["zr_max_run"], 1.0)
    return f
