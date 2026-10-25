"""Builds the leak-proof `ForecastContext` for one forecast origin.

Leakage guards (all tested in tests/):
  * target arrays are cut at the origin - future targets are not even present in the object
  * availability tweaks (leading zeros) and the stockout proxy are RECOMPUTED on the history slice,
    so a flag on day t never depends on what happens after the origin
  * simulated cold-start series only expose their last `history_days` days
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecastkit.masks import leading_zero_mask, stockout_mask
from forecastkit.schema import Dataset, ForecastContext


def build_context(ds: Dataset, origin_idx: int, horizon: int, mask_cfg: dict | None, cold: np.ndarray | None,
                  cold_days: int, quantiles, static, seed: int = 0) -> ForecastContext:
    Th = origin_idx + 1
    if Th + horizon > ds.T:
        raise ValueError("forecast window runs past the end of the data")
    mask_cfg = mask_cfg or {}
    y_raw = ds.Y[:, :Th].astype(np.float64)
    avail = ds.avail[:, :Th].copy()
    if cold is None:
        cold = np.zeros(ds.N, dtype=bool)
    if cold.any():
        avail[cold, : max(Th - cold_days, 0)] = False
    if mask_cfg.get("leading_zeros", True):
        avail &= ~leading_zero_mask(y_raw, avail)
    cens = stockout_mask(y_raw, avail, mask_cfg.get("stockout"))
    y_raw = np.where(avail, y_raw, np.nan)
    y = np.where(cens, np.nan, y_raw)
    return ForecastContext(
        origin=ds.dates[origin_idx], horizon=horizon, series_ids=ds.series_ids,
        hist_dates=ds.dates[:Th], fut_dates=ds.dates[Th:Th + horizon], y=y, y_raw=y_raw, available=avail,
        censored=cens, known_hist=ds.K[:, :Th], known_fut=ds.K[:, Th:Th + horizon], known_names=ds.known_names,
        static=static, cold=cold, quantiles=tuple(quantiles), season=ds.season, seed=seed, origin_index=origin_idx,
        meta=dict(ds.meta))
