"""Generic long-CSV adapter: point it at ANY table of (series keys, date, target) with a YAML column
mapping. This is the 'use it on your next project' entry point."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecastkit.features.calendar import calendar_features
from forecastkit.schema import Dataset
from forecastkit.utils import log, register_dataset


@register_dataset("csv")
def load_csv(path, series_cols, date_col, target_col, freq="D", static_cols=(), known_cols=(), available_col=None,
             missing_rows="zero", season=7, start=None, end=None, name="csv", cat_cols=None, item_col=None,
             day_slice_cols=()) -> Dataset:
    series_cols = [series_cols] if isinstance(series_cols, str) else list(series_cols)
    static_cols, known_cols = list(static_cols), list(known_cols)
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    if start:
        df = df[df[date_col] >= pd.Timestamp(start)]
    if end:
        df = df[df[date_col] <= pd.Timestamp(end)]
    df["_sid"] = df[series_cols].astype(str).agg("_".join, axis=1)
    dates = pd.date_range(df[date_col].min(), df[date_col].max(), freq=freq)
    sids = np.array(sorted(df["_sid"].unique()))
    N, T = len(sids), len(dates)
    row = {s: i for i, s in enumerate(sids)}
    di = dates.get_indexer(df[date_col])
    ri = df["_sid"].map(row).to_numpy()
    if (di < 0).any():
        raise ValueError(f"some dates do not align with freq={freq!r}")

    Y = np.zeros((N, T), dtype=np.float32)
    seen = np.zeros((N, T), dtype=bool)
    Y[ri, di] = df[target_col].to_numpy(np.float32)
    seen[ri, di] = True
    first = np.where(seen.any(1), seen.argmax(1), T)
    last = np.where(seen.any(1), T - 1 - seen[:, ::-1].argmax(1), -1)
    idx = np.arange(T)[None, :]
    avail = (idx >= first[:, None]) & (idx <= last[:, None])  # outside the series' life: not demand
    if missing_rows == "unavailable":
        avail &= seen
    if available_col:
        A = np.zeros((N, T), dtype=bool)
        A[ri, di] = df[available_col].astype(bool).to_numpy()
        avail &= A
    Y = np.where(avail, Y, 0.0).astype(np.float32)

    cal, cal_names = calendar_features(dates)
    parts = [np.broadcast_to(cal[None], (N, T, cal.shape[1]))]
    names = list(cal_names)
    for c in known_cols:
        M = np.full((N, T), np.nan, dtype=np.float32)
        M[ri, di] = df[c].to_numpy(np.float32)
        M = pd.DataFrame(M.T).ffill().bfill().to_numpy().T  # carry forward within series
        parts.append(np.nan_to_num(M)[:, :, None].astype(np.float32)); names.append(c)
    K = np.concatenate(parts, axis=2).astype(np.float32)

    first_rows = df.drop_duplicates("_sid").set_index("_sid")
    static = pd.DataFrame(index=sids)
    for c in dict.fromkeys(series_cols + static_cols):
        static[c] = first_rows.loc[sids, c].astype(str).to_numpy()
    cat_cols = list(cat_cols) if cat_cols is not None else [c for c in static.columns if c != item_col]
    ds = Dataset(name=name, series_ids=sids, dates=dates, Y=Y, avail=avail, K=K, known_names=names,
                 static=static, season=season, freq=freq,
                 meta={"static_cat_cols": cat_cols, "item_col": item_col, "day_slice_cols": list(day_slice_cols),
                       "source": str(path)})
    log.info("CSV dataset %s: %d series x %d steps, covariates=%s", name, N, T, names[-len(known_cols):] if known_cols else [])
    return ds
