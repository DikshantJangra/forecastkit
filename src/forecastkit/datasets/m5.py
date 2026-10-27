"""M5 Forecasting - Accuracy adapter (Walmart). Raw files expected in `raw_dir`:
calendar.csv, sell_prices.csv, sales_train_evaluation.csv   (download from the Kaggle competition page)

What this adapter does (and ONLY this adapter knows about M5):
  * freezes a stratified subset (stores x items) to a CSV and reuses it
  * melts wide -> a complete [series x day] grid, maps d_ -> real dates
  * availability = the item has a weekly price entry (no price => not on sale => not demand)
  * known-future covariates: calendar, event flags, SNAP of the series' state, price features
    (price features are causal: they use only prices up to the current week)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecastkit.features.calendar import calendar_features
from forecastkit.schema import Dataset
from forecastkit.utils import log, register_dataset

from .subset import freeze, stratified_subset

STATIC_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


@register_dataset("m5")
def load_m5(raw_dir="data/raw/m5", subset_file=None, stores=("CA_1", "TX_1", "WI_1"), items_per_store=300,
            seed=42, last_days=None, sales_file="sales_train_evaluation.csv", name="m5") -> Dataset:
    raw = Path(raw_dir)
    for f in ("calendar.csv", "sell_prices.csv", sales_file):
        if not (raw / f).exists():
            raise FileNotFoundError(
                f"{raw / f} not found. Download the M5 files from "
                "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data into data/raw/m5/ "
                "(or run `forecastkit synth --out data/raw/m5_synth` for a synthetic stand-in).")
    cal = pd.read_csv(raw / "calendar.csv")
    cal["date"] = pd.to_datetime(cal["date"])
    head = pd.read_csv(raw / sales_file, nrows=0).columns
    d_cols = [c for c in head if c.startswith("d_")]
    dtypes = {c: np.int16 for c in d_cols}
    sales = pd.read_csv(raw / sales_file, dtype=dtypes)
    sales = sales[sales.store_id.isin(list(stores))]

    # ---- frozen stratified subset -----------------------------------------
    if subset_file and Path(subset_file).exists():
        ids = pd.read_csv(subset_file)["id"]
        sub = sales[sales["id"].isin(ids)]
        log.info("loaded frozen subset %s (%d series)", subset_file, len(sub))
    else:
        chosen = stratified_subset(sales, d_cols, list(stores), items_per_store, seed)
        sub = sales[sales["id"].isin(chosen["id"])]
        if subset_file:
            freeze(chosen, subset_file)
            log.info("froze new subset to %s (%d series) - commit this file", subset_file, len(sub))
    sub = sub.reset_index(drop=True)

    if last_days:
        d_cols = d_cols[-int(last_days):]
    cal = cal.set_index("d").loc[d_cols].reset_index()
    dates = pd.DatetimeIndex(cal["date"])
    Y = sub[d_cols].to_numpy(dtype=np.float32)
    N, T = Y.shape

    # ---- weekly prices -> availability + price features ---------------------
    weeks = cal["wm_yr_wk"].to_numpy()
    uw = np.unique(weeks)
    w_index = {w: i for i, w in enumerate(uw)}
    w_of_day = np.array([w_index[w] for w in weeks])
    # price lookup also needs a few weeks BEFORE the window for the rolling mean: read what exists
    pr = pd.read_csv(raw / "sell_prices.csv", dtype={"wm_yr_wk": np.int32, "sell_price": np.float32})
    pr = pr[pr.store_id.isin(list(stores)) & pr.item_id.isin(sub.item_id.unique())]
    key = sub[["item_id", "store_id"]].reset_index().rename(columns={"index": "row"})
    pr = pr.merge(key, on=["item_id", "store_id"], how="inner")
    pr = pr[pr.wm_yr_wk.isin(uw)]
    price_w = np.full((N, len(uw)), np.nan, dtype=np.float32)
    price_w[pr["row"].to_numpy(), pr["wm_yr_wk"].map(w_index).to_numpy()] = pr["sell_price"].to_numpy()
    avail = ~np.isnan(price_w[:, w_of_day])

    pw = pd.DataFrame(price_w.T)
    past_mean = pw.shift(1).rolling(26, min_periods=1).mean().to_numpy().T
    prev = pw.shift(1).to_numpy().T
    with np.errstate(all="ignore"):
        rel_change = np.nan_to_num(price_w / prev - 1.0)
        vs_mean = np.nan_to_num(price_w / past_mean - 1.0)
        logp = np.log(price_w)
    series_mean_logp = np.nan_to_num(np.nanmean(np.where(np.isfinite(logp), logp, np.nan), axis=1), nan=0.0)
    logp = np.where(np.isfinite(logp), logp, series_mean_logp[:, None])
    promo = (vs_mean < -0.05).astype(np.float32)

    # ---- known-future covariates ---------------------------------------------
    cal_arr, cal_names = calendar_features(dates)
    ev1 = cal["event_type_1"].astype(object).fillna("")
    ev2 = cal["event_type_2"].astype(object).fillna("")
    ev_flags = {f"event_{t.lower()}": ((ev1 == t) | (ev2 == t)).to_numpy(np.float32)
                for t in ("Religious", "National", "Cultural", "Sporting")}
    ev_flags["event_any"] = ((ev1 != "") | (ev2 != "")).to_numpy(np.float32)
    snap_mat = np.stack([cal[f"snap_{s}"].to_numpy(np.float32) for s in ("CA", "TX", "WI")], axis=0)
    state_idx = sub["state_id"].map({"CA": 0, "TX": 1, "WI": 2}).to_numpy()
    snap = snap_mat[state_idx]  # [N,T]

    parts, names = [], []
    parts.append(np.broadcast_to(cal_arr[None], (N, T, cal_arr.shape[1])))
    names += cal_names
    for n_, v in ev_flags.items():
        parts.append(np.broadcast_to(v[None, :, None], (N, T, 1)))
        names.append(n_)
    parts.append(snap[:, :, None]); names.append("snap")
    for n_, arr in (("log_price", logp), ("price_rel_change", rel_change), ("price_vs_mean", vs_mean), ("promo", promo)):
        parts.append(arr[:, w_of_day][:, :, None].astype(np.float32)); names.append(n_)
    K = np.concatenate(parts, axis=2).astype(np.float32)

    sid = sub["id"].str.replace(r"_(evaluation|validation)$", "", regex=True).to_numpy()
    static = sub[STATIC_COLS].copy()
    static.index = sid
    ds = Dataset(name=name, series_ids=sid, dates=dates, Y=Y, avail=avail, K=K, known_names=names,
                 static=static, season=7, freq="D",
                 meta={"static_cat_cols": ["dept_id", "cat_id", "store_id", "state_id"], "item_col": "item_id",
                       "day_slice_cols": ["event_any", "snap", "promo"], "source": "M5 (Walmart)"})
    log.info("M5 dataset: %d series x %d days, %d covariates, %.1f%% of cells available",
             ds.N, ds.T, K.shape[2], 100 * avail.mean())
    return ds
