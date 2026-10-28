"""EDA for the M5 subset used by forecastkit.

One bounded, vectorized run. It loads the dataset THROUGH the repo loader (cache in
data/processed) and reuses the repo's own classify_demand, stockout_mask and
leading_zero_mask. It does not redefine those definitions.

Descriptive only. No EDA statistic is fed to any model. Models recompute masks and
classes causally per origin inside ForecastContext (see src/forecastkit/backtest/context.py).

Usage:
    python scripts/eda_m5.py \
        --config configs/m5.yaml \
        --fig-dir docs/figures/eda \
        --out-dir outputs/eda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastkit.config import load_config
from forecastkit.datasets import load_dataset
from forecastkit.features.intermittency import ADI_CUT, CV2_CUT, classify_demand
from forecastkit.masks import leading_zero_mask, stockout_mask

SEED = 42
DPI = 130


def parse_args():
    p = argparse.ArgumentParser(description="M5 subset EDA")
    p.add_argument("--config", default="configs/m5.yaml")
    p.add_argument("--fig-dir", default="docs/figures/eda")
    p.add_argument("--out-dir", default="outputs/eda")
    return p.parse_args()


def savefig(fig, fig_dir: Path, name: str):
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(SEED)
    fig_dir = Path(args.fig_dir)
    out_dir = Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    ds = load_dataset(cfg["dataset"], cache_dir=cfg.get("cache_dir", "data/processed"), use_cache=True)
    mask_cfg = cfg.get("masks", {})
    stockout_cfg = mask_cfg.get("stockout", {})

    N, T = ds.N, ds.T
    Y = np.asarray(ds.Y, dtype=np.float64)
    avail = ds.avail
    dates = ds.dates
    static = ds.static.reset_index().rename(columns={"index": "series_id"})
    static.columns = ["series_id"] + list(ds.static.columns)

    # ---- masks (repo functions, full window, descriptive only) -----------------
    lead = leading_zero_mask(Y, avail)
    eff_avail = avail & ~lead  # available and launched
    stock = stockout_mask(Y, eff_avail, stockout_cfg)

    # ---- intermittency classes (repo function, full window) --------------------
    info = classify_demand(Y, avail, end=T)
    info.index = ds.series_ids

    # ==========================================================================
    # CSV 1: dataset_summary
    # ==========================================================================
    summary = {
        "n_series": N,
        "n_days": T,
        "date_start": str(dates[0].date()),
        "date_end": str(dates[-1].date()),
        "n_covariates": ds.K.shape[2],
        "cells_total": N * T,
        "cells_available_pct": round(100 * avail.mean(), 4),
        "cells_launched_available_pct": round(100 * eff_avail.mean(), 4),
        "cells_stockout_flagged_pct": round(100 * stock.mean(), 4),
        "mean_daily_demand_per_series": round(float(np.nanmean(np.where(eff_avail, Y, np.nan))), 4),
        "overall_zero_share_launched_avail": round(float(((Y == 0) & eff_avail).sum() / eff_avail.sum()), 4),
        "season": ds.season,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "dataset_summary.csv", index=False)

    # ==========================================================================
    # Raw-file quality checks (read only needed rows/cols once)
    # ==========================================================================
    raw = Path(cfg["dataset"]["params"]["raw_dir"])
    stores = list(cfg["dataset"]["params"]["stores"])
    item_ids = set(ds.static["item_id"].unique())
    subset_ids = set(ds.series_ids)

    checks = []

    # sales file: read only id + d_ columns for subset rows
    sales_file = cfg["dataset"]["params"].get("sales_file", "sales_train_evaluation.csv")
    head = pd.read_csv(raw / sales_file, nrows=0).columns
    d_cols = [c for c in head if c.startswith("d_")]
    usecols = ["id", "store_id", "item_id"] + d_cols
    sales = pd.read_csv(raw / sales_file, usecols=usecols)
    sales["sid"] = sales["id"].str.replace(r"_(evaluation|validation)$", "", regex=True)
    sales_sub = sales[sales["sid"].isin(subset_ids)]
    sales_vals = sales_sub[d_cols].to_numpy()
    checks.append({"check": "sales_rows_in_subset", "value": int(len(sales_sub)), "note": "expected 1200"})
    checks.append({"check": "sales_duplicate_ids", "value": int(sales_sub["sid"].duplicated().sum()),
                   "note": "duplicate series ids in subset"})
    checks.append({"check": "sales_negative_values", "value": int((sales_vals < 0).sum()),
                   "note": "negative unit sales in raw"})
    checks.append({"check": "sales_null_values", "value": int(np.isnan(sales_vals.astype(float)).sum()),
                   "note": "missing cells in raw sales grid"})

    # sell_prices: filter to the 3 stores and subset items, read only needed cols
    pr = pd.read_csv(raw / "sell_prices.csv",
                     usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
                     dtype={"wm_yr_wk": np.int32, "sell_price": np.float32})
    pr = pr[pr["store_id"].isin(stores) & pr["item_id"].isin(item_ids)]
    checks.append({"check": "price_rows_subset", "value": int(len(pr)), "note": "weekly price rows for subset"})
    checks.append({"check": "price_duplicate_keys",
                   "value": int(pr.duplicated(subset=["store_id", "item_id", "wm_yr_wk"]).sum()),
                   "note": "duplicate store-item-week price rows"})
    checks.append({"check": "price_negative_or_zero", "value": int((pr["sell_price"] <= 0).sum()),
                   "note": "non-positive sell_price"})
    checks.append({"check": "price_null_values", "value": int(pr["sell_price"].isna().sum()),
                   "note": "missing sell_price"})

    # calendar: duplicate dates
    cal = pd.read_csv(raw / "calendar.csv", usecols=["date", "d", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI",
                                                     "event_type_1", "event_type_2"])
    checks.append({"check": "calendar_duplicate_dates", "value": int(cal["date"].duplicated().sum()),
                   "note": "duplicate calendar dates"})

    # processed grid checks
    checks.append({"check": "grid_null_where_available",
                   "value": int(np.isnan(np.where(avail, Y, 0)).sum()),
                   "note": "NaN target on available cells (should be 0)"})
    checks.append({"check": "grid_negative_targets", "value": int((Y[avail] < 0).sum()),
                   "note": "negative targets in processed grid"})
    pd.DataFrame(checks).to_csv(out_dir / "quality_checks.csv", index=False)

    # ==========================================================================
    # CSV: series_stats (per-series)
    # ==========================================================================
    n_avail = eff_avail.sum(1)
    n_zero = ((Y == 0) & eff_avail).sum(1)
    with np.errstate(all="ignore"):
        zero_share = np.where(n_avail > 0, n_zero / np.maximum(n_avail, 1), np.nan)
        mean_launched = np.where(n_avail > 0, (np.where(eff_avail, Y, 0)).sum(1) / np.maximum(n_avail, 1), np.nan)
        y_for_cv = np.where(eff_avail, Y, np.nan)
        std_all = np.nanstd(y_for_cv, axis=1)
        mean_all = np.nanmean(y_for_cv, axis=1)
        cv_all = np.where(mean_all > 0, std_all / mean_all, np.nan)
    stock_rate = np.where(n_avail > 0, stock.sum(1) / np.maximum(n_avail, 1), 0.0)

    series_stats = pd.DataFrame({
        "series_id": ds.series_ids,
        "item_id": ds.static["item_id"].to_numpy(),
        "dept_id": ds.static["dept_id"].to_numpy(),
        "cat_id": ds.static["cat_id"].to_numpy(),
        "store_id": ds.static["store_id"].to_numpy(),
        "state_id": ds.static["state_id"].to_numpy(),
        "adi": info["adi"].to_numpy(),
        "cv2": info["cv2"].to_numpy(),
        "intermittency": info["intermittency"].to_numpy(),
        "volume_tier": info["volume_tier"].to_numpy(),
        "mean_daily_demand": mean_launched,
        "cv": cv_all,
        "zero_share": zero_share,
        "n_available_days": n_avail,
        "stockout_flag_rate": stock_rate,
        "any_stockout_flag": (stock.sum(1) > 0),
    })
    series_stats.to_csv(out_dir / "series_stats.csv", index=False)

    # ==========================================================================
    # CSV: intermittency_counts
    # ==========================================================================
    ic = info["intermittency"].value_counts().rename_axis("class").reset_index(name="n_series")
    ic["pct"] = (100 * ic["n_series"] / N).round(2)
    ic.to_csv(out_dir / "intermittency_counts.csv", index=False)

    # ==========================================================================
    # Weekday / month / event / SNAP tables (long form, per day)
    # ==========================================================================
    # per (series, day) long frame of launched-available cells only
    day_avail = eff_avail
    # mean sales per day across series (use launched-available cells)
    masked_Y = np.where(day_avail, Y, np.nan)

    # weekday profile (normalised mean per category)
    dow = dates.dayofweek.to_numpy()
    month = dates.month.to_numpy()
    cat = ds.static["cat_id"].to_numpy()

    def profile_by(group_vals, group_name, levels):
        rows = []
        for c in np.unique(cat):
            rowmask = cat == c
            cat_daily = np.nanmean(masked_Y[rowmask], axis=0)  # mean over series per day
            overall = np.nanmean(cat_daily)
            for lv in levels:
                sel = group_vals == lv
                m = np.nanmean(cat_daily[sel])
                rows.append({group_name: int(lv), "cat_id": c,
                             "mean_sales": round(float(m), 5),
                             "normalised": round(float(m / overall), 5) if overall > 0 else np.nan})
        return pd.DataFrame(rows)

    wd = profile_by(dow, "weekday", range(7))
    wd.to_csv(out_dir / "weekday_profile.csv", index=False)
    mo = profile_by(month, "month", range(1, 13))
    mo.to_csv(out_dir / "month_profile.csv", index=False)

    # event comparison: event_any day vs non-event day, overall mean sales per day
    ev_any = ds.K[0, :, ds.known_names.index("event_any")].astype(bool)
    daily_overall = np.nanmean(masked_Y, axis=0)
    ev_tbl = pd.DataFrame({
        "group": ["event_day", "non_event_day"],
        "n_days": [int(ev_any.sum()), int((~ev_any).sum())],
        "mean_sales": [round(float(np.nanmean(daily_overall[ev_any])), 5),
                       round(float(np.nanmean(daily_overall[~ev_any])), 5)],
    })
    ev_tbl["ratio_vs_non_event"] = (ev_tbl["mean_sales"] / ev_tbl["mean_sales"].iloc[1]).round(4)
    ev_tbl.to_csv(out_dir / "event_table.csv", index=False)

    # SNAP: compare within the SAME state only. snap covariate is per-series (state-specific).
    snap_idx = ds.known_names.index("snap")
    snap_series = ds.K[:, :, snap_idx].astype(bool)  # [N,T] already state-specific per series
    snap_rows = []
    for st in np.unique(ds.static["state_id"].to_numpy()):
        rmask = ds.static["state_id"].to_numpy() == st
        yy = masked_Y[rmask]
        ss = snap_series[rmask]
        on = yy[ss]
        off = yy[~ss]
        m_on = float(np.nanmean(on))
        m_off = float(np.nanmean(off))
        snap_rows.append({"state_id": st,
                          "mean_sales_snap_day": round(m_on, 5),
                          "mean_sales_non_snap_day": round(m_off, 5),
                          "ratio_snap_vs_non": round(m_on / m_off, 4) if m_off > 0 else np.nan,
                          "n_snap_cells": int(np.isfinite(on).sum()),
                          "n_non_snap_cells": int(np.isfinite(off).sum())})
    snap_tbl = pd.DataFrame(snap_rows)
    snap_tbl.to_csv(out_dir / "snap_table.csv", index=False)

    # ==========================================================================
    # Price summary (within-series variation) and price vs sales relation
    # ==========================================================================
    log_price = ds.K[:, :, ds.known_names.index("log_price")]
    price = np.exp(log_price)  # recovered per-day price (per series)
    # within-series: any price change? relative changes?
    rel_change = ds.K[:, :, ds.known_names.index("price_rel_change")]  # price_t/price_{t-1} - 1
    changed = np.abs(rel_change) > 1e-6
    series_has_change = changed.any(1)
    price_summary = {
        "n_series": N,
        "share_series_with_any_price_change": round(float(series_has_change.mean()), 4),
        "mean_price_changes_per_series": round(float(changed.sum(1).mean()), 3),
        "median_abs_rel_change_nonzero": round(float(np.median(np.abs(rel_change[changed]))), 5)
        if changed.any() else 0.0,
        "p95_abs_rel_change_nonzero": round(float(np.percentile(np.abs(rel_change[changed]), 95)), 5)
        if changed.any() else 0.0,
    }
    pd.DataFrame([price_summary]).to_csv(out_dir / "price_summary.csv", index=False)

    # binned descriptive relation: relative price (vs series mean) vs relative sales (vs series mean)
    # use launched-available cells only
    with np.errstate(all="ignore"):
        series_mean_price = np.nanmean(np.where(eff_avail, price, np.nan), axis=1, keepdims=True)
        rel_price = price / series_mean_price - 1.0
        series_mean_sales = np.where(n_avail > 0, (np.where(eff_avail, Y, 0)).sum(1) / np.maximum(n_avail, 1), np.nan)
        rel_sales = np.where(series_mean_sales[:, None] > 0, Y / series_mean_sales[:, None] - 1.0, np.nan)
    valid = eff_avail & np.isfinite(rel_price) & np.isfinite(rel_sales)
    rp = rel_price[valid]
    rs = rel_sales[valid]
    bins = np.array([-np.inf, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, np.inf])
    binlabels = ["<-30%", "-30..-15%", "-15..-5%", "-5..5%", "5..15%", "15..30%", ">30%"]
    bidx = np.digitize(rp, bins[1:-1])
    pr_rel_rows = []
    for b in range(len(binlabels)):
        sel = bidx == b
        if sel.sum() == 0:
            continue
        pr_rel_rows.append({"rel_price_bin": binlabels[b], "n_cells": int(sel.sum()),
                            "mean_rel_sales": round(float(np.mean(rs[sel])), 5),
                            "median_rel_sales": round(float(np.median(rs[sel])), 5)})
    price_rel = pd.DataFrame(pr_rel_rows)
    price_rel.to_csv(out_dir / "price_vs_sales.csv", index=False)

    # ==========================================================================
    # Stockout run-length distribution and flag rate by class/category
    # ==========================================================================
    run_lengths = []
    for i in range(N):
        s = stock[i]
        if not s.any():
            continue
        d = np.diff(np.concatenate(([0], s.astype(np.int8), [0])))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        run_lengths.extend((ends - starts).tolist())
    run_lengths = np.array(run_lengths, dtype=int)

    so_by_class = series_stats.groupby("intermittency").agg(
        n_series=("series_id", "size"),
        share_series_any_flag=("any_stockout_flag", "mean"),
        mean_flag_rate=("stockout_flag_rate", "mean"),
    ).reset_index()
    so_by_class["share_series_any_flag"] = so_by_class["share_series_any_flag"].round(4)
    so_by_class["mean_flag_rate"] = so_by_class["mean_flag_rate"].round(5)
    so_by_class.to_csv(out_dir / "stockout_by_class.csv", index=False)

    # ==========================================================================
    # FIGURES
    # ==========================================================================
    # Fig 1: daily zero-sales share over time + per-series zero-share histogram
    with np.errstate(all="ignore"):
        daily_zero_share = ((Y == 0) & eff_avail).sum(0) / np.maximum(eff_avail.sum(0), 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(dates, daily_zero_share, lw=0.7, color="#1f77b4")
    ax[0].set_title(f"Daily zero-sales share over time (N={N} series)")
    ax[0].set_ylabel("share of launched-available series with 0 sales")
    ax[0].set_xlabel("date")
    ax[0].set_ylim(0, 1)
    ax[1].hist(zero_share[np.isfinite(zero_share)], bins=40, color="#1f77b4", edgecolor="white")
    ax[1].set_title(f"Per-series zero-share (n={int(np.isfinite(zero_share).sum())})")
    ax[1].set_xlabel("fraction of launched-available days with 0 sales")
    ax[1].set_ylabel("number of series")
    savefig(fig, fig_dir, "01_zero_share.png")

    # Fig 2: ADI vs CV2 scatter with SB cut lines
    adi = info["adi"].to_numpy()
    cv2 = info["cv2"].to_numpy()
    finite = np.isfinite(adi) & np.isfinite(cv2) & (adi > 0)
    cls = info["intermittency"].to_numpy()
    colors = {"smooth": "#2ca02c", "erratic": "#ff7f0e", "intermittent": "#1f77b4",
              "lumpy": "#d62728", "new_item": "#7f7f7f", "no_sales": "#000000"}
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for c in np.unique(cls):
        m = finite & (cls == c)
        ax.scatter(adi[m], np.maximum(cv2[m], 1e-3), s=10, alpha=0.5, label=f"{c} (n={int((cls==c).sum())})",
                   color=colors.get(c, "#999999"))
    ax.axvline(ADI_CUT, color="k", ls="--", lw=1)
    ax.axhline(CV2_CUT, color="k", ls="--", lw=1)
    ax.text(ADI_CUT * 1.02, ax.get_ylim()[1] * 0.5 if False else 1.0, f"ADI={ADI_CUT}", rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ADI = available days / nonzero events (log)")
    ax.set_ylabel("CV^2 of nonzero demand sizes (log)")
    ax.set_title(f"Syntetos-Boylan classes, cuts ADI={ADI_CUT} CV2={CV2_CUT} (N={N})")
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, fig_dir, "02_adi_cv2.png")

    # Fig 3: distribution of mean daily demand and CV across series (log)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    md = mean_launched[np.isfinite(mean_launched) & (mean_launched > 0)]
    ax[0].hist(md, bins=np.logspace(np.log10(md.min()), np.log10(md.max()), 40), color="#1f77b4",
               edgecolor="white")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("mean daily demand per series, units (log)")
    ax[0].set_ylabel("number of series")
    ax[0].set_title(f"Mean daily demand (n={len(md)})")
    cvv = cv_all[np.isfinite(cv_all) & (cv_all > 0)]
    ax[1].hist(cvv, bins=np.logspace(np.log10(cvv.min()), np.log10(cvv.max()), 40), color="#ff7f0e",
               edgecolor="white")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("CV of daily demand per series (log)")
    ax[1].set_ylabel("number of series")
    ax[1].set_title(f"Coefficient of variation (n={len(cvv)})")
    savefig(fig, fig_dir, "03_demand_cv.png")

    # Fig 4: weekday + month profiles by category, plus event/SNAP comparison
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for c in wd["cat_id"].unique():
        sub = wd[wd["cat_id"] == c].sort_values("weekday")
        ax[0, 0].plot(sub["weekday"], sub["normalised"], marker="o", label=c)
    ax[0, 0].set_xticks(range(7))
    ax[0, 0].set_xticklabels(dow_names)
    ax[0, 0].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[0, 0].set_ylabel("normalised mean sales (1.0 = category mean)")
    ax[0, 0].set_title("Weekday profile by category")
    ax[0, 0].legend(fontsize=8)
    for c in mo["cat_id"].unique():
        sub = mo[mo["cat_id"] == c].sort_values("month")
        ax[0, 1].plot(sub["month"], sub["normalised"], marker="o", label=c)
    ax[0, 1].set_xticks(range(1, 13))
    ax[0, 1].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[0, 1].set_ylabel("normalised mean sales (1.0 = category mean)")
    ax[0, 1].set_title("Month profile by category")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].bar(ev_tbl["group"], ev_tbl["mean_sales"], color=["#d62728", "#1f77b4"])
    ax[1, 0].set_ylabel("mean sales per launched-available cell (units)")
    ax[1, 0].set_title(f"Event day vs non-event day (n_days {ev_tbl['n_days'].tolist()})")
    x = np.arange(len(snap_tbl))
    w = 0.38
    ax[1, 1].bar(x - w / 2, snap_tbl["mean_sales_snap_day"], w, label="SNAP day", color="#2ca02c")
    ax[1, 1].bar(x + w / 2, snap_tbl["mean_sales_non_snap_day"], w, label="non-SNAP day", color="#9467bd")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(snap_tbl["state_id"])
    ax[1, 1].set_ylabel("mean sales per cell (units)")
    ax[1, 1].set_title("SNAP vs non-SNAP, within same state")
    ax[1, 1].legend(fontsize=8)
    savefig(fig, fig_dir, "04_seasonality_event_snap.png")

    # Fig 5: price variation + binned relative price vs relative sales
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    rc = np.abs(rel_change[changed])
    rc = rc[rc < np.percentile(rc, 99)] if rc.size else rc
    ax[0].hist(rc, bins=40, color="#1f77b4", edgecolor="white")
    ax[0].set_xlabel("abs relative weekly price change (clipped at p99)")
    ax[0].set_ylabel("number of weekly changes")
    ax[0].set_title(f"Within-series price changes\n{100*price_summary['share_series_with_any_price_change']:.1f}% of series change price at least once")
    ax[1].bar(price_rel["rel_price_bin"], price_rel["mean_rel_sales"], color="#ff7f0e", edgecolor="white")
    ax[1].axhline(0.0, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel("relative price vs series mean")
    ax[1].set_ylabel("mean relative sales vs series mean")
    ax[1].set_title("Price vs sales (within series, descriptive)\nmedian is -1 everywhere (zero-inflated), mean shown")
    ax[1].tick_params(axis="x", rotation=45)
    savefig(fig, fig_dir, "05_price.png")

    # Fig 6: stockout flag rate by class and run-length distribution
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    order = [c for c in ["smooth", "erratic", "intermittent", "lumpy", "new_item"] if c in so_by_class["intermittency"].values]
    sub = so_by_class.set_index("intermittency").loc[order]
    ax[0].bar(order, sub["share_series_any_flag"], color="#d62728", edgecolor="white")
    ax[0].set_ylabel("share of series with any stockout flag")
    ax[0].set_xlabel("Syntetos-Boylan class")
    ax[0].set_title("Stockout-proxy flag by class")
    ax[0].tick_params(axis="x", rotation=30)
    if run_lengths.size:
        ax[1].hist(run_lengths, bins=40, color="#d62728", edgecolor="white")
        ax[1].axvline(stockout_cfg.get("min_run", 28), color="k", ls="--", lw=1,
                      label=f"min_run={stockout_cfg.get('min_run', 28)}")
        ax[1].legend(fontsize=8)
    ax[1].set_xlabel("flagged zero-run length (days)")
    ax[1].set_ylabel("number of runs")
    ax[1].set_title(f"Flagged run lengths (n_runs={run_lengths.size})")
    savefig(fig, fig_dir, "06_stockout.png")

    print("EDA complete.")
    print(f"figures -> {fig_dir}")
    print(f"tables  -> {out_dir}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
