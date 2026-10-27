"""Model-agnostic rolling-origin backtester.

For every model and every origin: build the leak-proof context -> model.fit(ctx) -> model.predict(quantiles)
-> append to ONE long forecast table. The runner knows nothing about specific models or datasets.

Output dir  outputs/<run_name>/:
    config.yaml        exact resolved config (reproducibility)
    forecasts.parquet  model, series_id, origin, h, ds, stat ('mean' | 'q0.5' ...), value
    timing.csv         fit / predict seconds per model x origin
    series_info.csv    per-series static columns + intermittency class, volume tier, cold-start flag
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forecastkit.datasets import load_dataset
from forecastkit.features.intermittency import classify_demand
from forecastkit.schema import q_stat
from forecastkit.utils import get_model_class, log, set_seed

from .context import build_context
from .splitter import rolling_origins

DEFAULT_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def prepare(cfg: dict, use_cache: bool = True):
    """dataset + origins + per-series info (slice labels computed on data BEFORE the first test origin)."""
    ds = load_dataset(cfg["dataset"], cache_dir=cfg.get("cache_dir", "data/processed"), use_cache=use_cache)
    bt = cfg.get("backtest", {})
    H = int(bt.get("horizon", 28))
    orig = rolling_origins(ds.dates, H, int(bt.get("n_origins", 4)), int(bt.get("step", H)),
                           int(bt.get("val_origins", 1)), bt.get("origins"))
    cutoff = min(orig["val"] + orig["test"]) + 1          # labels use only data before every origin
    info = classify_demand(ds.Y, ds.avail, end=cutoff)
    info.index = ds.series_ids
    cs = bt.get("cold_start", {}) or {}
    cold = np.zeros(ds.N, dtype=bool)
    if cs.get("fraction", 0) > 0:
        rng = np.random.default_rng(int(cs.get("seed", 7)))
        long_hist = np.where(ds.avail[:, :cutoff].sum(1) >= int(cs.get("min_history", 200)))[0]
        n = max(1, round(cs["fraction"] * ds.N))
        cold[rng.choice(long_hist, size=min(n, len(long_hist)), replace=False)] = True
    info["intermittency"] = info["intermittency"].replace("no_sales", "new_item")  # nothing sold before first origin
    info["cold_start"] = cold
    static = ds.static.copy()
    for c in ("intermittency", "volume_tier", "cold_start"):
        static[c] = info[c].to_numpy()
    ds.static = static
    ds.meta["cold_days"] = int(cs.get("history_days", 28))
    return ds, orig, info


def make_models(cfg: dict, only=None) -> dict:
    out = {}
    for name, spec in cfg["models"].items():
        if only and name not in only:
            continue
        if spec.get("enabled", True) is False:
            continue
        out[name] = (spec["type"], dict(spec.get("params", {})))
    return out


def run_backtest(cfg: dict, out_dir=None, phase: str = "test", only=None, use_cache: bool = True,
                 save: bool = True) -> pd.DataFrame:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    ds, orig, info = prepare(cfg, use_cache)
    bt = cfg.get("backtest", {})
    H = int(bt.get("horizon", 28))
    qs = [float(q) for q in bt.get("quantiles", DEFAULT_QUANTILES)]
    origins = orig["val"] if phase == "val" else orig["test"]
    mask_cfg = cfg.get("masks", {})
    out_dir = Path(out_dir or Path(cfg.get("output_dir", "outputs")) / cfg["run_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cold = ds.static["cold_start"].to_numpy()
    log.info("run=%s phase=%s | %d series (%d cold) | origins=%s | H=%d", cfg["run_name"], phase, ds.N,
             cold.sum(), [str(ds.dates[o].date()) for o in origins], H)

    ctxs = {o: build_context(ds, o, H, mask_cfg, cold, ds.meta["cold_days"], qs, ds.static, seed=seed + o)
            for o in origins}
    rows, timing = [], []
    for mname, (mtype, params) in make_models(cfg, only).items():
        model = get_model_class(mtype)(**params)       # one instance across origins (warm starts allowed)
        for o in origins:
            ctx = ctxs[o]
            t0 = time.time()
            model.fit(ctx)
            t1 = time.time()
            fc = model.predict(qs)
            t2 = time.time()
            assert fc.quantiles.shape == (ds.N, H, len(qs)), f"{mname}: bad quantile shape {fc.quantiles.shape}"
            rows.append(_to_long(mname, ds.series_ids, ctx, fc, qs))
            timing.append({"model": mname, "origin": ctx.origin, "fit_s": t1 - t0, "predict_s": t2 - t1})
            log.info("  %-14s origin %s  fit %.1fs  predict %.1fs", mname, ctx.origin.date(), t1 - t0, t2 - t1)
        if hasattr(model, "diagnostics") and save:
            _save_model_extras(model, out_dir, mname)
    fc_df = pd.concat(rows, ignore_index=True)
    if save:
        suffix = "" if phase == "test" else f"_{phase}"
        _merge_save(fc_df, out_dir / f"forecasts{suffix}.parquet", only)
        t_df = pd.DataFrame(timing)
        tp = out_dir / f"timing{suffix}.csv"
        if only and tp.exists():
            old = pd.read_csv(tp)
            t_df = pd.concat([old[~old.model.isin(t_df.model.unique())], t_df])
        t_df.to_csv(tp, index=False)
        info.assign(**{c: ds.static[c].to_numpy() for c in ds.static.columns if c not in info.columns}
                    ).reset_index(names="series_id").to_csv(out_dir / "series_info.csv", index=False)
        (out_dir / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items()}, sort_keys=False))
    return fc_df


def _merge_save(df: pd.DataFrame, path: Path, only):
    if only and path.exists():
        old = pd.read_parquet(path)
        old = old[~old.model.isin(df.model.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(path, index=False)


def _to_long(model, series_ids, ctx, fc, qs) -> pd.DataFrame:
    N, H, _Q = fc.quantiles.shape
    sid = np.repeat(series_ids, H)
    h = np.tile(np.arange(1, H + 1), N)
    ds_ = np.tile(ctx.fut_dates.values, N)
    base = {"model": model, "series_id": sid, "origin": ctx.origin, "h": h.astype(np.int16), "ds": ds_}
    parts = []
    mean = fc.mean if fc.mean is not None else fc.quantiles[:, :, int(np.argmin(np.abs(np.array(qs) - 0.5)))]
    parts.append(pd.DataFrame({**base, "stat": "mean", "value": mean.reshape(-1)}))
    for j, q in enumerate(qs):
        parts.append(pd.DataFrame({**base, "stat": q_stat(q), "value": fc.quantiles[:, :, j].reshape(-1)}))
    out = pd.concat(parts, ignore_index=True)
    out["value"] = out["value"].astype(np.float32)
    for c in ("model", "stat"):
        out[c] = out[c].astype("category")
    return out


def _save_model_extras(model, out_dir: Path, mname: str):
    """Kalman diagnostics for 3 illustrative series (smooth, intermittent, cold-start) for the state-uncertainty figure."""
    try:
        import pickle
        c = model.ctx
        vol = np.nan_to_num(c.y).sum(1)
        cls = c.static["intermittency"].to_numpy() if "intermittency" in c.static else np.array([""] * c.N)
        pick = []
        for want in ("smooth", "intermittent"):
            sel = np.where((cls == want) & ~c.cold)[0]
            if len(sel):
                pick.append(int(sel[np.argmax(vol[sel])]))
        if c.cold.any():
            pick.append(int(np.where(c.cold)[0][0]))
        if not pick:
            pick = [int(np.argmax(vol))]
        with open(out_dir / f"diag_{mname}.pkl", "wb") as f:
            pickle.dump({c.series_ids[i]: model.diagnostics(i) for i in pick}, f)
    except Exception as e:  # noqa: BLE001 - diagnostics are optional
        log.warning("diagnostics skipped for %s: %s", mname, e)
