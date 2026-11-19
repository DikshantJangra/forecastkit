"""E04: stand-alone evaluation of the zero-run persistence model against non-learned baselines.

Target: Z = 1[recorded sales = 0] on available days of the 4 TEST windows (h = 1..28). Every model / baseline is
fitted on history <= origin only, exactly as inside the hybrid. Optional: DeepAR's own P(Y=0) from exported
artifacts (`--deepar name=dir`) is scored on the same cells.

    python scripts/phase2_zero_model.py --config configs/m5.yaml --out results/phase2/zero_model \
        --deepar deepar_masked=outputs/phase2/e01_repro_s42/artifacts/deepar
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecastkit.backtest.context import build_context
from forecastkit.backtest.runner import prepare
from forecastkit.config import load_config
from forecastkit.utils import setup_logging
from forecastkit.zerorun import KMRenewal, binary_scores, recent_zero_rate, reliability_table, series_zero_rate
from forecastkit.zerorun.features import current_run_length
from forecastkit.zerorun.model import ZeroRunModel

RUN_BUCKETS = [(-1, 0, "run=0"), (1, 6, "run=1-6"), (7, 27, "run=7-27"), (28, 10**6, "run>=28")]


def run_bucket(r):
    r = np.nan_to_num(r, nan=-1)
    out = np.full(r.shape, "no_sale_yet", dtype=object)
    for lo, hi, name in RUN_BUCKETS:
        out[(r >= max(lo, 0)) & (r <= hi)] = name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m5.yaml")
    ap.add_argument("--out", default="results/phase2/zero_model")
    ap.add_argument("--deepar", action="append", default=[], help="name=artifact_dir with pred_<origin>.npz (p0)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    ds, orig, info = prepare(cfg)
    H = int(cfg["backtest"]["horizon"])
    qs = cfg["backtest"]["quantiles"]
    cold = ds.static["cold_start"].to_numpy()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dar = dict(s.split("=", 1) for s in args.deepar)
    cells, meta = [], {}
    for o in orig["test"]:
        t0 = time.time()
        ctx = build_context(ds, o, H, cfg.get("masks", {}), cold, ds.meta["cold_days"], qs, ds.static, seed=seed + o)
        tag = str(ctx.origin.date())
        preds = {"series_zero_rate": series_zero_rate(ctx.y_raw, H), "recent_zero_rate_28": recent_zero_rate(ctx.y_raw, H, 28),
                 "km_renewal": KMRenewal().fit(ctx.y_raw).predict(H)}
        imp = {}
        for name, kw in (("lgbm_zero", {}), ("lgbm_zero_no_persistence", {"persistence_features": False})):
            zm = ZeroRunModel(seed=ctx.seed, **kw).fit(ctx)
            preds[name] = zm.predict_proba()
            imp[name] = {"best_iteration": zm.best_iteration, "n_train": zm.n_train, "importance": zm.feature_importance(25)}
        for name, d in dar.items():
            z = np.load(Path(d) / f"pred_{tag}.npz", allow_pickle=True)
            assert (z["series_ids"] == ctx.series_ids).all()
            preds[name] = np.clip(z["p0"].astype(float), 1e-6, 1 - 1e-6)
        meta[tag] = {"seconds": time.time() - t0, **imp}
        y = ds.Y[:, o + 1:o + 1 + H].astype(float)
        a = ds.avail[:, o + 1:o + 1 + H]
        r = current_run_length(ctx.y_raw)[:, -1]
        flag = ctx.censored[:, -1]
        s_idx, h_idx = np.nonzero(a)
        base = pd.DataFrame({"origin": tag, "series": s_idx, "h": h_idx + 1, "z": (y[s_idx, h_idx] == 0).astype(int),
                             "cold": cold[s_idx], "run_bucket": run_bucket(r)[s_idx], "proxy_flag_at_origin": flag[s_idx],
                             "intermittency": info["intermittency"].to_numpy()[s_idx]})
        for k, p in preds.items():
            base[k] = p[s_idx, h_idx]
        cells.append(base)
        print(f"{tag}: {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(cells, ignore_index=True)
    models = [c for c in df.columns if c not in ("origin", "series", "h", "z", "cold", "run_bucket",
                                                 "proxy_flag_at_origin", "intermittency")]
    df.to_parquet(out / "cells.parquet", index=False)

    def score(sub, extra):
        return [{**extra, "model": m, **binary_scores(sub[m].to_numpy(), sub.z.to_numpy())} for m in models]

    warm = df[~df.cold]
    rows = score(warm, {"slice": "all_warm"})
    for col in ("run_bucket", "intermittency", "origin", "proxy_flag_at_origin"):
        for lvl, g in warm.groupby(col):
            rows += score(g, {"slice": f"{col}={lvl}"})
    rows += score(df[df.cold], {"slice": "cold_start"})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "scores.csv", index=False)
    rel = pd.concat([reliability_table(warm[m].to_numpy(), warm.z.to_numpy()).assign(model=m) for m in models])
    rel.to_csv(out / "reliability.csv", index=False)
    # paired bootstrap over series (Brier difference lgbm_zero - baseline), resampling unit = series
    rng = np.random.default_rng(0)
    ser = warm.series.to_numpy()
    uniq = np.unique(ser)
    pos = {s: i for i, s in enumerate(uniq)}
    sid = np.array([pos[s] for s in ser])
    cnt = np.bincount(sid, minlength=len(uniq)).astype(float)
    B = rng.integers(0, len(uniq), size=(2000, len(uniq)))
    boot = []
    for ref in [m for m in models if m != "lgbm_zero"]:
        for metric, f in (("brier", lambda p, z: (p - z) ** 2),
                          ("log_loss", lambda p, z: -(z * np.log(p) + (1 - z) * np.log(1 - p)))):
            clip = lambda x: np.clip(x, 1e-6, 1 - 1e-6)            # same clipping as binary_scores
            d = f(clip(warm.lgbm_zero.to_numpy()), warm.z.to_numpy()) - f(clip(warm[ref].to_numpy()), warm.z.to_numpy())
            ds_ = np.bincount(sid, weights=d, minlength=len(uniq))
            bs = ds_[B].sum(1) / cnt[B].sum(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            boot.append({"model": "lgbm_zero", "reference": ref, "metric": metric, "diff": ds_.sum() / cnt.sum(),
                         "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)})
    pd.DataFrame(boot).to_csv(out / "bootstrap_vs_baselines.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    _plot(rel, out / "reliability.png")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab[tab.slice.isin(["all_warm", "run_bucket=run>=28", "run_bucket=run=0"])].round(4).to_string(index=False))
        print(pd.DataFrame(boot).round(5).to_string(index=False))


def _plot(rel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for m, g in rel.groupby("model"):
        ax.plot(g.mean_pred, g.obs_rate, "o-", ms=3, lw=1.2, label=m)
    ax.set_xlabel("predicted P(recorded zero)")
    ax.set_ylabel("observed zero rate")
    ax.set_title("Zero-probability reliability (test, warm series)", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()
