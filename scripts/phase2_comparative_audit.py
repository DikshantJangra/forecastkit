"""Phase 1 vs Phase 2 comparative audit from SAVED forecasts only (no training, no backtest).
    OMP_NUM_THREADS=1 python scripts/phase2_comparative_audit.py   -> results/phase2/comparative_audit/*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from forecastkit.backtest.context import build_context  # noqa: F401  (documentation: same context code)
from forecastkit.config import load_config
from forecastkit.evaluation.evaluator import Evaluator
from forecastkit.masks import leading_zero_mask, stockout_mask
from forecastkit.zerorun.features import current_run_length

OUT = Path("results/phase2/comparative_audit")
SEEDS = [42, 43, 44]
GROUPS = {"kalman_lgbm": ["kalman_lgbm"], "deepar_p1": [f"deepar_p1@s{s}" for s in SEEDS],
          "deepar_v2": [f"deepar_v2@s{s}" for s in SEEDS], "deepar_v2_nomask": [f"deepar_v2_nomask@s{s}" for s in SEEDS],
          "hybrid": [f"hybrid@s{s}" for s in SEEDS]}
PHASE = {"kalman_lgbm": "P1 (seed 42)", "deepar_p1": "P1 config, 3 seeds", "deepar_v2": "P2, 3 seeds",
         "deepar_v2_nomask": "P2, 3 seeds", "hybrid": "P2, 3 seeds"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/phase2/m5_phase2.yaml")
    p1 = pd.read_parquet("outputs/m5_main/forecasts.parquet")
    p1["model"] = p1.model.astype(str)
    parts = [p1[p1.model == "kalman_lgbm"]]
    for g, ms in GROUPS.items():
        for m in ms:
            if "@" not in m:
                continue
            base, s = m.split("@s")
            f = pd.read_parquet(f"outputs/phase2/runs/{base}_s{s}/forecasts.parquet")
            f["model"] = m
            parts.append(f)
    fc = pd.concat(parts, ignore_index=True)
    fc["model"], fc["stat"] = fc.model.astype(str), fc.stat.astype(str)
    cfg["ensembles"] = {}

    # ---------------- 1. fairness checks on the raw forecast files
    keys = {}
    for m, g in fc.groupby("model"):
        k = g[g.stat == "q0.5"][["series_id", "origin", "h", "ds"]]
        keys[m] = (set(k.series_id), sorted(pd.to_datetime(k.origin).unique()), sorted(k.h.unique()), len(k),
                   sorted(g.stat.unique()))
    ref = keys["kalman_lgbm"]
    fair = [{"model": m, "same_series": v[0] == ref[0], "n_series": len(v[0]), "same_origins": v[1] == ref[1],
             "origins": ";".join(str(pd.Timestamp(o).date()) for o in v[1]), "same_horizons": v[2] == ref[2],
             "H": max(v[2]), "n_median_cells": v[3], "same_stats": v[4] == ref[4]} for m, v in keys.items()]
    fair = pd.DataFrame(fair)
    fair.to_csv(OUT / "fairness.csv", index=False)

    ev = Evaluator(cfg, "outputs/m5_main", forecasts=fc)       # ONE truth cube Y/A/C shared by every model
    cold = ev.cold
    base_v = ev.valid(~cold, "all")

    # ---------------- causal state at origin and cell groups (mutually exclusive within each family)
    R, F = [], []
    for o in ev.oi:
        y = ev.ds.Y[:, :o + 1].astype(float)
        av = ev.ds.avail[:, :o + 1] & ~leading_zero_mask(y, ev.ds.avail[:, :o + 1])
        R.append(current_run_length(np.where(av, y, np.nan))[:, -1])
        F.append(stockout_mask(y, av, cfg["masks"]["stockout"])[:, -1])
    R = np.nan_to_num(np.stack(R), nan=-1)[:, :, None]
    cls = ev.ds.static["intermittency"].astype(str).to_numpy()[None, :, None]
    fam = {
        "outcome": {"y>0 (non-zero sales)": ev.Y > 0, "y=0 (zero sales)": ev.Y == 0},
        "hindsight flag": {"non-censored": ~ev.C, "censored (in flagged long zero run)": ev.C},
        "zero run at origin": {"0 (sold on origin day)": R == 0, "1-6 d": (R >= 1) & (R <= 6),
                               "7-27 d": (R >= 7) & (R <= 27), ">=28 d": R >= 28},
        "demand class": {c: cls == c for c in ("intermittent", "lumpy", "smooth", "erratic")},
    }

    def metrics(models, v):
        out = {}
        for k in ("WAPE", "MAE", "RMSE", "pinball", "cov80", "cov90"):
            out[k] = float(np.mean([ev.agg(m, v)[k] for m in models]))
        out["n"] = int(v.sum())
        return out

    # ---------------- 2. overall table
    rows = []
    for mode in ("all", "non_censored"):
        v = ev.valid(~cold, mode)
        for g, ms in GROUPS.items():
            rows.append({"eval_days": mode, "model": g, "phase": PHASE[g], **metrics(ms, v)})
    overall = pd.DataFrame(rows)
    overall.to_csv(OUT / "overall.csv", index=False)
    deltas = []
    for mode in ("all", "non_censored"):
        o = overall[overall.eval_days == mode].set_index("model")
        for ref_m in ("kalman_lgbm", "deepar_p1", "deepar_v2", "deepar_v2_nomask"):
            for k in ("WAPE", "MAE", "RMSE", "pinball", "cov80", "cov90"):
                a, b = o.loc["hybrid", k], o.loc[ref_m, k]
                deltas.append({"eval_days": mode, "reference": ref_m, "metric": k, "hybrid": a, "reference_value": b,
                               "abs_diff": a - b, "rel_diff_pct": 100 * (a - b) / b})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(OUT / "hybrid_minus_reference.csv", index=False)

    # ---------------- 3/4. subsets: metrics + hybrid - kalman_lgbm + contribution to the overall gap
    rows = []
    tot_ae = {g: np.mean([(ev.err[m].AE * base_v).sum() for m in ms]) for g, ms in GROUPS.items()}
    tot_pb = {g: np.mean([(ev.err[m].PB * base_v).sum() for m in ms]) for g, ms in GROUPS.items()}
    for fname, grp in fam.items():
        for gname, mask in grp.items():
            v = base_v & mask
            if v.sum() == 0:
                continue
            r = {"family": fname, "subset": gname, "n_cells": int(v.sum()), "share_cells": v.sum() / base_v.sum(),
                 "share_units": ev.Y[v].sum() / ev.Y[base_v].sum()}
            for g, ms in GROUPS.items():
                mt = metrics(ms, v)
                for k in ("WAPE", "MAE", "RMSE", "pinball"):
                    r[f"{g}_{k}"] = mt[k]
                # share of the overall abs-error / pinball-sum gap (hybrid - g) that this subset accounts for
                ae = np.mean([(ev.err[m].AE * v).sum() for m in ms])
                pb = np.mean([(ev.err[m].PB * v).sum() for m in ms])
                r[f"{g}_AEsum"], r[f"{g}_PBsum"] = ae, pb
            for k in ("WAPE", "MAE", "RMSE", "pinball"):
                r[f"hyb_minus_klgbm_{k}"] = r[f"hybrid_{k}"] - r[f"kalman_lgbm_{k}"]
                r[f"hyb_minus_dar_v2_{k}"] = r[f"hybrid_{k}"] - r[f"deepar_v2_{k}"]
            r["gap_share_AE_vs_klgbm"] = (r["hybrid_AEsum"] - r["kalman_lgbm_AEsum"]) / (tot_ae["hybrid"] - tot_ae["kalman_lgbm"])
            r["gap_share_PB_vs_klgbm"] = (r["hybrid_PBsum"] - r["kalman_lgbm_PBsum"]) / (tot_pb["hybrid"] - tot_pb["kalman_lgbm"])
            rows.append(r)
    sub = pd.DataFrame(rows)
    sub.to_csv(OUT / "subsets.csv", index=False)

    # ---------------- 5. paired bootstrap (series resampling, 500 draws)
    bs = []
    tests = [("all days", base_v), ("non-censored", ev.valid(~cold, "non_censored"))]
    tests += [(f"{f}: {g}", base_v & m) for f, grp in fam.items() for g, m in grp.items()]
    for label, v in tests:
        for ref_m in ("kalman_lgbm", "deepar_v2"):
            for met in ("pinball", "MAE") if "y=0" in label else ("pinball", "WAPE"):
                r = ev.paired_bootstrap(GROUPS["hybrid"], GROUPS[ref_m], met, v, n_boot=500, seed=0)
                bs.append({"cells": label, "reference": ref_m, "metric": met, "diff": r["diff"], "ci_low": r["ci_low"],
                           "ci_high": r["ci_high"], "significant": r["significant"],
                           "origins_hybrid_better": f"{r['origins_a_better']}/{r['n_origins']}"})
    bs = pd.DataFrame(bs)
    bs.to_csv(OUT / "bootstrap.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 200):
        print(fair.to_string(index=False))
        print(overall.round(4).to_string(index=False))
        print(sub[["family", "subset", "n_cells", "share_cells", "share_units", "kalman_lgbm_pinball", "hybrid_pinball",
                   "hyb_minus_klgbm_pinball", "hyb_minus_klgbm_MAE", "hyb_minus_klgbm_WAPE", "hyb_minus_klgbm_RMSE",
                   "gap_share_PB_vs_klgbm", "gap_share_AE_vs_klgbm", "hyb_minus_dar_v2_pinball"]].round(4).to_string(index=False))
        print(bs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
