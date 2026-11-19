"""Builds the Phase 2 results package (results/phase2/) from the forecast files of the Phase 2 runs and the
Phase 1 baselines. Every number in the report/slides comes from a CSV written here.

    python scripts/phase2_analysis.py            # after scripts/phase2_run_all.sh

Inputs : outputs/phase2/runs/<model>_s<seed>/{forecasts.parquet, timing.csv, artifacts/<model>/*.npz|json}
         outputs/m5_main/forecasts.parquet  (Phase 1 baselines; their forecasts are unchanged in Phase 2)
Naming : a Phase 2 run appears as `<model>@s<seed>`; Phase 1 baselines keep their names.
Cells  : warm series (simulated cold-start series reported separately), available days, 4 test origins x 28 days.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from forecastkit.config import load_config
from forecastkit.evaluation.evaluator import Evaluator
from forecastkit.evaluation.metrics import pit_histogram, randomized_pit
from forecastkit.masks import leading_zero_mask, stockout_mask
from forecastkit.utils import setup_logging
from forecastkit.zerorun import binary_scores, reliability_table
from forecastkit.zerorun.features import current_run_length

P1_BASELINES = ["naive", "snaive", "snaive4", "ma28", "kalman", "lgbm", "kalman_lgbm"]
SEEDS = [42, 43, 44]
MULTI = ["deepar_p1", "deepar_v2", "deepar_v2_nomask", "hybrid"]
METRICS = ["WAPE", "pinball"]


def load_forecasts(runs_root: Path, p1_dir: Path):
    parts, timing = [], []
    p1 = pd.read_parquet(p1_dir / "forecasts.parquet")
    p1["model"] = p1["model"].astype(str)
    parts.append(p1[p1.model.isin(P1_BASELINES)])
    t1 = pd.read_csv(p1_dir / "timing.csv")
    timing.append(t1[t1.model.isin(P1_BASELINES)])
    runs = {}
    for d in sorted(runs_root.glob("*_s*")):
        m = re.match(r"(.+)_s(\d+)$", d.name)
        if not m or not (d / "forecasts.parquet").exists():
            continue
        model, seed = m.group(1), int(m.group(2))
        f = pd.read_parquet(d / "forecasts.parquet")
        f["model"] = f["model"].astype(str)
        f = f[f.model == model].copy()
        name = f"{model}@s{seed}"
        f["model"] = name
        parts.append(f)
        t = pd.read_csv(d / "timing.csv")
        t = t[t.model == model].assign(model=name)
        timing.append(t)
        runs[name] = {"model": model, "seed": seed, "dir": d, "art": d / "artifacts" / model}
    fc = pd.concat(parts, ignore_index=True)
    fc["stat"] = fc["stat"].astype(str)
    return fc, pd.concat(timing, ignore_index=True), runs


def origin_states(ev: Evaluator, mask_cfg):
    """Causal per-(origin, series) state at the origin: current zero-run length and the history-only proxy flag."""
    ds = ev.ds
    R, F = [], []
    for o in ev.oi:
        Th = o + 1
        y = ds.Y[:, :Th].astype(float)
        av = ds.avail[:, :Th] & ~leading_zero_mask(y, ds.avail[:, :Th])
        yr = np.where(av, y, np.nan)
        R.append(current_run_length(yr)[:, -1])
        F.append(stockout_mask(y, av, mask_cfg.get("stockout"))[:, -1])
    return np.stack(R), np.stack(F)                         # [O,N]


def run_bucket(r):
    r = np.nan_to_num(r, nan=-1)
    out = np.full(r.shape, "no_sale_yet", dtype=object)
    out[r == 0] = "0"
    out[(r >= 1) & (r <= 6)] = "1-6"
    out[(r >= 7) & (r <= 27)] = "7-27"
    out[r >= 28] = ">=28"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2/m5_phase2.yaml")
    ap.add_argument("--runs", default="outputs/phase2/runs")
    ap.add_argument("--phase1", default="outputs/m5_main")
    ap.add_argument("--out", default="results/phase2")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    cfg["ensembles"] = {"ens_global_local": ["kalman", "lgbm", "deepar_p1@s42"],      # Phase 1 ensemble (identical)
                        "ens_global_local_v2": ["kalman", "lgbm", "deepar_v2@s42"]}
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fc, timing, runs = load_forecasts(Path(args.runs), Path(args.phase1))
    ev = Evaluator(cfg, args.phase1, forecasts=fc)
    models = ev.models
    cold = ev.cold
    wall = {}

    # ---------------------------------------------------------------- main tables
    tabs = {}
    for mode in ("all", "non_censored"):
        t = ev.main_table(mode)
        t.insert(0, "phase", ["P1" if m in P1_BASELINES or m == "ens_global_local" or m.startswith("deepar_p1@")
                              else "P2" for m in t.index])
        tabs[mode] = t
        t.to_csv(out / f"main_{mode}.csv")
    ev.cold_start().to_csv(out / "cold_start.csv")

    # ---------------------------------------------------------------- seeds
    rows = []
    for base in MULTI:
        for s in SEEDS:
            name = f"{base}@s{s}"
            if name not in models:
                continue
            for mode in ("all", "non_censored"):
                r = tabs[mode].loc[name]
                rows.append({"model": base, "seed": s, "eval_days": mode, **{k: r[k] for k in
                             ("WAPE", "MAE", "RMSE", "pinball", "WQL", "cov80", "cov90", "width80")}})
            cs = ev.cold_start().loc[name]
            rows.append({"model": base, "seed": s, "eval_days": "cold_start", **{k: cs[k] for k in
                         ("WAPE", "MAE", "RMSE", "pinball", "WQL", "cov80", "cov90", "width80")}})
    seeds = pd.DataFrame(rows)
    seeds.to_csv(out / "seeds_per_seed.csv", index=False)
    summ = seeds.groupby(["model", "eval_days"]).agg(
        n_seeds=("seed", "nunique"), WAPE_mean=("WAPE", "mean"), WAPE_std=("WAPE", "std"),
        pinball_mean=("pinball", "mean"), pinball_std=("pinball", "std"), MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"), cov80_mean=("cov80", "mean"), cov90_mean=("cov90", "mean")).reset_index()
    summ.to_csv(out / "seeds_summary.csv", index=False)

    # ---------------------------------------------------------------- pairwise + ablations
    def reps(base):
        return [f"{base}@s{s}" for s in SEEDS if f"{base}@s{s}" in models]

    def one(base):
        return f"{base}@s42"

    V = {"all": ev.valid(~cold, "all"), "non_censored": ev.valid(~cold, "non_censored")}
    comparisons = [
        # (experiment, question, a, b, unit)
        ("DeepAR P2 vs P1", "Do lags + validation checkpointing improve DeepAR?", reps("deepar_v2"), reps("deepar_p1"), "3 seeds"),
        ("Seasonal lags (no ES)", "Do lag-7/14/28 inputs help on their own?", [one("dar_v2_no_val")], [one("deepar_p1")], "seed 42"),
        ("Validation checkpoint", "Does best-checkpoint selection on the history validation block help?", [one("deepar_v2")], [one("dar_v2_no_val")], "seed 42"),
        ("Age feature", "Does log-age (days since first availability) help?", [one("dar_v2_age")], [one("deepar_v2")], "seed 42"),
        ("AdamW", "Decoupled weight decay 0.01 vs coupled L2 1e-5 (with age)", [one("dar_v2_age_adamw")], [one("dar_v2_age")], "seed 42"),
        ("Loss masking (DeepAR)", "Learning long zero runs as demand vs masking them", reps("deepar_v2_nomask"), reps("deepar_v2"), "3 seeds"),
        ("Hybrid vs DeepAR", "Does the zero-run persistence component add value?", reps("hybrid"), reps("deepar_v2"), "3 seeds"),
        ("Hybrid vs unmasked DeepAR", "Is the hurdle better than letting DeepAR learn the zeros itself?", reps("hybrid"), reps("deepar_v2_nomask"), "3 seeds"),
        ("Hybrid vs LightGBM", "Hybrid against the strongest tabular model", reps("hybrid"), ["lgbm"], "3 seeds vs 1"),
        ("Hybrid vs Kalman-LightGBM", "Hybrid against the Phase 1 hybrid", reps("hybrid"), ["kalman_lgbm"], "3 seeds vs 1"),
        ("Kalman-LightGBM vs LightGBM", "Phase 1 hybrid: does the Kalman state add value?", ["kalman_lgbm"], ["lgbm"], "single"),
        ("RB reconstruction control", "Is the hurdle code path neutral when q(0)=p_D(0)?", [one("hyb_none")], [one("deepar_v2")], "seed 42"),
        ("No persistence features", "Do explicit zero-run features matter inside the hybrid?", [one("hyb_no_persist")], [one("hybrid")], "seed 42"),
        ("No DL component", "Replace DeepAR's positive part by the empirical positive distribution", [one("hyb_no_dl")], [one("hybrid")], "seed 42"),
        ("Fusion: naive ZI", "pi*delta0 + (1-pi)*p_D (double-counts zeros)", [one("hyb_zi_naive")], [one("hybrid")], "seed 42"),
        ("Fusion: inflate-only ZI", "q(0) = max(pi, p_D(0))", [one("hyb_zi_inflate")], [one("hybrid")], "seed 42"),
        ("Fusion: gated", "Use pi only for series in a flagged zero run at the origin", [one("hyb_gated")], [one("hybrid")], "seed 42"),
        ("Hybrid on unmasked DeepAR", "Does the hurdle need the masked (latent-demand) DeepAR?", [one("hyb_nomask")], [one("hybrid")], "seed 42"),
        ("Hybrid on Phase 1 DeepAR", "Contribution of the DeepAR v2 changes inside the hybrid", [one("hyb_p1")], [one("hybrid")], "seed 42"),
        ("Ensemble P2 vs DeepAR v2", "Equal-weight quantile average (kalman, lgbm, deepar_v2)", ["ens_global_local_v2"], [one("deepar_v2")], "seed 42"),
    ]
    pw = []
    for exp, q, a, b, unit in comparisons:
        if not a or not b or not all(m in models for m in a + b):
            print("skip (missing runs):", exp)
            continue
        for mode, v in V.items():
            for met in METRICS:
                r = ev.paired_bootstrap(a, b, met, v, n_boot=args.n_boot, seed=0)
                ma = np.mean([ev.agg(m, v)[met] for m in a])
                mb = np.mean([ev.agg(m, v)[met] for m in b])
                pw.append({"experiment": exp, "question": q, "unit": unit, "eval_days": mode, "metric": met,
                           "changed": r["a"], "baseline": r["b"], "value_changed": ma, "value_baseline": mb, **{
                               k: r[k] for k in ("diff", "ci_low", "ci_high", "significant", "series_win_rate_a",
                                                 "origins_a_better", "n_origins", "per_origin_diff")}})
    pw = pd.DataFrame(pw)
    pw["rel_diff_pct"] = 100 * pw["diff"] / pw["value_baseline"]
    pw.to_csv(out / "pairwise_bootstrap.csv", index=False)

    # ---------------------------------------------------------------- slices (incl. causal zero-run state at origin)
    R, Fl = origin_states(ev, cfg.get("masks", {}))
    B = run_bucket(R)
    focus = [m for m in models if m in P1_BASELINES or m.endswith("@s42") or m.startswith("ens")]
    sl = []
    for mode in ("all", "non_censored"):
        for b in ("0", "1-6", "7-27", ">=28"):
            v = ev.valid(~cold, mode) & (B == b)[:, :, None]
            sl.append(ev.table(v, {"slice": f"run_at_origin={b}", "eval_days": mode}))
        for f in (True, False):
            v = ev.valid(~cold, mode) & (Fl == f)[:, :, None]
            sl.append(ev.table(v, {"slice": f"proxy_flag_at_origin={f}", "eval_days": mode}))
        for g in ("intermittent", "lumpy", "smooth", "erratic"):
            key = f"intermittency={g}"
            if key in ev.groups:
                sl.append(ev.table(ev.valid(ev.groups[key] & ~cold, mode), {"slice": key, "eval_days": mode}))
    slices = pd.concat(sl, ignore_index=True)
    slices = slices[slices.model.isin(focus) | slices.model.str.startswith(tuple(MULTI))]
    slices.to_csv(out / "slices.csv", index=False)
    share = pd.DataFrame({"bucket": B[~np.broadcast_to(cold, B.shape)]}).value_counts(normalize=True)
    share.to_csv(out / "run_bucket_share.csv")
    # slice-level significance: hybrid vs deepar_v2 inside the run buckets (3 seeds)
    ss = []
    for b in ("0", "1-6", "7-27", ">=28"):
        for mode in ("all", "non_censored"):
            v = ev.valid(~cold, mode) & (B == b)[:, :, None]
            for met in METRICS:
                if reps("hybrid") and reps("deepar_v2"):
                    r = ev.paired_bootstrap(reps("hybrid"), reps("deepar_v2"), met, v, n_boot=args.n_boot)
                    ss.append({"slice": f"run_at_origin={b}", "eval_days": mode, "metric": met, **r})
    for g in ("intermittent", "lumpy", "smooth", "erratic"):
        key = f"intermittency={g}"
        for mode in ("all", "non_censored"):
            v = ev.valid(ev.groups[key] & ~cold, mode)
            for met in METRICS:
                if reps("hybrid") and reps("deepar_v2"):
                    r = ev.paired_bootstrap(reps("hybrid"), reps("deepar_v2"), met, v, n_boot=args.n_boot)
                    ss.append({"slice": key, "eval_days": mode, "metric": met, **r})
    pd.DataFrame(ss).to_csv(out / "slices_bootstrap_hybrid_vs_deepar.csv", index=False)

    # ---------------------------------------------------------------- per origin, horizon, quantile calibration
    po = ev.per_origin()
    po[po.model.isin(focus)].to_csv(out / "per_origin.csv", index=False)
    bh = ev.by_horizon()
    bh[bh.model.isin(focus)].to_csv(out / "by_horizon.csv", index=False)
    cal = ev.calibration()
    cal[cal.model.isin(focus)].to_csv(out / "calibration_quantiles.csv", index=False)
    g = timing.groupby("model")[["fit_s", "predict_s"]].sum()
    g["total_s"] = g.fit_s + g.predict_s
    g.to_csv(out / "runtime.csv")

    # ---------------------------------------------------------------- PIT + zero-probability calibration (artifacts)
    Y, A = ev.Y, ev.A
    pit_rows, zrows, rel_all = [], [], []
    pits = {}
    for name, info in runs.items():
        files = sorted(info["art"].glob("pred_*.npz"))
        if len(files) != len(ev.origins):
            continue
        us, p0s, zs, bks = [], [], [], []
        rng = np.random.default_rng(0)
        for oi, o in enumerate(ev.origins):
            z = np.load(info["art"] / f"pred_{pd.Timestamp(o).date()}.npz", allow_pickle=True)
            assert (z["series_ids"] == ev.ds.series_ids).all()
            v = ev.valid(~cold, "all")[oi]
            s = z["samples"].transpose(0, 2, 1)             # [N,H,S]
            us.append(randomized_pit(Y[oi][v], s[v], rng))
            p0s.append(z["p0"][v])
            zs.append((Y[oi][v] == 0).astype(float))
            bks.append(np.broadcast_to(B[oi][:, None], v.shape)[v])
        u, p0, zz, bk = map(np.concatenate, (us, p0s, zs, bks))
        dens, dev = pit_histogram(u)
        pits[name] = dens
        pit_rows.append({"model": name, "pit_max_abs_dev": dev, "pit_mean": float(u.mean()), "n": len(u),
                         **{f"bin{k}": d for k, d in enumerate(dens)}})
        zrows.append({"model": name, "slice": "all_warm", **binary_scores(np.clip(p0, 1e-6, 1 - 1e-6), zz)})
        for b in ("0", "1-6", "7-27", ">=28"):
            m = bk == b
            zrows.append({"model": name, "slice": f"run_at_origin={b}", **binary_scores(np.clip(p0[m], 1e-6, 1 - 1e-6), zz[m])})
        rel_all.append(reliability_table(p0, zz).assign(model=name))
    pd.DataFrame(pit_rows).to_csv(out / "pit.csv", index=False)
    pd.DataFrame(zrows).to_csv(out / "zero_probability_scores.csv", index=False)
    if rel_all:
        pd.concat(rel_all).to_csv(out / "zero_probability_reliability.csv", index=False)

    # ---------------------------------------------------------------- training curves
    curves = []
    for name, info in runs.items():
        for f in sorted(info["art"].glob("train_*.json")) + sorted(info["art"].glob("meta_*.json")):
            h = json.loads(f.read_text())
            h = h.get("deepar_history", h) if f.name.startswith("meta_") else h
            if not h or "val_loss" not in h:
                continue
            curves.append({"model": name, "origin": h.get("origin"), "warm": h.get("warm"), "best_step": h.get("best_step"),
                           "best_val": h.get("best_val"), "stopped_step": h.get("stopped_step"),
                           "max_steps": h.get("max_steps"), "seconds": h.get("seconds"),
                           "final_train": h["train_loss"][-1] if h.get("train_loss") else np.nan,
                           "val_first": h["val_loss"][0] if h.get("val_loss") else np.nan})
            wall.setdefault(name, []).append(h)
    pd.DataFrame(curves).to_csv(out / "training_summary.csv", index=False)

    _figures(out / "figures", wall, pits, rel_all, tabs, summ, pw)
    _markdown(out, tabs, summ, pw, slices, pd.DataFrame(pit_rows), pd.DataFrame(zrows))
    print("results written to", out)


def _figures(fig_dir, wall, pits, rel_all, tabs, summ, pw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in ("deepar_v2@s42", "hybrid@s42"):
        hs = wall.get(name)
        if not hs:
            continue
        fig, axes = plt.subplots(1, len(hs), figsize=(3.0 * len(hs), 2.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, h in zip(axes, hs):
            ax.plot(h["train_step"], h["train_loss"], lw=0.8, alpha=0.7, label="train NLL (batch mean)")
            ax.plot(h["val_step"], h["val_loss"], "o-", ms=2.5, lw=1.2, label="validation NLL")
            if h.get("best_step") is not None:
                ax.axvline(h["best_step"], color="k", ls=":", lw=0.8)
            ax.set_title(f"{h.get('origin')} ({'fine-tune' if h.get('warm') else 'from scratch'})", fontsize=8)
            ax.set_xlabel("step", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("NLL per observed day", fontsize=8)
        axes[0].legend(fontsize=6.5)
        fig.tight_layout()
        fig.savefig(fig_dir / f"training_curves_{name.split('@')[0]}.png", dpi=170)
        plt.close(fig)
    if pits:
        sel = [m for m in ("deepar_p1@s42", "deepar_v2@s42", "deepar_v2_nomask@s42", "hybrid@s42") if m in pits]
        fig, axes = plt.subplots(1, len(sel), figsize=(2.6 * len(sel), 2.4), sharey=True)
        for ax, m in zip(np.atleast_1d(axes), sel):
            ax.bar(np.arange(10) / 10 + 0.05, pits[m], width=0.095, color="#4C72B0")
            ax.axhline(1, color="k", ls=":", lw=0.8)
            ax.set_title(m.replace("@s42", ""), fontsize=8)
            ax.set_xlabel("randomised PIT", fontsize=8)
            ax.tick_params(labelsize=7)
        np.atleast_1d(axes)[0].set_ylabel("density", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "pit_histograms.png", dpi=170)
        plt.close(fig)
    if rel_all:
        fig, ax = plt.subplots(figsize=(3.8, 3.5))
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        for r in rel_all:
            m = r.model.iloc[0]
            if m in ("deepar_v2@s42", "deepar_v2_nomask@s42", "hybrid@s42", "hyb_zi_naive@s42"):
                ax.plot(r.mean_pred, r.obs_rate, "o-", ms=2.5, lw=1.1, label=m.replace("@s42", ""))
        ax.set_xlabel("predicted P(Y=0)", fontsize=8)
        ax.set_ylabel("observed zero rate", fontsize=8)
        ax.legend(fontsize=6.5)
        ax.set_title("Zero-mass reliability (test, warm)", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "zero_reliability_models.png", dpi=170)
        plt.close(fig)
    # main comparison bar chart
    t = tabs["all"]
    order = [m for m in ("snaive4", "kalman", "lgbm", "kalman_lgbm", "ens_global_local", "deepar_p1@s42",
                         "deepar_v2@s42", "deepar_v2_nomask@s42", "hyb_no_dl@s42", "hybrid@s42") if m in t.index]
    if order:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
        for ax, (mode, met) in zip(axes, [("all", "pinball"), ("non_censored", "pinball")]):
            vals = tabs[mode].loc[order, met]
            ax.barh(range(len(order)), vals, color=["#C44E52" if m.startswith("hybrid") else "#4C72B0" for m in order])
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([m.replace("@s42", "") for m in order], fontsize=7)
            ax.set_xlim(vals.min() * 0.95, vals.max() * 1.01)
            ax.set_title(f"mean pinball, {mode.replace('_', '-')} days (seed 42)", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "main_pinball.png", dpi=170)
        plt.close(fig)


def _fmt(x, d=4):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def _markdown(out, tabs, summ, pw, slices, pit, zs):
    L = ["# Phase 2 results (auto-generated by scripts/phase2_analysis.py; do not edit by hand)", "",
         "Cells: warm series, available days, 4 test origins (2016-01-31 .. 2016-04-24) x 28 days. "
         "`@sNN` = seed. All-days = recorded sales; non-censored = days NOT in a hindsight-flagged long zero run.", ""]
    for mode in ("all", "non_censored"):
        t = tabs[mode]
        L += [f"## Main table ({mode})", "", "| model | phase | WAPE | MAE | RMSE | pinball | cov80 | cov90 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for m, r in t.sort_values("pinball").iterrows():
            L.append(f"| {m} | {r.phase} | {_fmt(r.WAPE)} | {_fmt(r.MAE)} | {_fmt(r.RMSE)} | {_fmt(r.pinball)} | {_fmt(r.cov80, 3)} | {_fmt(r.cov90, 3)} |")
        L.append("")
    L += ["## Seeds (mean +- sd over 3 seeds)", "", "| model | eval days | WAPE | pinball | cov80 |", "|---|---|---|---|---:|"]
    for _, r in summ.iterrows():
        L.append(f"| {r.model} | {r.eval_days} | {r.WAPE_mean:.4f} +- {r.WAPE_std:.4f} | {r.pinball_mean:.4f} +- {r.pinball_std:.4f} | {r.cov80_mean:.3f} |")
    L += ["", "## Paired bootstrap (series resampled, n_boot draws (1,000 for the tracked package); diff = changed - baseline, negative = changed better)", "",
          "| experiment | days | metric | changed | baseline | diff | 95% CI | sig. | origins better |", "|---|---|---|---:|---:|---:|---|---|---|"]
    for _, r in pw.iterrows():
        L.append(f"| {r.experiment} | {r.eval_days} | {r.metric} | {r.value_changed:.4f} | {r.value_baseline:.4f} | "
                 f"{r['diff']:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {'yes' if r.significant else 'no'} | {r.origins_a_better}/{r.n_origins} |")
    if len(pit):
        L += ["", "## Randomised PIT (max |density-1| over 10 bins; 0 = perfectly flat)", "", "| model | max dev | mean PIT |", "|---|---:|---:|"]
        for _, r in pit.iterrows():
            L.append(f"| {r.model} | {r.pit_max_abs_dev:.3f} | {r.pit_mean:.3f} |")
    if len(zs):
        z = zs[zs.slice == "all_warm"]
        L += ["", "## P(recorded zero) scores of the full predictive distributions", "", "| model | Brier | log loss | ECE | AUC |", "|---|---:|---:|---:|---:|"]
        for _, r in z.iterrows():
            L.append(f"| {r.model} | {r.brier:.4f} | {r.log_loss:.4f} | {r.ece:.4f} | {r.roc_auc:.4f} |")
    (out / "RESULTS_PHASE2.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
