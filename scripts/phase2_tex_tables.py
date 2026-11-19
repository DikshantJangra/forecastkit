"""Generates the LaTeX tables of the Phase 2 report from results/phase2/*.csv (no hand-typed numbers).
    python scripts/phase2_tex_tables.py      -> paper/generated/*.tex
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/phase2")
OUT = Path("paper/generated")
NAMES = {"snaive4": "Seasonal naive (4 wk)", "kalman": "Kalman (local)", "lgbm": "LightGBM",
         "kalman_lgbm": "Kalman--LightGBM (P1 hybrid)", "ens_global_local": "Ensemble P1 (K, L, DeepAR P1)",
         "ens_global_local_v2": "Ensemble P2 (K, L, DeepAR v2)", "deepar_p1": "DeepAR P1",
         "deepar_v2": "DeepAR v2", "deepar_v2_nomask": "DeepAR v2, unmasked", "hyb_no_dl": "Zero model + empirical sizes",
         "hybrid": "\\textbf{Hybrid (DeepAR v2 + zero model)}"}
MULTI = ["deepar_p1", "deepar_v2", "deepar_v2_nomask", "hybrid"]


def f(x, d=4):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def main_table():
    a = pd.read_csv(R / "main_all.csv", index_col=0)
    n = pd.read_csv(R / "main_non_censored.csv", index_col=0)
    c = pd.read_csv(R / "cold_start.csv", index_col=0)
    seeds = pd.read_csv(R / "seeds_summary.csv")
    rows = []
    for key in ["snaive4", "kalman", "lgbm", "kalman_lgbm", "ens_global_local", "ens_global_local_v2", "deepar_p1",
                "deepar_v2", "deepar_v2_nomask", "hyb_no_dl", "hybrid"]:
        if key in MULTI:
            def g(mode, col, key=key):
                s = seeds[(seeds.model == key) & (seeds.eval_days == mode)]
                return s[f"{col}_mean"].iloc[0] if len(s) else np.nan
            vals = [g("all", "WAPE"), g("all", "pinball"), g("all", "cov80"), g("all", "cov90"),
                    g("non_censored", "WAPE"), g("non_censored", "pinball"), g("cold_start", "WAPE"), g("cold_start", "pinball")]
            label = NAMES[key] + "$^{\\dagger}$"
        else:
            m = key if key in a.index else f"{key}@s42"
            if m not in a.index:
                continue
            vals = [a.loc[m, "WAPE"], a.loc[m, "pinball"], a.loc[m, "cov80"], a.loc[m, "cov90"], n.loc[m, "WAPE"],
                    n.loc[m, "pinball"], c.loc[m, "WAPE"] if m in c.index else np.nan,
                    c.loc[m, "pinball"] if m in c.index else np.nan]
            label = NAMES[key]
        rows.append((label, vals))
    best = [min(range(len(rows)), key=lambda i: rows[i][1][j] if not pd.isna(rows[i][1][j]) else 9) for j in range(8)]
    L = ["\\begin{table*}[t]", "\\caption{Test results, 4 origins $\\times$ 28 days. All days: 124,992 cells (recorded sales); "
         "non-censored: 114,127 cells; simulated cold start: 9,408 cells. $^{\\dagger}$mean over seeds 42, 43, 44 "
         "(sd in Table~\\ref{tab:seeds}); other Phase~2 rows seed 42. Lower is better except coverage (nominal 0.80/0.90).}",
         "\\label{tab:main2}", "\\centering", "\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
         "& \\multicolumn{4}{c}{All days} & \\multicolumn{2}{c}{Non-censored} & \\multicolumn{2}{c}{Cold start}\\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
         "Model & WAPE & Pinball & Cov80 & Cov90 & WAPE & Pinball & WAPE & Pinball\\\\", "\\midrule"]
    for i, (label, v) in enumerate(rows):
        cells = []
        for j, x in enumerate(v):
            s = f(x, 3 if j in (2, 3) else 4)
            if best[j] == i and j not in (2, 3):
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        L.append(f"{label} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(L)


def seeds_table():
    s = pd.read_csv(R / "seeds_summary.csv")
    L = ["\\begin{table}[t]", "\\caption{Seed variability: mean $\\pm$ sd over seeds 42, 43, 44 (warm series).}",
         "\\label{tab:seeds}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         "& \\multicolumn{2}{c}{All days} & \\multicolumn{2}{c}{Non-censored}\\\\", "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
         "Model & WAPE & Pinball & WAPE & Pinball\\\\", "\\midrule"]
    for m in MULTI:
        a = s[(s.model == m) & (s.eval_days == "all")]
        n = s[(s.model == m) & (s.eval_days == "non_censored")]
        if not len(a):
            continue
        a, n = a.iloc[0], n.iloc[0]
        name = "Hybrid" if m == "hybrid" else NAMES[m]
        L.append(f"{name} & "
                 f"{a.WAPE_mean:.4f}$\\pm${a.WAPE_std:.4f} & {a.pinball_mean:.4f}$\\pm${a.pinball_std:.4f} & "
                 f"{n.WAPE_mean:.4f}$\\pm${n.WAPE_std:.4f} & {n.pinball_mean:.4f}$\\pm${n.pinball_std:.4f}\\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L)


def ablation_table():
    p = pd.read_csv(R / "pairwise_bootstrap.csv")
    p = p[p.metric == "pinball"]
    L = ["\\begin{table*}[t]", "\\caption{Ablations and pairwise comparisons on mean pinball loss. $\\Delta$ = changed $-$ "
         "baseline (negative: change helps); 95\\% paired-bootstrap CI over series (1,000 draws); * = CI excludes 0; "
         "``orig.'' = number of the 4 origins on which the change helps.}", "\\label{tab:abl2}", "\\centering", "\\footnotesize",
         "\\begin{tabular}{@{}llrrlrrlr@{}}", "\\toprule",
         "& & \\multicolumn{3}{c}{All days} & \\multicolumn{3}{c}{Non-censored} & \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         "Experiment & Unit & Value & $\\Delta$ & 95\\% CI & Value & $\\Delta$ & 95\\% CI & orig.\\\\", "\\midrule"]
    for exp, g in p.groupby("experiment", sort=False):
        a = g[g.eval_days == "all"].iloc[0]
        n = g[g.eval_days == "non_censored"].iloc[0]
        star = lambda r: "*" if r.significant else ""
        L.append(f"{exp} & {a.unit} & {a.value_changed:.4f} & {a['diff']:+.4f}{star(a)} & [{a.ci_low:+.4f}, {a.ci_high:+.4f}] & "
                 f"{n.value_changed:.4f} & {n['diff']:+.4f}{star(n)} & [{n.ci_low:+.4f}, {n.ci_high:+.4f}] & {a.origins_a_better}/{a.n_origins}\\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(L).replace("_", "\\_").replace("\\\\_", "\\_")


def zero_table():
    z = pd.read_csv(R / "zero_model" / "scores.csv")
    names = {"series_zero_rate": "Series zero rate", "recent_zero_rate_28": "Zero rate, last 28 d",
             "km_renewal": "Kaplan--Meier renewal", "lgbm_zero": "\\textbf{LightGBM zero model}",
             "lgbm_zero_no_persistence": "\\quad without run features", "deepar_p1": "DeepAR P1 $p_D(0)$",
             "deepar_v2_masked": "DeepAR v2 $p_D(0)$", "deepar_v2_unmasked": "DeepAR v2 unmasked $p_D(0)$"}
    L = ["\\begin{table}[t]", "\\caption{Zero-run persistence model (E04): scores of $P(y_{T+h}=0)$ on the test cells of "
         "warm series. ECE: 10-bin expected calibration error.}", "\\label{tab:zero}", "\\centering", "\\footnotesize",
         "\\begin{tabular}{@{}lcccc|cc@{}}", "\\toprule",
         "& \\multicolumn{4}{c|}{All warm cells (124,992)} & \\multicolumn{2}{c}{Run $\\ge$28 d at origin}\\\\",
         "Predictor & Brier & Log loss & ECE & AUC & Brier & Log loss\\\\", "\\midrule"]
    for m, lab in names.items():
        a = z[(z.slice == "all_warm") & (z.model == m)]
        b = z[(z.slice == "run_bucket=run>=28") & (z.model == m)]
        if not len(a):
            continue
        a, b = a.iloc[0], b.iloc[0]
        L.append(f"{lab} & {a.brier:.4f} & {a.log_loss:.4f} & {a.ece:.4f} & {a.roc_auc:.3f} & {b.brier:.4f} & {b.log_loss:.4f}\\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L)


def slices_table():
    s = pd.read_csv(R / "slices_bootstrap_hybrid_vs_deepar.csv")
    s = s[s.metric == "pinball"]
    share = pd.read_csv(R / "run_bucket_share.csv")
    L = ["\\begin{table}[t]", "\\caption{Hybrid minus DeepAR v2 (3 seeds each), mean pinball, by the causal zero-run state at "
         "the origin and by demand class. * = 95\\% CI excludes 0.}", "\\label{tab:slices2}", "\\centering", "\\footnotesize",
         "\\begin{tabular}{@{}lrrr@{}}", "\\toprule", "Slice & series & $\\Delta$ all days & $\\Delta$ non-cens.\\\\", "\\midrule"]
    for sl, g in s.groupby("slice", sort=False):
        a = g[g.eval_days == "all"].iloc[0]
        n = g[g.eval_days == "non_censored"].iloc[0]
        lab = sl.replace("run_at_origin=", "zero run at origin ").replace("intermittency=", "class: ").replace(">=", "$\\ge$")
        L.append(f"{lab} & {a.n_series} & {a['diff']:+.4f}{'*' if a.significant else ''} & {n['diff']:+.4f}{'*' if n.significant else ''}\\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    _ = share
    return "\n".join(L)


def e02_table():
    t = pd.read_csv(R / "e02_validation_selection.csv")
    lab = {"deepar_p1": "Phase 1 DeepAR", "deepar_es": "+ validation checkpoint", "deepar_es_lags": "+ lags 7/14/28 (selected)",
           "deepar_es_lags_age": "+ age", "deepar_es_lags_age_adamw": "+ AdamW (wd 0.01)"}
    L = ["\\begin{table}[t]", "\\caption{DeepAR design selection on the \\emph{validation} origin 2016-01-03 (E02). "
         "Rule fixed in advance: lowest non-censored pinball; gaps $<0.5\\%$ go to the simpler model.}", "\\label{tab:e02}",
         "\\centering", "\\footnotesize", "\\begin{tabular}{@{}lcccr@{}}", "\\toprule",
         "Variant & WAPE & Pinball & Pinball$_{nc}$ & best step\\\\", "\\midrule"]
    for _, r in t.iterrows():
        bs = "--" if pd.isna(r.best_step) else f"{int(r.best_step)}"
        L.append(f"{lab[r.variant]} & {r.WAPE_all:.4f} & {r.pinball_all:.4f} & {r.pinball_noncens:.4f} & {bs}\\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L)


def zero_mass_table():
    z = pd.read_csv(R / "zero_probability_scores.csv")
    p = pd.read_csv(R / "pit.csv")
    keep = ["deepar_p1@s42", "deepar_v2@s42", "deepar_v2_nomask@s42", "hybrid@s42", "hyb_zi_naive@s42",
            "hyb_zi_inflate@s42", "hyb_gated@s42", "hyb_no_dl@s42"]
    L = ["\\begin{table}[t]", "\\caption{Calibration of the full predictive distributions (seed 42, warm series, all days): "
         "Brier score and ECE of the implied $q(0)$, and the maximum deviation of the randomised-PIT histogram from "
         "uniform (10 bins; 0 = flat).}", "\\label{tab:calib}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}lccc@{}}",
         "\\toprule", "Model & Brier $q(0)$ & ECE $q(0)$ & PIT max dev.\\\\", "\\midrule"]
    for m in keep:
        a = z[(z.model == m) & (z.slice == "all_warm")]
        b = p[p.model == m]
        if not len(a):
            continue
        tex = m.replace("@s42", "").replace("_", "\\_")
        L.append(f"{tex} & {a.brier.iloc[0]:.4f} & {a.ece.iloc[0]:.4f} & {b.pit_max_abs_dev.iloc[0]:.3f}\\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("tab_main2", main_table), ("tab_seeds", seeds_table), ("tab_abl2", ablation_table),
                     ("tab_zero", zero_table), ("tab_slices2", slices_table), ("tab_e02", e02_table),
                     ("tab_calib", zero_mass_table)]:
        tex = fn()
        width = "\\textwidth" if "table*" in tex else "\\columnwidth"
        tex = tex.replace("\\begin{tabular}", f"\\resizebox{{{width}}}{{!}}{{\\begin{{tabular}}", 1)
        tex = tex.replace("\\end{tabular}", "\\end{tabular}}", 1)
        (OUT / f"{name}.tex").write_text(tex + "\n")
        print("wrote", OUT / f"{name}.tex")
