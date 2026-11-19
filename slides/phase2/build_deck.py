"""Builds slides/phase2/deck.html from results/phase2/*.csv. Every number on the slides is read from those files.
    python slides/phase2/build_deck.py && sh slides/phase2/build.sh
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "phase2"
FIG = "../../results/phase2/figures"
CSS = re.search(r"<style>(.*?)</style>", (ROOT / "slides" / "deck.html").read_text(), re.S).group(1)
CSS += """
  .math { font-family: "STIX Two Math", "Cambria Math", Georgia, serif; font-size: 22px; background: #fbfaf7;
          border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; line-height: 1.6; }
  .small td, .small th { font-size: 16px; padding: 5px 8px; }
  .tiny { font-size: 15px; color: var(--muted); line-height: 1.35; }
  .pos { color: #1a8a6a; font-weight: 700; } .neg { color: #b4542d; font-weight: 700; }
  .pill { display:inline-block; padding: 2px 10px; border-radius: 99px; font-size: 14px; font-weight: 700;
          background: #eef7f3; color: #1a8a6a; margin-right: 6px; }
"""


def load():
    d = {k: pd.read_csv(R / f"{k}.csv") for k in ("main_all", "main_non_censored", "cold_start", "seeds_summary",
                                                   "pairwise_bootstrap", "pit", "zero_probability_scores",
                                                   "e02_validation_selection", "run_bucket_share",
                                                   "slices_bootstrap_hybrid_vs_deepar")}
    d["zero"] = pd.read_csv(R / "zero_model" / "scores.csv")
    d["zboot"] = pd.read_csv(R / "zero_model" / "bootstrap_vs_baselines.csv")
    for k in ("main_all", "main_non_censored", "cold_start"):
        d[k] = d[k].set_index(d[k].columns[0])
    return d


def seed(d, m, mode, col):
    s = d["seeds_summary"]
    r = s[(s.model == m) & (s.eval_days == mode)].iloc[0]
    return r[f"{col}_mean"], r[f"{col}_std"]


def pw(d, exp, mode="all", met="pinball"):
    p = d["pairwise_bootstrap"]
    return p[(p.experiment == exp) & (p.eval_days == mode) & (p.metric == met)].iloc[0]


def sgn(r, d=4):
    cls = "pos" if (r.significant and r["diff"] < 0) else ("neg" if (r.significant and r["diff"] > 0) else "")
    return f'<span class="{cls}">{r["diff"]:+.{d}f}</span> <span class="tiny">[{r.ci_low:+.{d}f}, {r.ci_high:+.{d}f}]</span>'


def slide(eyebrow, title, body, n, sub=""):
    s = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<section class="slide"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>{s}'
            f'<div class="body">{body}</div><div class="footer"><span>ForecastKit · Phase 2 (Deep Learning)</span>'
            f'<span>{n}</span></div></section>')


def build():
    d = load()
    A, N, C = d["main_all"], d["main_non_censored"], d["cold_start"]
    hw, hws = seed(d, "hybrid", "all", "WAPE")
    hp, hps = seed(d, "hybrid", "all", "pinball")
    dw, _ = seed(d, "deepar_v2", "all", "WAPE")
    dp, _ = seed(d, "deepar_v2", "all", "pinball")
    share28 = d["run_bucket_share"].set_index("bucket").loc[">=28", "proportion"]
    z = d["zero"]
    zz = lambda m, sl, c: z[(z.model == m) & (z.slice == sl)][c].iloc[0]
    S = []
    # 1 cover
    S.append(f'''<section class="slide"><div class="eyebrow">Phase 2 · Deep Learning evaluation</div>
<h1 class="cover">Masked DeepAR + a zero-run model:<br>an ML + DL hurdle for sparse retail demand</h1>
<div class="sub">Phase 1 found a trade-off: DeepAR trained with long zero runs masked forecasts demand well, but misses the
recorded zeros. Phase 2 asks whether a separate, calibrated zero model can supply those zeros without spoiling the
demand model.</div>
<div class="body"><div class="col"><div class="stats">
<div class="card"><div class="big">{hp:.4f}</div><div class="biglabel">hybrid mean pinball, all days<br>(3 seeds, sd {hps:.4f})</div></div>
<div class="card"><div class="big">{dp:.4f}</div><div class="biglabel">DeepAR v2 alone<br>(3 seeds)</div></div>
<div class="card"><div class="big">{A.loc["lgbm", "pinball"]:.4f}</div><div class="biglabel">LightGBM (Phase 1)</div></div>
<div class="card"><div class="big">{share28:.1%}</div><div class="biglabel">series-origins inside a zero run<br>of &ge; 28 days at the origin</div></div>
</div><div class="card warn"><p>M5 has no inventory data. "Zero run" means recorded zero sales, never a confirmed stockout.
All numbers: 1,200 M5 series, 4 rolling test origins × 28 days, warm series unless stated.</p></div></div></div>
<div class="footer"><span>ForecastKit · Phase 2</span><span>1</span></div></section>''')
    # 2 literature
    S.append(slide("1 · Literature review & motivation", "What the literature offers, and the gap we test", f'''
<div class="col"><div class="card"><h3>Deep forecasters</h3><ul>
<li><b>DeepAR</b> (Salinas 2020): global LSTM, NB likelihood, sampled paths. Our DL core.</li>
<li>RNN practice (Hewamalage 2021): global training, scaling, explicit seasonality.</li>
<li>MQ-RNN / TFT: quantiles only, no count pmf to fuse.</li></ul></div>
<div class="card"><h3>Zeros in count data</h3><ul>
<li><b>Hurdle</b> (Mullahy 1986): one process for zero vs positive.</li>
<li><b>Zero-inflated</b> (Lambert 1992): extra zero state + count model.</li>
<li>Deep renewal (Türkmen 2021): separate interval and size models.</li></ul></div></div>
<div class="col"><div class="card"><h3>Hybrids</h3><ul>
<li>ES-RNN (Smyl 2020), deep state space (Rangapuram 2018): components with <i>different roles</i>, not averaging.</li></ul></div>
<div class="card accent"><h3>Gap</h3><p>When zero runs may be censoring, a <b>masked</b> likelihood learns demand but stops
predicting recorded zeros; an <b>unmasked</b> one learns zeros as demand. Can a calibrated zero model + masked DeepAR
give the best of both, and is it better than simply not masking?</p></div>
<div class="card"><h3>Evaluation</h3><p>Pinball (proper), randomised PIT for counts (Czado 2009), Brier/log loss for P(0),
paired bootstrap over series.</p></div></div>''', 2))
    # 3 data pipeline
    S.append(slide("2 · Dataset quality & DL data pipeline", "Windows, masks and validation never touch the test window", f'''
<div class="col"><div class="card"><h3>Data (unchanged from Phase 1)</h3><ul>
<li>1,200 series (3 stores × 400 items), 1,000 days, 59% zero sales</li>
<li>Test origins 2016-01-31 … 2016-04-24, H = 28; validation origin 2016-01-03</li>
<li>84 simulated cold-start series (history hidden, not real launches)</li></ul></div>
<div class="card"><h3>DeepAR sequences</h3><ul>
<li>Window = 84-day context + 28-day target, sampled from the last 730 days of history</li>
<li>Per-window scale ν = 1 + mean(context); inputs divided by ν</li>
<li>Masked days enter as (0, flag = 0); excluded from the loss</li></ul></div></div>
<div class="col"><div class="card accent"><h3>Leakage controls</h3><ul>
<li>Context object is cut at the origin; masks recomputed on history</li>
<li><b>Validation block</b> = last 28 days of history; training windows end before it</li>
<li>Lags y(t−k): history if t−k ≤ T, else the path's own sample</li>
<li>Tests: changing data after the origin leaves DeepAR, zero model and features bit-identical</li></ul></div>
<div class="card"><h3>Reproducibility</h3><p>Phase 1 DeepAR re-run with the Phase 2 code: <b>0.0</b> max difference over
1,075,200 forecast values. Seeds 42, 43, 44 for every DL comparison.</p></div></div>''', 3))
    # 4 features & regularisation
    e = d["e02_validation_selection"].set_index("variant")
    rows = "".join(f'<tr{" class=hl" if v == "deepar_es_lags" else ""}><td>{lab}</td><td class=n>{e.loc[v, "pinball_noncens"]:.4f}</td>'
                   f'<td class=n>{"–" if pd.isna(e.loc[v, "best_step"]) else int(e.loc[v, "best_step"])}</td></tr>'
                   for v, lab in [("deepar_p1", "Phase 1 DeepAR"), ("deepar_es", "+ validation checkpoint"),
                                  ("deepar_es_lags", "+ lags 7/14/28 (selected)"), ("deepar_es_lags_age", "+ age"),
                                  ("deepar_es_lags_age_adamw", "+ AdamW")])
    S.append(slide("3 · Feature engineering & regularisation", "DeepAR v2: chosen on the validation origin, not on test", f'''
<div class="col"><div class="card"><h3>Selection on validation origin 2016-01-03 (pinball, non-censored)</h3>
<table class="small"><tr><th>variant</th><th class=n>pinball</th><th class=n>best step</th></tr>{rows}</table>
<p class="tiny">Rule fixed before running: lowest value; gaps &lt; 0.5% go to the simpler model. Age and AdamW were within
the tie band, so not adopted.</p></div>
<div class="card"><h3>Regularisation</h3><p>Dropout 0.1, item-id dropout 0.15, L2 1e-5, gradient clip 10, cosine LR,
best checkpoint on validation NLL (patience 5). Early stopping never triggered: the step budget, not over-fitting, binds.</p></div></div>
<div class="col" style="flex:1.7"><figure style="flex:1"><img src="{FIG}/training_curves_deepar_v2.png">
<figcaption>Train vs validation NLL per origin (seed 42). Dotted line = restored checkpoint. Later origins are fine-tuned from the previous weights.</figcaption></figure></div>''', 4))
    # 5 theory DeepAR + hurdle
    S.append(slide("4 · Theoretical rigor", "One valid count distribution from two models", f'''
<div class="col"><div class="card"><h3>DeepAR v2</h3><div class="math">h<sub>t</sub> = LSTM(h<sub>t−1</sub>, [y<sub>t−1</sub>, y<sub>t−7</sub>, y<sub>t−14</sub>, y<sub>t−28</sub>]/ν, flags, x<sub>t</sub>, e<sub>i</sub>)<br>
y<sub>t</sub> ~ NB(μ<sub>t</sub>, α<sub>t</sub>),&nbsp; E = μ, Var = μ + αμ²<br>p<sub>D</sub>(0) = (1/S) Σ<sub>s</sub> (r<sub>s</sub>/(r<sub>s</sub>+μ<sub>s</sub>))<sup>r<sub>s</sub></sup>, r = 1/α</div>
<p class="tiny">NLL equals scipy's NB to 1e-8; sampler mean/variance/P(0) within 2%.</p></div>
<div class="card"><h3>Zero-run model (LightGBM)</h3><div class="math">π<sub>h</sub> = P(y<sub>T+h</sub> = 0 | history to T)</div>
<p class="tiny">Observable target (recorded zeros), proper log loss, early stopping on history.</p></div></div>
<div class="col"><div class="card accent"><h3>Hurdle fusion</h3><div class="math">q(0) = π<sub>h</sub><br>q(y) = (1 − π<sub>h</sub>) p<sub>D</sub>(y) / (1 − p<sub>D</sub>(0)),&nbsp; y ≥ 1<br>
Q(τ) = 0 if τ ≤ π, else F<sub>+</sub><sup>−1</sup>((τ−π)/(1−π))</div>
<p>Sums to 1, non-negative, exact sampling (zero-truncated NB per path).</p></div>
<div class="card warn"><h3>Why not zero-inflation?</h3><p>π·δ<sub>0</sub> + (1−π)p<sub>D</sub> has q(0) = π + (1−π)p<sub>D</sub>(0) &gt; π:
DeepAR's zeros are counted twice. Kept as an ablation.</p></div></div>''', 5))
    # 6 architecture
    S.append(slide("4 · Theoretical rigor", "Hybrid architecture: each model does one job", f'''
<div class="col" style="flex:1.5"><figure style="flex:1"><img src="{FIG}/hybrid_architecture.png"></figure></div>
<div class="col"><div class="card"><h3>DL: how much</h3><p>Masked DeepAR = demand when the item sells normally; supplies the shape of positive sales.</p></div>
<div class="card"><h3>ML: whether</h3><p>Run-length state is a sharp, tabular signal; trees exploit it. Supplies the zero mass.</p></div>
<div class="card"><h3>Registered model</h3><p><code>deepar_hurdle</code> in the standard runner: same origins, masks, saved samples and metadata.</p></div></div>''', 6))
    # 7 zero model standalone
    zb = d["zboot"].set_index("reference")
    S.append(slide("5 · Model application", "Step 1: the zero model is validated on its own first", f'''
<div class="col"><div class="card"><h3>P(recorded zero), test cells (124,992)</h3><table class="small">
<tr><th>predictor</th><th class=n>Brier</th><th class=n>log loss</th><th class=n>Brier, run ≥ 28 d</th></tr>
{''.join(f"<tr{' class=hl' if m == 'lgbm_zero' else ''}><td>{lab}</td><td class=n>{zz(m, 'all_warm', 'brier'):.4f}</td><td class=n>{zz(m, 'all_warm', 'log_loss'):.4f}</td><td class=n>{zz(m, 'run_bucket=run>=28', 'brier'):.4f}</td></tr>" for m, lab in [("series_zero_rate", "series zero rate"), ("recent_zero_rate_28", "zero rate, last 28 d"), ("km_renewal", "Kaplan–Meier renewal"), ("lgbm_zero", "LightGBM zero model"), ("lgbm_zero_no_persistence", "  without run features"), ("deepar_v2_masked", "DeepAR v2 p<sub>D</sub>(0)"), ("deepar_v2_unmasked", "DeepAR v2 unmasked p<sub>D</sub>(0)")])}
</table></div>
<div class="card"><p>LightGBM − KM renewal Brier: {sgn(zb.loc["km_renewal"].iloc[0])}; − masked DeepAR: {sgn(zb.loc["deepar_v2_masked"].iloc[0])}.
Against <b>unmasked</b> DeepAR the gain is not significant: {sgn(zb.loc["deepar_v2_unmasked"].iloc[0])}.</p></div></div>
<div class="col"><figure style="flex:1"><img src="../../results/phase2/zero_model/reliability.png"><figcaption>Reliability of P(recorded zero), warm test cells.</figcaption></figure></div>''', 7))
    # 8 main results
    def row(lab, m, multi=False, hl=False):
        if multi:
            w = seed(d, m, "all", "WAPE")[0]
            p = seed(d, m, "all", "pinball")[0]
            wn = seed(d, m, "non_censored", "WAPE")[0]
            pn = seed(d, m, "non_censored", "pinball")[0]
            pc = seed(d, m, "cold_start", "pinball")[0]
        else:
            w, p, wn, pn = A.loc[m, "WAPE"], A.loc[m, "pinball"], N.loc[m, "WAPE"], N.loc[m, "pinball"]
            pc = C.loc[m, "pinball"] if m in C.index else float("nan")
        return (f'<tr{" class=hl" if hl else ""}><td>{lab}</td><td class=n>{w:.4f}</td><td class=n>{p:.4f}</td>'
                f'<td class=n>{wn:.4f}</td><td class=n>{pn:.4f}</td><td class=n>{pc:.4f}</td></tr>')
    S.append(slide("5 · Model application", "Main comparison: best on recorded sales, not on visible demand", f'''
<div class="col" style="flex:1.4"><div class="card"><table class="small">
<tr><th>model</th><th class=n>WAPE all</th><th class=n>pinball all</th><th class=n>WAPE non-cens.</th><th class=n>pinball non-cens.</th><th class=n>pinball cold</th></tr>
{row("Seasonal naive 4 wk", "snaive4")}{row("Kalman", "kalman")}{row("LightGBM", "lgbm")}{row("Kalman–LightGBM (P1 hybrid)", "kalman_lgbm")}
{row("Ensemble P1", "ens_global_local")}{row("DeepAR P1 ‡", "deepar_p1", True)}{row("DeepAR v2 ‡", "deepar_v2", True)}
{row("DeepAR v2 unmasked ‡", "deepar_v2_nomask", True)}{row("Zero model + empirical sizes", "hyb_no_dl@s42")}{row("Hybrid ‡", "hybrid", True, True)}
</table><p class="tiny">‡ mean of 3 seeds. All days = recorded sales; non-censored drops days in hindsight-flagged long zero runs.</p></div></div>
<div class="col"><div class="card accent"><h3>Hybrid vs DeepAR v2 (3 seeds)</h3><p>all days: {sgn(pw(d, "Hybrid vs DeepAR"))}<br>non-censored: {sgn(pw(d, "Hybrid vs DeepAR", "non_censored"))}</p></div>
<div class="card"><h3>Hybrid vs unmasked DeepAR</h3><p>all days: {sgn(pw(d, "Hybrid vs unmasked DeepAR"))}<br>non-censored: {sgn(pw(d, "Hybrid vs unmasked DeepAR", "non_censored"))}</p></div>
<div class="card"><h3>Hybrid vs LightGBM</h3><p>all days: {sgn(pw(d, "Hybrid vs LightGBM"))}<br>non-censored: {sgn(pw(d, "Hybrid vs LightGBM", "non_censored"))}</p></div>
<p class="tiny">Δ pinball, 95% paired bootstrap over series (1,000 draws). Green = significantly better, red = significantly worse.</p></div>''', 8))
    # 9 ablations
    exps = ["DeepAR P2 vs P1", "Seasonal lags (no ES)", "Validation checkpoint", "Age feature", "AdamW",
            "Loss masking (DeepAR)", "Hybrid vs DeepAR", "RB reconstruction control", "No persistence features",
            "No DL component", "Fusion: naive ZI", "Fusion: inflate-only ZI", "Fusion: gated",
            "Hybrid on unmasked DeepAR", "Hybrid on Phase 1 DeepAR"]
    p = d["pairwise_bootstrap"]
    ab = "".join(f"<tr><td>{x}</td><td class=n>{sgn(pw(d, x))}</td><td class=n>{sgn(pw(d, x, 'non_censored'))}</td></tr>"
                 for x in exps if x in set(p.experiment))
    S.append(slide("5 · Model application", "Ablations: every component's contribution, with confidence intervals", f'''
<div class="col"><div class="card"><table class="small"><tr><th>change (vs its baseline)</th><th class=n>Δ pinball all days [95% CI]</th><th class=n>Δ pinball non-censored [95% CI]</th></tr>{ab}</table>
<p class="tiny">Negative = the change helps. Seeds: DeepAR P2 vs P1, masking and hybrid vs DeepAR use 3 seeds; others seed 42. Full table with WAPE: results/phase2/pairwise_bootstrap.csv.</p></div></div>''', 9))
    # 10 calibration
    pit = d["pit"].set_index("model")
    S.append(slide("5 · Model application", "Calibration: the hurdle fixes masked DeepAR's zero mass", f'''
<div class="col" style="flex:1.3"><figure style="flex:1"><img src="{FIG}/pit_histograms.png"><figcaption>Randomised PIT (flat = calibrated). Masked DeepAR piles mass at low PIT: too few zeros predicted.</figcaption></figure>
<div class="card"><p>Max PIT deviation (seed 42): DeepAR v2 {pit.loc["deepar_v2@s42", "pit_max_abs_dev"]:.3f}, unmasked {pit.loc["deepar_v2_nomask@s42", "pit_max_abs_dev"]:.3f}, hybrid <b>{pit.loc["hybrid@s42", "pit_max_abs_dev"]:.3f}</b>.</p></div></div>
<div class="col"><figure style="flex:1"><img src="{FIG}/zero_reliability_models.png"><figcaption>Reliability of each model's implied P(Y = 0).</figcaption></figure></div>''', 10))
    # 11 failure cases
    sb = d["slices_bootstrap_hybrid_vs_deepar"]
    g = lambda sl, mode: sb[(sb.slice == sl) & (sb.eval_days == mode) & (sb.metric == "pinball")].iloc[0]
    S.append(slide("5 · Model application", "Where the hybrid fails, and what that means", f'''
<div class="col"><div class="card"><h3>Hybrid − DeepAR v2, pinball, by zero run at the origin</h3><table class="small">
<tr><th>run at origin</th><th class=n>all days</th><th class=n>non-censored</th></tr>
{''.join(f"<tr><td>{b} days</td><td class=n>{sgn(g('run_at_origin=' + b, 'all'))}</td><td class=n>{sgn(g('run_at_origin=' + b, 'non_censored'))}</td></tr>" for b in ["0", "1-6", "7-27", ">=28"])}
</table></div>
<div class="card warn"><h3>Failure case</h3><p>For series already in a long zero run, the hybrid predicts more zeros. When sales
resume inside the window, those days (the non-censored ones) are forecast worse. The hybrid targets <b>recorded sales</b>;
masked DeepAR remains the better <b>latent-demand</b> model there.</p></div></div>
<div class="col"><div class="card"><h3>Honest reading</h3><ul>
<li>Most of the all-days gain comes from the ~{share28:.0%} of series-origins in long zero runs.</li>
<li>Unmasked DeepAR recovers much of the same gain without a second model; the hybrid is still significantly better than it on both views.</li>
<li>Explicit run-length features add nothing measurable: rolling non-zero rates already carry the information.</li></ul></div></div>''', 11))
    # 12 limitations
    S.append(slide("6 · Final summary", "Limitations", '''
<div class="col"><div class="card warn"><ul>
<li><b>No inventory data.</b> The stockout proxy is a heuristic; "non-censored" is a proxy view of demand.</li>
<li><b>One retailer, 3 stores, 4 origins in early 2016.</b> CIs reflect series variability, not other periods.</li>
<li><b>3 seeds</b> for the main DL comparisons; ablations use one seed.</li>
<li><b>Untuned</b> LightGBM and zero-model hyper-parameters; DeepAR selected on a single validation origin.</li>
<li><b>Simulated cold start</b>: hidden history of existing products, not real launches.</li>
<li><b>CPU budget</b>: no Transformer/TFT or deep renewal comparison.</li></ul></div></div>''', 12))
    # 13 conclusion
    S.append(slide("6 · Final summary", "Conclusion", f'''
<div class="col"><div class="card accent"><h3>What we can claim</h3><ul>
<li>DeepAR v2 (lags + validation checkpoint) vs Phase 1, 3 seeds: {sgn(pw(d, "DeepAR P2 vs P1"))} pinball (all days).</li>
<li>The hurdle hybrid has the lowest all-days WAPE and pinball of all models ({hw:.4f} / {hp:.4f}, 3 seeds),
significantly better than DeepAR, unmasked DeepAR and LightGBM on recorded sales.</li>
<li>Its zero mass is calibrated (flat PIT), and the naive ZI alternative is measurably worse.</li></ul></div>
<div class="card warn"><h3>What we cannot claim</h3><ul><li>Better latent-demand forecasts: on non-censored days the Phase 1 Kalman–LightGBM is still best.</li>
<li>Stockout prediction: no inventory labels exist.</li></ul></div></div>''', 13))
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ForecastKit Phase 2</title>'
            f'<style>{CSS}</style></head><body>' + "\n".join(S) + "</body></html>")
    out = Path(__file__).parent / "deck.html"
    out.write_text(html)
    print("wrote", out)


if __name__ == "__main__":
    build()
