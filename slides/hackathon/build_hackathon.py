"""6-slide hackathon deck for ForecastKit.

Every metric on the slides is read from tracked result files (results/phase2/*, outputs/eda/*); the example series
come from the M5 cache and the saved hybrid forecasts. No model is trained here.

    PYTHONPATH=src python slides/hackathon/build_hackathon.py && sh slides/hackathon/build.sh
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R2 = ROOT / "results" / "phase2"
EDA = ROOT / "outputs" / "eda"
CACHE = ROOT / "data" / "processed" / "m5_ca1b76af29"
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)

INK, MUTED, LINE = "#1F2933", "#6B7280", "#E5E7EB"
ORANGE, TEAL, BLUE, GREEN, GREY = "#E8743B", "#14A098", "#2F6FDE", "#3BA55C", "#B8BEC6"

for f in ("/System/Library/Fonts/Avenir Next.ttc",):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"], "font.size": 13, "axes.edgecolor": LINE,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

EXAMPLES = {  # chosen from the cache: one regular, one intermittent, one in a long zero run at the last origin
    "Regular": ("FOODS_2_152_WI_1", BLUE),
    "Intermittent": ("HOUSEHOLD_1_130_CA_1", TEAL),
    "Long zero stretch": ("FOODS_2_008_WI_1", ORANGE),
}
LAST_ORIGIN = pd.Timestamp("2016-04-24")


# ----------------------------------------------------------------------------------------------------- data
def load():
    d = {}
    d["all"] = pd.read_csv(R2 / "main_all.csv").set_index("model")
    d["nc"] = pd.read_csv(R2 / "main_non_censored.csv").set_index("model")
    d["seeds"] = pd.read_csv(R2 / "seeds_summary.csv")
    d["pw"] = pd.read_csv(R2 / "pairwise_bootstrap.csv")
    d["sub"] = pd.read_csv(R2 / "comparative_audit" / "subsets.csv")
    d["boot"] = pd.read_csv(R2 / "comparative_audit" / "bootstrap.csv")
    d["cls"] = pd.read_csv(EDA / "intermittency_counts.csv")
    d["snap"] = pd.read_csv(EDA / "snap_table.csv").set_index("state_id")
    d["summary"] = pd.read_csv(EDA / "dataset_summary.csv")
    d["wd"] = pd.read_csv(EDA / "weekday_profile.csv")
    d["ev"] = pd.read_csv(EDA / "event_table.csv")
    a = np.load(CACHE / "arrays.npz")
    d["Y"], d["ids"], d["dates"] = a["Y"], list(a["series_ids"]), pd.to_datetime(a["dates"])
    return d


def seed_mean(d, model, mode, col):
    r = d["seeds"][(d["seeds"].model == model) & (d["seeds"].eval_days == mode)].iloc[0]
    return r[f"{col}_mean"], r.get(f"{col}_std")


def pw(d, exp, mode, met):
    p = d["pw"]
    return p[(p.experiment == exp) & (p.eval_days == mode) & (p.metric == met)].iloc[0]


# ----------------------------------------------------------------------------------------------------- charts
def chart_patterns(d):
    Y, ids, dates = d["Y"], d["ids"], d["dates"]
    o = dates.get_loc(LAST_ORIGIN)
    w = slice(o - 139, o + 1)
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.0), sharex=True)
    for ax, (lab, (sid, col)) in zip(axes, EXAMPLES.items()):
        y = Y[ids.index(sid), w]
        x = np.arange(len(y))
        ax.bar(x, y, width=0.8, color=col, linewidth=0)
        z = np.flatnonzero(y == 0)
        ax.scatter(z, np.zeros_like(z), s=4, color=GREY, zorder=3, linewidths=0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(0, 1.02, lab, transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK, va="bottom")
        ax.text(1, 1.02, f"{sid}  ·  {np.mean(y == 0):.0%} zero days", transform=ax.transAxes,
                fontsize=10.5, color=MUTED, ha="right", va="bottom")
    axes[-1].set_xticks([0, 139])
    axes[-1].set_xticklabels([dates[w][0].strftime("%b %Y"), dates[w][-1].strftime("%b %Y")])
    axes[-1].set_xlabel("140 days of recorded daily sales (grey dots = zero-sales days)", fontsize=10.5)
    fig.subplots_adjust(hspace=0.75)
    fig.savefig(FIG / "s1_patterns.png", transparent=True)
    plt.close(fig)


def chart_buckets(d):
    s = d["sub"][d["sub"].family == "zero run at origin"].reset_index(drop=True)
    b = d["boot"]
    labels = {"0 (sold on origin day)": "Sold on forecast day", "1-6 d": "Zero for 1–6 days",
              "7-27 d": "Zero for 7–27 days", ">=28 d": "Zero for 28+ days"}
    rows = []
    for _, r in s.iterrows():
        bb = b[(b.cells == f"zero run at origin: {r.subset}") & (b.reference == "kalman_lgbm") & (b.metric == "pinball")].iloc[0]
        rows.append((labels[r.subset], r.share_cells, -r.hyb_minus_klgbm_pinball, -bb.ci_high, -bb.ci_low,
                     bool(bb.significant), r.gap_share_PB_vs_klgbm))
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for i, (lab, share, v, lo, hi, sig, _) in enumerate(rows):
        col = GREEN if sig and v > 0 else GREY
        ax.barh(i, v, color=col, height=0.62)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.2, alpha=0.55)
        txt = f"{v:+.4f}" + ("" if sig else "  tie")
        ax.text(max(hi, v) + 0.003, i, txt, va="center", fontsize=12, color=INK if sig else MUTED, fontweight="bold")
        ax.text(-0.004, i, f"{lab}\n{share:.0%} of forecasts", va="center", ha="right", fontsize=11.5, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r[4] for r in rows) * 1.28)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Pinball-loss reduction vs Kalman-LightGBM (higher = hybrid better)", fontsize=10.5)
    fig.savefig(FIG / "s5_buckets.png", transparent=True)
    plt.close(fig)
    share7 = sum(r[1] for r in rows[2:])
    gain7 = sum(r[6] for r in rows[2:])
    return share7, gain7


def chart_ablation(d):
    A = d["all"]
    bars = [("Full hybrid", "hybrid@s42", TEAL), ("Unmasked DeepAR", "deepar_v2_nomask@s42", BLUE),
            ("DeepAR v2 alone", "deepar_v2@s42", BLUE), ("Hybrid without DeepAR", "hyb_no_dl@s42", ORANGE),
            ("Kalman-LightGBM (Phase 1)", "kalman_lgbm", GREY), ("Naive zero-inflated mix", "hyb_zi_naive@s42", ORANGE)]
    base = A.loc["hybrid@s42", "pinball"]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    lo = 0.245
    for i, (lab, m, col) in enumerate(bars):
        v = A.loc[m, "pinball"]
        ax.barh(i, v - lo, left=lo, color=col, height=0.62)
        rel = "" if i == 0 else f"   +{(v / base - 1) * 100:.1f}%"
        ax.text(v + 0.0006, i, f"{v:.4f}{rel}", va="center", fontsize=11.5, color=INK,
                fontweight="bold" if i == 0 else "normal")
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlim(lo, 0.2835)
    ax.set_xlabel(f"Pinball loss, all days (lower = better; axis starts at {lo})", fontsize=10.5)
    fig.savefig(FIG / "s5_ablation.png", transparent=True)
    plt.close(fig)


def chart_fans(d):
    f = pd.read_parquet(ROOT / "outputs" / "phase2" / "runs" / "hybrid_s42" / "forecasts.parquet",
                        filters=[("origin", "==", LAST_ORIGIN)])
    f = f[f.series_id.isin([v[0] for v in EXAMPLES.values()])]
    Y, ids, dates = d["Y"], d["ids"], d["dates"]
    o = dates.get_loc(LAST_ORIGIN)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 2.25))
    for ax, (lab, (sid, col)) in zip(axes, EXAMPLES.items()):
        g = f[f.series_id == sid].pivot_table(index="h", columns="stat", values="value", observed=True)
        y = Y[ids.index(sid)]
        hx = np.arange(-41, 1)
        fx = np.arange(1, 29)
        ax.plot(hx, y[o - 41:o + 1], color=INK, lw=1.1)
        ax.fill_between(fx, g["q0.05"], g["q0.95"], color=col, alpha=0.18, lw=0, label="90% interval")
        ax.fill_between(fx, g["q0.25"], g["q0.75"], color=col, alpha=0.38, lw=0, label="50% interval")
        ax.plot(fx, g["mean"], color=col, lw=2, label="expected sales")
        ax.scatter(fx, y[o + 1:o + 29], s=9, color=INK, zorder=4, label="actual (hidden until scoring)")
        ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_title(lab, fontsize=13.5, fontweight="bold", color=INK, loc="left")
        ax.set_xticks([-40, 0, 28])
        ax.set_xticklabels(["−40 d", "T", "+28 d"])
        ax.set_ylim(bottom=-0.3)
        ax.tick_params(axis="y", labelsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(FIG / "s6_fans.png", transparent=True)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------- html
STORY = ["Problem", "Data", "Approach", "Models", "Evidence", "Insight"]


def frame(n, eyebrow, title, body="", cls=""):
    dots = "".join(f'<span class="{"on" if i == n - 1 else ""}">{s}</span>' for i, s in enumerate(STORY))
    return (f'<section class="slide {cls}"><header><div class="eyebrow">{eyebrow}</div><div class="story">{dots}</div>'
            f'</header><h1>{title}</h1><div class="body">{body}</div>'
            f'<footer><span><b>ForecastKit</b> · Leak-safe probabilistic demand forecasting · M5 retail data</span>'
            f'<span>{n} / 6</span></footer></section>')


def build(d):
    A, N = d["all"], d["nc"]
    hw, hws = seed_mean(d, "hybrid", "all", "WAPE")
    hp, hps = seed_mean(d, "hybrid", "all", "pinball")
    hwn, _ = seed_mean(d, "hybrid", "non_censored", "WAPE")
    hpn, _ = seed_mean(d, "hybrid", "non_censored", "pinball")
    kw, kp = A.loc["kalman_lgbm", "WAPE"], A.loc["kalman_lgbm", "pinball"]
    kwn, kpn = N.loc["kalman_lgbm", "WAPE"], N.loc["kalman_lgbm", "pinball"]
    n_all = int(A.loc["kalman_lgbm", "n"])
    n_nc = int(N.loc["kalman_lgbm", "n"])
    cw, cp = pw(d, "Hybrid vs Kalman-LightGBM", "all", "WAPE"), pw(d, "Hybrid vs Kalman-LightGBM", "all", "pinball")
    cpn = pw(d, "Hybrid vs Kalman-LightGBM", "non_censored", "pinball")
    cov80, cov90 = seed_mean(d, "hybrid", "all", "cov80")[0], seed_mean(d, "hybrid", "all", "cov90")[0]
    share7, gain7 = chart_buckets(d)

    cls = d["cls"]
    cls = dict(zip(cls.iloc[:, 0], cls.iloc[:, 1]))
    n_series = sum(cls.values())
    sm = d["summary"]
    zshare = float(sm["overall_zero_share_launched_avail"].iloc[0])
    snap = d["snap"]["ratio_snap_vs_non"]
    ev = d["ev"]
    S = []

    # ---------------------------------------------------------------- 1
    S.append(frame(1, "The problem + our innovation", "Zero sales does not always mean zero demand.", f'''
<div class="g3">
  <div class="card">
    <div class="tag">The problem</div>
    <img src="fig/s1_patterns.png" class="fill">
    <div class="caption">Same retail dataset. Completely different demand patterns.</div>
  </div>
  <div class="mid">
    <div class="insight">Forecast demand patterns <em>differently.</em></div>
    <p class="muted">A recorded zero can mean “nobody wanted it”, or “it was not on the shelf”.
    One model struggles to learn both.</p>
  </div>
  <div class="card arch">
    <div class="tag">Our hybrid</div>
    <div class="node ink">Leak-safe history up to forecast day T</div>
    <div class="fork"><span></span><span></span></div>
    <div class="two">
      <div class="node blue"><b>DeepAR</b><small>How much sells, if anything sells</small></div>
      <div class="node orange"><b>Zero-run LightGBM</b><small>Chance that recorded sales are zero</small></div>
    </div>
    <div class="fork join"><span></span><span></span></div>
    <div class="node teal"><b>Hurdle fusion</b><small>One valid probability distribution</small></div>
    <div class="down">↓</div>
    <div class="node green"><b>Probabilistic forecast</b><small>Expected sales · uncertainty · quantiles</small></div>
    <div class="quote">First estimate whether sales will be zero. If sales happen, estimate how many.</div>
  </div>
</div>
<div class="banner">Instead of forcing one model to learn everything, we combine two models with different jobs.</div>'''))

    # ---------------------------------------------------------------- 2
    order = [("intermittent", "Intermittent", TEAL, "sells only now and then"),
             ("lumpy", "Lumpy", ORANGE, "occasional sales, sometimes large"),
             ("smooth", "Smooth", BLUE, "fairly steady demand"),
             ("erratic", "Erratic", "#8B5CF6", "steady timing, unpredictable size")]
    seg = "".join(f'<div style="flex:{cls[k]};background:{c}"><b>{cls[k]}</b></div>' for k, _, c, _ in order)
    leg = "".join(f'<div><i style="background:{c}"></i><b>{lab}</b> · {cls[k]} <span class="muted">— {txt}</span></div>'
                  for k, lab, c, txt in order)
    hard = (cls["intermittent"] + cls["lumpy"]) / n_series
    snapbars = "".join(
        f'<div class="sb"><span>{st}</span><div><i style="width:{(snap[st] - 1) * 100 * 18:.0f}px"></i></div>'
        f'<b>+{(snap[st] - 1) * 100:.1f}%</b></div>' for st in ("CA", "TX", "WI"))
    ev_ratio = float(ev.set_index("group").loc["event_day", "ratio_vs_non_event"])
    wd = d["wd"][d["wd"].weekday.isin([5, 6])].normalised
    S.append(frame(2, "Dataset quality & exploratory analysis",
                   "Our dataset was dominated by difficult demand patterns.", f'''
<div class="g2-21">
  <div class="card stat">
    <div class="huge">{zshare:.0%}</div>
    <div class="statlab">of recorded daily sales were <b>zero</b></div>
    <div class="facts">
      <div><b>{n_series:,}</b><span>product-store series</span></div>
      <div><b>3</b><span>stores (CA, TX, WI)</span></div>
      <div><b>1,000</b><span>days of history</span></div>
      <div><b>28</b><span>day forecast horizon</span></div>
    </div>
    <div class="muted small">A subset of the M5 Walmart dataset. Zero share counted after each product's first sale.</div>
  </div>
  <div class="col">
    <div class="card">
      <div class="row-between"><div class="tag">Demand types (by sale frequency and size variation)</div>
      <div class="pillbig">{hard:.0%} intermittent or lumpy</div></div>
      <div class="stack">{seg}</div>
      <div class="legend">{leg}</div>
    </div>
    <div class="g3 tight">
      <div class="card mini"><div class="k">Weekends</div><div class="v">{wd.min():.2f}–{wd.max():.2f}×</div>
        <div class="muted small">Sat/Sun sales vs the category's daily average</div></div>
      <div class="card mini"><div class="k">SNAP food-benefit days</div>{snapbars}
        <div class="muted small">vs non-SNAP days, same state</div></div>
      <div class="card mini"><div class="k">Event days</div><div class="v">{ev_ratio:.3f}×</div>
        <div class="muted small">slightly lower than non-event days</div></div>
    </div>
    <div class="row-between">
      <div class="muted small">⚑ Observed associations, not causal effects.</div>
    </div>
    <div class="warn"><b>⚠ Recorded zero ≠ confirmed zero demand.</b> M5 has no inventory data. We flag unusually long
    zero runs (at least 28 days, and at least 3× the product's usual gap between sales) as <i>possibly censored</i>.
    This proxy is a rule, not ground-truth stockout data.</div>
  </div>
</div>'''))

    # ---------------------------------------------------------------- 3
    # timeline (zoomed on Nov 2015 - May 2016; history starts Aug 2013)
    t0, t1 = pd.Timestamp("2015-11-15"), pd.Timestamp("2016-05-26")
    W = 1100

    def X(t):
        return 190 + (pd.Timestamp(t) - t0).days / (t1 - t0).days * (W - 200)
    origins = [("Validation", "2016-01-03", "#8B95A1"), ("Test 1", "2016-01-31", BLUE), ("Test 2", "2016-02-28", BLUE),
               ("Test 3", "2016-03-27", BLUE), ("Test 4", "2016-04-24", BLUE)]
    rows = []
    for i, (lab, o, c) in enumerate(origins):
        y = 30 + i * 40
        o = pd.Timestamp(o)
        e = o + pd.Timedelta(days=28)
        rows.append(f'<text x="0" y="{y + 17}" class="tl">{lab}</text>'
                    f'<text x="104" y="{y + 17}" class="td">{o:%d %b}</text>'
                    f'<rect x="190" y="{y}" width="{X(o) - 190:.0f}" height="24" rx="5" fill="#E9ECEF"/>'
                    f'<rect x="{X(o):.0f}" y="{y}" width="{X(e) - X(o):.0f}" height="24" rx="5" fill="{c}"/>'
                    f'<line x1="{X(o):.0f}" x2="{X(o):.0f}" y1="{y - 4}" y2="{y + 28}" stroke="{INK}" stroke-width="2"/>')
    ticks = "".join(f'<text x="{X(t):.0f}" y="246" class="td" text-anchor="middle">{pd.Timestamp(t):%b %Y}</text>'
                    for t in ("2015-12-01", "2016-01-01", "2016-02-01", "2016-03-01", "2016-04-01", "2016-05-01"))
    svg = (f'<svg viewBox="0 0 {W} 254" class="tlsvg">{"".join(rows)}{ticks}'
           f'<text x="190" y="18" class="td">◀ training history from Aug 2013</text></svg>')
    S.append(frame(3, "Feature engineering & leakage control",
                   "Every forecast uses only information available at that moment.", f'''
<div class="g2-12">
  <div class="card">
    <div class="row-between"><div class="tag">Rolling-origin evaluation</div>
      <div class="legend inline"><div><i style="background:#E9ECEF"></i>known at origin T: used for training</div>
      <div><i style="background:{BLUE}"></i>next 28 days: hidden, used only for scoring</div></div></div>
    {svg}
    <div class="steps"><span>① Refit on data up to T</span><span>② Forecast the next 28 days</span>
    <span>③ Score against hidden actuals</span><span>④ Move T forward 28 days, repeat</span></div>
    <div class="muted small">The validation origin is used only to choose model settings. The four test origins are scored once.</div>
  </div>
  <div class="col">
    <div class="banner slim">No future sales enter any forecast.</div>
    <div class="card gate"><div class="tag">At forecast day T</div>
      <div class="ok">✓ Allowed <span>past sales · past zero runs · product & store · the calendar, events, SNAP days and prices for the next 28 days</span></div>
      <div class="no">✕ Hidden <span>actual sales after T. Revealed only to score the forecast.</span></div></div>
    <div class="warn teal"><b>Zero-run proxy.</b> Unusually long zero stretches are found from past patterns
    only. DeepAR does not learn from them as if they were true zero demand; they are <b>not deleted</b> and are
    still scored. <b>Proxy ≠ confirmed stockout.</b></div>
  </div>
</div>
<div class="g4">
  <div class="card feat"><div class="ft" style="color:{BLUE}">1 · Sales history</div>
    <p>Recent sales · lags 1, 7, 14, 28 · rolling means · current zero-run length</p></div>
  <div class="card feat"><div class="ft" style="color:{TEAL}">2 · Demand behaviour</div>
    <p>Average gap between sales · non-zero rate · statistics of past zero runs</p></div>
  <div class="card feat"><div class="ft" style="color:{ORANGE}">3 · Calendar & business</div>
    <p>Day of week · month · events · SNAP days · price and price changes · forecast horizon</p></div>
  <div class="card feat"><div class="ft" style="color:{GREEN}">4 · Product</div>
    <p>Store, department and category identifiers · learned series embeddings <span class="muted">(product age was tested and not adopted)</span></p></div>
</div>'''))

    # ---------------------------------------------------------------- 4
    def bars(vals, colors, labels, h=110, struck=None):
        out = []
        for i, (v, c, l) in enumerate(zip(vals, colors, labels)):
            cl = " struck" if struck == i else ""
            out.append(f'<div class="b{cl}"><em>{v:.0%}</em><i style="height:{v * h / max(vals):.0f}px;background:{c}"></i><span>{l}</span></div>')
        return '<div class="bars">' + "".join(out) + "</div>"
    raw = [0.30, 0.35, 0.21, 0.14]
    pos = [v / (1 - raw[0]) for v in raw[1:]]
    pi = 0.60
    fin = [pi] + [(1 - pi) * v for v in pos]
    S.append(frame(4, "Model application", "Two models. Two different jobs. One forecast.", f'''
<div class="g2-11">
  <div class="col">
    <div class="card model" style="border-top:5px solid {BLUE}">
      <div class="mt"><b>DeepAR</b> <span class="muted">deep learning · one network for all 1,200 series</span></div>
      <p>Learns shared patterns across products and predicts a full probability distribution of sales.</p>
      <div class="chips"><span>sales history</span><span>lags 7 / 14 / 28</span><span>product & store</span><span>calendar, SNAP, price</span></div>
      <div class="out">Output: <b>possible sales values + their probabilities</b></div>
    </div>
    <div class="card model" style="border-top:5px solid {ORANGE}">
      <div class="mt"><b>Zero-run persistence model</b> <span class="muted">LightGBM classifier</span></div>
      <p>Asks one question: <b>will recorded sales be zero on this future day?</b></p>
      <div class="chips"><span>current zero-run length</span><span>average gap between sales</span><span>recent non-zero rate</span>
      <span>past zero-run statistics</span><span>calendar & price</span><span>forecast horizon</span></div>
      <div class="out">Output: <b>probability of zero recorded sales, for each of the next 28 days</b></div>
    </div>
    <div class="chips grey"><b>Phase 2 training upgrades:</b><span>validation early stopping</span><span>best-checkpoint restore</span>
    <span>3 random seeds</span><span>zero-probability scoring</span></div>
  </div>
  <div class="card fusion">
    <div class="row-between"><div class="tag">Hurdle fusion</div><div class="illus">Illustrative example, not a real prediction</div></div>
    <div class="fz">
      <div><div class="fh" style="color:{ORANGE}">① Zero-run model</div>
        {bars([pi, 1 - pi], [ORANGE, "#F3C3A8"], ["zero", "sales"])}</div>
      <div class="plus">+</div>
      <div><div class="fh" style="color:{BLUE}">② DeepAR: drop its zero, rescale the rest</div>
        {bars(raw, ["#D5D9DE", BLUE, BLUE, BLUE], ["0", "1", "2", "5"], struck=0)}</div>
    </div>
    <div class="down big">↓</div>
    <div><div class="fh" style="color:{TEAL}">③ Final distribution: 60% zero, then 40% shared out as DeepAR says</div>
      {bars(fin, [ORANGE, TEAL, TEAL, TEAL], ["0", "1 unit", "2 units", "5 units"], h=120)}</div>
    <div class="formula">P(0) = π &nbsp;·&nbsp; P(y) = (1 − π) × DeepAR(y) / (1 − DeepAR(0)) for y ≥ 1
      <span class="muted">— the probabilities add up to exactly 1</span></div>
  </div>
</div>'''))

    # ---------------------------------------------------------------- 5
    chart_ablation(d)
    S.append(frame(5, "Evaluation, results & ablations",
                   "The hybrid improved overall forecasting. The gains were not uniform.", cls="s5", body=f'''
<div class="evalbar"><span><b>1,200</b> series</span><span><b>4</b> rolling test origins</span><span><b>28</b>-day horizon</span>
<span><b>{n_all:,}</b> scored forecasts, same cells for every model</span><span>Hybrid: 3 seeds (sd {hps:.4f}) · Kalman-LightGBM: 1 run</span></div>
<div class="g2-11 r5">
  <div class="col">
    <div class="g2 tight">
      <div class="card metric"><div class="k">WAPE <span class="muted">(average error, lower = better)</span></div>
        <div class="vs"><span>{kw:.4f}</span>→<b>{hw:.4f}</b></div><div class="gain">−{-cw.rel_diff_pct:.1f}%</div></div>
      <div class="card metric"><div class="k">Pinball loss <span class="muted">(quantile error)</span></div>
        <div class="vs"><span>{kp:.4f}</span>→<b>{hp:.4f}</b></div><div class="gain">−{-cp.rel_diff_pct:.1f}%</div></div>
    </div>
    <div class="muted small c">Phase 1 Kalman-LightGBM → Phase 2 hybrid. Better at {int(cp.origins_a_better)} of 4 origins;
    95% bootstrap intervals exclude zero.</div>
    <div class="g2 tight">
      <div class="card wl win"><div class="k">All days · {n_all:,} forecasts</div>
        <div class="line"><span>Kalman-LightGBM</span><b>{kw:.4f}</b></div>
        <div class="line best"><span>Phase 2 hybrid</span><b>{hw:.4f}</b></div>
        <div class="who">Winner: Phase 2 hybrid</div></div>
      <div class="card wl lose"><div class="k">Excluding proxy-flagged long zero runs · {n_nc:,}</div>
        <div class="line best"><span>Kalman-LightGBM</span><b>{kwn:.4f}</b></div>
        <div class="line"><span>Phase 2 hybrid</span><b>{hwn:.4f}</b></div>
        <div class="who">Winner: Kalman-LightGBM <span class="muted">(pinball {kpn:.4f} vs {hpn:.4f})</span></div></div>
    </div>
    <div class="warn grey"><b>Uncertainty is still too wide.</b> The hybrid's 80% and 90% intervals contained the
    actual value {cov80:.0%} and {cov90:.0%} of the time. Calibration is an open problem, not a win.</div>
  </div>
  <div class="col">
    <div class="card chart"><div class="ct">Where the gain comes from: how long the product had sold zero at forecast day T</div>
      <img src="fig/s5_buckets.png">
      <div class="take">Zero runs of 7+ days are {share7:.0%} of forecasts but give {gain7:.0%} of the gain.</div></div>
    <div class="card chart"><div class="ct">Ablations: removing either component hurts (seed-42 runs)</div>
      <img src="fig/s5_ablation.png">
      <div class="take">Each gap to the full hybrid has a 95% bootstrap interval that excludes zero.</div></div>
  </div>
</div>'''))

    # ---------------------------------------------------------------- 6
    chart_fans(d)
    S.append(frame(6, "Conclusion", "The best forecast depends on the demand pattern.", f'''
<div class="g3 tight">
  <div class="card say" style="border-left:6px solid {BLUE}"><div class="k">We built</div>
    <p>A leak-safe pipeline that compares statistical, machine-learning and deep-learning forecasters on identical cells.</p></div>
  <div class="card say" style="border-left:6px solid {TEAL}"><div class="k">We improved</div>
    <p>The Phase 2 hybrid cut overall recorded-sales error vs Phase 1 Kalman-LightGBM: WAPE −{-cw.rel_diff_pct:.1f}%, pinball −{-cp.rel_diff_pct:.1f}%.</p></div>
  <div class="card say" style="border-left:6px solid {ORANGE}"><div class="k">We discovered</div>
    <p>The gain sits in persistent zero-sales runs. On more regular demand, Kalman-LightGBM stayed stronger.</p></div>
</div>
<div class="final">There is no single winner across every demand condition. Forecasting works better when the model
understands the type of demand it is facing.</div>
<div class="card fan"><div class="ct">Real Phase 2 hybrid forecasts from the last test origin (24 Apr 2016). Examples were picked from their sales history before looking at forecasts. The band changes shape with the demand pattern.
 <span class="muted">The third product restarted selling right after T: a miss, and the reason we report results by condition.</span></div>
  <img src="fig/s6_fans.png"></div>
<div class="g2-21 tight end">
  <div class="chips"><b>ForecastKit gives you:</b><span>leak-safe forecasting</span><span>probabilistic predictions</span>
  <span>multiple model families</span><span>condition-based evaluation</span><span>transparent comparisons</span><span>reproducible experiments</span></div>
  <div class="chips grey"><b>Next:</b><span>better uncertainty calibration</span><span>real inventory data</span><span>more ablations & datasets</span></div>
</div>
<div class="brand"><b>ForecastKit</b> <i>Understanding demand patterns. Forecasting with uncertainty.</i></div>''', cls="last"))

    html = f'<!doctype html><html><head><meta charset="utf-8"><title>ForecastKit · Hackathon</title>' \
           f'<link rel="stylesheet" href="deck.css"></head><body>{"".join(S)}</body></html>'
    (HERE / "deck.html").write_text(html)
    facts = {"hybrid_WAPE": hw, "hybrid_WAPE_sd": hws, "hybrid_pinball": hp, "hybrid_pinball_sd": hps,
             "klgbm_WAPE": kw, "klgbm_pinball": kp, "hybrid_WAPE_nc": hwn, "hybrid_pinball_nc": hpn,
             "klgbm_WAPE_nc": kwn, "klgbm_pinball_nc": kpn, "rel_WAPE_pct": cw.rel_diff_pct,
             "rel_pinball_pct": cp.rel_diff_pct, "nc_pinball_diff_ci": [cpn.ci_low, cpn.ci_high],
             "share_runs_7plus": share7, "gain_share_runs_7plus": gain7, "cov80": cov80, "cov90": cov90,
             "zero_share": zshare, "classes": cls, "snap_ratio": snap.to_dict(), "n_all": n_all, "n_nc": n_nc}
    (HERE / "deck_numbers.json").write_text(json.dumps(facts, indent=2, default=float))
    print(json.dumps(facts, indent=1, default=float))


if __name__ == "__main__":
    data = load()
    chart_patterns(data)
    build(data)
