"""All report figures. Every function takes an Evaluator (or its tables) and writes a PNG."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D", "#BAB0AC", "#EECA3B"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.grid": True, "grid.alpha": .25})


def colors(models):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def error_vs_horizon(ev, path, metrics=("WAPE", "pinball")):
    t = ev.error_by_horizon()
    col = colors(ev.models)
    fig, axs = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.6))
    axs = np.atleast_1d(axs)
    for ax, mt in zip(axs, metrics):
        for m in ev.models:
            d = t[t.model == m]
            ax.plot(d.h, d[mt], label=m, color=col[m], lw=1.6)
        ax.set_xlabel("forecast horizon (days)"); ax.set_ylabel(mt); ax.set_title(f"{mt} vs horizon")
    axs[0].legend(frameon=False, fontsize=8)
    return _save(fig, path)


def calibration_plot(ev, path):
    cal = ev.calibration()
    col = colors(ev.models)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1, 1.15]})
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    for m in ev.models:
        d = cal[cal.model == m]
        mid = (d.empirical + d.empirical_strict) / 2
        ax.plot(d.nominal, mid, "o-", ms=3, color=col[m], label=m, lw=1.2)
    ax.set_xlabel("nominal quantile level"); ax.set_ylabel("empirical P(y <= q-forecast) (mid of < and <=)")
    ax.set_title("Quantile calibration\n(count data: zeros blur the low quantiles)", fontsize=9)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    t = ev.main_table()
    x = np.arange(len(ev.models))
    for k, (c, lab) in enumerate((("cov80", "80% interval"), ("cov90", "90% interval"))):
        bx.bar(x + (k - .5) * .38, t[c].reindex(ev.models), .38, label=lab, color=["#9ecae1", "#3182bd"][k])
        bx.axhline([0.8, 0.9][k], color=["#9ecae1", "#3182bd"][k], ls="--", lw=1)
    bx.set_xticks(x); bx.set_xticklabels(ev.models, rotation=30, ha="right"); bx.set_ylim(0.6, 1.0)
    bx.set_ylabel("empirical coverage (dashed = target)"); bx.set_title("Central-interval coverage vs target", fontsize=9)
    bx.legend(frameon=False, fontsize=7)
    return _save(fig, path)


def pinball_by_quantile(ev, path):
    d = ev.pinball_by_quantile()
    col = colors(ev.models)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in ev.models:
        x = d[d.model == m]
        ax.plot(x.q, x.pinball, "o-", ms=3.5, color=col[m], label=m)
    ax.set_xlabel("quantile level"); ax.set_ylabel("pinball loss"); ax.set_title("Pinball loss by quantile level")
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, path)


def origin_stability(ev, path, metric="WAPE"):
    po = ev.per_origin()
    col = colors(ev.models)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    data = [po[po.model == m][metric].to_numpy() for m in ev.models]
    try:
        ax.boxplot(data, tick_labels=ev.models, patch_artist=True, boxprops={"alpha": .35})
    except TypeError:  # matplotlib < 3.9
        ax.boxplot(data, labels=ev.models, patch_artist=True, boxprops={"alpha": .35})
    for i, m in enumerate(ev.models):
        ax.scatter(np.full(len(data[i]), i + 1), data[i], color=col[m], zorder=3, s=18)
    ax.set_ylabel(metric); ax.set_title(f"{metric} across rolling origins")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, path)


def slice_heatmap(ev, path, metric="WAPE"):
    t = ev.by_slice()
    piv = t.pivot(index="slice", columns="model", values=metric)[ev.models].dropna(how="all")
    if piv.empty:
        return None
    fig, ax = plt.subplots(figsize=(1.0 + 0.95 * len(ev.models), 0.35 * len(piv) + 1.6))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.grid(False)
    for i in range(piv.shape[0]):
        best = np.nanargmin(piv.values[i])
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if j == best else None)
    ax.set_title(f"{metric} by slice (bold = best model in row)")
    fig.colorbar(im, ax=ax, fraction=.03)
    return _save(fig, path)


def runtime_bar(ev, path):
    rt = ev.runtime()
    if not len(rt):
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rt = rt.reindex([m for m in ev.models if m in rt.index])
    ax.barh(rt.index, rt.total_s, color=[colors(ev.models)[m] for m in rt.index])
    ax.set_xscale("log"); ax.set_xlabel("total train + predict seconds (log)"); ax.set_title("Compute cost")
    return _save(fig, path)


def cold_start_bar(ev, path, metric="WAPE"):
    cs = ev.cold_start()
    if not len(cs):
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    est = ev.main_table()[metric]
    x = np.arange(len(ev.models))
    ax.bar(x - .18, est.reindex(ev.models), .36, label="established series", color="#BAB0AC")
    ax.bar(x + .18, cs[metric].reindex(ev.models), .36, label=f"cold-start (last {ev.ds.meta['cold_days']}d only)",
           color="#E45756")
    ax.set_xticks(x); ax.set_xticklabels(ev.models, rotation=25, ha="right"); ax.set_ylabel(metric)
    ax.set_title("Cold start vs established"); ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


# ------------------------------------------------------------------------------ fan charts
def pick_series(ev) -> dict:
    """Pick representative series automatically: smooth, intermittent, lumpy, cold-start, stockout, event."""
    ds, picks = ev.ds, {}
    st, _ids = ds.static, ds.series_ids
    vol = np.nan_to_num(ev.info["mean_demand"].to_numpy())
    est = ~ev.cold
    for cls in ("smooth", "intermittent", "lumpy", "erratic"):
        sel = np.where((st["intermittency"].to_numpy() == cls) & est)[0]
        if len(sel):
            picks[cls] = int(sel[np.argsort(vol[sel])[len(sel) // 2]])
    if ev.cold.any():
        sel = np.where(ev.cold)[0]
        picks["cold-start"] = int(sel[np.argsort(vol[sel])[len(sel) // 2]])
    if ev.C.any():
        cnt = ev.C.sum((0, 2)) * est * (vol >= max(1.0, np.median(vol[est])))
        if cnt.max() > 0:
            picks["stockout (window)"] = int(np.argmax(cnt))
    if "event_any" in ev.day:
        cnt = (ev.day["event_any"] & ev.A).sum((0, 2))
        sel = np.where(est & (vol > np.median(vol)))[0]
        if len(sel):
            picks["event days"] = int(sel[0])
    out, seen = {}, set()
    for k, v in picks.items():
        if v not in seen:
            out[k] = v
            seen.add(v)
    return out


def fan_charts(ev, path, models=None, origin_i=-1, history=84):
    pref = [m for m in ("snaive4", "snaive", "kalman", "lgbm", "kalman_lgbm", "deepar") if m in ev.models]
    models = models or (pref[:1] + [m for m in pref if m in ("kalman", "lgbm", "deepar")][:3] or ev.models[:4])
    picks = pick_series(ev)
    if not picks:
        return None
    ds = ev.ds
    oi = ev.oi[origin_i]
    qs = np.array(ev.qs)
    j = lambda q: int(np.argmin(np.abs(qs - q)))
    fig, axs = plt.subplots(len(picks), len(models), figsize=(3.3 * len(models), 1.9 * len(picks)), sharex="row",
                            squeeze=False)
    col = colors(ev.models)
    for r, (label, s) in enumerate(picks.items()):
        lo = max(oi - history, 0)
        hist_x = ds.dates[lo:oi + 1]
        fut_x = ds.dates[oi + 1:oi + 1 + ev.H]
        for c, m in enumerate(models):
            ax = axs[r, c]
            if ev.cold[s]:   # models only saw the last `cold_days` days: fade the rest
                k = ds.meta.get("cold_days", 28)
                ax.plot(hist_x[:-k], ds.Y[s, lo:oi + 1][:-k], color="#555", lw=.8, alpha=.25)
                ax.plot(hist_x[-k:], ds.Y[s, lo:oi + 1][-k:], color="#555", lw=.9)
            else:
                ax.plot(hist_x, ds.Y[s, lo:oi + 1], color="#555", lw=.9)
            ax.plot(fut_x, ev.Y[origin_i, s], color="k", lw=1.2, marker=".", ms=3)
            Q = ev.Q[m][origin_i, s]
            for (a, b, al) in ((0.05, 0.95, .15), (0.1, 0.9, .22), (0.25, 0.75, .35)):
                if np.isclose(qs, a).any() and np.isclose(qs, b).any():
                    ax.fill_between(fut_x, Q[:, j(a)], Q[:, j(b)], color=col[m], alpha=al, lw=0)
            ax.plot(fut_x, Q[:, j(0.5)], color=col[m], lw=1.6)
            cm = ev.C[origin_i, s]
            if cm.any():
                ax.scatter(fut_x[cm], np.zeros(cm.sum()), marker="x", color="red", s=14, zorder=5)
            if r == 0:
                ax.set_title(m, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{label}\n{ds.series_ids[s][:18]}", fontsize=7)
            ax.tick_params(axis="x", labelrotation=30, labelsize=6)
    fig.suptitle(f"Forecast fans at origin {ds.dates[oi].date()} (median, 50/80/90% bands; black = actual; red x = stockout-flagged)",
                 fontsize=9, y=1.0)
    return _save(fig, path)


def kalman_state(run_dir, path, model="kalman"):
    p = Path(run_dir) / f"diag_{model}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        d = pickle.load(f)
    fig, axs = plt.subplots(len(d), 2, figsize=(9, 2.3 * len(d)), squeeze=False)
    for r, (sid, g) in enumerate(d.items()):
        ax = axs[r, 0]
        ax.plot(g["dates"], g["y"], ".", ms=2.5, color="#999", label=f"y ({g['transform']} scale)")
        ax.plot(g["dates"], g["level"], color="#4C78A8", lw=1.6, label="filtered level")
        ax.set_title(f"{sid}: level (q_level={g['q_level']:.0e}, q_seas={g['q_seas']:.0e})", fontsize=8)
        ax.legend(frameon=False, fontsize=6)
        ax = axs[r, 1]
        ax.plot(g["dates"], g["innovations"], lw=.6, color="#E45756")
        ax.axhline(0, color="k", lw=.5); ax.axhline(2, color="k", lw=.5, ls=":"); ax.axhline(-2, color="k", lw=.5, ls=":")
        ax.set_title("standardized one-step innovations (should look like white noise)", fontsize=8)
        for a_ in (axs[r, 0], axs[r, 1]):
            a_.tick_params(axis="x", labelrotation=20, labelsize=7)
    return _save(fig, path)


def make_all_figures(ev, run_dir, fig_dir=None, fan_models=None) -> dict:
    fig_dir = Path(fig_dir or Path(run_dir) / "figures")
    out = {
        "fan_charts": fan_charts(ev, fig_dir / "fan_charts.png", fan_models),
        "error_vs_horizon": error_vs_horizon(ev, fig_dir / "error_vs_horizon.png"),
        "calibration": calibration_plot(ev, fig_dir / "calibration.png"),
        "pinball_by_quantile": pinball_by_quantile(ev, fig_dir / "pinball_by_quantile.png"),
        "origin_stability": origin_stability(ev, fig_dir / "origin_stability.png"),
        "slice_heatmap": slice_heatmap(ev, fig_dir / "slice_heatmap.png"),
        "runtime": runtime_bar(ev, fig_dir / "runtime.png"),
        "cold_start": cold_start_bar(ev, fig_dir / "cold_start.png"),
        "kalman_state": kalman_state(run_dir, fig_dir / "kalman_state.png"),
    }
    return {k: v for k, v in out.items() if v is not None}
