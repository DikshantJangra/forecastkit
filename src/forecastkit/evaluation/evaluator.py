"""Reads ONLY the long forecast file (+ the dataset truth) and builds every table the report needs.
Works for any model - including forecasts produced outside this library."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecastkit.backtest.runner import prepare
from forecastkit.masks import stockout_mask
from forecastkit.utils import log

from .ensemble import add_ensembles


class Errors:
    """Element-wise errors of one model, shape [O, N, H] (and [..., Q] for per-quantile ones)."""

    def __init__(self, Q, M, Y, qs):
        qs = np.asarray(qs)
        self.qs = qs
        med = Q[..., int(np.argmin(np.abs(qs - 0.5)))]
        self.med, self.mean = med, M
        self.AE = np.abs(Y - med)
        self.SE = (Y - M) ** 2
        d = Y[..., None] - Q
        self.PBq = np.maximum(qs * d, (qs - 1) * d)           # [O,N,H,Q]
        self.PB = self.PBq.mean(-1)
        self.hit = (Y[..., None] <= Q)                         # calibration: P(y <= q_hat)
        self.hit_lt = (Y[..., None] < Q)                       # strict: for DISCRETE y the ideal lies in [P(y<q), P(y<=q)]
        self.Q = Q
        self.intervals = {}
        for cov in (0.8, 0.9):
            lo, hi = (1 - cov) / 2, 1 - (1 - cov) / 2
            if np.isclose(qs, lo).any() and np.isclose(qs, hi).any():
                jl, jh = int(np.argmin(np.abs(qs - lo))), int(np.argmin(np.abs(qs - hi)))
                self.intervals[cov] = ((Y >= Q[..., jl]) & (Y <= Q[..., jh]), Q[..., jh] - Q[..., jl])


class Evaluator:
    def __init__(self, cfg: dict, run_dir, forecasts: pd.DataFrame | None = None, phase: str = "test"):
        self.cfg, self.run_dir = cfg, Path(run_dir)
        ev = cfg.get("evaluation", {})
        self.ds, self.orig, self.info = prepare(cfg)
        fpath = self.run_dir / ("forecasts.parquet" if phase == "test" else f"forecasts_{phase}.parquet")
        self.fc = forecasts if forecasts is not None else pd.read_parquet(fpath)
        for c in ("model", "stat"):
            self.fc[c] = self.fc[c].astype(str)
        self.fc = add_ensembles(self.fc, cfg.get("ensembles"))
        self.models = list(dict.fromkeys(self.fc["model"]))
        self.origins = sorted(pd.to_datetime(self.fc["origin"]).unique())
        self.H = int(self.fc["h"].max())
        self.qs = sorted(float(s[1:]) for s in self.fc["stat"].unique() if s.startswith("q"))
        self.ev = ev
        self._build_truth()
        self._build_cubes()
        self.err = {m: Errors(self.Q[m], self.M[m], self.Y, self.qs) for m in self.models}
        self.cold = self.ds.static["cold_start"].to_numpy()
        self.groups = self._series_groups()

    # ------------------------------------------------------------------ truth + cubes
    def _build_truth(self):
        ds = self.ds
        O, _N, H = len(self.origins), ds.N, self.H
        oi = [int(ds.dates.get_loc(pd.Timestamp(o))) for o in self.origins]
        idx = np.array([[o + 1 + h for h in range(H)] for o in oi])             # [O,H] column indices
        eval_cfg = self.cfg.get("masks", {}).get("eval_stockout") or self.cfg.get("masks", {}).get("stockout")
        cens_full = stockout_mask(ds.Y, ds.avail, eval_cfg)                      # hindsight labels (evaluation only)
        self.Y = np.stack([ds.Y[:, idx[o]] for o in range(O)]).astype(np.float64)
        self.A = np.stack([ds.avail[:, idx[o]] for o in range(O)])
        self.C = np.stack([cens_full[:, idx[o]] for o in range(O)])
        self.day = {}
        for name in ds.meta.get("day_slice_cols", []) or []:
            if name in ds.known_names:
                j = ds.k_index(name)
                self.day[name] = np.stack([ds.K[:, idx[o], j] > 0.5 for o in range(O)])
        self.oi = oi

    def _build_cubes(self):
        ds, O, N, H = self.ds, len(self.origins), self.ds.N, self.H
        row = pd.Series(np.arange(N), index=ds.series_ids)
        orow = {pd.Timestamp(o): i for i, o in enumerate(self.origins)}
        self.Q, self.M = {}, {}
        for m in self.models:
            sub = self.fc[self.fc.model == m]
            r = row.reindex(sub.series_id.to_numpy()).to_numpy()
            o = sub.origin.map(lambda x: orow[pd.Timestamp(x)]).to_numpy()
            h = sub.h.to_numpy().astype(int) - 1
            Q = np.full((O, N, H, len(self.qs)), np.nan)
            M = np.full((O, N, H), np.nan)
            for stat, g in sub.groupby("stat"):
                sel = sub.stat.to_numpy() == stat
                if stat == "mean":
                    M[o[sel], r[sel], h[sel]] = sub.value.to_numpy()[sel]
                else:
                    j = int(np.argmin(np.abs(np.array(self.qs) - float(stat[1:]))))
                    Q[o[sel], r[sel], h[sel], j] = sub.value.to_numpy()[sel]
            if np.isnan(Q).any():
                raise ValueError(f"model {m}: forecasts missing for some (origin, series, horizon, quantile) cells")
            self.Q[m], self.M[m] = Q, np.where(np.isnan(M), Q[..., int(np.argmin(np.abs(np.array(self.qs) - .5)))], M)

    def _series_groups(self) -> dict:
        st = self.ds.static
        cols = self.ev.get("slice_cols") or (["intermittency", "volume_tier"] +
                                              [c for c in (self.ds.meta.get("static_cat_cols") or [])
                                               if st[c].nunique() <= 12])
        groups = {}
        for c in dict.fromkeys(cols):
            if c in st.columns:
                for lvl in sorted(st[c].astype(str).unique()):
                    groups[f"{c}={lvl}"] = (st[c].astype(str) == lvl).to_numpy()
        return groups

    # ------------------------------------------------------------------ aggregation
    def valid(self, series_mask=None, mode="all", h_sel=None, day=None, origin_idx=None) -> np.ndarray:
        v = self.A.copy()
        if mode == "non_censored":
            v &= ~self.C
        if series_mask is not None:
            v &= series_mask[None, :, None]
        if h_sel is not None:
            hm = np.zeros(self.H, dtype=bool)
            hm[[h - 1 for h in h_sel]] = True
            v &= hm[None, None, :]
        if day is not None:
            v &= self.day[day]
        if origin_idx is not None:
            om = np.zeros(len(self.origins), dtype=bool)
            om[origin_idx] = True
            v &= om[:, None, None]
        return v

    def agg(self, model: str, v: np.ndarray) -> dict:
        e, Y = self.err[model], self.Y
        n = int(v.sum())
        if n == 0:
            return {k: np.nan for k in ("n", "MAE", "RMSE", "WAPE", "pinball", "WQL", "cov80", "width80", "cov90",
                                        "width90")}
        sy = Y[v].sum()
        out = {"n": n, "MAE": e.AE[v].mean(), "RMSE": np.sqrt(e.SE[v].mean()),
               "WAPE": e.AE[v].sum() / sy if sy > 0 else np.nan, "pinball": e.PB[v].mean(),
               "WQL": (2 * e.PBq[v].sum(0) / sy).mean() if sy > 0 else np.nan}
        for cov in (0.8, 0.9):
            if cov in e.intervals:
                hit, w = e.intervals[cov]
                out[f"cov{int(cov * 100)}"], out[f"width{int(cov * 100)}"] = hit[v].mean(), w[v].mean()
            else:
                out[f"cov{int(cov * 100)}"], out[f"width{int(cov * 100)}"] = np.nan, np.nan
        return out

    def table(self, v: np.ndarray, extra: dict | None = None) -> pd.DataFrame:
        rows = [{"model": m, **(extra or {}), **self.agg(m, v)} for m in self.models]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ report tables
    def main_table(self, mode="all") -> pd.DataFrame:
        return self.table(self.valid(~self.cold, mode)).set_index("model")

    def by_horizon(self, hs=(1, 7, 14, 28), mode="all") -> pd.DataFrame:
        rows = []
        hs = sorted({h for h in hs if h <= self.H} | {self.H})
        for h in list(hs) + [f"1-{self.H}"]:
            sel = [h] if isinstance(h, int) else None
            t = self.table(self.valid(~self.cold, mode, h_sel=sel), {"h": h})
            rows.append(t)
        return pd.concat(rows, ignore_index=True)

    def error_by_horizon(self, mode="all") -> pd.DataFrame:
        rows = []
        for h in range(1, self.H + 1):
            rows.append(self.table(self.valid(~self.cold, mode, h_sel=[h]), {"h": h}))
        return pd.concat(rows, ignore_index=True)

    def by_slice(self, mode="all") -> pd.DataFrame:
        rows = []
        for g, sm in self.groups.items():
            if g.startswith("cold_start"):
                continue
            rows.append(self.table(self.valid(sm & ~self.cold, mode), {"slice": g}))
        for d in self.day:
            rows.append(self.table(self.valid(~self.cold, mode, day=d), {"slice": f"{d}=1"}))
        return pd.concat(rows, ignore_index=True)

    def cold_start(self, mode="all") -> pd.DataFrame:
        if not self.cold.any():
            return pd.DataFrame()
        return self.table(self.valid(self.cold, mode)).set_index("model")

    def censoring_effect(self) -> pd.DataFrame:
        rows = []
        for mode in ("all", "non_censored"):
            rows.append(self.table(self.valid(~self.cold, mode), {"eval_days": mode}))
        return pd.concat(rows, ignore_index=True)

    def per_origin(self) -> pd.DataFrame:
        rows = []
        for i, o in enumerate(self.origins):
            rows.append(self.table(self.valid(~self.cold, origin_idx=[i]), {"origin": pd.Timestamp(o).date()}))
        return pd.concat(rows, ignore_index=True)

    def origin_stability(self) -> pd.DataFrame:
        po = self.per_origin()
        g = po.groupby("model")[["MAE", "RMSE", "WAPE", "pinball"]].agg(["mean", "std"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        return g

    def calibration(self) -> pd.DataFrame:
        v = self.valid(~self.cold)
        rows = []
        for m in self.models:
            e = self.err[m]
            for j, q in enumerate(self.qs):
                le, lt = e.hit[..., j][v].mean(), e.hit_lt[..., j][v].mean()
                rows.append({"model": m, "nominal": q, "empirical": le, "empirical_strict": lt,
                             "inside_discrete_band": bool(lt <= q + 0.02 and le >= q - 0.02)})
        return pd.DataFrame(rows)

    def pinball_by_quantile(self) -> pd.DataFrame:
        v = self.valid(~self.cold)
        rows = []
        for m in self.models:
            for j, q in enumerate(self.qs):
                rows.append({"model": m, "q": q, "pinball": self.err[m].PBq[..., j][v].mean()})
        return pd.DataFrame(rows)

    def runtime(self) -> pd.DataFrame:
        p = self.run_dir / "timing.csv"
        if not p.exists():
            return pd.DataFrame()
        t = pd.read_csv(p)
        g = t.groupby("model")[["fit_s", "predict_s"]].sum()
        g["total_s"] = g.fit_s + g.predict_s
        return g

    def significance(self, reference: str | None = None, metric="WAPE", n_boot=1000, seed=0, mode="all") -> pd.DataFrame:
        """Paired bootstrap over SERIES of (model - reference). Negative diff = model better (lower)."""
        reference = reference or self.ev.get("reference_model") or self.models[0]
        v = self.valid(~self.cold, mode)
        keep = np.where(~self.cold)[0]

        def per_series(m):
            e = self.err[m]
            if metric == "WAPE":
                return (e.AE * v).sum((0, 2))[keep], (self.Y * v).sum((0, 2))[keep]
            return (e.PB * v).sum((0, 2))[keep], v.sum((0, 2))[keep].astype(float)

        rng = np.random.default_rng(seed)
        nr, dr = per_series(reference)
        idx = rng.integers(0, len(keep), size=(n_boot, len(keep)))
        rows = []
        for m in self.models:
            if m == reference:
                continue
            nm, dm = per_series(m)
            bm = nm[idx].sum(1) / np.maximum(dm[idx].sum(1), 1e-9)
            br = nr[idx].sum(1) / np.maximum(dr[idx].sum(1), 1e-9)
            d = bm - br
            point = nm.sum() / max(dm.sum(), 1e-9) - nr.sum() / max(dr.sum(), 1e-9)
            win = np.mean((nm / np.maximum(dm, 1e-9)) < (nr / np.maximum(dr, 1e-9)))
            rows.append({"model": m, "reference": reference, "metric": metric, "diff": point,
                         "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                         "significant": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
                         "series_win_rate": win})
        return pd.DataFrame(rows)

    def all_tables(self) -> dict:
        t = {"main_all_days": self.main_table("all"), "main_non_censored": self.main_table("non_censored"),
             "by_horizon": self.by_horizon(), "by_slice": self.by_slice(),
             "censoring_effect": self.censoring_effect(), "origin_stability": self.origin_stability(),
             "per_origin": self.per_origin(), "calibration": self.calibration(),
             "pinball_by_quantile": self.pinball_by_quantile(), "runtime": self.runtime(),
             "significance_wape": self.significance(metric="WAPE"),
             "significance_pinball": self.significance(metric="pinball")}
        cs = self.cold_start()
        if len(cs):
            t["cold_start"] = cs
        return t


def evaluate_run(cfg: dict, run_dir, out_sub="results") -> dict:
    ev = Evaluator(cfg, run_dir)
    tables = ev.all_tables()
    out = Path(run_dir) / out_sub
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if df is not None and len(df):
            df.to_csv(out / f"{name}.csv")
    log.info("wrote %d tables to %s", len(tables), out)
    return {"evaluator": ev, "tables": tables}
