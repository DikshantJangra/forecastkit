"""Local state-space model per SKU: local level + stochastic seasonal (dummy form), fitted with a
BATCHED Kalman filter (all series in parallel in numpy).

    y_t = level_t + gamma_t + eps_t          eps ~ N(0, sigma2)
    level_{t+1} = level_t + xi_t             xi  ~ N(0, q_level * sigma2)
    gamma_{t+1} = -sum_{j<m-1} gamma_{t-j} + w_t   w ~ N(0, q_seas * sigma2)

* missing/censored/unavailable days are simply *missing observations* (the filter skips the update)
* noise ratios (q_level, q_seas) picked per series by the concentrated log-likelihood over a small
  grid ('grid'), or one shared pair for all series ('shared'), or fixed ('fixed')   -> speed/accuracy trade-off
* sigma2 is concentrated out of the likelihood
* data are modelled on a transformed scale (sqrt by default) and quantiles mapped back (monotone => exact)
* diffuse prior: a series with little history gets WIDE intervals - this is the local model's cold-start story
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from forecastkit.schema import QuantileForecast
from forecastkit.utils import register_model
from .base import BaseForecaster

TRANSFORMS = {
    "sqrt": (np.sqrt, lambda z: np.maximum(z, 0.0) ** 2),
    "log1p": (np.log1p, lambda z: np.expm1(np.maximum(z, 0.0))),
    "none": (lambda y: y, lambda z: np.maximum(z, 0.0)),
}


class StateSpace:
    def __init__(self, m: int):
        d = max(m, 1)
        T = np.zeros((d, d))
        T[0, 0] = 1.0
        Z = np.zeros(d)
        Z[0] = 1.0
        if m > 1:
            T[1, 1:] = -1.0
            for i in range(2, d):
                T[i, i - 1] = 1.0
            Z[1] = 1.0
        self.T, self.Z, self.d, self.m = T, Z, d, m

    def Q(self, q_lvl, q_seas):
        B = len(q_lvl)
        Q = np.zeros((B, self.d, self.d))
        Q[:, 0, 0] = q_lvl
        if self.d > 1:
            Q[:, 1, 1] = q_seas
        return Q


def kf_filter(Y, ss: StateSpace, q_lvl, q_seas, a0, P0, burn_in=14, store=False):
    """Batched filter. Y [B,Tn] with NaN = missing. Variances are in units of sigma2 (R = 1).
    Returns dict(a, P (state predicted for time Tn), sum_logF, sum_v2F, n, [states, sd1])."""
    B, Tn = Y.shape
    d, T, Z = ss.d, ss.T, ss.Z
    a, P = a0.copy(), P0.copy()
    Q = ss.Q(q_lvl, q_seas)
    sum_logF, sum_v2F, n, seen = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
    st_a = np.zeros((B, Tn, d), dtype=np.float32) if store else None
    st_sd1 = np.zeros((B, Tn), dtype=np.float32) if store else None
    for t in range(Tn):
        y = Y[:, t]
        obs = ~np.isnan(y)
        PZ = P @ Z
        F = PZ @ Z + 1.0
        v = np.where(obs, y - a @ Z, 0.0)
        gain = PZ / F[:, None]
        a = np.where(obs[:, None], a + gain * v[:, None], a)
        P = np.where(obs[:, None, None], P - np.einsum("bi,bj->bij", PZ, PZ) / F[:, None, None], P)
        counted = obs & (seen >= burn_in)
        sum_logF += np.where(counted, np.log(F), 0.0)
        sum_v2F += np.where(counted, v * v / F, 0.0)
        n += counted
        seen += obs
        if store:
            st_a[:, t] = a
        a = a @ T.T
        P = np.matmul(np.matmul(T, P), T.T) + Q
        P = 0.5 * (P + np.transpose(P, (0, 2, 1)))
        if store:
            st_sd1[:, t] = np.sqrt((P @ Z) @ Z + 1.0)
    return {"a": a, "P": P, "sum_logF": sum_logF, "sum_v2F": sum_v2F, "n": n, "states": st_a, "sd1": st_sd1}


class KalmanCore:
    """Parameter selection + final filtering for a [N, Tn] block of (transformed) data."""

    def __init__(self, season=7, grid_level=(1e-3, 1e-2, 3e-2, 1e-1), grid_seas=(0.0, 1e-4, 1e-3, 1e-2),
                 param_mode="grid", burn_in=14, kappa=1e3, min_obs=28, fixed=(1e-2, 1e-3)):
        self.ss = StateSpace(season)
        self.grid = [(a, b) for a in grid_level for b in grid_seas]
        self.param_mode, self.burn_in, self.kappa, self.min_obs, self.fixed = param_mode, burn_in, kappa, min_obs, fixed

    def _init(self, Y):
        B = Y.shape[0]
        d = self.ss.d
        a0 = np.zeros((B, d))
        with np.errstate(all="ignore"):
            first = np.array([np.nanmean(r[~np.isnan(r)][:14]) if (~np.isnan(r)).any() else 0.0 for r in Y])
        a0[:, 0] = np.nan_to_num(first)
        P0 = np.broadcast_to(np.eye(d) * self.kappa, (B, d, d)).copy()
        return a0, P0

    def choose_params(self, Y):
        N = Y.shape[0]
        nobs = (~np.isnan(Y)).sum(1)
        if self.param_mode == "fixed":
            return np.full(N, self.fixed[0]), np.full(N, self.fixed[1])
        G = len(self.grid)
        Yr = np.repeat(Y, G, axis=0)
        gl = np.tile(np.array([g[0] for g in self.grid]), N)
        gs = np.tile(np.array([g[1] for g in self.grid]), N)
        a0, P0 = self._init(Yr)
        r = kf_filter(Yr, self.ss, gl, gs, a0, P0, self.burn_in)
        n = np.maximum(r["n"], 1)
        s2 = np.maximum(r["sum_v2F"] / n, 1e-3)
        ll = -0.5 * (n * np.log(s2) + r["sum_logF"])
        ll = np.where(r["n"] >= 5, ll, -np.inf).reshape(N, G)
        if self.param_mode == "shared":
            ok = np.isfinite(ll).all(1) & (nobs >= self.min_obs)
            best = int(np.argmax(ll[ok].sum(0))) if ok.any() else 0
            idx = np.full(N, best)
        else:
            idx = np.argmax(np.where(np.isfinite(ll).any(1, keepdims=True), ll, 0), axis=1)
        qd = np.array(self.grid)
        q_l, q_s = qd[idx, 0].copy(), qd[idx, 1].copy()
        short = nobs < self.min_obs  # too little data to estimate: default moderate noise
        q_l[short], q_s[short] = self.fixed
        return q_l, q_s

    def run(self, Y, q_l, q_s, store=False):
        a0, P0 = self._init(Y)
        r = kf_filter(Y, self.ss, q_l, q_s, a0, P0, self.burn_in, store=store)
        r["sigma2"] = np.maximum(r["sum_v2F"] / np.maximum(r["n"], 1), 1e-3)
        r["q_l"], r["q_s"] = q_l, q_s
        return r

    def forecast(self, r, H):
        """Predictive mean / variance on the transformed scale for h=1..H (state in r is already the 1-step prior)."""
        ss = self.ss
        a, P = r["a"].copy(), r["P"].copy()
        Q = ss.Q(r["q_l"], r["q_s"])
        mu = np.zeros((a.shape[0], H))
        var = np.zeros_like(mu)
        for h in range(H):
            mu[:, h] = a @ ss.Z
            var[:, h] = ((P @ ss.Z) @ ss.Z + 1.0) * r["sigma2"]
            a = a @ ss.T.T
            P = np.matmul(np.matmul(ss.T, P), ss.T.T) + Q
        return mu, var


@register_model("kalman")
class KalmanForecaster(BaseForecaster):
    """params: season, transform ('sqrt'|'log1p'|'none'), window (days of history used), param_mode
    ('grid'|'shared'|'fixed'), grid_level, grid_seas, use_mask (treat censored days as missing)"""

    def _core(self):
        c = self.ctx
        return KalmanCore(season=int(self.p("season", c.season)),
                          grid_level=tuple(self.p("grid_level", (1e-3, 1e-2, 3e-2, 1e-1))),
                          grid_seas=tuple(self.p("grid_seas", (0.0, 1e-4, 1e-3, 1e-2))),
                          param_mode=self.p("param_mode", "grid"), burn_in=int(self.p("burn_in", 14)),
                          kappa=float(self.p("kappa", 1e3)), min_obs=int(self.p("min_obs", 28)))

    def fit(self, ctx):
        self.ctx = ctx
        tf, self.inv = TRANSFORMS[self.p("transform", "sqrt")]
        self.tf = tf
        w = int(self.p("window", 365))
        y = ctx.target(self.p("use_mask", True)).astype(np.float64)[:, -w:]
        self._yt = tf(y)
        self.core = self._core()
        q_l, q_s = self.core.choose_params(self._yt)
        self.res = self.core.run(self._yt, q_l, q_s)
        self.mu, self.var = self.core.forecast(self.res, ctx.horizon)
        return self

    def predict(self, quantiles):
        qs = np.asarray(quantiles, dtype=float)
        sd = np.sqrt(self.var)
        z = norm.ppf(qs)
        zt = self.mu[:, :, None] + sd[:, :, None] * z[None, None, :]
        q = self.inv(zt)
        kind = self.p("transform", "sqrt")
        if kind == "sqrt":
            mean = np.maximum(self.mu, 0) ** 2 + self.var
        elif kind == "log1p":
            mean = np.expm1(np.minimum(self.mu + 0.5 * self.var, 12.0))
        else:
            mean = self.mu
        return self.finalize(q, mean=mean)

    # ------------------------------------------------------------ diagnostics (for the report)
    def diagnostics(self, i: int, max_days: int = 365) -> dict:
        """Filtered level, its std, seasonal part and standardized innovations for series index i."""
        y = self._yt[i:i + 1]
        r = self.core.run(y, self.res["q_l"][i:i + 1], self.res["q_s"][i:i + 1], store=True)
        a = r["states"][0].astype(np.float64)
        ss = self.core.ss
        level = a[:, 0]
        seas = a[:, 1] if ss.d > 1 else np.zeros_like(level)
        pred = np.concatenate([[a[0] @ ss.Z], (a[:-1] @ ss.T.T) @ ss.Z])  # 1-step-ahead predictions
        sd1 = r["sd1"][0] * np.sqrt(r["sigma2"][0])
        innov = (y[0] - np.concatenate([[np.nan], pred[1:]])) / np.concatenate([[np.nan], sd1[:-1]])
        hist_dates = self.ctx.hist_dates[-y.shape[1]:]
        return {"dates": hist_dates, "y": y[0], "level": level, "seasonal": seas, "level_sd": None,
                "innovations": innov, "q_level": float(r["q_l"][0]), "q_seas": float(r["q_s"][0]),
                "sigma2": float(r["sigma2"][0]), "transform": self.p("transform", "sqrt")}
