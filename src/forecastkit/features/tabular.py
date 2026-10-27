"""Leak-safe lag / rolling features for tabular (gradient-boosting) models.

Everything is computed from arrays that END at the forecast origin, and every feature at anchor day t
uses indices <= t only (cumulative-sum tricks; no centered windows, no future shifts).  Target-date
features (calendar, price, events) come from the known-future covariate array.
"""
from __future__ import annotations

import numpy as np


def causal_roll(Y: np.ndarray, windows) -> dict:
    """Y [N,T] with NaN = missing.  out['mean_w'][:, t] = mean of observed y over (t-w, t], NaN if none."""
    obs = ~np.isnan(Y)
    V = np.where(obs, Y, 0.0)
    cs = np.cumsum(V, 1)
    cc = np.cumsum(obs, 1)
    cs2 = np.cumsum(V * V, 1)
    cnz = np.cumsum((V > 0) & obs, 1)
    pad = lambda a: np.concatenate([np.zeros((a.shape[0], 1)), a], 1)
    cs, cc, cs2, cnz = pad(cs), pad(cc), pad(cs2), pad(cnz)
    T = Y.shape[1]
    t = np.arange(1, T + 1)
    out = {}
    for w in windows:
        lo = np.maximum(t - w, 0)
        s, c = cs[:, t] - cs[:, lo], cc[:, t] - cc[:, lo]
        s2, nz = cs2[:, t] - cs2[:, lo], cnz[:, t] - cnz[:, lo]
        with np.errstate(all="ignore"):
            mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            var = np.where(c > 1, (s2 - c * mean ** 2) / np.maximum(c - 1, 1), np.nan)
            out[f"mean_{w}"] = mean
            out[f"std_{w}"] = np.sqrt(np.maximum(var, 0))
            out[f"nzrate_{w}"] = np.where(c > 0, nz / np.maximum(c, 1), np.nan)
            out[f"maskfrac_{w}"] = 1.0 - c / (t - lo)[None, :]
    return out


def recency_features(Y: np.ndarray) -> dict:
    """days since last sale, age (days since first observation), expanding mean."""
    _N, T = Y.shape
    obs = ~np.isnan(Y)
    pos = obs & (Y > 0)
    idx = np.arange(T)[None, :]
    last_pos = np.maximum.accumulate(np.where(pos, idx, -1), axis=1)
    dsl = np.where(last_pos >= 0, idx - last_pos, np.nan)
    first_obs = np.where(obs.any(1), obs.argmax(1), T)
    age = np.where(idx >= first_obs[:, None], idx - first_obs[:, None], np.nan)
    V = np.where(obs, Y, 0.0)
    with np.errstate(all="ignore"):
        exp_mean = np.cumsum(V, 1) / np.maximum(np.cumsum(obs, 1), 1)
    return {"days_since_sale": dsl, "age": age, "mean_all": exp_mean}


class TabularFeatures:
    """Pre-computes causal per-day arrays once per origin; `rows(series, anchors, hs)` then assembles a
    feature matrix for any (series, anchor day t, horizon h) triples."""

    ROLL = (7, 14, 28, 56, 112)

    def __init__(self, y: np.ndarray, K_all: np.ndarray, known_names, static_codes: np.ndarray, static_names,
                 extra: dict | None = None, kf_state: dict | None = None):
        self.y = y                          # [N, Th] (NaN = masked) -- ends at the origin
        self.K = K_all                      # [N, Th+H, k] known-future covariates (hist + future)
        self.Th = y.shape[1]
        self.known_names = list(known_names)
        self.static_codes, self.static_names = static_codes, list(static_names)
        r = causal_roll(y, self.ROLL)
        r.update(recency_features(y))
        self.day_feats = r                  # name -> [N, Th]
        if extra:
            self.day_feats.update(extra)
        self.day_names = list(self.day_feats)
        # optional Kalman state (hybrid model): {'st': [N,Th,d] posterior states, 'Wh': [Hmax+1,d] = Z T^h, 'inv': fn}
        self.kf_state = kf_state
        self.feature_names = (["h", "y_t0", "y_t1", "y_t2"] + [f"slag_{k}" for k in range(1, 5)] + ["slag_mean"]
                              + self.day_names + [f"k_{n}" for n in self.known_names] + [f"s_{n}" for n in self.static_names]
                              + (["kf_mean_h"] if kf_state is not None else []))

    def rows(self, s: np.ndarray, t: np.ndarray, h: np.ndarray) -> np.ndarray:
        y = self.y
        c = np.ceil(h / 7.0).astype(int)    # aligned same-weekday lags (weekly seasonality)
        cols = [h.astype(np.float32)]
        for lag in (0, 1, 2):
            cols.append(y[s, np.maximum(t - lag, 0)])
        sl = []
        for k in range(1, 5):
            j = t + h - 7 * (c + k - 1)
            v = np.where(j >= 0, y[s, np.clip(j, 0, self.Th - 1)], np.nan)
            sl.append(v)
        cols += sl
        with np.errstate(all="ignore"):
            cols.append(np.nanmean(np.stack(sl), axis=0))
        for n in self.day_names:
            cols.append(self.day_feats[n][s, t])
        kmat = self.K[s, t + h]             # known covariates at the TARGET date
        cols += [kmat[:, i] for i in range(kmat.shape[1])]
        cols += [self.static_codes[s, i] for i in range(self.static_codes.shape[1])]
        if self.kf_state is not None:       # h-step-ahead mean of the local state-space model at the anchor
            a = self.kf_state["st"][s, t]
            m = np.einsum("rd,rd->r", a, self.kf_state["Wh"][h])
            cols.append(self.kf_state["inv"](m))
        return np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)
