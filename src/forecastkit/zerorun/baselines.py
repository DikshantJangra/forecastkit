"""Non-learned baselines for pi_h = P(Y_{t+h} = 0 | F_t). All use history <= origin only.

* series_zero_rate   expanding fraction of recorded-zero days of the series (constant over h)
* recent_zero_rate   same over the last `window` days
* KMRenewal          run-duration baseline: pooled Kaplan-Meier survival S(d) = P(run length > d) of zero runs,
                     stratified by the series' ADI quartile, with the run still open at the origin treated as
                     RIGHT-CENSORED (its final length is unknown). For a series whose current run has length
                     r >= 1:   P(run still open at t+h) = S(r+h-1) / S(r-1)
                     and once the run has ended the day is zero with the series' zero rate z_i (renewal
                     approximation):  pi_h = s_h + (1 - s_h) z_i.   For r = 0 (sold on the origin day) pi_h = z_i.
"""
from __future__ import annotations

import numpy as np


def series_zero_rate(y_hist: np.ndarray, H: int) -> np.ndarray:
    obs = ~np.isnan(y_hist)
    z = ((y_hist == 0) & obs).sum(1) / np.maximum(obs.sum(1), 1)
    z = np.where(obs.sum(1) > 0, z, 0.5)
    return np.repeat(z[:, None], H, 1)


def recent_zero_rate(y_hist: np.ndarray, H: int, window: int = 28) -> np.ndarray:
    return series_zero_rate(y_hist[:, -window:], H)


def _runs(y: np.ndarray):
    """(completed run lengths, open run length at the end or 0) for one series; runs start after the first sale."""
    pos = np.flatnonzero(np.nan_to_num(y) > 0)
    if len(pos) == 0:
        return np.zeros(0, dtype=int), 0
    gaps = pos[1:] - pos[:-1] - 1
    open_run = len(y) - 1 - pos[-1]
    return gaps[gaps > 0], int(open_run)


def kaplan_meier(durations: np.ndarray, observed: np.ndarray, dmax: int) -> np.ndarray:
    """S[d] = P(L > d), d = 0..dmax, from (possibly right-censored) integer durations >= 1."""
    durations = np.minimum(durations.astype(int), dmax + 1)
    deaths = np.bincount(durations[observed], minlength=dmax + 2)
    total = np.bincount(durations, minlength=dmax + 2)
    at_risk = total[::-1].cumsum()[::-1]                 # number with L >= d
    with np.errstate(all="ignore"):
        haz = np.where(at_risk > 0, deaths / np.maximum(at_risk, 1), 0.0)
    S = np.cumprod(1.0 - haz)                            # S[d] = prod_{k<=d} (1 - h_k) = P(L > d)
    return S[: dmax + 1]


class KMRenewal:
    def __init__(self, n_strata: int = 4, dmax: int = 1100):
        self.n_strata, self.dmax = n_strata, dmax

    def fit(self, y_hist: np.ndarray):
        N, _T = y_hist.shape
        obs = ~np.isnan(y_hist)
        nz = ((np.nan_to_num(y_hist) > 0) & obs).sum(1)
        adi = np.where(nz > 0, obs.sum(1) / np.maximum(nz, 1), np.inf)
        fin = np.isfinite(adi)
        self.edges = np.quantile(adi[fin], np.linspace(0, 1, self.n_strata + 1)[1:-1]) if fin.any() else np.array([])
        self.stratum = np.searchsorted(self.edges, adi)
        self.open_run = np.zeros(N, dtype=int)
        per = {k: ([], []) for k in range(self.n_strata + 1)}
        for i in range(N):
            done, op = _runs(y_hist[i])
            self.open_run[i] = op
            d, o = per[self.stratum[i]]
            d.extend(done.tolist())
            o.extend([True] * len(done))
            if op > 0:
                d.append(op)
                o.append(False)                          # right-censored at the origin
        self.S = {}
        for k, (d, o) in per.items():
            if len(d):
                self.S[k] = kaplan_meier(np.array(d), np.array(o, dtype=bool), self.dmax)
        all_d = np.concatenate([np.array(per[k][0]) for k in per if per[k][0]] or [np.zeros(0)])
        all_o = np.concatenate([np.array(per[k][1], dtype=bool) for k in per if per[k][1]] or [np.zeros(0, bool)])
        self.S_pool = kaplan_meier(all_d, all_o, self.dmax) if len(all_d) else np.ones(self.dmax + 1)
        self.z = series_zero_rate(y_hist, 1)[:, 0]
        self.no_sale = nz == 0
        return self

    def predict(self, H: int) -> np.ndarray:
        N = len(self.z)
        out = np.repeat(self.z[:, None], H, 1).astype(np.float64)
        hs = np.arange(1, H + 1)
        for i in range(N):
            r = self.open_run[i]
            if r <= 0:
                continue
            S = self.S.get(self.stratum[i], self.S_pool)
            Sx = np.concatenate([[1.0], S])              # Sx[d+1] = S[d], Sx[0] = P(L > -1) = 1
            num = Sx[np.minimum(r + hs - 1, self.dmax) + 1]
            den = max(Sx[min(r - 1, self.dmax) + 1], 1e-12)
            s = np.clip(num / den, 0, 1)
            out[i] = s + (1 - s) * self.z[i]
        out[self.no_sale] = 1.0 - 1e-3                    # nothing ever sold in the visible history
        return np.clip(out, 1e-6, 1 - 1e-6)
