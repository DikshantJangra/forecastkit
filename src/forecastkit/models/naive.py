"""Sanity-floor baselines: naive, seasonal naive (optionally averaged over n seasons), moving average.
Intervals = point forecast + empirical in-sample residual quantiles (per series), clipped at 0."""
from __future__ import annotations

import numpy as np

from forecastkit.utils import register_model

from .base import BaseForecaster


def _shift(y, k):
    out = np.full_like(y, np.nan)
    out[:, k:] = y[:, :-k]
    return out


@register_model("naive")
class NaiveForecaster(BaseForecaster):
    """method: 'naive' (last value) | 'seasonal' (mean of the last n_seasons same-weekday values) | 'ma' (window)"""

    def fit(self, ctx):
        self.ctx = ctx
        y = ctx.target(self.p("use_mask", True)).astype(np.float64)
        m, method = int(self.p("season", ctx.season)), self.p("method", "seasonal")
        n_s, window = int(self.p("n_seasons", 1)), int(self.p("window", 28))
        rw = int(self.p("resid_days", 180))
        self._horizon = ctx.horizon
        if method == "seasonal":
            fitted = np.nanmean(np.stack([_shift(y, m * k) for k in range(1, n_s + 1)]), axis=0)
            y[:, -m * n_s:]
            # point forecast for day h: mean over k of y[T - m*k + ((h-1) % m) + 1 ... ] aligned weekday values
            H = ctx.horizon
            pt = np.zeros((ctx.N, H))
            T = y.shape[1]
            for h in range(1, H + 1):
                vals = []
                for k in range(1, n_s + 1):
                    j = T + h - 1 - m * (int(np.ceil(h / m)) + k - 1)
                    vals.append(y[:, j] if j >= 0 else np.full(ctx.N, np.nan))
                pt[:, h - 1] = np.nanmean(np.stack(vals), axis=0)
        elif method == "naive":
            fitted = _shift(y, 1)
            lastv = np.array([row[~np.isnan(row)][-1] if (~np.isnan(row)).any() else 0.0 for row in y])
            pt = np.repeat(lastv[:, None], ctx.horizon, axis=1)
        elif method == "ma":
            T = y.shape[1]
            fitted = np.full_like(y, np.nan)
            cs = np.nancumsum(y, axis=1)
            cn = np.cumsum(~np.isnan(y), axis=1)
            for t in range(window, T):
                s = cs[:, t - 1] - (cs[:, t - 1 - window] if t - 1 - window >= 0 else 0)
                c = cn[:, t - 1] - (cn[:, t - 1 - window] if t - 1 - window >= 0 else 0)
                fitted[:, t] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            tail = y[:, -window:]
            mu = np.nanmean(tail, axis=1)
            pt = np.repeat(np.nan_to_num(mu)[:, None], ctx.horizon, axis=1)
        else:
            raise ValueError(method)
        pt = np.nan_to_num(pt, nan=0.0)
        resid = (y - fitted)[:, -rw:]
        self._pt, self._resid = pt, resid
        return self

    def predict(self, quantiles):
        qs = np.asarray(quantiles, dtype=float)
        resid = self._resid
        rq = np.zeros((resid.shape[0], len(qs)))
        for i in range(resid.shape[0]):
            r = resid[i][~np.isnan(resid[i])]
            rq[i] = np.quantile(r, qs) if len(r) >= 8 else 0.0
        q = self._pt[:, :, None] + rq[:, None, :]
        return self.finalize(q, mean=self._pt)


@register_model("seasonal_naive")
class SeasonalNaive(NaiveForecaster):
    def __init__(self, **params):
        params.setdefault("method", "seasonal")
        super().__init__(**params)


@register_model("moving_average")
class MovingAverage(NaiveForecaster):
    def __init__(self, **params):
        params.setdefault("method", "ma")
        super().__init__(**params)
