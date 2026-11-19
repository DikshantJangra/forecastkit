"""ML + DL hybrid: DeepAR (latent-demand distribution) + LightGBM zero-run persistence model (recorded zeros).

Why two components.  DeepAR is trained with the loss MASKED on days inside long zero runs (the stockout proxy),
so it learns the demand distribution *when the item sells normally*. Recorded sales, however, also contain
those long zero runs. The zero-run persistence model (forecastkit.zerorun) predicts exactly that part:
pi_h = P(Y_{t+h} = 0 | F_t) for recorded sales.

Fusion (Rao-Blackwellised hurdle; `fusion: hurdle`, the primary model).  Let p_D be DeepAR's predictive
marginal at horizon h, a mixture over S sample paths of NB(mu_s, alpha_s):
    p_D(y) = (1/S) sum_s NB(y; mu_s, alpha_s),     p_D(0) = (1/S) sum_s (r_s / (r_s + mu_s))^{r_s},  r_s = 1/alpha_s
The hybrid predictive distribution is the hurdle
    q(0) = pi_h,      q(y) = (1 - pi_h) * p_D(y) / (1 - p_D(0))      for y >= 1
i.e. the ML model owns the zero / non-zero decision and the DL model keeps the SHAPE of positive sales.
q is a valid pmf (q >= 0, sums to 1), non-negative, and its quantiles are
    Q(tau) = 0                                   if tau <= pi_h
           = F_+^{-1}((tau - pi_h) / (1 - pi_h))  otherwise,  F_+ = CDF of p_D(. | Y > 0)
and its mean is (1 - pi_h) E_D[Y | Y > 0] = (1 - pi_h) sum_s mu_s / sum_s (1 - p0_s).

Every fusion variant below changes ONLY the zero mass q(0); the positive conditional is DeepAR's in all of them,
so their differences isolate the zero-probability modelling:
    hurdle      q(0) = pi                                 (primary)
    zi_naive    q(0) = pi + (1 - pi) p_D(0)               (naive zero-inflated mixture pi*delta_0 + (1-pi)*p_D:
                                                           counts zeros twice, DeepAR's own zeros plus pi)
    zi_inflate  q(0) = max(pi, p_D(0))                    (consistent zero inflation: can only add zero mass)
    gated       q(0) = pi if the series is in a causally flagged zero run at the origin, else p_D(0)
    none        q(0) = p_D(0)                             (DeepAR reconstructed through the same code path; control)

`positive_source: empirical` replaces DeepAR by the empirical distribution of the series' positive sales in the
last `empirical_window` days (the "no DL component" ablation). Positive-part sampling for DeepAR is exact:
choose path s with probability proportional to 1 - p0_s, then draw from the zero-truncated NB(mu_s, alpha_s)
by inverse CDF.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from forecastkit.utils import register_model
from forecastkit.zerorun.model import ZeroRunModel

from .base import BaseForecaster

FUSIONS = ("hurdle", "zi_naive", "zi_inflate", "gated", "none")


def zero_mass(fusion: str, pi: np.ndarray, p0_d: np.ndarray, gate: np.ndarray | None = None) -> np.ndarray:
    """q(0) of the fused distribution, [N,H]."""
    if fusion == "hurdle":
        return pi
    if fusion == "zi_naive":
        return pi + (1 - pi) * p0_d
    if fusion == "zi_inflate":
        return np.maximum(pi, p0_d)
    if fusion == "gated":
        g = np.zeros(pi.shape[0], bool) if gate is None else gate
        return np.where(g[:, None], pi, p0_d)
    if fusion == "none":
        return p0_d
    raise ValueError(f"unknown fusion {fusion!r}; choose from {FUSIONS}")


def hurdle_quantiles(q0: np.ndarray, pos_sorted: np.ndarray, qs) -> np.ndarray:
    """Quantiles of q = q0*delta_0 + (1-q0)*F_+ with F_+ the empirical distribution of positive samples.
    q0 [N,H]; pos_sorted [N,H,M] ascending positive samples. Linear interpolation between order statistics,
    as numpy's default quantile. Returns [N,H,Q]."""
    N, H, M = pos_sorted.shape
    out = np.zeros((N, H, len(qs)), dtype=np.float64)
    for j, tau in enumerate(qs):
        lev = np.clip((tau - q0) / np.maximum(1 - q0, 1e-12), 0.0, 1.0)
        pos = lev * (M - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, M - 1)
        w = pos - lo
        v_lo = np.take_along_axis(pos_sorted, lo[..., None], -1)[..., 0]
        v_hi = np.take_along_axis(pos_sorted, hi[..., None], -1)[..., 0]
        out[..., j] = np.where(tau <= q0, 0.0, v_lo + w * (v_hi - v_lo))
    return out


def truncated_nb_mixture_samples(mu: np.ndarray, alpha: np.ndarray, M: int, rng) -> tuple:
    """Exact draws from p_D(. | Y > 0) where p_D = mean over paths of NB(mu_s, alpha_s).
    mu, alpha [N,S,H]. Returns (samples [N,M,H] >= 1, p0_D [N,H], E_D[Y | Y>0] [N,H])."""
    from scipy.stats import nbinom
    mu = np.maximum(mu.astype(np.float64), 1e-8)
    r = 1.0 / alpha.astype(np.float64)
    p = r / (r + mu)
    p0 = np.exp(r * np.log(p))                                      # per path NB(0)
    w = 1.0 - p0                                                    # path weight given Y > 0
    p0_d = p0.mean(1)
    N, S, H = mu.shape
    wt = np.transpose(w, (0, 2, 1)).reshape(N * H, S)               # [C,S]
    tot = wt.sum(1, keepdims=True)
    wt = np.where(tot > 0, wt / np.maximum(tot, 1e-300), 1.0 / S)
    cw = np.cumsum(wt, 1)
    cw[:, -1] = 1.0
    C = N * H
    u = rng.random((C, M))
    off = np.arange(C)[:, None]
    path = np.searchsorted((cw + off).ravel(), (u + off).ravel(), side="right").reshape(C, M) - off * S
    path = np.clip(path, 0, S - 1)
    gather = lambda a: np.take_along_axis(np.transpose(a, (0, 2, 1)).reshape(C, S), path, 1)
    r_s, p_s, f0 = gather(r), gather(p), gather(p0)
    v = rng.random((C, M))
    uu = np.minimum(f0 + (1.0 - f0) * v, 1.0 - 1e-12)
    y = nbinom.ppf(uu, r_s, p_s)
    y = np.where(np.isfinite(y), y, 1.0)
    y = np.maximum(y, 1.0)
    samples = y.reshape(N, H, M).transpose(0, 2, 1)
    with np.errstate(all="ignore"):
        m_pos = mu.sum(1) / np.maximum(w.sum(1), 1e-12)
    return samples.astype(np.float32), p0_d, np.maximum(m_pos, 1.0)


@register_model("deepar_hurdle")
class DeepARHurdle(BaseForecaster):
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "deepar": {}, "zero_model": {}, "fusion": "hurdle", "positive_source": "deepar", "positive_samples": 200,
        "empirical_window": 112, "export": True,
    }

    def pp(self, k):
        return self.params.get(k, self.DEFAULTS[k])

    def fit(self, ctx):
        from .deepar import DeepARForecaster
        self.ctx = ctx
        if self.pp("fusion") not in FUSIONS:
            raise ValueError(f"unknown fusion {self.pp('fusion')!r}")
        need_zero = self.pp("fusion") != "none"
        self.zero = ZeroRunModel(seed=int(ctx.seed), **dict(self.pp("zero_model"))).fit(ctx) if need_zero else None
        if self.pp("positive_source") == "deepar":
            if not hasattr(self, "dar"):                 # one DeepAR across origins -> same warm-start protocol
                self.dar = DeepARForecaster(**{**dict(self.pp("deepar")), "export": False})
            self.dar.fit(ctx)
        elif self.pp("positive_source") != "empirical":
            raise ValueError("positive_source must be 'deepar' or 'empirical'")
        return self

    def _empirical_positive(self, M, rng):
        ctx = self.ctx
        W = int(self.pp("empirical_window"))
        y = ctx.y[:, -W:]                                # masked history: censored days excluded
        N, H = ctx.N, ctx.horizon
        out = np.ones((N, M, H), dtype=np.float32)
        m_pos = np.ones((N, H))
        for i in range(N):
            v = y[i][np.nan_to_num(y[i]) > 0]
            if len(v):
                out[i] = rng.choice(v, size=(M, H), replace=True)
                m_pos[i] = v.mean()
        return out, m_pos

    def predict(self, quantiles):
        ctx = self.ctx
        N, H = ctx.N, ctx.horizon
        rng = np.random.default_rng(int(ctx.seed) + 4242)
        M = int(self.pp("positive_samples"))
        if self.pp("positive_source") == "deepar":
            self.dar.predict(quantiles)                  # draws the sample paths, keeps per-path NB params
            if self.dar.last_dist is not None:
                pos, p0_d, m_pos = truncated_nb_mixture_samples(self.dar.last_dist["mu"], self.dar.last_dist["alpha"],
                                                                M, rng)
            else:                                        # non-NB likelihood: sample-based fallback
                s = self.dar.last_samples
                p0_d = (s == 0).mean(1)
                pos = np.ones((N, M, H), dtype=np.float32)
                m_pos = np.ones((N, H))
                for i in range(N):
                    for h in range(H):
                        v = s[i, :, h][s[i, :, h] > 0]
                        if len(v):
                            pos[i, :, h] = rng.choice(v, M)
                            m_pos[i, h] = v.mean()
        else:
            pos, m_pos = self._empirical_positive(M, rng)
            p0_d = np.zeros((N, H))
        self.pi = self.zero.predict_proba() if self.zero is not None else np.full((N, H), np.nan)
        gate = ctx.censored[:, -1] if ctx.Th else np.zeros(N, bool)   # causal: flag computed on history only
        q0 = zero_mass(self.pp("fusion"), self.pi, p0_d, gate)
        self.q0, self.p0_d = q0, p0_d
        pos_sorted = np.sort(np.transpose(pos, (0, 2, 1)), -1)          # [N,H,M]
        Q = hurdle_quantiles(q0, pos_sorted, list(quantiles))
        mean = (1 - q0) * m_pos
        nz = rng.random((N, M, H)) >= q0[:, None, :]
        self.last_samples = (pos * nz).astype(np.float32)               # draws from the fused distribution
        return self.finalize(Q, mean=mean)

    def export(self, out_dir, ctx):
        if not bool(self.pp("export")):
            return
        import json
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = str(ctx.origin.date())
        np.savez_compressed(out / f"pred_{tag}.npz", samples=self.last_samples, p0=self.q0.astype(np.float32),
                            p0_deepar=np.asarray(self.p0_d, np.float32), pi=np.asarray(self.pi, np.float32),
                            series_ids=ctx.series_ids)
        meta = {"origin": tag, "fusion": self.pp("fusion"), "positive_source": self.pp("positive_source")}
        if self.zero is not None:
            meta.update({"zero_best_iteration": self.zero.best_iteration, "zero_n_train": self.zero.n_train,
                         "zero_val_logloss_curve": self.zero.val_curve,
                         "zero_feature_importance": self.zero.feature_importance()})
        if getattr(self, "dar", None) is not None and self.dar.last_history is not None:
            meta["deepar_history"] = self.dar.last_history
        (out / f"meta_{tag}.json").write_text(json.dumps(meta, default=float))
