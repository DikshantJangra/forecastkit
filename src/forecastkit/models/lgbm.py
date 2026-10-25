"""Gradient-boosted lag-feature model (the strong tabular 'AML' baseline) with DIRECT multi-horizon
forecasting by horizon buckets (default 1-7, 8-14, 15-21, 22-28), plus quantile objectives so it is
scored with pinball loss exactly like the probabilistic models (otherwise the comparison is unfair).

  * mean model per bucket: Poisson/Tweedie objective (count / intermittent demand)
  * one quantile model per (bucket, quantile)
  * censored / unavailable targets get weight 0 (rows dropped)
  * `KalmanLGBMForecaster` = the 'hybrid': Kalman filtered state / uncertainty fed in as features
"""
from __future__ import annotations

import numpy as np

from forecastkit.features.tabular import TabularFeatures
from forecastkit.schema import QuantileForecast
from forecastkit.utils import log, register_model
from .base import BaseForecaster

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise ImportError("lightgbm is required for the lgbm model: pip install lightgbm") from e


def _buckets(H, size):
    return [(lo, min(lo + size - 1, H)) for lo in range(1, H + 1, size)]


@register_model("lgbm")
class LGBMForecaster(BaseForecaster):
    DEFAULTS = dict(bucket_size=7, train_days=364, anchor_stride=7, objective="poisson", n_estimators=150,
                    learning_rate=0.06, num_leaves=63, min_child_samples=80, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, tweedie_variance_power=1.3,
                    use_item_id=False, use_mask=True, include_covariates=None, exclude_covariates=None,
                    num_threads=2, max_rows=600_000)

    def pp(self, k):
        return self.params.get(k, self.DEFAULTS[k])

    # --- hooks for subclasses --------------------------------------------------
    def extra_features(self, ctx, y):
        """Subclass hook -> (dict name -> [N,Th] causal day features, optional kalman-state dict)."""
        return {}, None

    # ---------------------------------------------------------------------------
    def _static(self, ctx):
        cols = list(ctx.meta.get("static_cat_cols") or [])
        if self.pp("use_item_id") and ctx.meta.get("item_col"):
            cols.append(ctx.meta["item_col"])
        codes = np.zeros((ctx.N, len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            codes[:, j] = ctx.static[c].astype("category").cat.codes.to_numpy()
        if self.pp("use_item_id") and ctx.meta.get("item_col"):
            codes[ctx.cold, -1] = -1  # cold-start items have no usable identity
        return codes, cols

    def _lgb_params(self, **over):
        p = dict(learning_rate=self.pp("learning_rate"), num_leaves=int(self.pp("num_leaves")),
                 min_child_samples=int(self.pp("min_child_samples")), feature_fraction=self.pp("feature_fraction"),
                 bagging_fraction=self.pp("bagging_fraction"), bagging_freq=int(self.pp("bagging_freq")),
                 lambda_l2=self.pp("lambda_l2"), num_threads=int(self.pp("num_threads")), verbose=-1,
                 seed=int(self.ctx.seed), deterministic=True, force_row_wise=True)
        p.update(over)
        return p

    def fit(self, ctx):
        self.ctx = ctx
        H, Th = ctx.horizon, ctx.Th
        y = ctx.target(self.pp("use_mask"))
        cov = ctx.covariate_idx(self.pp("include_covariates"), self.pp("exclude_covariates"))
        K_all = np.concatenate([ctx.known_hist, ctx.known_fut], axis=1)[:, :, cov]
        self.cov_names = [ctx.known_names[i] for i in cov]
        codes, scols = self._static(ctx)
        extra, kf_state = self.extra_features(ctx, y)
        self.feat = TabularFeatures(y, K_all, self.cov_names, codes, scols, extra=extra, kf_state=kf_state)
        self.cat_idx = [self.feat.feature_names.index(f"s_{c}") for c in scols]
        rng = np.random.default_rng(ctx.seed)
        qs = list(ctx.quantiles)
        self.models = {}
        T0 = Th - 1
        for (lo, hi) in _buckets(H, int(self.pp("bucket_size"))):
            X, yt = self._training_rows(y, T0, lo, hi, rng)
            if len(yt) < 500:
                self.models[(lo, hi)] = None
                continue
            ds_kw = dict(categorical_feature=self.cat_idx, free_raw_data=False)
            nround = int(self.pp("n_estimators"))
            obj = self.pp("objective")
            extra = {"tweedie_variance_power": self.pp("tweedie_variance_power")} if obj == "tweedie" else {}
            mean_m = lgb.train(self._lgb_params(objective=obj, **extra), lgb.Dataset(X, yt, **ds_kw), nround)
            qm = {}
            for q in qs:
                qm[q] = lgb.train(self._lgb_params(objective="quantile", alpha=float(q)),
                                  lgb.Dataset(X, yt, **ds_kw), nround)
            self.models[(lo, hi)] = (mean_m, qm)
        log.debug("lgbm fitted %d buckets", len(self.models))
        return self

    def _training_rows(self, y, T0, lo, hi, rng):
        stride, days = int(self.pp("anchor_stride")), int(self.pp("train_days"))
        first_anchor = max(T0 - days, 112 // 2)
        anchors = np.arange(T0 - lo, first_anchor, -stride)  # most recent first; t+h<=T0 enforced below
        N = y.shape[0]
        s_idx, t_idx, h_idx = [], [], []
        for h in range(lo, hi + 1):
            a = anchors[anchors + h <= T0]
            s_idx.append(np.repeat(np.arange(N), len(a)))
            t_idx.append(np.tile(a, N))
            h_idx.append(np.full(N * len(a), h))
        s, t, h = np.concatenate(s_idx), np.concatenate(t_idx), np.concatenate(h_idx)
        target = y[s, t + h]
        keep = ~np.isnan(target)                      # censored / unavailable targets: weight 0 == dropped
        s, t, h, target = s[keep], t[keep], h[keep], target[keep]
        cap = int(self.pp("max_rows"))
        if len(target) > cap:
            sel = rng.choice(len(target), cap, replace=False)
            s, t, h, target = s[sel], t[sel], h[sel], target[sel]
        return self.feat.rows(s, t, h), target.astype(np.float32)

    def predict(self, quantiles):
        ctx = self.ctx
        qs, H, N = list(quantiles), ctx.horizon, ctx.N
        T0 = ctx.Th - 1
        Q = np.zeros((N, H, len(qs)), dtype=np.float32)
        M = np.zeros((N, H), dtype=np.float32)
        s_all = np.arange(N)
        for (lo, hi), mm in self.models.items():
            for h in range(lo, hi + 1):
                X = self.feat.rows(s_all, np.full(N, T0), np.full(N, h))
                if mm is None:
                    Q[:, h - 1, :], M[:, h - 1] = 0.0, 0.0
                    continue
                mean_m, qm = mm
                M[:, h - 1] = mean_m.predict(X)
                for j, q in enumerate(qs):
                    Q[:, h - 1, j] = qm[q].predict(X)
        return self.finalize(Q, mean=M)

    def feature_importance(self, top=15):
        rows = {}
        for (lo, hi), mm in self.models.items():
            if mm is None:
                continue
            imp = mm[0].feature_importance("gain")
            for n, v in zip(self.feat.feature_names, imp):
                rows[n] = rows.get(n, 0.0) + float(v)
        tot = sum(rows.values()) or 1.0
        return dict(sorted(((k, v / tot) for k, v in rows.items()), key=lambda kv: -kv[1])[:top])


@register_model("kalman_lgbm")
class KalmanLGBMForecaster(LGBMForecaster):
    """HYBRID: the local state estimate of a per-SKU Kalman filter (filtered level, one-step predictive sd and
    the seasonal-aware h-step mean Z T^h a_t) is added as features to the global gradient-boosted model:
    local adaptivity + cross-series learning in one model.  Filtered states at anchor t use y<=t only."""

    def extra_features(self, ctx, y):
        from .kalman import TRANSFORMS, KalmanCore
        tf, inv = TRANSFORMS[self.p("kf_transform", "sqrt")]
        span = int(self.pp("train_days")) + 150
        yt = tf(y)
        core = KalmanCore(season=ctx.season, param_mode=self.p("kf_param_mode", "shared"))
        w = yt[:, -span:]
        q_l, q_s = core.choose_params(w)
        r = core.run(w, q_l, q_s, store=True)
        N, Th = y.shape
        off = Th - w.shape[1]
        st = np.full((N, Th, core.ss.d), np.nan, dtype=np.float32)
        st[:, off:] = r["states"]
        sd1 = np.full((N, Th), np.nan, dtype=np.float32)
        sd1[:, off:] = r["sd1"] * np.sqrt(r["sigma2"])[:, None]
        hmax = ctx.horizon + 1
        Wh = np.stack([core.ss.Z @ np.linalg.matrix_power(core.ss.T, k) for k in range(hmax + 1)])
        return ({"kf_level": inv(st[:, :, 0]), "kf_sd1": sd1}, {"st": st, "Wh": Wh, "inv": inv})
