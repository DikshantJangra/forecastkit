"""Zero-run persistence model: pi_h = P(Y_{t+h} = 0 | F_t), the probability that RECORDED sales are zero h days
after the origin t, for h = 1..H.

Target.  Z_{i,t+h} = 1[y_{i,t+h} = 0] on days the series is available (M5: a price exists). This is observable,
needs no inventory labels and is exactly the zero mass a predictive distribution of recorded sales must get
right. It is NOT a stockout probability.

Learner.  One LightGBM binary classifier (log loss, a proper scoring rule) over (series, anchor t, horizon h)
rows; h is a feature. Features: causal zero-run structure (features.py), the leak-safe lag / rolling /
recency features of the LightGBM forecaster, known-future calendar / price covariates at the target date,
and static hierarchy codes.

Training protocol (history only).  Let T0 be the origin. Validation rows: anchor T0 - H, h = 1..H (targets in
the last H days of history - the same shape as the real forecast task). Training rows: anchors with
t + h <= T0 - H. Early stopping on validation log loss picks the number of trees. Test windows are never seen.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from forecastkit.features.tabular import TabularFeatures
from forecastkit.utils import log

from .features import PERSISTENCE_FEATURES, zero_run_features

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise ImportError("lightgbm is required for the zero-run model: pip install lightgbm") from e


class ZeroRunModel:
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "train_days": 364, "anchor_stride": 7, "max_rows": 600_000, "early_stopping": 50, "num_threads": 4,
        "persistence_features": True, "use_item_id": False, "include_covariates": None, "exclude_covariates": None,
    }

    def __init__(self, seed: int = 0, **params):
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise KeyError(f"unknown zero-run model params: {sorted(unknown)}")
        self.params = {**self.DEFAULTS, **params}
        self.seed = int(seed)

    # ------------------------------------------------------------------ features
    def _features(self, ctx):
        y = ctx.y_raw                                      # recorded sales; NaN = unavailable (NOT censored)
        cov = ctx.covariate_idx(self.params["include_covariates"], self.params["exclude_covariates"])
        K_all = np.concatenate([ctx.known_hist, ctx.known_fut], axis=1)[:, :, cov]
        cols = list(ctx.meta.get("static_cat_cols") or [])
        if self.params["use_item_id"] and ctx.meta.get("item_col"):
            cols.append(ctx.meta["item_col"])
        codes = np.zeros((ctx.N, len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            codes[:, j] = ctx.static[c].astype("category").cat.codes.to_numpy()
        feat = TabularFeatures(y, K_all, [ctx.known_names[i] for i in cov], codes, cols, extra=zero_run_features(y))
        names = feat.feature_names
        drop = set(PERSISTENCE_FEATURES) if not self.params["persistence_features"] else set()
        self.keep = np.array([n not in drop for n in names])
        self.feature_names = [n for n in names if n not in drop]
        self.cat_idx = [self.feature_names.index(f"s_{c}") for c in cols]
        return feat

    def _rows(self, s, t, h):
        return self.feat.rows(s, t, h)[:, self.keep]

    def _lgb_params(self):
        p = self.params
        return {"objective": "binary", "learning_rate": p["learning_rate"], "num_leaves": int(p["num_leaves"]),
                "min_child_samples": int(p["min_child_samples"]), "feature_fraction": p["feature_fraction"],
                "bagging_fraction": p["bagging_fraction"], "bagging_freq": int(p["bagging_freq"]),
                "lambda_l2": p["lambda_l2"], "num_threads": int(p["num_threads"]), "verbose": -1,
                "seed": self.seed, "deterministic": True, "force_row_wise": True}

    # ------------------------------------------------------------------ fit / predict
    def fit(self, ctx):
        self.ctx = ctx
        H, Th = ctx.horizon, ctx.Th
        T0 = Th - 1
        self.feat = self._features(ctx)
        y = ctx.y_raw
        N = ctx.N
        rng = np.random.default_rng(self.seed)
        # validation: the most recent forecast-shaped block inside history
        tv = T0 - H
        sv = np.repeat(np.arange(N), H)
        hv = np.tile(np.arange(1, H + 1), N)
        lv = y[sv, tv + hv]
        okv = ~np.isnan(lv)
        Xv, Zv = self._rows(sv[okv], np.full(okv.sum(), tv), hv[okv]), (lv[okv] == 0).astype(np.float32)
        # training: anchors whose targets end before the validation block
        stride, days = int(self.params["anchor_stride"]), int(self.params["train_days"])
        last_target = T0 - H
        anchors = np.arange(last_target - 1, max(last_target - days, 56), -stride)
        s_l, t_l, h_l = [], [], []
        for h in range(1, H + 1):
            a = anchors[anchors + h <= last_target]
            s_l.append(np.repeat(np.arange(N), len(a)))
            t_l.append(np.tile(a, N))
            h_l.append(np.full(N * len(a), h))
        s, t, h = np.concatenate(s_l), np.concatenate(t_l), np.concatenate(h_l)
        lab = y[s, t + h]
        ok = ~np.isnan(lab)
        s, t, h, lab = s[ok], t[ok], h[ok], lab[ok]
        cap = int(self.params["max_rows"])
        if len(lab) > cap:
            sel = rng.choice(len(lab), cap, replace=False)
            s, t, h, lab = s[sel], t[sel], h[sel], lab[sel]
        X, Z = self._rows(s, t, h), (lab == 0).astype(np.float32)
        kw = {"categorical_feature": self.cat_idx, "free_raw_data": False}
        dtr = lgb.Dataset(X, Z, **kw)
        dva = lgb.Dataset(Xv, Zv, reference=dtr, **kw)
        ev: dict = {}
        self.booster = lgb.train(self._lgb_params(), dtr, int(self.params["n_estimators"]), valid_sets=[dva],
                                 valid_names=["val"],
                                 callbacks=[lgb.early_stopping(int(self.params["early_stopping"]), verbose=False),
                                            lgb.record_evaluation(ev)])
        self.best_iteration = int(self.booster.best_iteration or self.booster.current_iteration())
        self.val_curve = [float(v) for v in ev.get("val", {}).get("binary_logloss", [])]
        self.n_train, self.n_val, self.train_zero_rate = len(Z), len(Zv), float(Z.mean()) if len(Z) else np.nan
        log.debug("zero-run model: %d train rows, %d val rows, best iter %d, val logloss %.4f", len(Z), len(Zv),
                  self.best_iteration, min(self.val_curve) if self.val_curve else np.nan)
        return self

    def predict_proba(self) -> np.ndarray:
        """[N,H] pi_h at the origin."""
        ctx = self.ctx
        N, H, T0 = ctx.N, ctx.horizon, ctx.Th - 1
        out = np.zeros((N, H), dtype=np.float64)
        s = np.arange(N)
        for h in range(1, H + 1):
            out[:, h - 1] = self.booster.predict(self._rows(s, np.full(N, T0), np.full(N, h)),
                                                 num_iteration=self.best_iteration)
        return np.clip(out, 1e-6, 1 - 1e-6)

    def feature_importance(self, top=20) -> dict:
        imp = self.booster.feature_importance("gain", iteration=self.best_iteration)
        tot = float(imp.sum()) or 1.0
        return dict(sorted(((n, float(v) / tot) for n, v in zip(self.feature_names, imp)), key=lambda kv: -kv[1])[:top])
