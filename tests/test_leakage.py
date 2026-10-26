"""THE tests that matter: perturbing anything after the forecast origin must change nothing the
models can see, and nothing they predict."""
import copy

import numpy as np
import pytest

from forecastkit.backtest.context import build_context
from forecastkit.features.tabular import TabularFeatures
from forecastkit.utils import get_model_class

QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def _ctx(ds, cfg, origin_idx, perturb=False, seed=0):
    d = copy.deepcopy(ds)
    if perturb:
        rng = np.random.default_rng(seed)
        d.Y[:, origin_idx + 1:] = rng.integers(0, 500, d.Y[:, origin_idx + 1:].shape).astype(np.float32)
    cold = d.static["cold_start"].to_numpy()
    return build_context(d, origin_idx, cfg["backtest"]["horizon"], cfg["masks"], cold, 21, QS, d.static, seed=5)


def test_context_is_identical_when_the_future_is_scrambled(ds, cfg):
    o = ds._orig["test"][0]
    a, b = _ctx(ds, cfg, o), _ctx(ds, cfg, o, perturb=True)
    for name in ("y", "y_raw", "available", "censored", "known_hist", "known_fut"):
        np.testing.assert_array_equal(np.nan_to_num(getattr(a, name), nan=-1), np.nan_to_num(getattr(b, name), nan=-1), err_msg=name)
    assert a.y.shape[1] == o + 1                       # the target stops AT the origin


@pytest.mark.parametrize("name,params", [
    ("seasonal_naive", {"n_seasons": 3}),
    ("moving_average", {"window": 21}),
    ("kalman", {"window": 200}),
    ("lgbm", {"n_estimators": 8, "train_days": 150, "anchor_stride": 14}),
    ("kalman_lgbm", {"n_estimators": 8, "train_days": 150, "anchor_stride": 14}),
    ("deepar", {"steps": 6, "hidden": 8, "context_length": 28, "batch_size": 16, "num_samples": 10}),
])
def test_forecasts_do_not_depend_on_future_targets(ds, cfg, name, params):
    o = ds._orig["test"][0]
    fa = get_model_class(name)(**params).fit(_ctx(ds, cfg, o)).predict(QS)
    fb = get_model_class(name)(**params).fit(_ctx(ds, cfg, o, perturb=True)).predict(QS)
    np.testing.assert_allclose(fa.quantiles, fb.quantiles, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fa.mean, fb.mean, rtol=1e-5, atol=1e-5)


def test_tabular_features_at_anchor_ignore_later_values():
    rng = np.random.default_rng(0)
    N, T, H, k = 4, 200, 14, 3
    y = rng.poisson(2, (N, T)).astype(float)
    y[:, 50:60] = np.nan
    K = rng.normal(size=(N, T + H, k)).astype(np.float32)
    codes = rng.integers(0, 3, (N, 2)).astype(np.float32)
    t = 120
    y2 = y.copy()
    y2[:, t + 1:] = rng.integers(0, 999, y2[:, t + 1:].shape)
    f1 = TabularFeatures(y, K, list("abc"), codes, ["s1", "s2"])
    f2 = TabularFeatures(y2, K, list("abc"), codes, ["s1", "s2"])
    s = np.repeat(np.arange(N), H)
    h = np.tile(np.arange(1, H + 1), N)
    X1, X2 = f1.rows(s, np.full(len(s), t), h), f2.rows(s, np.full(len(s), t), h)
    np.testing.assert_array_equal(np.nan_to_num(X1, nan=-9), np.nan_to_num(X2, nan=-9))


def test_cold_start_series_expose_only_recent_days(ds, cfg):
    o = ds._orig["test"][0]
    ctx = _ctx(ds, cfg, o)
    assert ctx.cold.any()
    seen = (~np.isnan(ctx.y_raw[ctx.cold])).sum(1)
    assert (seen <= 21).all()


def test_stockout_flags_in_context_use_history_only(ds, cfg):
    o = ds._orig["test"][-1]
    a, b = _ctx(ds, cfg, o), _ctx(ds, cfg, o, perturb=True)
    np.testing.assert_array_equal(a.censored, b.censored)
