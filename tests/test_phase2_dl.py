"""Phase 2: DeepAR correctness (likelihood, sampler, quantiles, masking, causality, reproducibility, validation)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)   # macOS: torch + lightgbm ship separate OpenMP runtimes; multi-threaded torch ops after
                           # lightgbm is loaded can segfault. DeepAR itself runs with num_threads=1 (default).
from scipy import stats  # noqa: E402

from forecastkit.backtest.context import build_context  # noqa: E402
from forecastkit.models.deepar import DeepARForecaster, _Net  # noqa: E402

QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
SMALL = {"steps": 30, "finetune_steps": 10, "num_samples": 40, "hidden": 8, "context_length": 28, "batch_size": 32}


def _ctx(ds, k=0, H=14, mask=None):
    o = ds._orig["test"][k]
    mask = mask or {"leading_zeros": True, "stockout": {"enabled": True, "min_run": 21, "min_nonzero_events": 8}}
    return build_context(ds, o, H, mask, ds.static["cold_start"].to_numpy(), 21, QS, ds.static, seed=5 + o)


def _net():
    return _Net(0, [], [], 4, 1, 0.0, "negbin")


def test_nb_nll_matches_scipy():
    net = _net()
    mu = torch.tensor([0.05, 0.7, 3.0, 25.0, 300.0], dtype=torch.float64)
    alpha = torch.tensor([0.01, 0.5, 1.3, 2.0, 0.2], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 4.0, 17.0, 280.0], dtype=torch.float64)
    ours = net.nll((mu, alpha), y).numpy()
    r = 1 / alpha.numpy()
    ref = -stats.nbinom.logpmf(y.numpy(), r, r / (r + mu.numpy()))
    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-8)


def test_nb_sampler_moments():
    torch.manual_seed(0)
    net = _net()
    mu, alpha = torch.full((400_000,), 3.0), torch.full((400_000,), 0.5)
    z = net.sample((mu, alpha)).numpy()
    assert z.min() >= 0 and np.all(z == np.round(z))
    assert abs(z.mean() - 3.0) < 0.03                     # E = mu
    assert abs(z.var() - (3.0 + 0.5 * 9.0)) / 7.5 < 0.02  # Var = mu + alpha mu^2
    assert abs((z == 0).mean() - DeepARForecaster.nb_p0(3.0, 0.5)) < 0.005


def test_dist_params_are_valid():
    net = _net()
    raw = torch.randn(64, 7, 2) * 20
    mu, alpha = net.dist_params(raw, torch.rand(64) * 50 + 1)
    assert torch.all(mu > 0) and torch.all(alpha > 0) and torch.isfinite(net.nll((mu, alpha), torch.zeros_like(mu))).all()


@pytest.fixture(scope="module")
def fitted(ds):
    m = DeepARForecaster(**SMALL, seasonal_lags=[7, 14], age_feature=True, validation=True, eval_every=10, patience=2,
                         optimizer="adamw", weight_decay=0.01)
    ctx = _ctx(ds)
    m.fit(ctx)
    return m, ctx, m.predict(QS)


def test_forecast_shapes_nonneg_and_noncrossing(fitted):
    m, ctx, fc = fitted
    assert fc.quantiles.shape == (ctx.N, ctx.horizon, len(QS))
    assert np.all(fc.quantiles >= 0) and np.all(np.diff(fc.quantiles, axis=-1) >= 0)
    assert m.last_samples.shape == (ctx.N, int(SMALL["num_samples"]), ctx.horizon)
    np.testing.assert_allclose(fc.quantiles[..., 3], np.sort(np.quantile(m.last_samples, QS, axis=1).transpose(1, 2, 0), -1)[..., 3])
    p0 = m.zero_probability()
    assert p0.shape == (ctx.N, ctx.horizon) and np.all((p0 >= 0) & (p0 <= 1))
    # Rao-Blackwellised P(0) agrees with the sample frequency up to Monte-Carlo error
    assert abs(p0.mean() - (m.last_samples == 0).mean()) < 0.03


def test_validation_and_early_stopping_record_history(fitted):
    m, ctx, _ = fitted
    hst = m.last_history
    assert hst["val_step"] and hst["best_step"] is not None and len(hst["val_loss"]) == len(hst["val_step"])
    assert hst["best_val"] == pytest.approx(min(hst["val_loss"]))
    assert hst["stopped_step"] <= SMALL["steps"]


def test_training_windows_exclude_validation_block(ds):
    """With validation on, no training window may contain a target in the last H days of history."""
    m = DeepARForecaster(**SMALL, validation=True)
    ctx = _ctx(ds)
    m._prepare(ctx)
    g = torch.Generator().manual_seed(0)
    L = SMALL["context_length"] + ctx.horizon
    for _ in range(20):
        _i, pos, _t = m._batch(32, SMALL["context_length"], L, g, hold_out=ctx.horizon)
        assert int(pos.max()) <= ctx.Th - 1 - ctx.horizon


def test_masked_targets_are_excluded_from_loss(ds):
    """A NaN (censored / unavailable) target contributes nothing: the loss count drops by one per masked cell and
    the masked loss equals the full loss minus exactly that cell's NLL."""
    m = DeepARForecaster(**SMALL)
    ctx = _ctx(ds)
    m._prepare(ctx)
    torch.manual_seed(0)
    net = m._build(ctx)
    net.eval()
    i = torch.arange(4)
    pos = torch.arange(100, 142)[None, :].expand(4, -1)
    Ypad = torch.cat([torch.full((m.Y.shape[0], 1), float("nan")), m.Y], 1)
    tgt = torch.nan_to_num(m.Y[i[:, None].expand_as(pos), pos].clone(), nan=1.0)
    s_full, c_full = m._window_loss(net, i, pos, tgt, 28, Ypad, m.cats[i])
    s_last, c_last = m._window_loss(net, i, pos, tgt, 28, Ypad, m.cats[i], loss_from=41)   # only column 41
    tgt2 = tgt.clone()
    tgt2[:, 41] = float("nan")
    s_m, c_m = m._window_loss(net, i, pos, tgt2, 28, Ypad, m.cats[i])
    assert int(c_full - c_m) == 4 and torch.isfinite(s_m)
    assert float(s_m) == pytest.approx(float(s_full - s_last), rel=1e-5)


def test_deepar_is_causal_in_the_future(ds):
    """Changing data AFTER the origin must not change the forecast (lags and age included)."""
    import copy
    params = {**SMALL, "seasonal_lags": [7, 14, 28], "age_feature": True, "validation": True, "eval_every": 10}
    o = ds._orig["test"][0]
    d2 = copy.deepcopy(ds)
    d2.Y[:, o + 1:] = d2.Y[:, o + 1:] * 5 + 3
    out = []
    for d in (ds, d2):
        ctx = build_context(d, o, 14, {"leading_zeros": True}, d.static["cold_start"].to_numpy(), 21, QS, d.static, seed=9)
        out.append(DeepARForecaster(**params).fit(ctx).predict(QS).quantiles)
    np.testing.assert_array_equal(out[0], out[1])


def test_same_seed_is_reproducible(ds):
    ctx = _ctx(ds)
    a = DeepARForecaster(**SMALL, seasonal_lags=[7]).fit(ctx).predict(QS).quantiles
    b = DeepARForecaster(**SMALL, seasonal_lags=[7]).fit(ctx).predict(QS).quantiles
    np.testing.assert_array_equal(a, b)


def test_age_feature_is_causal_and_monotone():
    av = np.zeros((2, 10), bool)
    av[0, 3:] = True
    av[1, :] = True
    hist, fut = DeepARForecaster.age_arrays(av, 4)
    assert hist[0, :4].tolist() == [0, 0, 0, 0] and np.all(np.diff(np.concatenate([hist, fut], 1)[0, 3:]) > 0)
    assert fut.shape == (2, 4)
