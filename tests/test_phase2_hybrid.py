"""Phase 2: zero-run features (causality), baselines, zero model, hurdle fusion (validity), hybrid integration."""
import copy

import numpy as np
import pytest

from forecastkit.backtest.context import build_context
from forecastkit.zerorun import KMRenewal, binary_scores, zero_run_features
from forecastkit.zerorun.baselines import kaplan_meier
from forecastkit.zerorun.features import completed_run_stats, current_run_length

pytest.importorskip("lightgbm")
QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def test_run_features_on_toy_series():
    y = np.array([[np.nan, 0, 2, 0, 0, 0, 1, 0, 0, 3, 0]], float)
    r = current_run_length(y)[0]
    assert np.isnan(r[0]) and r.tolist()[2:] == [0, 1, 2, 3, 0, 1, 2, 0, 1]
    st = completed_run_stats(y, long_run=3)
    assert st["zr_runs_done"][0].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
    assert st["zr_max_run"][0, -1] == 3 and st["zr_mean_run"][0, -1] == 2.5 and st["zr_long_runs"][0, -1] == 1


def test_zero_run_features_are_causal():
    """Perturbing days > t must leave every feature on days <= t unchanged."""
    rng = np.random.default_rng(0)
    Y = rng.poisson(0.4, size=(30, 300)).astype(float)
    Y[:5, 100:200] = 0
    Y[3, :20] = np.nan
    t = 150
    Y2 = Y.copy()
    Y2[:, t + 1:] = rng.poisson(3.0, size=(30, 300 - t - 1))
    f1, f2 = zero_run_features(Y), zero_run_features(Y2)
    for k in f1:
        np.testing.assert_array_equal(np.nan_to_num(f1[k][:, :t + 1], nan=-7), np.nan_to_num(f2[k][:, :t + 1], nan=-7),
                                      err_msg=k)


def test_kaplan_meier_handles_censoring():
    d = np.array([1, 2, 2, 3, 5])
    obs = np.array([True, True, False, True, False])
    S = kaplan_meier(d, obs, 6)
    # at risk: d1:5 (1 death), d2:4 (1 death; 1 censored), d3:2 (1 death), d5:1 (censored)
    np.testing.assert_allclose(S[:4], [1, 4 / 5, 4 / 5 * 3 / 4, 4 / 5 * 3 / 4 * 1 / 2])
    assert S[6] == pytest.approx(S[3])


def test_km_renewal_probabilities_valid():
    rng = np.random.default_rng(1)
    Y = (rng.random((20, 200)) < 0.3) * rng.poisson(2, (20, 200)).astype(float)
    Y[0, -60:] = 0
    p = KMRenewal().fit(Y).predict(28)
    assert p.shape == (20, 28) and np.all((p > 0) & (p < 1))
    assert np.all(np.diff(p[0]) <= 1e-12)                 # run-continuation probability can only decay with h


def test_binary_scores_perfect_and_constant():
    z = np.array([0, 1, 1, 0, 1], float)
    s = binary_scores(np.clip(z, 1e-6, 1 - 1e-6), z)
    assert s["brier"] < 1e-10 and s["roc_auc"] == 1.0
    c = binary_scores(np.full(5, 0.6), z)
    assert c["brier"] == pytest.approx(0.24) and c["ece"] == pytest.approx(0.0)


# ---------------------------------------------------------------- fusion maths
def test_zero_mass_variants():
    from forecastkit.models.hybrid import zero_mass
    pi, p0 = np.array([[0.7, 0.2]]), np.array([[0.4, 0.5]])
    np.testing.assert_allclose(zero_mass("hurdle", pi, p0), pi)
    np.testing.assert_allclose(zero_mass("zi_naive", pi, p0), [[0.82, 0.6]])
    np.testing.assert_allclose(zero_mass("zi_inflate", pi, p0), [[0.7, 0.5]])
    np.testing.assert_allclose(zero_mass("gated", pi, p0, np.array([False])), p0)
    np.testing.assert_allclose(zero_mass("none", pi, p0), p0)


def test_truncated_mixture_sampler_is_exact():
    """Positive draws follow p_D(y | y>0) for a 2-path NB mixture; p0 and E[Y|Y>0] match closed forms."""
    from scipy.stats import nbinom

    from forecastkit.models.hybrid import truncated_nb_mixture_samples
    mu = np.array([0.3, 4.0])[None, :, None]
    al = np.array([1.5, 0.4])[None, :, None]
    s, p0, mpos = truncated_nb_mixture_samples(mu, al, 200_000, np.random.default_rng(0))
    r = 1 / al[0, :, 0]
    p = r / (r + mu[0, :, 0])
    f0 = nbinom.pmf(0, r, p)
    assert p0[0, 0] == pytest.approx(f0.mean(), rel=1e-9)
    assert mpos[0, 0] == pytest.approx(mu[0, :, 0].sum() / (1 - f0).sum(), rel=1e-9)
    assert s.min() >= 1
    for k in (1, 2, 5):
        exact = (nbinom.pmf(k, r, p)).mean() / (1 - f0.mean())
        assert (s == k).mean() == pytest.approx(exact, abs=0.004)


def test_hurdle_quantiles_valid_and_consistent():
    from forecastkit.models.hybrid import hurdle_quantiles
    rng = np.random.default_rng(0)
    pos = np.sort(rng.integers(1, 20, size=(3, 2, 500)).astype(float), -1)
    q0 = np.array([[0.0, 0.3], [0.6, 0.95], [0.5, 0.1]])
    Q = hurdle_quantiles(q0, pos, QS)
    assert np.all(Q >= 0) and np.all(np.diff(Q, axis=-1) >= -1e-12)
    for i in range(3):
        for h in range(2):
            for j, tau in enumerate(QS):
                if tau <= q0[i, h]:
                    assert Q[i, h, j] == 0
                else:
                    assert Q[i, h, j] >= 1
    # Monte-Carlo check of the mixture CDF at a quantile: P(Y <= Q(tau)) >= tau
    i, h = 0, 1
    draws = np.where(rng.random(200_000) < q0[i, h], 0, rng.choice(pos[i, h], 200_000))
    for j, tau in enumerate(QS):
        assert (draws <= Q[i, h, j]).mean() >= tau - 0.01


# ---------------------------------------------------------------- zero model + hybrid integration
def _ctx(ds, d=None, o=None):
    d = d or ds
    o = ds._orig["test"][0] if o is None else o
    m = {"leading_zeros": True, "stockout": {"enabled": True, "min_run": 21, "min_nonzero_events": 8}}
    return build_context(d, o, 14, m, d.static["cold_start"].to_numpy(), 21, QS, d.static, seed=3)


def test_zero_model_probabilities_and_no_future_leak(ds):
    from forecastkit.zerorun.model import ZeroRunModel
    ctx = _ctx(ds)
    zm = ZeroRunModel(seed=0, n_estimators=40, num_threads=1).fit(ctx)
    p = zm.predict_proba()
    assert p.shape == (ctx.N, ctx.horizon) and np.all((p > 0) & (p < 1))
    d2 = copy.deepcopy(ds)
    o = ds._orig["test"][0]
    d2.Y[:, o + 1:] = 0                                    # wipe the future: must not matter
    p2 = ZeroRunModel(seed=0, n_estimators=40, num_threads=1).fit(_ctx(ds, d2, o)).predict_proba()
    np.testing.assert_array_equal(p, p2)
    ZeroRunModel(seed=0, n_estimators=10, num_threads=1, persistence_features=False).fit(ctx)
    assert not ({"zr_run", "days_since_sale"} & set(ZeroRunModel(seed=0, n_estimators=10, num_threads=1,
                                                                 persistence_features=False).fit(ctx).feature_names))


@pytest.mark.parametrize("fusion", ["hurdle", "zi_naive", "zi_inflate", "gated", "none"])
def test_hybrid_end_to_end_valid_distribution(ds, fusion):
    from forecastkit.models.hybrid import DeepARHurdle
    dar = {"steps": 20, "num_samples": 30, "hidden": 8, "context_length": 28, "batch_size": 32}
    m = DeepARHurdle(deepar=dar, zero_model={"n_estimators": 20, "num_threads": 1}, fusion=fusion, positive_samples=60)
    ctx = _ctx(ds)
    fc = m.fit(ctx).predict(QS)
    Q = fc.quantiles
    assert Q.shape == (ctx.N, ctx.horizon, len(QS)) and np.all(Q >= 0) and np.all(np.diff(Q, axis=-1) >= 0)
    assert np.all((m.q0 >= 0) & (m.q0 <= 1)) and np.all(fc.mean >= 0)
    # quantiles below the zero mass are exactly zero; implied P(Y=0) of the samples ~ q0
    for j, tau in enumerate(QS):
        assert np.all(Q[..., j][m.q0 >= tau + 1e-9] == 0)
    assert abs((m.last_samples == 0).mean() - m.q0.mean()) < 0.03


def test_hybrid_deepar_component_equals_standalone_deepar(ds):
    from forecastkit.models.deepar import DeepARForecaster
    from forecastkit.models.hybrid import DeepARHurdle
    dar = {"steps": 20, "num_samples": 30, "hidden": 8, "context_length": 28, "batch_size": 32}
    ctx = _ctx(ds)
    a = DeepARForecaster(**dar).fit(ctx)
    a.predict(QS)
    h = DeepARHurdle(deepar=dar, zero_model={"n_estimators": 10, "num_threads": 1}).fit(ctx)
    h.predict(QS)
    np.testing.assert_array_equal(a.last_samples, h.dar.last_samples)


def test_hybrid_without_deepar_component(ds):
    from forecastkit.models.hybrid import DeepARHurdle
    m = DeepARHurdle(zero_model={"n_estimators": 20, "num_threads": 1}, positive_source="empirical")
    fc = m.fit(_ctx(ds)).predict(QS)
    assert np.all(fc.quantiles >= 0) and not hasattr(m, "dar")


def test_pipeline_runs_hybrid_and_writes_artifacts(cfg_copy, tmp_path):
    from forecastkit.backtest.runner import run_backtest
    from forecastkit.evaluation.evaluator import Evaluator
    cfg_copy["models"] = {"hyb": {"type": "deepar_hurdle", "params": {
        "deepar": {"steps": 15, "finetune_steps": 5, "num_samples": 20, "hidden": 8, "context_length": 28},
        "zero_model": {"n_estimators": 15, "num_threads": 1}}}}
    run_dir = tmp_path / "run"
    run_backtest(cfg_copy, out_dir=run_dir)
    ev = Evaluator(cfg_copy, run_dir)
    t = ev.main_table()
    assert "hyb" in t.index and np.isfinite(t.loc["hyb", "WAPE"])
    arts = sorted((run_dir / "artifacts" / "hyb").glob("pred_*.npz"))
    assert len(arts) == 2
    z = np.load(arts[0], allow_pickle=True)
    assert {"samples", "p0", "p0_deepar", "pi"} <= set(z.files)
