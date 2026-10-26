import numpy as np

from forecastkit.evaluation import metrics as M


def test_point_metrics_toy():
    y, f = np.array([0., 2., 4.]), np.array([1., 2., 2.])
    assert M.mae(y, f) == (1 + 0 + 2) / 3
    assert np.isclose(M.rmse(y, f), np.sqrt((1 + 0 + 4) / 3))
    assert np.isclose(M.wape(y, f), 3 / 6)


def test_wape_is_aggregated_not_mean_of_ratios():
    # series A tiny volume with big relative error, series B big volume with small relative error
    yA, fA = np.array([1.0]), np.array([3.0])      # ratio 2.0
    yB, fB = np.array([100.0]), np.array([101.0])  # ratio 0.01
    agg = M.wape(np.r_[yA, yB], np.r_[fA, fB])
    mean_of_ratios = (M.wape(yA, fA) + M.wape(yB, fB)) / 2
    assert np.isclose(agg, 3 / 101)
    assert not np.isclose(agg, mean_of_ratios)


def test_pinball_known_values():
    y = np.array([10.0])
    assert np.isclose(M.pinball(y, np.array([8.0]), 0.9), 0.9 * 2)    # under-forecast penalised by q
    assert np.isclose(M.pinball(y, np.array([12.0]), 0.9), 0.1 * 2)   # over-forecast penalised by 1-q
    assert np.isclose(M.pinball(y, np.array([10.0]), 0.5), 0.0)


def test_pinball_at_median_is_half_mae():
    rng = np.random.default_rng(0)
    y, f = rng.poisson(3, 500).astype(float), rng.poisson(3, 500).astype(float)
    assert np.isclose(M.pinball(y, f, 0.5), 0.5 * M.mae(y, f))


def test_coverage_and_width():
    y = np.array([1., 5., 9.])
    lo, hi = np.array([0., 0., 0.]), np.array([2., 6., 8.])
    assert np.isclose(M.coverage(y, lo, hi), 2 / 3)
    assert np.isclose(M.width(lo, hi), (2 + 6 + 8) / 3)


def test_perfect_quantile_forecast_has_zero_loss():
    y = np.array([1., 2., 3.])
    F = np.stack([y, y, y], axis=-1)
    assert M.mean_pinball(y, F, [0.1, 0.5, 0.9]) == 0.0
    assert M.wql(y, F, [0.1, 0.5, 0.9]) == 0.0
