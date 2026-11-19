"""Phase 2 evaluation additions: randomised PIT and the series-level paired bootstrap."""
import numpy as np

from forecastkit.evaluation.metrics import pit_histogram, randomized_pit


def test_randomized_pit_uniform_for_correct_count_forecast():
    rng = np.random.default_rng(0)
    mu = rng.gamma(1.0, 2.0, size=20_000)
    y = rng.poisson(mu)
    samples = rng.poisson(mu[:, None], size=(20_000, 400))
    u = randomized_pit(y, samples, rng)
    _, dev = pit_histogram(u)
    assert dev < 0.06 and 0.48 < u.mean() < 0.52


def test_randomized_pit_detects_overdispersed_forecast():
    rng = np.random.default_rng(1)
    y = rng.poisson(3.0, 20_000)
    too_wide = rng.negative_binomial(0.5, 0.5 / 3.5, size=(20_000, 400))      # mean 3, far too much spread
    dens, dev = pit_histogram(randomized_pit(y, too_wide, rng))
    assert dev > 0.2 and dens[0] < 1 and dens[-1] < 1                          # hump-shaped = over-dispersed


def test_paired_bootstrap_sign_and_identity(cfg_copy, tmp_path):
    from forecastkit.backtest.runner import run_backtest
    from forecastkit.evaluation.evaluator import Evaluator
    cfg_copy["models"] = {"snaive": {"type": "seasonal_naive", "params": {"n_seasons": 2}},
                          "naive": {"type": "naive", "params": {"method": "naive"}}}
    run_backtest(cfg_copy, out_dir=tmp_path)
    ev = Evaluator(cfg_copy, tmp_path)
    same = ev.paired_bootstrap("snaive", "snaive", "WAPE", n_boot=200)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0 and not same["significant"]
    ab = ev.paired_bootstrap("snaive", "naive", "pinball", n_boot=300)
    ba = ev.paired_bootstrap("naive", "snaive", "pinball", n_boot=300)
    assert np.isclose(ab["diff"], -ba["diff"]) and ab["ci_low"] <= ab["diff"] <= ab["ci_high"]
    t = ev.main_table()
    assert np.isclose(ab["diff"], t.loc["snaive", "pinball"] - t.loc["naive", "pinball"])
    rep = ev.paired_bootstrap(["snaive", "snaive"], "naive", "pinball", n_boot=300)
    assert np.isclose(rep["diff"], ab["diff"])                               # replicate averaging is neutral
