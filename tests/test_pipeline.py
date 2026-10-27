
import numpy as np
import pandas as pd
import pytest

from forecastkit.backtest import run_backtest
from forecastkit.backtest.context import build_context
from forecastkit.datasets import load_dataset, make_sample_csv
from forecastkit.evaluation import Evaluator
from forecastkit.evaluation.ensemble import add_ensembles
from forecastkit.utils import get_model_class

QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

FAST = {
    "naive": {"method": "naive"}, "seasonal_naive": {}, "moving_average": {"window": 14}, "kalman": {"window": 150},
    "lgbm": {"n_estimators": 6, "train_days": 120, "anchor_stride": 14},
    "kalman_lgbm": {"n_estimators": 6, "train_days": 120, "anchor_stride": 14},
    "deepar": {"steps": 5, "hidden": 8, "context_length": 28, "batch_size": 16, "num_samples": 12},
}


@pytest.mark.parametrize("name", list(FAST))
def test_model_interface_contract(ds, cfg, name):
    o = ds._orig["test"][0]
    ctx = build_context(ds, o, 14, cfg["masks"], ds.static["cold_start"].to_numpy(), 21, QS, ds.static, seed=1)
    m = get_model_class(name)(**FAST[name]).fit(ctx)
    fc = m.predict(QS)
    assert fc.quantiles.shape == (ds.N, 14, len(QS))
    assert np.isfinite(fc.quantiles).all() and (fc.quantiles >= 0).all()
    assert (np.diff(fc.quantiles, axis=-1) >= -1e-6).all(), "quantiles must not cross"
    assert fc.mean is None or (fc.mean.shape == (ds.N, 14) and np.isfinite(fc.mean).all())


def test_custom_model_plugs_in_by_import_path(cfg_copy):
    cfg = cfg_copy
    cfg["models"] = {"mine": {"type": "helpers:ConstModel"}}
    fc = run_backtest(cfg, save=False)
    assert set(fc.model.astype(str)) == {"mine"}
    assert {"model", "series_id", "origin", "h", "ds", "stat", "value"} <= set(fc.columns)


def test_forecast_file_schema_and_evaluator_roundtrip(cfg_copy, tmp_path):
    cfg = cfg_copy
    run_dir = tmp_path / "run"
    fc = run_backtest(cfg, out_dir=run_dir)
    assert (run_dir / "forecasts.parquet").exists() and (run_dir / "config.yaml").exists()
    ev = Evaluator(cfg, run_dir)
    t = ev.main_table()
    assert set(t.index) == {"snaive", "kalman"}
    # recompute MAE by hand from the long file
    d = fc[fc.stat.astype(str) == "q0.5"].copy()
    ds_ = ev.ds
    truth = pd.DataFrame({"series_id": np.repeat(ds_.series_ids, ds_.T), "ds": np.tile(ds_.dates, ds_.N),
                          "y": ds_.Y.reshape(-1), "av": ds_.avail.reshape(-1)})
    d = d.merge(truth, on=["series_id", "ds"])
    d = d[d.av & ~ev.cold[pd.Series(range(ds_.N), index=ds_.series_ids).reindex(d.series_id).to_numpy()]]
    for m in ("snaive", "kalman"):
        x = d[d.model.astype(str) == m]
        assert np.isclose((x.value - x.y).abs().mean(), t.loc[m, "MAE"], rtol=1e-4)


def test_perfect_forecast_scores_zero(cfg_copy, tmp_path):
    cfg = cfg_copy
    run_dir = tmp_path / "run"
    run_backtest(cfg, out_dir=run_dir, only=["snaive"])
    ev = Evaluator(cfg, run_dir)
    # overwrite the forecast with the truth in every quantile
    ev.Q["snaive"] = np.repeat(ev.Y[..., None], len(ev.qs), -1)
    ev.M["snaive"] = ev.Y.copy()
    from forecastkit.evaluation.evaluator import Errors
    ev.err["snaive"] = Errors(ev.Q["snaive"], ev.M["snaive"], ev.Y, ev.qs)
    r = ev.main_table().loc["snaive"]
    assert r.MAE == 0 and r.RMSE == 0 and r.WAPE == 0 and r.pinball == 0 and r.cov90 == 1.0 and r.width90 == 0


def test_ensemble_is_quantile_average(cfg_copy):
    fc = run_backtest(cfg_copy, save=False)
    ens = add_ensembles(fc, {"ens": ["snaive", "kalman"]})
    e = ens[ens.model.astype(str) == "ens"].set_index(["series_id", "origin", "h", "stat"])["value"].astype(float)
    a = fc[fc.model.astype(str) == "snaive"].set_index(["series_id", "origin", "h", "stat"])["value"].astype(float)
    b = fc[fc.model.astype(str) == "kalman"].set_index(["series_id", "origin", "h", "stat"])["value"].astype(float)
    np.testing.assert_allclose(e.sort_index().to_numpy(), ((a + b) / 2).sort_index().to_numpy(), rtol=1e-4, atol=1e-5)


def test_generic_csv_adapter_fills_gaps_and_marks_availability(tmp_path):
    p = make_sample_csv(tmp_path / "s.csv", n_skus=6, n_days=200)
    ds = load_dataset({"type": "csv", "params": {
        "path": str(p), "series_cols": ["shop", "sku"], "date_col": "date", "target_col": "units",
        "static_cols": ["category"], "known_cols": ["price", "promo"], "name": "bakery"}}, cache_dir=tmp_path / "c")
    assert ds.N == 6 and ds.T == 200 and ds.avail.all()                      # gaps filled, full grid
    assert {"price", "promo", "dow", "month"} <= set(ds.known_names)
    assert "category" in ds.static.columns
    ds2 = load_dataset({"type": "csv", "params": {
        "path": str(p), "series_cols": ["shop", "sku"], "date_col": "date", "target_col": "units",
        "static_cols": ["category"], "known_cols": ["price", "promo"], "name": "bakery"}}, cache_dir=tmp_path / "c")
    assert np.array_equal(ds.Y, ds2.Y)                                       # cache round-trip


def test_cli_end_to_end(cfg_copy, tmp_path):
    import yaml

    from forecastkit.cli import main
    cfg = cfg_copy
    cfg["output_dir"] = str(tmp_path / "out")
    cfg["models"] = {"snaive": cfg["models"]["snaive"], "kalman": cfg["models"]["kalman"]}
    cfg["ensembles"] = {"ens": ["snaive", "kalman"]}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    main(["all", str(p)])
    rd = tmp_path / "out" / "t"
    assert (rd / "forecasts.parquet").exists()
    assert (rd / "report" / "RESULTS.md").exists()
    assert (rd / "report" / "figures" / "fan_charts.png").exists()
    txt = (rd / "report" / "RESULTS.md").read_text()
    assert "ens" in txt and "Main results" in txt
