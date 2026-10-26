import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from forecastkit.datasets import load_dataset, make_synthetic_m5  # noqa: E402

QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("m5raw")
    make_synthetic_m5(d, n_items=10, n_days=520, stores=("CA_1", "TX_1"), seed=3)
    return d


@pytest.fixture(scope="session")
def cfg(raw_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return {
        "run_name": "t", "seed": 1, "output_dir": str(out), "cache_dir": str(out / "cache"),
        "dataset": {"type": "m5", "params": {"name": "tsynth", "raw_dir": str(raw_dir), "stores": ["CA_1", "TX_1"],
                                              "items_per_store": 10, "subset_file": str(out / "subset.csv")}},
        "masks": {"leading_zeros": True, "stockout": {"enabled": True, "rule": "zero_run", "min_run": 21,
                                                      "adi_multiplier": 3.0, "min_nonzero_events": 8}},
        "backtest": {"horizon": 14, "n_origins": 2, "step": 14, "val_origins": 1, "quantiles": QS,
                     "cold_start": {"fraction": 0.1, "history_days": 21, "seed": 3}},
        "models": {"snaive": {"type": "seasonal_naive", "params": {"n_seasons": 2}},
                   "kalman": {"type": "kalman", "params": {"window": 200}}},
        "evaluation": {"reference_model": "snaive"},
    }


@pytest.fixture()
def cfg_copy(cfg):
    return copy.deepcopy(cfg)


@pytest.fixture(scope="session")
def ds(cfg):
    from forecastkit.backtest.runner import prepare
    d, orig, info = prepare(cfg)
    d._orig = orig
    return d
