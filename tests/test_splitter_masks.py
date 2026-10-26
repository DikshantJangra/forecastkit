import numpy as np
import pandas as pd
import pytest

from forecastkit.backtest.splitter import rolling_origins
from forecastkit.masks import leading_zero_mask, stockout_mask


def test_origins_are_ordered_non_overlapping_and_end_at_last_day():
    dates = pd.date_range("2020-01-01", periods=400)
    o = rolling_origins(dates, horizon=28, n_origins=4, step=28, val_origins=1)
    test, val = o["test"], o["val"]
    assert test == sorted(test) and val[-1] < test[0]
    assert test[-1] + 28 == len(dates) - 1                     # last window ends on the last day
    windows = [(t + 1, t + 28) for t in test]
    for (a0, a1), (b0, b1) in zip(windows, windows[1:]):
        assert a1 < b0                                         # non-overlapping
    assert max(val) + 28 < min(test) + 1                       # validation windows never touch test windows


def test_origin_out_of_range_raises():
    dates = pd.date_range("2020-01-01", periods=60)
    with pytest.raises(ValueError):
        rolling_origins(dates, 28, 4, 28)


CFG = {"enabled": True, "min_run": 14, "adi_multiplier": 3.0, "min_nonzero_events": 5}


def _series(n=200, sell_every=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.where(np.arange(n) % sell_every == 0, rng.integers(1, 4, n), 0).astype(float)
    return y


def test_long_zero_run_in_regular_seller_is_flagged_short_run_is_not():
    y = _series()
    y[100:130] = 0          # 30-day gap: way longer than ADI=3 and min_run=14
    y[50:56] = 0            # short natural gap
    avail = np.ones_like(y, dtype=bool)
    m = stockout_mask(y[None], avail[None], CFG)[0]
    assert m[105:125].all() and not m[50:56].any()


def test_slow_seller_long_zeros_are_not_called_stockouts():
    y = np.zeros(300)
    y[::40] = 2.0           # ADI = 40: gaps of 40 days are NORMAL for this item
    m = stockout_mask(y[None], np.ones(300, bool)[None], CFG)[0]
    assert not m.any()


def test_stockout_mask_does_not_look_past_the_end_of_the_array():
    y = _series(240)
    y[150:200] = 0
    avail = np.ones_like(y, dtype=bool)
    full = stockout_mask(y[None], avail[None], CFG)[0]
    for cut in (120, 160, 180):                      # origin cuts INSIDE or before the run
        part = stockout_mask(y[None, :cut], avail[None, :cut], CFG)[0]
        # flags in the truncated history can only appear if the run is already long enough by then
        if cut - 150 < 14:
            assert not part[150:cut].any()
    # and when disabled nothing is flagged
    assert not stockout_mask(y[None], avail[None], {"enabled": False}).any()


def test_leading_zeros_are_not_demand():
    y = np.r_[np.zeros(30), 1, 0, 2, 0, 0, 3].astype(float)
    m = leading_zero_mask(y[None], np.ones(len(y), bool)[None])[0]
    assert m[:30].all() and not m[30:].any()
