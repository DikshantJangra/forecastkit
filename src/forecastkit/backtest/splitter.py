"""Rolling-origin splits. An origin is the LAST observed day; the forecast window is origin+1..origin+H.

Test origins end exactly at the last day of data; one extra earlier origin per `val_origins` is
reserved for tuning only, so the test origins are never used to pick hyper-parameters."""
from __future__ import annotations

import pandas as pd


def rolling_origins(dates: pd.DatetimeIndex, horizon: int, n_origins: int, step: int, val_origins: int = 1,
                    explicit: list | None = None) -> dict:
    T = len(dates)
    if explicit:
        idx = sorted(int(dates.get_loc(pd.Timestamp(o))) for o in explicit)
        test = idx
        val = [test[0] - step * (k + 1) for k in range(val_origins)][::-1]
    else:
        last = T - 1 - horizon
        test = [last - step * k for k in range(n_origins)][::-1]
        val = [test[0] - step * (k + 1) for k in range(val_origins)][::-1]
    for o in test + val:
        if o < 0 or o + horizon > T - 1:
            raise ValueError(f"origin index {o} leaves no room for a {horizon}-step window (T={T})")
    if step < horizon:
        import logging
        logging.getLogger("forecastkit").warning("step < horizon: forecast windows of adjacent origins overlap")
    return {"val": val, "test": test}
