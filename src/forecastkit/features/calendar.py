"""Calendar covariates built from dates only (always known in the future)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ALL_CAL = ("dow", "dow_onehot", "dom", "month", "month_cyc", "is_weekend")


def calendar_features(dates: pd.DatetimeIndex, which=ALL_CAL):
    cols, names = [], []
    dow = np.asarray(dates.day_of_week, dtype=np.int64)
    if "dow" in which:
        cols.append(dow.astype(np.float32)); names.append("dow")
    if "dow_onehot" in which:
        for d in range(7):
            cols.append((dow == d).astype(np.float32)); names.append(f"dow_{d}")
    if "dom" in which:
        cols.append(np.asarray(dates.day, dtype=np.float32)); names.append("dom")
    if "month" in which:
        cols.append(np.asarray(dates.month, dtype=np.float32)); names.append("month")
    if "month_cyc" in which:
        ang = 2 * np.pi * (np.asarray(dates.month, dtype=np.float32) - 1.0) / 12.0
        cols += [np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)]
        names += ["month_sin", "month_cos"]
    if "is_weekend" in which:
        cols.append((dow >= 5).astype(np.float32)); names.append("is_weekend")
    return np.stack(cols, axis=1), names  # [T, k]
