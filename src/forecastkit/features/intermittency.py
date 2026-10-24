"""Syntetos-Boylan demand classification + volume tiers (used for slicing results)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ADI_CUT, CV2_CUT = 1.32, 0.49


def classify_demand(Y: np.ndarray, avail: np.ndarray, end: int | None = None) -> pd.DataFrame:
    """ADI, CV^2 of non-zero demand and the SB class, from columns [:end] only.
    Computed before the first test origin so the slice labels never peek at test data."""
    end = Y.shape[1] if end is None else end
    y = np.nan_to_num(Y[:, :end].astype(np.float64))
    a = avail[:, :end]
    # ignore the pre-first-sale block
    pos = (y > 0) & a
    first = np.where(pos.any(1), pos.argmax(1), end)
    a = a & (np.arange(end)[None, :] >= first[:, None])
    n_avail = a.sum(1)
    n_nz = ((y > 0) & a).sum(1)
    adi = np.where(n_nz > 0, n_avail / np.maximum(n_nz, 1), np.inf)
    nzv = np.where((y > 0) & a, y, np.nan)
    with np.errstate(all="ignore"):
        mean_nz = np.nanmean(nzv, axis=1)
        std_nz = np.nanstd(nzv, axis=1)
        cv2 = np.where(n_nz > 1, (std_nz / mean_nz) ** 2, 0.0)
        mean_d = np.where(n_avail > 0, (y * a).sum(1) / np.maximum(n_avail, 1), 0.0)
    cls = np.where(adi < ADI_CUT, np.where(cv2 < CV2_CUT, "smooth", "erratic"),
                   np.where(cv2 < CV2_CUT, "intermittent", "lumpy"))
    cls = np.where(n_nz == 0, "no_sales", cls)
    df = pd.DataFrame({"adi": adi, "cv2": cv2, "mean_demand": mean_d, "intermittency": cls})
    pos_mean = df.loc[df.mean_demand > 0, "mean_demand"]
    if len(pos_mean) >= 3:
        lo, hi = pos_mean.quantile([1 / 3, 2 / 3])
        df["volume_tier"] = np.where(df.mean_demand <= lo, "low", np.where(df.mean_demand <= hi, "mid", "high"))
    else:
        df["volume_tier"] = "mid"
    return df
