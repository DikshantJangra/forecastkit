"""The canonical data contract.

Every dataset, whatever its origin, is converted by an *adapter* into a `Dataset`:

    Y      [N, T]    raw target (e.g. units sold), complete daily grid, no gaps
    avail  [N, T]    False where the series did not exist / was not on sale (NOT demand)
    K      [N, T, k] known-future covariates (calendar, price, events ...) incl. the future
    static DataFrame indexed by series_id (hierarchy / categorical attributes)

Nothing downstream (masks, models, backtester, evaluation) knows where the data came from.

Forecasts are always stored in ONE long format (`FORECAST_COLUMNS`), so evaluation works for any
model, including models trained outside this library.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FORECAST_COLUMNS = ["model", "series_id", "origin", "h", "ds", "stat", "value"]


def q_stat(q: float) -> str:
    """Name of a quantile in the `stat` column of the forecast file."""
    return f"q{q:g}"


@dataclass
class Dataset:
    name: str
    series_ids: np.ndarray
    dates: pd.DatetimeIndex
    Y: np.ndarray
    avail: np.ndarray
    K: np.ndarray
    known_names: list
    static: pd.DataFrame
    season: int = 7
    freq: str = "D"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.series_ids = np.asarray(self.series_ids).astype(str)
        self.Y = np.asarray(self.Y, dtype=np.float32)
        self.avail = np.asarray(self.avail, dtype=bool)
        self.K = np.asarray(self.K, dtype=np.float32)
        self.known_names = list(self.known_names)
        self.static = self.static.copy()
        self.static.index = self.static.index.astype(str)
        self.static = self.static.loc[self.series_ids]
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        N, T = self.Y.shape
        assert len(self.series_ids) == N, "series_ids length != N"
        assert len(self.dates) == T, "dates length != T"
        assert self.avail.shape == (N, T), "avail shape mismatch"
        assert self.K.shape[:2] == (N, T) and self.K.shape[2] == len(self.known_names), "K shape mismatch"
        assert self.dates.is_monotonic_increasing and self.dates.is_unique, "dates must be sorted and unique"
        if T > 1:
            diffs = np.unique(np.diff(self.dates.values).astype("timedelta64[D]").astype(int))
            if self.freq == "D":
                assert set(diffs) == {1}, "daily dataset must have a complete date index (no gaps)"
        assert np.isfinite(self.K).all(), "K must not contain NaN/inf (fill covariates in the adapter)"
        assert not np.isnan(self.Y[self.avail]).any(), "Y must be observed wherever avail=True"
        assert (self.Y[self.avail] >= 0).all(), "negative targets are not supported"

    @property
    def N(self) -> int:
        return self.Y.shape[0]

    @property
    def T(self) -> int:
        return self.Y.shape[1]

    def k_index(self, name: str) -> int:
        return self.known_names.index(name)

    # -------------------------------------------------------------- conversion
    def to_long(self) -> pd.DataFrame:
        sid = np.repeat(self.series_ids, self.T)
        ds = np.tile(self.dates.values, self.N)
        return pd.DataFrame(
            {"series_id": sid, "ds": ds, "y": self.Y.reshape(-1), "available": self.avail.reshape(-1)}
        )

    def subset(self, mask: np.ndarray) -> "Dataset":
        mask = np.asarray(mask, dtype=bool)
        return Dataset(self.name, self.series_ids[mask], self.dates, self.Y[mask], self.avail[mask],
                       self.K[mask], self.known_names, self.static[mask], self.season, self.freq, dict(self.meta))

    # ------------------------------------------------------------- persistence
    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "arrays.npz", Y=self.Y, avail=self.avail, K=self.K,
                            series_ids=self.series_ids, dates=self.dates.values.astype("datetime64[ns]"))
        self.static.reset_index(names="series_id").to_parquet(path / "static.parquet")
        (path / "meta.json").write_text(json.dumps(
            {"name": self.name, "known_names": self.known_names, "season": self.season,
             "freq": self.freq, "meta": self.meta}, default=str))

    @classmethod
    def load(cls, path) -> "Dataset":
        path = Path(path)
        z = np.load(path / "arrays.npz", allow_pickle=False)
        meta = json.loads((path / "meta.json").read_text())
        static = pd.read_parquet(path / "static.parquet").set_index("series_id")
        return cls(meta["name"], z["series_ids"], pd.DatetimeIndex(z["dates"]), z["Y"], z["avail"], z["K"],
                   meta["known_names"], static, meta["season"], meta["freq"], meta["meta"])


@dataclass
class ForecastContext:
    """Everything a model may see at one forecast origin. It is leak-proof by construction:
    the target arrays stop at the origin; only known-future covariates extend into the horizon."""

    origin: pd.Timestamp
    horizon: int
    series_ids: np.ndarray
    hist_dates: pd.DatetimeIndex
    fut_dates: pd.DatetimeIndex
    y: np.ndarray            # [N, Th] target, NaN where unavailable OR flagged censored (training mask)
    y_raw: np.ndarray        # [N, Th] target, NaN only where unavailable
    available: np.ndarray    # [N, Th] bool
    censored: np.ndarray     # [N, Th] bool (stockout proxy computed on history only)
    known_hist: np.ndarray   # [N, Th, k]
    known_fut: np.ndarray    # [N, H, k]
    known_names: list
    static: pd.DataFrame     # aligned with series_ids
    cold: np.ndarray         # [N] bool: simulated cold-start series (history truncated)
    quantiles: tuple
    season: int = 7
    seed: int = 0
    origin_index: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.y.shape[0]

    @property
    def Th(self) -> int:
        return self.y.shape[1]

    def target(self, use_mask: bool = True) -> np.ndarray:
        return self.y if use_mask else self.y_raw

    def covariate_idx(self, include=None, exclude=None) -> list:
        names = self.known_names
        idx = [i for i, n in enumerate(names) if (include is None or n in include)
               and (exclude is None or n not in exclude)]
        return idx


@dataclass
class QuantileForecast:
    """What every model's `predict` returns. Aligned with ctx.series_ids."""

    quantiles: np.ndarray            # [N, H, Q]
    mean: np.ndarray | None = None   # [N, H] (optional; falls back to the median downstream)
