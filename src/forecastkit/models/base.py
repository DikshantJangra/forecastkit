"""THE model interface. A new model = one file implementing these two methods.

    class MyModel(BaseForecaster):
        def fit(self, ctx): ...                    # may only use what is inside `ctx` (leak-proof by design)
        def predict(self, quantiles): ...          # -> QuantileForecast(quantiles=[N,H,Q], mean=[N,H] or None)

Then register it with @register_model("mine") or reference it in YAML as `type: "mypkg.mod:MyModel"`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from forecastkit.schema import ForecastContext, QuantileForecast


class BaseForecaster(ABC):
    registry_name = "base"

    def __init__(self, **params):
        self.params = params
        self.ctx: ForecastContext | None = None

    @abstractmethod
    def fit(self, ctx: ForecastContext) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, quantiles) -> QuantileForecast:
        ...

    # helpers ---------------------------------------------------------------
    def p(self, key, default):
        return self.params.get(key, default)

    @staticmethod
    def finalize(q: np.ndarray, mean=None) -> QuantileForecast:
        """Non-negative, non-crossing quantiles (rearrangement)."""
        q = np.sort(np.maximum(q, 0.0), axis=-1)
        if mean is not None:
            mean = np.maximum(mean, 0.0)
        return QuantileForecast(quantiles=q.astype(np.float32), mean=None if mean is None else mean.astype(np.float32))
