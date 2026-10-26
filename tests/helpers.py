"""A user-defined model living OUTSIDE the library, referenced from config by import path."""
import numpy as np

from forecastkit import BaseForecaster, QuantileForecast


class ConstModel(BaseForecaster):
    def fit(self, ctx):
        self.ctx = ctx
        self.level = np.nan_to_num(np.nanmean(ctx.y[:, -28:], axis=1))
        return self

    def predict(self, quantiles):
        H, qs = self.ctx.horizon, np.asarray(quantiles)
        q = self.level[:, None, None] * (0.5 + qs[None, None, :]) * np.ones((1, H, 1))
        return self.finalize(q, mean=np.repeat(self.level[:, None], H, 1))
