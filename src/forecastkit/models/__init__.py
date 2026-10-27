"""Models. Importing this package registers the built-ins."""
from .base import BaseForecaster  # noqa: F401
from .kalman import KalmanForecaster  # noqa: F401
from .naive import MovingAverage, NaiveForecaster, SeasonalNaive  # noqa: F401

try:  # optional heavy deps are imported lazily so the core library stays light
    from .lgbm import KalmanLGBMForecaster, LGBMForecaster  # noqa: F401
except ImportError:  # pragma: no cover
    pass
try:
    from .deepar import DeepARForecaster  # noqa: F401
except ImportError:  # pragma: no cover
    pass
