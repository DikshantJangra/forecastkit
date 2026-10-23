"""Models. Importing this package registers the built-ins."""
from .base import BaseForecaster  # noqa: F401
from .naive import NaiveForecaster, SeasonalNaive, MovingAverage  # noqa: F401
from .kalman import KalmanForecaster  # noqa: F401

try:  # optional heavy deps are imported lazily so the core library stays light
    from .lgbm import LGBMForecaster, KalmanLGBMForecaster  # noqa: F401
except ImportError:  # pragma: no cover
    pass
try:
    from .deepar import DeepARForecaster  # noqa: F401
except ImportError:  # pragma: no cover
    pass
