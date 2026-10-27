"""forecastkit - an open, model-agnostic probabilistic demand-forecasting toolkit."""
import warnings

# nan-aware reductions over all-missing rows are expected (unavailable series); keep logs readable
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", category=FutureWarning)

__version__ = "0.1.0"

from .models.base import BaseForecaster  # noqa: F401
from .schema import Dataset, ForecastContext, QuantileForecast  # noqa: F401
from .utils import register_dataset, register_model  # noqa: F401
