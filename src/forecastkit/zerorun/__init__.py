"""Zero-run persistence modelling: P(recorded sales = 0 at t+h | history to t). No inventory data is assumed;
see docs/PHASE2_THEORY.md for the target definition and its limits."""
from .baselines import KMRenewal, recent_zero_rate, series_zero_rate  # noqa: F401
from .features import PERSISTENCE_FEATURES, zero_run_features  # noqa: F401
from .metrics import binary_scores, reliability_table  # noqa: F401
