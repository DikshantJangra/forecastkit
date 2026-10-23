from __future__ import annotations

import logging
import random

import numpy as np

log = logging.getLogger("forecastkit")


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # torch optional
        pass
