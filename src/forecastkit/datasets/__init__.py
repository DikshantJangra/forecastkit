"""Dataset adapters. Add yours: write a function returning `forecastkit.schema.Dataset`, decorate it with
`@register_dataset("myname")` (or reference it in YAML as `type: "mypkg.mod:my_loader"`)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forecastkit.schema import Dataset
from forecastkit.utils import get_dataset_fn, log
from . import csv_adapter as _csv  # noqa: F401
from . import m5 as _m5  # noqa: F401
from .synthetic import make_sample_csv, make_synthetic_m5  # noqa: F401


def load_dataset(cfg: dict, cache_dir="data/processed", use_cache: bool = True) -> Dataset:
    """cfg = {type: <registered name or 'pkg.mod:fn'>, params: {...}}"""
    spec, params = cfg["type"], dict(cfg.get("params", {}))
    key = hashlib.md5(json.dumps({"t": spec, "p": params}, sort_keys=True, default=str).encode()).hexdigest()[:10]
    path = Path(cache_dir) / f"{params.get('name', spec)}_{key}"
    if use_cache and (path / "arrays.npz").exists():
        log.info("dataset cache hit: %s", path)
        return Dataset.load(path)
    ds = get_dataset_fn(spec)(**params)
    if use_cache:
        ds.save(path)
    return ds
