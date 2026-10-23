"""Tiny plugin registries. Anything can be plugged in by name OR by import path 'pkg.module:Name',
so users can add models/datasets in their own code without touching this library."""
from __future__ import annotations

import importlib

MODELS: dict = {}
DATASETS: dict = {}


def register_model(name: str):
    def deco(cls):
        MODELS[name] = cls
        cls.registry_name = name
        return cls
    return deco


def register_dataset(name: str):
    def deco(fn):
        DATASETS[name] = fn
        return fn
    return deco


def _import_path(spec: str):
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def get_model_class(spec: str):
    import forecastkit.models  # noqa: F401  (populates registry)
    if spec in MODELS:
        return MODELS[spec]
    if ":" in spec:
        return _import_path(spec)
    raise KeyError(f"unknown model type {spec!r}; registered: {sorted(MODELS)}")


def get_dataset_fn(spec: str):
    import forecastkit.datasets  # noqa: F401
    if spec in DATASETS:
        return DATASETS[spec]
    if ":" in spec:
        return _import_path(spec)
    raise KeyError(f"unknown dataset type {spec!r}; registered: {sorted(DATASETS)}")
