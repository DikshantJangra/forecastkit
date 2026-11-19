"""YAML config loading with `--set a.b.c=value` overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_path(cfg: dict, path: str, value) -> None:
    keys = path.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_overrides(cfg: dict, overrides) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like a.b.c=value, got {item!r}")
        path, raw = item.split("=", 1)
        set_by_path(cfg, path.strip(), yaml.safe_load(raw))
    return cfg


REPLACED_ON_EXTEND = ("models", "ensembles")   # a child config's model list replaces the parent's, never merges


def load_config(path, overrides=None) -> dict:
    """`extends: other.yaml` (path relative to this file) deep-merges this config onto the parent; the keys in
    REPLACED_ON_EXTEND are replaced wholesale so a child config runs exactly the models it lists."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        base.pop("_config_path", None)
        base.pop("run_name", None)
        for k in REPLACED_ON_EXTEND:
            if k in cfg:
                base.pop(k, None)
        cfg = deep_update(base, cfg)
    cfg = apply_overrides(cfg, overrides)
    cfg.setdefault("run_name", path.stem)
    cfg["_config_path"] = str(path)
    return cfg
