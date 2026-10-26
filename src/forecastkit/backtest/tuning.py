"""Hyper-parameter search that ONLY ever looks at the validation origin(s), which are earlier than every
test origin. Plain random search (no extra dependency). Output: best params + a YAML snippet to paste."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from forecastkit.evaluation.evaluator import Evaluator
from forecastkit.utils import log
from .runner import run_backtest

DEFAULT_SPACES = {
    "lgbm": {"num_leaves": [15, 31, 63, 127], "learning_rate": [0.03, 0.06, 0.1], "min_child_samples": [20, 50, 100, 200],
             "feature_fraction": [0.6, 0.8, 1.0], "n_estimators": [100, 150, 250], "objective": ["poisson", "tweedie"]},
    "kalman_lgbm": {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.06, 0.1], "min_child_samples": [50, 100, 200],
                    "n_estimators": [100, 150, 250]},
    "kalman": {"transform": ["sqrt", "log1p"], "window": [180, 365, 540], "param_mode": ["grid", "shared"]},
    "deepar": {"hidden": [24, 40, 64], "context_length": [56, 84, 112], "lr": [5e-4, 1e-3, 2e-3],
               "dropout": [0.05, 0.1, 0.2], "layers": [1, 2], "likelihood": ["negbin", "gaussian"]},
    "seasonal_naive": {"n_seasons": [1, 2, 4, 8]},
    "moving_average": {"window": [7, 14, 28, 56]},
}


def tune_model(cfg: dict, model_name: str, trials: int = 8, seed: int = 0, space: dict | None = None,
               objective: str = "pinball", out_dir="configs/tuned") -> dict:
    spec = cfg["models"][model_name]
    space = space or DEFAULT_SPACES.get(spec["type"])
    if not space:
        raise ValueError(f"no search space for model type {spec['type']!r}; pass --space file.yaml")
    rng = np.random.default_rng(seed)
    results = []
    for k in range(trials + 1):
        cand = {} if k == 0 else {p: v[int(rng.integers(len(v)))] for p, v in space.items()}  # trial 0 = current params
        c = copy.deepcopy(cfg)
        c["models"] = {model_name: {"type": spec["type"], "params": {**spec.get("params", {}), **cand}}}
        fc = run_backtest(c, phase="val", only=[model_name], save=False)
        ev = Evaluator(c, ".", forecasts=fc, phase="val")
        res = ev.main_table().loc[model_name]
        score = float(res[objective])
        results.append({"params": cand, "score": score, "MAE": float(res.MAE), "pinball": float(res.pinball)})
        log.info("tune %s trial %d/%d  %s=%.4f  %s", model_name, k, trials, objective, score, cand)
    best = min(results, key=lambda r: r["score"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snippet = {"models": {model_name: {"type": spec["type"], "params": {**spec.get("params", {}), **best["params"]}}}}
    (out / f"{model_name}.yaml").write_text(yaml.safe_dump(snippet, sort_keys=False))
    return {"best": best, "all": results, "snippet_path": str(out / f"{model_name}.yaml")}
