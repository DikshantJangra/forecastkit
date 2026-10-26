"""Ablation / sensitivity runner.

    base: configs/m5.yaml
    name: deepar_ablation
    models: [deepar]                       # only (re)run these models
    variants:
      full: {}
      no_item_id: {models.deepar.params.use_item_id: false}
      no_masking: {masks.stockout.enabled: false}

The EVALUATION stockout labels are frozen to the base config for every variant (`masks.eval_stockout`),
so a variant that changes how models *treat* stockouts is still scored against the same reference days."""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

from forecastkit.config import load_config, set_by_path
from forecastkit.evaluation.evaluator import Evaluator
from forecastkit.utils import log
from .runner import run_backtest


def run_ablation(spec_path, extra_overrides=None) -> pd.DataFrame:
    spec = yaml.safe_load(Path(spec_path).read_text())
    base = load_config(spec["base"], extra_overrides)
    name = spec.get("name", Path(spec_path).stem)
    only = spec.get("models")
    out_root = Path(base.get("output_dir", "outputs")) / name
    out_root.mkdir(parents=True, exist_ok=True)
    base["masks"] = copy.deepcopy(base.get("masks", {}))
    base["masks"].setdefault("eval_stockout", copy.deepcopy(base["masks"].get("stockout")))
    if only:
        base["models"] = {k: v for k, v in base["models"].items() if k in only}
    rows, cold_rows = [], []
    for vname, over in spec["variants"].items():
        cfg = copy.deepcopy(base)
        for k, v in (over or {}).items():
            set_by_path(cfg, k, v)
        cfg["run_name"] = f"{name}/{vname}"
        run_dir = out_root / vname
        log.info("=== ablation variant: %s %s", vname, over)
        fc = run_backtest(cfg, out_dir=run_dir)
        ev = Evaluator(cfg, run_dir)
        mt, cs = ev.main_table(), ev.cold_start()
        nc = ev.main_table("non_censored")[["MAE", "WAPE", "pinball"]].add_suffix("_noncens")  # stockout-fair view
        mt = mt.join(nc)
        mt.insert(0, "variant", vname)
        rows.append(mt.reset_index())
        if len(cs):
            cs.insert(0, "variant", vname)
            cold_rows.append(cs.reset_index())
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out_root / "ablation_table.csv", index=False)
    if cold_rows:
        pd.concat(cold_rows, ignore_index=True).to_csv(out_root / "ablation_cold_start.csv", index=False)
    return table
