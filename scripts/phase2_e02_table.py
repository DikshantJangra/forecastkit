"""E02 table: DeepAR design selection on the validation origin (2016-01-03). Writes results/phase2/e02_validation_selection.csv.
    python scripts/phase2_e02_table.py"""
import glob
import json
from pathlib import Path

import pandas as pd

from forecastkit.config import load_config
from forecastkit.evaluation.evaluator import Evaluator

cfg = load_config("configs/phase2/e02_deepar_select.yaml")
rows = []
for d in sorted(glob.glob("outputs/phase2/e02_val/*")):
    ev = Evaluator(cfg, d, phase="val")
    a, n = ev.main_table("all").iloc[0], ev.main_table("non_censored").iloc[0]
    h = json.loads(Path(glob.glob(d + "/artifacts/*/train_*.json")[0]).read_text())
    rows.append({"variant": Path(d).name, "origin": "2016-01-03", "WAPE_all": a.WAPE, "pinball_all": a.pinball,
                 "WAPE_noncens": n.WAPE, "pinball_noncens": n.pinball, "cov80": a.cov80, "best_step": h.get("best_step"),
                 "stopped_step": h.get("stopped_step"), "best_val_nll": h.get("best_val"), "train_seconds": h.get("seconds")})
t = pd.DataFrame(rows)
order = ["deepar_p1", "deepar_es", "deepar_es_lags", "deepar_es_lags_age", "deepar_es_lags_age_adamw"]
t = t.set_index("variant").loc[order].reset_index()
t["selected"] = t.variant == "deepar_es_lags"
Path("results/phase2").mkdir(parents=True, exist_ok=True)
t.to_csv("results/phase2/e02_validation_selection.csv", index=False)
print(t.round(5).to_string(index=False))
