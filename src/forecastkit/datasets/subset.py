"""Stratified, frozen subset selection for big hierarchical datasets (M5).

Stratify by (department x Syntetos-Boylan class) so smooth / intermittent / erratic / lumpy items
and every department are represented; seed-fixed; the chosen ids are written to a CSV and reused so
every experiment sees the same series.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecastkit.features.intermittency import classify_demand


def stratified_subset(sales: pd.DataFrame, d_cols: list, stores: list, items_per_store: int, seed: int = 42,
                      stratify=("dept_id", "intermittency")) -> pd.DataFrame:
    """sales: M5 wide frame (id, item_id, dept_id, cat_id, store_id, state_id, d_*). Returns chosen rows' ids."""
    picks = []
    for store in stores:
        sub = sales[sales.store_id == store].copy()
        Y = sub[d_cols].to_numpy(dtype=np.float64)
        cls = classify_demand(Y, np.ones_like(Y, dtype=bool))
        sub["intermittency"] = cls["intermittency"].values
        sub = sub[sub.intermittency != "no_sales"]
        n = min(items_per_store, len(sub))
        frac = n / len(sub)
        chosen = sub.groupby(list(stratify), group_keys=False).sample(frac=frac, random_state=seed)
        if len(chosen) > n:
            chosen = chosen.sample(n=n, random_state=seed)
        elif len(chosen) < n:
            rest = sub.drop(chosen.index)
            chosen = pd.concat([chosen, rest.sample(n=n - len(chosen), random_state=seed)])
        picks.append(chosen[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]])
    return pd.concat(picks, ignore_index=True)


def freeze(subset: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(path, index=False)
    return path
