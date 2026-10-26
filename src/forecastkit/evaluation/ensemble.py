"""Post-hoc ensembles on the forecast file (no re-training): quantile averaging ('Vincentization'),
which keeps quantiles non-crossing and is a strong, cheap combiner."""
from __future__ import annotations

import pandas as pd


def add_ensembles(fc: pd.DataFrame, spec: dict | None) -> pd.DataFrame:
    """spec = {ens_name: [model_a, model_b, ...]}  or  {ens_name: {models: [...], weights: [...]}}"""
    if not spec:
        return fc
    fc = fc.copy()
    for c in ("model", "stat"):
        fc[c] = fc[c].astype(str)
    out = [fc]
    for name, s in spec.items():
        models = s["models"] if isinstance(s, dict) else list(s)
        weights = (s.get("weights") if isinstance(s, dict) else None) or [1.0] * len(models)
        present = [m for m in models if m in set(fc.model)]
        if len(present) < 2:
            continue
        w = {m: float(wt) for m, wt in zip(models, weights) if m in present}
        tot = sum(w.values())
        sub = fc[fc.model.isin(present)].copy()
        sub["value"] = sub["value"].astype(float) * sub["model"].map(w) / tot
        agg = (sub.groupby(["series_id", "origin", "h", "ds", "stat"], observed=True, sort=False)["value"]
               .sum().reset_index())
        agg["model"] = name
        out.append(agg[fc.columns])
    return pd.concat(out, ignore_index=True)
