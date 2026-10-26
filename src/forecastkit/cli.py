"""forecastkit command line.

    forecastkit synth  --out data/raw/m5_synth        synthetic M5-format data (no Kaggle needed)
    forecastkit subset CONFIG                         freeze the stratified series subset
    forecastkit run    CONFIG [--set a.b=c] [--models a,b] [--phase test|val]
    forecastkit evaluate CONFIG                       tables (CSV) from the forecast file
    forecastkit report CONFIG                         RESULTS.md + figures
    forecastkit all    CONFIG                         run + evaluate + report
    forecastkit tune   CONFIG --model lgbm --trials 8 random search on the VALIDATION origin only
    forecastkit ablate ABLATION.yaml                  ablations / sensitivity table
    forecastkit list                                  registered models and datasets
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from forecastkit.config import load_config
from forecastkit.utils import log, setup_logging


def _cfg(args):
    return load_config(args.config, getattr(args, "set", None))


def _run_dir(cfg):
    return Path(cfg.get("output_dir", "outputs")) / cfg["run_name"]


def cmd_synth(args):
    from forecastkit.datasets import make_sample_csv, make_synthetic_m5
    p = make_synthetic_m5(args.out, n_items=args.items, n_days=args.days, seed=args.seed)
    print(f"synthetic M5-format data written to {p}")
    if args.sample_csv:
        print(f"non-M5 sample csv written to {make_sample_csv(args.sample_csv)}")


def cmd_subset(args):
    from forecastkit.backtest.runner import prepare
    cfg = _cfg(args)
    ds, orig, info = prepare(cfg, use_cache=False)
    print(ds.static.groupby(["cat_id" if "cat_id" in ds.static else ds.static.columns[0], "intermittency"]).size().unstack(fill_value=0))
    print("origins:", [str(ds.dates[o].date()) for o in orig["val"] + orig["test"]])


def cmd_run(args):
    from forecastkit.backtest import run_backtest
    cfg = _cfg(args)
    only = args.models.split(",") if args.models else None
    run_backtest(cfg, phase=args.phase, only=only, use_cache=not args.no_cache)


def cmd_evaluate(args):
    from forecastkit.evaluation import evaluate_run
    cfg = _cfg(args)
    out = evaluate_run(cfg, _run_dir(cfg))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out["tables"]["main_all_days"].round(3))


def cmd_report(args):
    from forecastkit.report import build_report
    cfg = _cfg(args)
    p = build_report(cfg, _run_dir(cfg), args.fan_models.split(",") if args.fan_models else None)
    print(f"report written to {p}")


def cmd_all(args):
    cmd_run(args)
    cmd_report(args)


def cmd_tune(args):
    from forecastkit.backtest.tuning import tune_model
    import yaml
    cfg = _cfg(args)
    space = yaml.safe_load(Path(args.space).read_text()) if args.space else None
    res = tune_model(cfg, args.model, args.trials, args.seed, space)
    print("best:", res["best"])
    print("paste this into your config ->", res["snippet_path"])


def cmd_ablate(args):
    from forecastkit.backtest.ablation import run_ablation
    t = run_ablation(args.spec, args.set)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(t[["variant", "model", "MAE", "WAPE", "pinball", "cov90", "WAPE_noncens", "pinball_noncens"]].round(3))


def cmd_list(args):
    from forecastkit.utils.registry import DATASETS, MODELS
    import forecastkit.datasets  # noqa: F401
    import forecastkit.models  # noqa: F401
    print("models  :", sorted(MODELS))
    print("datasets:", sorted(DATASETS))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="forecastkit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sp = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, config=True):
        p = sp.add_parser(name)
        if config:
            p.add_argument("config")
            p.add_argument("--set", action="append", default=[], metavar="a.b.c=value", help="override a config value")
        p.set_defaults(fn=fn)
        return p

    p = sp.add_parser("synth"); p.add_argument("--out", default="data/raw/m5_synth")
    p.add_argument("--items", type=int, default=40); p.add_argument("--days", type=int, default=800)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--sample-csv", default=None)
    p.set_defaults(fn=cmd_synth)
    add("subset", cmd_subset)
    p = add("run", cmd_run); p.add_argument("--models"); p.add_argument("--phase", default="test", choices=["test", "val"])
    p.add_argument("--no-cache", action="store_true")
    add("evaluate", cmd_evaluate)
    p = add("report", cmd_report); p.add_argument("--fan-models")
    p = add("all", cmd_all); p.add_argument("--models"); p.add_argument("--phase", default="test"); p.add_argument("--no-cache", action="store_true")
    p.add_argument("--fan-models")
    p = add("tune", cmd_tune); p.add_argument("--model", required=True); p.add_argument("--trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--space")
    p = sp.add_parser("ablate"); p.add_argument("spec"); p.add_argument("--set", action="append", default=[])
    p.set_defaults(fn=cmd_ablate)
    sp.add_parser("list").set_defaults(fn=cmd_list)
    args = ap.parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    args.fn(args)


if __name__ == "__main__":
    main()
