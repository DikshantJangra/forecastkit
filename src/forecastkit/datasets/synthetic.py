"""Synthetic retail data written in EXACT M5 file format (calendar.csv, sales_train_evaluation.csv,
sell_prices.csv), so the real M5 adapter can be tested/CI'd without a Kaggle download.

It deliberately contains the hard cases from the brief: intermittent/lumpy items, late launches
(cold start), stockout episodes (zeros while the item is priced), price promos, events, SNAP days,
slow level shifts and weekly seasonality shared inside a department (so global models have
something to learn across items).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEPTS = {"HOBBIES_1": "HOBBIES", "HOBBIES_2": "HOBBIES", "HOUSEHOLD_1": "HOUSEHOLD",
         "HOUSEHOLD_2": "HOUSEHOLD", "FOODS_1": "FOODS", "FOODS_2": "FOODS", "FOODS_3": "FOODS"}
STATE_OF = {"CA": "CA", "TX": "TX", "WI": "WI"}


def make_synthetic_m5(out_dir, n_items: int = 40, n_days: int = 800, stores=("CA_1", "TX_1", "WI_1"),
                      seed: int = 0, start: str = "2013-01-05") -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start, periods=n_days, freq="D")
    d_cols = [f"d_{i + 1}" for i in range(n_days)]
    wk = 11101 + (np.arange(n_days) // 7)
    dow = dates.dayofweek.values

    # ---- calendar ---------------------------------------------------------
    ev_days = rng.choice(np.arange(n_days), size=max(6, n_days // 30), replace=False)
    ev_type = rng.choice(["Religious", "National", "Cultural", "Sporting"], size=len(ev_days))
    ev1 = np.full(n_days, None, dtype=object)
    ty1 = np.full(n_days, None, dtype=object)
    for d, t in zip(ev_days, ev_type):
        ev1[d], ty1[d] = f"{t}Day{d}", t
    dom = dates.day.values
    cal = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "wm_yr_wk": wk, "weekday": dates.day_name(),
        "wday": (dow + 2 - 1) % 7 + 1, "month": dates.month, "year": dates.year, "d": d_cols,
        "event_name_1": ev1, "event_type_1": ty1, "event_name_2": None, "event_type_2": None,
        "snap_CA": (dom <= 10).astype(int), "snap_TX": ((dom >= 2) & (dom <= 11) & (dom % 2 == 0)).astype(int),
        "snap_WI": ((dom >= 3) & (dom <= 12)).astype(int)})
    is_event = np.array([e is not None for e in ev1])

    # ---- demand ------------------------------------------------------------
    dept_week = {d: np.clip(rng.normal(1, 0.35, 7), 0.4, None) for d in DEPTS}
    dept_year_amp = {c: rng.uniform(0.05, 0.3) for c in set(DEPTS.values())}
    items = []
    for i in range(n_items):
        dept = list(DEPTS)[rng.choice(len(DEPTS), p=[.08, .1, .12, .12, .18, .2, .2])]
        items.append((f"{dept}_{i:03d}", dept, DEPTS[dept]))

    sales_rows, price_rows = [], []
    week_idx = np.arange(n_days) // 7
    n_weeks = week_idx.max() + 1
    for store in stores:
        state = store.split("_")[0]
        snap = cal[f"snap_{state}"].values
        for item_id, dept, cat in items:
            klass = rng.choice(["smooth", "erratic", "intermittent", "lumpy"], p=[.3, .2, .3, .2])
            lam0 = {"smooth": rng.lognormal(1.6, .6), "erratic": rng.lognormal(0.9, .5),
                    "intermittent": rng.uniform(.08, .5), "lumpy": rng.uniform(.05, .25)}[klass]
            disp = {"smooth": .15, "erratic": .6, "intermittent": .5, "lumpy": 2.5}[klass]
            wprof = dept_week[dept] * np.clip(rng.normal(1, .08, 7), .6, None)
            wprof = wprof / wprof.mean()
            yr = 1 + dept_year_amp[cat] * np.sin(2 * np.pi * (np.arange(n_days) + rng.uniform(0, 365)) / 365.25)
            # slow level drift + occasional shift
            wk_step = rng.normal(0, .03, n_weeks)
            if rng.random() < .35:
                wk_step[rng.integers(10, n_weeks - 5)] += rng.choice([-.6, .6])
            level = np.exp(np.cumsum(wk_step))[week_idx]
            # prices
            p0 = rng.lognormal(1.0, .6)
            wp = p0 * np.exp(np.cumsum(rng.normal(0, .01, n_weeks)))
            promo = rng.random(n_weeks) < .06
            wp = np.where(promo, wp * rng.uniform(.7, .88, n_weeks), wp)
            price_d = wp[week_idx]
            elastic = (price_d / np.exp(np.log(wp).mean())) ** -1.4
            ev_mult = 1 + is_event * (.25 if cat != "HOBBIES" else .1)
            snap_mult = 1 + snap * (.18 if cat == "FOODS" else 0.)
            mu = lam0 * wprof[dow] * yr * level * elastic * ev_mult * snap_mult
            r = 1.0 / disp
            y = rng.negative_binomial(r, r / (r + mu)).astype(np.int32)
            # launch (cold start) and stockouts
            launch = 0
            if rng.random() < .22:
                launch = int(rng.integers(30, max(31, n_days - 120)))
            if rng.random() < .2:
                for _ in range(rng.integers(1, 3)):
                    L = int(rng.integers(21, 60))
                    s = int(rng.integers(max(launch, 60), max(launch + 61, n_days - L - 1)))
                    y[s:s + L] = 0
            y[:launch] = 0
            sales_rows.append({"id": f"{item_id}_{store}_evaluation", "item_id": item_id, "dept_id": dept,
                               "cat_id": cat, "store_id": store, "state_id": state,
                               **{c: int(v) for c, v in zip(d_cols, y)}})
            first_week = launch // 7
            for w in range(first_week, n_weeks):
                price_rows.append((store, item_id, 11101 + w, round(float(wp[w]), 2)))

    pd.DataFrame(sales_rows).to_csv(out / "sales_train_evaluation.csv", index=False)
    cal.to_csv(out / "calendar.csv", index=False)
    pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(
        out / "sell_prices.csv", index=False)
    return out


def make_sample_csv(path, n_skus: int = 24, n_days: int = 420, seed: int = 1) -> Path:
    """A NON-M5 dataset in a different shape (long CSV: shop, sku, date, units, price, promo) used
    to demonstrate the generic CSV adapter - proof that nothing in the library is M5-specific."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rows = []
    for k in range(n_skus):
        shop = f"shop_{k % 3}"
        cat = ["bread", "cake", "drink", "snack"][k % 4]
        lam = rng.lognormal(1.3, .6)
        wprof = np.clip(rng.normal(1, .3, 7), .4, None)
        promo = (rng.random(n_days) < .05).astype(int)
        price = np.round(rng.uniform(1, 6) * (1 - .2 * promo), 2)
        mu = lam * wprof[dates.dayofweek] * (1 + .5 * promo) * (1 + .15 * np.sin(2 * np.pi * np.arange(n_days) / 365))
        units = rng.poisson(mu)
        if k % 7 == 0:  # a stockout episode, recorded as zero sales
            units[200:230] = 0
        for d, u, p, pr in zip(dates, units, price, promo):
            rows.append((shop, f"sku_{k:03d}", cat, d.strftime("%Y-%m-%d"), int(u), float(p), int(pr)))
    df = pd.DataFrame(rows, columns=["shop", "sku", "category", "date", "units", "price", "promo"])
    df = df.drop(df.sample(frac=.01, random_state=seed).index)  # a few missing dates, to prove gap-filling
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
