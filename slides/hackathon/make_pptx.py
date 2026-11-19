"""Wrap the rendered slide PNGs into a 16:9 PPTX with speaker notes, and write SPEAKER_NOTES.md."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

HERE = Path(__file__).resolve().parent
F = json.loads((HERE / "deck_numbers.json").read_text())

NOTES = [
    # 1
    "Retail data records sales, not demand. A zero in the sales log can mean nobody wanted the product, or that it "
    "was not available. On the left are three real products from the same dataset: one sells every day, one sells "
    "now and then, and one has gone quiet for weeks. Asking one model to handle all three is hard. So we split the "
    "job. DeepAR, a deep-learning model, estimates how much sells if anything sells. A LightGBM model estimates the "
    "chance that recorded sales are zero. A hurdle fusion joins the two into one valid probability distribution.",
    # 2
    f"We used 1,200 product-store series from three Walmart stores in the M5 dataset, 1,000 days each, forecasting "
    f"28 days ahead. {F['zero_share']:.0%} of recorded daily sales are zero. About nine in ten series are "
    "intermittent or lumpy, meaning they sell only now and then. We also saw strong weekend effects, higher sales on "
    "SNAP food-benefit days in every state, and slightly lower sales on event days. These are associations, not "
    "causes. The big data-quality problem: M5 has no inventory data, so we cannot tell a true zero from an empty "
    "shelf. We use a rule to flag unusually long zero runs as possibly censored. It is a proxy, not ground truth.",
    # 3
    "Our core engineering rule: a forecast made on day T only sees what was known on day T. We test this with "
    "rolling-origin evaluation. At each of four test dates we refit on the past, forecast 28 days, and score against "
    "the hidden actuals. An earlier validation date is used only to choose settings. Inputs fall into four groups: "
    "sales history, demand behaviour such as the gap between sales, calendar and business signals like SNAP and "
    "price, and product identifiers. Long zero runs flagged by the proxy are not used to teach DeepAR that demand "
    "was zero, but they are kept and still scored.",
    # 4
    "Two models, two jobs. DeepAR is one neural network shared by all series. It outputs a full distribution of "
    "possible sales. The zero-run model is a LightGBM classifier that answers one question: will recorded sales be "
    "zero on this future day? It looks at how long the product has already sold nothing and how its past zero runs "
    "ended. The fusion on the right is an illustrative example. The zero-run model says 60% chance of zero. We drop "
    "DeepAR's own zero bar, rescale its positive bars, and share the remaining 40% between them. The result is one "
    "distribution whose probabilities add up to one. A naive mix that keeps DeepAR's zero counts zeros twice.",
    # 5
    f"All models are scored on the same {F['n_all']:,} forecasts. Overall, the hybrid beats our Phase 1 "
    f"Kalman-LightGBM: WAPE down {-F['rel_WAPE_pct']:.1f}% and pinball loss down {-F['rel_pinball_pct']:.1f}%, at "
    "every test origin, and the bootstrap intervals exclude zero. But the gain is not uniform. If we exclude the "
    "days the proxy flags as long zero runs, Kalman-LightGBM is slightly better. The chart on the right shows why: "
    f"products already in a zero run of a week or more are {F['share_runs_7plus']:.0%} of forecasts but give "
    f"{F['gain_share_runs_7plus']:.0%} of the gain. For products that sold recently, the two are tied. Ablations show "
    "both parts matter. Remove DeepAR, or mix naively, and the error goes up. One caveat: the hybrid's intervals are "
    "too wide, so calibration is still open. Phase 2 uses three seeds, Phase 1 one run; seed spread was tiny.",
    # 6
    "What did we learn? We built a leak-safe pipeline that compares statistical, machine-learning and deep-learning "
    "forecasters fairly. We improved overall recorded-sales error. And we found where the gain lives: in persistent "
    "zero-sales runs. On regular demand, Kalman-LightGBM is still stronger. So there is no single winner. The fans "
    "show the hybrid adapting its uncertainty to each pattern. Next steps: tighter uncertainty, real inventory data "
    "to separate true zero demand from stockouts, and more ablations on other datasets. ForecastKit: understanding "
    "demand patterns, forecasting with uncertainty.",
]


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    pngs = sorted((HERE / "png").glob("slide-*.png"))
    assert len(pngs) == 6, pngs
    for png, note in zip(pngs, NOTES):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.shapes.add_picture(str(png), 0, 0, prs.slide_width, prs.slide_height)
        s.notes_slide.notes_text_frame.text = note
    prs.save(HERE / "ForecastKit_Hackathon.pptx")
    md = ["# ForecastKit hackathon deck: speaker notes", ""]
    for i, n in enumerate(NOTES, 1):
        md += [f"## Slide {i}", "", n, ""]
    (HERE / "SPEAKER_NOTES.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
