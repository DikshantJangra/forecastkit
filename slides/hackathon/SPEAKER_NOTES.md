# ForecastKit hackathon deck: speaker notes

## Slide 1

Retail data records sales, not demand. A zero in the sales log can mean nobody wanted the product, or that it was not available. On the left are three real products from the same dataset: one sells every day, one sells now and then, and one has gone quiet for weeks. Asking one model to handle all three is hard. So we split the job. DeepAR, a deep-learning model, estimates how much sells if anything sells. A LightGBM model estimates the chance that recorded sales are zero. A hurdle fusion joins the two into one valid probability distribution.

## Slide 2

We used 1,200 product-store series from three Walmart stores in the M5 dataset, 1,000 days each, forecasting 28 days ahead. 59% of recorded daily sales are zero. About nine in ten series are intermittent or lumpy, meaning they sell only now and then. We also saw strong weekend effects, higher sales on SNAP food-benefit days in every state, and slightly lower sales on event days. These are associations, not causes. The big data-quality problem: M5 has no inventory data, so we cannot tell a true zero from an empty shelf. We use a rule to flag unusually long zero runs as possibly censored. It is a proxy, not ground truth.

## Slide 3

Our core engineering rule: a forecast made on day T only sees what was known on day T. We test this with rolling-origin evaluation. At each of four test dates we refit on the past, forecast 28 days, and score against the hidden actuals. An earlier validation date is used only to choose settings. Inputs fall into four groups: sales history, demand behaviour such as the gap between sales, calendar and business signals like SNAP and price, and product identifiers. Long zero runs flagged by the proxy are not used to teach DeepAR that demand was zero, but they are kept and still scored.

## Slide 4

Two models, two jobs. DeepAR is one neural network shared by all series. It outputs a full distribution of possible sales. The zero-run model is a LightGBM classifier that answers one question: will recorded sales be zero on this future day? It looks at how long the product has already sold nothing and how its past zero runs ended. The fusion on the right is an illustrative example. The zero-run model says 60% chance of zero. We drop DeepAR's own zero bar, rescale its positive bars, and share the remaining 40% between them. The result is one distribution whose probabilities add up to one. A naive mix that keeps DeepAR's zero counts zeros twice.

## Slide 5

All models are scored on the same 124,992 forecasts. Overall, the hybrid beats our Phase 1 Kalman-LightGBM: WAPE down 4.7% and pinball loss down 3.3%, at every test origin, and the bootstrap intervals exclude zero. But the gain is not uniform. If we exclude the days the proxy flags as long zero runs, Kalman-LightGBM is slightly better. The chart on the right shows why: products already in a zero run of a week or more are 19% of forecasts but give 92% of the gain. For products that sold recently, the two are tied. Ablations show both parts matter. Remove DeepAR, or mix naively, and the error goes up. One caveat: the hybrid's intervals are too wide, so calibration is still open. Phase 2 uses three seeds, Phase 1 one run; seed spread was tiny.

## Slide 6

What did we learn? We built a leak-safe pipeline that compares statistical, machine-learning and deep-learning forecasters fairly. We improved overall recorded-sales error. And we found where the gain lives: in persistent zero-sales runs. On regular demand, Kalman-LightGBM is still stronger. So there is no single winner. The fans show the hybrid adapting its uncertainty to each pattern. Next steps: tighter uncertainty, real inventory data to separate true zero demand from stockouts, and more ablations on other datasets. ForecastKit: understanding demand patterns, forecasting with uncertainty.
