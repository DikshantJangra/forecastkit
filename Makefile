# One-command reproduction.   make smoke  (2 min, synthetic)   |   make m5  (real M5 after download)
PY ?= python
CFG ?= configs/m5.yaml

install:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest -q

synth:
	forecastkit synth --out data/raw/m5_synth --sample-csv examples/data/bakery.csv

smoke: synth
	forecastkit all configs/m5_smoke.yaml

subset:
	forecastkit subset $(CFG)

baselines:
	forecastkit run $(CFG) --models naive,snaive,snaive4,ma28

kalman:
	forecastkit run $(CFG) --models kalman

lgbm:
	forecastkit run $(CFG) --models lgbm,kalman_lgbm

deepar:
	forecastkit run $(CFG) --models deepar

evaluate:
	forecastkit evaluate $(CFG)

report:
	forecastkit report $(CFG)

m5: subset
	forecastkit all $(CFG)

ablations:
	forecastkit ablate configs/ablations/deepar.yaml
	forecastkit ablate configs/ablations/stockout.yaml
	forecastkit ablate configs/ablations/kalman.yaml
	forecastkit ablate configs/ablations/lgbm.yaml

demo-csv: synth
	forecastkit all configs/example_csv.yaml

.PHONY: install test synth smoke subset baselines kalman lgbm deepar evaluate report m5 ablations demo-csv
