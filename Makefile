# One-command reproduction.   make smoke  (2 min, synthetic)   |   make m5  (real M5 after download)
PY ?= python
CHROME ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
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

streamlit:
	streamlit run app.py

site:
	cd frontend && npm run dev

build-site:
	cd frontend && npm run build

api:
	forecastkit api --host 0.0.0.0 --port 8000

hf-deploy:
	@echo "Deploying to Hugging Face Space..."
	./hf_spaces/deploy.sh $(SPACE)

.PHONY: install test synth smoke subset baselines kalman lgbm deepar evaluate report m5 ablations demo-csv streamlit site build-site api hf-deploy


# ---- Phase 1 deliverables (no model training) ----
eda:
	$(PY) scripts/eda_m5.py --config $(CFG) --fig-dir docs/figures/eda --out-dir outputs/eda

paper:
	cd paper && tectonic -X compile forecastkit_phase1.tex

slides:
	PYTHON=$(PY) sh slides/build.sh

.PHONY: eda paper slides


# ---- Phase 2 ----
phase2-runs:
	PY="$(PY)" JOBS=5 sh scripts/phase2_run_all.sh

phase2-zero:
	$(PY) scripts/phase2_zero_model.py --config configs/m5.yaml --out results/phase2/zero_model \
	  --deepar deepar_v2_masked=outputs/phase2/runs/deepar_v2_s42/artifacts/deepar_v2 \
	  --deepar deepar_v2_unmasked=outputs/phase2/runs/deepar_v2_nomask_s42/artifacts/deepar_v2_nomask \
	  --deepar deepar_p1=outputs/phase2/runs/deepar_p1_s42/artifacts/deepar_p1

phase2-results:
	$(PY) scripts/phase2_e02_table.py
	$(PY) scripts/phase2_analysis.py

paper2:
	cd paper && tectonic -X compile forecastkit_phase2.tex

slides2:
	PYTHON="$(PY)" CHROME="$(CHROME)" sh slides/phase2/build.sh

.PHONY: phase2-runs phase2-zero phase2-results paper2 slides2
