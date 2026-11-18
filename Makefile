# PathSnap: common tasks.
#
# Assumes the virtual environment in .venv is active, or set PYTHON explicitly:
#     make validate PYTHON=/path/to/venv/bin/python

PYTHON ?= python3
SHELL  := /bin/bash

# Some sandboxes make the default temp/matplotlib directories unwritable.
export TMPDIR      ?= $(CURDIR)/.tmp
export MPLCONFIGDIR ?= $(CURDIR)/.mpl

.PHONY: help setup validate eda labels recon clean features radius-sweep leakage train member2 hpo routes check-pipeline predict-batch member3

help:
	@echo "PathSnap targets"
	@echo "  setup      install Python dependencies"
	@echo "  validate   parse and validate every supplied trip"
	@echo "  eda        dataset measurements, figures and Folium maps"
	@echo "  labels     observation-level reference labels for every trip"
	@echo "  recon      dataset reconnaissance summary"
	@echo "  features   candidates + features + labels for every trip (Member 2; needs make labels)"
	@echo "  radius-sweep  candidate radius vs miss rate experiment"
	@echo "  leakage    prediction-path leakage checks"
	@echo "  train      train/compare models, select best, ablation + importance"
	@echo "  member2    labels -> features -> radius-sweep -> leakage -> train"
	@echo "  hpo        Optuna (Bayesian) tuning of LightGBM + untuned vs tuned (Member 3)"
	@echo "  routes     decoder tuning + classifier vs HMM route comparison (Member 3)"
	@echo "  check-pipeline  decoder/pipeline contract checks"
	@echo "  predict-batch   run the final pipeline on all supplied trips"
	@echo "  member3    hpo -> routes -> check-pipeline -> predict-batch"
	@echo "  clean      remove caches and temporary files"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

validate:
	@mkdir -p "$(TMPDIR)" "$(MPLCONFIGDIR)"
	$(PYTHON) scripts/validate_dataset.py $(ARGS)

eda:
	@mkdir -p "$(TMPDIR)" "$(MPLCONFIGDIR)"
	$(PYTHON) scripts/eda.py $(ARGS)

labels:
	@mkdir -p "$(TMPDIR)" "$(MPLCONFIGDIR)"
	$(PYTHON) scripts/build_labels.py $(ARGS)

recon:
	$(PYTHON) scripts/recon.py

features:
	@mkdir -p "$(TMPDIR)" "$(MPLCONFIGDIR)"
	$(PYTHON) scripts/build_features.py $(ARGS)

radius-sweep:
	@mkdir -p "$(TMPDIR)" "$(MPLCONFIGDIR)"
	$(PYTHON) scripts/radius_sweep.py

leakage:
	$(PYTHON) scripts/check_leakage.py

train:
	@mkdir -p "$(TMPDIR)" "$(MPLCONFIGDIR)"
	$(PYTHON) scripts/train.py $(ARGS)

member2: labels features radius-sweep leakage train

hpo:
	@mkdir -p "$(TMPDIR)" "$(MPLCONFIGDIR)"
	$(PYTHON) scripts/hpo.py $(ARGS)

routes:
	@mkdir -p "$(TMPDIR)" "$(MPLCONFIGDIR)"
	$(PYTHON) scripts/evaluate_routes.py $(ARGS)

check-pipeline:
	$(PYTHON) scripts/check_pipeline.py

predict-batch:
	$(PYTHON) scripts/predict_batch.py --partitions student_test train

member3: hpo routes check-pipeline predict-batch

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .tmp .mpl
