# PathSnap: common tasks.
#
# Assumes the virtual environment in .venv is active, or set PYTHON explicitly:
#     make validate PYTHON=/path/to/venv/bin/python

PYTHON ?= python3
SHELL  := /bin/bash

# Some sandboxes make the default temp/matplotlib directories unwritable.
export TMPDIR      ?= $(CURDIR)/.tmp
export MPLCONFIGDIR ?= $(CURDIR)/.mpl

.PHONY: help setup validate eda recon clean

help:
	@echo "PathSnap targets"
	@echo "  setup      install Python dependencies"
	@echo "  validate   parse and validate every supplied trip"
	@echo "  eda        dataset measurements, figures and Folium maps"
	@echo "  recon      dataset reconnaissance summary"
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

recon:
	$(PYTHON) scripts/recon.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .tmp .mpl
