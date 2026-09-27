RUNS ?= ../DuraSeed-v1/runs
PYTHON ?= python
TECTONIC ?= tectonic
export PYTHONPATH := src
export HF_HUB_OFFLINE := 1

.PHONY: all outputs summaries numbers check figures test inputs

all: summaries numbers check figures test

outputs: ; $(PYTHON) src/collect.py --runs $(RUNS)
summaries: ; $(PYTHON) src/analyze_math.py && $(PYTHON) src/analyze_tces.py && $(PYTHON) src/analyze_depth.py && $(PYTHON) src/analyze_revision.py
numbers: ; $(PYTHON) src/paper_numbers.py
# recompute every number; fail if the written file differs or the paper uses an undefined key, and list
# the keys the paper does not use (FINAL=1 also fails while a key is pending)
check: ; $(PYTHON) src/paper_numbers.py --check
# a fixed PDF creation date (2026-09-24 00:00 UTC), so that regenerated figures are byte-identical
figures: ; SOURCE_DATE_EPOCH=1790208000 $(PYTHON) src/figures.py
test: ; $(PYTHON) -m pytest -q tests
inputs: ; $(PYTHON) src/build_inputs.py
