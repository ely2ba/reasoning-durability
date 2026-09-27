"""Shared test helpers: repository paths, the frozen DuraSeed-v1 runs, and imports of src/common.

Tests that need data outside this repository (the DuraSeed-v1 runs, the Hugging Face cache) skip
cleanly when it is absent. DuraSeed-v1 is looked for beside this repository (as the Makefile's RUNS);
set DURASEED_V1 or HF_HUB_CACHE to point elsewhere.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tests never touch the network: tokenizers come from the local cache
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
DURASEED = Path(os.environ.get("DURASEED_V1", REPO.parent / "DuraSeed-v1"))
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
FULL_REPLAY = os.environ.get("REPLAY_FULL") == "1"  # otherwise replay tests read a fixed sample per file

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def common(module, *names):
    """Return names from src/common/<module>.py; skip (not fail) while the module or names are missing.

    A module that exists but fails to import for another reason raises, so real errors surface.
    """
    try:
        mod = importlib.import_module(f"common.{module}")
    except ModuleNotFoundError as error:
        if error.name and (error.name == "common" or error.name.startswith("common.")):
            pytest.skip(f"src/common/{module}.py is not written yet")
        raise
    missing = [name for name in names if not hasattr(mod, name)]
    if missing:
        pytest.skip(f"src/common/{module}.py does not define {missing} yet")
    values = [getattr(mod, name) for name in names]
    return values[0] if len(values) == 1 else values


def first_of(module, *candidates):
    """The first of several candidate function names that src/common/<module>.py defines."""
    try:
        mod = importlib.import_module(f"common.{module}")
    except ModuleNotFoundError as error:
        if error.name and (error.name == "common" or error.name.startswith("common.")):
            pytest.skip(f"src/common/{module}.py is not written yet")
        raise
    for name in candidates:
        if hasattr(mod, name):
            return getattr(mod, name)
    pytest.skip(f"src/common/{module}.py defines none of {list(candidates)} yet")


def needs(path, what):
    """Skip unless `path` exists."""
    if not Path(path).exists():
        pytest.skip(f"{what} not found at {path}")
    return Path(path)


def sample_rows(lines, per_file=60):
    """All rows under REPLAY_FULL=1, else an evenly spaced fixed sample (deterministic)."""
    if FULL_REPLAY or len(lines) <= per_file:
        return lines
    step = len(lines) / per_file
    return [lines[int(i * step)] for i in range(per_file)]
