"""run_tces.py finds a state in its run folder: its own drilled state first, else one supplied as <label>/state.json;
a state it has neither of fails with a message that says where to put it and where the paper's states are listed."""

import json
from types import SimpleNamespace

import pytest


def test_own_supplied_and_missing_states(tmp_path):
    run_tces = pytest.importorskip("run_tces")  # needs the experiment dependencies (tinker, transformers)
    run = SimpleNamespace(root=tmp_path)  # Tces.state reads only the run folder
    with pytest.raises(FileNotFoundError) as missing:
        run_tces.Tces.state(run, "M0")
    assert "data/checkpoints.json" in str(missing.value) and str(tmp_path / "M0" / "state.json") in str(missing.value)
    for path, value in ((tmp_path / "M0" / "state.json", "supplied M0"), (tmp_path / "F-u140" / "state.json", "supplied"),
                        (tmp_path / "F" / "stage_a" / "u140-state.json", "own")):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"state_path": value}))
    assert run_tces.Tces.state(run, "M0") == "supplied M0"
    assert run_tces.Tces.state(run, "F-u140") == "own"  # this run's own drilled state comes first
