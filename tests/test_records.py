"""The new modules reproduce the recorded numbers from the frozen DuraSeed-v1 runs.

- divergence.aggregate/summary reproduce the full-length TCES scoring (depth-full-20260924) and the
  math runs' depth values (pilot, 35B, Nemotron: B, A, kept part, top-token gain, B', the paired
  D-u140 - O-u140 difference and the base NLL);
- stats.loss reproduces every recorded loss and excess point estimate exactly, from per-problem
  accuracies read off the stored samples;
- divergence.text_nll reproduces the registered fit checks on the drilled texts (main-phase.md N2, anchor.md);
- analyze_revision reproduces every value of the revision parts' summaries exactly, intervals included, and
  data/revision_ids.json lists the training data those runs used.
Each test skips when its run is absent.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import DURASEED, REPO, common, needs

pytestmark = pytest.mark.duraseed  # every test here needs the original DuraSeed-v1 runs

MATH = DURASEED / "runs" / "math"
BODY = {"pilot-20260924": 3, "main-35b-20260924": 3, "main-nemotron-20260924": 2}  # forced prefix length
TOL = 1e-7


def summaries(run):
    root = needs(MATH / run, f"DuraSeed-v1 run {run}")
    return root, json.loads((root / "summary.json").read_text())


# ---- divergence ---------------------------------------------------------------------------------------

def test_full_length_tces_scoring():
    load, aggregate, summary = common("divergence", "load", "aggregate", "summary")
    root = needs(DURASEED / "runs" / "depth-profile" / "depth-full-20260924", "the full-length TCES scoring")
    recorded = json.loads((root / "summary.json").read_text())
    base = load(root / "M0" / "common.npz")
    checked = 0
    for label, want in recorded.items():
        if not isinstance(want, dict) or not (root / label / "common.npz").exists():
            continue
        got = summary(aggregate(base, load(root / label / "common.npz"), 3))
        assert abs(got["B"] - want["B"]) < TOL and abs(got["A"] - want["A"]) < TOL, label
        assert abs(got["alternative_probability_kept"] - want["alternative_probability_kept"]) < 1e-6, label
        if want.get("B_share_kept_top") is None:
            assert got["kept_share"] is None, label
        else:
            assert abs(got["kept_share"] - want["B_share_kept_top"]) < 1e-6, label
        checked += 1
    assert abs(summary(aggregate(base, base, 3))["base_nll"] - recorded["base_nll"]) < TOL
    assert checked >= 8


@pytest.mark.parametrize("run", sorted(BODY))
def test_math_depth_values(run):
    load, aggregate, summary = common("divergence", "load", "aggregate", "summary")
    root, recorded = summaries(run)
    depth, body = root / "depth", BODY[run]
    base = load(depth / "M0" / "common.npz")
    aggs = {}
    for label, want in recorded["depth"].items():
        aggs[label] = aggregate(base, load(depth / label / "common.npz"), body)
        got = summary(aggs[label])
        for key in ("B", "A", "B_kept_top", "top_probability_gain_where_kept"):
            if key in want:
                assert abs(got[key] - want[key]) < TOL, (run, label, key, got[key], want[key])
        if "B_own" in want:  # B' is the base-minus-state gap on the state's own texts, sign reversed
            own = summary(aggregate(load(depth / "M0" / f"own-{label}.npz"), load(depth / label / "own.npz"), body))
            assert abs(-own["B"] - want["B_own"]) < TOL, (run, label, "B_own")
    paired = recorded["B_D140_minus_O140"]
    diff = (np.sum(aggs["D-u140"]["d"]) - np.sum(aggs["O-u140"]["d"])) / np.sum(aggs["D-u140"]["n"])
    assert abs(diff - paired["diff"]) < TOL
    assert abs(summary(aggs["M0"])["base_nll"] - paired["base_nll"]) < TOL


# ---- stats.loss ------------------------------------------------------------------------------------------

def per_problem(path):
    sums = {}
    for row in map(json.loads, Path(path).open()):
        sums.setdefault(row["task_id"], []).append(float(row["correct"]))
    return {k: float(np.mean(v)) for k, v in sums.items()}


def forced(run_root, name, when, mode="forced"):
    return per_problem(run_root / "eval" / "math500" / name / f"{when}-{mode}.jsonl")


@pytest.mark.parametrize("run", sorted(BODY))
def test_losses_and_excesses_are_reproduced(run):
    loss = common("stats", "loss")
    root, recorded = summaries(run)
    acc = {}
    checked = 0
    for key, want in recorded["loss"].items():
        parts = key.split("|")
        if " minus " in parts[0]:
            first, other = parts[0].split(" minus ")
            stage, mode = parts[1], "forced"
            names = [(first, "u0"), (f"{first}+{stage}", "u20"), (other, "u0"), (f"{other}+{stage}", "u20")]
        else:
            label, stage, mode = parts
            names = [(label, "u0"), (f"{label}+{stage}", "u20")]
        names = [(name, when, mode) for name, when in names]
        for name, when, m in names:
            acc.setdefault((name, when, m), forced(root, name, when, m))
        ids = sorted(acc[names[0]])
        vec = [np.array([acc[n][i] for i in ids]) for n in names]
        got = loss(vec[0], vec[1]) if len(vec) == 2 else loss(vec[0], vec[1], other=(vec[2], vec[3]))
        assert abs(got["points"] - want["points"]) < 1e-9, (run, key, got["points"], want["points"])
        checked += 1
    assert checked >= 24


def test_main_phase_9b_excesses_are_reproduced():
    loss = common("stats", "loss")
    pilot = needs(MATH / "pilot-20260924", "the math pilot")
    main, recorded = summaries("main-9b-20260924")
    runs = {"D-u140": pilot, "O-u140": pilot, "P-u140": main, "D2-u140": main, "D-r128-u140": main}

    def vectors(label, stage):
        before, after = forced(runs[label], label, "u0"), forced(runs[label], f"{label}+{stage}", "u20")
        ids = sorted(before)
        return np.array([before[i] for i in ids]), np.array([after[i] for i in ids])

    p = recorded["predictions"]
    for group, first, other in (("M3", "D-u140", "P-u140"), ("M3", "P-u140", "O-u140"),
                                ("M3", "D-u140", "O-u140"), ("M2", "D2-u140", "O-u140")):
        for stage in ("it", "b"):
            got = loss(*vectors(first, stage), other=vectors(other, stage))
            want = p[group][f"{first.split('-u')[0].replace('-r128', '')}_minus_{other.split('-u')[0]}"][stage]
            assert abs(got["points"] - want["points"]) < 1e-9, (first, other, stage)
    for stage in ("it", "b"):
        ratio = loss(*vectors("D-r128-u140", stage))["points"] / loss(*vectors("D-u140", stage))["points"]
        assert abs(ratio - p["N2"]["loss_ratio_to_rank32"][stage]) < 1e-9


def test_drilled_text_fit_reproduces_the_registered_checks():
    """Both checks average the NLL over every scored token of the drilled texts, the forced prefix included, from
    float32 scores; the pipeline sums per-text totals, so it agrees to float32 precision."""
    text_nll = common("divergence", "text_nll")
    texts = pd.read_csv(REPO / "outputs" / "depth_texts.csv.gz")
    for run, path, fields in (("main-9b-20260924", MATH / "main-9b-20260924", ("predictions", "N2", "drilled_text_nll")),
                              ("anchor-20260924", DURASEED / "runs/anchor/anchor-20260924", ("drilled_texts", "nll_per_token"))):
        want = json.loads(needs(path / "summary.json", f"DuraSeed-v1 run {run}").read_text())
        for field in fields:
            want = want[field]
        got = text_nll(texts[(texts.run == run) & (texts.set == "drilled")])
        assert got.keys() == want.keys(), run
        for label, value in want.items():
            assert abs(got[label] - value) < 1e-6, (run, label, got[label], value)


# ---- the revision (records/revision.md) --------------------------------------------------------------------

REVISION = {"RP": "math/revision-rp-20260926", "RS": "math/revision-rs-20260926", "RL": "math/revision-rl-20260926",
            "SH": "math/revision-sh-20260926", "SD": "math/revision-sd-20260926", "SK": "skill/skill-20260926"}


def unmatched(want, got, path=""):
    """The leaves of `want` that `got` does not hold exactly, with the same type."""
    if isinstance(want, dict):
        return [m for key in want for m in unmatched(want[key], got.get(key) if isinstance(got, dict) else None,
                                                      f"{path}/{key}")]
    if isinstance(want, list) and isinstance(got, list) and len(want) == len(got):
        return [m for i, (w, g) in enumerate(zip(want, got)) for m in unmatched(w, g, f"{path}[{i}]")]
    return [] if want == got and type(want) is type(got) else [(path, want, got)]


def test_revision_is_reproduced_exactly():
    roots = {part: needs(DURASEED / "runs" / run, f"DuraSeed-v1 run {run}") for part, run in REVISION.items()}
    import analyze_revision  # from this repository's outputs/ and data/

    got = analyze_revision.summary()
    for part, root in roots.items():
        want = json.loads((root / "summary.json").read_text())
        assert not unmatched(want, got[part]), (part, unmatched(want, got[part])[:5])
    gate = json.loads((roots["SK"] / "gate.json").read_text())
    del gate["rule"]  # the rule in words
    assert not unmatched(gate, got["SK"]["gate"])
    assert not unmatched(json.loads((roots["SK"] / "calibration.json").read_text())["accuracy"],
                         got["SK"]["calibration"]["accuracy"])


def test_revision_ids_equal_the_runs():
    frozen = json.loads((REPO / "data" / "revision_ids.json").read_text())

    def kept(run, name, key, keep):
        return [r[key] for r in map(json.loads, needs(MATH / run / name, f"{run}/{name}").open()) if keep(r)]

    relearn = kept("revision-rl-20260926", "rft.jsonl", "task_id", lambda r: r["correct"])[:640]  # the first 640 kept
    assert frozen["rl_relearn"]["task_ids"] == relearn
    assert frozen["rl_habit"]["ids"] == kept("revision-rl-20260926", "habit.jsonl", "id", lambda r: r["kept"])
    assert frozen["sh_sharp"]["task_ids"] == kept("revision-sh-20260926", "sharp-t05.jsonl", "task_id",
                                                  lambda r: r["generation"])
    examples = needs(MATH / "revision-rs-20260926" / "examples.json", "RS's examples.json")
    assert frozen["rs_no_robots"]["ids"] == json.loads(examples.read_text())["examples"]
    for seed in ("s2", "s3"):
        arms = needs(MATH / "revision-sd-20260926" / seed / "arms.json", f"SD {seed}'s arms.json")
        assert frozen[f"sd_{seed}"]["task_ids"] == json.loads(arms.read_text())["D"], seed
