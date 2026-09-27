#!/usr/bin/env python3
"""Part SK's registered outcomes (records/revision.md), from a run of src/run_skill.py (runs/skill/<run-id>/eval/),
written to the run's summary.json.

Per later stage (IT at 1e-4 for 60 updates, the primary; IT at the defaults), from the forced evaluations on the
300 held-out problems:
  break               D-S at least 10 points below O-S after the stage, interval above zero; read both ways, as
                      after-stage accuracy (the registered wording) and as extra loss from each arm's handoff
  consolidation       D-S within 3 points of O-S after the stage, read the same two ways
  the fix costs nothing  P-S within 3 points of the better of D-S and O-S, at handoff and after the stage
and, descriptively, D-S against O-S at handoff. Intervals are paired item bootstraps (20,000 resamples), each
from a fresh generator seeded 20260924, with DuraSeed-v1's arithmetic (tools/analyze_skill.py), so that its
summary.json and run_skill.py's gate.json are reproduced to the last digit. src/analyze_revision.py computes the
same from outputs/.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import stats

REPO = Path(__file__).resolve().parents[1]
ARMS, STAGES = ("D-S", "O-S", "P-S"), {"it-lr1e-4": "primary", "it": "default"}


def per_problem(path):
    """Each problem's share of correct draws, keyed by task id."""
    draws = {}
    for row in map(json.loads, path.open()):
        draws.setdefault(row["task_id"], []).append(float(row["correct"]))
    return {task_id: sum(v) / len(v) for task_id, v in draws.items()}


def change(before, after, other=None):
    """Mean of before - after over problems in points, less another (before, after) pair's with `other`, and its
    paired item-bootstrap interval."""
    ids = sorted(before)
    loss = np.array([before[i] - after[i] for i in ids])
    if other is not None:
        loss = loss - np.array([other[0][i] - other[1][i] for i in ids])
    picks = np.random.default_rng(stats.SEED).integers(0, len(ids), size=(stats.RESAMPLES, len(ids)))
    return {"points": float(100 * loss.mean()), "interval": stats.percentiles(100 * loss[picks].mean(1))}


def summary(root):
    ev = {name: per_problem(root / "eval" / f"{name}.jsonl") for name in
          ["M0", *(f"{a}-u140" for a in ARMS), *(f"{a}+{s}" for a in ARMS for s in STAGES)]}
    acc = {name: 100 * float(np.mean(list(v.values()))) for name, v in ev.items()}
    hand = {a: ev[f"{a}-u140"] for a in ARMS}
    out = {"accuracy": acc, "handoff_D_minus_O": change(hand["D-S"], hand["O-S"]), "stages": {}}
    for stage, role in STAGES.items():
        after = {a: ev[f"{a}+{stage}"] for a in ARMS}
        excess = change(hand["D-S"], after["D-S"], other=(hand["O-S"], after["O-S"]))
        below = change(after["O-S"], after["D-S"])  # how far D-S ends below O-S
        best_hand = max(("D-S", "O-S"), key=lambda a: acc[f"{a}-u140"])
        best_after = max(("D-S", "O-S"), key=lambda a: acc[f"{a}+{stage}"])
        fix = {"handoff": acc["P-S-u140"] - acc[f"{best_hand}-u140"],
               "after": acc[f"P-S+{stage}"] - acc[f"{best_after}+{stage}"]}
        out["stages"][stage] = {
            "role": role, "loss": {a: change(hand[a], after[a]) for a in ARMS}, "D_excess_over_O": excess,
            "O_minus_D_after": below, "P_minus_best": fix,
            "outcomes": {
                "break_after_accuracy": below["points"] >= 10 and below["interval"][0] > 0,
                "break_extra_loss": excess["points"] >= 10 and excess["interval"][0] > 0,
                "consolidation_after_accuracy": abs(below["points"]) <= 3,
                "consolidation_extra_loss": abs(excess["points"]) <= 3,
                "fix_costs_nothing": abs(fix["handoff"]) <= 3 and abs(fix["after"]) <= 3}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="skill-20260926", help="under runs/skill")
    root = REPO / "runs/skill" / parser.parse_args().run_id
    out = summary(root)
    (root / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
