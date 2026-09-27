"""Revision results (records/revision.md). RP: replay of each arm's own training texts during the later stage;
RS: one epoch of No Robots at the reasoning rate; RL: relearning after the later stage, with a habit control;
SH: sharpening without repetition; SD: two more runs (seeds) with the drilled subset redrawn; SK: a skill the base
lacks within 2,048 tokens. Reads outputs/ and data/revision_ids.json; writes outputs/summaries/revision.json.

Each part reproduces DuraSeed-v1's analysis to the last digit: tools/analyze_revision.py for RP, RS, RL, SH (with
the SH runner's gate) and SD, tools/analyze_skill.py and run_skill.py's gate for SK. Accuracy is per problem (the
mean over four draws), under the primary score (the last boxed integer, `correct`) and, on MATH-500, the
secondary, lenient one (the text's last integer, `last_integer_correct`). A change is the mean of before - after
over problems, in points; its interval is a paired item bootstrap (20,000 resamples, 95%). Each MATH-500 part
draws all its intervals from one generator (seed 20260924) in the original's order, the lenient score's and
SH's two B intervals included, so the intervals are the original's; the gates and SK draw each interval from a
fresh generator with the same seed. The smoke and the calibration are shares of samples, as the runner records
them.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import stats

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
PILOT, MAIN, TEACHER = "pilot-20260924", "main-9b-20260924", "teacher-20260924"
ROBUST, FOLLOWUP = "robustness-20260924", "robustness-followup-20260924"  # R2 and R4: the stage without replay
RP, RS, RL, SH = "revision-rp-20260926", "revision-rs-20260926", "revision-rl-20260926", "revision-sh-20260926"
SEEDS = {"s2": "revision-sd-20260926/s2", "s3": "revision-sd-20260926/s3"}  # SD: a redrawn subset, order and init
SCORES = {"primary": "correct", "lenient": "last_integer_correct"}
HANDOFF = {"M0": PILOT, "D-u140": PILOT, "O-u140": PILOT, "P-u140": MAIN, "D-T-u140": TEACHER, "O-T-u140": TEACHER}
AFTER = {"D-u140+it": PILOT, "D-u140+b": PILOT, "D-T-u140+it": TEACHER, "D-T-u140+b": TEACHER, "O-u140+it": PILOT}
REFERENCE = {"D-u140+it": "O-u140+it", "D-u140+b": "O-u140+b", "D-T-u140+it": "O-T-u140+it", "D-T-u140+b": "O-T-u140+b"}
HABIT = ("D-u140+it", "D-u140+b")
SHARP, GATE_B = "O-sharp-t05-u140", 0.05  # SH's arm, sampled at temperature 0.5, and its gate: B at least 0.05
ARMS, STAGES = ("D-S", "O-S", "P-S"), {"it-lr1e-4": "primary", "it": "default"}  # SK
GATE_GAIN, GATE_CAPPED = 30.0, 0.15  # SK's gate: O-S-u140 at least 30 points above the base, at most 15% capped


def fresh():
    return np.random.default_rng(stats.SEED)


def change(before, after, rng=None, other=None):
    """Mean of before - after over problems in points, less another pair's change with `other`; with `rng`, its
    paired item-bootstrap interval (DuraSeed-v1's change_interval, draw for draw)."""
    ids = before.index
    loss = before.to_numpy() - after.loc[ids].to_numpy()
    if other is not None:
        loss = loss - (other[0].loc[ids].to_numpy() - other[1].loc[ids].to_numpy())
    out = {"points": float(100 * loss.mean())}
    if rng is not None:
        picks = rng.integers(0, len(ids), size=(stats.RESAMPLES, len(ids)))
        out["interval"] = stats.percentiles(100 * loss[picks].mean(1))
    return out


def recovered(start, now, reference, rng):
    """R_k = (acc_k - acc_0) / (acc_ref - acc_0), with its paired item-bootstrap interval."""
    x0, xk, xr = (a.loc[start.index].to_numpy() for a in (start, now, reference))
    picks = rng.integers(0, len(x0), size=(stats.RESAMPLES, len(x0)))
    boot = (xk[picks].mean(1) - x0[picks].mean(1)) / (xr[picks].mean(1) - x0[picks].mean(1))
    return {"R": float((xk.mean() - x0.mean()) / (xr.mean() - x0.mean())), "interval": stats.percentiles(boot)}


def b_value(texts, rng):
    """B on the base's common texts (sum of d over sum of n) and its item-bootstrap interval, as DuraSeed-v1's
    b_value computes them. The scores are float32 and it sums them in float32, so the per-text sums (their float32
    values, written out) are read back as float32."""
    d, n = texts.d.to_numpy(np.float32), texts.n.to_numpy()
    picks = rng.integers(0, len(d), size=(stats.RESAMPLES, len(d)))
    return {"B": float(d.sum() / n.sum()), "interval": stats.percentiles(d[picks].sum(1) / n[picks].sum(1))}


def mean(acc):
    return {s: 100 * float(acc[s].mean()) for s in SCORES}


def loss(pair, rng):
    """A (handoff, after) pair's loss in points, under both scores."""
    return {s: change(pair[0][s], pair[1][s], rng) for s in SCORES}


def excess(a, b, rng):
    """Pair a's loss minus pair b's, under both scores."""
    return {s: change(a[0][s], a[1][s], rng, other=(b[0][s], b[1][s])) for s in SCORES}


def evaluations(samples):
    """Per-problem forced MATH-500 accuracy under both scores, by (run, label, update)."""
    forced = samples[(samples.bench == "math500") & (samples["mode"] == "forced") & (samples.rule == "first")]
    return {key: {s: g.groupby("task_id")[column].mean() for s, column in SCORES.items()}
            for key, g in forced.groupby(["run", "label", "when"])}


def handoff(ev, label):
    return ev[HANDOFF[label], label, "u0"]


def rp(ev, rng):
    """RP: R2's 60 updates of instruction tuning at 1e-4 with 2 of each batch of 32 replayed from the arm's own
    training texts. The reference is the excess without replay (R2 for own solutions, R4 for teacher traces)."""
    reference = {}
    for drilled, once, run in (("D-u140", "O-u140", ROBUST), ("D-T-u140", "O-T-u140", FOLLOWUP)):
        (d0, d1), (o0, o1) = ((handoff(ev, x), ev[run, f"{x}+it-lr1e-4@60", "u60"]) for x in (drilled, once))
        reference[drilled] = change(d0["primary"], d1["primary"], other=(o0["primary"], o1["primary"]))["points"]
    out = {"accuracy": {}, "reference_excess": reference, "excess": {}, "reduction": {}, "readings": {},
           "predictions": {}}
    pairs = {}
    for label in ("D-u140", "O-u140", "D-T-u140", "O-T-u140"):
        pairs[label] = handoff(ev, label), ev[RP, f"{label}+it-replay", "u60"]
        out["accuracy"][label] = {"handoff": mean(pairs[label][0]), "u60": mean(pairs[label][1])}
    for drilled, once in (("D-u140", "O-u140"), ("D-T-u140", "O-T-u140")):
        e = out["excess"][drilled] = excess(pairs[drilled], pairs[once], rng)
        points, upper = e["primary"]["points"], e["primary"]["interval"][1]
        cut = out["reduction"][drilled] = 1 - points / reference[drilled]
        out["readings"][drilled] = ("replay prevents the break" if points <= 3 and upper < 6
                                    else "replay mitigates" if cut >= 0.5 else "replay fails" if cut < 0.25
                                    else "a reduction of 25-50%, which no registered reading names")
    out["predictions"]["own_solutions_mitigated"] = out["reduction"]["D-u140"] >= 0.5
    return out


def rs(ev, rng, rows):
    """RS: one pass over the No Robots rows at 1e-4, batch 32, from the base (a fresh adapter), D, O and P."""
    updates = len(rows) // 32
    out = {"updates": updates, "rows": len(rows), "accuracy": {}, "loss": {}, "excess": {}}
    pairs = {}
    for label in ("M0", "D-u140", "O-u140", "P-u140"):
        pairs[label] = handoff(ev, label), ev[RS, f"{label}+nr-epoch", f"u{updates}"]
        out["accuracy"][label] = {"handoff": mean(pairs[label][0]), "after": mean(pairs[label][1])}
        out["loss"][label] = loss(pairs[label], rng)
    d = out["excess"]["D-u140"] = excess(pairs["D-u140"], pairs["O-u140"], rng)["primary"]
    p = out["excess"]["P-u140"] = excess(pairs["P-u140"], pairs["O-u140"], rng)["primary"]
    holds = d["points"] >= 5 and d["interval"][0] > 0 and abs(p["points"]) <= 3
    out["predictions"] = {"D_over_O_at_least_5_above_zero": d["points"] >= 5 and d["interval"][0] > 0,
                          "P_over_O_within_3": abs(p["points"]) <= 3, "holds": holds}
    out["reading"] = ("the abstract leads with this number; the 3e-4 stage becomes the stress test" if holds
                      else "the break does not survive one realistic epoch" if d["interval"][0] <= 0
                      else "neither registered reading applies")
    return out


def rl(ev, rng):
    """RL: from each after-stage state, R_k after k updates of relearning (and of the habit control, after 20)
    against the matching once-trained after-state; from O+IT, the ceiling control, the loss instead."""
    out = {"accuracy": {}, "R": {}, "ceiling_loss": {}, "predictions": {}, "readings": {}}
    for origin, run in AFTER.items():
        start = ev[run, origin, "u20"]
        out["accuracy"][f"{origin}@0"] = mean(start)
        if origin in REFERENCE:
            target = ev[run, REFERENCE[origin], "u20"]
            out["accuracy"][f"{REFERENCE[origin]} (acc_ref)"] = mean(target)
        for tag in ("relearn", "habit"):
            for k in (1, 2, 5, 20):
                now, key = ev.get((RL, f"{origin}+{tag}", f"u{k}")), f"{origin}+{tag}@{k}"
                if now is None:  # the habit control runs from D+IT and D+AO, and is evaluated after update 20
                    continue
                out["accuracy"][key] = mean(now)
                if origin in REFERENCE:
                    out["R"][key] = {s: recovered(start[s], now[s], target[s], rng) for s in SCORES}
                else:
                    out["ceiling_loss"][key] = loss((start, now), rng)
    r, p = {key: v["primary"]["R"] for key, v in out["R"].items()}, out["predictions"]
    p["1_fast_relearning"] = {"R_5": r["D-u140+it+relearn@5"], "holds": r["D-u140+it+relearn@5"] >= 0.5}
    points = [out["ceiling_loss"][f"O-u140+it+relearn@{k}"]["primary"]["points"] for k in (1, 2, 5, 20)]
    p["2_ceiling_holds"] = {"loss_points": points, "holds": all(abs(x) <= 2 for x in points)}
    r5, r20 = r["D-u140+it+relearn@5"], r["D-u140+it+relearn@20"]
    out["readings"]["D+IT"] = ("suppressed: restored by about 160 examples of reasoning training" if r5 >= 0.75
                               else "not cheaply restored" if r20 <= 0.5 else "partly restored")
    for origin in HABIT:
        habit, math = r[f"{origin}+habit@20"], r[f"{origin}+relearn@20"]
        out["readings"][f"habit|{origin}"] = (
            "retraining the reasoning format alone restores much of the loss" if habit >= 0.5
            else "the loss concerns mathematical competence rather than format" if habit < 0.25 and math >= 0.5
            else "neither habit reading applies")
    return out


def sh(ev, texts, rng, sharp):
    """SH: O's problems, each with the base's own solution sampled at temperature 0.5, trained as O. The gate (the
    arm's B on the base's common texts), the paper's B of D and O, and the arm's excess over O after IT and AO."""
    def common(run, state):
        return texts[(texts.run == run) & (texts.set == "common") & (texts.state == state)]

    b = b_value(common(SH, SHARP), fresh())
    gate = {"arms": {SHARP: {"temperature": sharp["temperature"], "texts": len(sharp["task_ids"]), **b}},
            "final": SHARP, "gate": GATE_B}
    out = {"gate": gate, "gate_passed": b["B"] >= GATE_B, "accuracy": {}, "excess": {},
           "B_paper": {x: b_value(common(PILOT, x), rng)["B"] for x in ("D-u140", "O-u140")}}
    pairs = {}
    for name, run, label in (("O-sharp", SH, SHARP), ("O", PILOT, "O-u140"), ("D", PILOT, "D-u140")):
        start, after = ev[run, label, "u0"], {s: ev[run, f"{label}+{s}", "u20"] for s in ("it", "b")}
        pairs[name] = {s: (start, a) for s, a in after.items()}
        out["accuracy"][name] = {"handoff": mean(start), **{s: mean(a) for s, a in after.items()}}
    out["excess"] = {s: excess(pairs["O-sharp"][s], pairs["O"][s], rng) for s in ("it", "b")}
    d_ao = excess(pairs["D"]["b"], pairs["O"]["b"], rng)["primary"]["points"]
    ao = out["excess"]["b"]["primary"]
    breaks = ao["points"] >= 10 and ao["interval"][0] > 0
    within = all(abs(out["excess"][s]["primary"]["points"]) <= 3 for s in ("it", "b"))
    out["reading"] = ("sharpening alone breaks it" if breaks else "sharpening alone does not break it" if within
                      else "neither registered reading applies")
    out["prediction"] = {"D_excess_after_AO": d_ao, "holds": ao["points"] < d_ao / 2}
    return out


def sd(ev, rng):
    """SD: each run's excesses per later stage (IT at the defaults; IT at 1e-4 for 60 updates), with the paper's
    run and D2 beside the seeds; the seeds' predictions; and the pooled mean with the spread across runs."""
    later = {"it": ("it", "u20"), "it-lr1e-4": ("it-lr1e-4", "u60")}

    def arm(run, label, after=None):  # a state at handoff and after each later stage that was evaluated
        after = after or {s: (run, f"{label}+{tag}", when) for s, (tag, when) in later.items()}
        return {"u0": ev[run, label, "u0"], **{s: ev[key] for s, key in after.items() if key in ev}}

    pilot = {x: arm(PILOT, x, {"it": (PILOT, f"{x}+it", "u20"), "it-lr1e-4": (ROBUST, f"{x}+it-lr1e-4@60", "u60")})
             for x in ("D-u140", "O-u140")}
    runs = {"paper": {"D": pilot["D-u140"], "O": pilot["O-u140"], "P": arm(MAIN, "P-u140")},
            "D2": {"D": arm(MAIN, "D2-u140"), "O": pilot["O-u140"]},
            **{seed: {x: arm(run, f"{x}-u140") for x in "DOP"} for seed, run in SEEDS.items()}}
    out = {"accuracy": {}, "excess": {}, "predictions": {}, "pooled": {}}
    for name, run in runs.items():
        out["accuracy"][name] = {x: {s: mean(a) for s, a in states.items()} for x, states in run.items()}
        for s in later:
            for a, b in (("D", "O"), ("D", "P"), ("P", "O")):
                if a in run and b in run and s in run[a] and s in run[b]:
                    out["excess"].setdefault(name, {}).setdefault(s, {})[f"{a}-{b}"] = excess(
                        (run[a]["u0"], run[a][s]), (run[b]["u0"], run[b][s]), rng)["primary"]
    for seed in SEEDS:
        for s, e in out["excess"][seed].items():
            out["predictions"][f"{seed}|{s}"] = {
                "1_intervals_above_zero": e["D-O"]["interval"][0] > 0 and e["D-P"]["interval"][0] > 0,
                "2_at_least_5": e["D-O"]["points"] >= 5 and e["D-P"]["points"] >= 5,
                "3_P_within_3_of_O": abs(e["P-O"]["points"]) <= 3}
    for s in later:
        for c in ("D-O", "D-P", "P-O"):
            values = {name: e[s][c]["points"] for name, e in out["excess"].items() if c in e.get(s, {})}
            v = list(values.values())
            out["pooled"][f"{s}|{c}"] = {"runs": values, "mean": float(np.mean(v)), "min": min(v), "max": max(v),
                                         "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
    return out


def sk(skill):
    """SK: the base's smoke (4,096 and 8,192 tokens) and calibration (2,048) as shares of samples; the gate; and,
    per later stage, the losses, D-S's excess over O-S, O-S minus D-S after the stage, P-S minus the better of
    the two, and the registered outcomes read both ways (after-stage accuracy and extra loss)."""
    smoke = {f"{cap}|{setting}": {"correct": float(g.correct.mean()), "capped": float(g.capped.mean()),
                                  "mean_tokens": float(g.tokens.mean())}
             for (cap, setting), g in skill[skill.set == "smoke"].groupby(["cap", "setting"])}
    calibration = skill[skill.set == "calibration"].groupby("setting")
    calibration = {"accuracy": {s: float(g.correct.mean()) for s, g in calibration},
                   "capped": {s: float(g.capped.mean()) for s, g in calibration}}
    evals = skill[skill.set == "eval"]
    ev = {label: g.groupby("task_id").correct.mean() for label, g in evals.groupby("label")}
    acc = {name: 100 * float(v.mean()) for name, v in ev.items()}
    gain, capped = change(ev["O-S-u140"], ev["M0"], fresh()), float(evals[evals.label == "O-S-u140"].capped.mean())
    gate = {"gain": gain, "capped_share": capped,
            "passed": gain["points"] >= GATE_GAIN and gain["interval"][0] > 0 and capped <= GATE_CAPPED}
    hand = {a: ev[f"{a}-u140"] for a in ARMS}
    out = {"smoke": smoke, "calibration": calibration, "gate": gate, "accuracy": acc,
           "handoff_D_minus_O": change(hand["D-S"], hand["O-S"], fresh()), "stages": {},
           "handoff_P_minus_D": change(hand["P-S"], hand["D-S"], fresh())}  # post hoc, descriptive: not registered
    for stage, role in STAGES.items():
        after = {a: ev[f"{a}+{stage}"] for a in ARMS}
        lost = {a: change(hand[a], after[a], fresh()) for a in ARMS}
        extra = change(hand["D-S"], after["D-S"], fresh(), other=(hand["O-S"], after["O-S"]))
        below = change(after["O-S"], after["D-S"], fresh())  # how far D-S ends below O-S
        best_hand = max(("D-S", "O-S"), key=lambda a: acc[f"{a}-u140"])
        best_after = max(("D-S", "O-S"), key=lambda a: acc[f"{a}+{stage}"])
        fix = {"handoff": acc["P-S-u140"] - acc[f"{best_hand}-u140"],
               "after": acc[f"P-S+{stage}"] - acc[f"{best_after}+{stage}"]}
        out["stages"][stage] = {
            "role": role, "loss": lost, "D_excess_over_O": extra, "O_minus_D_after": below, "P_minus_best": fix,
            "outcomes": {
                "break_after_accuracy": below["points"] >= 10 and below["interval"][0] > 0,
                "break_extra_loss": extra["points"] >= 10 and extra["interval"][0] > 0,
                "consolidation_after_accuracy": abs(below["points"]) <= 3,
                "consolidation_extra_loss": abs(extra["points"]) <= 3,
                "fix_costs_nothing": abs(fix["handoff"]) <= 3 and abs(fix["after"]) <= 3}}
    return out


def summary():
    """Every part's results, from outputs/ and data/revision_ids.json."""
    ev = evaluations(pd.read_csv(OUT / "math_samples.csv.gz", dtype={"level": str}))
    texts = pd.read_csv(OUT / "depth_texts.csv.gz")
    ids = json.loads((REPO / "data" / "revision_ids.json").read_text())
    return {"RP": rp(ev, fresh()), "RS": rs(ev, fresh(), ids["rs_no_robots"]["ids"]), "RL": rl(ev, fresh()),
            "SH": sh(ev, texts, fresh(), ids["sh_sharp"]), "SD": sd(ev, fresh()),
            "SK": sk(pd.read_csv(OUT / "skill_samples.csv.gz"))}


def main():
    (OUT / "summaries").mkdir(exist_ok=True)
    (OUT / "summaries" / "revision.json").write_text(json.dumps(summary(), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
