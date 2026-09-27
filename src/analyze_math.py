"""Math results (paper sections 3-5): accuracy at handoff and after each later stage, losses and
excess losses with paired item-bootstrap intervals, AIME, how samples end, the re-forcing probe, and
divergence from the base. Reads outputs/; writes outputs/summaries/math.json.

Accuracy is per problem (the mean over draws), then averaged over the 221 MATH-500 problems of
levels 3-5. A condition is a state at handoff ("u0") or after 20 updates of a stage ("+it", "+b").
"""

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import divergence, stats

OUT = Path(__file__).resolve().parents[1] / "outputs"
PILOT, MAIN, TEACHER = "pilot-20260924", "main-9b-20260924", "teacher-20260924"
FOLLOWUP = "robustness-followup-20260924"  # R4 and R5
FAMILY_RUNS = (PILOT, "main-35b-20260924", "main-nemotron-20260924")  # full pilot designs
EVALUATED = ("M0", "D-u20", "D-u60", "D-u140", "O-u140")
SCORED = ("M0", "D-u20", "D-u60", "D-u140", "O-u60", "O-u140")
MAIN_ARMS = ("P-u140", "D2-u140", "D-r128-u140")
STAGES = ("it", "b")
BUDGETS = (2048, 4096, 8192)


def name(label, stage):
    return (label, "u0") if stage is None else (f"{label}+{stage}", "u20")


class Samples:
    def __init__(self, table):
        self.groups = dict(tuple(table.groupby(["run", "bench", "label", "when", "mode", "rule"])))

    def rows(self, run, label, stage=None, mode="forced", rule="first", bench="math500"):
        label, when = name(label, stage)
        return self.groups.get((run, bench, label, when, mode, rule))

    def per_problem(self, *args, key="correct", **kwargs):
        rows = self.rows(*args, **kwargs)
        return None if rows is None else rows.groupby("task_id")[key].mean()


def loss(before, after, other=None):
    ids = before.index
    if other is not None:
        other = (other[0].loc[ids].to_numpy(), other[1].loc[ids].to_numpy())
    return stats.loss(before.to_numpy(), after.loc[ids].to_numpy(), other)


def failure_form(rows, cap):
    """How the samples of one condition end: lengths, cap hits, missing boxes, loops, budgets."""
    out = {"mean_tokens": float(rows.tokens.mean()), "median_tokens": float(rows.tokens.median()),
           "under_100": 100 * float((rows.tokens < 100).mean()), "capped": 100 * float(rows.capped.mean()),
           "unboxed": 100 * float((rows.boxed == 0).mean()), "ends_turn": 100 * float((rows.end == "user").mean()),
           "closed_think": 100 * float(rows.closed_think.mean()),
           "lenient": 100 * float(rows.lenient_correct.mean()),
           "capped_looping": 100 * float(rows.looping[rows.capped == 1].mean()) if rows.capped.any() else 0.0}
    for k in BUDGETS:
        if k < cap:
            out[f"within_{k}"] = 100 * float(rows[f"within_{k}"].mean())
    return out


def depth(texts, run, labels):
    """B, A, the kept/flipped split and their intervals on the base's common texts; B' on own texts."""
    out, sums = {}, {}
    for label in labels:
        rows = texts[(texts.run == run) & (texts.set == "common") & (texts.state == label)]
        if rows.empty:
            continue
        rows = rows.sort_values("task_id")
        out[label] = divergence.summary(rows) | {"B_interval": stats.ratio_interval(rows.d, rows.n)["interval"]}
        sums[label] = rows.set_index("task_id")[["d", "n"]]
        own = texts[(texts.run == run) & (texts.set == "own") & (texts.state == label)]
        if not own.empty:
            b_own = stats.ratio_interval(-own.d, own.n)
            out[label] |= {"B_own": b_own["estimate"], "B_own_interval": b_own["interval"]}
    return out, sums


def paired_b(sums, x, y):
    ids = sums[x].index
    diff = sums[x].d - sums[y].d.loc[ids]
    return stats.ratio_interval(diff.to_numpy(), sums[x].n.to_numpy())


def family_run(s, texts, run):
    """The pilot design on one base model: states, both stages, AIME, depth and the registered
    predictions of docs/experiments/math-pilot.md (applied unchanged to the 35B and Nemotron)."""
    acc, think, lost = {}, {}, {}
    for label in EVALUATED:
        for stage in (None, *STAGES):
            for mode in ("forced", "unforced"):
                key = f"{name(label, stage)[0]}|{mode}"
                p = s.per_problem(run, label, stage, mode)
                if p is not None:
                    acc[key] = 100 * float(p.mean())
                    think[key] = 100 * float(s.per_problem(run, label, stage, mode, key="think").mean())
    for label in EVALUATED:
        for stage in STAGES:
            for mode in ("forced", "unforced"):
                before, after = s.per_problem(run, label, None, mode), s.per_problem(run, label, stage, mode)
                if before is not None and after is not None:
                    lost[f"{label}|{stage}|{mode}"] = loss(before, after)
    excess = {}
    for stage in STAGES:
        for other in ("D-u20", "O-u140"):
            parts = [s.per_problem(run, x, st) for x in ("D-u140", other) for st in (None, stage)]
            if all(p is not None for p in parts):
                excess[f"D-u140 minus {other}|{stage}"] = loss(parts[0], parts[1], other=(parts[2], parts[3]))
    aime = {}
    for label in ("M0", "D-u140", "O-u140"):
        for stage in (None, "it"):
            p = s.per_problem(run, label, stage, bench="aime")
            if p is not None:
                aime[name(label, stage)[0]] = 100 * float(p.mean())
    states, sums = depth(texts, run, SCORED)
    out = {"accuracy": acc, "think_open": think, "loss": lost, "excess": excess, "aime_forced": aime, "depth": states}
    if {"D-u140", "O-u140"} <= set(sums):
        paired = paired_b(sums, "D-u140", "O-u140")
        nll = states["M0"]["base_nll"]
        out["B_D140_minus_O140"] = {"diff": paired["estimate"], "interval": paired["interval"], "base_nll": nll}
        out["predictions"] = predictions(out)
    return out


def predictions(r):
    """The pilot's registered predictions 1-5 and its handoff parity check."""
    b, lost, paired = {k: v["B"] for k, v in r["depth"].items()}, r["loss"], r["B_D140_minus_O140"]
    p = {"1": paired["interval"][0] > 0 and paired["diff"] >= 0.05 * paired["base_nll"],
         "2": b.get("D-u20", np.nan) < b.get("D-u60", np.nan) < b.get("D-u140", np.nan)}
    for stage in STAGES:
        keys = [f"D-u140 minus {o}|{stage}" for o in ("D-u20", "O-u140")]
        if all(k in r["excess"] for k in keys):
            p[f"3|{stage}"] = all(r["excess"][k]["interval"][0] > 0 for k in keys)
        if all(f"{x}|{stage}|forced" in lost for x in EVALUATED):
            p[f"4|{stage}"] = stats.spearman([b[x] for x in EVALUATED],
                                             [lost[f"{x}|{stage}|forced"]["points"] for x in EVALUATED])
        for x in ("M0", "O-u140"):
            if f"{x}|{stage}|unforced" in lost:
                p[f"5|{x}|{stage}"] = lost[f"{x}|{stage}|unforced"]["points"] > lost[f"{x}|{stage}|forced"]["points"]
    gap = abs(r["accuracy"]["D-u140|forced"] - r["accuracy"]["O-u140|forced"])
    p["parity"] = {"handoff_forced_gap": gap, "holds": gap <= 5}
    return p


def main_arms(s, texts):
    """The 9B main arms against the pilot's D-u140 and O-u140 on the same problems
    (docs/experiments/main-phase.md, items M2, M3 and N2)."""
    out = {"accuracy": {}, "excess": {}, "loss": {}}
    refs = {x: (PILOT, x) for x in ("D-u140", "O-u140")} | {x: (MAIN, x) for x in MAIN_ARMS}
    have = {x for x, (run, label) in refs.items() if s.rows(run, label, "b") is not None}
    for x in have:
        run, label = refs[x]
        for stage in (None, *STAGES):
            out["accuracy"][f"{name(label, stage)[0]}|forced"] = 100 * float(s.per_problem(run, label, stage).mean())
        for stage in STAGES:
            out["loss"][f"{x}|{stage}"] = loss(s.per_problem(run, label), s.per_problem(run, label, stage))
    for x in have & set(MAIN_ARMS):
        for y in ("D-u140", "O-u140"):
            for stage in STAGES:
                a, b = (s.per_problem(*refs[z]) for z in (x, y))
                a2, b2 = (s.per_problem(*refs[z], stage) for z in (x, y))
                out["excess"][f"{x} minus {y}|{stage}"] = loss(a, a2, other=(b, b2))
    states, _ = depth(texts, MAIN, MAIN_ARMS)
    out["depth"] = states
    drilled = texts[(texts.run == MAIN) & (texts.set == "drilled")]
    if not drilled.empty:  # N2's registered fit check: every scored token of D's texts, the forced prefix included
        out["drilled_text_nll"] = divergence.text_nll(drilled)
    return out


def teacher_run(s, texts):
    """records/teacher.md: the pilot design on gpt-oss-120b's traces (D-T, D-T-u20, O-T, P-T)."""
    states, out = ("D-T-u20", "D-T-u140", "O-T-u140", "P-T-u140"), {"accuracy": {}, "loss": {}, "excess": {}}
    for label in states:
        for stage in (None, *STAGES):
            p = s.per_problem(TEACHER, label, stage)
            if p is not None:
                out["accuracy"][f"{name(label, stage)[0]}|forced"] = 100 * float(p.mean())
        free = s.per_problem(TEACHER, label, None, "unforced")
        if free is not None:
            out["accuracy"][f"{label}|unforced"] = 100 * float(free.mean())
        for stage in STAGES:
            before, after = s.per_problem(TEACHER, label), s.per_problem(TEACHER, label, stage)
            if before is not None and after is not None:
                out["loss"][f"{label}|{stage}"] = loss(before, after)
    for x, y in (("D-T-u140", "O-T-u140"), ("D-T-u140", "D-T-u20"), ("P-T-u140", "O-T-u140")):
        for stage in STAGES:
            parts = [s.per_problem(TEACHER, z, st) for z in (x, y) for st in (None, stage)]
            if all(p is not None for p in parts):
                out["excess"][f"{x} minus {y}|{stage}"] = loss(parts[0], parts[1], other=(parts[2], parts[3]))
    out["aime_forced"] = {name(label, stage)[0]: 100 * float(p.mean()) for label in ("D-T-u140", "O-T-u140")
                          for stage in (None, "it") if (p := s.per_problem(TEACHER, label, stage, bench="aime")) is not None}
    out["depth"], _ = depth(texts, TEACHER, states)
    for kind in ("displacement", "displacement_common"):
        rows = texts[(texts.run == TEACHER) & (texts.set == kind)]
        out[kind] = {state: float(g.d.sum() / g.n.sum()) for state, g in rows.groupby("state")}
    held = texts[(texts.run == TEACHER) & (texts.set == "heldout")]
    if not held.empty:
        first = held[held.state == held.state.iloc[0]]
        out["heldout_nll"] = {"M0": float(-first.base_lp.sum() / first.n.sum())} | {
            state: float(-(g.base_lp.sum() - g.d.sum()) / g.n.sum()) for state, g in held.groupby("state")}
    return out


def robustness(s, texts):
    """records/robustness.md. R1: displacement of each state's own reasoning after k instruction-tuning
    updates, and forced accuracy of D and O; R2: the drilled excess at learning rate 1e-4."""
    run, out = "robustness-20260924", {"delta": {}, "forced": {}, "predictions": {}}
    for state, g in texts[(texts.run == run) & (texts.set == "displacement_common")].groupby("state"):
        out.setdefault("delta_common", {})[state] = float(g.d.sum() / g.n.sum())
        out["delta_common"][f"{state}|interval"] = stats.ratio_interval(g.sort_values("task_id").d,
                                                                       g.sort_values("task_id").n)["interval"]
    rows = texts[(texts.run == run) & (texts.set == "displacement")]
    draws = {}
    for state, g in rows.groupby("state"):
        g = g.sort_values("task_id")
        out["delta"][state] = float(g.d.sum() / g.n.sum())
        out["delta"][f"{state}|over_handoff_nll"] = float(g.d.sum() / -g.base_lp.sum())
        draws[state] = stats.ratio_draws(g.d, g.n)
        out["delta"][f"{state}|interval"] = stats.percentiles(draws[state])
    if {"D-u140@u2", "O-u140@u2", "P-u140@u2"} <= set(draws):
        stat = draws["D-u140@u2"] - 2 * np.maximum(draws["O-u140@u2"], draws["P-u140@u2"])
        out["predictions"]["R1.1"] = {"difference": out["delta"]["D-u140@u2"] - 2 * max(
            out["delta"]["O-u140@u2"], out["delta"]["P-u140@u2"]), "interval": stats.percentiles(stat)}
    for label in ("D-u140", "O-u140"):
        points = {"u0": s.per_problem(PILOT, label), "u20": s.per_problem(PILOT, label, "it")}
        for tag, k in (("", 2), ("", 5), ("-long", 40), ("-long", 60)):
            rows_k = s.groups.get((run, "math500", f"{label}+it{tag}@{k}", f"u{k}", "forced", "first"))
            if rows_k is not None:
                points[f"u{k}"] = rows_k.groupby("task_id").correct.mean()
        out["forced"][label] = {k: 100 * float(v.mean()) for k, v in points.items()}
        for k in (20, 60):
            rows_k = s.groups.get((run, "math500", f"{label}+it-lr1e-4@{k}", f"u{k}", "forced", "first"))
            if rows_k is not None:
                points[f"lr{k}"] = rows_k.groupby("task_id").correct.mean()
                out["forced"][label][f"lr1e-4@{k}"] = 100 * float(points[f"lr{k}"].mean())
        out.setdefault("_points", {})[label] = points
    points = out.pop("_points", {})
    for k in (20, 60):
        if all(f"lr{k}" in points.get(x, {}) for x in ("D-u140", "O-u140")):
            d, o = points["D-u140"], points["O-u140"]
            out[f"excess_lr1e-4@{k}"] = loss(d["u0"], d[f"lr{k}"], other=(o["u0"], o[f"lr{k}"]))
    for k in (20, 40, 60):
        if all(f"u{k}" in points.get(x, {}) for x in ("D-u140", "O-u140")):
            d, o = points["D-u140"], points["O-u140"]
            out[f"excess_it@{k}"] = loss(d["u0"], d[f"u{k}"], other=(o["u0"], o[f"u{k}"]))
    return out


def followup(s, train, out):
    """records/robustness.md, R4 and R5 (as DuraSeed-v1's analyze_robustness.followup), added to `out`:
    forced accuracy after the 1e-4 stage (R4: instruction tuning from the teacher states, after 20 and
    60 updates; R5: broad chat data from all four states, after 60), the drilled-minus-once-trained
    excesses, R5.3's losses, the training loss over updates 1-5 and 56-60, and R5's data mix."""
    handoff = {"D-u140": PILOT, "O-u140": PILOT, "D-T-u140": TEACHER, "O-T-u140": TEACHER}
    acc = {label: {0: s.per_problem(run, label)} for label, run in handoff.items()}
    for part, tag, saves, labels in (("r4", "it-lr1e-4", (20, 60), ("D-T-u140", "O-T-u140")),
                                     ("r5", "chat-lr1e-4", (60,), tuple(handoff))):
        out[part] = {"forced": {}, "train_loss": {}}
        for label in labels:
            for k in saves:
                rows = s.groups.get((FOLLOWUP, "math500", f"{label}+{tag}@{k}", f"u{k}", "forced", "first"))
                if rows is not None:
                    acc[label][f"{part}@{k}"] = rows.groupby("task_id").correct.mean()
                    out[part]["forced"][f"{label}@{k}"] = 100 * float(acc[label][f"{part}@{k}"].mean())
            logs = train[(train.run == FOLLOWUP) & (train.stage == part) & (train.label == label)]
            per_example = logs.set_index("update").loss_sum.sort_index() / 32
            if {*range(1, 6), *range(56, 61)} <= set(per_example.index):
                out[part]["train_loss"][label] = {"u1_5": float(per_example.loc[1:5].mean()),
                                                  "u56_60": float(per_example.loc[56:60].mean())}
    for part, k, drilled, once in (("r4", 20, "D-T-u140", "O-T-u140"), ("r4", 60, "D-T-u140", "O-T-u140"),
                                   ("r5", 60, "D-u140", "O-u140"), ("r5", 60, "D-T-u140", "O-T-u140")):
        key = f"{part}@{k}"
        if key in acc[drilled] and key in acc[once]:
            out[part][f"excess|{drilled}|u{k}"] = loss(acc[drilled][0], acc[drilled][key],
                                                       other=(acc[once][0], acc[once][key]))
    p = out["predictions"]
    if e := out["r4"].get("excess|D-T-u140|u60"):
        p["R4.1"] = {"excess": e["points"], "interval": e["interval"], "holds": e["points"] >= 10 and e["interval"][0] > 0}
    for prediction, drilled in (("R5.1", "D-u140"), ("R5.2", "D-T-u140")):
        if e := out["r5"].get(f"excess|{drilled}|u60"):
            p[prediction] = {"excess": e["points"], "interval": e["interval"], "holds": e["interval"][0] > 0,
                             "at_least_10": e["points"] >= 10}
    forced = out["r5"]["forced"]
    if all(f"{x}@60" in forced for x in ("O-u140", "O-T-u140")):
        lost = {x: 100 * float(acc[x][0].mean()) - forced[f"{x}@60"] for x in ("O-u140", "O-T-u140")}
        p["R5.3"] = {"lost": lost, "holds": max(lost.values()) <= 5}
    chat = json.loads((OUT.parent / "data" / "chat_ids.json").read_text())["rows"]  # R5's frozen data
    out["r5_data"] = {"rows": len(chat), "mix": dict(collections.Counter(r["source"] for r in chat).most_common()),
                      "mathlike": sum(r["mathlike"] for r in chat),
                      "mathlike_share": 100 * sum(r["mathlike"] for r in chat) / len(chat)}


def reforcing(s, run, labels):
    """docs/experiments/reforcing.md: accuracy under each continuation rule and the excess of
    D-u140 over O-u140 with every condition scored under the rule."""
    out = {}
    for rule in ("first", "budget", "answer"):
        acc = {}
        for label in labels:
            for stage in (None, *STAGES):
                rows = s.rows(run, label, stage, rule=rule)
                if rows is not None:
                    acc[name(label, stage)[0]] = {
                        "accuracy": 100 * float(rows.groupby("task_id").correct.mean().mean()),
                        "continued": 100 * float((rows.continuations > 0).mean()) if rule != "first" else 0.0,
                        "continuations": float(rows.continuations.mean()) if rule != "first" else 0.0,
                        "unboxed": 100 * float((rows.boxed == 0).mean()), "median_tokens": float(rows.tokens.median())}
        excess = {}
        for stage in STAGES:
            parts = [s.per_problem(run, x, st, rule=rule) for x in ("D-u140", "O-u140") for st in (None, stage)]
            if all(p is not None for p in parts):
                excess[stage] = loss(parts[0], parts[1], other=(parts[2], parts[3]))
        out[rule] = {"conditions": acc, "excess_D140_over_O140": excess}
    return out


def main():
    samples = pd.read_csv(OUT / "math_samples.csv.gz", dtype={"level": str})
    s = Samples(samples)
    texts = pd.read_csv(OUT / "depth_texts.csv.gz")
    texts = texts[texts.run.str.startswith(("pilot", "main", "teacher", "robustness"))]
    result = {run: family_run(s, texts, run) for run in FAMILY_RUNS if s.rows(run, "M0") is not None}
    result[MAIN] = main_arms(s, texts)
    if s.rows(TEACHER, "D-T-u140") is not None:
        result[TEACHER] = teacher_run(s, texts)
    if (texts.run == "robustness-20260924").any():
        result["robustness-20260924"] = robustness(s, texts)
        if (samples.run == FOLLOWUP).any():
            followup(s, pd.read_csv(OUT / "train_log.csv.gz"), result["robustness-20260924"])
    for run in FAMILY_RUNS:
        if run in result:
            result[run]["reforcing"] = reforcing(s, run, ("M0", "D-u140", "O-u140"))
            result[run]["failure_form"] = {
                f"{label}|{when}": failure_form(g, cap=32768 if bench == "aime" else 16384)
                for (bench, label, when), g in samples[(samples.run == run) & (samples["mode"] == "forced")
                                                       & (samples.rule == "first")].groupby(["bench", "label", "when"])}
    (OUT / "summaries").mkdir(exist_ok=True)
    (OUT / "summaries" / "math.json").write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
