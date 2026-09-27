r"""Every number in the paper, computed from the frozen results.

One registry maps each key the text uses as \nv{key} to a value, a rounding rule, a source and a
status. Values come from outputs/summaries/*.json where a summary holds the field, else from the
per-sample tables outputs/*.csv.gz, from the original study's package replay-v1/, or from data/.
Design values, registered thresholds, cited numbers and the few results that only a record states are
constants, each with its source. A key with no source yet is pending: it prints ?? and FINAL=1 refuses it.

    python src/paper_numbers.py          write paper/numbers.generated.tex and outputs/summaries/numbers.json
    python src/paper_numbers.py --check  fail if that file is stale or the paper uses an undefined key, and
                                         list the keys the paper does not use (FINAL=1: fail on pending keys)

To add a number: yield it from the family function of its experiment (or add a family to FAMILIES).

Rounding rules:
- accuracy, points and their intervals: 1 decimal (a signed change shows its +);
- B: 3 decimals, or 4 when |B| < 0.01;
- shares: an integer percent, or 1 decimal under 10%; cap hits and unboxed answers: 1 decimal;
- probabilities, correlations and ratios: 2 decimals; a correlation over five or fewer points, 1;
- half-skip: 1 decimal (reruns 2);
- counts and tokens: integers with a thousands comma.
Halves round away from zero, a difference is taken before rounding, and a lower bound ("at least")
rounds down.
"""

import json
import os
import re
import sys
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from common import divergence, stats

REPO = Path(__file__).resolve().parents[1]
OUT, REPLAY = REPO / "outputs", REPO / "replay-v1"
TEX = REPO / "paper" / "numbers.generated.tex"  # paper/numbers.tex once reviewed
PAPER = [*sorted((REPO / "paper" / "sections").glob("*.tex")), REPO / "paper" / "main.tex"]

# ---------------------------------------------------------------------------------------------------
# Constants

DESIGN = {  # key: (value, what it is and where it is fixed)
    "d.rank": (32, "LoRA rank of every arm"),
    "d.rank.alt": (128, "rank of the wide arm (main-phase.md N2)"),
    "d.items": (192, "TCES targeted items, and as many held out (data/tces/panel.jsonl)"),
    "d.budget.xs": (2048, "smallest budget read from the math samples"),
    "d.budget.short": (4096, "TCES probe cap before the 16k probes"),
    "d.budget.mid": (8192, "intermediate budget read from the same samples"),
    "d.budget.long": (16384, "cap of the 16k TCES probes and of MATH-500"),
    "d.budget.aime": (32768, "AIME cap"),
    "d.topp": (0.95, "sampling top-p"),
    "d.T.updates": (30, "RL teacher updates (endpoint-clone.md)"),
    "d.S.samples": (6400, "clone corpus: 800 prompts x 8 samples (endpoint-clone.md)"),
    "d.S.perprompt": (8, "samples per clone prompt"),
    "d.S.updates": (230, "clone state S-u230"),
    "d.S.passes": (1.15, "230 x 32 / 6,400"),
    "d.S.presentations": (9.2, "each clone prompt seen 230 x 32 / 800 times"),
    "d.drill.n": (579, "drilled set, TCES and math (data/tces/subsets.json, data/math/)"),
    "d.drill.updates": (140, "updates of the drilled and once-trained arms"),
    "d.upd.low": (20, "first saved state"),
    "d.upd.mid": (60, "second saved state"),
    "d.pass.low": (1.1, "20 x 32 / 579"),
    "d.pass.mid": (3.3, "60 x 32 / 579"),
    "d.pass.high": (7.7, "140 x 32 / 579"),
    "d.O.n": (4480, "once-trained set, 140 x 32 problems"),
    "d.O.pass.mid": (0.43, "60 x 32 / 4,480"),
    "d.O.pass.high": (1.0, "140 x 32 / 4,480"),
    "d.P.draws": (24, "most draws per problem for the fresh-solution arm (main-phase.md M3)"),
    "d.P.kept": (8, "kept samples per problem for the fresh-solution arm (main-phase.md M3)"),
    "d.P.passes": (1.0, "4,480 slots over 4,474 distinct solutions"),
    "d.R.fit": (20, "registered drilled-fit margin of the rank-128 arm, % of the NLL drop (main-phase.md N2)"),
    "d.RS.updates": (200, "solver state R-S47-u200"),
    "d.RS.passes": (11.1, "200 x 32 / 579"),
    "mt.drill.req": (0.8, "teacher.md, changes before the student phase 4: the drilling ratio required"),
    "mt.held.req": (20, "teacher.md, the same: the drop in held-out NLL required, %"),
    "mt.p5.band": (0.01, "teacher.md prediction 5: the band for D-T's displacement on the base's texts, nats per token"),
    "d.R2.updates": (60, "robustness.md R2: instruction tuning at 1e-4 for 60 updates, the total step of 20 at 3e-4"),
    "d.R1.match": (0.1, "robustness.md R1 trajectory check: the replayed losses at updates 1-5 match the stage's within 0.1%"),
    "d.R5.rows": (1920, "robustness.md R5: distinct single-turn chat rows, 60 updates x 32 (data/chat_ids.json)"),
    "d.acq.lr": ("$10^{-4}$", "supervised acquisition learning rate"),
    "d.stage.lr": ("$3\\times10^{-4}$", "later-stage learning rate"),
    "d.stage.lrlow": ("$10^{-4}$", "gentler Stage-B learning rate (repetition-fragility.md D2)"),
    "d.stage.batch": (32, "examples per update"),
    "d.stage.updates": (20, "updates of a later stage"),
    "d.norobots": (640, "No Robots examples of the instruction-tuning stage (repetition-fragility.md D3)"),
    "d.anchor.extra": (32, "anchor examples added to each batch (anchor.md)"),
    "d.anchor.whigh": (1, "FA1 anchor weight"),
    "d.anchor.wlow": (0.25, "FA025 anchor weight"),
    "d.anchor.fit": (80, "registered drilled-fit threshold, % of F-u140's NLL drop (anchor.md)"),
    "d.anchor.parity": (5, "registered anchor parity margin, points (anchor.md)"),
    "d.parity.math": (5, "registered math parity margin, points (math-pilot.md)"),
    "d.math.pool": (36447, "math training pool (data/math/pool_ids.jsonl)"),
    "d.math.distinct": (58674, "distinct problems in the source (data/math/pool_stats.json)"),
    "d.math.eval": (221, "MATH-500 problems at levels 3-5 with integer answers"),
    "d.math.evalall": (311, "MATH-500 problems with integer answers"),
    "d.math.aime": (59, "AIME 2025-26 problems, less one"),
    "d.math.easy": (256, "easy-problem screen, tokens"),
    "d.math.cut": (2048, "math common-set cut for B, tokens"),
    "d.tces.cut": (1024, "TCES common-set cut for B in the first scoring, tokens"),
    "d.decision.tokens": (8, "length of a decision reply, tokens"),
    "d.horizon": (480, "long Stage-B horizon, updates (decision-competence.md C)"),
    "d.active": (3, "active parameters, billions, of Qwen3.5-35B-A3B and Nemotron-3-Nano-30B-A3B"),
    "d.reforce.times": (8, "most continuations of one sample (reforcing.md)"),
    "d.reforce.tokens": (1024, "continue while under 1,024 tokens or unboxed (reforcing.md)"),
    "d.RP.replay": (6.25, "revision.md RP: 2 of each batch of 32 are the arm's own training texts, %"),
    "d.RL.n": (640, "revision.md RL: relearning problems, and habit prompts, 20 updates x 32 (data/revision_ids.json)"),
    "d.RL.u5": (160, "revision.md RL readings: the examples of 5 updates x 32"),
    "d.SH.temp": (0.5, "revision.md SH: the sampling temperature of O-sharp's solutions"),
    "d.SH.draws": (4, "revision.md SH: most draws per problem"),
    "d.SK.cap": (2048, "revision.md SK amendment: the skill is solving within 2,048 tokens"),
    "d.SK.items": (300, "revision.md SK: held-out evaluation items, and as many calibration items"),
    "d.SK.smoke": (5, "run_skill.py smoke: the first 5 calibration items of each setting, four draws each"),
    "d.SD.shared": (91, "revision.md SD, deviation 6: problems the two seeds' drilled subsets share (both disjoint from D, D2)"),
}

CITED = {  # other papers' numbers, as the text cites them
    "lit.kop.n": (400, "kopiczko2026repetition abstract: 400 samples"),
    "lit.kop.ratio": (128, "kopiczko2026repetition abstract: 128 epochs against 51,200 samples once"),
    "lit.s1.epochs": (5, "muennighoff2025s1 training setup: \"we train for 5 epochs with a batch size of 16\""),
    "lit.limo.epochs": (15, "ye2025limo training setup: \"trained for 15 epochs with a batch size of 64\""),
}

RECORDED = {  # results that only a record states: no table in outputs/ holds them
    "m.calib.direct": ("83", "records/math-pilot.md Calibration: direct samples writing working to the 256 cap, %"),
    "m.scorer.units": ("30", "records/math-pilot.md Prechecks 2: unit examples of the scorer"),
    "m.scorer.hand": ("20", "records/math-pilot.md Prechecks 2: base solutions checked by hand"),
    "aime.ci": ("13--16", "records/math-pilot.md Measures: AIME interval, about +-13 to 16 points"),
    "m35.pre.mad": ("0.0034", "records/main-phase.md Changes: 35B fresh-LoRA identity precheck, mean absolute difference"),
    "m35.pre.two": ("0.0030", "records/main-phase.md Changes: two 35B base samplers, mean absolute difference"),
    "m35.pre.thr": ("0.001", "records/main-phase.md Changes: registered identity criterion"),
    "m35.noise.mad": ("0.0003", "records/main-phase.md M1 noise floor: mean absolute per-token difference, nats"),
    "mnem.pre.pos": ("1,032", "records/main-phase.md Launch (M5): positions a fresh LoRA reproduces exactly"),
    "mnem.tmpl.n": ("7,047", "records/main-phase.md Launch (M5): prompts rendered token-identically"),
    "tces.RP20.think": ("99", "records/decision-competence.md A: R-P-u20 replies opening <think> by update 12, %"),
    "tces.RP20.upd": ("12", "records/decision-competence.md A: the update of that share"),
    "anc.samples.capped": ("55", "records/anchor.md Results: 2,458 of 4,480 anchor texts reach the 4,096 cap, %"),
    "mt.traces.P": ("7.99", "records/teacher.md, teacher phase result: kept traces per P-T problem, on average"),
    "r5.wordprob": ("1--1.5", "records/robustness.md, correction before launch (an independent reading): worked math "
                    "word problems, about 15-30 of R5's 1,920 rows, %"),
    "comp.total": ("2,437", "records/compute.md, Total: list-price Tinker compute of the runs the paper reports, US "
                   "dollars ($2,437.28: $1,845.11 before the revision and $592.17 for its parts; with calibrations and "
                   "aborted launches $2,458.48)"),
    "sk.text.mean": ("871", "records/revision.md SK implementation: mean worked-solution length, base 7 with 5 digits, tokens"),
    "sk.text.max": ("1,937", "records/revision.md SK implementation: the longest of the same"),
}
LEDGER = "records/predictions-ledger.md §4 (Counts), with T2 counted as the rival reading, not a failure"
RECORDED |= {key: (value, f"{LEDGER}: {what}") for key, value, what in (
    ("reg.total", "70", "registered predictions"), ("reg.held", "50", "held"), ("reg.failed", "17", "failed"),
    ("reg.failed.partial", "7", "failed in part"), ("reg.unread", "3", "not readable"),
    ("reg.rival", "1", "a rival reading (T2)"), ("reg.math", "44", "registered, mathematics"),
    ("reg.failed.math", "5", "failed, mathematics"), ("reg.arith", "26", "registered, the arithmetic task"),
    ("reg.failed.arith", "12", "failed, the arithmetic task"))}
TEACHER_ARMS = {"D": 575, "O": 4441}  # records/teacher.md, teacher phase result: problems in D-T and O-T

REGISTERED = {  # thresholds of registered predictions, as the record's prediction text states them (records/)
    "r4.req": (10, "robustness.md R4.1: D-T's excess forced loss over O-T's after update 60 at least 10 points"),
    "r5.req": (5, "robustness.md R5.3: O and O-T each lose at most 5 points"),
    "r5.req.sec": (10, "robustness.md R5.1 and R5.2, secondary: each excess at least 10 points"),
    "reg.MP1.thr": (5, "math-pilot.md prediction 1: B(D-u140) - B(O-u140) at least 5% of the base's per-token NLL"),
    "reg.M3.2.thr": (20, "main-phase.md M3 item 2: D-u140's loss exceeds P-u140's by at least 20 points, each stage"),
    "reg.M2.1.thr": (10, "main-phase.md M2 item 1: D2-u140's excess over O-u140 at least 10 points, each stage"),
    "reg.T1.thr": (10, "teacher.md prediction 1: D-T-u140's excess over O-T-u140 and D-T-u20 at least 10 points"),
    "reg.T2.thr": (3, "teacher.md prediction 2 (consolidation): the excess under 3 points"),
    "reg.T2.thr2": (10, "teacher.md, changes before the student phase 3: prediction 2's upper bound under 10"),
    "reg.T4.thr": (5, "teacher.md prediction 4: P-T-u140's excess over O-T-u140 at most 5 points"),
    "reg.T5.thr": (10, "teacher.md prediction 5: D-T's displacement on its own texts at least 10 times O-T's"),
    "reg.R1.1.thr": (2, "robustness.md R1 prediction 1: delta_2(D) at least 2 x max(delta_2(O), delta_2(P))"),
    "reg.R1.2.thr": (77.4, "robustness.md R1 prediction 2: D's forced accuracy at k = 2 above 77.4%"),
    "reg.R1.3.thr": (1.5, "robustness.md R1 prediction 3: delta_k(P) at most 1.5 x delta_k(O) at k = 1, 2, 5"),
    "reg.R1.4.thr": (10, "robustness.md R1 prediction 4: training loss at updates 2-5 within 10% across the states"),
    "reg.R2.1.thr": (10, "robustness.md R2 prediction 1: D-u140's excess over O-u140 at least 10 points"),
    "reg.R3a.thr": (20, "robustness.md R3a: D's excess over O after update 60 at least 20 points"),
    "reg.R3b.thr": (2, "robustness.md R3b: delta^common_2(D) at least 2 x max(delta^common_2(O), delta^common_2(P))"),
    "reg.DP4.thr": (0.7, "depth-profile.md prediction 4: rank correlation of B and the forced loss at least 0.7"),
    "reg.CT1.thr": (15, "depth-profile.md cap test: F-u140 and U-u140 under 15% forced within 16,384 tokens"),
    "reg.CT2.thr": (10, "depth-profile.md cap test: S-u230 gains at most 10 points over its 4,096-token accuracy"),
    "reg.AN1.thr": (40, "anchor.md prediction 1: FA1 at least 40% forced within 16,384 tokens after Stage B"),
    "reg.AN3.thr": (20, "anchor.md prediction 3: U2 at most 20% after Stage B"),
    "reg.AN3.thr2": (0.06, "anchor.md prediction 3: B(U2) at least 0.06"),
    "reg.AN4.thr": (20, "anchor.md prediction 4: FN at most 20% after Stage B"),
    "reg.C-3.thr": (10, "decision-competence.md C prediction 3: halving at least ten times the half-skip update"),
    "reg.C2.thr": (10, "decision-competence.md C2: the clone above the teacher at update 80 by more than 10 points"),
    "reg.C3.thr": ("3--6", "decision-competence.md C3: both half-skip updates between 3 and 6"),
    "reg.C3.thr2": (1, "decision-competence.md C3: within one update of the first run"),
    "reg.RR-D3b.thr": (10, "repetition-fragility.md D3b: the 140-update states lose at least 10 points unforced"),
    "reg.RR-D3b.thr2": (3, "repetition-fragility.md D3b: the 20-update states lose at most 3"),
    "reg.RR-D4.thr": (30, "repetition-fragility.md D4: F-u140 forced below 30% at update 2"),
    "reg.RL1.thr": (0.5, "revision.md RL prediction 1: R_5 of D+IT at least 0.5"),
    "reg.RL2.thr": (2, "revision.md RL prediction 2: O+IT within 2 points of its after-stage accuracy at every k"),
    "reg.RL.suppressed": (0.75, "revision.md RL readings: R_5 of D+IT at least 0.75 reads as suppressed and restored"),
    "reg.RL.habit": (0.5, "revision.md RL readings: a habit R_20 of at least 0.5 drops the claim of more than a lost habit"),
    "reg.RP.prevent": (3, "revision.md RP readings: replay prevents the break with an excess of at most 3 points"),
    "reg.RP.upper": (6, "revision.md RP readings: and the excess's upper bound under 6"),
    "reg.RP.mitigate": (50, "revision.md RP readings: replay mitigates with a reduction of at least 50%"),
    "reg.RP.fail": (25, "revision.md RP readings: replay fails with a reduction under 25%"),
    "reg.RS.D": (5, "revision.md RS prediction: D's excess over O at least 5 points, its interval above zero"),
    "reg.RS.P": (3, "revision.md RS prediction: P's excess over O within 3 points"),
    "reg.SH.gate": (0.05, "revision.md SH gate: B of O-sharp at least 0.05 nats per token"),
    "reg.SH.break": (10, "revision.md SH readings: sharpening alone breaks it with an excess after AO of at least 10"),
    "reg.SH.within": (3, "revision.md SH readings: it does not with O-sharp within 3 points of O after both stages"),
    "reg.SH.half": (36.5, "revision.md SH prediction: O-sharp's excess after AO under half of D's 73.1"),
    "reg.SK.cal": (25, "revision.md SK calibration: the chosen setting leaves the base at most 25% accurate"),
    "reg.SK.gate": (30, "revision.md SK gate: O-S-u140 at least 30 points above the base, its interval above zero"),
    "reg.SK.capped": (15, "revision.md SK gate: at most 15% of O-S-u140's responses at the cap"),
    "reg.SK.break": (10, "revision.md SK outcomes: a break, D-S at least 10 points below O-S after the stage"),
    "reg.SK.within": (3, "revision.md SK outcomes: consolidation (D-S within 3 points of O-S) and the fix costing "
                      "nothing (P-S within 3 of the better)"),
}

CALIBRATION = {  # records/math-pilot.md Calibration: forced, unforced, direct, median forced tokens
    "l12": ("1.00", "1.00", "0.58", "850"), "l3": ("0.92", "0.83", "0.42", "1,326"),
    "l4": ("0.88", "0.71", "0.17", "954"), "l5": ("0.62", "0.58", "0.08", "2,928"),
    "oly": ("0.46", "0.42", "0.00", "6,144 (cap)"), "num": ("0.73", "0.67", "0.19", "1,410")}

# ---------------------------------------------------------------------------------------------------
# Rounding

DECIMALS = {"acc": 1, "signed": 1, "cap": 1, "hs": 1, "few": 1, "prob": 2, "loss": 2, "hs2": 2, "nll": 3,
            "B.about": 3, "count": 0, "noise": 5}  # noise: the 35B's noise floor, whose interval is ±0.00002


def number(x, decimals, plus=False, rounding=ROUND_HALF_UP):
    """x rounded (half away from zero by default), with a LaTeX minus sign and a thousands comma."""
    q = Decimal(f"{x:.9f}").quantize(Decimal(1).scaleb(-decimals), rounding=rounding)
    text = f"{abs(q):,}"
    return "$-$" + text if q < 0 else "+" + text if plus and q > 0 else text


def fmt(value, rule):
    """The text a value prints as: pending prints ??, an interval [lo, hi], a constant as written."""
    if value is None:
        return r"\textbf{??}"
    if rule.startswith("range "):  # smallest--largest, each end by the named rule
        return "--".join(fmt(v, rule[6:]) for v in value)
    if isinstance(value, list):
        return "[" + ", ".join(fmt(v, rule) for v in value) + "]"
    if isinstance(value, str):
        return value
    if rule == "stated":
        return f"{value:,}"
    if rule == "B":
        return number(value, 3 if abs(value) >= 0.01 else 4)
    if rule == "B.atleast":
        return number(value, 3 if abs(value) >= 0.01 else 4, rounding=ROUND_DOWN)
    if rule == "share":
        return number(value, 0 if abs(value) >= 10 else 1)
    return number(value, DECIMALS[rule], plus=rule == "signed")


# ---------------------------------------------------------------------------------------------------
# Data


class Data:
    """The frozen summaries and, for what no summary holds, the per-sample tables."""

    def __init__(self):
        self.math, self.tces, self.depth, self.revision = (json.loads((OUT / "summaries" / f"{name}.json").read_text())
                                                           for name in ("math", "tces", "depth", "revision"))
        samples = pd.read_csv(OUT / "tces_samples.csv.gz")
        self.tces_rows = dict(tuple(samples[samples.role == "targeted"].groupby(["run", "label", "update", "kind"])))
        maths = pd.read_csv(OUT / "math_samples.csv.gz", dtype={"level": str})
        self.math_rows = dict(tuple(maths.groupby(["run", "bench", "label", "when", "mode", "rule"])))
        self.texts = pd.read_csv(OUT / "depth_texts.csv.gz")
        self.train = pd.read_csv(OUT / "train_log.csv.gz")
        self.p_samples = pd.read_csv(OUT / "math_p_samples.csv.gz")
        self.pool = json.loads((REPO / "data" / "math" / "pool_stats.json").read_text())


def pct(values):
    return 100 * float(np.mean(values))


def flip(interval):
    """The interval of -x from the interval of x."""
    return [-interval[1], -interval[0]]


def cond(d, run, label, update, kind="think"):
    """tces.json's summary of one TCES probe condition (targeted items) and its source."""
    key = f"{run}|{label}|u{update}|{kind}"
    return d.tces["conditions"][key], f'tces.json conditions["{key}"]'


def items(d, run, label, update, kind="think"):
    """Per-item accuracy (the mean over draws) of one TCES condition."""
    return d.tces_rows[(run, label, update, kind)].groupby("task_id").correct.mean()


def paired_loss(before, after, other=None):
    """Points lost from `before` to `after` (per-item Series), less a second pair's loss on the same
    items, with common.stats' item-bootstrap interval."""
    ids = before.index
    if other is not None:
        other = (other[0].loc[ids].to_numpy(), other[1].loc[ids].to_numpy())
    return stats.loss(before.to_numpy(), after.loc[ids].to_numpy(), other)


def first_b(d, run, label):
    """B over the first 1,024 response tokens of M0's common texts."""
    t = d.texts
    return divergence.summary(t[(t.run == run) & (t.set == "common") & (t.state == label)])["B"]


# ---------------------------------------------------------------------------------------------------
# The families of keys. Each yields (key, value, rule, source); a value of None is pending.


def constants(d):
    """Design values, registered thresholds, cited numbers, and results only a record states."""
    for table, prefix in ((DESIGN, "design: "), (REGISTERED, "registered: records/"), (CITED, ""), (RECORDED, "")):
        for key, (value, source) in table.items():
            yield key, value, "stated", prefix + source
    for row, values in CALIBRATION.items():
        for column, value in zip(("forced", "free", "direct", "med"), values):
            yield f"cal.{row}.{column}", value, "stated", f"records/math-pilot.md Calibration: {row}, {column}"


TCES4 = {  # paper tag: (probe run at 4,096 tokens, state)
    "M0": ("dc-20260923b", "M0"), "T": ("dc-20260923b", "T-u30"), "S": ("dc-20260923b", "S-u230"),
    "RP": ("dc-20260923b", "RP47-u140"), "RS": ("dc-20260923b", "RS47-u200"),
    "RPorig": ("dc-20260923b", "B-RPorig"), "RSorig": ("dc-20260923b", "B-RSorig"),
    "U20": ("rep-20260923-forced", "U-u20"), "F20": ("rep-20260923-forced", "F-u20"),
    "U60": ("rep-20260923-forced60", "U-u60"), "F60": ("rep-20260923-forced60", "F-u60"),
    "U140": ("rep-20260923-forced", "U-u140"), "F140": ("rep-20260923-forced", "F-u140")}
LONG = "dc-long-20260923"  # Stage B continued to update 480 from M0, T, S, RP and RS
GENTLE = {"U20": "rep-20260923-gentle", "F20": "rep-20260923-gentle", "U140": "rep-20260923-gentle",
          "F140": "rep-20260923-gentle-F140"}  # Stage B at learning rate 1e-4 (F-u140 was rerun)
INSTRUCT = "rep-20260923-instruct"


def probe4(tag, update):
    run, label = TCES4[tag]
    return (LONG if update > 20 and tag in ("M0", "T", "S", "RP", "RS") else run), label


def tces_4k(d):
    """TCES accuracy within 4,096 tokens, forced into <think>, and its change over Stage B."""
    for tag in TCES4:
        for update in (0, 20, 80, 160, 320, 480):
            run, label = probe4(tag, update)
            if f"{run}|{label}|u{update}|think" in d.tces["conditions"]:
                c, src = cond(d, run, label, update)
                yield f"tces4.{tag}.u{update}", c["accuracy"], "acc", src + ".accuracy"
                yield f"tces4s.{tag}.u{update}", c["strict"], "acc", src + ".strict"
        if not tag.endswith("orig"):  # the original study's states were not probed unforced
            c, src = cond(d, *TCES4[tag], 0)
            yield f"tces4r.{tag}.u0", c["resumed"], "share", src + ".resumed"
            c, src = cond(d, *TCES4[tag], 0, "unforced")
            yield f"tces4free.{tag}.u0", c["accuracy"], "acc", src + ".accuracy"
    strict = [cond(d, *TCES4[tag], 0)[0]["strict"] for tag in ("M0", "T", "S", "U20", "F20")]
    yield "tces4s.handoff.lasting.max", max(strict), "acc", "largest of tces4s.{M0,T,S,U20,F20}.u0"
    yield "tces4s.handoff.lasting.min", min(strict), "acc", "smallest of the same"
    loss = d.tces["loss"]
    dose = {"M0": "M0", "T": "T", "S": "S", "U.low": "U20", "U.mid": "U60", "U.high": "U140",
            "F.low": "F20", "F.mid": "F60", "F.high": "F140"}  # 1.1, 3.3 and 7.7 passes
    for key, tag in dose.items():
        name = f"4k|{TCES4[tag][1]}|Stage B"
        yield f"tces4.dose.{key}", -loss[name]["points"], "signed", f'minus tces.json loss["{name}"].points'
    for tag in ("RS", "U60", "F60"):
        name = f"4k|{TCES4[tag][1]}|Stage B"
        yield f"tces4loss.{tag}", loss[name]["points"], "acc", f'tces.json loss["{name}"].points'
    for update in (0, 20):
        c, src = cond(d, *TCES4["RP"], update)
        yield f"tces4cap.RP.u{update}", c["capped"], "cap", src + ".capped"


def tces_stages(d):
    """TCES states after a gentler Stage B or after instruction tuning (4,096 tokens), and reruns."""
    for tag, run in GENTLE.items():
        c, src = cond(d, run, TCES4[tag][1], 20)
        yield f"tces4g.{tag}", c["accuracy"], "acc", src + ".accuracy"
        yield f"tces4gs.{tag}", c["strict"], "acc", src + ".strict"
    for tag in ("M0", "U20", "F20", "U140", "F140"):
        c, src = cond(d, INSTRUCT, TCES4[tag][1], 20)
        yield f"tces4i.{tag}", c["accuracy"], "acc", src + ".accuracy"
        yield f"tces4is.{tag}", c["strict"], "acc", src + ".strict"
    c, src = cond(d, INSTRUCT, "M0", 20)
    yield "tces4ir.M0", c["resumed"], "share", src + ".resumed"
    keep = [cond(d, INSTRUCT, TCES4[tag][1], 20)[0]["accuracy"] for tag in ("M0", "U20", "F20")]
    yield "tces4i.keep.max", max(keep), "acc", "largest of tces4i.M0, tces4i.U20, tces4i.F20"
    yield "tces4i.keep.min", min(keep), "acc", "smallest of the same"
    medians = [float(d.tces_rows[("rep-20260923-instruct-unforced", TCES4[tag][1], 20, "unforced")].tokens.median())
               for tag in ("U20", "F20")]
    source = "tces_samples rep-20260923-instruct-unforced U-u20 and F-u20 u20: median unforced tokens"
    yield "tces4i.median.max", max(medians), "count", "larger " + source
    yield "tces4i.median.min", min(medians), "count", "smaller " + source
    free = [cond(d, "rep-20260923-forced", TCES4[tag][1], 0, "unforced")[0]["accuracy"]
            - cond(d, "rep-20260923-instruct-unforced", TCES4[tag][1], 20, "unforced")[0]["accuracy"] for tag in ("U20", "F20")]
    yield ("reg.RR-D3b.obs", [min(free), max(free)], "range acc", "tces.json conditions: U-u20's and F-u20's unforced "
           "accuracy at handoff (rep-20260923-forced u0) less after 20 IT updates (rep-20260923-instruct-unforced u20)")
    for arm in "UF":  # change of the drilled state minus change of the lightly trained one, per stage
        for stage, name in (("b", "Stage B"), ("g", "Stage B at 1e-4"), ("i", "instruction tuning")):
            pairs = []
            for tag in (f"{arm}140", f"{arm}20"):
                after = {"b": TCES4[tag][0], "g": GENTLE[tag], "i": INSTRUCT}[stage]
                pairs.append((items(d, *TCES4[tag], 0), items(d, after, TCES4[tag][1], 20)))
            excess = paired_loss(*pairs[0], other=pairs[1])
            source = f"change of {arm}-u140 minus change of {arm}-u20 over {name}, 4k probes, item bootstrap"
            yield f"tces4.dmo.{arm}.{stage}", -excess["points"], "acc", source
            if stage == "b":
                yield f"tces4.dmo.{arm}.{stage}.ci", flip(excess["interval"]), "acc", source
    once = [d.tces["loss"][f"4k|{label}|{name}"]["points"] for label in ("U-u20", "F-u20")
            for name in ("Stage B at 1e-4", "instruction tuning")]
    yield "tces4.once.maxloss", max(once), "acc", 'largest tces.json loss["4k|U-u20 or F-u20|Stage B at 1e-4 or IT"]'
    c, src = cond(d, "rep-20260923-cliff2", "F-u140", 2)
    yield "tces4.cliff.F140", c["accuracy"], "acc", src + ".accuracy"
    for tag, run, label in (("S", "dc-long80-rep", "S-u230"), ("T", "dc-long80-rep-T", "T-u30")):
        c, src = cond(d, run, label, 80)
        yield f"tces4.rerun.{tag}", c["accuracy"], "acc", src + ".accuracy"
        yield f"tces4s.rerun.{tag}", c["strict"], "acc", src + ".strict"
    spread = cond(d, "dc-long80-rep-T", "T-u30", 80)[0]["accuracy"] - cond(d, LONG, "T-u30", 80)[0]["accuracy"]
    yield "tces4.rerun.spread", spread, "acc", "tces4.rerun.T - tces4.T.u80"


CAP16 = {"M0": "M0", "T": "T-u30", "S": "S-u230", "U140": "U-u140", "F140": "F-u140"}


def probe16(tag, update):
    if update == 0:
        return "cap16k-u0-20260924", CAP16[tag]
    return ("cap16k-stageb-20260924" if tag in ("M0", "T") else "cap-20260924"), CAP16[tag]


def tces_16k(d):
    """TCES accuracy within 16,384 tokens at handoff and after Stage B, and how the searches end."""
    handoff, within4 = {}, {}
    for tag in CAP16:
        for update in (0, 20):
            run, label = probe16(tag, update)
            c, src = cond(d, run, label, update)
            rows = d.tces_rows[(run, label, update, "think")]
            yield f"tces16.{tag}.u{update}", c["accuracy"], "acc", src + ".accuracy"
            yield f"tces16s.{tag}.u{update}", c["strict"], "acc", src + ".strict"
            yield f"tces16r.{tag}.u{update}", c["resumed"], "share", src + ".resumed"
            yield f"tces16w4.{tag}.u{update}", c["within_4096"], "acc", src + ".within_4096"
            yield (f"tces16w8.{tag}.u{update}", pct(rows.correct.astype(bool) & (rows.tokens <= 8192)), "acc",
                   f"tces_samples {run} {label} u{update} think: correct within 8,192 tokens")
            if update == 20:
                yield f"tces16cap.{tag}.u20", c["capped"], "cap", src + ".capped"
                yield f"tces16over4.{tag}.u20", c["over_4096"], "share", src + ".over_4096"
        c = cond(d, *probe16(tag, 0), 0)[0]
        handoff[tag], within4[tag] = c["accuracy"], c["within_4096"]
    c, src = cond(d, *probe16("F140", 20), 20)
    yield "tces16.draws", c["n"], "count", src + ".n: 192 targeted items x 2 draws"
    for tag in ("U140", "F140"):
        c, src = cond(d, *probe16(tag, 20), 20)
        yield f"tces16capn.{tag}", c["capped_count"], "count", src + ".capped_count"
        yield f"tces16scan.{tag}", c["capped_with_solution"], "count", src + ".capped_with_solution"
    lost = {tag: d.tces["loss"][f"16k|{CAP16[tag]}|Stage B|16384"]["points"] for tag in CAP16}
    for tag in ("S", "U140", "F140"):
        yield f"tces16loss.{tag}", lost[tag], "acc", f'tces.json loss["16k|{CAP16[tag]}|Stage B|16384"].points'
    for tag in ("U140", "F140"):
        excess, src = d.tces["excess_over_clone"][CAP16[tag]], f'tces.json excess_over_clone["{CAP16[tag]}"]'
        yield f"tces16ex.{tag}", excess["points"], "acc", src + ".points: the answer-only loss beyond the clone's"
        yield f"tces16ex.{tag}.ci", excess["interval"], "acc", src + ".interval"
    lasting = [lost[tag] for tag in ("M0", "T", "S")]
    yield "tces16.lastingloss.max", max(lasting), "acc", "largest of the 16k losses of M0, T-u30, S-u230"
    yield "tces16.lastingloss.min", min(lasting), "acc", "smallest of the same"
    yield "tces16.handoff.max", max(handoff.values()), "acc", "largest tces16.*.u0"
    yield "tces16.handoff.min", min(handoff.values()), "acc", "smallest tces16.*.u0"
    yield "tces16.handoff.spread", max(handoff.values()) - min(handoff.values()), "acc", "their difference"
    gain = [handoff[tag] - within4[tag] for tag in CAP16]
    yield "tces16.gain.max", max(gain), "acc", "largest tces16.*.u0 - tces16w4.*.u0"
    yield "tces16.gain.min", min(gain), "acc", "smallest of the same"


def tces_clock(d):
    """The decision clock: half-skip updates, forced halving, the solver's skip share."""
    hs = d.tces["half_skip"]
    for tag, key in (("M0", "dc-20260923b|M0"), ("T", "dc-20260923b|T-u30"), ("S", "dc-20260923b|S-u230"),
                     ("RP", "dc-20260923b|RP47-u140"), ("RP60", "dc-20260923b|RP47-u60"),
                     ("U140", "rep-20260923|U-u140")):
        yield f"tces.hs.{tag}", hs[key], "hs", f'tces.json half_skip["{key}"]'
    for tag, label in (("S", "S-u230"), ("T", "T-u30")):
        yield f"tces.hs.rep.{tag}", hs[f"dc-decision-rep|{label}"], "hs2", f'tces.json half_skip["dc-decision-rep|{label}"]'
    for tag in ("M0", "T", "S", "RP", "RS"):
        label = TCES4[tag][1]
        halves = d.tces["forced_halving"][label]["halves_at"]
        yield f"tces.halves.{tag}", halves, "count", f'tces.json forced_halving["{label}"].halves_at'
    c, src = cond(d, "dc-20260923b", "RS47-u200", 20, "decision")
    yield "tces.RS.skip20", c["skip"], "share", src + ".skip"
    n = d.texts[(d.texts.set == "common") & (d.texts.state == "M0")].groupby("run").n.sum()
    yield ("tces.frac.tokens", 100 * n["depth-20260923"] / n["depth-full-20260924"], "share",
           "depth_texts M0 common texts: positions scored at 1,024 over positions scored at 4,096")


E1 = {"M0": "E1-M0", "RP": "E1-RP20", "RS": "E1-RS220"}  # the original study's pair on its panel
E2 = {"M0": "E2-M0", "T": "E2-T-u30", "S": "E2-S-u230", "RP": "E2-RP47-u140", "RS": "E2-RS47-u200"}
PANELS = "dc-panels-20260923"


def tces_panels(d):
    """The E1 panel (256 items x 16 draws) and the held-out E2 panel."""
    per_item = {}
    for tag, label in E1.items():
        c, src = cond(d, PANELS, label, 0)
        by_item = d.tces_rows[(PANELS, label, 0, "think")].groupby("task_id")
        correct, strict = by_item.correct.mean(), by_item.strict_correct.mean()
        per_item[tag] = correct
        rows = f"tces_samples {PANELS} {label}: items"
        yield f"e1.{tag}.acc", c["accuracy"], "acc", src + ".accuracy"
        yield f"e1.{tag}.strict", c["strict"], "acc", src + ".strict"
        yield f"e1.{tag}.always", int((correct == 1).sum()), "count", rows + " correct on every draw"
        yield f"e1.{tag}.never", int((correct == 0).sum()), "count", rows + " with no correct draw"
        yield f"e1.{tag}.snever", int((strict == 0).sum()), "count", rows + " with no strict-correct draw"
    yield "pair.corr", per_item["RP"].corr(per_item["RS"]), "prob", "E1: Pearson correlation of per-item accuracy, RP20 and RS220"
    with_m0 = [per_item["M0"].corr(per_item[tag]) for tag in ("RP", "RS")]
    yield "pair.corr.m0.max", max(with_m0), "prob", "E1: the larger of M0's correlations with RP20 and RS220"
    yield "pair.corr.m0.min", min(with_m0), "prob", "E1: the smaller of the same"
    for tag, label in E2.items():
        c, src = cond(d, PANELS, label, 0)
        yield f"e2.{tag}", c["accuracy"], "acc", src + ".accuracy (held-out items)"
        yield f"e2s.{tag}", c["strict"], "acc", src + ".strict (held-out items)"


def tces_losses(d):
    """Tinker's summed token loss over the first later-stage updates."""
    for tag in ("M0", "T", "S", "RP", "RS", "U20", "U60", "U140", "F20", "F60", "F140"):
        key = "|".join(TCES4[tag])
        for update in ("1", "2", "3"):
            loss = d.tces["next_batch_loss"][key][update]
            yield f"lj.{tag}.u{update}", loss, "loss", f'tces.json next_batch_loss["{key}"]["{update}"]'
    t = d.train[(d.train.stage == "later") & (d.train["update"] == 2)].set_index(["run", "label"]).loss_sum
    for stage, run, group, others in (("gentle", "rep-20260923-gentle", "once", ("U-u20", "F-u20")),
                                      ("it", INSTRUCT, "others", ("M0", "U-u20", "F-u20"))):
        for tag in ("U140", "F140"):
            label = TCES4[tag][1]
            yield f"lj.{stage}.{tag}", t[run, label], "loss", f"train_log {run} {label} later update 2: loss_sum"
        values = [t[run, label] for label in others]
        yield f"lj.{stage}.{group}.max", max(values), "loss", f"train_log {run} {', '.join(others)} update 2: largest"
        yield f"lj.{stage}.{group}.min", min(values), "loss", "smallest of the same"


SEARCH = ("M0", "T", "S", "RP", "U20", "U60", "U140", "F20", "F60", "F140")  # analyze_depth's search states


def correlations(d):
    """What ranks the forced accuracy lost under Stage B (4,096 tokens) across the search states."""
    rank = d.depth["rank_correlation_with_4k_loss"]
    for key, group, field in (("rho.B", "search", "B_4096"), ("rho.B.RS", "with_solver", "B_4096"),
                              ("rho.group", "search", "group"), ("rho.passes", "search", "passes")):
        yield key, rank[group][field], "prob", f'depth.json rank_correlation_with_4k_loss["{group}"]["{field}"]'
    dose = d.depth["dose"]
    lost = {tag: dose[TCES4[tag][1]]["lost_4k"] for tag in (*SEARCH, "RS")}
    y = [lost[tag] for tag in SEARCH]
    handoff = [cond(d, *TCES4[tag], 0)[0]["accuracy"] for tag in SEARCH]
    free = [cond(d, *TCES4[tag], 0, "unforced")[0]["accuracy"] for tag in SEARCH]
    source = " at handoff against depth.json dose lost_4k, the ten search states"
    yield "rho.fa", stats.spearman(handoff, y), "prob", "Spearman: forced accuracy" + source
    yield "rho.ua", stats.spearman(free, y), "prob", "Spearman: unforced accuracy" + source
    batch = {tag: d.tces["next_batch_loss"]["|".join(TCES4[tag])] for tag in (*SEARCH, "RS")}
    for update in ("1", "2"):
        x = [batch[tag][update] for tag in SEARCH]
        yield f"rho.loss{update}", stats.spearman(x, y), "prob", f"Spearman: loss at later update {update}" + source
    b = {tag: dose[TCES4[tag][1]]["B_4096"] for tag in SEARCH}
    for key, group in (("rho.within.low", ("T", "S", "U20", "F20")), ("rho.within.high", ("U140", "F140", "RP"))):
        yield (key, stats.spearman([b[t] for t in group], [lost[t] for t in group]), "few",
               f"Spearman of B_4096 and lost_4k within {', '.join(group)}")
    origins = (*SEARCH, "RS")
    yield ("pearson.loss2", float(np.corrcoef([batch[t]["2"] for t in origins], [-lost[t] for t in origins])[0, 1]),
           "prob", "Pearson: loss at later update 2 against the forced change over Stage B, eleven origins")


DEPTH = {"M0": "M0", "T": "T-u30", "S": "S-u230", "U20": "U-u20", "U60": "U-u60", "U140": "U-u140",
         "F20": "F-u20", "F60": "F-u60", "F140": "F-u140", "RP": "RP47-u140", "RP20": "RP47-u20",
         "RP60": "RP47-u60", "RS": "RS47-u200", "U2": "U2-u140", "FA1": "FA1-u140", "FA025": "FA025-u140",
         "FN": "FN-u140"}
BINS = ("4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1024")


def tces_divergence(d):
    """B, A and the probability kept on M0's alternatives: over the first 1,024 tokens (b, a, br) and
    over the full samples (n1, n1x)."""
    first = d.depth["1024"]["states"]
    for tag in ("T", "S", "U20", "U60", "U140", "F20", "F60", "F140", "RP", "RP20", "RP60", "RS"):
        yield f"b.{tag}", first[DEPTH[tag]]["B"], "B", f'depth.json ["1024"].states["{DEPTH[tag]}"].B'
    for tag in ("RP", "RP20", "RP60", "RS"):
        yield f"a.{tag}", 100 * first[DEPTH[tag]]["A"], "acc", f'depth.json ["1024"].states["{DEPTH[tag]}"].A, %'
    yield "b.offset.lo", first["M0"]["top_p_offset_lower_bound"], "B", 'depth.json ["1024"].states["M0"].top_p_offset_lower_bound'
    yield "b.offset.hi", -np.log(0.95), "B", "-ln 0.95, the offset when M0's nucleus is its top token alone"
    base = first["M0"]["branch_top_probability_base"]
    source = 'depth.json ["1024"].states[...].branch_top_probability_state (all of M0\'s branch points)'
    yield "br.m0top", base, "prob", 'depth.json ["1024"].states["M0"].branch_top_probability_base'
    yield "br.U140", first["U-u140"]["branch_top_probability_state"], "prob", source
    yield "br.F140", first["F-u140"]["branch_top_probability_state"], "prob", source
    lasting = ("T-u30", "S-u230", "U-u20", "F-u20")  # every state scores the same branch points, so pooling is the mean
    yield ("br.last.to", np.mean([first[label]["branch_top_probability_state"] for label in lasting]), "prob",
           source + ", pooled over T-u30, S-u230, U-u20 and F-u20")
    full = d.depth["4096"]["states"]
    nll = full["M0"]["base_nll"]
    for group, tags in (("n1", ("T", "S", "U140", "F140", "U2", "FA1", "FA025", "FN")),
                        ("n1x", ("U20", "U60", "F20", "F60", "RP", "RS"))):
        for tag in tags:
            s, src = full[DEPTH[tag]], f'depth.json ["4096"].states["{DEPTH[tag]}"]'
            yield f"{group}.b.{tag}", s["B"], "B", src + ".B"
            yield f"{group}.a.{tag}", 100 * s["A"], "acc", src + ".A, %"
            yield f"{group}.alt.{tag}", s["alternative_probability_kept"], "prob", src + ".alternative_probability_kept"
    yield "n1.bfrac.M0", 0, "stated", "definition: B of M0 against itself is 0"
    for group, tags in (("n1", ("T", "S", "U140", "F140", "U2", "FA1", "FA025", "FN")), ("n1x", ("U60", "F60"))):
        for tag in tags:
            yield f"{group}.bfrac.{tag}", 100 * full[DEPTH[tag]]["B"] / nll, "share", f'B of {DEPTH[tag]} over M0\'s NLL, 4,096 tokens, %'
    for tag in ("F140", "U2"):
        yield f"n1.kept.{tag}", 100 * full[DEPTH[tag]]["kept_share"], "share", f'depth.json ["4096"].states["{DEPTH[tag]}"].kept_share, %'
    yield "n1.nll", nll, "nll", 'depth.json ["4096"].states["M0"].base_nll'
    drilled = min(full[label]["B"] for label in ("U-u140", "F-u140", "RP47-u140", "U2-u140"))
    lasting = max(full[label]["B"] for label in ("M0", "T-u30", "S-u230"))
    yield ("n1x.b.gap", drilled - lasting, "B.atleast",
           "lowest B at 7.7 passes without added data (U, F, RP, U2) minus highest lasting B (M0, T, S), 4,096 tokens")
    b1024 = {label: first[label]["B"] for label in first} | {DEPTH[t]: first_b(d, ANCHOR, DEPTH[t]) for t in ARMS}
    shrink = [100 * (1 - full[label]["B"] / b1024[label]) for label in full
              if label in b1024 and full[label]["B"] > 0 and b1024[label] > 0]
    source = "share of B at 1,024 tokens lost at 4,096, over the states with positive B"
    yield "n1x.shrink.min", min(shrink), "share", "smallest " + source
    yield "n1x.shrink.max", max(shrink), "share", "largest " + source
    sums = d.texts[(d.texts.run == "depth-20260923") & (d.texts.set == "common")].groupby("state").sum(numeric_only=True)
    profile = {}
    for tag in ("T", "S", "U20", "F20", "U60", "F60", "U140", "F140", "RP", "RS"):
        for position in BINS:
            column = position.replace("-", "_")
            profile[tag, position] = sums.loc[DEPTH[tag], f"d_{column}"] / sums.loc[DEPTH[tag], f"n_{column}"]
            yield (f"bprof.{tag}.{position}", profile[tag, position], "B",
                   f"depth_texts depth-20260923 {DEPTH[tag]}, common texts: B over response positions {position}")
    late = [abs(profile[tag, position]) for tag in ("T", "S", "U20", "F20") for position in BINS[1:]]
    yield "bprof.lasting.max", max(late), "B.about", "largest |B| from position 8 on over T-u30, S-u230, U-u20, F-u20"


ANCHOR = "anchor-20260924"
ARMS = {"FA1": "FA1-u140", "FA025": "FA025-u140", "FN": "FN-u140", "U2": "U2-u140"}


def anchor(d):
    """The anchor arms: drilled on U or F's set with M0's own samples added (records/anchor.md)."""
    handoff16, handoff4 = {}, {}
    for tag, label in ARMS.items():
        h, src_h = cond(d, ANCHOR, label, 0)
        a, src_a = cond(d, ANCHOR, label, 20)
        free, src_f = cond(d, ANCHOR, label, 0, "unforced")
        handoff16[tag], handoff4[tag] = h["accuracy"], h["within_4096"]
        yield f"anc.h16.{tag}", h["accuracy"], "acc", src_h + ".accuracy"
        yield f"anc.h4.{tag}", h["within_4096"], "acc", src_h + ".within_4096"
        yield f"anc.a16.{tag}", a["accuracy"], "acc", src_a + ".accuracy"
        yield f"anc.a4.{tag}", a["within_4096"], "acc", src_a + ".within_4096"
        yield f"anc.cap.{tag}", a["capped"], "cap", src_a + ".capped"
        yield f"anc.strict.{tag}", a["strict"], "acc", src_a + ".strict"
        yield f"anc.free.{tag}", free["within_4096"], "acc", src_f + ".within_4096"
        gain, src = d.tces["anchor_minus_F_after"][label], f'tces.json anchor_minus_F_after["{label}"]'
        yield f"anc.d16.{tag}", gain["estimate"], "signed", src + ".estimate"
        yield f"anc.d16ci.{tag}", gain["interval"], "acc", src + ".interval"
        yield f"anc.b.{tag}", first_b(d, ANCHOR, label), "B", f"depth_texts {ANCHOR} {label}, common texts: B at 1,024 tokens"
    f16, src16 = cond(d, "cap16k-u0-20260924", "F-u140", 0)
    f4, src4 = cond(d, "rep-20260923-forced", "F-u140", 0)  # F-u140's references (anchor.md Results)
    yield "anc.h4.F140", f4["accuracy"], "acc", src4 + ".accuracy"
    c, src = cond(d, "cap-20260924", "F-u140", 20)
    yield "anc.a4.F140", c["within_4096"], "acc", src + ".within_4096"
    c, src = cond(d, "rep-20260923-forced", "F-u140", 0, "unforced")
    yield "anc.free.F140", c["accuracy"], "acc", src + ".accuracy"
    for tag in ("FA1", "FA025"):
        yield f"anc.gap16.{tag}", f16["accuracy"] - handoff16[tag], "acc", f"{src16}.accuracy - anc.h16.{tag}"
        yield f"anc.gap4.{tag}", f4["accuracy"] - handoff4[tag], "acc", f"anc.h4.F140 - anc.h4.{tag}"
    source = "handoff accuracy within 16,384 tokens minus within 4,096, same samples"
    yield "anc.gain.F140", f16["accuracy"] - f16["within_4096"], "acc", f"{src16}: {source}"
    yield "anc.gain.FA1", handoff16["FA1"] - handoff4["FA1"], "acc", f"anc.h16.FA1 - anc.h4.FA1: {source}"
    for tag in ("FA1", "FA025", "FN"):
        share = d.depth["drilled_text_fit"]["drop_share_of_F140"][ARMS[tag]]
        yield f"anc.fit.{tag}", 100 * share, "share", f'depth.json drilled_text_fit.drop_share_of_F140["{ARMS[tag]}"], %'
    name = "16k|FA1-u140|Stage B|16384"
    yield "anc.loss16.FA1", d.tces["loss"][name]["points"], "acc", f'tces.json loss["{name}"].points'


def original_study(d):
    """The original study's matched pair, trace R-P@20 and solver R-S@220 (replay-v1/, unchanged)."""
    follow = json.loads((REPLAY / "followup" / "readout.json").read_text())["blocks"]["11"]
    for tag, arm in (("RP", "R-P"), ("RS", "R-S")):
        path = f"followup/profiles/seed-11-{arm}.json"
        panels = json.loads((REPLAY / path).read_text())["panels"]
        src = f"replay-v1/{path} panels"
        yield f"pair.acc.{tag}", 100 * panels["targeted"]["exact_success_rate"], "acc", src + ".targeted.exact_success_rate"
        yield f"pair1.held.{tag}", 100 * panels["sentinel"]["exact_success_rate"], "acc", src + ".sentinel.exact_success_rate"
        yield (f"pair.never.{tag}", 256 * (1 - panels["targeted"]["mean_pass_at_k"]["16"]), "count",
               src + '.targeted: 256 x (1 - mean_pass_at_k["16"])')
        if tag == "RS":
            yield "pair.median.RS", panels["targeted"]["completion_token_length"]["median"], "count", src + ".targeted.completion_token_length.median"
        path = f"order-seed47-20260909/{arm}-profile.json"
        second = json.loads((REPLAY / path).read_text())["panels"]
        yield f"pair2.held.{tag}", 100 * second["sentinel"]["exact_success_rate"], "acc", f"replay-v1/{path} panels.sentinel.exact_success_rate"
        half = follow["arms"][arm]["curves"]["targeted"]["half_life"]["update"]
        yield f"pair.hl.{tag}", half, "hs2", f'replay-v1/followup/readout.json blocks["11"].arms["{arm}"].curves.targeted.half_life.update'
        wins = follow["matching"]["selected"][f"{arm}_successes"]
        yield f"pair.match.{tag}", wins, "count", f'replay-v1/followup/readout.json blocks["11"].matching.selected["{arm}_successes"]'
    monitor = {m["update"]: m["targeted"]["tokens_median"] for m in follow["arms"]["R-P"]["failures"]["a_monitor"]}
    for update in (5, 10, 20):
        yield (f"pair.len.u{update}", monitor[update], "count",
               f'replay-v1/followup/readout.json blocks["11"].arms["R-P"].failures.a_monitor, update {update}: targeted.tokens_median')
    curve = json.loads((REPLAY / "order-seed47-20260909" / "readout.json").read_text())
    curve = curve["blocks"]["47"]["arms"]["R-P"]["curves"]["targeted"]
    raw = dict(zip(curve["grid"], curve["raw"]))
    for update in (0, 5):
        yield (f"pair2.RP.u{update}", 100 * raw[update], "acc",
               f'replay-v1/order-seed47-20260909/readout.json blocks["47"].arms["R-P"].curves.targeted.raw at update {update}')


PILOT, MAIN = "pilot-20260924", "main-9b-20260924"
M35, NEMO = "main-35b-20260924", "main-nemotron-20260924"
MATH = {"base": "M0", "D20": "D-u20", "D60": "D-u60", "D140": "D-u140", "O60": "O-u60", "O140": "O-u140",
        "P": "P-u140", "D2": "D2-u140", "R": "D-r128-u140"}
AFTER = {"h": "", "it": "+it", "b": "+b"}  # at handoff, after instruction tuning, after Stage B
EVALUATED = ("base", "D20", "D60", "D140", "O140")


def acc(d, run, tag, stage, mode="forced"):
    key = f"{MATH[tag]}{AFTER[stage]}|{mode}"
    return d.math[run]["accuracy"][key], f'math.json {run} accuracy["{key}"]'


def form(d, run, tag, stage):
    """How one condition's forced MATH-500 samples end (math.json failure_form)."""
    key = f"{MATH[tag]}{AFTER[stage]}|{'u0' if stage == 'h' else 'u20'}"
    return d.math[run]["failure_form"][key], f'math.json {run} failure_form["{key}"]'


def math_rows(d, run, tag, stage, mode="forced", bench="math500", rule="first"):
    label, when = MATH[tag] + AFTER[stage], "u0" if stage == "h" else "u20"
    return d.math_rows[(run, bench, label, when, mode, rule)], f"math_samples {run} {bench} {label} {when} {mode} {rule}"


def math_pilot(d):
    """The 9B math pilot: accuracy, losses and excesses, AIME, how samples end, divergence."""
    r = d.math[PILOT]
    for stage in AFTER:
        for tag in EVALUATED:
            value, src = acc(d, PILOT, tag, stage)
            yield f"m.{stage}.{tag}", value, "acc", src
    for stage, tag in [("h", tag) for tag in EVALUATED] + [("b", "O140")]:
        value, src = acc(d, PILOT, tag, stage, "unforced")
        yield f"mfree.{stage}.{tag}", value, "acc", src
    after_it = [acc(d, PILOT, tag, "it", "unforced")[0] for tag in EVALUATED]
    yield "mfree.it.max", max(after_it), "acc", "largest unforced accuracy after instruction tuning, five states"
    yield "mfree.it.min", min(after_it), "acc", "smallest of the same"
    for key, mode in (("m.it.gapDO", "forced"), ("mfree.it.gapDO", "unforced")):
        gap = acc(d, PILOT, "O140", "it", mode)[0] - acc(d, PILOT, "D140", "it", mode)[0]
        yield key, gap, "acc", f"math.json {PILOT} accuracy: O-u140+it minus D-u140+it, {mode}"
    handoffs = [acc(d, PILOT, tag, "h")[0] for tag in EVALUATED] + [acc(d, MAIN, tag, "h")[0] for tag in ("P", "D2", "R")]
    yield "m.h.all.max", max(handoffs), "acc", "largest 9B forced handoff accuracy, pilot and main-phase states"
    yield "m.h.all.min", min(handoffs), "acc", "smallest of the same"
    lost = [r["loss"][f"{label}|{stage}|forced"]["points"] for label in ("M0", "O-u140") for stage in ("it", "b")]
    yield "m.otherloss.max", max(lost), "acc", f"largest forced loss of M0 and O-u140 under either stage (math.json {PILOT} loss)"
    for stage in ("it", "b"):
        key = f"D-u140 minus O-u140|{stage}"
        excess, src = r["excess"][key], f'math.json {PILOT} excess["{key}"]'
        yield f"mex.O.{stage}", excess["points"], "acc", src + ".points"
        yield f"mexci.O.{stage}", excess["interval"], "acc", src + ".interval"
    for stage in ("it", "b"):
        key = f"D-u60|{stage}|forced"
        yield f"mloss.{stage}.D60", r["loss"][key]["points"], "acc", f'math.json {PILOT} loss["{key}"].points'
        yield f"mlossci.{stage}.D60", r["loss"][key]["interval"], "acc", f'math.json {PILOT} loss["{key}"].interval'
    for tag in ("base", "O140"):
        key = f"{MATH[tag]}|it|forced"
        yield f"mgain.it.{tag}", -r["loss"][key]["points"], "acc", f'minus math.json {PILOT} loss["{key}"].points'
        yield f"mgainci.it.{tag}", flip(r["loss"][key]["interval"]), "acc", f'minus math.json {PILOT} loss["{key}"].interval'
    yield "m.parity", r["predictions"]["parity"]["handoff_forced_gap"], "acc", f"math.json {PILOT} predictions.parity.handoff_forced_gap"
    for stage in ("b", "it"):
        yield f"m.rho.{stage}", r["predictions"][f"4|{stage}"], "few", f'math.json {PILOT} predictions["4|{stage}"]'
    depth, src = r["depth"], f"math.json {PILOT} depth"
    nll = depth["M0"]["base_nll"]
    for tag in ("D60", "D140"):
        yield f"mb.{tag}.3", depth[MATH[tag]]["B"], "B", f'{src}["{MATH[tag]}"].B'
    yield "mb.nll", nll, "nll", f'{src}["M0"].base_nll'
    yield "mb.D140.pdrop", 100 * (1 - np.exp(-depth["D-u140"]["B"])), "share", \
        f'100(1 - exp(-B)) of {src}["D-u140"].B: the per-token probability the base\'s texts lose, geometric mean, %'
    for tag in ("D20", "D60", "D140", "O140"):
        yield f"mb.frac.{tag}", 100 * depth[MATH[tag]]["B"] / nll, "share", f"B of {MATH[tag]} over the base's NLL, %"
    yield "m.bfrac.base", 0, "stated", "definition: B of the base against itself is 0"
    yield "mb.kept", 100 * depth["D-u140"]["kept_share"], "share", f'{src}["D-u140"].kept_share, %'
    yield "mb.alt.D140", depth["D-u140"]["alternative_probability_kept"], "prob", f'{src}["D-u140"].alternative_probability_kept'


def math_aime_and_forms(d):
    """The pilot on AIME, and how its forced MATH-500 samples end at each budget."""
    r = d.math[PILOT]
    for stage in ("h", "it"):
        for tag in ("base", "D140", "O140"):
            key = MATH[tag] + AFTER[stage]
            yield f"aime.{stage}.{tag}", r["aime_forced"][key], "acc", f'math.json {PILOT} aime_forced["{key}"]'
    rows, src = math_rows(d, PILOT, "base", "h", bench="aime")
    yield "aime.base.h16", pct(rows.within_16384), "acc", src + ": correct within 16,384 tokens"
    rows, src = math_rows(d, PILOT, "D140", "it", bench="aime")
    capped = rows.capped == 1
    yield "aime.it.D140.cap", pct(capped), "cap", src + ": share capped"
    yield "aime.it.D140.loop", pct(rows.looping[capped]), "share", src + ": share of capped samples looping"
    yield "aime.it.D140.unboxed", pct(rows.boxed == 0), "cap", src + ": share with no \\boxed"
    for stage in AFTER:
        for tag in ("base", "D140", "O140", "D60"):
            f, src = form(d, PILOT, tag, stage)
            for k in (2048, 4096, 8192) if tag != "D60" else (4096,):
                yield f"mbud.{k}.{stage}.{tag}", f[f"within_{k}"], "acc", f"{src}.within_{k}"
            if tag == "D60":
                continue
            value, src_acc = acc(d, PILOT, tag, stage)
            yield f"mbud.16384.{stage}.{tag}", value, "acc", src_acc
            yield f"mcap.{stage}.{tag}", f["capped"], "cap", src + ".capped"
            yield f"munbox.{stage}.{tag}", f["unboxed"], "cap", src + ".unboxed"
            yield f"mtok.{stage}.{tag}", f["mean_tokens"], "count", src + ".mean_tokens"
    for tag in ("D20", "D60"):
        f, src = form(d, PILOT, tag, "b")
        yield f"mcap.b.{tag}", f["capped"], "cap", src + ".capped"
    for stage, tag, name in (("b", "D", "D140"), ("b", "O", "O140"), ("b", "base", "base"), ("it", "D", "D140")):
        value, src_acc = acc(d, PILOT, name, stage)
        f, src = form(d, PILOT, name, stage)
        yield f"mf.{stage}.{tag}.acc", value, "acc", src_acc
        yield f"mf.{stage}.{tag}.int", f["lenient"], "acc", src + ".lenient"
        yield f"mf.{stage}.{tag}.cap", f["capped"], "cap", src + ".capped"
        yield f"mf.{stage}.{tag}.unbox", f["unboxed"], "cap", src + ".unboxed"
        yield f"mf.{stage}.{tag}.closes", f["closed_think"], "share", src + ".closed_think"
        yield f"mf.{stage}.{tag}.ends", f["ends_turn"], "share", src + ".ends_turn"
        yield f"mf.{stage}.{tag}.short", f["under_100"], "share", src + ".under_100"
        yield f"mf.{stage}.{tag}.med", f["median_tokens"], "count", src + ".median_tokens"
    lenient = {(stage, tag): form(d, PILOT, tag, stage)[0]["lenient"] for stage in AFTER for tag in ("D140", "O140")}
    for stage in ("b", "it"):
        credit = lenient[stage, "D140"] - acc(d, PILOT, "D140", stage)[0]
        yield f"mf.credit.{stage}", credit, "acc", f"mf.{stage}.D.int - mf.{stage}.D.acc"
        excess = (lenient["h", "D140"] - lenient[stage, "D140"]) - (lenient["h", "O140"] - lenient[stage, "O140"])
        yield f"mf.lenient.{stage}", excess, "acc", "D-u140's loss minus O-u140's, both scored leniently (failure_form)"
    rows, src = math_rows(d, PILOT, "D140", "b")
    yield "mf.n", len(rows), "count", src + ": samples"
    yield "mf.closes.n", int(rows.closed_think.sum()), "count", src + ": samples closing </think>"
    yield "mf.short.tokens", 100, "stated", "analyze_math.py failure_form: under 100 tokens is short"


def math_data(d):
    """How the math training pool and the pilot's training data were built."""
    pool, src = d.pool, "data/math/pool_stats.json"
    yield "m.src.rows", pool["source_rows"], "count", src + " source_rows"
    yield "m.contam", pool["dropped"]["contamination"], "count", src + " dropped.contamination"
    for key, field in (("nonint", "non_integer_answer"), ("multi", "several_final_answers"),
                       ("dup", "inconsistent_duplicate_answers"), ("fig", "figures"), ("long", "prompt_over_512_tokens"),
                       ("guess", "guessable_or_not_single_answer")):
        yield f"m.scr.{key}", pool["dropped"][field], "count", f"{src} dropped.{field}"
    yield "m.scr.prompt", pool["prompt_token_limit"], "count", src + " prompt_token_limit"
    agree = pool["reference_agreement"]
    yield "m.ref.rows", agree["rows"], "count", src + " reference_agreement.rows"
    yield "m.ref.agree", 100 * agree["agree_with_code_output"] / agree["rows"], "share", src + " reference_agreement, %"


def replications(d):
    """The pilot design on the 35B and on Nemotron."""
    for prefix, run in (("m35", M35), ("mnem", NEMO)):
        r = d.math[run]
        for stage in AFTER:
            for tag in EVALUATED:
                value, src = acc(d, run, tag, stage)
                yield f"{prefix}.{stage}.{tag}", value, "acc", src
        for stage in ("h", "it"):
            for tag in ("base", "D140", "O140"):
                key = MATH[tag] + AFTER[stage]
                yield f"{prefix}.aime.{stage}.{tag}", r["aime_forced"][key], "acc", f'math.json {run} aime_forced["{key}"]'
        for tag in ("base", "D140", "O140"):
            value, src = acc(d, run, tag, "h", "unforced")
            yield f"{prefix}.free.h.{tag}", value, "acc", src
            f, src = form(d, run, tag, "b")
            yield f"{prefix}.cap.b.{tag}", f["capped"], "cap", src + ".capped"
        depth, src = r["depth"], f"math.json {run} depth"
        nll = depth["M0"]["base_nll"]
        yield f"{prefix}.nll", nll, "nll", f'{src}["M0"].base_nll'
        for tag in ("D140", "O140"):
            yield f"{prefix}.B.{tag}", depth[MATH[tag]]["B"], "B", f'{src}["{MATH[tag]}"].B'
        for tag in ("D60", "D140", "O140"):
            yield f"{prefix}.bfrac.{tag}", 100 * depth[MATH[tag]]["B"] / nll, "share", f"B of {MATH[tag]} over the base's NLL, %"
        yield f"{prefix}.kept", 100 * depth["D-u140"]["kept_share"], "share", f'{src}["D-u140"].kept_share, %'
        for stage in ("it", "b"):
            key = f"D-u140 minus O-u140|{stage}"
            yield f"{prefix}.ex.{stage}", r["excess"][key]["points"], "acc", f'math.json {run} excess["{key}"].points'
            yield f"{prefix}.ex.{stage}.ci", r["excess"][key]["interval"], "acc", f'math.json {run} excess["{key}"].interval'
        gap = r["predictions"]["parity"]["handoff_forced_gap"]
        yield f"{prefix}.parity", gap, "acc", f"math.json {run} predictions.parity.handoff_forced_gap"
        for stage in ("b", "it"):
            yield f"{prefix}.rho.{stage}", r["predictions"][f"4|{stage}"], "few", f'math.json {run} predictions["4|{stage}"]'
    for prefix, run, stage in (("m35", M35, "b"), ("mnem", NEMO, "it")):  # the stage that breaks each drilled state
        f, src = form(d, run, "D140", stage)
        yield f"{prefix}.unbox.{stage}.D140", f["unboxed"], "cap", src + ".unboxed"
        key = f"D-u60|{stage}|forced"
        loss = d.math[run]["loss"][key]
        yield f"{prefix}.loss.{stage}.D60", loss["points"], "acc", f'math.json {run} loss["{key}"].points'
        yield f"{prefix}.loss.{stage}.D60.ci", loss["interval"], "acc", f'math.json {run} loss["{key}"].interval'
    f, src = form(d, M35, "D140", "b")
    yield "m35.loop.b.D140", f["capped_looping"], "share", src + ".capped_looping"
    yield "m35.med.b.D140", f["median_tokens"], "count", src + ".median_tokens"
    rows, src = math_rows(d, M35, "D140", "b")
    yield "m35.short.b.D140", pct(rows.tokens <= 1024), "share", src + ": share of samples within 1,024 tokens"
    f, src = form(d, NEMO, "D140", "it")
    yield "mnem.cap.it.D140", f["capped"], "cap", src + ".capped"
    yield "mnem.tok.it.D140", f["mean_tokens"], "count", src + ".mean_tokens"
    t = d.texts
    rows = t[(t.run == M35) & (t.set == "common") & (t.state == "M0-second")]
    noise = stats.ratio_interval(rows.d, rows.n)
    yield "m35.noise.B", noise["estimate"], "B", f"depth_texts {M35} M0-second: B of a second base sampler"
    yield "m35.noise.ci", noise["interval"], "noise", f"depth_texts {M35} M0-second: item-bootstrap interval"


def main_arms(d):
    """The main phase on the 9B: fresh solutions (P), a second subset (D2), rank 128 (R)."""
    r = d.math[MAIN]
    nll = r["depth"]["P-u140"]["base_nll"]
    for tag in ("P", "D2", "R"):
        for stage in AFTER:
            value, src = acc(d, MAIN, tag, stage)
            yield f"mm.{tag}.{stage}", value, "acc", src
        rows, src = math_rows(d, MAIN, tag, "h", mode="unforced")
        yield f"mm.{tag}.free", pct(rows.correct), "acc", src + ": mean correct"
        rows, src = math_rows(d, MAIN, tag, "b")
        yield f"mm.{tag}.cap.b", pct(rows.capped), "cap", src + ": share capped"
        if tag != "P":
            yield f"mm.{tag}.med.b", float(rows.tokens.median()), "count", src + ": median tokens"
        yield f"mm.{tag}.bfrac", 100 * r["depth"][MATH[tag]]["B"] / nll, "share", f"B of {MATH[tag]} over the base's NLL, %"
    for tag in ("P", "D2"):
        yield f"mm.{tag}.B", r["depth"][MATH[tag]]["B"], "B", f'math.json {MAIN} depth["{MATH[tag]}"].B'
    for stage in ("h", "it"):
        rows, src = math_rows(d, MAIN, "P", stage, bench="aime")
        yield f"mm.P.aime.{stage}", pct(rows.correct), "acc", src + ": mean correct"
    p = d.p_samples
    yield "mm.P.distinct", int((p.distinct == 8).sum()), "count", "math_p_samples: problems with 8 distinct solutions"
    yield ("mm.P.repeat", int((8 - p.distinct[p.distinct < 8]).sum()), "count",
           "math_p_samples: repeated slots, each problem short of 8 solutions being visited 8 times")
    yield "mm.slots", 4480, "stated", "design: 140 updates x 32"
    for stage in ("it", "b"):
        for key, name in (("D2O", "D2-u140 minus O-u140"), ("RO", "D-r128-u140 minus O-u140"),
                          ("PO", "P-u140 minus O-u140")):
            excess, src = r["excess"][f"{name}|{stage}"], f'math.json {MAIN} excess["{name}|{stage}"]'
            yield f"mm.{key}.{stage}", excess["points"], "acc", src + ".points"
            yield f"mm.{key}.{stage}.ci", excess["interval"], "acc", src + ".interval"
        excess, src = r["excess"][f"P-u140 minus D-u140|{stage}"], f'minus math.json {MAIN} excess["P-u140 minus D-u140|{stage}"]'
        yield f"mm.DP.{stage}", -excess["points"], "acc", src + ".points"
        yield f"mm.DP.{stage}.ci", flip(excess["interval"]), "acc", src + ".interval"
    rows = d.texts[(d.texts.run == MAIN) & (d.texts.set == "drilled") & (d.texts.state == "D-u140")]
    agree, src = divergence.summary(rows), f"depth_texts {MAIN} drilled texts of D-u140"
    yield "tok.D", 100 * agree["token_accuracy_state"], "acc", src + ": positions where D-u140's top token is the text's, %"
    yield "tok.base", 100 * agree["token_accuracy_base"], "acc", src + ": positions where M0's top token is the text's, %"
    texts = r["drilled_text_nll"]
    for tag, label in (("base", "M0"), ("r32", "D-u140"), ("r128", "D-r128-u140")):
        yield f"mm.R.nll.{tag}", texts[label], "nll", f'math.json {MAIN} drilled_text_nll["{label}"]'
    drop = {label: texts["M0"] - texts[label] for label in ("D-u140", "D-r128-u140")}
    yield "mm.R.fitdiff", 100 * (drop["D-r128-u140"] / drop["D-u140"] - 1), "share", "rank 128's NLL drop on D's texts over rank 32's, less 1, %"
    for stage in ("b", "it"):
        ratio = r["loss"][f"D-r128-u140|{stage}"]["points"] / r["loss"][f"D-u140|{stage}"]["points"]
        yield f"mm.R.ratio.{stage}", ratio, "prob", f'math.json {MAIN} loss["D-r128-u140|{stage}"] over loss["D-u140|{stage}"]'


def reforcing(d):
    """Continuing the drilled searches with "Wait" (records/reforcing.md), on the pilot."""
    rf = d.math[PILOT]["reforcing"]
    for rule, prefix in (("answer", "rfa"), ("budget", "rfb")):
        for tag, label in (("D", "D-u140"), ("O", "O-u140"), ("base", "M0")):
            for stage in AFTER:
                key = label + AFTER[stage]
                yield (prefix + f".{tag}" + ("" if stage == "h" else f".{stage}"), rf[rule]["conditions"][key]["accuracy"],
                       "acc", f'math.json {PILOT} reforcing.{rule}.conditions["{key}"].accuracy')
        for stage in ("it", "b"):
            excess, src = rf[rule]["excess_D140_over_O140"][stage], f"math.json {PILOT} reforcing.{rule}.excess_D140_over_O140.{stage}"
            yield f"{prefix}.ex.{stage}", excess["points"], "acc", src + ".points"
            yield f"{prefix}.ex.{stage}.ci", excess["interval"], "acc", src + ".interval"
    first, budget = rf["first"]["conditions"], rf["budget"]["conditions"]
    src = f"math.json {PILOT} reforcing.budget.conditions"
    for stage in ("it", "b"):
        yield f"rfb.cont.D.{stage}", budget[f"D-u140+{stage}"]["continued"], "share", f'{src}["D-u140+{stage}"].continued'
    yield "rfb.med.D.b", budget["D-u140+b"]["median_tokens"], "count", f'{src}["D-u140+b"].median_tokens'
    rows, src = math_rows(d, PILOT, "D140", "b", rule="budget")
    yield "rfb.cap.D.b", pct(rows.capped), "cap", src + ": share capped"
    cost = {key: first[key]["accuracy"] - budget[key]["accuracy"] for key in ("M0", "M0+it", "D-u140", "O-u140", "O-u140+it", "O-u140+b")}
    source = "first-sample minus budget-rule accuracy"
    yield "rf.cost.min", min(cost.values()), "acc", "smallest " + source + " of " + ", ".join(cost)
    yield "rf.cost.max", max(cost.values()), "acc", "largest of the same"
    yield "rf.cost.baseb", first["M0+b"]["accuracy"] - budget["M0+b"]["accuracy"], "acc", source + ", M0+b"
    for stage in ("it", "b"):
        key = f"D-u140+{stage}"
        yield f"rf.gain.{stage}", budget[key]["accuracy"] - first[key]["accuracy"], "acc", f"budget-rule minus first-sample accuracy, {key}"
        registered = rf["first"]["excess_D140_over_O140"][stage]["points"]
        share = 100 * (1 - rf["budget"]["excess_D140_over_O140"][stage]["points"] / registered)
        yield f"rf.share.{stage}", share, "share", f"share of the {stage} excess the budget rule removes, %"


ROBUST = "robustness-20260924"


def robustness(d):
    """When and how the drilled 9B breaks under instruction tuning (R1), at a smaller step (R2) and over
    longer training (R3) (records/robustness.md)."""
    r, src = d.math[ROBUST], f"math.json {ROBUST}"
    for tag, label in (("D", "D-u140"), ("O", "O-u140"), ("P", "P-u140")):
        yield (f"r1.d.{tag}.u2", r["delta"][f"{label}@u2"], "B",
               f'{src} delta["{label}@u2"]: KL(handoff || after 2 updates) on its own handoff texts')
    ratios = [r["delta"][f"P-u140@u{k}"] / r["delta"][f"O-u140@u{k}"] for k in (1, 2, 5)]
    yield "reg.R1.3.obs", [min(ratios), max(ratios)], "range prob", f'{src} delta["P-u140@uk"] over delta["O-u140@uk"], k = 1, 2, 5'
    yield "reg.R1.2.obs", r["forced"]["D-u140"]["u2"], "acc", f'{src} forced["D-u140"]["u2"]: after 2 updates of instruction tuning'
    yield "reg.R3b.obs", r["delta_common"]["D-u140@u2"], "B", f'{src} delta_common["D-u140@u2"]: on the base\'s texts'
    t = d.train
    loss = t[(t.run == ROBUST) & (t.stage == "r1")].pivot(index="update", columns="label", values="loss_sum").loc[2:5]
    spread = 100 * (loss.max(axis=1) / loss.min(axis=1) - 1)
    source = f"train_log {ROBUST} r1: the four states' training loss at updates 2-5, max/min - 1, %"
    yield "r1.lossspread.max", spread.max(), "share", "largest " + source
    yield "reg.R1.4.obs.min", spread.min(), "share", "smallest " + source
    texts = d.texts

    def nll(run, kind, state, model="state"):  # per-token NLL of D-u140's own handoff texts (lp_state = lp_base - d)
        rows = texts[(texts.run == run) & (texts.set == kind) & (texts.state == state)]
        return -(rows.base_lp.sum() - (rows.d.sum() if model == "state" else 0)) / rows.n.sum()

    base, handoff = nll(PILOT, "own", "D-u140", model="base"), nll(PILOT, "own", "D-u140")
    source = "depth_texts: per-token NLL of D-u140's own handoff texts"
    yield "r1.own.base", base, "nll", f"{source} under the base ({PILOT} own)"
    yield "r1.own.D20", nll(ROBUST, "displacement", "D-u140@u20"), "nll", f"{source} after 20 updates ({ROBUST} displacement)"
    yield ("r1.own.gone.u2", 100 * (nll(ROBUST, "displacement", "D-u140@u2") - handoff) / (base - handoff), "share",
           f"{source}: the rise after 2 updates over the base's NLL less the handoff's, %")
    yield "r2.ex20", r["excess_lr1e-4@20"]["points"], "acc", f'{src} ["excess_lr1e-4@20"].points'
    yield "r2.ex", r["excess_lr1e-4@60"]["points"], "acc", f'{src} ["excess_lr1e-4@60"].points'
    yield "r2.ex.ci", r["excess_lr1e-4@60"]["interval"], "acc", f'{src} ["excess_lr1e-4@60"].interval'
    yield ("r2.examples", DESIGN["d.R2.updates"][0] * DESIGN["d.stage.batch"][0], "count",
           "design: 60 updates x 32 distinct examples, each seen once")
    yield "r3.ex60", r["excess_it@60"]["points"], "acc", f'{src} ["excess_it@60"].points'
    yield "r3.ex60.ci", r["excess_it@60"]["interval"], "acc", f'{src} ["excess_it@60"].interval'
    for tag, label, update in (("D", "D-u140", 40), ("D", "D-u140", 60), ("O", "O-u140", 60)):
        yield f"r3.f.{tag}.u{update}", r["forced"][label][f"u{update}"], "acc", f'{src} forced["{label}"]["u{update}"]'
    common = [-r["delta_common"][f"D-u140@u{update}"] for update in (2, 20)]
    yield "r3b.D.approx", np.mean(common), "B.about", f'minus {src} delta_common["D-u140@u2"] and ["D-u140@u20"], their mean (on the base\'s texts)'


FOLLOW = {"D": "D-u140", "O": "O-u140", "DT": "D-T-u140", "OT": "O-T-u140"}
MIX = {"aya": "tulu_v3.9_aya_100k", "flan": "flan_v2_converted", "codealpaca": "evol_codealpaca_heval_decontaminated",
       "wildjailbreak": "tulu_v3.9_wildjailbreak_decontaminated_50k", "wildchat": "tulu_v3.9_wildchat_100k",
       "wildguard": "tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
       "phcode": "personahub_code_v2_34999", "phif": "personahub_ifdata_manual_seed_v3_29980",
       "coconot": "coconot_converted", "sciriff": "tulu_v3.9_sciriff_10k", "oasst": "oasst1_converted",
       "tablegpt": "tulu_v3.9_table_gpt_5k"}  # R5's sources, all ai2-adapt-dev/


def robustness_followup(d):
    """R2's forced accuracies, R4 (R2 from the teacher states) and R5 (broad chat data at R2's step)."""
    r, src = d.math[ROBUST], f"math.json {ROBUST}"
    for tag in ("D", "O"):
        for k in (20, 60):
            yield f"r2.f.{tag}.u{k}", r["forced"][FOLLOW[tag]][f"lr1e-4@{k}"], "acc", f'{src} forced["{FOLLOW[tag]}"]["lr1e-4@{k}"]'
    for part, tags, saves in (("r4", ("DT", "OT"), (20, 60)), ("r5", ("D", "O", "DT", "OT"), (60,))):
        for tag in tags:
            label = FOLLOW[tag]
            for k in saves:
                key = f"{part}.f.{tag}" + (f".u{k}" if part == "r4" else "")
                yield key, r[part]["forced"][f"{label}@{k}"], "acc", f'{src} {part}.forced["{label}@{k}"]'
            for when, field in (("u1-5", "u1_5"), ("u56-60", "u56_60")):
                yield (f"{part}.loss.{tag}.{when}", r[part]["train_loss"][label][field], "loss",
                       f'{src} {part}.train_loss["{label}"]["{field}"]: per-example training loss')
    for key, part, name in (("r4.ex20", "r4", "excess|D-T-u140|u20"), ("r4.ex60", "r4", "excess|D-T-u140|u60"),
                            ("r5.ex.D", "r5", "excess|D-u140|u60"), ("r5.ex.DT", "r5", "excess|D-T-u140|u60")):
        yield key, r[part][name]["points"], "acc", f'{src} {part}["{name}"].points'
        yield f"{key}.ci", r[part][name]["interval"], "acc", f'{src} {part}["{name}"].interval'
    for tag in ("O", "OT"):  # R5.3: forced at handoff minus after 60 updates
        lost = r["predictions"]["R5.3"]["lost"][FOLLOW[tag]]
        yield f"r5.lost.{tag}", lost, "acc", f'{src} predictions["R5.3"].lost["{FOLLOW[tag]}"]'
    data = r["r5_data"]
    for key, source in MIX.items():
        yield f"r5.mix.{key}", data["mix"][f"ai2-adapt-dev/{source}"], "count", f"{src} r5_data.mix: ai2-adapt-dev/{source}"
    yield "r5.mathlike", data["mathlike_share"], "acc", f"{src} r5_data.mathlike_share: rows matching the registered pattern, %"
    yield "r5.mathlike.n", data["mathlike"], "count", f"{src} r5_data.mathlike: rows matching the registered pattern"
    run, every = "robustness-followup-20260924", []  # the failure form at update 60: R4 (teacher states) and R5
    for kind, states in (("D", ("D-u140", "D-T-u140")), ("O", ("O-u140", "O-T-u140"))):
        labels = [f"{state}+chat-lr1e-4@60" for state in states] + [f"{states[1]}+it-lr1e-4@60"]
        rows = [d.math_rows[(run, "math500", label, "u60", "forced", "first")] for label in labels]
        every += rows
        for name, rule, what, values in (
                ("med", "count", "median tokens", [float(g.tokens.median()) for g in rows]),
                ("close", "share", "share closing </think>", [pct(g.closed_think) for g in rows]),
                ("unbox", "cap", "share with no \\boxed", [pct(g.boxed == 0) for g in rows])):
            source = f"{what} of {', '.join(labels)} (math_samples {run}, u60 forced)"
            yield f"r45.{name}.{kind}.min", min(values), rule, "smallest " + source
            yield f"r45.{name}.{kind}.max", max(values), rule, "largest " + source
    caps = [pct(g.capped) for g in every]
    yield "r45.cap.min", min(caps), "cap", f"smallest share capped of the six states above (math_samples {run}, u60 forced)"
    yield "r45.cap.max", max(caps), "cap", f"largest share capped of the six states above (math_samples {run}, u60 forced)"


TEACHER_RUN = "teacher-20260924"
TEACHER_STATES = {"D": "D-T-u140", "D20": "D-T-u20", "O": "O-T-u140", "P": "P-T-u140"}


def teacher(d):
    """The pilot's design on gpt-oss-120b's traces (records/teacher.md): D-T drilled on one trace per
    problem, D-T-u20 at 1.1 passes, O-T once, P-T a fresh trace at every visit."""
    r, src = d.math[TEACHER_RUN], f"math.json {TEACHER_RUN}"
    acc, nll = r["accuracy"], d.math[PILOT]["depth"]["M0"]["base_nll"]
    for tag, n in TEACHER_ARMS.items():
        yield f"mt.n.{tag}", n, "count", f"records/teacher.md, teacher phase result: problems in {tag}-T"
    for tag, updates in (("D", DESIGN["d.drill.updates"][0]), ("D20", 20)):
        yield f"mt.pass.{tag}", updates * 32 / TEACHER_ARMS["D"], "acc", f"design: {updates} x 32 / {TEACHER_ARMS['D']}"
    for tag, label in TEACHER_STATES.items():
        yield f"mt.{tag}.free", acc[f"{label}|unforced"], "acc", f'{src} accuracy["{label}|unforced"]'
        for stage, suffix in AFTER.items():
            yield f"mt.{tag}.{stage}", acc[f"{label}{suffix}|forced"], "acc", f'{src} accuracy["{label}{suffix}|forced"]'
        rows = d.math_rows[(TEACHER_RUN, "math500", f"{label}+b", "u20", "forced", "first")]
        yield f"mt.cap.b.{tag}", pct(rows.capped), "cap", f"math_samples {TEACHER_RUN} {label}+b forced: share capped"
        b = r["depth"][label]["B"]
        yield f"mt.B.{tag}", b, "B", f'{src} depth["{label}"].B, on the pilot\'s base texts'
        yield f"mt.bfrac.{tag}", 100 * b / nll, "share", f"B of {label} over the pilot base's NLL, %"
    after = {stage: d.math_rows[(TEACHER_RUN, "math500", f"D-T-u140+{stage}", "u20", "forced", "first")]
             for stage in ("b", "it")}
    for stage, rows in after.items():
        yield f"mt.med.{stage}.D", float(rows.tokens.median()), "count", f"math_samples {TEACHER_RUN} D-T-u140+{stage}: median tokens"
    yield "mt.unbox.it.D", pct(after["it"].boxed == 0), "cap", f"math_samples {TEACHER_RUN} D-T-u140+it: share with no \\boxed"
    gap = abs(acc["D-T-u140|forced"] - acc["O-T-u140|forced"])
    yield "mt.parity", gap, "acc", f"{src} accuracy: |D-T-u140 - O-T-u140|, forced at handoff"
    for key, (x, y) in (("ex.O", ("D-T-u140", "O-T-u140")), ("ex.D20", ("D-T-u140", "D-T-u20")),
                        ("PO", ("P-T-u140", "O-T-u140"))):
        for stage in ("it", "b"):
            excess = r["excess"][f"{x} minus {y}|{stage}"]
            yield f"mt.{key}.{stage}", excess["points"], "acc", f'{src} excess["{x} minus {y}|{stage}"].points'
            yield f"mt.{key}.{stage}.ci", excess["interval"], "acc", f'{src} excess["{x} minus {y}|{stage}"].interval'
    for tag in ("D", "O"):
        for stage, suffix in (("h", ""), ("it", "+it")):
            key = TEACHER_STATES[tag] + suffix
            yield f"mt.aime.{tag}.{stage}", r["aime_forced"][key], "acc", f'{src} aime_forced["{key}"]'
    held = r["heldout_nll"]
    for tag, label in (("base", "M0"), ("O", "O-T-u140"), ("D", "D-T-u140")):
        yield f"mt.held.{tag}", held[label], "nll", f'{src} heldout_nll["{label}"]: held-out teacher traces'
    yield "mt.held.drop", 100 * (1 - held["O-T-u140"] / held["M0"]), "share", "held-out NLL under O-T-u140 below the base's, %"
    t = d.train[d.train.stage == "acquisition"]
    fall = {}
    for run, label in ((TEACHER_RUN, "D-T"), (PILOT, "D")):
        loss = t[(t.run == run) & (t.label == label)].set_index("update").loss_sum.sort_index()  # logs come in file order
        fall[label] = loss.loc[16:20].mean() - loss.loc[136:140].mean()
    yield ("mt.drill.ratio", fall["D-T"] / fall["D"], "prob",
           "train_log: the fall of the training loss from updates 16-20 to 136-140, D-T's over the pilot D's")
    own, common = r["displacement"], r["displacement_common"]
    for tag in ("D", "O", "P"):  # prediction 5: log-likelihood lost after 2 updates of instruction tuning
        state = f"{TEACHER_STATES[tag]}@u2"
        yield f"mt.d.{tag}.own", own[state], "B", f'{src} displacement["{state}"]: its own handoff texts'
        yield f"mt.d.{tag}.base", common[state], "B", f'{src} displacement_common["{state}"]: the base\'s texts'
    yield "mt.d.D.toward", -common["D-T-u140@u2"], "B", f"minus {src} displacement_common[D-T-u140@u2]"
    yield ("mt.d.ratio", own["D-T-u140@u2"] / own["O-T-u140@u2"], "count",
           "D-T-u140's own-text displacement after 2 updates over O-T-u140's")


def replay(d):
    """RP: R2's instruction tuning with 2 of each batch of 32 replayed from the arm's own training texts
    (records/revision.md). The excesses without replay are r2.ex and r4.ex60."""
    r, src = d.revision["RP"], "revision.json RP"
    for tag, label in (("D", "D-u140"), ("O", "O-u140"), ("DT", "D-T-u140"), ("OT", "O-T-u140")):
        for when, field in (("h", "handoff"), ("a", "u60")):
            yield f"rp.{tag}.{when}", r["accuracy"][label][field]["primary"], "acc", f'{src} accuracy["{label}"].{field}.primary'
    for tag, label in (("D", "D-u140"), ("DT", "D-T-u140")):
        e, name = r["excess"][label]["primary"], f'{src} excess["{label}"].primary'
        yield f"rp.ex.{tag}", e["points"], "acc", name + ".points: over the once-trained arm"
        yield f"rp.ex.{tag}.ci", e["interval"], "acc", name + ".interval"
        yield (f"rp.cut.{tag}", 100 * r["reduction"][label], "share",
               f'{src} reduction["{label}"], %: 1 - the excess over the excess without replay')


def realistic_stage(d):
    """RS: one pass over the No Robots rows at 1e-4 from the base (a fresh adapter), D, O and P."""
    r, src = d.revision["RS"], "revision.json RS"
    yield "rs.updates", r["updates"], "count", f"{src} updates: the rows at 32 per update"
    yield "rs.rows", r["rows"], "count", f"{src} rows: data/revision_ids.json rs_no_robots"
    for tag, label in (("base", "M0"), ("D", "D-u140"), ("O", "O-u140"), ("P", "P-u140")):
        for when, field in (("h", "handoff"), ("a", "after")):
            yield f"rs.{tag}.{when}", r["accuracy"][label][field]["primary"], "acc", f'{src} accuracy["{label}"].{field}.primary'
        lost, name = r["loss"][label]["primary"], f'{src} loss["{label}"].primary'
        yield f"rs.loss.{tag}", lost["points"], "acc", name + ".points"
        yield f"rs.loss.{tag}.ci", lost["interval"], "acc", name + ".interval"
    for tag, label in (("D", "D-u140"), ("P", "P-u140")):
        e, name = r["excess"][label], f'{src} excess["{label}"]'
        yield f"rs.ex.{tag}", e["points"], "acc", name + ".points: over O"
        yield f"rs.ex.{tag}.ci", e["interval"], "acc", name + ".interval"


RL_ORIGINS = {"D.it": ("D-u140+it", "O-u140+it"), "D.b": ("D-u140+b", "O-u140+b"),
              "DT.it": ("D-T-u140+it", "O-T-u140+it"), "DT.b": ("D-T-u140+b", "O-T-u140+b")}  # origin, reference


def relearning(d):
    """RL: from each after-stage state, accuracy and the recovered share R_k after k updates of relearning, and of
    the habit control after 20; O+IT's ceiling control."""
    r, src = d.revision["RL"], "revision.json RL"
    acc, refs = r["accuracy"], []
    for tag, (origin, ref) in RL_ORIGINS.items():
        yield f"rl.{tag}.u0", acc[f"{origin}@0"]["primary"], "acc", f'{src} accuracy["{origin}@0"].primary'
        refs.append(acc[f"{ref} (acc_ref)"]["primary"])
        yield f"rl.{tag}.ref", refs[-1], "acc", f'{src} accuracy["{ref} (acc_ref)"].primary'
        curve = [(f"u{k}", f"R{k}", f"{origin}+relearn@{k}") for k in (1, 2, 5, 20)] + [("hab", "habR", f"{origin}+habit@20")]
        for when, share, key in curve:
            if key in r["R"]:  # the habit control ran from D+IT and D+AO only
                yield f"rl.{tag}.{when}", acc[key]["primary"], "acc", f'{src} accuracy["{key}"].primary'
                yield f"rl.{tag}.{share}", r["R"][key]["primary"]["R"], "prob", f'{src} R["{key}"].primary.R'
                yield f"rl.{tag}.{share}.ci", r["R"][key]["primary"]["interval"], "prob", f'{src} R["{key}"].primary.interval'
    yield "rl.ref", [min(refs), max(refs)], "range acc", "smallest and largest rl.*.ref, the once-trained after-states"
    yield "rl.O.it.u0", acc["O-u140+it@0"]["primary"], "acc", f'{src} accuracy["O-u140+it@0"].primary'
    lost = []
    for k in (1, 2, 5, 20):
        key = f"O-u140+it+relearn@{k}"
        c = r["ceiling_loss"][key]["primary"]
        lost.append(abs(c["points"]))
        yield f"rl.O.it.u{k}", acc[key]["primary"], "acc", f'{src} accuracy["{key}"].primary'
        yield f"rl.ceil.u{k}", c["points"], "acc", f'{src} ceiling_loss["{key}"].primary.points'
        yield f"rl.ceil.u{k}.ci", c["interval"], "acc", f'{src} ceiling_loss["{key}"].primary.interval'
    yield "rl.ceil.max", max(lost), "acc", "largest |rl.ceil.u*|: how far O+IT moves at any k"


def sharpening(d):
    """SH: O's problems with the base's own solutions sampled at temperature 0.5, trained as O; its B against the
    paper's D and O (as the SH analysis scores them; mb.D140.3 is the same B of D), its accuracy, and its excess
    over O after each stage."""
    r, src = d.revision["SH"], "revision.json SH"
    name = f'{src} gate.arms["{r["gate"]["final"]}"]'
    arm = r["gate"]["arms"][r["gate"]["final"]]
    yield "sh.B", arm["B"], "B", name + ".B, on the base's common texts"
    yield "sh.B.ci", arm["interval"], "B", name + ".interval"
    yield "sh.texts", arm["texts"], "count", name + ".texts: data/revision_ids.json sh_sharp"
    for tag, label in (("D", "D-u140"), ("O", "O-u140")):
        yield f"sh.B.{tag}", r["B_paper"][label], "B", f'{src} B_paper["{label}"]'
    for when, field in (("h", "handoff"), ("it", "it"), ("b", "b")):
        yield f"sh.{when}", r["accuracy"]["O-sharp"][field]["primary"], "acc", f'{src} accuracy["O-sharp"].{field}.primary'
    for stage in ("it", "b"):
        e, name = r["excess"][stage]["primary"], f'{src} excess["{stage}"].primary'
        yield f"sh.ex.{stage}", e["points"], "acc", name + ".points: over O"
        yield f"sh.ex.{stage}.ci", e["interval"], "acc", name + ".interval"


SD_STAGES = {"it": "it", "lr": "it-lr1e-4"}  # instruction tuning at the defaults, and at 1e-4 for 60 updates


def seeds(d):
    """SD: two more runs of the 9B design (seeds s2 and s3: a redrawn drilled subset, a new order of O's texts and a
    new initialization), with the registered predictions and the pooled excess over the runs."""
    r, src = d.revision["SD"], "revision.json SD"
    parity, fresh = [], []
    for s in ("s2", "s3"):
        acc = r["accuracy"][s]
        for tag in "DOP":
            for key, stage in (("h", "u0"), *SD_STAGES.items()):
                yield f"sd.{s}.{tag}.{key}", acc[tag][stage]["primary"], "acc", f'{src} accuracy["{s}"]["{tag}"]["{stage}"].primary'
        parity.append(abs(acc["D"]["u0"]["primary"] - acc["O"]["u0"]["primary"]))
        for key, stage in SD_STAGES.items():
            for short, pair in (("ex", "D-O"), ("exP", "D-P"), ("PO", "P-O")):
                e, name = r["excess"][s][stage][pair], f'{src} excess["{s}"]["{stage}"]["{pair}"]'
                yield f"sd.{s}.{short}.{key}", e["points"], "acc", name + ".points"
                yield f"sd.{s}.{short}.{key}.ci", e["interval"], "acc", name + ".interval"
            fresh.append(abs(r["excess"][s][stage]["P-O"]["points"]))
    yield "sd.PO.absmax", max(fresh), "acc", "largest |sd.*.PO.*|: P's excess over O, both seeds and stages"
    yield "sd.parity.max", max(parity), "acc", "largest |D - O| forced accuracy at handoff, both seeds"
    for key, stage in SD_STAGES.items():  # the registered pooling: every run with the excess (paper, D2, seeds)
        pooled = r["pooled"][f"{stage}|D-O"]
        name = f'{src} pooled["{stage}|D-O"], over {", ".join(pooled["runs"])}'
        for stat in ("mean", "min", "max", "sd"):
            yield f"sd.pool.{key}.{stat}", pooled[stat], "acc", f"{name}: {stat}"
    for key, field in (("SD1", "1_intervals_above_zero"), ("SD2", "2_at_least_5"), ("SD3", "3_P_within_3_of_O")):
        held = all(cell[field] for cell in r["predictions"].values())
        yield f"sd.out.{key}", "held" if held else "failed", "stated", f'{src} predictions, all four cells: "{field}"'


SK_SETTINGS = {"b5": "base7-5", "b6": "base7-6", "w12": "words-12", "w16": "words-16"}
SK_STAGES = {"lr": "it-lr1e-4", "it": "it"}  # instruction tuning at 1e-4 for 60 updates (primary), at the defaults


def skill(d):
    """SK: a skill the base lacks within 2,048 tokens (base-7 multiplication of two 5-digit numbers): the smoke and
    the calibration of the base, the learning gate, and D-S, O-S and P-S at handoff and after each stage."""
    r, src = d.revision["SK"], "revision.json SK"
    for cap in (4096, 8192):
        for tag, setting in SK_SETTINGS.items():
            s, name = r["smoke"][f"{cap}|{setting}"], f'{src} smoke["{cap}|{setting}"]'
            yield f"sk.smoke{cap // 1024}.{tag}", 100 * s["correct"], "acc", name + ".correct, %"
            yield f"sk.smoke{cap // 1024}cap.{tag}", 100 * s["capped"], "cap", name + ".capped, %"
    yield "sk.smoke8tok.b5", r["smoke"]["8192|base7-5"]["mean_tokens"], "count", f'{src} smoke["8192|base7-5"].mean_tokens'
    yield "sk.cal", 100 * r["calibration"]["accuracy"]["base7-5"], "acc", f'{src} calibration.accuracy["base7-5"], %'
    yield "sk.cal.cap", 100 * r["calibration"]["capped"]["base7-5"], "cap", f'{src} calibration.capped["base7-5"], %'
    gate = r["gate"]
    yield "sk.gate", gate["gain"]["points"], "acc", f"{src} gate.gain.points: O-S-u140 minus the base"
    yield "sk.gate.ci", gate["gain"]["interval"], "acc", f"{src} gate.gain.interval"
    yield "sk.gate.cap", 100 * gate["capped_share"], "cap", f"{src} gate.capped_share, %"
    acc = r["accuracy"]
    yield "sk.base", acc["M0"], "acc", f'{src} accuracy["M0"]'
    for tag in "DOP":
        yield f"sk.{tag}.h", acc[f"{tag}-S-u140"], "acc", f'{src} accuracy["{tag}-S-u140"]'
        for stage, key in SK_STAGES.items():
            yield f"sk.{tag}.{stage}", acc[f"{tag}-S+{key}"], "acc", f'{src} accuracy["{tag}-S+{key}"]'
    gap = r["handoff_D_minus_O"]
    yield "sk.hDO", gap["points"], "acc", f"{src} handoff_D_minus_O.points"
    yield "sk.hDO.ci", gap["interval"], "acc", f"{src} handoff_D_minus_O.interval"
    gap, post_hoc = r["handoff_P_minus_D"], "post hoc, descriptive, not registered"
    yield "sk.PD.h", gap["points"], "acc", f"{src} handoff_P_minus_D.points ({post_hoc})"
    yield "sk.PD.h.ci", gap["interval"], "acc", f"{src} handoff_P_minus_D.interval ({post_hoc})"
    yield ("sk.fix.h", r["stages"]["it-lr1e-4"]["P_minus_best"]["handoff"], "acc",
           f'{src} stages["it-lr1e-4"].P_minus_best.handoff (the same in both stages)')
    for stage, key in SK_STAGES.items():
        s, name = r["stages"][key], f'{src} stages["{key}"]'
        for tag in "DOP":
            yield f"sk.loss.{tag}.{stage}", s["loss"][f"{tag}-S"]["points"], "acc", f'{name}.loss["{tag}-S"].points'
            yield f"sk.loss.{tag}.{stage}.ci", s["loss"][f"{tag}-S"]["interval"], "acc", f'{name}.loss["{tag}-S"].interval'
        for short, field in (("ex", "D_excess_over_O"), ("OD", "O_minus_D_after")):
            yield f"sk.{short}.{stage}", s[field]["points"], "acc", f"{name}.{field}.points"
            yield f"sk.{short}.{stage}.ci", s[field]["interval"], "acc", f"{name}.{field}.interval"
        yield f"sk.fix.{stage}", s["P_minus_best"]["after"], "acc", f"{name}.P_minus_best.after"
    for tag in "DOP":  # each model's loss in the primary stage as a share of its own handoff accuracy
        lost = r["stages"]["it-lr1e-4"]["loss"][f"{tag}-S"]["points"]
        yield (f"sk.share.{tag}.lr", 100 * lost / acc[f"{tag}-S-u140"], "acc", f'100 x {src} stages["it-lr1e-4"]'
               f'.loss["{tag}-S"].points / accuracy["{tag}-S-u140"], % ({post_hoc})')


FAMILIES = (constants, tces_4k, tces_stages, tces_16k, tces_clock, tces_panels, tces_losses, correlations,
            tces_divergence, anchor, original_study, math_pilot, math_aime_and_forms, math_data, replications,
            main_arms, reforcing, robustness, robustness_followup, teacher, replay, realistic_stage, relearning,
            sharpening, seeds, skill)

# ---------------------------------------------------------------------------------------------------
# Output and checks


def plain(value):
    """Python numbers for JSON."""
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


def build(d):
    registry = {}
    for family in FAMILIES:
        for key, value, rule, source in family(d):
            assert key not in registry, f"{key} is defined twice"
            value = plain(value)
            registry[key] = {"value": value, "text": fmt(value, rule), "rule": rule, "source": source,
                             "status": "pending" if value is None else "final"}
    return registry


HEADER = r"""% numbers.generated.tex: every number in the paper's text.
% Written by src/paper_numbers.py from outputs/, replay-v1/ and data/; do not edit. Usage in the text: \nv{key}.
% Each line ends with its source. A pending value prints ?? until its result exists.
\makeatletter
\newcommand{\defnum}[2]{\expandafter\def\csname nv@#1\endcsname{#2}}
\newcommand{\nv}[1]{\ifcsname nv@#1\endcsname\csname nv@#1\endcsname\else\textbf{??}\PackageWarning{numbers}{Undefined number #1}\fi}
\makeatother
"""


def render(registry):
    lines = [f"\\defnum{{{key}}}{{{e['text']}}} % {e['source']}" for key, e in registry.items()]
    return HEADER + "\n" + "\n".join(lines) + "\n"


def defined(text):
    return dict(re.findall(r"^\\defnum\{([^}]*)\}\{(.*?)\} %", text, re.M))


def used_keys():
    """Every \nv{key} in the paper's text, comments removed; None where the text is not present."""
    if not PAPER[-1].exists():
        return None
    used = set()
    for path in PAPER:
        for line in path.read_text().splitlines():
            used.update(re.findall(r"\\nv\{([^}]*)\}", re.sub(r"(?<!\\)%.*", "", line)))
    return used


def check(registry, text):
    failures = []
    if not TEX.exists():
        failures.append(f"{TEX.name} does not exist: run python src/paper_numbers.py")
    elif TEX.read_text() != text:
        old, new = defined(TEX.read_text()), defined(text)
        stale = [k for k in sorted(old.keys() | new.keys()) if old.get(k) != new.get(k)]
        failures.append(f"{TEX.name} is stale ({len(stale)} keys differ):\n  " + "\n  ".join(
            f"{k}: {old.get(k, '(absent)')} -> {new.get(k, '(absent)')}" for k in stale[:40]))
    used = used_keys()
    if used is None:  # a copy without the paper's text: check the registry alone
        print(f"{len(registry)} keys; the paper's text is not present, so only the registry is checked")
        for failure in failures:
            print("FAIL: " + failure)
        return not failures
    undefined = sorted(used - registry.keys())
    if undefined:
        failures.append(f"the paper uses {len(undefined)} undefined keys: " + ", ".join(undefined))
    pending = [k for k, e in registry.items() if e["status"] == "pending"]
    if pending and os.environ.get("FINAL") == "1":
        failures.append(f"FINAL=1 and {len(pending)} keys are pending: " + ", ".join(pending))
    unused = sorted(registry.keys() - used)
    print(f"{len(registry)} keys: {len(registry) - len(pending)} final, {len(pending)} pending; "
          f"the paper uses {len(used)}")
    if unused:
        print(f"{len(unused)} keys the paper does not use: " + ", ".join(unused))
    for failure in failures:
        print("FAIL: " + failure)
    return not failures


def main():
    registry = build(Data())
    text = render(registry)
    if "--check" in sys.argv[1:]:
        sys.exit(0 if check(registry, text) else 1)
    TEX.write_text(text)
    (OUT / "summaries" / "numbers.json").write_text(json.dumps(registry, indent=1))
    pending = sum(e["status"] == "pending" for e in registry.values())
    print(f"wrote {TEX.relative_to(REPO)} and outputs/summaries/numbers.json: {len(registry)} keys, {pending} pending")


if __name__ == "__main__":
    main()
