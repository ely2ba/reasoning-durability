"""The TCES verifier (src/common/tces.py): correct(completion, task) and the probes' rule score(text, task).

Replay equivalence: every stored TCES forced or free completion in DuraSeed-v1
(runs/{decision-probe,repetition-probe,anchor}/**/{think,competence,unforced}.jsonl: 201 files,
89,472 rows) must score exactly as stored under the probes' rule

    correct = verify(TEMPLATE_TAG.sub("", text), task)

`text` already holds the forced prefix and any continuation resumed after a restated answer
template, so resumption needs no re-simulation; the strict score is `correct and resumed == 0`.
An independent check before panel.jsonl existed (tasks taken from DuraSeed-v1): this rule reproduces all
89,472 stored scores with DuraSeed-v1's verifier; without the template removal only 87% agree.

Tasks come from data/tces/panel.jsonl (384 monitor items). Rows of the E1 profile panel are skipped
and counted. All rows are scored (about 20 s); the rare failure cases are too few to sample.
"""

import json
import re
from fractions import Fraction

import pytest

from conftest import DURASEED, REPO, common, needs

TEMPLATE_TAG = re.compile(r"<answer>([^\d]*?)</answer>", re.S)  # a restated template: no digits
KINDS = ("think.jsonl", "competence.jsonl", "unforced.jsonl")
RUN_DIRS = ("decision-probe", "repetition-probe", "anchor")
TASK = {"operands": [3, 3, 4], "allowed_ops": ["+", "-", "*", "/"], "target": [10, 1], "constraints": {}}


def rule():
    """The probes' rule from the module (score), or built here from its verifier."""
    try:
        return common("tces", "score")
    except pytest.skip.Exception:
        verify = common("tces", "correct")
        return lambda text, task: verify(TEMPLATE_TAG.sub("", text), task)


# ---- unit cases (no data needed) -------------------------------------------------------------------

@pytest.mark.parametrize("completion, expected", [
    ("<answer>3*3+4-3</answer>", False),            # uses 3 three times
    ("<answer>3+3+4</answer>", True),
    ("<answer>(3+3)+4</answer>", True),
    ("<answer>4*3-3/1</answer>", False),             # 1 is not an operand
    ("<answer>-3+3*4+1</answer>", False),            # unary sign
    ("<answer>3 3 4</answer>", False),               # adjacent numbers must be rejected, not crash
    ("<answer>33+4</answer>", False),
    ("<answer>(3+3+4</answer>", False),              # unbalanced
    ("<answer>3+3+4</answer> <answer>3+3+4</answer>", False),  # exactly one answer tag
    ("no tag 3+3+4", False),
    ("<answer>3 + 3 + 4</answer>", True),            # whitespace is allowed
])
def test_verifier_cases(completion, expected):
    correct = common("tces", "correct")
    assert bool(correct(completion, TASK)) is expected


def test_exact_rational_target():
    correct = common("tces", "correct")
    task = {"operands": [1, 3, 5], "allowed_ops": ["+", "-", "*", "/"], "target": [16, 3], "constraints": {}}
    assert 5 + Fraction(1, 3) == Fraction(16, 3)
    assert correct("<answer>5+1/3</answer>", task)
    assert not correct("<answer>5-1/3</answer>", task)


def test_rule_removes_restated_templates_only():
    score = rule()
    assert score("<answer>EXPRESSION</answer> ... <answer>3+3+4</answer>", TASK)
    assert not score("<answer>3*4-3</answer> ... <answer>3+3+4</answer>", TASK)  # two real answers
    assert TEMPLATE_TAG.sub("", "a <answer>EXPR</answer> b <answer>(3+4)</answer>") == "a  b <answer>(3+4)</answer>"


@pytest.mark.parametrize("text, expected", [
    ("... <answer>EXPRESSION</answer>", True),
    ('... <answer>EXPRESSION</answer>".', True),     # up to three trailing non-word characters
    ("... <answer>3+4</answer>", False),             # a real answer has digits
    ("... <answer>EXPRESSION</answer> and then", False),
    ("no tag", False),
])
def test_template_stop(text, expected):
    template_stop = common("tces", "template_stop")
    assert bool(template_stop(text)) is expected


# ---- replay against the stored runs -----------------------------------------------------------

def panel_tasks():
    path = needs(REPO / "data" / "tces" / "panel.jsonl", "data/tces/panel.jsonl")
    rows = [json.loads(line) for line in path.open()]
    assert len(rows) == 384 and len({r["task_id"] for r in rows}) == 384
    assert sum(r.get("role") == "targeted" for r in rows) in (0, 192)
    return {r["task_id"]: r.get("task", r) for r in rows}


def stored_files():
    runs = needs(DURASEED / "runs", "DuraSeed-v1 runs")
    files = sorted(p for d in RUN_DIRS for p in (runs / d).glob("**/*.jsonl") if p.name in KINDS)
    if not files:
        pytest.skip("no stored TCES probe files")
    return runs, files


@pytest.mark.duraseed
def test_stored_strict_score_is_correct_without_resumption():
    runs, files = stored_files()
    bad = [str(path.relative_to(runs)) for path in files for row in map(json.loads, path.open())
           if "resumed" in row and row["strict_correct"] != (row["correct"] and row["resumed"] == 0)]
    assert not bad, f"{len(bad)} rows break strict = correct and not resumed: {bad[:5]}"


@pytest.mark.duraseed
def test_replay_equivalence():
    score = rule()
    tasks = panel_tasks()
    runs, files = stored_files()
    checked, wrong, foreign = 0, [], {}
    for path in files:
        for row in map(json.loads, path.open()):
            task = tasks.get(row["task_id"])
            if task is None:
                name = str(path.relative_to(runs).parent.parent)
                foreign[name] = foreign.get(name, 0) + 1
                continue
            checked += 1
            if bool(score(row["text"], task)) != row["correct"]:
                wrong.append((str(path.relative_to(runs)), row["task_id"][:16], row["draw"]))
    assert checked == 77_184, checked  # 201 files; the 12,288 E1 rows (256 problems x 16 draws x 3) are outside the panel
    assert all("E1-" in name for name in foreign), f"rows outside the panel: {foreign}"
    assert not wrong, f"{len(wrong)} of {checked} rows differ from the stored score (e.g. {wrong[:5]})"
