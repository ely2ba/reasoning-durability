"""The math scorer (src/common/maths.py): extract(text) -> int | None.

1. The 30 unit examples of the registered scorer (DuraSeed-v1 tools/math_score.py, SHA-256 f9194799...).
2. It recovers all 311 integer answers of MATH-500 from the reference solutions.
3. Replay: on stored math evaluations (DuraSeed-v1 runs/math/*/eval), it gives the stored `correct`.
"""

import glob
import json
import re

import pytest

from conftest import DURASEED, HF_CACHE, REPO, common, needs, sample_rows

EXAMPLES = [
    (r"\boxed{\text{5}}", 5),
    (r"\boxed{5.}", 5),
    (r"\boxed{5.0}", 5),
    (r"\boxed{-3}", -3),
    (r"\boxed{1,000}", 1000),
    (r"\boxed{\frac{1}{2}}", None),
    (r"First \boxed{3}, corrected: \boxed{4}", 4),
    (r"The answer is \boxed{5", None),
    (r"\boxed{3} then \boxed{5", None),
    (r"The answer is 5.", None),
    (r"\boxed{1{,}000}", 1000),
    (r"\boxed{12\,345}", 12345),
    (r"\boxed{-1{,}234{,}567}", -1234567),
    (r"\boxed{1,2}", None),
    (r"\boxed{2x - 3y + z - 6 = 0}", None),
    (r"\boxed{5.5}", None),
    (r"$\boxed{ 42 }$", 42),
    (r"\boxed{\boxed{7}}", 7),
    (r"\boxed {8}", 8),
    (r"\boxed 9", None),
    (r"\boxed{}", None),
    (r"\boxed{x = 5}", 5),
    (r"\boxed{x=5, y=3}", None),
    (r"\boxed{45^\circ}", 45),
    (r"\boxed{\$18} or \boxed{25\%}", 25),
    (r"\boxed{12 \text{ cm}}", 12),
    (r"\boxed{\text{(B)}}", None),
    (r"\boxed{2 \text{ and } 100}", None),
    (r"\boxed{2^{10}}", None),
    ("\\boxed{\u22124}", -4),
]


def test_thirty_unit_examples():
    extract = common("maths", "extract")
    assert len(EXAMPLES) == 30
    wrong = [(text, want, extract(text)) for text, want in EXAMPLES
             if extract(text) != want or type(extract(text)) is not type(want)]
    assert not wrong, f"{len(wrong)} of 30 unit examples differ: {wrong}"


def math500_rows():
    files = glob.glob(str(HF_CACHE / "datasets--HuggingFaceH4--MATH-500" / "snapshots" / "*" / "test.jsonl"))
    if not files:
        pytest.skip(f"MATH-500 not in the Hugging Face cache ({HF_CACHE})")
    return [json.loads(line) for line in open(files[0])]


@pytest.mark.duraseed
def test_math500_reference_answers_recovered():
    extract = common("maths", "extract")
    rows = [r for r in math500_rows() if re.fullmatch(r"-?\d+", r["answer"].strip())]
    assert len(rows) == 311
    wrong = [(r["unique_id"], r["answer"], extract(r["solution"])) for r in rows
             if extract(r["solution"]) != int(r["answer"])]
    assert not wrong, f"{len(wrong)} of 311 reference answers not recovered: {wrong[:10]}"


@pytest.mark.duraseed
def test_frozen_eval_ids_are_the_221_integer_answers_at_levels_3_to_5():
    path = REPO / "data" / "math" / "eval_math500_ids.json"
    if not path.exists():
        pytest.skip("data/math/eval_math500_ids.json not built yet")
    frozen = json.loads(path.read_text())
    items = frozen["items"] if isinstance(frozen, dict) else frozen
    items = [i if isinstance(i, dict) else {"unique_id": i} for i in items]
    source = {r["unique_id"]: r for r in math500_rows()}
    expected = {u for u, r in source.items() if re.fullmatch(r"-?\d+", r["answer"].strip()) and r["level"] >= 3}
    assert len(expected) == 221 and len(items) == 221 and {i["unique_id"] for i in items} == expected
    for i in items:
        assert i.get("answer", int(source[i["unique_id"]]["answer"])) == int(source[i["unique_id"]]["answer"])
        assert i.get("level", source[i["unique_id"]]["level"]) == source[i["unique_id"]]["level"]


def stored_answers():
    data = needs(DURASEED / "runs" / "math" / "data" / "eval_math500.jsonl", "DuraSeed-v1 math eval set")
    answers = {r["unique_id"]: int(r["answer"]) for r in map(json.loads, data.open())}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return answers
    for year in (2025, 2026):
        found = glob.glob(str(HF_CACHE / f"datasets--MathArena--aime_{year}" / "snapshots" / "*" / "data" / "*.parquet"))
        for row in (pq.read_table(found[0]).to_pylist() if found else []):
            answers[f"aime{year}-{row['problem_idx']}"] = int(row["answer"])
    return answers


@pytest.mark.duraseed
def test_replay_stored_math_scores():
    """Every stored `correct` (sampled per file, or all rows with REPLAY_FULL=1) is reproduced."""
    extract = common("maths", "extract")
    runs = needs(DURASEED / "runs" / "math", "DuraSeed-v1 math runs")
    answers = stored_answers()
    files = sorted(runs.glob("*/eval/*/*/*.jsonl"))
    if not files:
        pytest.skip("no stored math evaluations")
    checked, wrong, unknown = 0, [], 0
    for path in files:
        for line in sample_rows(path.open().readlines()):
            row = json.loads(line)
            if row["task_id"] not in answers:
                unknown += 1
                continue
            checked += 1
            if (extract(row["text"]) == answers[row["task_id"]]) != row["correct"]:
                wrong.append((str(path.relative_to(runs)), row["task_id"], row["draw"]))
    assert checked > 0
    assert not wrong, f"{len(wrong)} of {checked} stored scores differ (e.g. {wrong[:5]})"
