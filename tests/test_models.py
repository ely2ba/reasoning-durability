"""Model families (src/common/models.py): Family.strip_turn_end must agree with the function the
re-forcing probe ran (DuraSeed-v1 tools/run_reforce.py), on every stored forced sample of the math
pilot and of the Nemotron run.

The DuraSeed-v1 function is read from its source file and run in isolation (its module imports the
whole DuraSeed-v1 runtime), with the end tokens run_reforce.py uses: the tokenizer's EOS for Qwen,
<|im_end|> and </s> for Nemotron.
"""

import ast
import json
from types import SimpleNamespace

import pytest

from conftest import DURASEED, common, needs

pytest.importorskip("transformers", reason="needs the experiments extra: pip install -e \".[experiments]\"")

MATH = DURASEED / "runs" / "math"
NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def duraseed_strip():
    """strip_turn_end and TURN_ENDS from DuraSeed-v1 tools/run_reforce.py, executed on their own."""
    source = needs(DURASEED / "tools" / "run_reforce.py", "DuraSeed-v1 tools/run_reforce.py").read_text()
    tree = ast.parse(source)
    namespace = {}
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        if isinstance(target, ast.Tuple) and any(getattr(t, "id", None) == "TURN_ENDS" for t in target.elts):
            exec(compile(ast.Module([node], []), "run_reforce.py", "exec"), namespace)
        if isinstance(node, ast.FunctionDef) and node.name == "strip_turn_end":
            exec(compile(ast.Module([node], []), "run_reforce.py", "exec"), namespace)
    assert "TURN_ENDS" in namespace and "strip_turn_end" in namespace
    return namespace["strip_turn_end"]


def stored_forced(run):
    root = needs(MATH / run / "eval" / "math500", f"DuraSeed-v1 run {run}")
    files = sorted(root.glob("*/*-forced.jsonl"))
    if not files:
        pytest.skip(f"no forced samples in {run}")
    return files


@pytest.mark.duraseed
@pytest.mark.parametrize("run, model", [("pilot-20260924", "Qwen/Qwen3.5-9B-Base"), ("main-nemotron-20260924", NEMOTRON)])
def test_strip_turn_end_matches_the_probe(run, model):
    Family = common("models", "Family")
    files = stored_forced(run)
    family = Family(model)
    reference = duraseed_strip()
    ctx = SimpleNamespace(tok=family.tok, end_tokens={11, 2} if model == NEMOTRON else {family.tok.eos_token_id})
    checked, differ, stripped = 0, [], 0
    for path in files:
        for row in map(json.loads, path.open()):
            ids = row["token_ids"]
            mine, theirs = family.strip_turn_end(ids), reference(ctx, ids)
            checked += 1
            stripped += len(mine) != len(ids)
            if mine != theirs:
                differ.append((path.parent.name, row["task_id"], row["draw"]))
    assert checked > 5000 and stripped > 0
    assert not differ, f"{len(differ)} of {checked} samples differ: {differ[:5]}"


@pytest.mark.duraseed
def test_strip_turn_end_removes_exactly_the_end():
    """On the pilot's samples the stripped text plus the removed end is the stored text."""
    Family = common("models", "Family")
    files = stored_forced("pilot-20260924")
    family = Family("Qwen/Qwen3.5-9B-Base")
    ends = {"\n\nUser:", "\nUser:", family.tok.eos_token}
    for path in files[:6]:
        for row in map(json.loads, path.open()):
            if row["capped"]:
                continue
            text, kept = family.tok.decode(row["token_ids"]), family.tok.decode(family.strip_turn_end(row["token_ids"]))
            assert text.startswith(kept) and text[len(kept):] in ends, (row["task_id"], repr(text[-20:]))
