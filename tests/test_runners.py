"""Offline equivalence of the experiment runners with the DuraSeed-v1 code that produced the paper's runs.

Both codebases are imported: this repository from src/ (first on sys.path), DuraSeed-v1 from its src/
and tools/. Every comparison is exact at the token level (loss weights to a relative 1e-9); a failure names the
item, the field and the first differing index. Skips when DuraSeed-v1, its runs, the Hugging Face
cache or this repository's inputs/ are absent. No network, no Tinker session.

run_math.py   1 Math.prompt vs the stored rft prompts (pilot, 35B, Nemotron); 2 pilot_sets and the D2
              rows vs arms.json and d2.json; 3 the P slots vs run_math_main.p_rows; 4 raw_datums vs
              duraseed.endpoint_training.raw_datums; 5 Family.later_stages vs Context.datums,
              instruction_datums and nemotron_format.answer_only_datums
run_tces.py   6 the panel, stops and prefixes vs Context; 7 the drill rows vs run_repetition_probe.subsets;
              8 the anchor rows and datums vs run_anchor, the FN datums vs instruction_datums;
              9 Family.completion vs _completion, tces.template_stop vs run_decision_probe.template_stop;
              10 the score texts vs run_depth_full and the stored M0 scores; 11 inputs/math vs the runs' inputs
run_teacher   12 the student arms, P-T slots and held-out traces vs run_teacher.arms, the run's arms.json,
              its logged training tokens and its scored held-out texts; parse, the teacher prompts and
              problems; the stored traces meet the keep rule; the replayed probe's tokens, losses and texts
run_robustness 13 R2's 1,920 examples vs instruction_datums; the states, R1's texts and R3's base texts vs
              run_robustness and the stored scores; R1-R3's logged training tokens
               14 R5's frozen chat ids vs the SHA rule (chat_datums), the run and the keyword flags; the chat
              datums vs chat_datums; R4 and R5's logged training tokens
"""

import asyncio
import collections
import hashlib
import importlib
import importlib.util
import json
import math
import sys
from types import SimpleNamespace

import pytest

from conftest import DURASEED, HF_CACHE, REPO, SRC, needs, sample_rows

pytestmark = pytest.mark.duraseed  # every test here needs the original DuraSeed-v1 runs

QWEN, NEMOTRON = "Qwen/Qwen3.5-9B-Base", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MATH = DURASEED / "runs/math"
CORPUS = DURASEED / "runs/endpoint-clone/endpoint-clone-20260910T162817Z/acquisition/corpus"
ANCHOR = DURASEED / "runs/anchor/anchor-20260924/anchor.jsonl"
SHARD = HF_CACHE / "datasets--allenai--tulu-3-sft-mixture"


def jsonl(path):
    return [json.loads(line) for line in open(path)]


def first_diff(a, b):
    """None when equal, else (first differing index, mine, theirs, my length, their length)."""
    a, b = list(a), list(b)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y, len(a), len(b)
    return None if len(a) == len(b) else (min(len(a), len(b)), "end", "end", len(a), len(b))


def fields(d):
    ints = lambda v: v.tolist() if hasattr(v, "tolist") else list(v)  # noqa: E731
    return (list(d.model_input.to_ints()), ints(d.loss_fn_inputs["target_tokens"]), ints(d.loss_fn_inputs["weights"]))


def same_datums(mine, theirs, what):
    assert len(mine) == len(theirs), f"{what}: {len(mine)} datums, DuraSeed-v1 {len(theirs)}"
    for i, (x, y) in enumerate(zip(mine, theirs)):
        (xi, xt, xw), (yi, yt, yw) = fields(x), fields(y)
        for name, u, v in (("model_input", xi, yi), ("target_tokens", xt, yt)):
            diff = first_diff(u, v)
            assert diff is None, f"{what}: datum {i}, {name}: (index, mine, theirs, lengths) {diff}"
        assert len(xw) == len(yw), f"{what}: datum {i}, weights: lengths {len(xw)} and {len(yw)}"
        bad = next((j for j, (u, v) in enumerate(zip(xw, yw)) if not math.isclose(u, v, rel_tol=1e-9)), None)
        assert bad is None, f"{what}: datum {i}, weights: index {bad}, {xw[bad]!r} and {yw[bad]!r}"


def pairs_of(rows):
    return [dict(prompt=r["prompt_token_ids"], response=r["generation"]["completion_token_ids"]) for r in rows]


# ---- the two codebases -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    """DuraSeed-v1's tools, SDK bundle and decision-probe Context (tokenizer, panel, Stage-B datums)."""
    needs(DURASEED / "tools/run_decision_probe.py", "DuraSeed-v1")
    for path in (DURASEED / "src", DURASEED / "tools"):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    try:
        mods = {n: importlib.import_module(n) for n in ("run_decision_probe", "run_repetition_probe", "run_anchor",
                                                          "run_instruction_continuation", "run_math_main",
                                                          "nemotron_format")}
        mods["training"] = importlib.import_module("duraseed.endpoint_training")
        mods["sampling"] = importlib.import_module("duraseed.runtime.sampling")
        sdk = importlib.import_module("duraseed.runtime").load_sdk()
        ctx = mods["run_decision_probe"].Context(DURASEED, tmp_path_factory.mktemp("ctx"), cap=0.0, prior=0.0)
    except (ImportError, OSError) as error:
        pytest.skip(f"DuraSeed-v1 cannot be loaded here: {error}")
    ctx.sdk = sdk
    return SimpleNamespace(ctx=ctx, sdk=sdk, **mods)


@pytest.fixture(scope="module")
def new():
    try:
        return SimpleNamespace(run_math=importlib.import_module("run_math"), run_tces=importlib.import_module("run_tces"),
                               models=importlib.import_module("common.models"),
                               tces=importlib.import_module("common.tces"),
                               io=importlib.import_module("common.tinker_io"))
    except ImportError as error:
        pytest.skip(f"this repository's runners cannot be imported here: {error}")


@pytest.fixture(scope="module")
def family(new):
    cache = {}

    def get(model):
        if model not in cache:
            cache[model] = new.models.Family(model)
        return cache[model]
    return get


@pytest.fixture(scope="module")
def maths(new, family, tmp_path_factory):
    """A Math object with no Tinker session (prompt() and the P slots need none)."""
    for name in ("pool", "eval_math500", "aime"):
        needs(REPO / f"inputs/math/{name}.jsonl", f"inputs/math/{name}.jsonl")
    return lambda model: new.run_math.Math(None, family(model), tmp_path_factory.mktemp("math"))


@pytest.fixture(scope="module")
def robots():
    return jsonl(needs(REPO / "inputs/no_robots.jsonl", "inputs/no_robots.jsonl"))


@pytest.fixture(scope="module")
def instruction(ds):
    """DuraSeed-v1's D3 (640) and FN (4,480) No Robots streams: (datums, ids) each."""
    needs(SHARD, "the cached Tulu-3 SFT mixture")
    ic = ds.run_instruction_continuation
    d3 = ic.instruction_datums(ds.ctx)
    return {"d3": d3, "fn": ic.instruction_datums(ds.ctx, examples=4480, namespace="anchor-20260924-fn", exclude=d3[1])}


# ---- run_math.py -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("run, model", [("pilot-20260924", QWEN), ("main-35b-20260924", QWEN),
                                        ("main-nemotron-20260924", NEMOTRON)])
def test_1_prompt_matches_every_stored_rft_prompt(maths, run, model):
    rows = [r for r in jsonl(needs(MATH / run / "rft.jsonl", f"{run} rft.jsonl")) if not r["easy"]]
    m = maths(model)
    problems = {p["task_id"]: p["problem"] for p in m.pool}
    for r in rows:
        diff = first_diff(m.prompt(problems[r["task_id"]]), r["prompt_token_ids"])
        assert diff is None, f"{run} {r['task_id']}: prompt_token_ids {diff}"
    assert sum(r["correct"] for r in rows) >= 4480


@pytest.mark.parametrize("run", ["pilot-20260924", "main-35b-20260924", "main-nemotron-20260924"])
def test_2_pilot_sets_equal_arms_json(new, run):
    root = needs(MATH / run / "arms.json", f"{run} arms.json").parent
    sets, arms = new.run_math.pilot_sets(root), json.loads((root / "arms.json").read_text())
    for arm in ("D", "O"):
        assert len(sets[arm]) == len(arms[arm]) == {"D": 579, "O": 4480}[arm]
        diff = first_diff([r["task_id"] for r in sets[arm]], arms[arm])
        assert diff is None, f"{run} {arm}: task order {diff}"


def test_2_d2_rows_equal_the_main_run(ds, new):
    sets = new.run_math.pilot_sets(needs(MATH / "pilot-20260924", "the pilot"))
    stored = jsonl(needs(MATH / "main-9b-20260924/d2.json", "main-9b d2.json"))  # JSON lines of task ids
    inside, ordered = {r["task_id"] for r in sets["D"]}, new.run_math.ordered
    chosen = ordered([r for r in sets["O"] if r["task_id"] not in inside], "main-20260924:s2")[:new.run_math.SMALL]
    mine = [r["task_id"] for r in ordered(chosen, "main-20260924:s2:order")]
    assert first_diff(mine, stored) is None, f"D2 task order {first_diff(mine, stored)}"
    assert mine == [r["task_id"] for r in ds.run_math_main.d2_rows(sets)]


def test_3_p_slots_equal_p_rows(ds, new, maths):
    main = needs(MATH / "main-9b-20260924/p_samples.jsonl", "main-9b p_samples.jsonl").parent
    d_rows = new.run_math.pilot_sets(MATH / "pilot-20260924")["D"]
    m = maths(QWEN)
    m.root = main  # every D problem is in p_samples.jsonl, so nothing is sampled and nothing is written
    mine = asyncio.run(new.run_math.p_samples(m, None, d_rows))
    have = {r["task_id"]: r for r in jsonl(main / "p_samples.jsonl")}
    theirs = ds.run_math_main.p_rows([have[r["task_id"]] for r in d_rows])
    assert len(mine) == len(theirs) == 4480
    for s, (x, y) in enumerate(zip(mine, theirs)):
        assert x["prompt_token_ids"] == y["prompt_token_ids"], f"P slot {s}: prompt_token_ids"
        diff = first_diff(x["generation"]["completion_token_ids"], y["generation"]["completion_token_ids"])
        assert diff is None, f"P slot {s}: completion_token_ids {diff}"


@pytest.mark.parametrize("arm", ["D", "O"])
def test_4_raw_datums_equal(ds, new, arm):
    rows = new.run_math.pilot_sets(needs(MATH / "pilot-20260924", "the pilot"))[arm]
    same_datums(new.io.raw_datums(pairs_of(rows), new.run_math.MAX_LENGTH),
                ds.training.raw_datums(SimpleNamespace(sdk=ds.sdk), rows, max_length=17000), f"raw_datums {arm}")


def test_5_later_stages_qwen(ds, family, robots, instruction):
    diff = first_diff([r["id"] for r in robots[:640]], instruction["d3"][1])
    assert diff is None, f"inputs/no_robots.jsonl[:640] ids {diff}"
    stages = family(QWEN).later_stages()
    same_datums(stages["b"], ds.ctx.datums, "Stage B, Qwen")
    same_datums(stages["it"], instruction["d3"][0], "instruction tuning, Qwen")


def test_5_later_stages_nemotron(ds, family, robots):
    needs(SHARD, "the cached Tulu-3 SFT mixture")
    nf = ds.nemotron_format
    pairs, tok = nf.stage_pairs(DURASEED, ds.ctx.stage_b_records), nf.tokenizer()
    stages = family(NEMOTRON).later_stages()
    for stage in ("b", "it"):
        same_datums(stages[stage], nf.answer_only_datums(ds.sdk, tok, pairs[stage]), f"{stage}, Nemotron")


# ---- run_tces.py -----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tces_runner(new, tmp_path_factory):
    return new.run_tces.Tces(None, tmp_path_factory.mktemp("tces"), 16384)  # creates no session


def test_6_panel_stops_and_prefixes_equal_context(ds, new, family, tces_runner):
    items, panel = ds.ctx.items, tces_runner.items
    diff = first_diff([p["task_id"] for p in panel], [i["task_id"] for i in items])
    assert diff is None, f"panel task order {diff}"
    for p, i in zip(panel, items):
        assert p["role"] == i["role"], f"{p['task_id']}: role"
        diff = first_diff(p["prompt_token_ids"], i["prompt"])
        assert diff is None, f"{p['task_id']}: prompt_token_ids {diff}"
        assert family(QWEN).render(p["prompt"]) == i["prompt"], f"{p['task_id']}: the prompt text renders differently"
        task = i["task"].model_dump(mode="json")
        assert p["task"]["operands"] == task["operands"], f"{p['task_id']}: operands"
        assert p["task"]["allowed_ops"] == task["allowed_ops"], f"{p['task_id']}: allowed_ops"
        assert p["task"]["target"] == [task["target"]["numerator"], task["target"]["denominator"]], p["task_id"]
        limits = {**new.tces.LIMITS, **p["task"]["constraints"]}  # the new verifier's limits for this item
        assert {k: task["constraints"][k] for k in limits} == limits, f"{p['task_id']}: constraints"
        assert task["constraints"]["use_each_once"] is True, f"{p['task_id']}: use_each_once (always enforced here)"
    assert new.run_tces.STOPS == ds.ctx.stops, "probe stops"
    assert tces_runner.prefixes == ds.ctx.prefixes, "forced prefixes"


def test_7_drill_rows_equal_subsets(ds, new, monkeypatch):
    monkeypatch.setattr(new.run_tces, "CORPUS", needs(CORPUS, "the clone corpus"))
    corpus = new.run_tces.Tces.corpus(None)
    assert len(corpus) == 6400, "sample ids are not unique across the corpus"
    subsets = json.loads(needs(REPO / "data/tces/subsets.json", "data/tces/subsets.json").read_text())
    rp = ds.run_repetition_probe
    theirs = {"U": rp.subsets(DURASEED)["U"], "F": rp.subsets(DURASEED)["F"],
              "U2": rp.subsets(DURASEED, "anchor-20260924-u2")["U"]}
    for arm, rows in theirs.items():
        ids = subsets[arm]["sample_ids"]
        diff = first_diff(ids, [r["generation"]["sample_id"] for r in rows])
        assert diff is None, f"{arm}: sample_ids {diff}"
        for k, (x, y) in enumerate(zip([corpus[i] for i in ids], pairs_of(rows))):
            for name in ("prompt", "response"):
                diff = first_diff(x[name], y[name])
                assert diff is None, f"{arm} row {k} ({ids[k]}): {name} {diff}"


def test_8_anchor_prompts_and_rows_follow_run_anchor(ds, new, tces_runner):
    """The runner's anchor construction (prompts, hashed order, six draws, first 4,480) gives the stored rows."""
    prompts = {r["task_id"]: r["prompt_token_ids"] for r in jsonl(REPO / "data/tces/clone_prompts.jsonl")}
    order = sorted(prompts, key=lambda t: hashlib.sha256(f"anchor-20260924:{t}".encode()).hexdigest())
    theirs = ds.run_anchor.corpus_prompts(DURASEED)
    diff = first_diff(order, [t for t, _ in theirs])
    assert diff is None, f"prompt order {diff}"
    for t, ids in theirs:
        assert prompts[t] == ids, f"{t}: clone prompt tokens"
    stored, think = jsonl(needs(ANCHOR, "anchor.jsonl")), tces_runner.prefixes["think"]
    expected = [(t, i) for t in order for i in range(new.run_tces.PER_PROMPT)][:new.run_tces.ANCHORS]
    diff = first_diff(expected, [(r["task_id"], r["draw"]) for r in stored])
    assert diff is None, f"anchor rows (task_id, draw) {diff}"
    for k, r in enumerate(stored):
        assert r["prompt_token_ids"] == prompts[r["task_id"]] + think, f"anchor row {k}: prompt"


@pytest.mark.parametrize("weight", [1.0, 0.25])
def test_8_anchor_datums_equal_weighted(ds, new, weight):
    stored = jsonl(needs(ANCHOR, "anchor.jsonl"))
    same_datums(new.io.raw_datums(pairs_of(stored), new.run_tces.MAX_LENGTH + 8, weight),
                ds.run_anchor.weighted(ds.ctx, stored, weight), f"anchor datums, weight {weight}")


def test_8_released_anchor_file_is_readable_by_the_runner(new):
    """drill() reads the released anchor.jsonl through run_tces.training_rows, which gives raw_datums
    the `prompt` and `response` it needs, token for token."""
    stored = jsonl(needs(ANCHOR, "anchor.jsonl"))
    rows = new.run_tces.training_rows(stored)
    assert all(r["prompt"] == s["prompt_token_ids"] and r["response"] == s["generation"]["completion_token_ids"]
               for r, s in zip(rows, stored))


def test_8_fn_datums_equal_instruction_datums(family, robots, instruction):
    fn = robots[640:640 + 4480]
    diff = first_diff([r["id"] for r in fn], instruction["fn"][1])
    assert diff is None, f"inputs/no_robots.jsonl[640:] ids {diff}"
    same_datums(family(QWEN).datums([(r["prompt"], r["completion"]) for r in fn], 1024), instruction["fn"][0], "FN")


def test_9_completion_equals_decision_probe_completion(ds, family):
    """Teacher samples (clone corpus), M0's anchor samples and math samples ending on EOS or a user turn."""
    groups = sorted(needs(CORPUS, "the clone corpus").glob("group-*/samples.json"))
    sequences = [r["generation"]["completion_token_ids"] for g in groups for r in json.loads(g.read_text())["records"]][::4]
    sequences += [r["generation"]["completion_token_ids"] for r in jsonl(needs(ANCHOR, "anchor.jsonl"))][::4]
    for name, when in (("M0", "u0"), ("D-u140+b", "u20"), ("D-u140+it", "u20")):
        for mode in ("forced", "unforced"):
            path = needs(MATH / "pilot-20260924/eval/math500" / name / f"{when}-{mode}.jsonl", f"{name} {mode}")
            sequences += [r["token_ids"] for r in jsonl(path)]
    f, tok = family(QWEN), ds.ctx.local.tokenizer
    for k, ids in enumerate(sequences):
        mine, theirs = f.completion(ids), ds.sampling._completion(tok, tuple(ids))
        assert mine == theirs, f"sequence {k}: text differs from character {first_diff(mine, theirs)}"
    assert len(sequences) > 5000


def test_9_template_stop_equals_decision_probe(ds, new):
    texts = ["<answer>EXPRESSION</answer>", "<answer>EXPRESSION</answer>\".", "<answer>3+4</answer>", "x </answer>",
             "<answer>E</answer> and more", "<answer></answer>", "<answer>(a+b)</answer>)", "no tags"]
    for folder in ("decision-probe", "repetition-probe", "anchor"):
        for path in sorted((DURASEED / "runs" / folder).glob("**/*.jsonl")):
            if path.stem in ("think", "competence", "unforced", "decision"):
                texts += [json.loads(line)["text"] for line in sample_rows(open(path).readlines(), 20)]
    for text in texts:
        assert new.tces.template_stop(text) == ds.run_decision_probe.template_stop(text), repr(text[-60:])
    assert len(texts) > 1000


def test_10_score_texts_equal_depth_full(ds):
    """run_tces.py score reads data/tces/common_texts.jsonl: equal to run_depth_full's texts (4,096) and to
    the tokens stored with depth-full-20260924's M0 scores."""
    import numpy as np
    stored = needs(DURASEED / "runs/depth-profile/depth-full-20260924/M0/common.npz", "depth-full M0 scores")
    dp = importlib.import_module("run_depth_profile")
    ds.ctx.repo = DURASEED
    ids = dp.chosen(ds.ctx)
    theirs = dp.texts(ds.ctx, dp.COMMON["targeted"], ids["targeted"], 4096)
    theirs += dp.texts(ds.ctx, dp.COMMON["sentinel"], ids["sentinel"], 4096)
    mine = jsonl(REPO / "data/tces/common_texts.jsonl")
    assert [r["task_id"] for r in mine] == [r["task_id"] for r in theirs], "common_texts order"
    z = np.load(stored)
    assert [str(t) for t in z["task_ids"]] == [r["task_id"] for r in mine], "depth-full M0 order"
    for k, (x, y) in enumerate(zip(mine, theirs)):
        assert x["prompt"] == y["prompt"], f"common text {k}: prompt"
        diff = first_diff(x["response"], y["response"])
        assert diff is None, f"common text {k}: response {diff}"
        diff = first_diff(x["response"], [t for t in z["tokens"][k].tolist() if t >= 0])
        assert diff is None, f"common text {k}: response against the stored scores {diff}"


def test_11_math_inputs_equal_the_runs_inputs(ds):
    """The rebuilt inputs/math files equal what the runs read: pool order and answers decide D and O."""
    data = needs(DURASEED / "runs/math/data", "DuraSeed-v1 math data")
    for name, key in (("pool", "task_id"), ("eval_math500", "unique_id")):
        mine, theirs = jsonl(needs(REPO / f"inputs/math/{name}.jsonl", name)), jsonl(data / f"{name}.jsonl")
        fields = sorted(set(mine[0]) & set(theirs[0]))
        assert key in fields and len(mine) == len(theirs), (name, len(mine), len(theirs))
        bad = next((i for i, (x, y) in enumerate(zip(mine, theirs)) if any(x[f] != y[f] for f in fields)), None)
        assert bad is None, f"{name}: row {bad} differs in {[f for f in fields if mine[bad][f] != theirs[bad][f]]}"
    aime = importlib.import_module("run_math_pilot").aime_items()
    mine = jsonl(REPO / "inputs/math/aime.jsonl")
    assert [(r["unique_id"], r["problem"], int(r["answer"])) for r in mine] == \
        [(r["unique_id"], r["problem"], r["answer"]) for r in aime], "AIME items"


# ---- run_teacher.py and run_robustness.py ------------------------------------------------------------------

TEACHER_RUN, ROBUST_RUN = MATH / "teacher-20260924", MATH / "robustness-20260924"


def load(name, path):
    """A module from its file under a name of its own: DuraSeed-v1's tools/ has a run_teacher.py and a
    run_robustness.py too, and its modules put tools/ first on sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_rows(mine, theirs, what):
    assert len(mine) == len(theirs), f"{what}: {len(mine)} rows, DuraSeed-v1 {len(theirs)}"
    for k, (x, y) in enumerate(zip(mine, theirs)):
        assert x["task_id"] == y["task_id"], f"{what} row {k}: task_id"
        for name in ("prompt_token_ids", "completion_token_ids"):
            u, v = (x[name], y[name]) if name == "prompt_token_ids" else (x["generation"][name], y["generation"][name])
            diff = first_diff(u, v)
            assert diff is None, f"{what} row {k} ({x['task_id']}): {name} {diff}"


def logged_tokens(lengths, folder, what):
    """Every logged update's training tokens equal its batch's: 32 datums taken in order and cycled."""
    logs = [p for p in needs(folder, what).glob("u*.json") if p.stem[1:].isdigit()]
    assert logs, f"{what}: no training logs"
    for path in logs:
        update = int(path.stem[1:])
        batch = sum(lengths[((update - 1) * 32 + j) % len(lengths)] for j in range(32))
        assert batch == json.loads(path.read_text())["train_tokens"], f"{what} update {update}: training tokens"


@pytest.fixture(scope="module")
def ports(ds):
    return SimpleNamespace(teacher=load("new_run_teacher", SRC / "run_teacher.py"),
                           robustness=load("new_run_robustness", SRC / "run_robustness.py"),
                           ds_teacher=load("ds_run_teacher", DURASEED / "tools/run_teacher.py"),
                           ds_robustness=load("ds_run_robustness", DURASEED / "tools/run_robustness.py"))


@pytest.fixture(scope="module")
def math_ctx(ds, tmp_path_factory):
    """A DuraSeed-v1 Context set up for the 9B, as run_teacher and run_robustness set up theirs."""
    pilot = importlib.import_module("run_math_pilot")
    ctx = ds.run_decision_probe.Context(DURASEED, tmp_path_factory.mktemp("math-ctx"), cap=0.0, prior=0.0)
    ctx.repo, ctx.sdk = DURASEED, ds.sdk
    importlib.import_module("math_models").configure(ctx, pilot.BASE_MODEL, pilot.STOPS)
    return ctx


@pytest.fixture(scope="module")
def teacher(ports, maths, math_ctx):
    """Both codebases' student arms on the stored teacher traces: (rows, held, dropped) each."""
    have = {r["task_id"]: r for r in jsonl(needs(TEACHER_RUN / "teacher/traces.jsonl", "the teacher traces"))}
    sets = json.loads((MATH / "pilot-20260924/arms.json").read_text())
    pool = {p["task_id"]: p for p in jsonl(MATH / "data/pool.jsonl")}
    return SimpleNamespace(have=have, sets=sets, mine=ports.teacher.arms(maths(QWEN), sets, have),
                           theirs=ports.ds_teacher.arms(math_ctx, pool, have))


def test_12_teacher_arms_equal_run_teacher(teacher):
    (rows, held, dropped), (their_rows, their_held, their_dropped) = teacher.mine, teacher.theirs
    stored = json.loads(needs(TEACHER_RUN / "arms.json", "the teacher run's arms.json").read_text())
    assert dropped == their_dropped == stored["dropped"], "dropped counts"
    for arm in ("D-T", "O-T", "P-T"):
        diff = first_diff([r["task_id"] for r in rows[arm]], stored[arm])
        assert diff is None, f"{arm}: task order against the run's arms.json {diff}"
        same_rows(rows[arm], their_rows[arm], arm)
    same_rows(held, their_held, "held-out traces")


def test_12_teacher_datums_equal_run_teacher(ds, new, teacher):
    """D-T's datums; O-T's and P-T's rows are equal token for token and go through the same raw_datums (test 4)."""
    same_datums(new.io.raw_datums(pairs_of(teacher.mine[0]["D-T"]), new.run_math.MAX_LENGTH),
                ds.training.raw_datums(SimpleNamespace(sdk=ds.sdk), teacher.theirs[0]["D-T"], max_length=17000), "D-T")


@pytest.mark.parametrize("arm", ["D-T", "O-T", "P-T"])
def test_12_teacher_batches_equal_the_logged_training_tokens(teacher, arm):
    rows = teacher.mine[0][arm]
    logged_tokens([len(r["prompt_token_ids"]) + len(r["generation"]["completion_token_ids"]) - 1 for r in rows],
                  TEACHER_RUN / "train" / arm, arm)


def test_12_heldout_texts_equal_the_scored_ones(ports, teacher):
    import numpy as np
    stored = np.load(needs(TEACHER_RUN / "heldout/M0/traces.npz", "the scored held-out traces"))
    texts = ports.teacher.held_texts(teacher.mine[1])
    assert [t["task_id"] for t in texts] == [str(t) for t in stored["task_ids"]], "held-out task order"
    for k, t in enumerate(texts):
        diff = first_diff(t["response"], [x for x in stored["tokens"][k].tolist() if x >= 0])
        assert diff is None, f"held-out text {k}: response against the stored scores {diff}"


def test_12_teacher_prompts_and_problems_follow_run_teacher(ports, teacher):
    pilot = importlib.import_module("run_math_pilot")
    _, renderer = ports.ds_teacher.teacher_format()
    pool = {p["task_id"]: p for p in jsonl(REPO / "inputs/math/pool.jsonl")}
    problems, stored = ports.teacher.teacher_problems(pool, teacher.sets), list(teacher.have.values())
    diff = first_diff([(p["task_id"], wanted) for p, wanted, _ in problems], [(r["task_id"], r["wanted"]) for r in stored])
    assert diff is None, f"teacher problems against the stored traces (in the order written) {diff}"
    for p, _, _ in problems[::40]:
        theirs = renderer.build_generation_prompt([{"role": "user", "content": pilot.INSTR + p["problem"]}]).to_ints()
        assert ports.teacher.teacher_prompt(renderer, p) == list(theirs), f"{p['task_id']}: teacher prompt"


def test_12_stored_traces_meet_the_keep_rule(ports, teacher):
    extract, theirs = importlib.import_module("common.maths").extract, importlib.import_module("math_score").extract
    pool = {p["task_id"]: p for p in jsonl(REPO / "inputs/math/pool.jsonl")}
    for problem, wanted, most in ports.teacher.teacher_problems(pool, teacher.sets):
        r = teacher.have[problem["task_id"]]
        assert len(r["kept"]) <= wanted and r["draws"] <= most, f"{r['task_id']}: traces or draws"
        for t in r["kept"]:
            assert extract(t["final"]) == theirs(t["final"]) == problem["answer"], f"{r['task_id']}: answer"
            assert t["teacher_tokens"] < 16384 and t["draw"] < r["draws"], f"{r['task_id']}: length or draw"


@pytest.mark.parametrize("label", ["D-T-u140", "O-T-u140", "P-T-u140"])
def test_12_probe_replays_the_stage_and_scores_the_same_texts(maths, family, label):
    """--phase probe: its two updates are the instruction-tuning stage's own first batches (tokens exact,
    losses within 0.1% of the stage's logs), and it scores the stored own texts and the pilot's base texts."""
    import numpy as np
    folder = needs(TEACHER_RUN / "r1" / label, f"the replayed probe of {label}")
    logged_tokens([int(d.model_input.length) for d in family(QWEN).later_stages()["it"]], folder / "train", label)
    for update in (1, 2):
        replay, stage = (json.loads((path / f"u{update}.json").read_text())["metrics"]["loss:sum"]
                         for path in (folder / "train", TEACHER_RUN / "later" / f"{label}+it"))
        assert math.isclose(replay, stage, rel_tol=1e-3), f"{label} update {update}: loss {replay}, the stage's {stage}"
    own, common = maths(QWEN), maths(QWEN)
    own.root, common.root = TEACHER_RUN, MATH / "pilot-20260924"  # stored texts: nothing is sampled or written
    for kind, texts in (("own", asyncio.run(own.depth_texts(label, None))),
                        ("common", asyncio.run(common.depth_texts("M0", None)))):
        stored = np.load(folder / "u2" / f"{kind}.npz")
        assert [t["task_id"] for t in texts] == [str(t) for t in stored["task_ids"]], f"{label} {kind}: order"
        for k, t in enumerate(texts):
            diff = first_diff(t["response"], [x for x in stored["tokens"][k].tolist() if x >= 0])
            assert diff is None, f"{label} {kind} text {k}: response against the probe's scores {diff}"


def test_12_parse_equals_run_teacher(ports):
    a, f = "<|channel|>analysis<|message|>", "<|channel|>final<|message|>"
    replies = [f"{a} think <|end|><|start|>assistant{f} \\boxed{{5}} <|return|>", f"{a}x<|end|>", f"{f}x<|return|>",
               f"{a}x<|end|>{a}y<|end|>{f}z<|return|>", f"{a}x <|call|> y<|end|>{f}z<|return|>",
               f"{a}x<|end|><|start|>assistant<|channel|>commentary to=python<|message|>1+1<|call|>{f}z<|return|>",
               f"{a}x<|end|>{f}z", f"{a}{f}", ""]
    for text in replies:
        assert ports.teacher.parse(text) == ports.ds_teacher.parse(text), text


def test_13_r2_examples_equal_run_robustness(ds, ports, family, instruction):
    """R2 and R3 train on the 640 instruction-tuning examples, then the first 1,280 of the FN stream."""
    config = needs(DURASEED / "runs/repetition-probe/rep-20260923-instruct/config.json", "the D3 config")
    it = instruction["d3"][0]
    fn = ds.run_instruction_continuation.instruction_datums(
        ds.ctx, examples=ports.ds_robustness.GENTLE * 32 - len(it), namespace="anchor-20260924-fn",
        exclude=json.loads(config.read_text())["examples"])[0]
    same_datums(ports.robustness.instruction(family(QWEN), ports.robustness.LONG * 32), it + fn, "1,920 examples")


@pytest.mark.parametrize("part, label, count", [("r1", label, 640) for label in ("M0", "D-u140", "O-u140", "P-u140")]
                         + [(part, label, 1920) for part in ("r2", "r3") for label in ("D-u140", "O-u140")])
def test_13_robustness_batches_equal_the_logged_training_tokens(ports, family, part, label, count):
    datums = ports.robustness.instruction(family(QWEN), count)
    logged_tokens([int(d.model_input.length) for d in datums], ROBUST_RUN / part / label / "train", f"{part} {label}")


def test_13_states_are_run_robustness_states(ports):
    runs = {"pilot": MATH / "pilot-20260924", "main": MATH / "main-9b-20260924", "teacher": TEACHER_RUN}
    for label, (run, path) in ports.ds_robustness.ARMS.items():  # R1-R3's states and R4-R5's teacher states
        theirs = None if path is None else json.loads((DURASEED / run / path).read_text())["state_path"]
        assert ports.robustness.state(runs, label) == theirs, f"{label}: handoff state"
    for label, (run, path) in ports.ds_robustness.STATES.items():
        later = json.loads((DURASEED / run / "later" / f"{label}+it" / "state.json").read_text())["state_path"]
        assert ports.robustness.state(runs, label, later=True) == later, f"{label}: state after instruction tuning"


@pytest.mark.parametrize("label", ["M0", "D-u140", "O-u140", "P-u140"])
def test_13_r1_texts_equal_run_robustness_and_the_scored_texts(ports, maths, math_ctx, label):
    import numpy as np
    items = [it for it in jsonl(MATH / "data/eval_math500.jsonl") if it["level"] >= 3]
    theirs, _ = ports.ds_robustness.own_texts(math_ctx, label, items)
    runs = {"pilot": MATH / "pilot-20260924", "main": MATH / "main-9b-20260924"}
    mine = asyncio.run(ports.robustness.own_texts(maths(QWEN), runs, label))
    stored = np.load(needs(ROBUST_RUN / "r1" / label / "u1" / "own.npz", f"R1's scores of {label}"))
    assert [t["task_id"] for t in mine] == [t["task_id"] for t in theirs] == [str(t) for t in stored["task_ids"]]
    for k, (x, y) in enumerate(zip(mine, theirs)):
        for name in ("prompt", "response"):
            diff = first_diff(x[name], y[name])
            assert diff is None, f"{label} text {k}: {name} {diff}"
        diff = first_diff(x["response"], [t for t in stored["tokens"][k].tolist() if t >= 0])
        assert diff is None, f"{label} text {k}: response against R1's stored scores {diff}"


def test_13_r3_scores_the_pilots_base_texts(ports, maths):
    import numpy as np
    runs = {"pilot": MATH / "pilot-20260924", "main": MATH / "main-9b-20260924"}
    common = asyncio.run(ports.robustness.own_texts(maths(QWEN), runs, "M0"))
    for label in ("D-u140", "O-u140", "P-u140"):
        for update in (2, 20):
            stored = np.load(needs(ROBUST_RUN / "r3" / label / f"u{update}" / "common.npz", f"R3's scores of {label}"))
            assert [t["task_id"] for t in common] == [str(t) for t in stored["task_ids"]], f"{label} u{update}: order"
            for k, t in enumerate(common):
                diff = first_diff(t["response"], [x for x in stored["tokens"][k].tolist() if x >= 0])
                assert diff is None, f"{label} u{update} text {k}: response {diff}"


FOLLOWUP_RUN = MATH / "robustness-followup-20260924"


@pytest.fixture(scope="module")
def chat(ports, math_ctx):
    """DuraSeed-v1's R5 examples by its SHA rule over the cached mixture: (datums, config)."""
    needs(SHARD, "the cached Tulu-3 SFT mixture")
    return ports.ds_robustness.chat_datums(math_ctx)


@pytest.fixture(scope="module")
def followup_datums(ports, family):
    """The port's datums for R4 (R2's 1,920 examples) and R5 (the chat rows)."""
    needs(REPO / "inputs/chat.jsonl", "inputs/chat.jsonl (make inputs)")
    return {"r4": ports.robustness.instruction(family(QWEN), 1920),
            "r5": ports.robustness.instruction(family(QWEN), 1920, "chat")}


def test_14_chat_ids_follow_the_sha_rule_and_equal_the_run(ports, chat):
    frozen = json.loads(needs(REPO / "data/chat_ids.json", "data/chat_ids.json").read_text())
    ids, (_, config) = [r["id"] for r in frozen["rows"]], chat
    diff = first_diff(ids, config["examples"])
    assert diff is None, f"data/chat_ids.json against the SHA rule of chat_datums {diff}"
    run = json.loads(needs(FOLLOWUP_RUN / "r5/examples.json", "R5's examples.json").read_text())
    assert ids == run["examples"], "data/chat_ids.json against the run's r5/examples.json"
    mix = collections.Counter(r["source"] for r in frozen["rows"])
    assert dict(mix) == config["sources"] == run["sources"], "the source mix"
    rows = jsonl(REPO / "inputs/chat.jsonl")
    flags = [int(bool(ports.ds_robustness.MATHLIKE.search(r["prompt"] + r["completion"]))) for r in rows]
    assert flags == [r["mathlike"] for r in frozen["rows"]], "the keyword flags against DuraSeed-v1's MATHLIKE"
    assert sum(flags) == round(config["mathlike_share"] * len(ids)) == 200, "the keyword share"


def test_14_chat_datums_equal_chat_datums(chat, followup_datums):
    same_datums(followup_datums["r5"], chat[0], "R5's chat datums")


@pytest.mark.parametrize("part, label", [("r4", "D-T-u140"), ("r4", "O-T-u140")]
                         + [("r5", label) for label in ("D-u140", "O-u140", "D-T-u140", "O-T-u140")])
def test_14_followup_batches_equal_the_logged_training_tokens(followup_datums, part, label):
    logged_tokens([int(d.model_input.length) for d in followup_datums[part]], FOLLOWUP_RUN / part / label / "train",
                  f"{part} {label}")
