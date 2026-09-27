"""Offline equivalence of the revision runners (records/revision.md) with the runs that DuraSeed-v1's
tools/run_revision.py, run_revision_arms.py and run_skill.py produced. Every selection is rebuilt from local files
and compared with what the runs wrote, and every training batch with the run's logged training tokens (32 datums
taken in order and cycled; test_runners.py checks the datums themselves). Where a runner resumes from a stored
file (rft.jsonl, sharp-t05.jsonl, p_samples.jsonl), it runs here with no Tinker session, so it can sample nothing.

RL  the screen (the pool after 7,680, less data/revision_ids.json's skipped problems, which are the paper's
    arms'), its keep rule and the 640 kept; the habit rows' prompts, rounds and keep rule; every stage's batches
RP  the own rows (against the runs' arms.json) and replayed.json; every update's 30 + 2 against the logs
RS  inputs/no_robots_epoch.jsonl against instruction_datums (8,545 qualify) and examples.json; the logs
SH  the sharp rows against data/revision_ids.json and the keep rule; the logs; the gate's B, to the last digit
SD  each seed's arms against its arms.json and data/revision_ids.json (outside D and D2; the seeds share 91); the
    P texts; the logs
SK  the arms and D-S/O-S/P-S texts (DuraSeed-v1's solver) against arms.json and the logs; the stored samples
    re-scored; gate.json and summary.json, to the last digit
and data/checkpoints.json's revision entries against the runs' state files.
"""

import asyncio
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from conftest import DURASEED, HF_CACHE, REPO, SRC, needs
from test_runners import first_diff, jsonl, logged_tokens, same_datums

pytestmark = pytest.mark.duraseed  # every test here needs the original DuraSeed-v1 runs

QWEN, MATH = "Qwen/Qwen3.5-9B-Base", DURASEED / "runs/math"
RL, RP, RS, SH = (MATH / f"revision-{part}-20260926" for part in ("rl", "rp", "rs", "sh"))
SD, SK, PILOT = MATH / "revision-sd-20260926", DURASEED / "runs/skill/skill-20260926", MATH / "pilot-20260924"
PAPER = ("pilot-20260924", "main-35b-20260924", "main-nemotron-20260924", "teacher-20260924")


def frozen():
    return json.loads((REPO / "data/revision_ids.json").read_text())


def lengths(rows):
    return [len(r["prompt_token_ids"]) + len(r["generation"]["completion_token_ids"]) - 1 for r in rows]


@pytest.fixture(scope="module")
def new():
    """This repository's runners, imported by name with src/ first: DuraSeed-v1's tools/ has modules of the same
    names, and test_runners.py may have put tools/ first on sys.path."""
    for path in (PILOT / "rft.jsonl", RL, RP, RS, SH, SD, SK):
        needs(path, "the run")
    needs(HF_CACHE / "models--Qwen--Qwen3.5-9B-Base", "the Qwen3.5-9B-Base tokenizer")
    sys.path.insert(0, str(SRC))
    try:
        mods = {name: importlib.import_module(name) for name in ("run_revision", "run_revision_arms", "run_skill",
                                                                 "analyze_skill", "run_math", "run_robustness")}
        family = importlib.import_module("common.models").Family(QWEN)
    except ImportError as error:
        pytest.skip(f"this repository's runners cannot be imported here: {error}")
    return SimpleNamespace(f=family, **mods)


@pytest.fixture(scope="module")
def math(new, tmp_path_factory):
    """A Math object with no Tinker session, and the pilot's D and O."""
    needs(REPO / "inputs/math/pool.jsonl", "inputs/ (make inputs)")
    return lambda root: new.run_math.Math(None, new.f, root or tmp_path_factory.mktemp("math"))


@pytest.fixture(scope="module")
def sets(new):
    return new.run_math.pilot_sets(PILOT)  # asserts D and O against the pilot's arms.json


# ---- RL ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def relearn(new, math):
    return asyncio.run(new.run_revision.relearn_rows(math(RL), None))  # every chunk is stored: nothing is sampled


def test_rl_screen_follows_the_rule_and_keeps_the_frozen_640(new, math, relearn):
    m, skipped = math(None), frozen()["rl_relearn"]["skipped"]
    trained = {t for run in PAPER for arm in json.loads((MATH / run / "arms.json").read_text()).values()
               if isinstance(arm, list) for t in arm}
    assert skipped == [p["task_id"] for p in m.pool[7680:] if p["task_id"] in trained], "the paper's arms' problems"
    screen = [p for p in m.pool[7680:] if p["task_id"] not in set(skipped)]
    stored = jsonl(RL / "rft.jsonl")
    assert len(stored) % 512 == 0 and [r["task_id"] for r in stored] == [p["task_id"] for p in screen[:len(stored)]]
    problems, cap = {p["task_id"]: p for p in m.pool}, new.run_math.CAP
    extract = importlib.import_module("common.maths").extract
    for r in (r for r in stored if not r["easy"]):
        p, tokens = problems[r["task_id"]], r["generation"]["completion_token_ids"][len(new.f.think):]
        assert r["prompt_token_ids"] == m.prompt(p["problem"]), r["task_id"]
        assert r["correct"] == (len(tokens) < cap and extract(new.f.tok.decode(tokens)) == p["answer"]), r["task_id"]
    assert [r["task_id"] for r in relearn] == frozen()["rl_relearn"]["task_ids"]


def test_rl_habit_rows_follow_r5_the_rounds_and_the_keep_rule(new):
    rows, chat = jsonl(RL / "habit.jsonl"), jsonl(needs(REPO / "inputs/chat.jsonl", "inputs/chat.jsonl"))
    assert [r["id"] for r in rows] == [c["id"] for c in chat[:len(rows)]], "one prompt each, in R5's order"
    screened, kept = 0, 0
    while kept < 640:  # each round asks for as many prompts as are still missing
        size = 640 - kept
        kept, screened = kept + sum(r["kept"] for r in rows[screened:screened + size]), screened + size
    assert screened == len(rows)
    for r, c in zip(rows, chat):
        completion = r["generation"]["completion_token_ids"]
        assert r["prompt_token_ids"] == new.f.render(c["prompt"]) and completion[:3] == new.f.think, r["id"]
        assert new.run_revision.habit_kept(new.f, r["prompt_token_ids"], completion[3:]) == r["kept"], r["id"]
    assert [r["id"] for r in rows if r["kept"]] == frozen()["rl_habit"]["ids"]


@pytest.mark.parametrize("origin", ["D-u140+it", "D-u140+b", "D-T-u140+it", "D-T-u140+b", "O-u140+it"])
def test_rl_batches_equal_the_logged_training_tokens(new, relearn, origin):
    assert origin in new.run_revision.AFTER
    logged_tokens(lengths(relearn), RL / "stage" / f"{origin}+relearn", f"{origin}+relearn")
    if origin in new.run_revision.HABIT:
        habit = [r for r in jsonl(RL / "habit.jsonl") if r["kept"]]
        logged_tokens(lengths(habit), RL / "stage" / f"{origin}+habit", f"{origin}+habit")


# ---- RP and RS --------------------------------------------------------------------------------------------------

def test_rp_replayed_texts_and_batches_equal_the_run(new, math):
    runs = {"pilot": PILOT, "teacher": MATH / "teacher-20260924"}
    rows = new.run_revision.own_rows(math(None), runs)  # asserts each arm against its run's arms.json
    stored = json.loads((RP / "replayed.json").read_text())
    assert {label: [rs[i % len(rs)]["task_id"] for i in range(120)] for label, rs in rows.items()} == stored
    gentle = new.run_robustness.instruction(new.f, 1920)
    for label, rs in rows.items():
        datums = new.run_revision.replayed(gentle, rs)
        instruction = [d for u in range(60) for d in datums[32 * u:32 * u + 30]]
        assert len(datums) == 60 * 32 and all(x is y for x, y in zip(instruction, gentle[:1800], strict=True)), label
        logged_tokens([int(d.model_input.length) for d in datums], RP / "stage" / f"{label}+it-replay", label)


def test_rs_rows_and_datums_equal_instruction_datums(new, tmp_path_factory):
    """inputs/no_robots_epoch.jsonl: the first 8,544 of the 8,545 No Robots rows that D3's rule admits."""
    for path in (DURASEED / "src", DURASEED / "tools"):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    try:
        ic = importlib.import_module("run_instruction_continuation")
        ctx = importlib.import_module("run_decision_probe").Context(DURASEED, tmp_path_factory.mktemp("ctx"), cap=0.0,
                                                                     prior=0.0)
    except (ImportError, OSError) as error:
        pytest.skip(f"DuraSeed-v1 cannot be loaded here: {error}")
    theirs, ids = ic.instruction_datums(ctx, examples=8545)
    with pytest.raises(SystemExit):
        ic.instruction_datums(ctx, examples=8546)
    rows = jsonl(needs(REPO / "inputs/no_robots_epoch.jsonl", "inputs/no_robots_epoch.jsonl (make inputs)"))
    assert [r["id"] for r in rows] == ids[:8544] == json.loads((RS / "examples.json").read_text())["examples"]
    mine = new.run_robustness.instruction(new.f, new.run_revision.RS_UPDATES * 32, "no_robots_epoch")
    same_datums(mine, theirs[:8544], "RS")
    for label in new.run_revision.RS_ARMS:
        logged_tokens([int(d.model_input.length) for d in mine], RS / "stage" / f"{label}+nr-epoch", label)


# ---- SH and SD --------------------------------------------------------------------------------------------------

def test_sh_rows_training_and_gate_equal_the_run(new, math, sets):
    m = math(SH)
    rows = asyncio.run(new.run_revision_arms.sharp_rows(m, None, sets["O"], 0.5))  # all stored: nothing is sampled
    assert [r["task_id"] for r in rows] == frozen()["sh_sharp"]["task_ids"]
    answers, prompts = {p["task_id"]: p["answer"] for p in m.pool}, {r["task_id"]: r["prompt_token_ids"] for r in sets["O"]}
    extract = importlib.import_module("common.maths").extract
    for r in rows:
        tokens = r["generation"]["completion_token_ids"][3:]
        assert r["prompt_token_ids"] == prompts[r["task_id"]] and 1 <= r["draws"] <= 4, r["task_id"]
        assert len(tokens) < 16384 and extract(new.f.tok.decode(tokens)) == answers[r["task_id"]], r["task_id"]
    logged_tokens(lengths(rows), SH / "train" / "O-sharp-t05-u140", "O-sharp")
    gate = json.loads((SH / "gate.json").read_text())["arms"]["O-sharp-t05-u140"]
    b = new.run_revision_arms.b_value(PILOT, SH / "depth" / "O-sharp-t05-u140" / "common.npz")
    assert b == {"B": gate["B"], "interval": gate["interval"]} and gate["texts"] == len(rows)
    for tag, datums in new.f.later_stages().items():
        logged_tokens([int(d.model_input.length) for d in datums], SH / "stage" / f"O-sharp-t05-u140+{tag}", tag)


@pytest.mark.parametrize("seed", ["s2", "s3"])
def test_sd_arms_texts_and_batches_equal_the_run(new, math, sets, seed):
    rows = new.run_revision_arms.seed_rows(sets, seed)
    stored, subset = json.loads((SD / seed / "arms.json").read_text()), frozen()[f"sd_{seed}"]
    assert [r["task_id"] for r in rows["D"]] == subset["task_ids"], f"{seed}: D against data/revision_ids.json"
    assert new.run_revision_arms.LORA_SEEDS[seed] == subset["lora_seed"] and subset["namespace"] == f"revision-sd-{seed}"
    for arm in ("D", "O"):
        diff = first_diff([r["task_id"] for r in rows[arm]], stored[arm])
        assert diff is None, f"{seed} {arm}: task order {diff}"
    outside = {r["task_id"] for r in sets["D"] + new.run_math.d2_rows(sets)}
    assert len(stored["D"]) == 579 and not outside & set(stored["D"]) and sorted(stored["O"]) == sorted(
        r["task_id"] for r in sets["O"])
    p = asyncio.run(new.run_math.p_samples(math(SD / seed), None, rows["D"]))  # all stored: nothing is sampled
    assert [s["generation"]["completion_token_ids"] for s in p[:579]] == [
        r["generation"]["completion_token_ids"] for r in rows["D"]], "P's first visit is D's own text"
    distinct = new.run_robustness.instruction(new.f, 1920)
    for arm, arm_rows in (("D", rows["D"]), ("O", rows["O"]), ("P", p)):
        logged_tokens(lengths(arm_rows), SD / seed / "train" / f"{arm}-u140", f"{seed} {arm}")
        for tag, datums in (("it", distinct[:640]), ("it-lr1e-4", distinct)):
            logged_tokens([int(d.model_input.length) for d in datums], SD / seed / "stage" / f"{arm}-u140+{tag}", tag)


def test_sd_seeds_share_91_problems():
    s2, s3 = (json.loads((needs(SD / seed / "arms.json", "the run")).read_text())["D"] for seed in ("s2", "s3"))
    assert len(set(s2) & set(s3)) == 91


def test_checkpoints_list_every_revision_state_as_its_file():
    """data/checkpoints.json's revision entries are the revision runs' state files, one for one: the label from the
    file's place in its run, and the creation time from the file."""
    from datetime import UTC, datetime

    listed = {s["label"]: s for s in json.loads((REPO / "data/checkpoints.json").read_text())["states"]
              if s["origin_run"].startswith(("runs/math/revision-", "runs/skill/"))}
    found = {}
    for root in (RL, RP, RS, SH, SD / "s2", SD / "s3", SK):
        prefix = "" if root == SK else "math9b-" + (f"{root.name}-" if root.parent == SD else "")
        for path in needs(root, "the run").rglob("*state*.json"):
            where, name = path.parent.parent.name, path.parent.name
            if where == "stage":  # u<k>-state.json after update k of a stage
                label = f"{prefix}{name}-{path.name.removesuffix('-state.json')}"
            elif where == "later":  # SK's later stages: 20 updates at 3e-4, or 60 at 1e-4
                label = f"{name}-u{60 if name.endswith('lr1e-4') else 20}"
            else:  # train/<arm>/state.json at handoff
                label = f"{name}-u140" if root == SK else f"{prefix}{name}"
            found[label] = (f"runs/{root.relative_to(DURASEED / 'runs')}",
                            datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    assert len(found) == 60 and set(found) == set(listed), sorted(set(found) ^ set(listed))
    for label, (run, created) in found.items():
        assert (listed[label]["origin_run"], listed[label]["created_utc"]) == (run, created), label


# ---- SK ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def skill_rows(new):
    base7, splits = (getattr(importlib.import_module("common.skill"), name) for name in ("base7", "splits"))
    return new.run_skill.arm_rows(new.f, base7, splits(base7, 5)["train"])


def test_sk_problems_and_texts_equal_duraseeds(new):
    """src/common/skill against DuraSeed-v1's duraseed.tasks.skill: every setting's splits, and the prompts,
    answers and worked solutions of base7-5's 4,480 train problems (the arms' texts) and of 200 of each other's."""
    if str(DURASEED / "src") not in sys.path:
        sys.path.append(str(DURASEED / "src"))
    mine, theirs = importlib.import_module("common.skill"), importlib.import_module("duraseed.tasks.skill")
    for task, size in new.run_skill.LADDER:
        other = importlib.import_module(f"duraseed.tasks.skill.{task.NAME}")
        ours, original = mine.splits(task, size), theirs.splits(other, size)
        assert {k: [p.task_id for p in v] for k, v in ours.items()} == {k: [p.task_id for p in v] for k, v in
                                                                         original.items()}, (task.NAME, size)
        count = 4480 if (task.NAME, size) == ("base7", 5) else 200
        for p, q in zip(ours["train"][:count], original["train"][:count]):
            assert (p.prompt, p.answer, task.solutions(p)) == (q.prompt, q.answer, other.solutions(q)), p.task_id


def test_sk_arms_and_batches_equal_the_run(new, skill_rows):
    stored = json.loads((SK / "arms.json").read_text())
    distinct = new.run_robustness.instruction(new.f, 1920)
    for arm, rows in skill_rows.items():
        assert [r["task_id"] for r in rows] == stored[arm], arm
        logged_tokens(lengths(rows), SK / "train" / arm, arm)
        for tag, datums in (("it", distinct[:640]), ("it-lr1e-4", distinct)):
            logged_tokens([int(d.model_input.length) for d in datums], SK / "later" / f"{arm}+{tag}", f"{arm}+{tag}")


def test_sk_stored_samples_rescore_and_the_results_reproduce(new):
    skill = importlib.import_module("common.skill")
    problems = {p.task_id: (task, p) for task, size in new.run_skill.LADDER
                for split in skill.splits(task, size).values() for p in split}
    files = [p for group in ("smoke-cap4096", "smoke-cap8192", "calibration", "eval") for p in (SK / group).glob("*.jsonl")]
    assert len(files) == 4 + 4 + 1 + 10
    for path in files:
        for r in jsonl(path):
            task, p = problems[r["task_id"]]
            text = new.f.tok.decode(r["token_ids"])
            assert text == r["text"] and task.verify(text, p) == r["correct"], (path.name, r["task_id"], r["draw"])
    assert new.analyze_skill.summary(SK) == json.loads((SK / "summary.json").read_text())
    gate = json.loads((SK / "gate.json").read_text())
    per_problem, change = new.analyze_skill.per_problem, new.analyze_skill.change
    assert change(per_problem(SK / "eval/O-S-u140.jsonl"), per_problem(SK / "eval/M0.jsonl")) == gate["gain"]
