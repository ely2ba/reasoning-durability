"""Build outputs/ from a DuraSeed-v1 runs tree: compact per-sample tables with no text.

    python src/collect.py --runs /path/to/DuraSeed-v1/runs

Every paper number is computed from these tables; outputs/README.md documents each column. Runs
still in progress are read as they stand (a line cut short by a running writer is skipped), and
re-running rewrites every table. The script also copies the files of the original study's results
package that the paper reads (replay-v1/) from the DuraSeed-v1 checkout that holds the runs tree.
The records (records/) are frozen in this repository.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common import divergence, maths, tces
from common.models import NEMOTRON, Family

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
MATH_RUNS = {"pilot-20260924": "qwen", "main-9b-20260924": "qwen", "main-35b-20260924": "qwen",
             "main-nemotron-20260924": "nemotron", "teacher-20260924": "qwen", "robustness-20260924": "qwen",
             "robustness-followup-20260924": "qwen", "revision-rp-20260926": "qwen", "revision-rs-20260926": "qwen",
             "revision-rl-20260926": "qwen", "revision-sh-20260926": "qwen", "revision-sd-20260926/s2": "qwen",
             "revision-sd-20260926/s3": "qwen"}  # revision-*: records/revision.md
ON_PILOT_BASE = ("main-9b-20260924", "teacher-20260924", "revision-sh-20260926")  # scored on the pilot's base texts
BODY = {"qwen": 3, "nemotron": 2}
PROFILE_RUN, PROFILE_BINS = "depth-20260923", (4, 8, 16, 32, 64, 128, 256, 512, 1025)  # 1-indexed positions
PROFILE_COUNTS = tuple(f"n_{lo}_{hi - 1}" for lo, hi in zip(PROFILE_BINS, PROFILE_BINS[1:]))
CAPS, BUDGETS = {"math500": 16384, "aime": 32768}, (2048, 4096, 8192, 16384)
TCES_KINDS, DECISION_TOKENS = ("decision", "think", "competence", "unforced"), 8
REPLAY = ("followup/README.md", "followup/readout.md", "followup/readout.json", "followup/profiles/seed-11-R-P.json",
          "followup/profiles/seed-11-R-S.json", "order-seed47-20260909/readout.md",
          "order-seed47-20260909/readout.json", "order-seed47-20260909/R-P-profile.json",
          "order-seed47-20260909/R-S-profile.json")  # the six files the paper reads and their three notes
INTEGER = re.compile(r"-?\d{1,3}(?:,\d{3})+|-?\d+")
LAST_INTEGER = re.compile(r"-?\d+(?:,\d{3})*")  # the revision's lenient score (DuraSeed-v1 analyze_revision.py)
SKILL_SETS = ("smoke-cap4096", "smoke-cap8192", "smoke-cap2048", "calibration", "eval")  # 2,048: a src/run_skill.py smoke


def rows(path):
    """The JSON lines of a file, skipping any line cut short by a writer still running."""
    out = []
    for line in path.open():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def flag(value):
    return None if value is None else int(bool(value))


def dirs(root):
    return sorted(p for p in root.glob("*") if p.is_dir()) if root.is_dir() else []


def write(records, name, integers=()):
    """Gzip CSV with 0/1 flags and counts as integers, missing values empty; replaced atomically."""
    df = pd.DataFrame(records) if isinstance(records, list) else records
    for column in integers:
        if column in df:
            df[column] = df[column].astype("Int64")
    tmp = OUT / f".{name}.tmp"
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(OUT / name)
    return name, len(df)


def tces_samples(runs):
    """One row per probe sample. The cap is the run's forced (and unforced) budget; decision
    probes sample 8 tokens. Both are checked on the data: a sample stopped by length is at the cap."""
    tasks = {p["task_id"]: p["task"] for p in tces.panel(REPO)}
    roots = [d for d in dirs(runs / "decision-probe") if d.name != "dc-20260923"]  # dc-20260923 was aborted
    roots += dirs(runs / "repetition-probe") + [runs / "anchor" / "anchor-20260924"]
    out = []
    for run in roots:
        cap = 16384 if run.name.startswith(("cap-", "cap16k-", "anchor-")) else 4096
        for path in sorted(run.glob("*/u*/*.jsonl")):
            if path.stem not in TCES_KINDS:
                continue
            kind, budget = path.stem, DECISION_TOKENS if path.stem == "decision" else cap
            for r in rows(path):
                assert r["tokens"] <= budget and (r["stop"] != "length" or r["tokens"] == budget), (path, r["tokens"])
                task = None if kind == "decision" else tasks.get(r["task_id"])  # E1-panel items are not in the panel
                out.append(dict(run=run.name, label=path.parents[1].name, update=int(path.parent.name[1:]), kind=kind,
                                cap=cap, task_id=r["task_id"], role=r["role"], draw=r["draw"], tokens=r["tokens"],
                                stop=r["stop"], skip=flag(r.get("skip")), correct=flag(r.get("correct")),
                                strict_correct=flag(r.get("strict_correct")), resumed=r.get("resumed"),
                                found=None if task is None else int(tces.found(r["text"], task))))
    return out


def lenient(r, text, answer):
    """Correct, or no box at all and the last integer of the text (thousands commas removed) is the answer."""
    if r["correct"]:
        return 1
    last = (INTEGER.findall(text) or [""])[-1].replace(",", "")
    return int("\\boxed" not in text and 0 < len(last) < 40 and int(last) == answer)  # a looping text can hold a huge one


def last_integer(text):
    """The text's last integer (thousands commas allowed, `{,}` read as one), or None when there is none or it
    runs to 100 characters (a looping text): the revision's lenient score."""
    found = LAST_INTEGER.findall(text.replace("{,}", ",").replace("−", "-"))
    return int(found[-1].replace(",", "")) if found and len(found[-1]) < 100 else None


def math_row(r, family, answer, cap):
    ids, text = r["token_ids"], r["text"]
    end = ("cap" if r["capped"] else "eos" if ids and ids[-1] in family.ends
           else "user" if text.endswith("\nUser:") else "other")
    within = {f"within_{k}": (int(maths.extract(text if len(ids) <= k else family.tok.decode(ids[:k])) == answer)
                              if k < cap else None) for k in BUDGETS}
    tail = ids[-2000:]
    looping = len({tuple(tail[i:i + 16]) for i in range(len(tail) - 15)}) < 0.5 * (len(tail) - 15)
    return dict(task_id=r["task_id"], level=r["level"], draw=r["draw"], tokens=r["tokens"], capped=int(r["capped"]),
                end=end, boxed=int(r["boxed"]), correct=int(r["correct"]),
                tagged=None if r.get("tagged") is None else str(r["tagged"]),
                think=flag(r.get("think")), closed_think=int("</think>" in text), **within,
                lenient_correct=lenient(r, text, answer), last_integer_correct=int(last_integer(text) == answer),
                looping=int(looping) if r["capped"] else None,
                continuations=r.get("continuations"), first_correct=flag(r.get("first_correct")))


def math_samples(runs, families):
    items = [json.loads((REPO / "data/math" / name).read_text())["items"] for name in ("eval_math500_ids.json", "aime_ids.json")]
    answers = {it["unique_id"]: it["answer"] for it in items[0]} | {it["id"]: it["answer"] for it in items[1]}
    out = []
    for run, family in MATH_RUNS.items():
        root = runs / "math" / run
        files = [(bench, "first", p) for bench in CAPS for p in sorted((root / "eval" / bench).glob("*/*.jsonl"))]
        files += [("math500", rule, p) for rule in ("budget", "answer")
                  for p in sorted((root / "reforce" / rule).glob("*/*-forced.jsonl"))]
        for bench, rule, path in files:
            if ".partial" in path.name:  # a probe still running; only finished files are collected
                continue
            when, mode = path.stem.split("-", 1)
            head = dict(run=run, bench=bench, label=path.parent.name, when=when, mode=mode, rule=rule)
            out += [head | math_row(r, families[family], answers[r["task_id"]], CAPS[bench]) for r in rows(path)]
    return out


def skill_samples(runs):
    """One row per sample of the revision's skill part SK: the base on each setting of the ladder in the smoke
    (4,096 and 8,192 tokens; a re-run's at 2,048) and in the calibration, and every state on the chosen setting
    (eval/). The cap is checked on the data: a sample is capped exactly when it reaches the cap."""
    out = []
    for run in dirs(runs / "skill"):
        known = run / "calibration.json"  # written after the smoke, by the calibration
        calibration = json.loads(known.read_text()) if known.exists() else {}
        for group in SKILL_SETS:
            cap = int(group.removeprefix("smoke-cap")) if group.startswith("smoke") else calibration["cap_tokens"]
            for path in sorted((run / group).glob("*.jsonl")):
                setting, label = (calibration["chosen"], path.stem) if group == "eval" else (path.stem, "M0")
                for r in rows(path):
                    assert r["tokens"] <= cap and r["capped"] == (r["tokens"] >= cap), (path, r["tokens"])
                    out.append(dict(run=run.name, set=group.split("-")[0], setting=setting, label=label, cap=cap,
                                    task_id=r["task_id"], draw=r["draw"], tokens=r["tokens"], capped=int(r["capped"]),
                                    boxed=int(r["boxed"]), correct=int(r["correct"])))
    return out


def depth_pairs(runs):
    """(run, set, base label, base scores, state label, state scores, body start) for every scored pair."""
    dp, anchor, pairs = runs / "depth-profile", runs / "anchor" / "anchor-20260924", []
    for run in ("depth-20260923", "depth-full-20260924"):
        for d in dirs(dp / run):
            pairs.append((run, "common", "M0", dp / run / "M0/common.npz", d.name, d / "common.npz", 3))
            pairs.append((run, "own", "M0", dp / run / f"M0/own-{d.name}.npz", d.name, d / "own.npz", 3))
    m0_common = dp / "depth-20260923/M0/common.npz"
    pairs += [("anchor-20260924", "common", "M0", m0_common, label, anchor / label / "common.npz", 3)
              for label in ("FA1-u140", "FA025-u140", "FN-u140", "U2-u140")]
    pairs += [("anchor-20260924", "drilled", "M0", anchor / "M0/drilled.npz", label, anchor / label / "drilled.npz", 3)
              for label in ("FA1-u140", "FA025-u140", "FN-u140", "F-u20", "F-u140")]
    for run, family in MATH_RUNS.items():
        root, body = runs / "math" / run, BODY[family]
        common = runs / "math/pilot-20260924/depth/M0/common.npz" if run in ON_PILOT_BASE else root / "depth/M0/common.npz"
        for d in dirs(root / "depth"):
            pairs.append((run, "common", "M0", common, d.name, d / "common.npz", body))
            pairs.append((run, "own", "M0", root / f"depth/M0/own-{d.name}.npz", d.name, d / "own.npz", body))
        pairs += [(run, "drilled", "M0", root / "drilled/M0/drilled.npz", d.name, d / "drilled.npz", body)
                  for d in dirs(root / "drilled") if d.name != "M0"]
        pairs += [(run, "heldout", "M0", root / "heldout/M0/traces.npz", d.name, d / "traces.npz", body)
                  for d in dirs(root / "heldout") if d.name != "M0"]
    pilot, main = runs / "math/pilot-20260924/depth", runs / "math/main-9b-20260924/depth"
    handoff = {"M0": pilot / "M0/common.npz", "D-u140": pilot / "D-u140/own.npz", "O-u140": pilot / "O-u140/own.npz",
               "P-u140": main / "P-u140/own.npz"}  # robustness R1: each state's own texts, at handoff and after k
    for label, path in handoff.items():
        for d in dirs(runs / "math/robustness-20260924/r1" / label):
            pairs.append(("robustness-20260924", "displacement", label, path, f"{label}@{d.name}", d / "own.npz", 3))
    teacher = runs / "math/teacher-20260924"  # prediction 5: the teacher states after two updates
    for d in dirs(teacher / "r1"):
        for kind, name in (("own", "displacement"), ("common", "displacement_common")):
            pairs.append(("teacher-20260924", name, d.name, teacher / "depth" / d.name / f"{kind}.npz",
                          f"{d.name}@u2", d / "u2" / f"{kind}.npz", 3))
    for label in ("D-u140", "O-u140", "P-u140"):  # R3b: the same measure on the base's texts
        path = (main if label == "P-u140" else pilot) / label / "common.npz"
        for d in dirs(runs / "math/robustness-20260924/r3" / label):
            pairs.append(("robustness-20260924", "displacement_common", label, path, f"{label}@{d.name}",
                          d / "common.npz", 3))
    return [p for p in pairs if p[3].exists() and p[5].exists()]


def profile(base, state):
    """Per-text sums of d and of scored positions in each bin [lo, hi) of 1-indexed response positions,
    as in DuraSeed-v1's analyze_depth_profile (the first bin starts after the three prefix tokens)."""
    scored = ~np.isnan(base["lp"])
    position = np.arange(1, scored.shape[1] + 1)
    d, out = np.where(scored, base["lp"] - state["lp"], 0.0), {}
    for lo, hi in zip(PROFILE_BINS, PROFILE_BINS[1:]):
        window = scored & (position >= lo) & (position < hi)
        out[f"d_{lo}_{hi - 1}"], out[f"n_{lo}_{hi - 1}"] = np.where(window, d, 0.0).sum(1), window.sum(1)
    return out


def prefix(base, state, body):
    """Per-text sums over the forced prefix, the scored positions before the body. The registered fit checks
    on the drilled texts (main-phase.md N2, anchor.md) average the NLL over every scored token, prefix included."""
    before = ~np.isnan(base["lp"]) & (np.arange(base["lp"].shape[1]) < body)
    return {"n_prefix": before.sum(1), "base_lp_prefix": np.where(before, base["lp"], 0.0).sum(1),
            "d_prefix": np.where(before, base["lp"] - state["lp"], 0.0).sum(1)}


def depth_texts(runs):
    frames = []
    for run, name, base_label, base_path, state_label, state_path, body in depth_pairs(runs):
        base, state = divergence.load(base_path), divergence.load(state_path)
        assert (base["task_ids"] == state["task_ids"]).all(), (base_path, state_path)
        agg = divergence.aggregate(base, state, body)  # asserts that both files score the same tokens
        if run == PROFILE_RUN and name == "common":
            agg |= profile(base, state)
        if name == "drilled":
            agg |= prefix(base, state, body)
        frame = pd.DataFrame({key: np.asarray(value) for key, value in agg.items()})
        frame.insert(0, "task_id", base["task_ids"])
        for i, (key, value) in enumerate([("run", run), ("length", base["tokens"].shape[1]), ("set", name),
                                          ("base", base_label), ("state", state_label)]):
            frame.insert(i, key, value)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def train_log(runs):
    """Per-update training records: acquisition (stage_a/, train/) and later stages (u*/train.json, later/)."""
    patterns = (("*/stage_a/u*.json", "acquisition", 1), ("*/u*/train.json", "later", 1),
                ("train/*/u*.json", "acquisition", 0), ("later/*/u*.json", "later", 0),
                *((f"{part}/*/train/u*.json", part, 1) for part in ("r1", "r2", "r3", "r4", "r5")))
    out = []
    for family in ("decision-probe", "repetition-probe", "anchor", "math"):
        names = sorted(MATH_RUNS) if family == "math" else [d.name for d in dirs(runs / family)]
        for name in names:  # math: the collected runs only, not a revision part still running
            for pattern, stage, depth in patterns:
                for path in sorted((runs / family / name).glob(pattern)):
                    if "state" in path.name:
                        continue
                    r = json.loads(path.read_text())
                    out.append(dict(run=name, label=path.parents[depth].name, stage=stage, update=r["update"],
                                    train_tokens=r["train_tokens"], loss_sum=r["metrics"].get("loss:sum"),
                                    batch=r.get("batch")))
    return out


def copy_replay(source):
    """The files of the original study's results package that the paper reads, with the Tinker session
    identifiers removed from the seed-47 readout (replay-v1/README.md)."""
    for name in REPLAY:
        (REPO / "replay-v1" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "artifacts/replay-v1" / name, REPO / "replay-v1" / name)
    readout = REPO / "replay-v1/order-seed47-20260909/readout.json"
    readout.write_text(re.sub(r'"session_ids":\[[^\]]*\],', "", readout.read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, required=True, help="DuraSeed-v1's runs/ directory")
    runs = parser.parse_args().runs.resolve()
    OUT.mkdir(exist_ok=True)
    copy_replay(runs.parent)
    families = {"qwen": Family("Qwen/Qwen3.5-9B-Base")}  # tokenizers from the local cache (HF_HUB_OFFLINE=1)
    if (runs / "math/main-nemotron-20260924").exists():
        families["nemotron"] = Family(NEMOTRON)
    flags = ("skip", "correct", "strict_correct", "resumed", "found", "capped", "boxed", "think",
             "closed_think", "within_2048", "within_4096", "within_8192", "within_16384", "lenient_correct",
             "last_integer_correct", "looping", "continuations", "first_correct", "easy", "batch")
    rft = [dict(run=path.parent.name, task_id=r["task_id"], easy=int(r["easy"]), tokens=r.get("tokens"),
                capped=flag(r.get("capped")), correct=int(r["correct"]))
           for path in sorted((runs / "math").glob("*/rft.jsonl")) for r in rows(path)]
    p_path = runs / "math/main-9b-20260924/p_samples.jsonl"
    p_samples = [dict(task_id=r["task_id"], distinct=len({tuple(s) for s in r["samples"]}), draws=r["draws"])
                 for r in (rows(p_path) if p_path.exists() else [])]
    results = [write(tces_samples(runs), "tces_samples.csv.gz", flags),
               write(math_samples(runs, families), "math_samples.csv.gz", flags + ("tokens", "draw")),
               write(depth_texts(runs), "depth_texts.csv.gz", PROFILE_COUNTS + ("n_prefix",)),
               write(train_log(runs), "train_log.csv.gz", flags),
               write(rft, "math_rft.csv.gz", flags + ("tokens",)),
               write(p_samples, "math_p_samples.csv.gz"),
               write(skill_samples(runs), "skill_samples.csv.gz")]
    for name, n in results:
        print(f"{name}: {n:,} rows, {(OUT / name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
