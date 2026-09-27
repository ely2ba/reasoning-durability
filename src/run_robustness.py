#!/usr/bin/env python3
"""Robustness of the break on the 9B (records/robustness.md), from the pilot's and main run's states.

  R1  timing: instruction tuning (the pilot's 640 examples at 3e-4) for 5 updates from the base (a
      fresh LoRA), D-u140, O-u140 and P-u140, states saved after updates 1, 2 and 5. Each state's
      own handoff texts are scored after 1, 2, 5 and 20 updates (displacement), and D-u140 and
      O-u140 are evaluated forced after 2 and 5.
  R2  a smaller step: instruction tuning at 1e-4 for 60 updates on 1,920 distinct examples (the 640,
      then the first 1,280 of the FN stream) from D-u140 and O-u140, forced after 20 and 60.
  R3  longer: the same 1,920 examples at 3e-4 for 60 updates from D-u140 and O-u140 (forced after 40
      and 60), and the base's texts scored after updates 2 and 20 for D-u140, O-u140 and P-u140.
  R4  R2 from the teacher run's D-T-u140 and O-T-u140 (records/teacher.md).
  R5  a broad later stage: R2's schedule on 1,920 other examples, single-turn rows from the Tulu-3 SFT
      mixture's non-math sources (data/chat_ids.json), from D-u140, O-u140, D-T-u140 and O-T-u140,
      forced after 60.
Every step skips work whose output exists, so a relaunch resumes.
"""

import argparse
import asyncio
import json

from common.models import Family, jsonl
from common.tinker_io import Session, stamp, write
from run_math import DEPTH_LENGTH, LATER_LR, REPO, Math, guarded, train_state

BASE = "Qwen/Qwen3.5-9B-Base"
ORIGINS = {"M0": "pilot", "D-u140": "pilot", "O-u140": "pilot", "P-u140": "main",  # the run that holds each state
           "D-T-u140": "teacher", "O-T-u140": "teacher"}
EARLY, EVALUATED_EARLY, LONG, GENTLE_LR = (1, 2, 5), (2, 5), 60, 1e-4
GENTLE_SAVES, LONG_SAVES = (20, 60), (40, 60)


def instruction(family, count, name="no_robots"):
    """Datums of the first `count` rows of inputs/<name>.jsonl, in order: the pilot's 640 No Robots
    examples, then the FN stream; or R5's chat rows."""
    rows = jsonl(REPO / "inputs" / f"{name}.jsonl")[:count]
    return family.datums([(r["prompt"], r["completion"]) for r in rows], 1024)


def state(runs, label, later=False):
    """The state's path in its run, at handoff or after its 20 updates of instruction tuning."""
    root = runs[ORIGINS[label]]
    if later:
        return json.loads((root / "later" / f"{label}+it" / "state.json").read_text())["state_path"]
    return train_state(root, label)


async def start(m, runs, label):
    """A training client at the state's handoff: a fresh LoRA for the base."""
    path = state(runs, label)
    return await (m.s.fresh_lora() if path is None else m.s.service.create_training_client_from_state_async(path))


async def own_texts(m, runs, label):
    """The state's own forced handoff texts, as stored (the base's are the pilot's common texts)."""
    return await Math(m.s, m.f, runs[ORIGINS[label]]).depth_texts(label, None)


def forced(m, name, update):
    return m.root / "eval" / "math500" / name / f"u{update}-forced.jsonl"


async def r1(m, runs, label, datums):
    texts, folder, evaluated = await own_texts(m, runs, label), m.root / "r1" / label, label in ("D-u140", "O-u140")
    done = all((folder / f"u{k}" / "own.npz").exists() for k in EARLY) and (
        not evaluated or all(forced(m, f"{label}+it@{k}", k).exists() for k in EVALUATED_EARLY))
    jobs = []
    if not done:
        model = await start(m, runs, label)
        async for update in m.s.fit(model, datums, LATER_LR, max(EARLY), lambda u: folder / "train" / f"u{u}.json"):
            if update in EARLY:
                write(folder / f"u{update}" / "state.json",
                      {"state_path": await m.s.save(model, f"robustness-{label}-it-u{update}")})
                sampler = await model.save_weights_and_get_sampling_client_async()
                jobs.append(m.s.score_set(sampler, texts, folder / f"u{update}" / "own.npz", DEPTH_LENGTH))
                if evaluated and update in EVALUATED_EARLY:
                    jobs.append(m.evaluate(sampler, f"{label}+it@{update}", f"u{update}", modes=("forced",)))
    after = await m.s.sampler(state(runs, label, later=True))  # the pilot's (or main run's) 20-update state
    await asyncio.gather(*jobs, m.s.score_set(after, texts, folder / "u20" / "own.npz", DEPTH_LENGTH))
    print(f"{stamp()} r1 {label} done", flush=True)


async def r2(m, runs, label, datums, part="r2", tag="it-lr1e-4", saves=GENTLE_SAVES):
    """R2's schedule; R4 reuses it unchanged from the teacher states, R5 with other data."""
    folder = m.root / part / label
    if all(forced(m, f"{label}+{tag}@{k}", k).exists() for k in saves):
        return
    model, jobs = await start(m, runs, label), []
    async for update in m.s.fit(model, datums, GENTLE_LR, LONG, lambda u: folder / "train" / f"u{u}.json"):
        if update in saves:
            write(folder / f"u{update}" / "state.json",
                  {"state_path": await m.s.save(model, f"robustness-{label}-{tag}-u{update}")})
            sampler = await model.save_weights_and_get_sampling_client_async()
            jobs.append(m.evaluate(sampler, f"{label}+{tag}@{update}", f"u{update}", modes=("forced",)))
    await asyncio.gather(*jobs)
    print(f"{stamp()} {part} {label} done", flush=True)


async def r3(m, runs, label, datums, common):
    jobs = []
    for update in (2, 20):  # R3b: the base's texts under R1's 2-update state and the 20-update state
        out = m.root / "r3" / label / f"u{update}" / "common.npz"
        if not out.exists():
            path = (json.loads((m.root / "r1" / label / "u2" / "state.json").read_text())["state_path"]
                    if update == 2 else state(runs, label, later=True))
            jobs.append(m.s.score_set(await m.s.sampler(path), common, out, DEPTH_LENGTH))
    if label != "P-u140" and not all(forced(m, f"{label}+it-long@{k}", k).exists() for k in LONG_SAVES):
        model = await start(m, runs, label)
        async for update in m.s.fit(model, datums, LATER_LR, LONG, lambda u: m.root / "r3" / label / "train" / f"u{u}.json"):
            if update in LONG_SAVES:
                sampler = await model.save_weights_and_get_sampling_client_async()
                jobs.append(m.evaluate(sampler, f"{label}+it-long@{update}", f"u{update}", modes=("forced",)))
    await asyncio.gather(*jobs)
    print(f"{stamp()} r3 {label} done", flush=True)


async def execute(args):
    root = REPO / "runs/math" / args.run_id
    runs = {name: REPO / "runs/math" / run for name, run in
            (("pilot", args.pilot_run), ("main", args.main_run), ("teacher", args.teacher_run))}
    session = Session(root, BASE, args.limit)
    m = Math(session, Family(BASE), root)
    distinct = instruction(m.f, LONG * 32)  # the first 640 are the instruction-tuning stage itself
    jobs = []
    if "R1" in args.parts:
        jobs += [guarded(root, f"r1-{label}", r1(m, runs, label, distinct[:640])) for label in ORIGINS]
    if "R2" in args.parts:
        jobs += [guarded(root, f"r2-{label}", r2(m, runs, label, distinct)) for label in ("D-u140", "O-u140")]
    if "R3" in args.parts:
        common = await own_texts(m, runs, "M0")
        jobs += [guarded(root, f"r3-{label}", r3(m, runs, label, distinct, common))
                 for label in ("D-u140", "O-u140", "P-u140")]
    if "R4" in args.parts:
        jobs += [guarded(root, f"r4-{label}", r2(m, runs, label, distinct, part="r4")) for label in ("D-T-u140", "O-T-u140")]
    if "R5" in args.parts:
        chat = instruction(m.f, LONG * 32, "chat")
        jobs += [guarded(root, f"r5-{label}", r2(m, runs, label, chat, part="r5", tag="chat-lr1e-4", saves=(LONG,)))
                 for label in ("D-u140", "O-u140", "D-T-u140", "O-T-u140")]
    await asyncio.gather(*jobs)
    print(f"done; list-price spend ${session.ledger.total():.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="robustness-20260924")
    parser.add_argument("--parts", nargs="+", choices=("R1", "R2", "R3", "R4", "R5"), default=("R1", "R2"))
    parser.add_argument("--pilot-run", default="pilot-20260924")
    parser.add_argument("--main-run", default="main-9b-20260924")
    parser.add_argument("--teacher-run", default="teacher-20260924")
    parser.add_argument("--limit", type=float, required=True, help="this run's list-price limit in USD")
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
