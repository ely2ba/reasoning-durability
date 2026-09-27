#!/usr/bin/env python3
"""Revision parts RL, RP and RS (records/revision.md), which continue the paper's states on the 9B.

  RL  relearning. From the after-stage states D+IT, D+AO, D-T+IT, D-T+AO and O+IT (the ceiling control), 20
      updates on the base's own correct forced solutions to 640 pool problems: the pool after the pilot's 7,680
      screened problems, in pool order, less the 52 an arm of the paper trained on (data/revision_ids.json),
      under the pilot's screen and keep rule in chunks of 512; evaluated after updates 1, 2, 5 and 20. From D+IT
      and D+AO only, the habit control: 20 updates on the base's forced responses (top-p 1) to R5's chat
      prompts, one per prompt in R5's order until 640 are kept (the reasoning block closes, an answer follows,
      the response is uncapped and fits the training length); evaluated after update 20.
  RP  replay. R2's instruction tuning (60 updates) in which each batch of 32 holds 30 of R2's examples in order
      and the arm's next 2 own training texts, in its training order, cycled; from D-u140, O-u140, D-T-u140 and
      O-T-u140, evaluated after update 60.
  RS  a realistic stage: one pass over the 8,544 No Robots rows of inputs/no_robots_epoch.jsonl (the paper's
      SHA-256 selection continued; 267 updates) from the base (a fresh LoRA), D-u140, O-u140 and P-u140,
      evaluated after the last update.

Every stage runs at 1e-4 with batch 32 and a fresh optimizer, continuing its origin's adapter; every
evaluation is forced (MATH-500 levels 3-5, 4 draws, 16,384 tokens). Every step skips work whose output exists,
so a relaunch resumes.
"""

import argparse
import asyncio
import json

from common.models import Family, jsonl
from common.tinker_io import Session, raw_datums, stamp, write
from run_math import CAP, MAX_LENGTH, REPO, TTL_DAYS, Math, guarded, pilot_sets
from run_robustness import BASE, LONG, ORIGINS, instruction, state
from run_teacher import arms as teacher_arms

LR, SCREENED, RELEARN, UPDATES = 1e-4, 7680, 640, 20
AFTER, HABIT = ("D-u140+it", "D-u140+b", "D-T-u140+it", "D-T-u140+b", "O-u140+it"), ("D-u140+it", "D-u140+b")
SAVES = {"relearn": (1, 2, 5, 20), "habit": (20,)}
INSTRUCTION, REPLAYED = 30, 2  # RP's batch of 32
RS_UPDATES, RS_ARMS = 267, ("M0", "D-u140", "O-u140", "P-u140")  # 8,545 rows qualify; one is left over


def texts(rows):
    return [dict(prompt=r["prompt_token_ids"], response=r["generation"]["completion_token_ids"]) for r in rows]


async def stage(m, label, path, datums, lr, updates, saves, tag):
    """`updates` updates of 32 of `datums`, taken in order, from the state at `path` (a fresh LoRA when None), at
    `lr` with a fresh optimizer. The states after the updates in `saves` are saved and then evaluated forced; a
    relaunch trains again only if one of them is missing."""
    name, folder = f"{label}+{tag}", m.root / "stage" / f"{label}+{tag}"
    evals = {k: m.root / "eval" / "math500" / name / f"u{k}-forced.jsonl" for k in saves}
    if all(p.exists() for p in evals.values()):
        return
    states = {k: folder / f"u{k}-state.json" for k in saves}
    if not all(p.exists() for p in states.values()):
        model = await (m.s.fresh_lora() if path is None else m.s.service.create_training_client_from_state_async(path))
        async for update in m.s.fit(model, datums, lr, updates, lambda u: folder / f"u{u}.json"):
            if update in states:
                saved = await m.s.save(model, f"{m.root.name}-{name.replace('+', '-')}-u{update}", TTL_DAYS)
                write(states[update], {"state_path": saved})
    jobs = [m.evaluate(await m.s.sampler(json.loads(states[k].read_text())["state_path"]), name, f"u{k}",
                       modes=("forced",)) for k in saves if not evals[k].exists()]
    await asyncio.gather(*jobs)
    print(f"{stamp()} {name} done", flush=True)


async def relearn_rows(m, base):
    """RL's math data: the first 640 kept by the pilot's screen and keep rule (Math.rft) over the pool after the
    pilot's 7,680 screened problems, in pool order, less the problems an arm of the paper trained on."""
    skipped = set(json.loads((REPO / "data/revision_ids.json").read_text())["rl_relearn"]["skipped"])
    fresh = [p for p in m.pool[SCREENED:] if p["task_id"] not in skipped]
    return [r for r in await m.rft(base, fresh, RELEARN) if r["correct"]][:RELEARN]


def habit_kept(f, prompt, tokens):
    """The reasoning block closes and a final answer follows it (an end of text or a new user turn aside); the
    response also ends within the cap and fits the pilot's training length."""
    _, closed, answer = f.tok.decode(tokens).rpartition("</think>")
    for end in ("<|endoftext|>", *f.stops):
        answer = answer.removesuffix(end)
    return (bool(closed) and bool(answer.strip()) and len(tokens) < CAP
            and len(prompt) + len(f.think) + len(tokens) - 1 <= MAX_LENGTH)


async def habit_rows(m, base):
    """RL's habit data: the base's forced response (top-p 1) to R5's chat prompts, one per prompt in R5's order,
    until 640 are kept; saved after every round, and a relaunch continues."""
    out, chat = m.root / "habit.jsonl", jsonl(REPO / "inputs/chat.jsonl")
    rows = jsonl(out) if out.exists() else []
    while sum(r["kept"] for r in rows) < RELEARN and len(rows) < len(chat):
        chunk = chat[len(rows):len(rows) + RELEARN - sum(r["kept"] for r in rows)]
        prompts = [m.f.render(r["prompt"]) for r in chunk]
        drawn = await asyncio.gather(*(m.draw(base, p + m.f.think, 1, top_p=1.0) for p in prompts))
        rows += [dict(id=r["id"], kept=habit_kept(m.f, p, t), tokens=len(t), prompt_token_ids=p,
                      generation={"completion_token_ids": m.f.think + t}) for r, p, [t] in zip(chunk, prompts, drawn)]
        write(out, rows)
        print(f"{stamp()} habit {len(rows)} prompts, {sum(r['kept'] for r in rows)} kept, "
              f"spent=${m.s.ledger.total():.2f}", flush=True)
    return [r for r in rows if r["kept"]]


def own_rows(m, runs):
    """RP's arms' own training rows in training order, rebuilt as trained and checked against the runs' arms.json."""
    pilot, teacher = runs["pilot"], runs["teacher"]
    have = {r["task_id"]: r for r in jsonl(teacher / "teacher/traces.jsonl")}
    rows = pilot_sets(pilot) | teacher_arms(m, json.loads((pilot / "arms.json").read_text()), have)[0]
    for arm in ("D-T", "O-T"):
        assert [r["task_id"] for r in rows[arm]] == json.loads((teacher / "arms.json").read_text())[arm], arm
    return {f"{arm}-u140": rows[arm] for arm in ("D", "O", "D-T", "O-T")}


def replayed(gentle, rows):
    """RP's 60 batches: 30 of R2's examples in order, then the arm's next 2 own texts, cycled. The own datums are
    built over the whole training set, so they keep the normalization they were trained with."""
    own = raw_datums(texts(rows), MAX_LENGTH)
    return [d for u in range(LONG) for d in gentle[u * INSTRUCTION:(u + 1) * INSTRUCTION]
            + [own[(u * REPLAYED + j) % len(own)] for j in range(REPLAYED)]]


async def execute(args):
    root = REPO / "runs/math" / (args.run_id or f"revision-{args.part.lower()}-20260926")
    runs = {name: REPO / "runs/math" / run for name, run in
            (("pilot", args.pilot_run), ("main", args.main_run), ("teacher", args.teacher_run))}
    session = Session(root, BASE, args.limit)
    m = Math(session, Family(BASE), root)
    if args.part == "RL":
        base = await session.service.create_sampling_client_async(base_model=BASE)
        math, habit = await asyncio.gather(relearn_rows(m, base), habit_rows(m, base))
        jobs = {}
        for tag, rows, origins in (("relearn", math, AFTER), ("habit", habit, HABIT)):
            datums = raw_datums(texts(rows), MAX_LENGTH)
            for origin in origins:  # the state after the origin's own later stage, in its run
                later = runs[ORIGINS[origin.split("+")[0]]] / "later" / origin / "state.json"
                jobs[f"{origin}+{tag}"] = stage(m, origin, json.loads(later.read_text())["state_path"], datums, LR,
                                                UPDATES, SAVES[tag], tag)
    elif args.part == "RP":
        gentle, rows = instruction(m.f, LONG * 32), own_rows(m, runs)
        write(root / "replayed.json", {label: [rs[i % len(rs)]["task_id"] for i in range(LONG * REPLAYED)]
                                       for label, rs in rows.items()})
        jobs = {label: stage(m, label, state(runs, label), replayed(gentle, rs), LR, LONG, (LONG,), "it-replay")
                for label, rs in rows.items()}
    else:
        datums = instruction(m.f, RS_UPDATES * 32, "no_robots_epoch")
        jobs = {label: stage(m, label, state(runs, label), datums, LR, RS_UPDATES, (RS_UPDATES,), "nr-epoch")
                for label in RS_ARMS}
    await asyncio.gather(*(guarded(root, label, job) for label, job in jobs.items()))
    print(f"done; list-price spend ${session.ledger.total():.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("RL", "RP", "RS"), required=True)
    parser.add_argument("--run-id", help="under runs/math; by default the paper's, revision-<part>-20260926")
    parser.add_argument("--pilot-run", default="pilot-20260924")
    parser.add_argument("--main-run", default="main-9b-20260924")
    parser.add_argument("--teacher-run", default="teacher-20260924")
    parser.add_argument("--limit", type=float, required=True, help="this part's list-price limit in USD")
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
