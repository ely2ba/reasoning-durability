#!/usr/bin/env python3
"""Part SK of records/revision.md: a skill the base lacks within a fixed token budget, in four phases, each
launched separately and reviewed before the next (`--phase`, in this order):

  smoke        the base on the first 5 calibration problems of each setting of the ladder (20 problems):
               prompts, verifiers and response lengths.
  calibration  the base's forced accuracy on the 300 calibration problems of the next setting of the ladder
               (base-7 products of 5 then 6 digits, then 12 then 16 words under a custom alphabet), one new
               setting per launch; the first at or below 25% is chosen.
  gate         O-S on the chosen setting with the pilot's recipe (a fresh rank-32 LoRA, 1e-4, batch 32, 140
               updates: 4,480 problems once), and the base, evaluated on the 300 evaluation problems. The gate:
               O-S-u140 at least 30 points above the base, the paired interval above zero, and at most 15% of
               its responses at the cap.
  rest         if the gate passed: D-S (the first 579 train problems, one solution each, cycled for 7.7
               passes) and P-S (D-S's problems, a different solution at each visit), evaluated at handoff; then
               each arm's later stages from its u140 state (fresh optimizer), evaluated: instruction tuning at
               1e-4 for 60 updates on R2's 1,920 examples (the primary stage, first), then at 3e-4 for 20
               updates on the pilot's 640 No Robots examples.

This runs the final design: every evaluation, the smoke's included, is forced into the reasoning block with
four draws of at most 2,048 tokens (the record's amendment: the skill is solving within 2,048 tokens, about
twice the mean 5-digit solution and above the longest), temperature 1, top-p 0.95, scored by the task's exact
verifier. The paper's smoke ran before that amendment, at 4,096 and then 8,192 tokens (its results are
runs/skill/skill-20260926/smoke-cap4096/ and smoke-cap8192/); the smoke here writes smoke-cap2048/.

The problems come from src/common/skill's seeded generator (data/skill_ids.json lists them). A training text is
"\\n<think>\\n", the solver's reasoning, "\\n</think>\\n\\n", its boxed answer and end of text, in the base's
tokens. Every step skips work whose output exists, so a relaunch resumes. `--limit` is the run's cumulative
list-price limit: the ledger carries the spend of earlier phases, and stops the run before the limit.
"""

import argparse
import asyncio
import hashlib
import json

import numpy as np

from analyze_skill import change, per_problem
from common.models import Family, jsonl
from common.skill import base7, splits, words
from common.tinker_io import Session, stamp, write
from run_math import LARGE, LATER_LR, LATER_UPDATES, REPO, SMALL, TTL_DAYS, UPDATES, Math, guarded
from run_robustness import BASE, GENTLE_LR, LONG, instruction

LADDER = ((base7, 5), (base7, 6), (words, 12), (words, 16))  # the record's order
CAP, DRAWS, CEILING, GAIN, CAPPED, SMOKE = 2048, 4, 0.25, 30.0, 0.15, 5
STAGES = {"it": (LATER_LR, LATER_UPDATES), "it-lr1e-4": (GENTLE_LR, LONG)}


async def evaluate(m, sampler, label, task, problems, folder):
    """Forced responses to `problems`, scored by the task's verifier; the rows (read back when they exist)."""
    out = folder / f"{label}.jsonl"
    if not out.exists():
        drawn = await asyncio.gather(*(m.s.sample(sampler, m.f.render(p.prompt) + m.f.think, DRAWS, CAP, m.f.stops)
                                       for p in problems))
        write(out, [dict(task_id=p.task_id, draw=k, tokens=len(t), capped=len(t) >= CAP, boxed="\\boxed" in text,
                         correct=task.verify(text, p), text=text, token_ids=t)
                    for p, sequences in zip(problems, drawn) for k, t in enumerate(sequences)
                    for text in [m.f.tok.decode(t)]])
    rows = jsonl(out)
    print(f"{stamp()} {label}: correct={np.mean([r['correct'] for r in rows]):.3f} "
          f"capped={np.mean([r['capped'] for r in rows]):.3f} tokens={np.mean([r['tokens'] for r in rows]):.0f} "
          f"spent=${m.s.ledger.total():.2f}", flush=True)
    return rows


async def calibrate(m, base):
    """The ladder in the record's order, one new setting per launch so that each is reviewed before the next is
    paid for, until the base's forced accuracy on one is at most CEILING; recorded in calibration.json."""
    record = {"cap_tokens": CAP, "draws": DRAWS, "ceiling": CEILING, "accuracy": {}, "chosen": None}
    for task, size in LADDER:
        name = f"{task.NAME}-{size}"
        new = not (m.root / "calibration" / f"{name}.jsonl").exists()
        rows = await evaluate(m, base, name, task, splits(task, size)["calibration"], m.root / "calibration")
        record["accuracy"][name] = float(np.mean([r["correct"] for r in rows]))
        record["chosen"] = name if record["accuracy"][name] <= CEILING else None
        if new or record["chosen"]:
            break
    write(m.root / "calibration.json", record)
    status = (f"chosen: {record['chosen']}" if record["chosen"] else "none yet; relaunch for the next setting"
              if len(record["accuracy"]) < len(LADDER) else "no setting qualifies; the part stops")
    print(f"calibration {record['accuracy']}: {status}", flush=True)


def arm_rows(f, task, train):
    """The arms' rows as the pilot's D and O and the main phase's P: D-S is the first 579 train problems and O-S
    all 4,480, each arm in its own hashed order; P-S's slot s visits D-S's problem s mod 579 with its solution
    s // 579, so its first visit is D-S's text and no text repeats. A problem's first solution is its D-S and
    O-S text."""
    def order(arm):
        return lambda p: hashlib.sha256(f"skill-20260926:{arm}:{p.task_id}".encode()).hexdigest()

    def row(problem, text):
        assert task.verify(text, problem), problem.task_id  # only verified texts are trained on
        body = f.tok.encode(text, add_special_tokens=False)
        return dict(task_id=problem.task_id, prompt_token_ids=f.render(problem.prompt),
                    generation={"completion_token_ids": f.think + body + [f.tok.eos_token_id]})

    texts = {p.task_id: task.solutions(p) for p in train[:LARGE]}
    d, o = sorted(train[:SMALL], key=order("D-S")), sorted(train[:LARGE], key=order("O-S"))
    return {"D-S": [row(p, texts[p.task_id][0]) for p in d],
            "O-S": [row(p, texts[p.task_id][0]) for p in o],
            "P-S": [row(d[s % len(d)], texts[d[s % len(d)].task_id][s // len(d)]) for s in range(UPDATES * 32)]}


async def handoff(m, arm, path, task, problems):
    if not (m.root / "eval" / f"{arm}-u140.jsonl").exists():
        await evaluate(m, await m.s.sampler(path), f"{arm}-u140", task, problems, m.root / "eval")


def gate(root):
    """O-S-u140 at least GAIN points above the base (paired item bootstrap, 20,000 resamples) with the interval
    above zero, and at most CAPPED of its responses at the cap; recorded in gate.json."""
    trained = root / "eval" / "O-S-u140.jsonl"
    gain = change(per_problem(trained), per_problem(root / "eval" / "M0.jsonl"))
    capped = float(np.mean([r["capped"] for r in jsonl(trained)]))
    passed = gain["points"] >= GAIN and gain["interval"][0] > 0 and capped <= CAPPED
    write(root / "gate.json", {"gain": gain, "capped_share": capped, "passed": passed,
                               "rule": f"gain >= {GAIN} points with its interval above 0, capped share <= {CAPPED}"})
    print(f"gate {'passed' if passed else 'FAILED; the part stops'}: O-S minus base {gain['points']:.1f} points "
          f"{gain['interval']}, capped {capped:.3f}", flush=True)


async def stage(m, arm, path, name, datums, task, problems):
    """A later stage from the arm's u140 state (fresh optimizer), saved so that a relaunch evaluates the same
    realization, then evaluated."""
    label, (lr, updates) = f"{arm}+{name}", STAGES[name]
    if (m.root / "eval" / f"{label}.jsonl").exists():
        return
    saved = m.root / "later" / label / "state.json"
    if saved.exists():
        sampler = await m.s.sampler(json.loads(saved.read_text())["state_path"])
    else:
        model = await m.s.service.create_training_client_from_state_async(path)
        async for _ in m.s.fit(model, datums, lr, updates, lambda u: m.root / "later" / label / f"u{u}.json"):
            pass
        write(saved, {"state_path": await m.s.save(model, f"{m.root.name}-{arm}-{name}", TTL_DAYS)})
        sampler = await model.save_weights_and_get_sampling_client_async()
    await evaluate(m, sampler, label, task, problems, m.root / "eval")


async def execute(args):
    root = REPO / "runs/skill" / args.run_id
    session = Session(root, BASE, args.limit)
    m = Math(session, Family(BASE), root)  # for its training (Math.train); the evaluations are the task's own
    base = await session.service.create_sampling_client_async(base_model=BASE)
    if args.phase == "smoke":
        for task, size in LADDER:
            await evaluate(m, base, f"{task.NAME}-{size}", task, splits(task, size)["calibration"][:SMOKE],
                           root / f"smoke-cap{CAP}")
        return
    if args.phase == "calibration":
        await calibrate(m, base)
        return
    chosen = json.loads((root / "calibration.json").read_text())["chosen"]
    if chosen is None:
        print("no setting is chosen; see calibration.json", flush=True)
        return
    task, size = next((task, size) for task, size in LADDER if f"{task.NAME}-{size}" == chosen)
    parts = splits(task, size)
    problems, rows = parts["evaluation"], arm_rows(m.f, task, parts["train"])
    write(root / "arms.json", {"setting": chosen, "evaluation": [p.task_id for p in problems],
                               **{arm: [r["task_id"] for r in rs] for arm, rs in rows.items()}})
    states = {arm: root / "train" / arm / f"u{UPDATES}-state.json" for arm in rows}  # as Math.train saves them
    if args.phase == "gate":
        await asyncio.gather(guarded(root, "eval-M0", evaluate(m, base, "M0", task, problems, root / "eval")),
                             guarded(root, "train-O-S", m.train("O-S", rows["O-S"])))
        if not states["O-S"].exists() or not (root / "eval" / "M0.jsonl").exists():
            print("O-S's training or the base's evaluation failed (see errors/); a relaunch resumes", flush=True)
            return
        await handoff(m, "O-S", json.loads(states["O-S"].read_text())["state_path"], task, problems)
        gate(root)
        return
    if not json.loads((root / "gate.json").read_text())["passed"]:
        print("the gate failed; the part stops", flush=True)
        return
    distinct = instruction(m.f, LONG * 32)  # R2's 1,920 examples; the first 640 are the default stage's

    async def primary(arm):  # D-S and P-S are trained here (O-S's state exists), evaluated, then the primary stage
        path = (await m.train(arm, rows[arm]))[UPDATES]
        await handoff(m, arm, path, task, problems)
        await stage(m, arm, path, "it-lr1e-4", distinct, task, problems)

    await asyncio.gather(*(guarded(root, f"{arm}+it-lr1e-4", primary(arm)) for arm in rows))
    await asyncio.gather(*(guarded(root, f"{arm}+it", stage(m, arm, json.loads(state.read_text())["state_path"], "it",
                                                             distinct[:640], task, problems))
                           for arm, state in states.items() if state.exists()))
    print(f"done; list-price spend ${session.ledger.total():.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="skill-20260926", help="under runs/skill")
    parser.add_argument("--phase", choices=("smoke", "calibration", "gate", "rest"), required=True,
                        help="in this order, each reviewed before the next")
    parser.add_argument("--limit", type=float, required=True, help="the run's cumulative list-price limit in USD")
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
