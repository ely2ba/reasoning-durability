#!/usr/bin/env python3
"""The teacher experiment (records/teacher.md): the pilot's design on another model's reasoning.

  teacher  gpt-oss-120b (high reasoning, top-p 1) writes traces on the pilot's problems: up to 8
           correct ones for each of D's 579 problems in at most 24 draws, and one for each of O's
           other problems in at most 4. A reply is kept if it has exactly one analysis and one final
           channel and no other control text, is under 16,384 tokens, and its final answer's last box
           is the answer. Saved after every 256 problems.
  student  Qwen3.5-9B-Base learns the traces with the pilot's recipe (run_math.Math). A trace becomes
           "\\n<think>\\n" + analysis + "\\n</think>\\n\\n" + final + end of text in the student's tokens,
           and is usable if it fits the pilot's training length.
             D-T  D's problems, one trace each, cycled for 140 updates (7.7 passes); also saved at 20
             O-T  O's problems, one trace each, seen once
             P-T  D's problems, a different usable trace at each visit
           A problem enters D-T and O-T only if its first kept trace is usable and came within the
           teacher's first 4 draws. Each state is evaluated at handoff (forced and free) and forced
           after 20 updates of each later stage, and scored on the pilot's base texts; D-T-u140 and
           O-T-u140 also on AIME and on 200 held-out teacher traces (as is the base). The u140
           states' own texts and the base's are scored again after 2 updates of instruction tuning.
  probe    prediction 5 by replay (as run): the stage's first 2 updates replayed from each u140
           state, then its own handoff texts and the base's texts scored.
Every step skips work whose output exists, so a relaunch resumes.
"""

import argparse
import asyncio
import glob
import json
from pathlib import Path

from common import maths
from common.models import Family, jsonl
from common.tinker_io import Session, stamp, write
from run_math import CAP, DEPTH_LENGTH, LATER_LR, MAX_LENGTH, REPO, UPDATES, Math, guarded, train_state

TEACHER, BASE = "openai/gpt-oss-120b", "Qwen/Qwen3.5-9B-Base"
PER_D, D_DRAWS, O_DRAWS, CHUNK = 8, 24, 4, 256
BRANCHES = {"D-T": (20, 140), "O-T": (140,), "P-T": (140,)}
AIME_LABELS, EARLY_LABELS = ("D-T-u140", "O-T-u140"), ("D-T-u140", "O-T-u140", "P-T-u140")
HELD, HELD_LENGTH = 200, 4096  # held-out teacher traces: did the student learn the teacher?


def teacher_format():
    """gpt-oss-120b's tokenizer (from the local cache) and its high-reasoning harmony renderer."""
    from tinker_cookbook.renderers import get_renderer
    from transformers import AutoTokenizer
    path = glob.glob(str(Path("~/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/*").expanduser()))[0]
    tok = AutoTokenizer.from_pretrained(path)
    return tok, get_renderer("gpt_oss_high_reasoning", tok)


def teacher_prompt(renderer, problem):
    return list(renderer.build_generation_prompt(
        [{"role": "user", "content": maths.INSTRUCTION + problem["problem"]}]).to_ints())


def teacher_problems(pool, sets):
    """(problem, traces wanted, most draws): D's problems, then O's others, in the pilot's orders."""
    inside = set(sets["D"])
    return ([(pool[t], PER_D, D_DRAWS) for t in sets["D"]]
            + [(pool[t], 1, O_DRAWS) for t in sets["O"] if t not in inside])


def parse(text):
    """The analysis and final channels of a harmony reply, or None if it is not a clean two-part reply."""
    if text.count("<|channel|>analysis<|message|>") != 1 or text.count("<|channel|>final<|message|>") != 1:
        return None
    analysis = text.split("<|channel|>analysis<|message|>")[1].split("<|end|>")[0]
    final = text.split("<|channel|>final<|message|>")[1].split("<|return|>")[0]
    if "<|" in analysis or "<|" in final:  # another channel, a tool call or stray control text
        return None
    return analysis.strip(), final.strip()


async def traces(session, root, problems):
    """Teacher traces for (problem, wanted, draws) triples, saved after every chunk of problems."""
    tok, renderer = teacher_format()
    sampler = await session.service.create_sampling_client_async(base_model=TEACHER)
    stops, out = renderer.get_stop_sequences(), root / "teacher" / "traces.jsonl"
    have = {r["task_id"]: r for r in jsonl(out)} if out.exists() else {}

    async def one(problem, wanted, draws):
        prompt, kept, used, lengths = teacher_prompt(renderer, problem), [], 0, []
        while len(kept) < wanted and used < draws:
            n = min(wanted - len(kept), draws - used)
            for i, tokens in enumerate(await session.sample(sampler, prompt, n, CAP, stops, top_p=1.0)):
                lengths.append(len(tokens))
                parts = parse(tok.decode(tokens))
                if parts and len(tokens) < CAP and maths.extract(parts[1]) == problem["answer"] and len(kept) < wanted:
                    kept.append(dict(analysis=parts[0], final=parts[1], teacher_tokens=len(tokens), draw=used + i))
            used += n
        return dict(task_id=problem["task_id"], wanted=wanted, draws=used, lengths=lengths, kept=kept)

    todo = [t for t in problems if t[0]["task_id"] not in have]
    for start in range(0, len(todo), CHUNK):
        for row in await asyncio.gather(*(one(*t) for t in todo[start:start + CHUNK])):
            have[row["task_id"]] = row
        write(out, list(have.values()))
        print(f"{stamp()} teacher {len(have)} problems, {sum(len(r['kept']) for r in have.values())} traces, "
              f"spent=${session.ledger.total():.2f}", flush=True)


def student_rows(m, problems, have, order):
    """The student's training row for each (task_id, trace index) in `order`."""
    rows = []
    for task_id, k in order:
        trace = have[task_id]["kept"][k]
        body = m.f.tok.encode(trace["analysis"] + "\n</think>\n\n" + trace["final"], add_special_tokens=False)
        rows.append(dict(task_id=task_id, prompt_token_ids=m.prompt(problems[task_id]["problem"]),
                         generation={"completion_token_ids": m.f.think + body + [m.f.tok.eos_token_id]}))
    return rows


def arms(m, sets, have):
    """D-T, O-T and P-T rows on the pilot's problems and orders (`sets`: its arms.json), the held-out
    traces (each D-T problem's second usable trace, which only P-T trains on) and what was dropped."""
    problems = {p["task_id"]: p for p in m.pool}

    def usable(task_id):
        kept = have.get(task_id, {}).get("kept", [])
        rows = student_rows(m, problems, have, [(task_id, k) for k in range(len(kept))])
        return [(k, r) for k, r in enumerate(rows)
                if len(r["prompt_token_ids"]) + len(r["generation"]["completion_token_ids"]) <= MAX_LENGTH]

    def eligible(task_id):  # the first kept trace itself is usable and came within 4 draws, for D and O alike
        return bool(use[task_id]) and use[task_id][0][0] == 0 and have[task_id]["kept"][0]["draw"] < O_DRAWS

    def visit(slot):  # P-T's slot s visits D-T's problem s mod |D-T| for the (s // |D-T|)-th time
        usable_traces = use[d[slot % len(d)]]
        return usable_traces[(slot // len(d)) % len(usable_traces)][1]

    use = {t: usable(t) for t in dict.fromkeys(sets["D"] + sets["O"])}
    d, o = [t for t in sets["D"] if eligible(t)], [t for t in sets["O"] if eligible(t)]
    rows = {"D-T": [use[t][0][1] for t in d], "O-T": [use[t][0][1] for t in o],
            "P-T": [visit(s) for s in range(UPDATES * 32)]}
    held = [use[t][1][1] for t in d if len(use[t]) > 1]
    long = sum(len(have.get(t, {}).get("kept", [])) - len(use[t]) for t in use)
    return rows, held, {"D": len(sets["D"]) - len(d), "O": len(sets["O"]) - len(o), "too_long_traces": long}


def held_texts(held):
    texts = [dict(task_id=r["task_id"], prompt=r["prompt_token_ids"], response=r["generation"]["completion_token_ids"])
             for r in held]
    return [t for t in texts if len(t["response"]) <= HELD_LENGTH][:HELD]


async def student(m, pilot):
    s, sets = m.s, json.loads((pilot / "arms.json").read_text())
    have = {r["task_id"]: r for r in jsonl(m.root / "teacher" / "traces.jsonl")}
    assert set(sets["O"]) <= set(have), "the teacher phase has not finished"
    rows, held, dropped = arms(m, sets, have)
    write(m.root / "arms.json", {"dropped": dropped, **{arm: [r["task_id"] for r in rs] for arm, rs in rows.items()}})
    held = held_texts(held)
    common = await Math(s, m.f, pilot).depth_texts("M0", None)  # the pilot's base texts, as stored
    stages = m.f.later_stages()
    base = await s.service.create_sampling_client_async(base_model=BASE)

    def after_update_2(label, own):
        """Prediction 5: the state's own texts and the base's, scored after update 2 of instruction tuning."""
        async def hook(model, update):
            if update != 2:
                return []
            after = await model.save_weights_and_get_sampling_client_async()
            return [s.score_set(after, texts, m.root / "r1" / label / "u2" / f"{kind}.npz", DEPTH_LENGTH)
                    for kind, texts in (("own", own), ("common", common))]
        return hook

    async def arm(name):
        jobs = []
        for update, path in (await m.train(name, rows[name], branches=BRANCHES[name])).items():
            label = f"{name}-u{update}"
            sampler = await s.sampler(path)
            jobs += [m.evaluate(sampler, label, "u0"), m.common(sampler, label, common)]
            if label in AIME_LABELS:
                jobs += [m.evaluate(sampler, label, "u0", "aime", ("forced",)),
                         s.score_set(sampler, held, m.root / "heldout" / label / "traces.npz", HELD_LENGTH)]
            hook = None
            if label in EARLY_LABELS:
                own = await m.depth_texts(label, sampler)
                jobs.append(s.score_set(sampler, own, m.root / "depth" / label / "own.npz", DEPTH_LENGTH))
                hook = after_update_2(label, own)
            jobs += [m.later(label, path, stage, datums, aime=stage == "it" and label in AIME_LABELS,
                             modes=("forced",), hook=hook if stage == "it" else None)
                     for stage, datums in stages.items()]
        await asyncio.gather(*jobs)

    await asyncio.gather(*(guarded(m.root, name, arm(name)) for name in BRANCHES),
                         s.score_set(base, held, m.root / "heldout" / "M0" / "traces.npz", HELD_LENGTH))


async def probe(m, pilot):
    """The student phase's scoring after update 2, replayed: the stage's own first two batches from each
    u140 state (the replayed training logs can be checked against the stage's), then the two text sets."""
    common = await Math(m.s, m.f, pilot).depth_texts("M0", None)
    datums = m.f.later_stages()["it"]

    async def one(label):
        folder = m.root / "r1" / label
        if all((folder / "u2" / f"{kind}.npz").exists() for kind in ("own", "common")):
            return
        own = await m.depth_texts(label, None)  # stored at handoff
        model = await m.s.service.create_training_client_from_state_async(train_state(m.root, label))
        async for _ in m.s.fit(model, datums, LATER_LR, 2, lambda u: folder / "train" / f"u{u}.json"):
            pass
        after = await model.save_weights_and_get_sampling_client_async()
        await asyncio.gather(*(m.s.score_set(after, texts, folder / "u2" / f"{kind}.npz", DEPTH_LENGTH)
                               for kind, texts in (("own", own), ("common", common))))

    await asyncio.gather(*(one(label) for label in EARLY_LABELS))


async def execute(args):
    root, pilot = REPO / "runs/math" / args.run_id, REPO / "runs/math" / args.pilot_run
    if args.phase == "teacher":
        session = Session(root / "teacher", TEACHER, args.limit)
        pool = {p["task_id"]: p for p in jsonl(REPO / "inputs/math/pool.jsonl")}
        await traces(session, root, teacher_problems(pool, json.loads((pilot / "arms.json").read_text())))
    else:
        session = Session(root, BASE, args.limit)
        await (student if args.phase == "student" else probe)(Math(session, Family(BASE), root), pilot)
    print(f"done; list-price spend ${session.ledger.total():.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="teacher-20260924")
    parser.add_argument("--phase", choices=("teacher", "student", "probe"), required=True)
    parser.add_argument("--pilot-run", default="pilot-20260924")
    parser.add_argument("--limit", type=float, required=True, help="this phase's list-price limit in USD")
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
