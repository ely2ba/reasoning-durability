#!/usr/bin/env python3
"""Revision parts SH and SD (records/revision.md), which train new arms on the 9B.

  SH  sharpening without repetition. O-sharp takes O's 4,480 problems in O's order, each with the base's first
      correct, uncapped forced sample at temperature 0.5 (top-p 1, up to 4 draws; a problem without one drops
      out and training fills its slot by cycling from the start, as O-T's did), and is trained exactly as O.
      Before any evaluation its B on the pilot's base texts must reach 0.05 nats per token; if it does not, the
      arm is sampled and trained once more at 0.3, and that arm is final. The final arm is evaluated at handoff
      and after IT and AO at the defaults (3e-4, 20 updates).
  SD  seeds. One launch per seed (s2 or s3), each in its own subfolder with its own ledger. A seed draws 579 of
      O's problems outside D's and D2's, D's order, and a new order of O's 4,480 texts, each by SHA-256 in the
      namespace revision-sd-<seed>, and LoRA seed 2 or 3. D, O and P are trained as in the paper (P's fresh
      samples under the main phase's rule), then evaluated at handoff, after IT at the defaults and after IT on
      R2's schedule (1e-4, 60 updates).

Evaluations are forced, as in the paper. Every step skips work whose output exists, so a relaunch resumes.
"""

import argparse
import asyncio

import numpy as np

from common import divergence, maths, stats
from common.models import Family, jsonl
from common.tinker_io import Session, stamp, write
from run_math import (CAP, DEPTH_LENGTH, LATER_LR, LATER_UPDATES, REPO, SMALL, UPDATES, Math, d2_rows, guarded,
                      ordered, p_samples, pilot_sets)
from run_revision import stage
from run_robustness import BASE, GENTLE_LR, LONG, instruction

TEMPERATURES, GATE, DRAWS, CHUNK = (0.5, 0.3), 0.05, 4, 512
LORA_SEEDS = {"s2": 2, "s3": 3}  # the paper's arms used 20260924


async def sharp_rows(m, base, o_rows, temperature):
    """For each of O's problems, in O's order, the base's first correct and uncapped forced sample at
    `temperature` (top-p 1) within 4 draws, saved after every chunk of 512 problems. A problem without one
    drops out."""
    out = m.root / f"sharp-t{round(10 * temperature):02d}.jsonl"
    have = {r["task_id"]: r for r in jsonl(out)} if out.exists() else {}
    answers = {p["task_id"]: p["answer"] for p in m.pool}

    async def one(row):
        for draw in range(1, DRAWS + 1):
            [tokens] = await m.s.sample(base, row["prompt_token_ids"] + m.f.think, 1, CAP, m.f.stops, 1.0, temperature)
            if len(tokens) < CAP and maths.extract(m.f.tok.decode(tokens)) == answers[row["task_id"]]:
                return dict(task_id=row["task_id"], draws=draw, prompt_token_ids=row["prompt_token_ids"],
                            generation={"completion_token_ids": m.f.think + tokens})
        return dict(task_id=row["task_id"], draws=DRAWS, generation=None)

    todo = [r for r in o_rows if r["task_id"] not in have]
    for start in range(0, len(todo), CHUNK):
        for row in await asyncio.gather(*(one(r) for r in todo[start:start + CHUNK])):
            have[row["task_id"]] = row
        write(out, list(have.values()))
        print(f"{stamp()} T={temperature}: {len(have)} problems, {sum(r['generation'] is not None for r in have.values())}"
              f" kept, spent=${m.s.ledger.total():.2f}", flush=True)
    return [have[r["task_id"]] for r in o_rows if have[r["task_id"]]["generation"]]


def b_value(pilot, scores):
    """B on the pilot's base texts against the pilot's base scores, with its item-bootstrap interval, as the paper
    computes it: per-text sums of the float32 differences, one generator seeded 20260924."""
    agg = divergence.aggregate(divergence.load(pilot / "depth/M0/common.npz"), divergence.load(scores), 3)
    d, n = agg["d"], agg["n"]
    picks = np.random.default_rng(stats.SEED).integers(0, len(d), size=(stats.RESAMPLES, len(d)))
    return {"B": float(d.sum() / n.sum()), "interval": stats.percentiles(d[picks].sum(1) / n[picks].sum(1))}


async def gated(m, base, pilot, o_rows):
    """O-sharp at 0.5, and once more at 0.3 if its B misses the gate; the final arm's label and state."""
    common = await Math(m.s, m.f, pilot).depth_texts("M0", None)  # the pilot's base texts, as stored
    record = {"gate": GATE, "arms": {}}
    for temperature in TEMPERATURES:
        label = f"O-sharp-t{round(10 * temperature):02d}-u140"
        rows = await sharp_rows(m, base, o_rows, temperature)
        path = (await m.train(label, rows))[UPDATES]
        scores = m.root / "depth" / label / "common.npz"
        if not scores.exists():
            await m.s.score_set(await m.s.sampler(path), common, scores, DEPTH_LENGTH)
        b = b_value(pilot, scores)
        record["arms"][label] = {"temperature": temperature, "texts": len(rows), **b}
        record["final"] = label
        write(m.root / "gate.json", record)
        print(f"{stamp()} gate {label}: B={b['B']:.4f}", flush=True)
        if b["B"] >= GATE:
            break
    return label, path


async def evaluated(m, label, path, stages):
    """The arm forced at handoff, and after each (tag, datums, learning rate, updates) stage from its state."""
    jobs = [m.evaluate(await m.s.sampler(path), label, "u0", modes=("forced",))]
    jobs += [stage(m, label, path, datums, lr, updates, (updates,), tag) for tag, datums, lr, updates in stages]
    await asyncio.gather(*jobs)


def seed_rows(sets, seed):
    """For `seed`: the first 579 of O's problems outside D's and D2's in the seed's SHA-256 order, as D in a
    second such order, and O's 4,480 texts in a third."""
    taken = {r["task_id"] for r in sets["D"] + d2_rows(sets)}
    subset = ordered([r for r in sets["O"] if r["task_id"] not in taken], f"revision-sd-{seed}:subset")[:SMALL]
    return {"D": ordered(subset, f"revision-sd-{seed}:D"), "O": ordered(sets["O"], f"revision-sd-{seed}:O")}


async def seeds(m, base, sets, seed):
    rows = seed_rows(sets, seed)
    write(m.root / "arms.json", {arm: [r["task_id"] for r in rs] for arm, rs in rows.items()})
    distinct = instruction(m.f, LONG * 32)  # R2's 1,920 examples; the first 640 are the default stage's
    stages = [("it", distinct[:640], LATER_LR, LATER_UPDATES), ("it-lr1e-4", distinct, GENTLE_LR, LONG)]

    async def arm(name, arm_rows):
        path = (await m.train(f"{name}-u140", arm_rows, seed=LORA_SEEDS[seed]))[UPDATES]
        await evaluated(m, f"{name}-u140", path, stages)

    async def p_arm():  # P's texts: up to 8 correct samples per D problem, D's own first (run_math.p_samples)
        await arm("P", await p_samples(m, base, rows["D"]))

    await asyncio.gather(guarded(m.root, "D", arm("D", rows["D"])), guarded(m.root, "O", arm("O", rows["O"])),
                         guarded(m.root, "P", p_arm()))


async def execute(args):
    pilot = REPO / "runs/math" / args.pilot_run
    root = REPO / "runs/math" / (args.run_id or f"revision-{args.part.lower()}-20260926") / (args.seed or "")
    session = Session(root, BASE, args.limit)
    m = Math(session, Family(BASE), root)
    sets = pilot_sets(pilot)
    base = await session.service.create_sampling_client_async(base_model=BASE)
    if args.part == "SH":
        label, path = await gated(m, base, pilot, sets["O"])
        stages = [(tag, datums, LATER_LR, LATER_UPDATES) for tag, datums in m.f.later_stages().items()]
        await guarded(root, label, evaluated(m, label, path, stages))
    else:
        await seeds(m, base, sets, args.seed)
    print(f"done; list-price spend ${session.ledger.total():.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("SH", "SD"), required=True)
    parser.add_argument("--seed", choices=tuple(LORA_SEEDS), help="SD only: the seed this launch trains")
    parser.add_argument("--run-id", help="under runs/math; by default the paper's, revision-<part>-20260926")
    parser.add_argument("--pilot-run", default="pilot-20260924")
    parser.add_argument("--limit", type=float, required=True, help="this launch's list-price limit in USD")
    args = parser.parse_args()
    if (args.part == "SD") != (args.seed is not None):
        parser.error("--seed is required for SD, and only for SD")
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
