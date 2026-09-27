#!/usr/bin/env python3
"""Competition-math experiments (records/math-pilot.md, main-phase.md, reforcing.md).

A base model's own correct forced solutions train LoRA arms for 140 updates; the base and the
states are evaluated before and after 20 updates of each later stage, and scored against the base.
Every step skips work whose output exists, so a relaunch resumes, and released files copied into
the run folder are used as they are.

  rft      the pool in order: a problem is easy if the base answers it within 256 tokens after an
           empty think block; otherwise one forced sample at top-p 1, kept if correct and uncapped,
           until 4,480 are kept
  arms     D: the first 579 kept, O: the first 4,480, each in its own hashed order (the pilot);
           P: D's problems with a fresh correct sample at every visit; D2: 579 of O outside D;
           R128: D at LoRA rank 128 (the main arms, on a pilot run of the same base)
  train    a fresh LoRA, 140 updates of 32 at 1e-4
  evaluate MATH-500 levels 3-5, forced (4 draws) and free (2), 16,384 tokens; AIME forced, 32,768
  later    20 updates of instruction tuning (640 No Robots) or Stage B (answer-only program
           synthesis) at 3e-4, then evaluate
  depth    B: the base's forced texts (top-p 1, 2,048 tokens) scored under each state
  reforce  continue forced samples that end early with "\\n\\nWait" (the budget and answer rules)
"""

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path

from common import maths
from common.models import Family, jsonl
from common.tinker_io import Session, raw_datums, stamp, write

REPO = Path(__file__).resolve().parents[1]
CAP, AIME_CAP, MAX_LENGTH, DEPTH_LENGTH = 16384, 32768, 17000, 2048
DRAWS, SMALL, LARGE, UPDATES = {"forced": 4, "unforced": 2}, 579, 4480, 140
BRANCHES = {"D": (20, 60, 140), "O": (60, 140)}
EVALUATED, AIME_STATES = ("M0", "D-u20", "D-u60", "D-u140", "O-u140"), ("M0", "D-u140", "O-u140")
LR, LATER_LR, LATER_UPDATES, TTL_DAYS = 1e-4, 3e-4, 20, 30
PER_PROBLEM, MAX_DRAWS = 8, 24  # the P arm: kept samples and draws per problem
CUE, MIN_TOKENS, CONTINUATIONS = "\n\nWait", 1024, 8
RULES = {"budget": lambda n, boxed: n < MIN_TOKENS or not boxed, "answer": lambda n, boxed: not boxed}
TAG = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


def ordered(rows, namespace):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{namespace}:{r['task_id']}".encode()).hexdigest())


class Math:
    def __init__(self, session, family, root):
        self.s, self.f, self.root = session, family, root
        self.pool = jsonl(REPO / "inputs/math/pool.jsonl")
        self.items = [it for it in jsonl(REPO / "inputs/math/eval_math500.jsonl") if it["level"] >= 3]
        self.aime = jsonl(REPO / "inputs/math/aime.jsonl")

    def prompt(self, problem):
        return self.f.render(maths.INSTRUCTION + problem)

    async def draw(self, sampler, tokens, n, cap=CAP, top_p=0.95):
        return await self.s.sample(sampler, tokens, n, cap, self.f.stops, top_p)

    async def rft(self, base, pool=None, until=LARGE):
        """The screen over `pool` (the whole pool by default) in chunks of 512, until `until` are kept."""
        out = self.root / "rft.jsonl"
        rows = [json.loads(line) for line in out.open()] if out.exists() else []
        done = {r["task_id"] for r in rows}
        remaining = [p for p in (self.pool if pool is None else pool) if p["task_id"] not in done]
        while sum(r["correct"] for r in rows) < until and remaining:
            chunk, remaining = remaining[:512], remaining[512:]
            direct = await asyncio.gather(*(self.draw(base, self.prompt(p["problem"]) + self.f.direct, 1, 256)
                                            for p in chunk))
            easy = {p["task_id"] for p, [t] in zip(chunk, direct) if maths.extract(self.f.tok.decode(t)) == p["answer"]}
            hard = [p for p in chunk if p["task_id"] not in easy]
            drawn = dict(zip((p["task_id"] for p in hard), await asyncio.gather(
                *(self.draw(base, self.prompt(p["problem"]) + self.f.think, 1, top_p=1.0) for p in hard))))
            for p in chunk:
                if p["task_id"] in easy:
                    rows.append(dict(task_id=p["task_id"], easy=True, correct=False))
                    continue
                [tokens] = drawn[p["task_id"]]
                rows.append(dict(task_id=p["task_id"], easy=False, prompt_token_ids=self.prompt(p["problem"]),
                                 generation={"completion_token_ids": self.f.think + tokens}, tokens=len(tokens),
                                 capped=len(tokens) >= CAP, correct=len(tokens) < CAP
                                 and maths.extract(self.f.tok.decode(tokens)) == p["answer"]))
            write(out, rows)
            kept = sum(r["correct"] for r in rows)
            print(f"{stamp()} rft {len(rows)} screened, {kept} kept, spent=${self.s.ledger.total():.2f}", flush=True)
            if len(rows) >= 2048 and kept < 0.2 * len(rows):
                raise RuntimeError(f"keep rate collapsed: {kept} of {len(rows)}")
        return rows

    async def train(self, label, rows, rank=32, branches=(UPDATES,), seed=20260924):
        """A fresh LoRA (initialized from `seed`) trained on [{prompt_token_ids, generation.completion_token_ids}];
        returns the state paths saved after the updates in `branches`, keyed by update."""
        folder = self.root / "train" / label
        paths = {u: folder / f"u{u}-state.json" for u in branches}
        if not all(p.exists() for p in paths.values()):
            texts = [dict(prompt=r["prompt_token_ids"], response=r["generation"]["completion_token_ids"]) for r in rows]
            model = await self.s.fresh_lora(rank, seed)
            async for update in self.s.fit(model, raw_datums(texts, MAX_LENGTH), LR, UPDATES, lambda u: folder / f"u{u}.json"):
                if update in paths:
                    path = await self.s.save(model, f"{self.root.name}-{label}-u{update}", TTL_DAYS)
                    write(paths[update], {"state_path": path})
                    print(f"{stamp()} {label} u{update} saved", flush=True)
        return {u: json.loads(p.read_text())["state_path"] for u, p in paths.items()}

    async def evaluate(self, sampler, name, when, bench="math500", modes=("forced", "unforced")):
        items, cap = (self.items, CAP) if bench == "math500" else (self.aime, AIME_CAP)
        for mode in modes:
            out = self.root / "eval" / bench / name / f"{when}-{mode}.jsonl"
            if out.exists():
                continue
            prefix = self.f.think if mode == "forced" else []
            drawn = await asyncio.gather(*(self.draw(sampler, self.prompt(it["problem"]) + prefix, DRAWS[mode], cap)
                                           for it in items))
            rows = []
            for it, sequences in zip(items, drawn):
                for draw, tokens in enumerate(sequences):
                    text = self.f.tok.decode(tokens)
                    tags = TAG.findall(text)  # Stage B's answer-tag format, a secondary reading
                    rows.append(dict(task_id=it["unique_id"], level=it["level"], draw=draw, tokens=len(tokens),
                                     capped=len(tokens) >= cap, correct=maths.extract(text) == int(it["answer"]),
                                     boxed="\\boxed" in text, tagged=int(tags[-1]) if tags else None,
                                     think=mode == "forced" or text.lstrip().startswith("<think>"),
                                     text=text, token_ids=tokens))
            write(out, rows)
            print(f"{stamp()} {bench} {name} {when} {mode}: correct={sum(r['correct'] for r in rows) / len(rows):.3f} "
                  f"spent=${self.s.ledger.total():.2f}", flush=True)

    async def later(self, label, path, stage, datums, aime=False, modes=("forced", "unforced"), hook=None):
        """20 updates of a later stage from a state (a fresh LoRA for the base), saved, then evaluated.
        `hook(model, update)`, awaited after each update when the stage is trained, returns more jobs
        to run with the evaluations (the teacher run's probe after update 2)."""
        name, folder = f"{label}+{stage}", self.root / "later" / f"{label}+{stage}"
        saved, jobs = folder / "state.json", []
        if saved.exists():
            sampler = await self.s.sampler(json.loads(saved.read_text())["state_path"])
        else:
            model = await (self.s.fresh_lora() if path is None else
                           self.s.service.create_training_client_from_state_async(path))
            async for update in self.s.fit(model, datums, LATER_LR, LATER_UPDATES, lambda u: folder / f"u{u}.json"):
                jobs += await hook(model, update) if hook else []
            write(saved, {"state_path": await self.s.save(model, f"{self.root.name}-{label}-{stage}", TTL_DAYS)})
            sampler = await model.save_weights_and_get_sampling_client_async()
        jobs.append(self.evaluate(sampler, name, "u20", modes=modes))
        if aime:
            jobs.append(self.evaluate(sampler, name, "u20", "aime", ("forced",)))
        await asyncio.gather(*jobs)

    async def depth_texts(self, label, sampler):
        """One forced sample per problem at top-p 1, cut at 2,048 tokens with the prefix."""
        out = self.root / "depth" / label / "texts.jsonl"
        if not out.exists():
            drawn = await asyncio.gather(*(self.draw(sampler, self.prompt(it["problem"]) + self.f.think, 1,
                                                     DEPTH_LENGTH - len(self.f.think), top_p=1.0) for it in self.items))
            write(out, [dict(task_id=it["unique_id"], token_ids=seq) for it, [seq] in zip(self.items, drawn)])
        rows = {r["task_id"]: r["token_ids"] for r in map(json.loads, out.open())}
        return [dict(task_id=it["unique_id"], prompt=self.prompt(it["problem"]), response=self.f.think + rows[it["unique_id"]])
                for it in self.items]

    async def own(self, base, label, sampler):
        """B': the state's own forced texts scored under the state and under the base."""
        own = await self.depth_texts(label, sampler)
        await asyncio.gather(self.s.score_set(sampler, own, self.root / "depth" / label / "own.npz", DEPTH_LENGTH),
                             self.s.score_set(base, own, self.root / "depth" / "M0" / f"own-{label}.npz", DEPTH_LENGTH))

    async def common(self, sampler, label, texts):
        await self.s.score_set(sampler, texts, self.root / "depth" / label / "common.npz", DEPTH_LENGTH)

    async def continue_sample(self, sampler, prompt, row, answer, rule):
        """reforcing.md: remove the end of turn, append the cue, resume; at most 8 times, 16,384 in all."""
        ids, times = list(row["token_ids"]), 0
        while times < CONTINUATIONS and len(ids) < CAP and RULES[rule](len(ids), "\\boxed" in self.f.tok.decode(ids)):
            kept = self.f.strip_turn_end(ids) + self.f.tok.encode(CUE, add_special_tokens=False)
            if len(kept) >= CAP:
                break
            [more] = await self.draw(sampler, prompt + kept, 1, CAP - len(kept))
            ids, times = kept + more, times + 1
        text = self.f.tok.decode(ids)
        return dict(task_id=row["task_id"], level=row["level"], draw=row["draw"], first_correct=row["correct"],
                    first_tokens=row["tokens"], continuations=times, tokens=len(ids), capped=len(ids) >= CAP,
                    boxed="\\boxed" in text, correct=maths.extract(text) == answer, text=text, token_ids=ids)

    async def reforce(self, sampler, name, when):
        rows = [json.loads(line) for line in (self.root / "eval/math500" / name / f"{when}-forced.jsonl").open()]
        items = {it["unique_id"]: it for it in self.items}
        for rule in RULES:
            out = self.root / "reforce" / rule / name / f"{when}-forced.jsonl"
            if not out.exists():
                write(out, await asyncio.gather(*(self.continue_sample(
                    sampler, self.prompt(items[r["task_id"]]["problem"]) + self.f.think, r,
                    int(items[r["task_id"]]["answer"]), rule) for r in rows)))


async def pilot_design(m, base):
    """records/math-pilot.md on one base model."""
    kept = [r for r in await m.rft(base) if r["correct"]]
    sets = {"D": ordered(kept[:SMALL], "math-pilot-20260924:D"), "O": ordered(kept[:LARGE], "math-pilot-20260924:O")}
    write(m.root / "arms.json", {arm: [r["task_id"] for r in rows] for arm, rows in sets.items()})
    paths = {}
    for arm, got in zip(sets, await asyncio.gather(*(m.train(a, sets[a], branches=BRANCHES[a]) for a in sets))):
        paths |= {f"{arm}-u{u}": p for u, p in got.items()}
    samplers = {"M0": base} | dict(zip(paths, await asyncio.gather(*(m.s.sampler(p) for p in paths.values()))))
    stages = m.f.later_stages()
    common = await m.depth_texts("M0", base)
    jobs = [m.evaluate(samplers[label], label, "u0") for label in EVALUATED]
    jobs += [m.evaluate(samplers[label], label, "u0", "aime", ("forced",)) for label in AIME_STATES]
    jobs += [m.later(label, paths.get(label), stage, datums, aime=stage == "it" and label in AIME_STATES)
             for label in EVALUATED for stage, datums in stages.items()]
    jobs += [m.common(s, label, common) for label, s in samplers.items()]
    jobs += [m.own(base, label, samplers[label]) for label in EVALUATED[1:]]
    await asyncio.gather(*jobs)


async def p_samples(m, base, d_rows):
    """The P arm's texts: up to 8 correct uncapped forced samples per D problem at top-p 1 (at most
    24 draws), D's own sample first."""
    out = m.root / "p_samples.jsonl"
    have = {r["task_id"]: r for r in map(json.loads, out.open())} if out.exists() else {}
    pool = {p["task_id"]: p for p in m.pool}

    async def one(row):
        problem, kept, draws = pool[row["task_id"]], [row["generation"]["completion_token_ids"]], 1
        while len(kept) < PER_PROBLEM and draws < MAX_DRAWS:
            n = min(PER_PROBLEM - len(kept), MAX_DRAWS - draws)
            for tokens in await m.draw(base, m.prompt(problem["problem"]) + m.f.think, n, top_p=1.0):
                if len(kept) < PER_PROBLEM and len(tokens) < CAP and maths.extract(m.f.tok.decode(tokens)) == problem["answer"]:
                    kept.append(m.f.think + tokens)
            draws += n
        return dict(task_id=row["task_id"], prompt_token_ids=row["prompt_token_ids"], samples=kept, draws=draws)

    todo = [r for r in d_rows if r["task_id"] not in have]
    for start in range(0, len(todo), 64):
        for result in await asyncio.gather(*(one(r) for r in todo[start:start + 64])):
            have[result["task_id"]] = result
        write(out, list(have.values()))
    samples = [have[r["task_id"]] for r in d_rows]  # slot s visits problem s mod 579 for the (s // 579)-th time
    return [dict(prompt_token_ids=samples[s % SMALL]["prompt_token_ids"], generation={"completion_token_ids":
            samples[s % SMALL]["samples"][(s // SMALL) % len(samples[s % SMALL]["samples"])]}) for s in range(UPDATES * 32)]


async def main_arms(m, base, pilot, arms):
    """records/main-phase.md on the 9B: P, D2 and rank 128, against the pilot's own D and O."""
    sets = pilot_sets(pilot)
    rows = {}
    if "P" in arms:
        rows["P-u140"] = (await p_samples(m, base, sets["D"]), 32)
    if "D2" in arms:
        rows["D2-u140"] = (d2_rows(sets), 32)
    if "R128" in arms:
        rows["D-r128-u140"] = (sets["D"], 128)
    stages, common = m.f.later_stages(), await m.depth_texts("M0", base)

    async def drilled_fit(r128):
        """N2: D's drilled texts under the base, the pilot's rank-32 D-u140 and the rank-128 arm."""
        texts = [dict(task_id=r["task_id"], prompt=r["prompt_token_ids"], response=r["generation"]["completion_token_ids"])
                 for r in sets["D"]]
        d140 = await m.s.sampler(train_state(pilot, "D-u140"))
        await asyncio.gather(*(m.s.score_set(s, texts, m.root / "drilled" / label / "drilled.npz", MAX_LENGTH)
                               for label, s in (("M0", base), ("D-u140", d140), ("D-r128-u140", r128))))

    async def arm(label, data, rank):
        path = (await m.train(label, data, rank))[UPDATES]
        sampler = await m.s.sampler(path)
        jobs = [m.evaluate(sampler, label, "u0"), m.common(sampler, label, common), m.own(base, label, sampler)]
        jobs += [m.later(label, path, stage, d, aime=label == "P-u140" and stage == "it") for stage, d in stages.items()]
        if label == "P-u140":
            jobs.append(m.evaluate(sampler, label, "u0", "aime", ("forced",)))
        if label == "D-r128-u140":
            jobs.append(drilled_fit(sampler))
        await asyncio.gather(*jobs)

    await asyncio.gather(*(arm(label, data, rank) for label, (data, rank) in rows.items()))


def pilot_sets(pilot):
    rows = [json.loads(line) for line in (pilot / "rft.jsonl").open()]
    kept = [r for r in rows if r["correct"]]
    sets = {"D": ordered(kept[:SMALL], "math-pilot-20260924:D"), "O": ordered(kept[:LARGE], "math-pilot-20260924:O")}
    arms = json.loads((pilot / "arms.json").read_text())
    assert all([r["task_id"] for r in sets[arm]] == arms[arm] for arm in sets)
    return sets


def d2_rows(sets):
    """D2: 579 of O's problems outside D, drawn by hash, in a hashed order."""
    inside = {r["task_id"] for r in sets["D"]}
    chosen = ordered([r for r in sets["O"] if r["task_id"] not in inside], "main-20260924:s2")[:SMALL]
    return ordered(chosen, "main-20260924:s2:order")


async def execute(args):
    root = REPO / "runs/math" / args.run_id
    session = Session(root, args.model, args.limit)
    m = Math(session, Family(args.model), root)
    base = await session.service.create_sampling_client_async(base_model=args.model)
    if args.reforce:
        for label in args.reforce:
            for name, when in ((label, "u0"), (f"{label}+it", "u20"), (f"{label}+b", "u20")):
                state = root / "later" / name / "state.json" if "+" in name else None
                path = json.loads(state.read_text())["state_path"] if state else train_state(root, label)
                await m.reforce(base if path is None else await session.sampler(path), name, when)
    elif args.arms:
        await main_arms(m, base, REPO / "runs/math" / args.pilot_run, set(args.arms))
    else:
        await pilot_design(m, base)
    print(f"done; list-price spend ${session.ledger.total():.2f}", flush=True)


def train_state(root, label):
    if label == "M0":
        return None
    arm, update = label.rsplit("-u", 1)
    for path in (root / "train" / arm / f"u{update}-state.json", root / "train" / label / f"u{UPDATES}-state.json",
                 root / "train" / label / "state.json"):  # the last: DuraSeed-v1's main run, as released
        if path.exists():
            return json.loads(path.read_text())["state_path"]
    raise FileNotFoundError(label)


async def guarded(root, label, job):
    """Run one arm's job; a failure is recorded in errors/ and the other jobs continue."""
    try:
        await job
    except Exception as error:
        write(root / "errors" / f"{label}.json", {"type": type(error).__name__, "message": str(error)})
        print(f"{label} FAILED: {error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--limit", type=float, required=True, help="this run's list-price limit in USD")
    parser.add_argument("--arms", nargs="*", choices=("P", "D2", "R128"), help="the main arms (needs --pilot-run)")
    parser.add_argument("--pilot-run", default="pilot-20260924")
    parser.add_argument("--reforce", nargs="*", help="labels whose forced samples to continue (reforcing.md)")
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
