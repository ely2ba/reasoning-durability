#!/usr/bin/env python3
"""TCES experiments on Qwen3.5-9B-Base (records/decision-competence.md, repetition-fragility.md,
depth-profile.md, anchor.md). TCES is exact arithmetic search: combine given integers with + - * /
to reach a target.

  drill   from M0's weights, 140 updates at 1e-4 on 579 texts of the clone corpus (the RL teacher's
          own samples) cycled in a fixed order: U (any), F (correct only), U2 (a second draw of U).
          FA1, FA025 and FN drill F while each batch also takes 32 fresh examples: M0's own forced
          samples at loss weight 1 or 0.25, or No Robots instructions
  later   from a state's weights (fresh optimizer), 20 updates of Stage B (answer-only program
          synthesis) or instruction tuning, with probes after the listed updates:
            decision  8 tokens, 4 draws, targeted and held-out items: does the reply open with an answer?
            think     forced with "\\n<think>\\n", 2 draws, targeted items
            unforced  the plain prompt, 2 draws, targeted items
          a forced search that stops on a restated answer template (a tag without digits) is resumed
  score   M0's forced texts on the panel scored under each state, for B

States are this run's own (runs/tces/<run-id>/<arm>/stage_a/u<N>-state.json) or supplied in its
folder as <label>/state.json, {"state_path": ...}: M0 for drill, and the paper's states for later and
score (data/checkpoints.json lists them; they are available on request). Every step skips work whose
output exists.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from common import tces
from common.models import Family, jsonl
from common.tinker_io import Session, raw_datums, stamp, write

REPO = Path(__file__).resolve().parents[1]
MODEL, CORPUS = "Qwen/Qwen3.5-9B-Base", REPO / "release/clone-corpus"
PREFIXES = {"think": "\n<think>\n", "competence": " Let's think step by step.\n\n"}
STOPS, DECISION_TOKENS, RESUMES = ["\n\nUser:", "</answer>", "\nUser:"], 8, 16
DRILL_LR, UPDATES, MAX_LENGTH, BRANCHES = 1e-4, 140, 4231, {"U": (20, 60, 140), "F": (20, 60, 140)}
ANCHORS, PER_PROMPT, ANCHOR_CAP = 4480, 6, 4096


def training_rows(rows):
    """Rows as raw_datums reads them; the released files keep the original runs' field names."""
    return [r if "prompt" in r else dict(r, prompt=r["prompt_token_ids"], response=r["generation"]["completion_token_ids"])
            for r in rows]


class Tces:
    def __init__(self, session, root, cap):
        self.s, self.root, self.cap, self.f = session, root, cap, Family(MODEL)
        self.items = jsonl(REPO / "data/tces/panel.jsonl")
        self.prefixes = {k: self.f.tok.encode(v, add_special_tokens=False) for k, v in PREFIXES.items()}

    def state(self, label):
        arm, _, update = label.rpartition("-u")
        supplied = self.root / label / "state.json"
        for path in (self.root / arm / "stage_a" / f"u{update}-state.json", supplied):
            if path.exists():
                return json.loads(path.read_text())["state_path"]
        raise FileNotFoundError(f"state {label}: not trained in this run, and not supplied as {supplied} "
                                '({"state_path": ...}); data/checkpoints.json lists the paper\'s states, which are '
                                "available on request (README.md)")

    async def probe(self, sampler, label, update, kind, roles, draws):
        out = self.root / label / f"u{update}" / f"{kind}.jsonl"
        if out.exists():
            return
        cap, prefix = (DECISION_TOKENS if kind == "decision" else self.cap), self.prefixes.get(kind, [])

        async def resume(prompt, tokens):
            resumed = 0
            while (kind != "decision" and len(tokens) < cap and resumed < RESUMES
                   and tces.template_stop(self.f.completion(tokens))):
                [more] = await self.s.sample(sampler, prompt + tokens, 1, cap - len(tokens), STOPS)
                if not more:
                    break
                tokens, resumed = tokens + more, resumed + 1
            return tokens, resumed

        async def one(item):
            prompt = item["prompt_token_ids"] + prefix
            first = await self.s.sample(sampler, prompt, draws, cap, STOPS)
            rows = []
            for draw, (tokens, resumed) in enumerate(await asyncio.gather(*(resume(prompt, t) for t in first))):
                text = self.f.completion(tokens)
                row = dict(task_id=item["task_id"], role=item["role"], draw=draw, tokens=len(tokens),
                           stop="length" if len(tokens) >= cap else "stop")
                if kind == "decision":
                    row.update(text=text, skip=text.lstrip().startswith("<answer>"))
                else:
                    full = PREFIXES.get(kind, "") + text
                    correct = tces.score(full, item["task"])
                    row.update(text=full, resumed=resumed, correct=correct, strict_correct=correct and resumed == 0)
                rows.append(row)
            return rows

        rows = [r for group in await asyncio.gather(*(one(it) for it in self.items if it["role"] in roles))
                for r in group]
        write(out, rows)
        key = "skip" if kind == "decision" else "correct"
        print(f"{stamp()} {label} u{update} {kind}: {key}={sum(r[key] for r in rows) / len(rows):.3f} "
              f"spent=${self.s.ledger.total():.2f}", flush=True)

    async def later(self, label, datums, lr, last, decision=(), forced=(), unforced=(), prefixes=("think",)):
        """`last` updates of a later stage from a state's weights, probing after the listed updates."""
        model = await self.s.service.create_training_client_from_state_async(self.state(label))
        probes = []

        def kinds(update):
            out = [("decision", ("targeted", "sentinel"), 4)] if update in decision else []
            out += [(k, ("targeted",), 2) for k in prefixes] if update in forced else []
            return out + ([("unforced", ("targeted",), 2)] if update in unforced else [])

        async def probe_now(update):
            if kinds(update):
                sampler = await model.save_weights_and_get_sampling_client_async()
                probes.extend(asyncio.create_task(self.probe(sampler, label, update, *k)) for k in kinds(update))

        await probe_now(0)
        async for update in self.s.fit(model, datums, lr, last, lambda u: self.root / label / f"u{u}" / "train.json"):
            await probe_now(update)
        await asyncio.gather(*probes)

    def corpus(self):
        rows = {}
        for path in sorted(CORPUS.glob("group-*/samples.json")):
            for r in json.loads(path.read_text())["records"]:
                rows[r["generation"]["sample_id"]] = dict(prompt=r["prompt_token_ids"],
                                                          response=r["generation"]["completion_token_ids"])
        return rows

    async def anchor_samples(self, m0):
        """Six forced samples of M0 per clone prompt at top-p 1 (so fitting them is the sampled gradient
        of KL(M0 || state)); the first 4,480 in the prompts' hashed order."""
        out, part = self.root / "anchor.jsonl", self.root / "anchor.partial.jsonl"
        if not out.exists():
            prompts = {r["task_id"]: r["prompt_token_ids"] for r in jsonl(REPO / "data/tces/clone_prompts.jsonl")}
            order = sorted(prompts, key=lambda t: hashlib.sha256(f"anchor-20260924:{t}".encode()).hexdigest())
            rows = jsonl(part) if part.exists() else []
            todo = order[len(rows) // PER_PROMPT:]
            for start in range(0, len(todo), 100):  # saved per chunk, so a failure loses at most one
                chunk = todo[start:start + 100]
                drawn = await asyncio.gather(*(self.s.sample(m0, prompts[t] + self.prefixes["think"], PER_PROMPT,
                                                             ANCHOR_CAP, STOPS[::2], top_p=1.0) for t in chunk))
                rows += [dict(task_id=t, draw=i, prompt=prompts[t] + self.prefixes["think"], response=s)
                         for t, group in zip(chunk, drawn) for i, s in enumerate(group)]
                write(part, rows)
            write(out, rows[:ANCHORS])
        return training_rows(jsonl(out))

    async def drill(self, arm, rows, extra=()):
        """140 updates from M0's weights on `rows` cycled, with `extra` datums seen once."""
        branches = BRANCHES.get(arm, (UPDATES,))
        states = {u: self.root / arm / "stage_a" / f"u{u}-state.json" for u in branches}
        if all(p.exists() for p in states.values()):
            return
        model = await self.s.service.create_training_client_from_state_async(self.state("M0"))
        log = lambda u: self.root / arm / "stage_a" / f"u{u}.json"  # noqa: E731
        async for update in self.s.fit(model, raw_datums(rows, MAX_LENGTH), DRILL_LR, UPDATES, log, extra):
            if update in states:
                write(states[update], {"state_path": await self.s.save(model, f"{self.root.name}-{arm}-u{update}")})
                print(f"{stamp()} {arm} u{update} saved", flush=True)


async def drill(t, arms):
    subsets = json.loads((REPO / "data/tces/subsets.json").read_text())
    corpus = t.corpus()
    rows = {arm: [corpus[i] for i in subsets[arm]["sample_ids"]] for arm in ("U", "F", "U2")}
    jobs = [t.drill(arm, rows[arm]) for arm in ("U", "F", "U2") if arm in arms]
    if {"FA1", "FA025"} & set(arms):
        anchors = await t.anchor_samples(await t.s.sampler(t.state("M0")))
        jobs += [t.drill(arm, rows["F"], raw_datums(anchors, MAX_LENGTH + 8, weight))
                 for arm, weight in (("FA1", 1.0), ("FA025", 0.25)) if arm in arms]
    if "FN" in arms:
        robots = jsonl(REPO / "inputs/no_robots.jsonl")[640:640 + ANCHORS]  # the FN stream, after the 640 of IT
        jobs.append(t.drill("FN", rows["F"], t.f.datums([(r["prompt"], r["completion"]) for r in robots], 1024)))
    await asyncio.gather(*jobs)


async def score(t, labels, length):
    texts = [dict(r, response=r["response"][:length]) for r in jsonl(REPO / "data/tces/common_texts.jsonl")]
    samplers = [await t.s.sampler(t.state(label)) for label in labels]
    await asyncio.gather(*(t.s.score_set(s, texts, t.root / f"depth-{length}" / label / "common.npz", length)
                           for label, s in zip(labels, samplers)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=float, required=True, help="this run's list-price limit in USD")
    parser.add_argument("--cap", type=int, default=16384, help="the forced probes' token budget (4,096 or 16,384)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("drill").add_argument("--arms", nargs="+", default=("U", "F", "U2", "FA1", "FA025", "FN"))
    later = sub.add_parser("later")
    later.add_argument("--states", nargs="+", required=True)
    later.add_argument("--stage", choices=("b", "it"), default="b")
    later.add_argument("--lr", type=float, default=3e-4)
    later.add_argument("--last", type=int, default=20)
    for name in ("decision", "forced", "unforced"):
        later.add_argument(f"--{name}", type=int, nargs="*", default=(), help="updates after which to probe")
    later.add_argument("--prefixes", nargs="+", default=("think",), choices=tuple(PREFIXES))
    scoring = sub.add_parser("score")
    scoring.add_argument("--states", nargs="+", required=True)
    scoring.add_argument("--length", type=int, default=4096)
    args = parser.parse_args()
    root = REPO / "runs/tces" / args.run_id
    t = Tces(Session(root, MODEL, args.limit), root, args.cap)
    if args.command == "drill":
        job = drill(t, set(args.arms))
    elif args.command == "score":
        job = score(t, args.states, args.length)
    else:
        datums = t.f.later_stages()[args.stage]

        async def stages():  # asyncio.run takes a coroutine, and gather must run inside its loop
            await asyncio.gather(*(t.later(label, datums, args.lr, args.last, set(args.decision), set(args.forced),
                                           set(args.unforced), tuple(args.prefixes)) for label in args.states))
        job = stages()
    asyncio.run(job)
    print(f"done; list-price spend ${t.s.ledger.total():.2f}", flush=True)


if __name__ == "__main__":
    main()
