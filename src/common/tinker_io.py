"""Everything the experiments ask of Tinker: sampling, teacher-forced scoring, LoRA training on
exact tokens, saving states, and a list-price ledger that refuses a request that could pass the
spend limit. Pinned to tinker 0.25.0 and tinker-cookbook 0.5.3, the versions the paper used.

Training: Adam (beta 0.9/0.95, eps 1e-12, no weight decay, no clipping), 32 examples per update
taken in order and cycled, cross-entropy on target tokens with per-token weights.
"""

import asyncio
import json
import os
from datetime import UTC, datetime
from math import fsum

import numpy as np
import tinker

PRICES = {  # list prices per million tokens (prefill, sample, train), 2026-09-24
    "Qwen/Qwen3.5-9B-Base": (0.66, 1.995, 1.463),
    "Qwen/Qwen3.5-35B-A3B-Base": (0.54, 1.335, 1.177),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": (0.39, 0.99, 0.88),
    "openai/gpt-oss-120b": (0.70, 2.00, 2.00),  # assumed, as in the teacher run: no list price in our records
}
BATCH, ADAM = 32, dict(beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0, grad_clip_norm=0.0)


def write(path, value):
    """Atomic write: a list as JSON lines, anything else as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(value, list):
        temp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in value))
    else:
        temp.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(temp, path)


def stamp():
    return datetime.now(UTC).strftime("%H:%M:%S")


class Ledger:
    """List-price spend of one run, kept in `path`; in-flight worst cases are held against `limit`."""

    def __init__(self, path, model, limit):
        self.path, self.prices, self.limit, self.held = path, PRICES[model], limit, 0.0
        self.spent = json.loads(path.read_text())["tokens"] if path.exists() else [0, 0, 0]

    def cost(self, prefill=0, sample=0, train=0):
        return fsum(n * p for n, p in zip((prefill, sample, train), self.prices)) / 1e6

    def total(self):
        return self.cost(*self.spent)

    def hold(self, **worst):
        amount = self.cost(**worst)
        if self.total() + self.held + amount > self.limit:
            raise RuntimeError(f"the run's limit of ${self.limit} would be passed; stopping")
        self.held += amount
        return amount

    def settle(self, held, prefill=0, sample=0, train=0):
        self.held -= held
        self.spent = [a + b for a, b in zip(self.spent, (prefill, sample, train))]
        write(self.path, {"tokens": self.spent, "usd": round(self.total(), 4), "limit_usd": self.limit})


class Session:
    """One run's Tinker service, ledger and concurrency limit."""

    def __init__(self, root, model, limit, concurrency=128):
        self.root, self.model = root, model
        self.ledger = Ledger(root / "billing.json", model, limit)
        self.pool = asyncio.Semaphore(concurrency)
        self.service = tinker.ServiceClient(project_id=os.environ.get("TINKER_PROJECT_ID"))

    async def sample(self, sampler, tokens, n, cap, stops, top_p=0.95, temperature=1.0):
        """n samples, at temperature 1 unless given (one unseeded call: seeded calls return identical copies)."""
        async with self.pool:
            held = self.ledger.hold(prefill=len(tokens) * n, sample=cap * n)
            try:
                out = await sampler.sample_async(
                    prompt=tinker.ModelInput.from_ints(tokens), num_samples=n,
                    sampling_params=tinker.SamplingParams(max_tokens=cap, stop=stops, temperature=temperature,
                                                          top_k=-1, top_p=top_p))
            except Exception:
                self.ledger.settle(held)
                raise
            sequences = [list(s.tokens) for s in out.sequences]
            self.ledger.settle(held, prefill=len(tokens) * n, sample=sum(map(len, sequences)))
        return sequences

    async def score(self, sampler, prompt, response):
        """Teacher-forced log-probability of each response token, and the top-1 token and its
        log-probability at each response position."""
        tokens = prompt + response
        async with self.pool:
            held = self.ledger.hold(prefill=len(tokens), sample=1)
            try:
                out = await sampler.sample_async(
                    prompt=tinker.ModelInput.from_ints(tokens), num_samples=1,
                    sampling_params=tinker.SamplingParams(max_tokens=1),
                    include_prompt_logprobs=True, topk_prompt_logprobs=1)
            except Exception:
                self.ledger.settle(held)
                raise
            self.ledger.settle(held, prefill=len(tokens), sample=1)
        n, top = len(prompt), out.topk_prompt_logprobs_np
        return (np.asarray(out.prompt_logprobs_np[n:], np.float32), np.asarray(top.token_ids[n:, 0], np.int32),
                np.asarray(top.logprobs[n:, 0], np.float32))

    async def score_set(self, sampler, texts, out, length):
        """Scores of [{task_id, prompt, response}] padded to `length` positions, saved as .npz."""
        if out.exists():
            return
        results = await asyncio.gather(*(self.score(sampler, t["prompt"], t["response"]) for t in texts))
        shape = (len(texts), length)
        arrays = dict(lp=np.full(shape, np.nan, np.float32), top=np.full(shape, -1, np.int32),
                      top_lp=np.full(shape, np.nan, np.float32), tokens=np.full(shape, -1, np.int32))
        for i, (text, (lp, top, top_lp)) in enumerate(zip(texts, results)):
            k = len(text["response"])
            arrays["lp"][i, :k], arrays["top"][i, :k], arrays["top_lp"][i, :k] = lp, top, top_lp
            arrays["tokens"][i, :k] = text["response"]
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, task_ids=np.array([t["task_id"] for t in texts]), **arrays)
        print(f"{stamp()} scored {out.parent.name}/{out.name} spent=${self.ledger.total():.2f}", flush=True)

    async def sampler(self, state_path):
        model = await self.service.create_training_client_from_state_async(state_path)
        return await model.save_weights_and_get_sampling_client_async()

    async def fresh_lora(self, rank=32, seed=20260924):
        return await self.service.create_lora_training_client_async(
            base_model=self.model, rank=rank, seed=seed, train_mlp=True, train_attn=True, train_unembed=True)

    async def fit(self, model, datums, lr, updates, log, extra=()):
        """`updates` Adam steps on batches of 32 datums taken in order and cycled; with `extra`, each
        batch also holds the next 32 datums of that stream, each seen once. `log(update)` names each
        update's record. Yields each update."""
        for update in range(1, updates + 1):
            batch = [datums[((update - 1) * BATCH + j) % len(datums)] for j in range(BATCH)]
            batch += list(extra[(update - 1) * BATCH:update * BATCH])
            tokens = sum(int(d.model_input.length) for d in batch)
            held = self.ledger.hold(train=tokens)
            forward = await model.forward_backward_async(batch, "cross_entropy")
            step = await model.optim_step_async(tinker.AdamParams(learning_rate=lr, **ADAM))
            result, _ = await asyncio.gather(forward.result_async(), step.result_async())
            self.ledger.settle(held, train=tokens)
            write(log(update), {"update": update, "batch": len(batch), "train_tokens": tokens,
                                "metrics": dict(result.metrics)})
            yield update

    async def save(self, model, name, ttl_days=None):
        """Save the training state; with no lifetime it never expires."""
        future = await model.save_state_async(name, ttl_seconds=ttl_days and ttl_days * 86400)
        return (await future.result_async()).path


def raw_datums(rows, max_length, weight=1.0):
    """Datums that predict exactly the sampled response tokens after each prompt, no rendering; every
    response token weighs `weight` / (the mean response length of the whole set)."""
    mean = fsum(len(r["response"]) for r in rows) / len(rows)
    datums = []
    for r in rows:
        prompt, response = list(r["prompt"]), list(r["response"])
        assert prompt and len(prompt) + len(response) - 1 <= max_length, "the datum would be truncated"
        datums.append(tinker.Datum(model_input=tinker.ModelInput.from_ints((prompt + response)[:-1]),
                                   loss_fn_inputs={"target_tokens": [0] * (len(prompt) - 1) + response,
                                                   "weights": [0.0] * (len(prompt) - 1)
                                                   + [weight / mean] * len(response)}))
    return datums


def answer_datums(prompts, targets):
    """Datums for (prompt tokens, target tokens) pairs with each example's weights summing to 1."""
    return [tinker.Datum(model_input=tinker.ModelInput.from_ints((p + t)[:-1]),
                         loss_fn_inputs={"target_tokens": [0] * (len(p) - 1) + t,
                                         "weights": [0.0] * (len(p) - 1) + [1.0 / len(t)] * len(t)})
            for p, t in zip(prompts, targets)]
