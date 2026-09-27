"""Synthetic skills for part SK of records/revision.md: multiplying two base-7
numbers (`base7`) and sorting words under a custom alphabet (`words`).

Each task module has a seeded generator (`generate`), an exact verifier that reads the last
\\boxed{} of a response (`verify`), and a solver (`solutions`) that writes several worked
responses per problem in the forced-reasoning format: reasoning, then CLOSE, then a boxed
answer. The solver computes every displayed step and reads its final answer from those steps,
so a wrong step cannot hide behind a right answer; its methods are shuffled per problem, so a
problem's first response is a random method.
"""

import random

SPLITS = {"calibration": 300, "evaluation": 300, "train": 4480}
CLOSE = "\n</think>\n\n"


def last_boxed(text):
    """Content of the last \\boxed{...}, nested braces matched; None if absent or unclosed."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, begin = 1, start + len("\\boxed{")
    for end in range(begin, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return text[begin:end]
    return None


def splits(task, size):
    """{split: problems} for `task` (the module base7 or words) at `size`. The splits are
    consecutive slices of one seeded stream from which repeated problems are dropped, so
    calibration, evaluation and train problems are disjoint."""
    rng, seen, stream = random.Random(f"skill-20260926:{task.NAME}:{size}"), set(), []
    while len(stream) < sum(SPLITS.values()):
        problem = task.generate(rng, size)
        if problem.key not in seen:
            seen.add(problem.key)
            stream.append(problem)
    out, start = {}, 0
    for name, count in SPLITS.items():
        out[name], start = stream[start:start + count], start + count
    return out
