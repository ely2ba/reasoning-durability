"""Properties of the synthetic skills of part SK (records/revision.md; src/common/skill): exact answers, a
verifier that accepts the answer's written forms and rejects near misses, worked solutions that all verify and
differ in their steps, seeded determinism, disjoint splits, and the problems data/skill_ids.json froze. No data
outside this repository."""

import json
import random
import re
from itertools import combinations

from common.skill import CLOSE, SPLITS, base7, splits, words
from conftest import REPO

SETTINGS = ((base7, 5), (base7, 6), (words, 12), (words, 16))
SAMPLE = 40


def trace(task, problem, text):
    """The numbers (and for words, the problem's words) of a response's reasoning, in order. Two
    responses with the same trace would differ only in wording."""
    tokens = re.findall(r"[a-z]+|\d+", text.split(CLOSE)[0])
    return tuple(t for t in tokens if t.isdigit() or (task is words and t in problem.words))


def accepted(task, answer):
    """Ways of writing the right answer that the verifier must accept."""
    texts = [f"\\boxed{{{answer}}}", f"First \\boxed{{0}}, corrected: \\boxed{{{answer}}}."]
    if task is base7:
        return texts + [f"\\boxed{{{answer}_7}}", f"\\boxed{{{answer}_{{7}}}}", f"$\\boxed{{({answer})_7}}$"]
    return texts + [f"\\boxed{{{answer.replace(', ', ',')}}}", f"\\boxed{{{answer.replace(',', '')}}}",
                    f"\\boxed{{\\text{{{answer}}}}}"]


def perturbed(task, problem):
    """Wrong answers close to the right one."""
    if task is base7:
        answer = problem.answer
        wrong = {answer[1:], answer + "0", answer[::-1], str(int(answer, 7))}
        wrong |= {answer[:i] + d + answer[i + 1:] for i in range(len(answer)) for d in "0123456"}
        wrong |= {answer[:i] + answer[i + 1] + answer[i] + answer[i + 2:] for i in range(len(answer) - 1)}
        return wrong - {answer}
    order = problem.order
    lists = [order[:i] + [order[i + 1], order[i]] + order[i + 2:] for i in range(len(order) - 1)]
    lists += [order[1:], order + order[:1], order[::-1], sorted(order), list(problem.words)]
    return {", ".join(x) for x in lists} - {problem.answer}


def edge_cases():
    """A square, and words that are prefixes of each other."""
    alphabet = "".join(random.Random(0).sample(words.LETTERS, 26))
    return [(base7, base7.Problem("34512", "34512")), (base7, base7.Problem("100000", "600001")),
            (words, words.Problem(alphabet, ("abc", "abcde", "abd", "zab", "ab", "abcd", "zzz", "qab")))]


def test_true_answers_are_exact():
    rng = random.Random(1)
    for n in [0, 6, 7, 48, 49] + [rng.randrange(7 ** 12) for _ in range(500)]:
        assert int(base7.to7(n), 7) == n
    for task, size in SETTINGS:
        for problem in splits(task, size)["evaluation"][:SAMPLE]:
            if task is base7:
                assert int(problem.answer, 7) == int(problem.a, 7) * int(problem.b, 7)
                assert len(problem.a) == len(problem.b) == size and "0" not in problem.a[0] + problem.b[0]
            else:
                standard = str.maketrans(problem.alphabet, words.LETTERS)
                assert problem.order == sorted(problem.words, key=lambda w: w.translate(standard))
                assert sorted(problem.alphabet) == list(words.LETTERS)
                assert len(set(problem.words)) == size


def test_verifier_accepts_the_answer_and_rejects_perturbed_ones():
    for task, size in SETTINGS:
        for problem in splits(task, size)["evaluation"][:SAMPLE]:
            for text in accepted(task, problem.answer):
                assert task.verify(text, problem), (problem.task_id, text)
            assert not task.verify(problem.answer, problem), problem.task_id  # no box
            assert not task.verify(f"\\boxed{{{problem.answer}", problem), problem.task_id  # unclosed
            for wrong in perturbed(task, problem):
                assert not task.verify(f"\\boxed{{{wrong}}}", problem), (problem.task_id, wrong)
                assert not task.verify(f"\\boxed{{{problem.answer}}} or \\boxed{{{wrong}}}", problem), wrong


def test_every_solution_verifies_and_the_solutions_differ_in_their_steps():
    cases = [(task, p) for task, size in SETTINGS for p in splits(task, size)["train"][:SAMPLE]] + edge_cases()
    for task, problem in cases:
        texts = task.solutions(problem)
        assert len(texts) >= 8, problem.task_id
        assert len({trace(task, problem, t) for t in texts}) == len(texts), problem.task_id
        for text in texts:
            assert text.count(CLOSE) == 1, problem.task_id
            assert text.endswith("}") and task.verify(text, problem), text[-300:]


def test_generation_and_solutions_are_deterministic_per_seed():
    for task, size in SETTINGS:
        parts = splits(task, size)
        assert parts == splits(task, size)
        assert task.solutions(parts["train"][0]) == task.solutions(parts["train"][0])
    assert splits(base7, 5)["train"][:10] != splits(base7, 6)["train"][:10]


def test_calibration_evaluation_and_train_problems_are_disjoint():
    for task, size in SETTINGS:
        parts = splits(task, size)
        assert {name: len(problems) for name, problems in parts.items()} == SPLITS
        keys = {name: {p.key for p in problems} for name, problems in parts.items()}
        assert sum(map(len, keys.values())) == sum(SPLITS.values())  # no repeats within a split
        for first, second in combinations(keys, 2):
            assert not keys[first] & keys[second], (first, second)
        ids = [p.task_id for problems in parts.values() for p in problems]
        assert len(set(ids)) == len(ids)


def test_the_generator_rebuilds_the_frozen_problems():
    """data/skill_ids.json: every setting's calibration problems, and base7-5's evaluation and train problems (a
    change in Python's random would show here)."""
    frozen = json.loads((REPO / "data" / "skill_ids.json").read_text())
    for task, size in SETTINGS:
        name = f"{task.NAME}-{size}"
        assert [p.task_id for p in splits(task, size)["calibration"]] == frozen["calibration"][name], name
    parts = splits(base7, 5)
    assert frozen["chosen"] == "base7-5" and [p.task_id for p in parts["evaluation"]] == frozen["evaluation"]
    assert [p.task_id for p in parts["train"]] == frozen["train"]
