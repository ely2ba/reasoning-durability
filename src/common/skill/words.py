"""Sorting `size` words under a custom alphabet (part SK of records/revision.md).

`solutions` writes 12 worked responses per problem, each opening with the letters' ranks:
- insertion sort, taking the words in the given or the reversed order and placing each by
  comparing it with the sorted list from the front or from the back (4);
- selection sort, taking the smallest remaining word each time, or the largest and filling the
  list from the end (2);
- merge sort, splitting into halves or dealing alternately, or merging neighbours bottom-up (3);
- grouping by first letter, walking the alphabet or the words, ties split by the next letter (2);
- rewriting each word as its letter ranks and ordering the codes (1).
"""

import hashlib
import random
import re
from dataclasses import dataclass

from . import CLOSE, last_boxed

NAME = "words"
INSTRUCTION = "Solve the following problem. Put your final answer in \\boxed{}.\n\n"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class Problem:
    alphabet: str
    words: tuple[str, ...]

    @property
    def rank(self):
        return {c: i + 1 for i, c in enumerate(self.alphabet)}

    @property
    def task_id(self):
        return "words-" + hashlib.sha256(f"{self.alphabet}:{','.join(self.words)}".encode()).hexdigest()[:16]

    @property
    def key(self):  # the words' order in the prompt does not change the problem
        return self.alphabet, frozenset(self.words)

    @property
    def order(self):
        rank = self.rank
        return sorted(self.words, key=lambda w: [rank[c] for c in w])

    @property
    def answer(self):
        return ", ".join(self.order)

    @property
    def prompt(self):
        return (INSTRUCTION + f"A custom alphabet orders the 26 letters as follows: {' '.join(self.alphabet)}. "
                f"Sort these {len(self.words)} words in increasing order under this alphabet, comparing them "
                f"letter by letter (a word that is a prefix of another comes first): {', '.join(self.words)}. "
                "Give the sorted words separated by commas.")


def generate(rng, size):
    """A uniform random alphabet and `size` distinct words of 3 to 7 uniform random letters."""
    alphabet, words = "".join(rng.sample(LETTERS, 26)), []
    while len(words) < size:
        word = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 7)))
        if word not in words:
            words.append(word)
    return Problem(alphabet, tuple(words))


def verify(text, problem):
    """True if the last \\boxed{} lists exactly the sorted words, separated by commas or spaces."""
    content = last_boxed(text)
    if content is None:
        return False
    content = re.sub(r"\\text\{([^{}]*)\}", r"\1", content)
    return re.sub(r"\\[,;:! ]", " ", content).replace(",", " ").split() == problem.order


def compare(rank, u, v):
    """Whether u comes before v, and why."""
    for i, (p, q) in enumerate(zip(u, v)):
        if p != q:
            sign = "<" if rank[p] < rank[q] else ">"
            return sign == "<", f"{u} {sign} {v} (letter {i + 1}: {p}={rank[p]} {sign} {q}={rank[q]})"
    sign = "<" if len(u) < len(v) else ">"
    return sign == "<", f"{u} {sign} {v} ({min(u, v, key=len)} is a prefix)"


def insertion(problem, reverse, from_front):
    rank, done = problem.rank, []
    words = problem.words[::-1] if reverse else problem.words
    lines = [f"Insertion sort: take the words in {'reverse' if reverse else 'the given'} order and place each "
             f"by comparing it with the sorted list from the {'front' if from_front else 'back'}."]
    for word in words:
        notes, pos = [], 0 if from_front else len(done)
        while pos < len(done) if from_front else pos > 0:
            first, note = compare(rank, word, done[pos] if from_front else done[pos - 1])
            notes.append(note)
            if first == from_front:  # before the first larger word, or after the last smaller one
                break
            pos += 1 if from_front else -1
        done.insert(pos, word)
        lines.append(f"{word}: " + ("; ".join(notes) + " → " if notes else "") + ", ".join(done))
    return lines, done


def selection(problem, smallest):
    rank, left, picked = problem.rank, list(problem.words), []
    lines = ["Selection sort: take the smallest remaining word each time." if smallest else
             "Selection sort: take the largest remaining word each time and fill the list from the end."]
    while left:
        notes, candidates, i = [], left, 0
        while len(candidates) > 1:
            ended = [w for w in candidates if len(w) == i]  # a prefix of every other candidate
            if ended:
                notes.append(f"{ended[0]} ends, a prefix of the others")
                candidates = ended if smallest else [w for w in candidates if len(w) > i]
                continue
            best = (min if smallest else max)(rank[w[i]] for w in candidates)
            notes.append(f"letter {i + 1}: " + ", ".join(f"{w} {w[i]}={rank[w[i]]}" for w in candidates))
            candidates, i = [w for w in candidates if rank[w[i]] == best], i + 1
        picked.append(candidates[0])
        left.remove(candidates[0])
        position = len(picked) if smallest else len(problem.words) + 1 - len(picked)
        lines.append(f"Position {position}: " + ("; ".join(notes) if notes else "only one left") + f" → {picked[-1]}")
    return lines, picked if smallest else picked[::-1]


def merge(rank, left, right, lines):
    out, notes, i, j = [], [], 0, 0
    while i < len(left) and j < len(right):
        first, note = compare(rank, left[i], right[j])
        notes.append(note)
        out.append(left[i] if first else right[j])
        i, j = (i + 1, j) if first else (i, j + 1)
    out += left[i:] + right[j:]
    lines.append(f"Merge [{', '.join(left)}] and [{', '.join(right)}]: " + "; ".join(notes) + f" → [{', '.join(out)}]")
    return out


def merge_sort(rank, words, lines, dealt):
    if len(words) < 2:
        return list(words)
    half = len(words) // 2
    left, right = (words[::2], words[1::2]) if dealt else (words[:half], words[half:])
    return merge(rank, merge_sort(rank, left, lines, dealt), merge_sort(rank, right, lines, dealt), lines)


def merges(problem, how):
    rank, lines = problem.rank, [{
        "halves": "Merge sort: split the list into halves until single words remain, then merge back.",
        "dealt": "Merge sort: deal the list alternately into two piles until single words remain, then merge back.",
        "bottom-up": "Merge sort from the bottom: merge neighbouring words into pairs, then neighbouring pairs, "
                     "and so on."}[how]]
    if how != "bottom-up":
        return lines, merge_sort(rank, list(problem.words), lines, how == "dealt")
    runs = [[w] for w in problem.words]
    while len(runs) > 1:
        runs = [merge(rank, *runs[k:k + 2], lines) if k + 1 < len(runs) else runs[k] for k in range(0, len(runs), 2)]
    return lines, runs[0]


def group(rank, words, i, lines, by_alphabet):
    """`words`, which share their first i letters, ordered by letter i + 1 and then the next ones."""
    ended = [w for w in words if len(w) == i]
    groups = {}
    for w in words:
        if len(w) > i:
            groups.setdefault(w[i], []).append(w)
    letters = sorted(groups, key=rank.get)
    listed = "; ".join(f"{c}={rank[c]}: {', '.join(groups[c])}" for c in letters)
    head = (f"{', '.join(ended)} ends first; " if ended else "")
    if by_alphabet:
        where = f" of the words starting {words[0][:i]}" if i else ""
        lines.append(f"Letter {i + 1}{where}, walking the alphabet: {head}{listed}")
    else:
        ranks = ", ".join(f"{w} {w[i]}={rank[w[i]]}" for w in words if len(w) > i)
        lines.append(f"Letter {i + 1} of each word: {ranks}. {head}Grouped by rank: {listed}")
    out = ended
    for c in letters:
        out = out + (groups[c] if len(groups[c]) == 1 else group(rank, groups[c], i + 1, lines, by_alphabet))
    return out


def grouping(problem, by_alphabet):
    lines = ["Group the words by their first letter in the custom order, then split each tie by the next letter."]
    return lines, group(problem.rank, list(problem.words), 0, lines, by_alphabet)


def codes(problem):
    rank = problem.rank
    code = {w: ".".join(f"{rank[c]:02d}" for c in w) for w in problem.words}
    order = sorted(problem.words, key=code.get)  # two-digit ranks: the codes order as strings
    return ["Rewrite each word as its letter ranks (two digits each); the words then order as their codes do, "
            "number by number, a shorter code first when it is a prefix.",
            "; ".join(f"{w} = {code[w]}" for w in problem.words) + ".",
            "Codes in increasing order: " + ", ".join(f"{code[w]} ({w})" for w in order) + "."], order


def solutions(problem):
    """The 12 worked responses, in an order shuffled per problem (seeded by its id)."""
    methods = [insertion(problem, reverse, front) for reverse in (False, True) for front in (True, False)]
    methods += [selection(problem, smallest) for smallest in (True, False)]
    methods += [merges(problem, how) for how in ("halves", "dealt", "bottom-up")]
    methods += [grouping(problem, by_alphabet) for by_alphabet in (True, False)]
    methods.append(codes(problem))
    table = "Ranks in the custom alphabet: " + ", ".join(f"{c}={r}" for c, r in problem.rank.items()) + "."
    texts = list(dict.fromkeys("\n".join([table, *lines, f"Sorted: {', '.join(order)}."]) + CLOSE
                               + f"\\boxed{{{', '.join(order)}}}" for lines, order in methods))
    random.Random(f"skill-20260926:{problem.task_id}").shuffle(texts)
    return texts
