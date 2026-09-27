"""Multiplying two base-7 numbers of `size` digits (part SK of records/revision.md).

`solutions` writes 20 worked responses per problem (10 for a square, where swapping the
operands changes nothing), from three methods:
- long multiplication in base 7: either number on top, the lower number's digits taken from
  the right or from the left, and the partial products added at the end or kept as a running
  total (8);
- through base 10: Horner's rule or place values into base 10, the product by the decimal
  places of either factor, and repeated division or descending powers of 7 back (8);
- digit products collected by place and carried in base 7, in one pass or two, with either
  number first (4).
"""

import random
import re
from dataclasses import dataclass
from itertools import product

from . import CLOSE, last_boxed

NAME = "base7"
INSTRUCTION = "Solve the following math problem. Put your final answer in \\boxed{}.\n\n"


def to7(n):
    """The base-7 numeral of n >= 0."""
    out = ""
    while True:
        n, r = divmod(n, 7)
        out = str(r) + out
        if not n:
            return out


@dataclass(frozen=True)
class Problem:
    a: str
    b: str

    @property
    def task_id(self):
        return f"base7-{self.a}-{self.b}"

    @property
    def key(self):  # a × b and b × a are one problem
        return tuple(sorted((self.a, self.b)))

    @property
    def answer(self):
        return to7(int(self.a, 7) * int(self.b, 7))

    @property
    def prompt(self):
        return (INSTRUCTION + f"Multiply the base-7 numbers {self.a}_7 and {self.b}_7. "
                "Give the product in base 7.")


def generate(rng, size):
    """Two independent uniform base-7 numbers of `size` digits (no leading zero)."""
    low, high = 7 ** (size - 1), 7 ** size - 1
    return Problem(to7(rng.randint(low, high)), to7(rng.randint(low, high)))


def verify(text, problem):
    """True if the last \\boxed{} holds the product's base-7 digits exactly; spaces and a base
    subscript (_7, _{7}, (…)_7) are allowed."""
    content = last_boxed(text)
    if content is None:
        return False
    content = re.sub(r"\s|\\[,;:! ]", "", content)
    content = re.sub(r"_\{?\(?7\)?\}?$", "", content)
    if content.startswith("(") and content.endswith(")"):
        content = content[1:-1]
    return content == problem.answer


def step(value):
    """`value` written as a base-7 digit and a carry, the carry and the digit."""
    carry, digit = divmod(value, 7)
    return (f"{value} = {carry}·7+{digit} → {digit}, carry {carry}" if carry else f"{value} → {digit}"), carry, digit


def times_digit(x, d):
    """x × d (d from 2 to 6), digit by digit from the right, and the result."""
    parts, carry, out = [], 0, ""
    for c in reversed(x):
        text, carry_next, digit = step(int(c) * d + carry)
        parts.append(f"{c}×{d}{f'+{carry}' if carry else ''} = {text}")
        carry, out = carry_next, str(digit) + out
    if carry:
        parts.append(f"the last carry {carry} goes in front")
        out = str(carry) + out
    return "; ".join(parts) + f"; so {x} × {d} = {out}", out


def add_columns(numbers):
    """Base-7 numbers added place by place from the right, and the sum."""
    lines, carry, out = [], 0, ""
    for place in range(max(map(len, numbers))):
        column = [n[-1 - place] for n in numbers if place < len(n)]
        if len(column) == 1 and not carry:
            lines.append(f"place {place}: {column[0]}")
            out = column[0] + out
            continue
        terms = "+".join(column) + (f" + carry {carry}" if carry else "")
        text, carry, digit = step(sum(map(int, column)) + carry)
        lines.append(f"place {place}: {terms} = {text}")
        out = str(digit) + out
    if carry:
        lines.append(f"the last carry {carry} goes in front")
        out = str(carry) + out
    return lines, out


def long_multiplication(top, bottom, from_right, running):
    side = "from the right" if from_right else "from the left"
    lines = [f"Long multiplication in base 7 with {top} on top, taking the digits of {bottom} {side}. "
             "A digit product plus carry v is written v = q·7+r: keep r, carry q."]
    places = range(len(bottom)) if from_right else reversed(range(len(bottom)))
    partials, total = [], None
    for place in places:
        d = int(bottom[-1 - place])
        if d == 0:
            lines.append(f"Digit 0 at place {place} adds nothing.")
            continue
        text, digits = times_digit(top, d) if d > 1 else (f"{top} × 1 = {top}", top)
        partial = digits + "0" * place
        lines.append(f"Digit {d} at place {place}: {text}" + (f"; shifted {place}: {partial}." if place else "."))
        if running and partials:
            steps, new = add_columns([total, partial])
            lines.append(f"Running total {total} + {partial}: " + "; ".join(steps) + f"; total {new}.")
            total = new
        elif running:
            total = partial
        partials.append(partial)
    if not running:
        steps, total = add_columns(partials) if len(partials) > 1 else ([], partials[0])
        lines += ["Add the partial products place by place:", *steps] if steps else []
    lines.append(f"So {top} × {bottom} = {total} in base 7.")
    return lines, total


def to_ten(x, horner):
    """x (base 7) in base 10, by Horner's rule or by place values, and its value."""
    if horner:
        value, parts = int(x[0]), []
        for c in x[1:]:
            parts.append(f"{value}·7+{c} = {value * 7 + int(c)}")
            value = value * 7 + int(c)
        return f"{x}_7 by Horner's rule: {x[0]}; " + "; ".join(parts) + ".", value
    terms = [(int(c), 7 ** (len(x) - 1 - k)) for k, c in enumerate(x) if c != "0"]
    value = sum(d * p for d, p in terms)
    return (f"{x}_7 by place values: " + " + ".join(f"{d}·{p}" for d, p in terms) + " = "
            + " + ".join(str(d * p) for d, p in terms) + f" = {value}."), value


def decimal_product(x, y):
    """x × y in base 10 by the decimal places of y, and the product."""
    places = [int(c) * 10 ** (len(str(y)) - 1 - k) for k, c in enumerate(str(y)) if c != "0"]
    lines = [f"{x} × {y} by the decimal places of {y}: " + "; ".join(f"{x}×{p} = {x * p}" for p in places) + "."]
    total, sums = x * places[0], []
    for p in places[1:]:
        sums.append(f"{total} + {x * p} = {total + x * p}")
        total += x * p
    if sums:
        lines.append("Adding: " + "; ".join(sums) + ".")
    return lines, total


def from_ten(n, divide):
    """n back to base 7, by repeated division or by descending powers of 7, and its digits."""
    digits = ""
    if divide:
        lines = ["Back to base 7 by repeated division by 7:"]
        while n:
            lines.append(f"{n} = 7·{n // 7} + {n % 7}")
            n, digits = n // 7, str(n % 7) + digits
        return lines + [f"The remainders from last to first: {digits}."], digits
    top = 0
    while 7 ** (top + 1) <= n:
        top += 1
    lines = ["Back to base 7 by descending powers: " + ", ".join(f"7^{k} = {7 ** k}" for k in range(top, 0, -1)) + "."]
    for k in range(top, 0, -1):
        d, rest = divmod(n, 7 ** k)
        lines.append(f"{n} = {d}·{7 ** k} + {rest} → digit {d}")
        n, digits = rest, digits + str(d)
    lines.append(f"units digit {n}. Digits: {digits + str(n)}.")
    return lines, digits + str(n)


def through_ten(first, second, horner, divide):
    line_a, x = to_ten(first, horner)
    line_b, y = to_ten(second, horner)
    product_lines, n = decimal_product(x, y)
    back, digits = from_ten(n, divide)
    return (["Convert both numbers to base 10, multiply there, and convert back.", line_a, line_b,
             *product_lines, *back, f"So {first}_7 × {second}_7 = {digits}_7."], digits)


def by_place(first, second, two_pass):
    """Products of digits at places i and k − i summed for each place k, then carried."""
    x, y = first[::-1], second[::-1]
    sums = []
    for k in range(len(x) + len(y) - 1):
        terms = [(x[i], y[k - i]) for i in range(len(x)) if 0 <= k - i < len(y)]
        sums.append((" + ".join(f"{p}·{q}" for p, q in terms), sum(int(p) * int(q) for p, q in terms)))
    lines = [f"Multiply digit by digit and collect by place (places count from the right, from 0): place k "
             f"collects each digit of {first} at place i times the digit of {second} at place k−i. "
             "Then carry in base 7 from place 0 up.",
             f"Digits by place: {first} → {', '.join(x)}; {second} → {', '.join(y)}."]
    if two_pass:
        lines += [f"place {k}: {terms} = {s}" for k, (terms, s) in enumerate(sums)] + ["Now carry:"]
    carry, digits, k = 0, "", 0
    while k < len(sums) or carry:
        terms, s = sums[k] if k < len(sums) else ("", 0)
        text, carry_next, digit = step(s + carry)
        added = (f"{s} + carry {carry} = " if terms else f"carry {carry} = ") if carry else ""
        head = f"place {k}: " + (f"{terms} = " if terms and not two_pass else "")
        lines.append(head + added + text)
        carry, digits, k = carry_next, str(digit) + digits, k + 1
    lines.append(f"So {first} × {second} = {digits} in base 7.")
    return lines, digits


def solutions(problem):
    """The distinct worked responses, in an order shuffled per problem (seeded by its id)."""
    a, b = problem.a, problem.b
    methods = [long_multiplication(top, bottom, right, running)
               for (top, bottom), right, running in product(((a, b), (b, a)), (True, False), (False, True))]
    methods += [through_ten(first, second, horner, divide)
                for (first, second), horner, divide in product(((a, b), (b, a)), (True, False), (True, False))]
    methods += [by_place(first, second, two_pass) for (first, second), two_pass in product(((a, b), (b, a)), (False, True))]
    texts = list(dict.fromkeys("\n".join(lines) + CLOSE + f"{a}_7 × {b}_7 = {result}_7.\n\n\\boxed{{{result}}}"
                               for lines, result in methods))
    random.Random(f"skill-20260926:{problem.task_id}").shuffle(texts)
    return texts
