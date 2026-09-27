"""TCES: exact verification of arithmetic-search answers, and the probes' scoring rule.

A task gives operands, a target (an exact rational), the allowed operators and
limits. A completion is correct when it holds exactly one `<answer>...</answer>`
pair whose expression uses each operand exactly once, only allowed binary
operators (no unary signs), stays within the limits, and equals the target in
exact arithmetic. This is a lean port of DuraSeed-v1's verifier
(`src/duraseed/tasks/tces/`); `tests/test_tces_verifier.py` replays every stored
completion through both.

The probes' scoring rule: a forced search that restates the answer template
(`<answer>EXPRESSION</answer>`, a tag without digits) is not answering, so such
tags are removed before verifying, and generation that stopped on one was
resumed (the resumed score); the strict score counts those responses as wrong.
"""

import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

TEMPLATE_TAG = re.compile(r"<answer>([^\d]*?)</answer>", re.S)
WHITESPACE, OPERATORS = " \t\n\r\f\v", "+-*/"
LIMITS = {"max_abs_intermediate": 10_000, "max_denominator": 1_000, "max_tree_depth": 5,
          "max_ast_nodes": 31, "max_answer_length": 1_024}


def tokens(text, max_digits, max_tokens):
    """ASCII integers, operators and parentheses; None on any other character or a limit."""
    if any(ord(c) > 0x7F for c in text):
        return None
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c in WHITESPACE:
            i += 1
            continue
        if "0" <= c <= "9":
            j = i
            while j < len(text) and "0" <= text[j] <= "9":
                j += 1
            if j - i > max_digits:
                return None
            out.append(int(text[i:j]))
            i = j
        elif c in OPERATORS or c in "()":
            out.append(c)
            i += 1
        else:
            return None
        if len(out) > max_tokens:
            return None
    return out


def parse(toks, max_nodes, max_parens=64):
    """expression := product (+|- product)*; product := primary (*|/ primary)*;
    primary := INTEGER | ( expression ). Returns a tree of ints and (op, left, right), or None."""
    pos, nodes = 0, 0

    def node(tree):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise ValueError
        return tree

    def primary(depth):
        nonlocal pos
        tok = toks[pos] if pos < len(toks) else None
        if isinstance(tok, int):
            pos += 1
            return node(tok)
        if tok == "(" and depth < max_parens:
            pos += 1
            tree = expression(depth + 1)
            if pos >= len(toks) or toks[pos] != ")":
                raise ValueError
            pos += 1
            return tree
        raise ValueError  # unary sign, empty, or unexpected token

    def chain(sub, ops, depth):
        nonlocal pos
        tree = sub(depth)
        while pos < len(toks) and toks[pos] in ops:
            op = toks[pos]
            pos += 1
            right = sub(depth)
            tree = node((op, tree, right))
        return tree

    def product(depth):
        return chain(primary, ("*", "/"), depth)

    def expression(depth):
        return chain(product, ("+", "-"), depth)

    try:
        tree = expression(0)
    except (ValueError, IndexError):
        return None
    return tree if pos == len(toks) else None


def depth(tree):
    return 1 if isinstance(tree, int) else 1 + max(depth(tree[1]), depth(tree[2]))


def leaves(tree):
    return [tree] if isinstance(tree, int) else leaves(tree[1]) + leaves(tree[2])


def ops(tree):
    return [] if isinstance(tree, int) else [tree[0], *ops(tree[1]), *ops(tree[2])]


def evaluate(tree, max_abs, max_den):
    if isinstance(tree, int):
        return Fraction(tree)
    left, right = evaluate(tree[1], max_abs, max_den), evaluate(tree[2], max_abs, max_den)
    if left is None or right is None or (tree[0] == "/" and right == 0):
        return None
    value = {"+": left + right, "-": left - right, "*": left * right}.get(tree[0]) if tree[0] != "/" else left / right
    return None if abs(value) > max_abs or value.denominator > max_den else value


def correct(completion, task):
    """True when the completion solves the task under DuraSeed-v1's exact rules."""
    limits = {**LIMITS, **task.get("constraints", {})}
    if completion.count("<answer>") != 1 or completion.count("</answer>") != 1:
        return False
    start, end = completion.find("<answer>") + len("<answer>"), completion.find("</answer>")
    span = completion[start:end]
    if end < start or len(span) > limits["max_answer_length"] or not span.strip():
        return False
    toks = tokens(span, min(limits["max_answer_length"], 1_024), limits["max_answer_length"] + 1)
    tree = parse(toks, limits["max_ast_nodes"]) if toks else None
    if tree is None or depth(tree) > limits["max_tree_depth"]:
        return False
    if Counter(leaves(tree)) != Counter(task["operands"]) or not set(ops(tree)) <= set(task["allowed_ops"]):
        return False
    value = evaluate(tree, limits["max_abs_intermediate"], limits["max_denominator"])
    return value is not None and value == Fraction(task["target"][0], task["target"][1])


def score(text, task):
    """The probes' rule: restated templates removed, then exact verification."""
    return correct(TEMPLATE_TAG.sub("", text), task)


RUN = re.compile(r"[0-9(][0-9+\-*/() ]*[0-9)]")


def found(text, task):
    """True when any arithmetic span of the text, whole or cut at an opening parenthesis, solves the
    task: the search wrote a solution down, whether or not it answered with it."""
    for match in RUN.finditer(text):
        span = match.group(0)
        spans = {span, span.strip("() ")} | {span[i:] for i, c in enumerate(span) if c == "("}
        if any(correct(f"<answer>{s}</answer>", task) for s in spans):
            return True
    return False


def template_stop(text):
    """True when generation stopped on a restated template (a tag without digits); such a stop is
    resumed from the model's own tokens. The final token may carry up to three trailing characters."""
    close = text.rfind("</answer>")
    start = text.rfind("<answer>", 0, max(close, 0))
    tail = text[close + 9:]
    return (close >= 0 and start >= 0 and len(tail) <= 3 and not re.search(r"\w", tail)
            and not re.search(r"\d", text[start + 8:close]))


def panel(repo):
    """The 384 monitor items (192 targeted, 192 held-out) with prompt tokens and task specs."""
    return [json.loads(line) for line in (Path(repo) / "data/tces/panel.jsonl").open()]
