"""Competition math: the exact integer scorer and the prompt instruction.

Scoring takes the last \\boxed{...} (nested and escaped braces handled; an unclosed last box means
no answer), normalizes units, degrees, currency, percent, thousands separators and `x = 5`, then
parses an integer (DuraSeed-v1's tools/math_score.py, unchanged).
"""

import re

_BOXED = re.compile(r"\\boxed\s*\{")
_TEXT = re.compile(
    r"\\(?:text|textbf|textit|textrm|textnormal|mathrm|mathbf|mathit|mbox|operatorname)"
    r"\s*\{([^{}]*)\}"
)
_DROP = re.compile(r"\\[$%]|[$%°]|\^\s*\{?\s*\\circ\s*\}?|\\displaystyle")
_SPACE = re.compile(r"\\[,!;: ]|~")
_NAME = re.compile(r"\s*[A-Za-z]{1,3}(?:_\{?[A-Za-z0-9]+\}?)?\s*")
_INT = re.compile(r"([+-]?)\s*(\d{1,3}(?:,\d{3})+|\d{1,3}(?: \d{3})+|\d+)(?:\.0*)?")


def last_boxed(text):
    """Content of the last \\boxed{...}, or None if absent or unclosed."""
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    depth, i = 1, starts[-1]
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[starts[-1]:i]
        i += 1
    return None


def _unwrap(m):
    return m.group(1) if re.search(r"\d", m.group(1)) else "#"


def parse_int(s):
    """Normalized integer value of a boxed answer string, or None."""
    s = _DROP.sub("", s.replace("\u2212", "-"))
    while True:
        t = _TEXT.sub(_unwrap, s)
        if t == s:
            break
        s = t
    s = _SPACE.sub(" ", s.replace("{,}", ","))
    if s.count("=") == 1 and _NAME.fullmatch(s.split("=")[0]):
        s = s.split("=")[1]
    s = s.strip(" #\t\n")
    if s.endswith("."):
        s = s[:-1].rstrip()
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    m = _INT.fullmatch(s)
    if not m:
        return None
    value = int(re.sub(r"[, ]", "", m.group(2)))
    return -value if m.group(1) == "-" else value


def extract(text):
    content = last_boxed(text)
    return None if content is None else parse_int(content)


INSTRUCTION = "Solve the following math problem. Put your final answer in \\boxed{}.\n\n"
