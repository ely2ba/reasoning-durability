"""The README's explainer: the study in 21 slides, as one animated GIF (assets/explainer.gif).

Written for readers new to the paper, in plain words: the question, the experiment, the break, its cause, what is
lost, the early warning, how general it is and what prevents it. Five slides move: the three models' training
solutions as cards, and the models' own answers to one held-out problem before and after the intense round of chat
training (the instruction stress test) and after five updates of reasoning training. The answers are the models'
first forced responses (draw 0) to MATH-500 problem test/number_theory/89, copied verbatim from the Hugging Face
dataset into assets/explainer-responses.json and checked against outputs/math_samples.csv.gz. Every number comes
from the registry (outputs/summaries/numbers.json), or from outputs/summaries/math.json for the two the paper's text
does not use. The first frame is the summary slide, because GitHub shows the first frame when a GIF is paused or
motion is reduced; a bar at the top shows each slide's time, and no text changes while something moves.
"""

import colorsys
import json
import math
import random
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT, RESPONSES = ROOT / "assets/explainer.gif", ROOT / "assets/explainer-responses.json"
W, H, S = 800, 450, 2  # layout in points, drawn at S pixels a point so that text stays sharp on dense screens
BG, PANE, BORDER, TEXT, SUB, DIM = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#9198a1", "#6e7681"
RED, GREY, BLUE, GREEN, AMBER, WRONG, PURPLE = "#e5534b", "#c9d1d9", "#4c8fe0", "#57ab5a", "#d4a72c", "#f47067", "#a371f7"
LINE, GAP, WRAP = 12.5, 5, 56  # answer text: line height, blank line height, characters a line
FPS, FAST_FPS, RATE = 10, 4, 80  # frames a second while writing and while fast-forwarding; tokens a second
LONGEST = 2200  # tokens that fill the answer-length bar
PERCENTS = ("m.h.base", "m.h.D140", "m.h.O140", "mm.P.h", "rs.D.h", "rs.O.h", "rs.P.h", "rs.D.a", "rs.O.a", "rs.P.a",
            "m.it.D140", "m.it.O140", "mm.P.it", "m.b.D140", "m.b.O140", "mm.P.b", "rl.D.it.u5", "rl.O.it.u5")


def font(size, weight="Regular"):
    """Avenir Next, as in the README's figures; matplotlib's DejaVu Sans where it is not installed."""
    path = Path("/System/Library/Fonts/Avenir Next.ttc")
    if path.exists():
        index = {"Bold": 0, "Demi Bold": 2, "Italic": 4, "Medium": 5, "Regular": 7}[weight]
        return ImageFont.truetype(str(path), round(size * S), index=index)
    return fallback("DejaVuSans-Bold.ttf" if "Bold" in weight else "DejaVuSans.ttf", size - 1)


def mono(size, bold=False):
    """Menlo, or DejaVu Sans Mono."""
    path = Path("/System/Library/Fonts/Menlo.ttc")
    if path.exists():
        return ImageFont.truetype(str(path), round(size * S), index=int(bold))
    return fallback("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf", size)


def fallback(name, size):
    import matplotlib
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf" / name), round(size * S))


OVER, TITLE, BODY, SMALL, PROSE = font(10.5, "Demi Bold"), font(21, "Demi Bold"), font(13, "Medium"), font(11), font(12.5)
NAME, CHIP, BIG, HUGE, ITALIC = (font(14, "Demi Bold"), font(12.5, "Demi Bold"), font(26, "Bold"), font(34, "Bold"),
                                 font(12.5, "Italic"))
TEXTF, LABEL = font(14), font(12, "Demi Bold")
CODE, CODE_BOLD = mono(10), mono(10, bold=True)


def P(*values):
    """Points to pixels."""
    return [round(v * S) for v in values]


def mix(a, b, t):
    """The colour t of the way from a to b."""
    a, b = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (a, b))
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def load():
    """The problem, the five responses (checked against outputs/), and the numbers: N as printed (accuracies in whole
    percents), V as values."""
    import pandas as pd
    doc = json.loads(RESPONSES.read_text())
    table = pd.read_csv(ROOT / "outputs/math_samples.csv.gz", low_memory=False,
                        usecols=["run", "bench", "label", "when", "mode", "rule", "task_id", "draw", "tokens", "correct"])
    table = table[(table.bench == "math500") & (table["mode"] == "forced") & (table.rule == "first")
                  & (table.task_id == doc["task_id"])]
    responses = {}
    for r in doc["responses"]:
        state = table[(table.run == r["run"]) & (table.label == r["label"]) & (table["when"] == r["when"])]
        row = state[state.draw == r["draw"]]
        assert row[["tokens", "correct"]].values.tolist() == [[r["tokens"], int(r["correct"])]], r["label"]
        responses[r["label"]] = {**r, "tries": len(state), "right": int(state.correct.sum())}
    registry = json.loads((ROOT / "outputs/summaries/numbers.json").read_text())
    assert round(registry["d.pass.high"]["value"]) == 8  # "about 8 times"
    N = {k: v["text"] for k, v in registry.items()}
    V = {k: v["value"] for k, v in registry.items()}
    N.update({k: f"{round(V[k])}%" for k in PERCENTS})
    failure = json.loads((ROOT / "outputs/summaries/math.json").read_text())["pilot-20260924"]["failure_form"]
    V["o.ends"], V["o.median"] = failure["O-u140+it|u20"]["ends_turn"], failure["O-u140+it|u20"]["median_tokens"]
    for key, B in (("D", V["sh.B.D"]), ("SH", V["sh.B"]), ("O", V["sh.B.O"]), ("P", V["mm.P.B"])):
        V[f"drop.{key}"] = 100 * (1 - math.exp(-B))  # how much less likely, per token, the base model's solutions are
    return doc, responses, N, V


# ---------------------------------------------------------------- drawing pieces

def text(d, x, y, words, face=TEXTF, fill=SUB, width=752, gap=6):
    """Words wrapped to width points; returns the y below them."""
    for para in words.split("\n"):
        line = ""
        for word in para.split():
            if line and face.getlength(f"{line} {word}") / S > width:
                d.text(P(x, y), line, font=face, fill=fill)
                y, line = y + face.size / S * 1.45, word
            else:
                line = f"{line} {word}".strip()
        d.text(P(x, y), line, font=face, fill=fill)
        y += face.size / S * 1.45 + gap
    return y


def icon(d, x, y, ok, color):
    """A tick or a cross centred on (x, y); the text fonts lack both."""
    if ok:
        d.line(P(x - 4, y, x - 1.2, y + 3, x + 4, y - 3.2), fill=color, width=round(1.8 * S), joint="curve")
    else:
        d.line(P(x - 3.3, y - 3.3, x + 3.3, y + 3.3), fill=color, width=round(1.8 * S))
        d.line(P(x - 3.3, y + 3.3, x + 3.3, y - 3.3), fill=color, width=round(1.8 * S))


def arrow(d, x0, x1, y, color=DIM):
    d.line(P(x0, y, x1 - 5, y), fill=color, width=round(1.5 * S))
    d.polygon(P(x1, y, x1 - 7, y - 4, x1 - 7, y + 4), fill=color)


def box(d, x, y, w, h, title, words="", color=SUB, fill=PANE, right=0):
    """A rounded box with a coloured title and some words under it; right keeps room beside the title."""
    d.rounded_rectangle(P(x, y, x + w, y + h), radius=8 * S, fill=fill, outline=mix(fill, color, 0.45), width=S)
    below = text(d, x + 12, y + 10, title, LABEL, color, w - 24 - right, 2)
    if words:
        below = text(d, x + 12, below + 2, words, SMALL, SUB, w - 24, 3)
    assert below <= y + h, title  # the words fit the box


def bars(d, x, y, rows, top=100, width=360):
    """Horizontal bars, one row per (label, colour, value, printed value), on a scale from 0 to top."""
    for i, (label, color, value, printed) in enumerate(rows):
        yy = y + 30 * i
        d.text(P(x, yy + 9), label, font=LABEL, fill=color, anchor="lm")
        assert LABEL.getlength(label) / S < 164, label
        bx = x + 170
        d.rounded_rectangle(P(bx, yy + 3, bx + width, yy + 15), radius=3 * S, fill=PANE)
        d.rounded_rectangle(P(bx, yy + 3, bx + max(width * value / top, 3), yy + 15), radius=3 * S, fill=color)
        d.text(P(bx + width + 10, yy + 9), printed, font=LABEL, fill=TEXT, anchor="lm")


def verdict(d, right, y, answer, ok):
    """The final answer, marked right or wrong."""
    color, label = (GREEN, "correct") if ok else (WRONG, "wrong")
    x = right - (35 + CHIP.getlength(answer) / S + 5 + SMALL.getlength(label) / S)
    d.rounded_rectangle(P(x, y, right, y + 22), radius=11 * S, fill=mix(BG, color, 0.2))
    icon(d, x + 13, y + 11, ok, color)
    d.text(P(x + 24, y + 11), answer, font=CHIP, fill=color, anchor="lm")
    d.text(P(x + 29 + CHIP.getlength(answer) / S, y + 11.5), label, font=SMALL, fill=color, anchor="lm")


def strip(d, y, label, words, color):
    """A one-line box under the title: the problem, or the training in between."""
    d.rounded_rectangle(P(24, y, 776, y + 22), radius=6 * S, fill=mix(BG, color, 0.1) if color != DIM else PANE)
    d.text(P(36, y + 11.5), label, font=OVER, fill=color, anchor="lm")
    x = 50 + OVER.getlength(label) / S
    for i, part in enumerate(re.split(r"\$(\w)\$", words)):  # the problem's $k$ in italics
        face = ITALIC if i % 2 else PROSE
        d.text(P(x, y + 11.5), part, font=face, fill=TEXT if color != DIM else SUB, anchor="lm")
        x += face.getlength(part) / S


# ---------------------------------------------------------------- the models' answers

def wrap(response):
    """A response as display lines, without the end-of-text token."""
    lines = []
    for para in response.replace("<|endoftext|>", "").split("\n"):
        lines += textwrap.wrap(para, WRAP) or [""]
    return lines


def model(arm, response, N, whole=False):
    """A pane's state: who writes, what, and how much of it is shown (0-1). A finished answer shows its opening
    and its end, or, if whole, as much of its end as fits."""
    about = {"D": f"the same {N['d.drill.n']} solutions, about 8 times each",
             "O": f"{N['d.O.n']} different solutions, once each"}
    return {"name": {"D": "Drilled", "O": "Once-trained"}[arm], "color": {"D": RED, "O": GREY}[arm],
            "about": about[arm], "lines": wrap(response["text"]), "tokens": response["tokens"],
            "answer": re.findall(r"\\boxed\{([^}]*)\}", response["text"])[-1], "ok": bool(response["correct"]),
            "ends": response["text"].rstrip().endswith("User:"), "shown": 0.0, "fast": False, "whole": whole}


def shown_lines(lines, fraction):
    """The first fraction of the wrapped response, the last line maybe cut."""
    n, out = round(fraction * sum(len(line) + 1 for line in lines)), []
    for line in lines:
        if n <= 0:
            break
        out.append(line[:n])
        n -= len(line) + 1
    return out


def tail(lines, room, page=1):
    """The last lines that fit in room points, like a terminal that scrolls, page lines at a time."""
    start, used = len(lines), 0
    while start and used + (LINE if lines[start - 1] else GAP) <= room:
        start -= 1
        used += LINE if lines[start] else GAP
    return lines[-(-start // page) * page:]


def gist(lines, room):
    """A long answer in brief: as many of its opening lines as fit, a gap, and the line with its final answer."""
    head, used = [], 2 * LINE
    for line in lines:
        used += LINE if line else GAP
        if used > room:
            break
        head.append(line)
    while head and not head[-1]:
        head.pop()
    return head + ["\u22ee", [line for line in lines if "\\boxed" in line][-1]]


def pane(d, box_, p):
    """One model's answer: its name and training, what it has written so far, its length and result."""
    x, y, w, h = box_
    done, top = p["shown"] >= 1, y + 42
    d.rounded_rectangle(P(x, y, x + w, y + h), radius=8 * S, fill=PANE, outline=BORDER, width=S)
    d.text(P(x + 14, y + 7), p["name"], font=NAME, fill=p["color"])
    d.text(P(x + 22 + NAME.getlength(p["name"]) / S, y + 10.5), p["about"], font=SMALL, fill=DIM)
    d.line(P(x, y + 32, x + w, y + 32), fill=BORDER, width=S)
    if done:
        lines = tail(p["lines"], h - 80) if p["whole"] else gist(p["lines"], h - 80)
    else:
        lines = tail(shown_lines(p["lines"], p["shown"]), h - 80, page=8)
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        fill, face = (DIM, CODE) if line == "</think>" else (TEXT, CODE_BOLD) if "\\boxed" in line else (SUB, CODE)
        if last and done and p["ends"]:  # the response opened a new user turn
            fill, face = WRONG, CODE_BOLD
            end = x + 14 + CODE_BOLD.getlength(line) / S + 9
            d.line(P(end + 3, top + 6.5, end + 16, top + 6.5), fill=WRONG, width=round(1.2 * S))
            d.polygon(P(end, top + 6.5, end + 5, top + 3.5, end + 5, top + 9.5), fill=WRONG)
            d.text(P(end + 21, top + 6.5), "it stops here, as if its chat turn were over", font=SMALL, fill=WRONG,
                   anchor="lm")
        if line == "\u22ee":  # the fonts lack the vertical ellipsis
            for dy in (2.5, 6, 9.5):
                d.ellipse(P(x + 16, top + dy - 1, x + 18, top + dy + 1), fill=DIM)
        else:
            d.text(P(x + 14, top), line, font=face, fill=fill)
        if last and not done:  # the cursor
            cx = x + 14 + CODE.getlength(line) / S + 1
            d.rectangle(P(cx, top + 1, cx + 5.5, top + 11), fill=p["color"])
        top += LINE if line else GAP
    d.text(P(x + 14, y + h - 19), "answer length", font=SMALL, fill=DIM, anchor="lm")
    bx = x + 22 + SMALL.getlength("answer length") / S
    d.rounded_rectangle(P(bx, y + h - 21.5, bx + 120, y + h - 16.5), radius=2.5 * S, fill=BORDER)
    if p["shown"] > 0:
        filled = 120 * min(1, p["tokens"] * p["shown"] / LONGEST)
        d.rounded_rectangle(P(bx, y + h - 21.5, bx + max(filled, 5), y + h - 16.5), radius=2.5 * S, fill=p["color"])
    if done:
        verdict(d, x + w - 12, y + h - 30, p["answer"], p["ok"])
    elif p["fast"]:
        d.text(P(x + w - 14, y + h - 19), "fast-forward", font=SMALL, fill=DIM, anchor="rm")
        fx = x + w - 34 - SMALL.getlength("fast-forward") / S
        for dx in (0, 7):
            d.polygon(P(fx + dx, y + h - 23.5, fx + dx + 6.5, y + h - 19, fx + dx, y + h - 14.5), fill=DIM)


# ---------------------------------------------------------------- cards: what the models train on

def cards(seed, passes=8, n=12):
    """Training solutions, pass by pass, as rows of cards (colour, line lengths, times seen): the colour is the
    problem and the lines its solution. Drilled: the same solutions again; fresh solutions: the same problems with
    new solutions; once-trained: new problems."""
    rng = random.Random(seed)
    lines = lambda: [rng.choice((4, 7, 10, 13)) for _ in range(4)]  # noqa: E731

    def colour(i):
        r, g, b = colorsys.hsv_to_rgb((0.04 + i * 0.618034) % 1, 0.5, 0.92)
        return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))

    own = [lines() for _ in range(n)]
    return [{"D": [(colour(i), own[i], k + 1) for i in range(n)], "P": [(colour(i), lines(), 1) for i in range(n)],
             "O": [(colour(n * (k + 1) + i), lines(), 1) for i in range(n)],
             "s1": [(colour(i), own[i], 5) for i in range(n)], "limo": [(colour(i + 3), own[i], 15) for i in range(n)]}
            for k in range(passes)]


def deck(d, rows, texts, numbers=None, heading="", top=150, step=120):
    """Rows of cards, one per model: its name and what it learns from, its cards with how often each solution has
    been seen, and its accuracy if numbers are given."""
    if numbers:
        d.text(P(776, top - 22), heading.upper(), font=OVER, fill=DIM, anchor="ra")
    for i, (key, name, color, about) in enumerate(rows):
        y = top + step * i
        d.text(P(24, y), name, font=NAME, fill=color)
        for j, line in enumerate(about.split("\n")):
            d.text(P(24, y + 22 + 15 * j), line, font=SMALL, fill=SUB)
        for k, (hue, lengths, seen) in enumerate(texts[key]):
            cx = 262 + 31 * k
            d.rounded_rectangle(P(cx, y - 2, cx + 24, y + 30), radius=3 * S, fill=mix(PANE, hue, 0.22), outline=hue,
                                width=S)
            for n, length in enumerate(lengths):
                d.line(P(cx + 5, y + 5 + 6 * n, cx + 5 + length, y + 5 + 6 * n), fill=mix(hue, TEXT, 0.35),
                       width=round(1.4 * S))
            d.text(P(cx + 12, y + 41), f"\u00d7{seen}", font=CHIP if seen > 1 else SMALL,
                   fill=color if seen > 1 else DIM, anchor="mm")
        if numbers:
            d.text(P(776, y - 6), numbers[key][0], font=BIG, fill=numbers[key][1], anchor="ra")


# ---------------------------------------------------------------- slide bodies

def score(d, N):
    """The summary: each model's accuracy after each stage, as a small table of big numbers."""
    columns = ((384, "after math training"), (550, "after a short, intense round of chat training (no math)"),
               (712, "after a quick math refresher"))
    for x, head in columns:
        for j, line in enumerate(textwrap.wrap(head, 22)):
            d.text(P(x, 96 + 15 * j), line, font=SMALL, fill=SUB, anchor="ma")
    rows = (("Drilled", RED, f"the same {N['d.drill.n']} solutions,\nabout 8 times each",
             (N["m.h.D140"], N["m.it.D140"], N["rl.D.it.u5"])),
            ("Once-trained", GREY, f"{N['d.O.n']} different solutions,\nonce each",
             (N["m.h.O140"], N["m.it.O140"], N["rl.O.it.u5"])))
    for i, (name, color, about, values) in enumerate(rows):
        y = 168 + 104 * i
        d.text(P(24, y), name, font=NAME, fill=color)
        for j, line in enumerate(about.split("\n")):
            d.text(P(24, y + 22 + 15 * j), line, font=SMALL, fill=SUB)
        for k, ((x, _), value) in enumerate(zip(columns, values)):
            d.text(P(x, y + 20), value, font=HUGE, fill=RED if (i, k) == (0, 1) else TEXT, anchor="mm")
            if k:
                arrow(d, columns[k - 1][0] + 42, x - 42, y + 21)


def table(d, columns, rows, top=112):
    """Big numbers in a grid: columns of (x, heading), rows of (name, colour, [(value, colour)])."""
    for x, head in columns:
        for j, line in enumerate(textwrap.wrap(head, 20)):
            d.text(P(x, top + 15 * j), line, font=SMALL, fill=SUB, anchor="ma")
    for i, (name, color, values) in enumerate(rows):
        y = top + 70 + 78 * i
        d.text(P(24, y + 20), name, font=NAME, fill=color, anchor="lm")
        for (x, _), (value, fill) in zip(columns, values):
            d.text(P(x, y + 20), value, font=HUGE, fill=fill, anchor="mm")


def piles(d, N, V):
    """One study's comparison: a few examples many times against many examples once."""
    d.text(P(24, 150), "ONE STUDY, SAME AMOUNT OF TRAINING", font=OVER, fill=DIM)
    for k in range(4):  # a small stack
        d.rounded_rectangle(P(60 + 5 * k, 186 - 5 * k, 110 + 5 * k, 250 - 5 * k), radius=4 * S,
                            fill=mix(PANE, GREEN, 0.25), outline=GREEN, width=S)
    d.text(P(185, 196), f"{N['lit.kop.n']} examples, each {N['lit.kop.ratio']} times", font=LABEL, fill=TEXT)
    d.text(P(185, 216), "did better", font=TEXTF, fill=GREEN)
    for k in range(64):  # a wide pile
        x, y = 430 + 11 * (k % 16), 186 + 16 * (k // 16)
        d.rounded_rectangle(P(x, y, x + 9, y + 13), radius=1.5 * S, fill=mix(PANE, GREY, 0.2), outline=DIM, width=1)
    d.text(P(430, 262), f"{round(V['lit.kop.n'] * V['lit.kop.ratio']):,} different examples, once each", font=LABEL,
           fill=TEXT)


def timeline(d):
    """Where later training comes in."""
    steps = (("Untrained model", "", SUB), ("Reasoning training", "on worked solutions", TEXT),
             ("Later training", "to chat, to follow instructions, or for someone's own task", AMBER),
             ("Does it still reason?", "", TEXT))
    xs = (24, 204, 434, 634)
    widths = (150, 170, 170, 142)
    for (title, words, color), x, w in zip(steps, xs, widths):
        box(d, x, 184, w, 88, title, words, color)
    for i in range(3):
        arrow(d, xs[i] + widths[i] + 6, xs[i + 1] - 6, 228)
    d.line(P(412, 170, 412, 286), fill=GREEN, width=round(1.5 * S))
    d.text(P(412, 162), "HANDOFF", font=OVER, fill=GREEN, anchor="ms")


def loop(d, N):
    """Where the training solutions come from."""
    steps = (("Untrained model", f"already solves {N['m.h.base']} of the test problems when it reasons", SUB),
             ("It writes solutions", "to other math problems, step by step", TEXT),
             ("Keep the correct ones", "and drop the rest", TEXT),
             ("Train three copies", "on those solutions, in three ways", TEXT))
    for i, (title, words, color) in enumerate(steps):
        x = 24 + 190 * i
        box(d, x, 176, 172, 84, title, words, color)
        if i:
            arrow(d, x - 16, x - 4, 218)


def reply(d, x, y, w, heading=""):
    """A reply laid out the way the models are trained to: reasoning first, then the answer."""
    if heading:
        d.text(P(x, y - 18), heading.upper(), font=OVER, fill=DIM)
    d.rounded_rectangle(P(x, y, x + w, y + 120), radius=8 * S, fill=PANE, outline=BORDER, width=S)
    d.rounded_rectangle(P(x + 12, y + 12, x + w - 12, y + 80), radius=5 * S, fill=mix(PANE, BLUE, 0.12))
    d.text(P(x + 22, y + 22), "REASONING", font=OVER, fill=BLUE)
    for k, share in enumerate((0.82, 0.74, 0.86)):
        d.rounded_rectangle(P(x + 22, y + 40 + 11 * k, x + 22 + (w - 44) * share, y + 45 + 11 * k), radius=2 * S,
                            fill=mix(PANE, BLUE, 0.35))
    d.text(P(x + 22, y + 96), "ANSWER", font=OVER, fill=GREEN)
    d.rounded_rectangle(P(x + 86, y + 92, x + 166, y + 104), radius=2 * S, fill=mix(PANE, GREEN, 0.45))


def anatomy(d, doc):
    """How each test answer is scored: the model reasons first, then answers."""
    strip(d, 126, "PROBLEM", doc["problem"], DIM)
    reply(d, 24, 170, 536)
    d.text(P(580, 186), "We start every answer", font=LABEL, fill=TEXT)
    d.text(P(580, 204), "with the model's reasoning.", font=LABEL, fill=TEXT)
    text(d, 580, 232, "Each model answers each problem 4 times; we count the share of right answers.", SMALL, SUB,
         196, 3)


def stages(d, N):
    """The three later stages, with how hard each pushes."""
    items = (("Ordinary chat training", f"one pass over {N['rs.rows']} everyday requests with human-written "
                                        "replies, as strong as the math training", 1),
             ("Intense chat training", f"{N['d.stage.updates']} quick steps on {N['d.norobots']} of those requests, "
                                       "three times as strong", 3),
             ("Intense answers-only training", f"{N['d.stage.updates']} quick steps on puzzles whose answers come "
                                               "with no reasoning at all, three times as strong", 3))
    for i, (title, words, strength) in enumerate(items):
        x = 24 + 256 * i
        box(d, x, 150, 240, 150, title, words, AMBER)
        d.text(P(x + 12, 272), "STRENGTH", font=OVER, fill=DIM)
        for k in range(3):
            d.rounded_rectangle(P(x + 80 + 22 * k, 272, x + 98 + 22 * k, 280), radius=2 * S,
                                fill=AMBER if k < strength else BORDER)


def badges(d):
    """Where the break recurs."""
    items = (("Solutions from a much stronger model", "gpt-oss-120b's long reasoning, the kind s1 and LIMO train on: "
                                                     "it breaks even more"),
             ("Two more training runs", "new problems, a new order and a new start"),
             ("A second set of drilled problems", "a different 579"),
             ("A larger training setup", "more of the model's weights trained"),
             ("A larger model", "Qwen3.5-35B"),
             ("Another model family", "NVIDIA's Nemotron 3 Nano"),
             ("A different task", "arithmetic puzzles instead of math problems"))
    for i, (title, words) in enumerate(items):
        x, y = 24 + 254 * (i % 3), 110 + 94 * (i // 3)
        box(d, x, y, 238, 84, title, words, GREEN, right=18)
        icon(d, x + 222, y + 16, True, GREEN)


def fixes(d, N):
    """What prevents the break, and where the finding applies."""
    box(d, 24, 92, 368, 104, "A new solution each time a problem comes around",
        "Prevented the damage everywhere we tried it: that is the fresh-solution model.", GREEN)
    box(d, 408, 92, 368, 104, f"Mix {N['d.RP.replay']}% of the original solutions into the later training",
        "Prevented it in the ordinary round of chat training.", GREEN)
    box(d, 24, 214, 752, 110, "Where the finding applies",
        "Later training that doesn't mix the reasoning data back in, like adapting a released model to your own "
        "task.\nSkills the model already has, which is what s1 and LIMO sharpen. On a skill it could not do "
        "before, repetition mostly made it learn less.", AMBER)


def frame(sc):
    """One picture of a slide."""
    im = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(im)
    d.text(P(24, 17), sc["over"].upper(), font=OVER, fill=DIM)
    d.text(P(24, 31), sc["title"], font=TITLE, fill=TEXT)
    assert TITLE.getlength(sc["title"]) / S <= 752, sc["title"]
    if sc.get("words"):
        text(d, 24, 70, sc["words"], TEXTF, SUB)
    for i, (label, words, color) in enumerate(sc.get("strips", ())):
        strip(d, 64 + 26 * i, label, words, color)
    top = 64 + 26 * len(sc.get("strips", ())) + 4
    for key, p in sc.get("panes", {}).items():
        pane(d, ((24 if key == "D" else 408), top, 368, 396 - top), p)
    if "deck" in sc:
        deck(d, **sc["deck"])
    if "bars" in sc:
        bars(d, **sc["bars"])
    for draw in sc.get("draw", ()):
        draw(d)
    if "back" in sc:  # the drilled model's accuracy, before and after the refresher
        label, *values = sc["back"]
        d.text(P(592, top + 30), label.upper(), font=OVER, fill=DIM, anchor="ma")
        d.text(P(516, top + 110), values[0], font=HUGE, fill=RED, anchor="mm")
        if len(values) > 1:
            arrow(d, 562, 622, top + 111)
            d.text(P(668, top + 110), values[1], font=HUGE, fill=GREEN, anchor="mm")
            d.text(P(592, top + 160), "The skill was hidden, not lost.", font=BODY, fill=TEXT, anchor="ma")
    for i, (words, color) in enumerate(sc.get("footer", ())):
        face = BODY if i == 0 else SMALL
        assert face.getlength(words) / S <= 752, words
        d.text(P(24, 404 + 19 * i), words, font=face, fill=color)
    return im


def bar(im, slide, share, count):
    """The progress bar: one segment a slide, filled as far as it has played, and the slide's number."""
    d = ImageDraw.Draw(im)
    width = (752 - 3 * (count - 1)) / count
    for i in range(count):
        x = 24 + i * (width + 3)
        d.rounded_rectangle(P(x, 5.5, x + width, 8), radius=S, fill=BORDER)
        filled = width if i < slide else width * share if i == slide else 0
        if filled:
            d.rounded_rectangle(P(x, 5.5, x + max(filled, 2.5), 8), radius=S, fill=SUB)
    d.text(P(776, 17), f"{slide + 1} / {count}", font=OVER, fill=DIM, anchor="ra")
    return im


# ---------------------------------------------------------------- motion

def stream(scene, keys, start=160, leap=300, pause=0):
    """The panes in keys writing their answers at one rate, RATE tokens a second until start tokens, then
    fast-forward, leap tokens a frame; the picture holds pause milliseconds more where one answer ends first."""
    shots, written = [], 0
    longest = max(scene["panes"][k]["tokens"] for k in keys)
    while written < longest:
        fast, before = written >= start, written
        written = min(longest, written + (leap if fast else RATE // FPS))
        panes = dict(scene["panes"])
        for k in keys:
            p = panes[k]
            panes[k] = {**p, "shown": min(1.0, written / p["tokens"]), "fast": fast and written < p["tokens"]}
        ends = any(before < scene["panes"][k]["tokens"] <= written < longest for k in keys)
        shots.append(({**scene, "panes": panes}, 1000 // (FAST_FPS if fast else FPS) + (pause if ends else 0)))
    return shots


def sweep(before, after, keys, color, steps=14):
    """A band of training passing down the panes in keys, leaving them as in after."""
    a, b, out = frame(before), frame(after), []
    top = 64 + 26 * len(before.get("strips", ())) + 4
    for k in range(1, steps + 1):
        im = a.copy()
        for key in keys:
            x, y, w, h = (24 if key == "D" else 408), top, 368, 396 - top
            edge = y + (h + 30) * k / steps
            region = tuple(P(x, y, x + w + 1, min(edge, y + h + 1)))
            im.paste(b.crop(region), region[:2])
            band_top, bottom = P(max(y + 1, edge - 24), min(edge, y + h))
            if bottom > band_top:
                band = (P(x + 1)[0], band_top, P(x + w)[0], bottom)
                tint = Image.new("RGB", (band[2] - band[0], bottom - band_top), color)
                im.paste(Image.blend(im.crop(band), tint, 0.16), band[:2])
            if edge < y + h:
                ImageDraw.Draw(im).line(P(x + 1, edge, x + w - 1, edge), fill=color, width=round(1.5 * S))
        out.append((im, 1000 // FPS))
    return out


def still(scene, words=0):
    """A slide that does not move, held long enough to read: 2 s and a quarter second a word, 8 to 16 s."""
    count = sum(len(str(scene.get(k, "")).split()) for k in ("title", "words")) + words
    count += sum(len(w.split()) for w, _ in scene.get("footer", ()))
    return [(scene, min(16000, max(8000, round(2000 + 250 * count, -2))))]


# ---------------------------------------------------------------- the slides

def slides(doc, R, N, V):
    """The 21 slides, each a list of (scene or picture, milliseconds) ending on its finished picture."""
    problem = ("PROBLEM", doc["problem"], DIM)
    handoff = {"D": model("D", R["D-u140"], N), "O": model("O", R["O-u140"], N)}
    solved = {k: {**p, "shown": 1.0} for k, p in handoff.items()}
    after = {"D": model("D", R["D-u140+it"], N, whole=True), "O": model("O", R["O-u140+it"], N)}
    broken = {k: {**p, "shown": 1.0} for k, p in after.items()}
    healed = model("D", R["D-u140+it+relearn"], N)
    passes = cards(7)
    about = {"D": f"{N['d.drill.n']} problems, one solution each,\nabout 8 times each",
             "P": f"the same {N['d.drill.n']} problems,\na new solution every time",
             "O": f"{N['d.O.n']} problems, one solution each,\nonce each"}
    three = [("D", "Drilled", RED, about["D"]), ("P", "Fresh solutions", BLUE, about["P"]),
             ("O", "Once-trained", GREY, about["O"])]
    tries = R["D-u140+it"]["tries"]
    assert R["D-u140+it"]["right"] == 0 and R["O-u140+it"]["right"] == tries  # "wrong in all 4 tries"
    back = f"Drilled model, on all {N['d.math.eval']} test problems"
    it_strip = ("INTENSE CHAT TRAINING", f"{N['d.stage.updates']} quick steps on everyday requests, nothing about "
                                         "math, three times as strong", AMBER)

    deck_ = []
    deck_.append(still({"over": "Fine until fine-tuned · the study in one slide",
                        "title": "Repetition looks free. Then later training breaks it.", "draw": [lambda d: score(d, N)],
                        "footer": ((f"Share of {N['d.math.eval']} new competition math problems solved. An ordinary "
                                    f"round of chat training costs the drilled model less: {N['rs.D.a']}.", SUB),
                                   ("The next 20 slides explain the study.", DIM))}, 20))

    # the question
    q = "The question"
    deck_.append(still({"over": q, "title": "Models can learn to reason from very few worked solutions.",
                        "words": "A worked solution is a problem solved step by step, down to the answer. Recipes "
                                 "such as s1 and LIMO train on 1,000 of them or fewer, and make up for so few by "
                                 "going over them again and again.",
                        "deck": {"rows": [("s1", "s1", TEXT, f"each solution\n{N['lit.s1.epochs']} times"),
                                          ("limo", "LIMO", TEXT, f"each solution\n{N['lit.limo.epochs']} times")],
                                 "texts": passes[0], "top": 176, "step": 96}}))
    deck_.append(still({"over": q, "title": "Judged right after training, the repetition looks free.",
                        "words": "We call the point where this training ends handoff; it is where models are usually "
                                 "tested. At handoff, going over a few solutions many times does as well as using "
                                 "many different ones, and sometimes better.",
                        "draw": [lambda d: piles(d, N, V)],
                        "footer": (("So repeating the solutions looks like a free way to save data.", SUB),)}, 12))
    deck_.append(still({"over": q, "title": "But a model is rarely finished at handoff.",
                        "words": "Released models are often trained again, by their makers or by whoever adapts them "
                                 "next. That later training usually has nothing to do with reasoning.",
                        "draw": [timeline],
                        "footer": (("Our question: does repeating the same solutions make the reasoning more fragile "
                                    "to that next step?", TEXT),)}, 20))

    # the experiment
    e = "The experiment"
    five = {"over": e, "title": "We trained three copies of one model, changing only the repetition.",
            "footer": (("Colour marks the problem; \u00d7n counts how often the model has seen that exact solution.",
                        SUB), (f"All three take the same {N['d.drill.updates']} training steps of 32 solutions. The "
                               "model is Qwen3.5-9B-Base, an open 9-billion-parameter model.", DIM))}
    deck_.append([({**five, "deck": {"rows": three, "texts": passes[0], "top": 104, "step": 94}}, 2500),
                  *[({**five, "deck": {"rows": three, "texts": texts, "top": 104, "step": 94}}, 450)
                    for texts in passes[1:]],
                  ({**five, "deck": {"rows": three, "texts": passes[-1], "top": 104, "step": 94}}, 12000)])
    deck_.append(still({"over": e, "title": "The solutions are the model's own, so there is nothing new to learn.",
                        "words": "Training a model on solutions it could already write teaches it no new math.",
                        "draw": [lambda d: loop(d, N)],
                        "footer": (("So whatever later separates the three copies comes from how the solutions were "
                                    "shown, not from anything new they learned.", TEXT),)}, 30))
    deck_.append(still({"over": e, "title": f"We test on {N['d.math.eval']} new problems, reasoning first.",
                        "words": f"{N['d.math.eval']} competition problems from the MATH-500 set that no model "
                                 "trained on.",
                        "draw": [lambda d: anatomy(d, doc)],
                        "footer": (("Starting the reasoning for the model means we measure how well it reasons, not "
                                    "whether it decides to.", SUB),)}, 40))
    deck_.append(still({"over": e, "title": "Right after training, all three copies are equally good.",
                        "bars": {"x": 24, "y": 130, "rows": [
                            ("Untrained model", SUB, V["m.h.base"], N["m.h.base"]),
                            ("Drilled", RED, V["m.h.D140"], N["m.h.D140"]),
                            ("Fresh solutions", BLUE, V["mm.P.h"], N["mm.P.h"]),
                            ("Once-trained", GREY, V["m.h.O140"], N["m.h.O140"])]},
                        "footer": ((f"Share of right answers on the {N['d.math.eval']} test problems.", SUB),
                                   ("Nothing in the usual check sets the drilled model apart.", DIM))}, 12))

    # the break
    b = "The break"
    deck_.append(still({"over": b, "title": "Then each copy gets the same later training, with no math in it.",
                        "draw": [lambda d: stages(d, N)],
                        "footer": (("None of these teaches the model to solve math problems. The two intense rounds "
                                    "are stress tests.", SUB),)}, 60))
    deck_.append(still({"over": b, "title": "An ordinary round of chat training already hurts the drilled model.",
                        "bars": {"x": 24, "y": 110, "rows": [
                            ("Drilled, before", RED, V["rs.D.h"], N["rs.D.h"]),
                            ("Drilled, after", RED, V["rs.D.a"], N["rs.D.a"]),
                            ("Fresh solutions, before", BLUE, V["rs.P.h"], N["rs.P.h"]),
                            ("Fresh solutions, after", BLUE, V["rs.P.a"], N["rs.P.a"]),
                            ("Once-trained, before", GREY, V["rs.O.h"], N["rs.O.h"]),
                            ("Once-trained, after", GREY, V["rs.O.a"], N["rs.O.a"])]},
                        "footer": ((f"The drilled model loses about {round(V['rs.ex.D'])} points more than the "
                                    "once-trained model, well beyond what chance would explain.", TEXT),
                                   ("Share of right answers on the 221 test problems.", DIM))}, 20))
    columns = ((360, "ordinary chat training"), (530, "intense chat training"), (700, "intense answers-only training"))
    deck_.append(still({"over": b, "title": "The harder the later training, the further it falls.",
                        "draw": [lambda d: table(d, columns, [
                            ("Drilled", RED, [(N["rs.D.a"], RED), (N["m.it.D140"], RED), (N["m.b.D140"], RED)]),
                            ("Fresh solutions", BLUE, [(N["rs.P.a"], TEXT), (N["mm.P.it"], TEXT), (N["mm.P.b"], TEXT)]),
                            ("Once-trained", GREY, [(N["rs.O.a"], TEXT), (N["m.it.O140"], TEXT),
                                                    (N["m.b.O140"], TEXT)])])],
                        "footer": ((f"Share of right answers on the {N['d.math.eval']} test problems after each kind of "
                                    "later training. All three started at about 95%.", SUB),)}, 20))
    one = {"over": b, "title": "One test problem, before the later training: both get it right.",
           "strips": (problem,), "footer": (("The models' own answers, sped up and shortened.", DIM),)}
    deck_.append([({**one, "panes": handoff}, 2500), *stream({**one, "panes": handoff}, "DO"),
                  ({**one, "panes": solved}, 9000)])
    two = {"over": b, "title": "The same problem after intense chat training: the drilled model gives up.",
           "strips": (it_strip, problem)}
    deck_.append([({**two, "panes": solved}, 3000), *sweep({**two, "panes": solved}, {**two, "panes": after}, "DO",
                                                           AMBER),
                  ({**two, "panes": after}, 500), *stream({**two, "panes": after}, "DO", start=330, pause=1500),
                  ({**two, "panes": broken,
                    "footer": ((f"The drilled model got this problem wrong in all {tries} of its tries; the "
                                f"once-trained model got it right in all {tries}.", SUB),)}, 11000)])
    deck_.append(still({"over": b, "title": "Across all problems, the drilled model quits partway.",
                        "bars": {"x": 24, "y": 120, "top": 100, "rows": [
                            ("Drilled", RED, V["mf.it.D.ends"], N["mf.it.D.ends"] + "%"),
                            ("Once-trained", GREY, V["o.ends"], f"{round(V['o.ends'])}%")]},
                        "draw": [lambda d: [d.text(P(24, 100), "ANSWERS THAT STOP PARTWAY, AS IF THE MODEL'S CHAT TURN "
                                                              "WERE OVER", font=OVER, fill=DIM),
                                            d.text(P(24, 220), "TYPICAL ANSWER LENGTH, IN TOKENS (ROUGHLY, WORDS)", font=OVER,
                                                   fill=DIM),
                                            bars(d, 24, 240, [("Drilled", RED, V["mf.it.D.med"], N["mf.it.D.med"]),
                                                              ("Once-trained", GREY, V["o.median"],
                                                               f"{round(V['o.median']):,}")], top=1500)]],
                        "footer": (("After intense chat training. It doesn't reason badly for longer; it stops "
                                    "reasoning.", SUB),)}, 30))

    # why
    w = "Why"
    why = {"over": w, "title": "The cause is seeing the same texts again, not having few problems.",
           "footer": (("The fresh-solution model has as few problems as the drilled one, and came through unharmed.",
                       SUB), ("It sees each solution about once, in every later round we tried. We registered this "
                              "test before running it.", DIM))}
    numbers = {"D": (N["m.it.D140"], RED), "P": (N["mm.P.it"], TEXT), "O": (N["m.it.O140"], TEXT)}
    deck_.append([({**why, "deck": {"rows": three, "texts": passes[0], "top": 104, "step": 94}}, 2500),
                  *[({**why, "deck": {"rows": three, "texts": texts, "top": 104, "step": 94}}, 450)
                    for texts in passes[1:]],
                  ({**why, "deck": {"rows": three, "texts": passes[-1], "top": 104, "step": 94, "numbers": numbers,
                                    "heading": "after intense chat training"}}, 12000)])

    # what is lost
    lost = "What is lost"
    refresher = {"over": lost, "title": "A quick refresher brings the drilled model's reasoning back.",
                 "strips": (("REFRESHER", f"five small training steps on {N['d.RL.u5']} worked solutions to new math "
                                          "problems", GREEN), problem)}
    deck_.append([({**refresher, "panes": {"D": broken["D"]}, "back": (back, N["m.it.D140"])}, 2500),
                  *sweep({**refresher, "panes": {"D": broken["D"]}, "back": (back, N["m.it.D140"])},
                         {**refresher, "panes": {"D": healed}, "back": (back, N["m.it.D140"])}, "D", GREEN),
                  *stream({**refresher, "panes": {"D": healed}, "back": (back, N["m.it.D140"])}, "D"),
                  ({**refresher, "panes": {"D": {**healed, "shown": 1.0}}, "back": (back, N["m.it.D140"],
                                                                                    N["rl.D.it.u5"]),
                    "footer": ((f"Just nudging it to keep going, by adding \"Wait\" when it stops, wins back only "
                                f"{N['rf.share.it']}% of what it lost.", SUB),)}, 10000)])
    deck_.append(still({"over": lost, "title": "Even training on the layout of reasoning alone brings it back.",
                        "words": f"{N['d.stage.updates']} small training steps on the untrained model's replies to "
                                 "everyday chat prompts. The replies show how to lay out reasoning, but contain "
                                 "almost no math.",
                        "draw": [lambda d: [reply(d, 24, 170, 400, "a reply to an everyday request, almost no math"),
                                            d.text(P(600, 205), f"{round(100 * V['rl.D.it.habR'])}%", font=HUGE,
                                                   fill=GREEN, anchor="mm"),
                                            text(d, 480, 236, "of what the drilled model lost comes back", SMALL, SUB,
                                                 240, 3)]],
                        "footer": (("So the later training switched off the habit of reasoning; the math was still "
                                    "there.", TEXT),
                                   ("Anyone who fine-tunes a drilled model and stops there still loses it.", DIM))},
                       10))

    # seeing it coming
    c = "Seeing it coming"
    drop = lambda key: f"{V[f'drop.{key}']:.0f}%" if V[f"drop.{key}"] >= 1 else "under 0.1%"  # noqa: E731
    deck_.append(still({"over": c, "title": "Accuracy gives no warning. Another measurement does.",
                        "words": "Right after training, ask how likely each copy finds the untrained model's own "
                                 "worked solutions, word for word.",
                        "bars": {"x": 24, "y": 150, "top": 12, "rows": [
                            ("Drilled", RED, V["drop.D"], drop("D")),
                            ("Fresh solutions", BLUE, V["drop.P"], drop("P")),
                            ("Once-trained", GREY, V["drop.O"], drop("O"))]},
                        "draw": [lambda d: d.text(P(24, 128), "HOW MUCH LESS LIKELY THEY ARE", font=OVER, fill=DIM)],
                        "footer": (("The drilled model has moved away from where it started; the other two have "
                                    "barely moved.", SUB),)}, 10))
    deck_.append(still({"over": c, "title": "Is it just over-confidence? No.",
                        "words": f"Most of that change ({N['mb.kept']}%) is over-confidence: the drilled model is "
                                 "even surer of the words the untrained model already preferred. So we trained a "
                                 "fourth copy nearly as over-confident without repeating anything, once through the "
                                 "untrained model's most likely solutions.",
                        "bars": {"x": 24, "y": 196, "top": 12, "rows": [
                            ("Drilled", RED, V["drop.D"], drop("D") + "   fragile"),
                            ("Over-confident", PURPLE, V["drop.SH"], drop("SH") + "   not fragile"),
                            ("Once-trained", GREY, V["drop.O"], drop("O") + "   not fragile")]},
                        "draw": [lambda d: d.text(P(24, 176), "HOW FAR EACH MOVED, AND WHAT LATER TRAINING DID",
                                                  font=OVER, fill=DIM)],
                        "footer": (("So the repeated texts matter beyond the over-confidence they cause.", TEXT),
                                   ("The measurement flags drilled models, but doesn't decide the outcome on its own.",
                                    DIM))},
                       10))

    # how general, what to do
    deck_.append(still({"over": "How general", "title": "It is not a quirk of one setup.",
                        "words": "We repeated the experiment in other settings. The drilled model broke again in "
                                 "every one:",
                        "draw": [badges],
                        "footer": (("How much it breaks varies a lot from run to run, so we claim the direction of "
                                    "the effect, not its size.", SUB),)}, 70))
    deck_.append(still({"over": "What to do", "title": "Easy to prevent, once you know to look.",
                        "draw": [lambda d: fixes(d, N)],
                        "footer": (("In short: repetition looks free at handoff, but the next round of training can "
                                    "switch the reasoning off.", TEXT),)}, 60))
    assert len(deck_) == 21  # "the next 20 slides", on the first
    return deck_


# ---------------------------------------------------------------- the GIF

def picture(shot, progress=None):
    im = shot.copy() if isinstance(shot, Image.Image) else frame(shot)
    return bar(im, *progress) if progress else im


def main():
    deck_ = slides(*load())
    played = []  # every slide in turn, under a progress bar that moves at least once a second
    for i, shots in enumerate(deck_):
        total, clock = sum(ms for _, ms in shots), 0
        for shot, ms in shots:
            for part in range(0, ms, 1000):
                played.append((shot, min(1000, ms - part), (i, (clock + part) / total, len(deck_))))
            clock += ms
    samples = [picture(shots[-1][0]) for shots in deck_[:: 3]]
    samples.append(next(shot for shots in deck_ for shot, _ in shots if isinstance(shot, Image.Image)))
    sheet = Image.new("RGB", (W * S, H * S * len(samples)))
    for i, im in enumerate(samples):
        sheet.paste(im, (0, H * S * i))
    palette = sheet.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = (picture(shot, progress).quantize(palette=palette, dither=Image.Dither.NONE)
              for shot, _, progress in played)
    next(frames).save(OUT, save_all=True, append_images=frames, duration=[ms for _, ms, _ in played], loop=0)


if __name__ == "__main__":
    main()
