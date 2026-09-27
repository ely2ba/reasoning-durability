"""The paper's figures, drawn from outputs/ into paper/figures/.

  fig1  every MATH-500 test problem for the drilled, once-trained and fresh-solution 9B models, at handoff and
        after one epoch at the reasoning rate and the two stress stages
  fig2  the drilled model's extra loss by later stage (grouped by learning rate), and in every run, model and task
  fig3  what later training takes: the decision to reason, and what forcing still finds
  fig5  relearning after the stress stages, and each 9B model's divergence at handoff beside its extra loss, with O
        sharpened without repetition (records/revision.md RL, SH)
  figA1 the arithmetic task's divergence; figA2 forced accuracy through 60 updates at two step sizes;
  figA3 the likelihood lost since handoff under instruction tuning
and, into assets/, the README's two figures in light and dark: the design beside the outcome, and relearning
"""

import functools
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARIES, OUT = ROOT / "outputs/summaries", ROOT / "paper/figures"
DRILLED, FRESH, ONCE, BASE, FAINT = "#b03a2e", "#2a67a8", "#2f3338", "#9a9a9a", "#d9d9d9"
SHARP, GRID, GREY = "#c58b2a", "#ececec", "#777777"  # the sharpened model (records/revision.md SH); grid; notes
# a problem's fate since handoff, in the order of every bar: solved, weakened, lost, not solved at handoff
FATES = (("#34393e", "solved"), ("#e8a598", "weakened"), (DRILLED, "lost since handoff"),
         ("#dcdcdc", "not solved at handoff"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444",
                     "axes.titlelocation": "left", "axes.titlepad": 5, "lines.linewidth": 1.4,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def grid(ax, axis="y"):
    """Light gridlines behind the data."""
    ax.grid(axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


NOTE, TITLE = 6.5, 7.8  # the smallest text and the panel titles of every figure after Figure 1 (points)


def save(fig, name):
    """Write the figure at exactly its canvas size (no tight bounding box), so the paper never rescales it."""
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(OUT / name)


def place(fig, left, bottom, width, height):
    """An axes placed in inches from the figure's bottom left corner."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def panel(fig, x, y, letter, text):
    """A panel's title in inches from the figure's top left corner, its letter in bold."""
    w, h = fig.get_size_inches()
    fig.text(x / w, 1 - y / h, rf"$\bf{{{letter}}}$  {text}", fontsize=TITLE, color=ONCE, va="center")


def sizes(ax):
    """Tick and axis-label sizes of the appendix figures."""
    ax.tick_params(labelsize=NOTE)
    ax.xaxis.label.set_size(NOTE + 0.4)
    ax.yaxis.label.set_size(NOTE + 0.4)


def load(name):
    return json.loads((SUMMARIES / f"{name}.json").read_text())


@functools.cache
def math_samples():
    """Forced MATH-500 samples scored by their first answer, from outputs/math_samples.csv.gz."""
    import pandas as pd
    df = pd.read_csv(ROOT / "outputs/math_samples.csv.gz", low_memory=False,
                     usecols=["run", "bench", "label", "when", "mode", "rule", "task_id", "level", "correct"])
    return df[(df.bench == "math500") & (df["mode"] == "forced") & (df.rule == "first")]


@functools.cache
def counts():
    """Correct samples (0-4) per problem for every evaluated state, keyed by (run, label, when)."""
    return {key: g.groupby("task_id").correct.sum() for key, g in math_samples().groupby(["run", "label", "when"])}


def problem_grids():
    """Correct samples (0-4) per problem for the 9B drilled, once-trained and fresh-solution models at handoff and
    after each later stage, every grid in one order: easiest to hardest at handoff."""
    states = {
        "D": [("pilot-20260924", "D-u140", "u0"), ("revision-rs-20260926", "D-u140+nr-epoch", "u267"),
              ("pilot-20260924", "D-u140+it", "u20"), ("pilot-20260924", "D-u140+b", "u20")],
        "O": [("pilot-20260924", "O-u140", "u0"), ("revision-rs-20260926", "O-u140+nr-epoch", "u267"),
              ("pilot-20260924", "O-u140+it", "u20"), ("pilot-20260924", "O-u140+b", "u20")],
        "P": [("main-9b-20260924", "P-u140", "u0"), ("revision-rs-20260926", "P-u140+nr-epoch", "u267"),
              ("main-9b-20260924", "P-u140+it", "u20"), ("main-9b-20260924", "P-u140+b", "u20")]}
    grids = {arm: [counts()[key] for key in keys] for arm, keys in states.items()}
    levels = math_samples().groupby("task_id")["level"].first().astype(str)
    handoff = sum(grids[arm][0] for arm in "DOP")
    order = sorted(handoff.index, key=lambda t: (-handoff[t], levels[t], t))
    return {arm: [c.reindex(order).to_numpy() for c in cs] for arm, cs in grids.items()}


def canvas(width, height):
    """A figure drawn in inches from its top left corner, without axes."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def header(ax, x, y, letter, title, subtitle=None):
    """A panel's title, its letter in bold, with a grey line beneath."""
    ax.text(x, y, rf"$\bf{{{letter}}}$  {title}" if letter else title, fontsize=7.6, color=ONCE, va="center")
    if subtitle:
        ax.text(x, y + 0.16, subtitle, fontsize=6.3, color=GREY, va="center")


def fate_key(ax, x, y, marker="s", size=14):
    """The colours of FATES, each followed by its name, spaced by the names' measured widths."""
    renderer = ax.figure.canvas.get_renderer()
    for color, name in FATES:
        ax.scatter([x + 0.03], [y], s=size, c=[color], marker=marker, linewidths=0)
        label = ax.text(x + 0.09, y, name, fontsize=6.4, color="#555555", va="center", ha="left")
        x += 0.09 + label.get_window_extent(renderer).width / ax.figure.dpi + 0.3


def fig1(math, revision):
    """Every MATH-500 test problem (levels 3-5, 221 dots per grid, each in the same place in every grid, easiest
    first) for the drilled, once-trained and fresh-solution 9B models at handoff and after each later stage:
    (a) one epoch of instructions at the reasoning rate (records/revision.md RS); (b) 20 updates of instructions and
    (c) 20 of answer-only training at three times that rate, coloured by FATES."""
    grids = problem_grids()
    rows, cols = 10, 23  # 230 places for 221 problems
    width, left, gap_x, gap_y, top = 5.5, 0.98, 0.12, 0.17, 0.42
    pitch = (width - left - 0.02 - 3 * gap_x) / (4 * cols)
    height = top + 3 * rows * pitch + 2 * gap_y + 0.24
    fig, ax = canvas(width, height)
    (solved, _), (weakened, _), (lost, _), (faint, _) = FATES
    arms = (("D", "drilled", DRILLED, "579 solutions, 7.7 passes"),
            ("O", "once-trained", ONCE, "4,480 solutions, once"),
            ("P", "fresh solutions", FRESH, "579 problems, new\nsolution at each visit"))
    heads = (("", "At handoff", "after reasoning training"),
             ("a", "One epoch", "instructions, reasoning rate"),
             ("b", "20 updates", "instructions, 3× the rate"),
             ("c", "20 updates", "answer-only, 3× the rate"))
    for c, (letter, title, subtitle) in enumerate(heads):
        header(ax, left + c * (cols * pitch + gap_x), 0.11, letter, title, subtitle)
    divider = left + cols * pitch + gap_x / 2
    ax.plot([divider, divider], [0.05, top + 3 * rows * pitch + 2 * gap_y], color="#cfcfcf", linewidth=0.6)
    for r, (arm, name, color, detail) in enumerate(arms):
        y0 = top + r * (rows * pitch + gap_y)
        ax.text(0.02, y0 + 0.1, name, fontsize=7.6, color=color, va="center", ha="left")
        ax.text(0.02, y0 + 0.3, detail, fontsize=6.0, color=GREY, va="center", ha="left", linespacing=0.95)
        start = grids[arm][0]
        for c, after in enumerate(grids[arm]):
            x0 = left + c * (cols * pitch + gap_x)
            colors = []
            for before, now in zip(start, after):
                if now >= 3:
                    colors.append(solved)
                elif before >= 3 and c > 0:
                    colors.append(lost if now == 0 else weakened)
                else:
                    colors.append(faint)
            i = np.arange(len(after))
            ax.scatter(x0 + (i % cols + 0.5) * pitch, y0 + (i // cols + 0.5) * pitch, s=6.4, c=colors,
                       linewidths=0, marker="o")
            ax.text(x0 + cols * pitch, y0 - 0.058, f"{100 * after.mean() / 4:.1f}%", fontsize=6.4,
                    color=DRILLED if (arm == "D" and c > 0) else "#666666", ha="right", va="center")
    fate_key(ax, left, height - 0.09, marker="o", size=12)
    ax.text(0.02, height - 0.09, "each dot is a test problem", fontsize=6.4, color=GREY, va="center", ha="left",
            style="italic")
    save(fig, "fig1.pdf")
    return fig


def no_excess(ax, axis="y"):
    """Zero extra loss, and the registered margin of 3 points around it."""
    (ax.axhspan if axis == "y" else ax.axvspan)(-3, 3, color="#efefef", linewidth=0, zorder=0)
    (ax.axhline if axis == "y" else ax.axvline)(0, color="#9a9a9a", linewidth=0.8, zorder=1)


VALUE, RULE = "#4d4d4d", "#e6e6e6"  # Figures 2 and 3: printed values; gridlines


def axes_in(fig, left, top, width, height):
    """Axes placed in inches from the figure's top left corner."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, 1 - (top + height) / h, width / w, height / h])


def text_in(fig, x, y, text, **kw):
    """Text placed in inches from the figure's top left corner."""
    w, h = fig.get_size_inches()
    return fig.text(x / w, 1 - y / h, text, **{"fontsize": NOTE, "color": GREY, **kw})


def line_in(fig, xs, ys, **kw):
    """A line drawn in inches from the figure's top left corner."""
    w, h = fig.get_size_inches()
    fig.add_artist(Line2D([x / w for x in xs], [1 - y / h for y in ys],
                          **{"color": "#444444", "linewidth": 0.6, **kw}))


def break_marks(fig, x, y, vertical=False, size=0.022):
    """Two short slanted strokes marking a break in an axis at (x, y) inches."""
    for off in (-0.012, 0.012):
        if vertical:
            line_in(fig, [x - size, x + size], [y + off + size / 2, y + off - size / 2])
        else:
            line_in(fig, [x + off - size / 2, x + off + size / 2], [y + size, y - size])


def text_width(fig, text, size=NOTE):
    """The width of a text in inches, as it will print."""
    t = fig.text(0, 0, text, fontsize=size)
    width = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return width


def dot(ax, x, y, color, hollow=False, interval=None, horizontal=False, size=3.3, zorder=5):
    """An estimate with its 95% interval: filled for the base model's own solutions, hollow for the traces."""
    if interval is not None:
        xs, ys = (interval, [y, y]) if horizontal else ([x, x], interval)
        ax.plot(xs, ys, color=color, linewidth=0.85, solid_capstyle="butt", zorder=zorder - 1)
    ax.plot([x], [y], "o", ms=size, color=color, mfc="white" if hollow else color, mew=0.9, zorder=zorder,
            clip_on=False)


def mark_key(fig, x, y, items):
    """Marks and what they stand for, in a row from (x, y) inches; an item without a colour is a plain word."""
    for color, hollow, text in items:
        if color is not None:
            line_in(fig, [x], [y], marker="o", ms=3.3, color=color, mfc="white" if hollow else color, mew=0.9,
                    linestyle="")
            x += 0.06
        text_in(fig, x, y, text, color=ONCE, va="center")
        x += text_width(fig, text) + (0.14 if color is not None else 0.06)


def bare(ax):
    """An axes with only its bottom spine and x ticks."""
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=NOTE, pad=1.5, length=2)


def signed(value):
    """A value as printed beside its mark, with a true minus sign."""
    return f"{value:.1f}".replace("-", "−")


def excess_interval(run, label, reference, stage):
    """The extra loss of `label` over `reference` in a stress stage, with its paired 95% interval, from the
    per-problem samples (for the drilled checkpoints whose excess the summaries do not store)."""
    from common import stats
    before, after = (counts()[(run, lab, when)] / 4 for lab, when in ((label, "u0"), (f"{label}+{stage}", "u20")))
    other = [counts()[(run, lab, when)].reindex(before.index).to_numpy() / 4
             for lab, when in ((reference, "u0"), (f"{reference}+{stage}", "u20"))]
    return stats.loss(before.to_numpy(), after.reindex(before.index).to_numpy(), other)


def fig2(math, tces, revision):
    """The drilled model's extra loss over the once-trained model, in points with 95% intervals, on one scale in both
    panels. (a) The 9B model's later stages in two blocks by learning rate, 1e-4 (the reasoning training's; robustness
    r2, r4, r5, one epoch RS, replay RP) and 3e-4 (the stress stage, r3), own solutions filled above gpt-oss-120b's
    traces hollow, every value printed, and a bracket joining the two stages with equal learning rate x updates.
    (b) One column per stage and one row per run, model and task (the two more runs, records/revision.md SD), the
    fresh-solution control in blue on its drilled model's row, "not run" where a stage was not run, and the
    arithmetic task against its clone."""
    rob = math["robustness-20260924"]
    pilot, teacher = math["pilot-20260924"]["excess"], math["teacher-20260924"]["excess"]
    r4, r5, replay, runs = rob["r4"], rob["r5"], revision["RP"]["excess"], revision["SD"]["excess"]
    main, big, nemotron = (math[run]["excess"] for run in ("main-9b-20260924", "main-35b-20260924",
                                                            "main-nemotron-20260924"))
    traces = lambda stage: teacher[f"D-T-u140 minus O-T-u140|{stage}"]  # noqa: E731
    rates = [("reasoning rate (0.0001)", [("20 updates", rob["excess_lr1e-4@20"], r4["excess|D-T-u140|u20"]),
                                          ("60 updates", rob["excess_lr1e-4@60"], r4["excess|D-T-u140|u60"]),
                                          ("one epoch (267)", revision["RS"]["excess"]["D-u140"], None),
                                          ("broad chat, 60", r5["excess|D-u140|u60"], r5["excess|D-T-u140|u60"]),
                                          ("with replay, 60", replay["D-u140"]["primary"],
                                           replay["D-T-u140"]["primary"])]),
             ("three times the rate (0.0003)", [("20 updates", rob["excess_it@20"], traces("it")),
                                                ("40 updates", rob["excess_it@40"], None),
                                                ("60 updates", rob["excess_it@60"], None),
                                                ("answer-only, 20", pilot["D-u140 minus O-u140|b"], traces("b"))])]
    # (b): name, group, on traces, then the drilled and fresh-solution estimates in the three stages (None: not run)
    none = [None, None, None]
    settings = [("9B, first run", 0, False,
                 [rob["excess_lr1e-4@60"], pilot["D-u140 minus O-u140|it"], pilot["D-u140 minus O-u140|b"]],
                 [None, main["P-u140 minus O-u140|it"], main["P-u140 minus O-u140|b"]]),
                *((name, 0, False, [runs[run]["it-lr1e-4"]["D-O"], runs[run]["it"]["D-O"], None],
                   [runs[run]["it-lr1e-4"]["P-O"], runs[run]["it"]["P-O"], None])
                  for name, run in (("second run", "s2"), ("third run", "s3"))),
                *((name, 0, False, [None, main[f"{label} minus O-u140|it"], main[f"{label} minus O-u140|b"]], none)
                  for name, label in (("second drilled set", "D2-u140"), ("rank-128 adapters", "D-r128-u140"))),
                ("gpt-oss-120b traces", 1, True, [r4["excess|D-T-u140|u60"], traces("it"), traces("b")],
                 [None, teacher["P-T-u140 minus O-T-u140|it"], teacher["P-T-u140 minus O-T-u140|b"]]),
                *((name, 2, False, [None, table["D-u140 minus O-u140|it"], table["D-u140 minus O-u140|b"]], none)
                  for name, table in (("Qwen3.5-35B-A3B", big), ("Nemotron-3 Nano", nemotron)))]
    arithmetic = [("correct only", tces["excess_over_clone"]["F-u140"]),
                  ("unfiltered", tces["excess_over_clone"]["U-u140"])]

    width, height, bottom = 5.5, 2.5, 2.17
    fig = plt.figure(figsize=(width, height))
    left, label_room, room = 0.72, 0.73, 110  # (a) runs on to 110 points for its printed values
    spans = [(-5, 25), (-5, 67), (-5, 97)]  # (b)'s three columns, on (a)'s scale
    inch = (width - 0.02 - left - 0.15 - label_room - 0.06 - 0.2) / (room + 5 + sum(h - lo for lo, h in spans))
    panel(fig, 0.02, 0.1, "a", "The rate of the later stage sets the size")
    mark_key(fig, 0.06, 0.27, [(DRILLED, False, "own solutions"), (DRILLED, True, "gpt-oss-120b traces")])
    stage_rows(fig, rates, left, 0.37, bottom, inch, room)
    x1 = width - 0.02
    x0 = x1 - sum(h - lo for lo, h in spans) * inch - 0.2
    panel(fig, x0 - 0.06 - label_room, 0.1, "b", "Every run, model and task")
    setting_columns(fig, settings, arithmetic, x0, x0 - 0.06, 0.49, bottom, inch, spans)
    for middle in (left + 105 * inch / 2, (x0 + x1) / 2):
        text_in(fig, middle, height - 0.07, "extra loss (points)", fontsize=NOTE + 0.3, color=ONCE, ha="center",
                va="center")
    save(fig, "fig2.pdf")
    return fig


def stage_rows(fig, rates, left, top, bottom, inch, room, low=-5, high=100, sub=0.27):
    """Figure 2a: a heading per learning rate and beneath it a row per later stage, own solutions above traces,
    each value printed beside its interval on the side where both of a row's values fit; then the bracket."""
    ax = axes_in(fig, left, top, (room - low) * inch, bottom - top)
    y, rows, heads, blocks = 0.0, {}, [], []
    for head, stages in rates:
        heads.append((head, y))
        y += 0.85
        first = y
        for name, own, traces in stages:
            rows[(head, name)] = (y, own, traces)
            y += 1
        blocks.append((first - 0.55, y - 0.45))
        y += 0.05
    ax.set(ylim=(y - 0.5, -0.45), xlim=(low, room))
    inches = blended_transform_factory(fig.dpi_scale_trans, ax.transData)  # x in inches, y in rows
    for head, yh in heads:
        ax.text(0.02, yh, head, transform=inches, fontsize=NOTE + 0.3, color=ONCE, va="center")
    for y0, y1 in blocks:
        ax.plot([0, 0], [y0, y1], color="#9a9a9a", linewidth=0.7, zorder=1)
        for x in range(20, high + 1, 20):
            ax.plot([x, x], [y0, y1], color=RULE, linewidth=0.5, zorder=0)
    ends = {}
    for (head, name), (yr, own, traces) in rows.items():
        ax.text(left - 0.05, yr, name, transform=inches, fontsize=NOTE, color=ONCE, va="center", ha="right")
        pairs = [(own, yr, False)] if traces is None else [(own, yr - sub, False), (traces, yr + sub, True)]
        right = all(e["interval"][1] + 2.4 + text_width(fig, signed(e["points"])) / inch <= room for e, _, _ in pairs)
        ends[(head, name)] = []
        for e, ye, hollow in pairs:
            dot(ax, e["points"], ye, DRILLED, hollow, e["interval"], horizontal=True)
            x = e["interval"][1] + 2.4 if right else e["interval"][0] - 2.4
            ax.text(x, ye, signed(e["points"]), ha="left" if right else "right", va="center", fontsize=NOTE,
                    color=VALUE, zorder=6)
            ends[(head, name)].append(x + text_width(fig, signed(e["points"])) / inch if right else x)
    # the two stages with the same learning rate x updates: 60 updates at 1e-4 and 20 at 3e-4
    (slow, _), (fast, _) = rates
    y1, y2, x = rows[(slow, "60 updates")][0], rows[(fast, "20 updates")][0], 97
    line = {"color": "#8c8c8c", "linewidth": 0.6, "zorder": 2}
    ax.plot([x, x], [y1, y2], **line)
    for yb, key in ((y1, (slow, "60 updates")), (y2, (fast, "20 updates"))):
        ax.plot([max(ends[key]) + 2.5, x], [yb, yb], linestyle=(0, (1, 1.6)), **line)
    ax.text(x - 2.5, (y1 + y2) / 2 - 0.5, "same learning\nrate × updates", ha="right", va="center", fontsize=NOTE,
            color=GREY, linespacing=1.0, style="italic")
    bare(ax)
    ax.spines["bottom"].set_bounds(low, high)
    ax.set_xticks(list(range(0, high + 1, 20)))


def setting_columns(fig, settings, arithmetic, x0, label_right, top, bottom, inch, spans, gap=0.1):
    """Figure 2b: a column per stage (60 updates at the reasoning rate, then instructions and answer-only at three
    times it) and a row per setting, the drilled model in red and the fresh-solution model in blue on its row."""
    widths = [(h - lo) * inch for lo, h in spans]
    lefts = [x0, x0 + widths[0] + gap, x0 + widths[0] + widths[1] + 2 * gap]
    ys, y, group = [], 0.0, 0
    for _, g, *_ in settings:
        if g != group:
            y, group = y + 0.45, g
        ys.append(y)
        y += 1
    arith = [y + 0.45, y + 1.45]
    columns = []
    for (lo, hi), left, w in zip(spans, lefts, widths):
        cx = axes_in(fig, left, top, w, bottom - top)
        cx.set(xlim=(lo, hi), ylim=(arith[1] + 0.6, -0.6), xticks=list(range(0, hi + 1, 20)))
        no_excess(cx, "x")
        for x in range(20, hi + 1, 20):
            cx.axvline(x, color=RULE, linewidth=0.5, zorder=0)
        bare(cx)
        cx.spines["bottom"].set_bounds(lo, hi)
        columns.append(cx)
    inches = blended_transform_factory(fig.dpi_scale_trans, columns[0].transData)
    for (name, _, on_traces, drilled, fresh), yr in zip(settings, ys):
        for c, cx in enumerate(columns):
            if drilled[c] is None:
                lo, hi = spans[c]
                cx.text((lo + hi) / 2, yr, "not run", ha="center", va="center", fontsize=NOTE, color="#a3a3a3",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6}, zorder=4)
                continue
            dot(cx, drilled[c]["points"], yr, DRILLED, on_traces, drilled[c]["interval"], horizontal=True)
            if fresh[c] is not None:
                dot(cx, fresh[c]["points"], yr, FRESH, on_traces, fresh[c]["interval"], horizontal=True)
        columns[0].text(label_right, yr, name, transform=inches, fontsize=NOTE, color=ONCE, va="center",
                        ha="right")
    for (name, e), yr in zip(arithmetic, arith):
        dot(columns[2], e["points"], yr, DRILLED, interval=e["interval"], horizontal=True)
        columns[2].text(e["interval"][0] - 2.5, yr, name, ha="right", va="center", fontsize=NOTE, color=GREY)
    columns[0].text(label_right, sum(arith) / 2, "arithmetic search,\nagainst its clone", transform=inches,
                    fontsize=NOTE, color=ONCE, va="center", ha="right", linespacing=1.05)
    first = settings[0]
    columns[1].text(first[3][1]["interval"][0] - 2.5, ys[0], "drilled", ha="right", va="center", fontsize=NOTE,
                    color=DRILLED)
    columns[2].text(first[4][2]["interval"][1] + 2.5, ys[0], "fresh solutions", ha="left", va="center",
                    fontsize=NOTE, color=FRESH)
    rights = [lo + w for lo, w in zip(lefts, widths)]
    for (a, b), text in (((lefts[0], rights[0]), "reasoning rate"), ((lefts[1], rights[2]), "three times the rate")):
        text_in(fig, (a + b) / 2, top - 0.22, text, color=ONCE, ha="center", va="center")
        line_in(fig, [a, b], [top - 0.145] * 2, color="#bdbdbd", linewidth=0.5)
    for left, right, text in zip(lefts, rights, ("60 updates", "20 updates", "answer-only, 20")):
        text_in(fig, (left + right) / 2, top - 0.08, text, color=ONCE, ha="center", va="center")


def fig5(math, revision):
    """(a) Relearning: forced accuracy (95% intervals) of each drilled model broken by a stress stage, then after 1-20
    updates of reasoning training (records/revision.md RL), one row per stage with 90-97% magnified above the rest of
    the scale; beside it both models nudged with "Wait", without training (records/reforcing.md), and 20 updates on
    chat responses that show only the format of reasoning. (b) One row per 9B model: the divergence B at handoff (log
    scale) and the extra loss after each stress stage, on a scale broken at 5 points; the sharpened model (SH) in
    gold."""
    width, height = 5.5, 2.4
    fig = plt.figure(figsize=(width, height))
    panel(fig, 0.02, 0.09, "a", "Suppressed, not lost: training brings it back")
    panel(fig, 2.54, 0.09, "b", "B flags the drilled models, not the damage")
    relearning(fig, math, revision)
    divergence_table(fig, math, revision)
    save(fig, "fig5.pdf")
    return fig


def relearning(fig, math, revision, left=0.34, top=0.42, width=2.02):
    """Figure 3a: one row per stress stage; within a row a magnified strip (90-97%) above the rest of the scale, the
    two drilled models set slightly apart, segments that cross the break dotted so that no bend there reads as data.
    The bars are the paired 95% intervals the paper reports: of R_k (the share of the gap to the once-trained model
    closed) drawn on the accuracy scale, and of the once-trained model's own change under the same training."""
    rl = revision["RL"]
    accuracy = {key: v["primary"] for key, v in rl["accuracy"].items()}
    share = {key: v["primary"] for key, v in rl["R"].items()}
    nudged = math["pilot-20260924"]["reforcing"]["budget"]["conditions"]
    zoom_h, main_h, gap, row_gap = 0.33, 0.43, 0.035, 0.09
    wait_x, first, last, habit_x = 0.12, 0.47, 1.58, 1.88
    at = lambda k: first + np.log1p(k) * (last - first) / np.log1p(20)  # noqa: E731 (inches along the row)
    mark_key(fig, 0.02, top - 0.13, [(None, False, "drilled:"), (DRILLED, False, "own solutions"),
                                     (DRILLED, True, "gpt-oss-120b traces"), (ONCE, False, "once-trained")])
    for row, stage in enumerate(("it", "b")):
        y0 = top + row * (zoom_h + gap + main_h + row_gap)
        zoom, main = axes_in(fig, left, y0, width, zoom_h), axes_in(fig, left, y0 + zoom_h + gap, width, main_h)
        for ax, ylim, ticks in ((zoom, (90, 97), [90, 95]), (main, (-9, 90), [0, 25, 50, 75])):
            ax.set(xlim=(0, width), ylim=ylim, yticks=ticks)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False, length=2, pad=1.5, labelsize=NOTE)
            ax.patch.set_alpha(0)
        zoom.axhspan(90, 97, color="#f3f3f3", linewidth=0, zorder=0)
        break_marks(fig, left, y0 + zoom_h + gap / 2, vertical=True)

        def put(x, y, color, hollow=False, interval=None, size=3.3):
            """A point in whichever part of the scale holds it; its interval drawn across both."""
            for ax, (lo, hi) in ((zoom, (90, 99)), (main, (-9, 90))):
                if interval is not None:
                    ax.plot([x, x], interval, color=color, linewidth=0.8, solid_capstyle="butt", zorder=4)
                if lo <= y < hi:
                    dot(ax, x, y, color, hollow, size=size)

        reference = accuracy[f"O-u140+{stage} (acc_ref)"]  # the once-trained model after the same stress stage
        for ax in (zoom, main):
            for x0, x1 in ((at(0) - 0.1, at(20) + 0.1), (habit_x - 0.1, habit_x + 0.1)):
                ax.plot([x0, x1], [reference] * 2, color=ONCE, linewidth=0.6, zorder=2)
        if stage == "it":  # the once-trained model trained the same way, with its paired change
            start = accuracy["O-u140+it@0"]
            for k in (1, 2, 5, 20):
                low, high = rl["ceiling_loss"][f"O-u140+it+relearn@{k}"]["primary"]["interval"]
                put(at(k), accuracy[f"O-u140+it+relearn@{k}"], ONCE, interval=[start - high, start - low], size=2.4)
        for model, own, sign in (("D-u140", "O-u140", -1), ("D-T-u140", "O-T-u140", 1)):
            a0 = accuracy[f"{model}+{stage}@0"]
            closing = accuracy[f"{own}+{stage} (acc_ref)"] - a0  # the gap to the once-trained model at the start
            xs = [at(k) + sign * 0.065 for k in (0, 1, 2, 5, 20)]
            ys = [a0] + [accuracy[f"{model}+{stage}+relearn@{k}"] for k in (1, 2, 5, 20)]
            hollow = sign > 0
            for (xa, ya), (xb, yb) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
                crossing = (ya >= 90) != (yb >= 90)
                for ax in (zoom, main):
                    ax.plot([xa, xb], [ya, yb], color=DRILLED, zorder=3, solid_capstyle="round",
                            linewidth=0.6 if crossing else (0.8 if hollow else 1.2),
                            linestyle=(0, (1, 1.5)) if crossing else ((0, (3, 1.5)) if hollow else "-"))
            put(xs[0], ys[0], DRILLED, hollow)
            for x, y, k in zip(xs[1:], ys[1:], (1, 2, 5, 20)):
                low, high = share[f"{model}+{stage}+relearn@{k}"]["interval"]
                put(x, y, DRILLED, hollow, [a0 + low * closing, a0 + high * closing])
            main.text(xs[0] - 0.05, a0, f"{a0:.1f}", ha="right", va="center", fontsize=NOTE, color=GREY)
            if model == "D-u140":  # 20 updates on chat responses that show only the format of reasoning
                low, high = share[f"{model}+{stage}+habit@20"]["interval"]
                put(habit_x, accuracy[f"{model}+{stage}+habit@20"], DRILLED,
                    interval=[a0 + low * closing, a0 + high * closing])
        put(wait_x, nudged[f"D-u140+{stage}"]["accuracy"], DRILLED)
        put(wait_x, nudged[f"O-u140+{stage}"]["accuracy"], ONCE, size=2.4)
        main.text(at(20) + 0.1, -2, "broken by\n" + ("instructions" if stage == "it" else "answer-only"), ha="right",
                  va="bottom", fontsize=NOTE, color=ONCE, linespacing=1.0)
        base = y0 + zoom_h + gap + main_h
        for x0, x1 in ((wait_x - 0.1, wait_x + 0.1), (at(0) - 0.1, at(20) + 0.1), (habit_x - 0.1, habit_x + 0.1)):
            line_in(fig, [left + x0, left + x1], [base, base])
    for k in (0, 1, 2, 5, 20):
        text_in(fig, left + at(k), base + 0.075, str(k), ha="center", va="center", color="#444444")
    for x, upper, lower in ((wait_x, "“Wait”", "no training"),
                            ((at(0) + at(20)) / 2, "", "updates of reasoning training"),
                            (habit_x, "format", "only, 20")):
        text_in(fig, left + x, base + 0.075, upper, ha="center", va="center", color="#444444")
        text_in(fig, left + x, base + 0.19, lower, ha="center", va="center", color="#444444")
    text_in(fig, 0.05, (top + base) / 2, "forced accuracy (%)", rotation=90, ha="center", va="center", color="#444444")


def divergence_table(fig, math, revision, left=2.54, top=0.2, right=5.42, bottom=2.14):
    """Figure 3b: one row per 9B model in two groups (each text seen once; drilled), each sorted by B; B at handoff
    on a log scale, then the extra loss after each stress stage on a scale broken at 5 points so that the values
    near zero stay apart; the once-trained models are the reference, and the sharpened model is banded in gold."""
    pilot, main, teacher = (math[run] for run in ("pilot-20260924", "main-9b-20260924", "teacher-20260924"))
    depth = lambda run, label: (run["depth"][label]["B"], run["depth"][label]["B_interval"])  # noqa: E731
    stored = lambda run, label, ref: {s: run["excess"][f"{label} minus {ref}|{s}"] for s in ("it", "b")}  # noqa: E731
    computed = lambda run, label, ref: {s: excess_interval(run, label, ref, s) for s in ("it", "b")}  # noqa: E731
    sharp = revision["SH"]["gate"]["arms"][revision["SH"]["gate"]["final"]]
    models = {"O": ("once-trained", ONCE, False, depth(pilot, "O-u140"), None),
              "P": ("fresh solutions", FRESH, False, depth(main, "P-u140"), stored(main, "P-u140", "O-u140")),
              "OT": ("once-trained, traces", ONCE, True, depth(teacher, "O-T-u140"), None),
              "PT": ("fresh traces", FRESH, True, depth(teacher, "P-T-u140"), stored(teacher, "P-T-u140", "O-T-u140")),
              "S": ("sharpened", SHARP, False, (sharp["B"], sharp["interval"]),
                    {s: revision["SH"]["excess"][s]["primary"] for s in ("it", "b")}),
              "D20": ("1.1 passes", DRILLED, False, depth(pilot, "D-u20"),
                      computed("pilot-20260924", "D-u20", "O-u140")),
              "D60": ("3.3 passes", DRILLED, False, depth(pilot, "D-u60"),
                      computed("pilot-20260924", "D-u60", "O-u140")),
              "DT20": ("traces, 1.1 passes", DRILLED, True, depth(teacher, "D-T-u20"),
                       computed("teacher-20260924", "D-T-u20", "O-T-u140")),
              "D2": ("7.7, second set", DRILLED, False, depth(main, "D2-u140"), stored(main, "D2-u140", "O-u140")),
              "D": ("7.7 passes", DRILLED, False, depth(pilot, "D-u140"), stored(pilot, "D-u140", "O-u140")),
              "Dr": ("7.7, rank 128", DRILLED, False, depth(main, "D-r128-u140"),
                     stored(main, "D-r128-u140", "O-u140")),
              "DT": ("traces, 7.8 passes", DRILLED, True, depth(teacher, "D-T-u140"),
                     stored(teacher, "D-T-u140", "O-T-u140"))}
    groups = (("each text seen once", ["O", "P", "OT", "PT", "S"]),
              ("drilled: the same texts again", ["D20", "D60", "DT20", "D2", "D", "Dr", "DT"]))
    rows, n = [], 0
    for heading, keys in groups:
        rows.append((None, heading, n))
        rows += [(key, None, n + 1 + i) for i, key in enumerate(keys)]
        n += 1 + len(keys)
    table_top = top + 0.3
    row_h = (bottom - table_top) / n
    label_w, b_w, loss_w, near_w, gap = 0.8, 0.61, 0.6, 0.2, 0.035
    x_b, x_ao = left + label_w + 0.06, right - loss_w
    x_it = x_ao - 0.13 - loss_w
    near_lim = (-4.5, 4.5)

    def column(x, w, xlim, log=False):
        ax = axes_in(fig, x, table_top, w, n * row_h)
        if log:
            ax.set_xscale("log")
        ax.set(ylim=(n - 0.5, -0.5), xlim=xlim)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, labelleft=False, length=2, pad=1.5, labelsize=NOTE)
        grid(ax, "x")
        ax.patch.set_alpha(0)
        return ax

    bx = column(x_b, b_w, (2.2e-4, 0.22), log=True)
    bx.set_xticks([1e-3, 1e-2, 1e-1], ["0.001", "0.01", "0.1"])
    bx.xaxis.set_minor_locator(plt.NullLocator())
    loss = {}
    for stage, x0 in (("it", x_it), ("b", x_ao)):
        near = column(x0, near_w, near_lim)
        far = column(x0 + near_w + gap, loss_w - near_w - gap, (5, 100))
        near.set_xticks([-3, 3], ["−3", "3"])
        far.set_xticks([25, 50, 75, 100], ["", "50", "", "100"])
        near.axvspan(-3, 3, color="#ececec", linewidth=0, zorder=0)
        near.axvline(0, color="#9a9a9a", linewidth=0.7, zorder=1)
        loss[stage] = (near, far)
        break_marks(fig, x0 + near_w + gap / 2, table_top + n * row_h)
    text_in(fig, x_b + b_w / 2, top + 0.1, "$B$ at handoff", ha="center", va="center", color=ONCE)
    text_in(fig, x_b + b_w / 2, top + 0.22, "(nats per token)", ha="center", va="center")
    text_in(fig, (x_it + right) / 2, top + 0.1, "extra loss (points) after", ha="center", va="center", color=ONCE)
    text_in(fig, x_it + loss_w / 2, top + 0.22, "instructions", ha="center", va="center")
    text_in(fig, x_ao + loss_w / 2, top + 0.22, "answer-only", ha="center", va="center")
    for key, heading, y in rows:
        ty = table_top + (y + 0.5) * row_h
        if key is None:
            text_in(fig, left + label_w, ty, heading, ha="right", va="center", style="italic")
            continue
        name, color, hollow, (b, b_interval), excess = models[key]
        if key == "S":
            for ax in (bx, *loss["it"], *loss["b"]):
                ax.axhspan(y - 0.5, y + 0.5, color="#f6ecd6", linewidth=0, zorder=0.5)
        text_in(fig, left + label_w, ty, name, ha="right", va="center", color=ONCE)
        dot(bx, b, y, color, hollow, b_interval, horizontal=True)
        if excess is None:
            text_in(fig, (x_it + right) / 2, ty, "reference: zero by definition", ha="center", va="center",
                    style="italic", bbox={"fc": "white", "ec": "none", "pad": 0.4})
            continue
        for stage in ("it", "b"):
            e = excess[stage]
            near, far = loss[stage]
            dot(near if e["points"] < near_lim[1] else far, e["points"], y, color, hollow, e["interval"],
                horizontal=True)


def fig3(math, tces):
    """Later training takes the decision first. (a) The arithmetic task: the update of the answer-only stage at which
    half of each checkpoint's openings skip the search (a second run of the first six updates hollow; the band spans
    the checkpoints not drilled). (b) The 9B models: the change in free and forced MATH-500 accuracy between handoff
    and twenty updates of instruction tuning, with paired intervals."""
    half, loss = tces["half_skip"], math["pilot-20260924"]["loss"]
    fig = plt.figure(figsize=(5.5, 2.35))
    width, height = fig.get_size_inches()
    panel(fig, 0.04, 0.13, "a", "Arithmetic: when the decision goes")
    panel(fig, 2.78, 0.13, "b", "Math: the decision goes, competence stays")
    ax = place(fig, 1.2, 0.5, 1.36, 1.55)
    drilled = lambda arm: [(f"{p} passes", [half[f"rep-20260923|{arm}-u{u}"]], DRILLED)  # noqa: E731
                           for p, u in (("1.1", 20), ("3.3", 60), ("7.7", 140))]
    rows = [("not drilled", None), ("starting model", [half["dc-20260923b|M0"]], BASE),
            *((name, [half[f"dc-20260923b|{label}"], half[f"dc-decision-rep|{label}"]], ONCE)
              for name, label in (("RL teacher", "T-u30"), ("clone of the teacher", "S-u230"))),
            ("drilled, correct samples", None), *drilled("F"), ("drilled, all samples", None), *drilled("U")]
    y, lasting = 0, []
    for name, values, *color in rows:
        label = {"transform": ax.get_yaxis_transform(), "ha": "right", "va": "center"}
        if values is None:  # a group's heading
            ax.text(-0.03, y + 0.1, name, fontsize=6.4, color=GREY, style="italic", **label)
            y -= 0.85
            continue
        for j, value in enumerate(values):  # a repeat run just below the first
            ax.plot([value], [y + (0 if len(values) == 1 else 0.2 - 0.4 * j)], "o", color=color[0], ms=3.8,
                    mfc=color[0] if j == 0 else "white", mew=0.9, zorder=4)
        ax.text(-0.03, y, name, fontsize=NOTE, color=ONCE, **label)
        lasting += values if color[0] != DRILLED else []
        y -= 1
    ax.axvspan(min(lasting), max(lasting), color="#efefef", linewidth=0, zorder=0)
    grid(ax, "x")
    ax.set(xlim=(2.3, 5.3), ylim=(y + 0.5, 0.75), yticks=[], xticks=[3, 4, 5])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("update at which half the replies skip\nthe search ($\\leftarrow$ sooner)", linespacing=0.95)
    sizes(ax)
    ax.text(5.28, y + 1.0, "hollow: a\nrepeat run", fontsize=6.2, color=GREY, ha="right", va="center", linespacing=0.9)

    models = [("base", "M0", ONCE), ("once-trained", "O-u140", ONCE), ("drilled, 1.1 passes", "D-u20", DRILLED),
              ("drilled, 3.3 passes", "D-u60", DRILLED), ("drilled, 7.7 passes", "D-u140", DRILLED)]
    for left, mode, head in ((3.84, "unforced", "answering freely"), (4.72, "forced", "reasoning forced")):
        bx = place(fig, left, 0.5, 0.74, 1.55)
        bx.axvline(0, color="#9a9a9a", linewidth=0.8, zorder=1)
        for i, (name, label, color) in enumerate(models):
            e = loss[f"{label}|it|{mode}"]  # a loss: the change is its negative
            change, (low, high) = -e["points"], (-e["interval"][1], -e["interval"][0])
            bx.errorbar([change], [-i], xerr=[[change - low], [high - change]], fmt="o", color=color, ms=3.6,
                        elinewidth=0.9, capsize=0, zorder=4)
            if mode == "unforced":
                bx.text(-0.04, -i, name, transform=bx.get_yaxis_transform(), ha="right", va="center", fontsize=NOTE,
                        color=color)
        bx.set(xlim=(-82, 8), ylim=(-len(models) + 0.5, 0.6), yticks=[], xticks=[-75, -50, -25, 0])
        bx.spines["left"].set_visible(False)
        grid(bx, "x")
        sizes(bx)
        bx.set_title(head, fontsize=NOTE, pad=3, color=GREY)
    fig.text((3.84 + 4.72 + 0.74) / 2 / width, 0.1 / height, "change since handoff (points)", fontsize=NOTE + 0.4,
             ha="center", va="center")
    save(fig, "fig3.pdf")
    return fig


def fig_appendix_movement(math):
    """How far instruction tuning at 3e-4 moves each 9B model. (a) How much less likely each model's own handoff texts
    become (the KL from its handoff; log scale, 95% intervals; the base, from a fresh adapter, dashed; hollow: the
    models trained on gpt-oss-120b's traces, after 2 updates). (b) How much more likely the base model's texts become,
    one row per model, with the band registered for the traces' models."""
    moved, common = math["robustness-20260924"]["delta"], math["robustness-20260924"]["delta_common"]
    traces = math["teacher-20260924"]
    fig = plt.figure(figsize=(5.5, 2.25))
    width, height = fig.get_size_inches()
    panel(fig, 0.04, 0.13, "a", "Off its own handoff texts")
    panel(fig, 2.92, 0.13, "b", "On the base model's texts")
    ax, steps = place(fig, 0.62, 0.46, 1.95, 1.5), [1, 2, 5, 20]
    for label, color, style in (("M0", BASE, (0, (3, 2))), ("O-u140", ONCE, "-"), ("P-u140", FRESH, "-"),
                                ("D-u140", DRILLED, "-")):
        ys = [moved[f"{label}@u{k}"] for k in steps]
        low, high = zip(*(moved.get(f"{label}@u{k}|interval", (y, y)) for k, y in zip(steps, ys)))
        ax.plot(np.log(steps), ys, color=color, linestyle=style, linewidth=1.3, zorder=3)
        ax.errorbar(np.log(steps), ys, yerr=[np.subtract(ys, low), np.subtract(high, ys)], fmt="o", color=color,
                    ms=3.0, elinewidth=0.8, capsize=0, zorder=4)
    for label, color, dx in (("O-T-u140", ONCE, 0.2), ("P-T-u140", FRESH, 0.34), ("D-T-u140", DRILLED, 0.2)):
        ax.plot([np.log(2) + dx], [traces["displacement"][f"{label}@u2"]], "o", color=color, ms=3.6, mfc="white",
                mew=0.9, zorder=5)
    ax.set_yscale("log")
    ax.set(ylim=(5e-5, 0.3), xlim=(-0.2, np.log(20) + 0.25), xticks=np.log(steps),
           xlabel="update of instruction tuning")
    ax.set_xticklabels([str(k) for k in steps])
    ax.set_ylabel("own texts, less likely by\n(nats per token)", linespacing=0.95)
    grid(ax, "y")
    sizes(ax)
    note = {"fontsize": NOTE, "va": "center", "linespacing": 0.92}
    right = np.log(20) + 0.2
    ax.text(np.log(5), moved["D-u140@u5"] * 1.9, "drilled", color=DRILLED, ha="center", **note)
    ax.text(np.log(5), moved["M0@u5"] * 2.1, "base, new adapter", color=BASE, ha="center", **note)
    ax.text(right, moved["O-u140@u20"] * 0.42, "once-trained\nand fresh", color=ONCE, ha="right", **note)
    ax.text(right, 1.1e-4, "hollow: on traces,\nafter 2 updates", color=GREY, ha="right", **note)

    rows = [("own solutions", None), ("once-trained", ("O-u140", ONCE, True)),
            ("fresh solutions", ("P-u140", FRESH, True)), ("drilled", ("D-u140", DRILLED, True)),
            ("gpt-oss-120b traces", None), ("once-trained", ("O-T-u140", ONCE, False)),
            ("fresh traces", ("P-T-u140", FRESH, False)), ("drilled", ("D-T-u140", DRILLED, False))]
    ys, y = [], 0
    for name, spec in rows:
        ys.append(y)
        y -= 0.8 if spec is None else 1
    band = 0.01  # registered for the traces' models after 2 updates (records/teacher.md)
    columns = {k: place(fig, left, 0.46, 0.7, height - 0.46 - 0.52) for left, k in ((3.92, 2), (4.78, 20))}
    for k, bx in columns.items():
        bx.axvline(0, color="#9a9a9a", linewidth=0.8, zorder=1)
        for (name, spec), row in zip(rows, ys):
            if spec is None:
                continue
            label, color, own = spec
            if own:  # a positive change makes the base's texts less likely: plot its negative
                more = -common[f"{label}@u{k}"]
                low, high = (-v for v in reversed(common[f"{label}@u{k}|interval"]))
                bx.errorbar([more], [row], xerr=[[more - low], [high - more]], fmt="o", color=color, ms=3.4,
                            elinewidth=0.9, capsize=0, zorder=4)
            elif k == 2:
                bx.add_patch(plt.Rectangle((-band, row - 0.4), 2 * band, 0.8, color="#ececec", linewidth=0, zorder=0))
                bx.plot([-traces["displacement_common"][f"{label}@u2"]], [row], "o", color=color, ms=3.6,
                        mfc="white", mew=0.9, zorder=4)
            else:
                bx.text(0.0, row, "not run", fontsize=6.2, color="#a8a8a8", ha="center", va="center", style="italic")
        bx.set(xlim=(-0.012, 0.032), ylim=(y + 0.4, 0.5), yticks=[], xticks=[-0.01, 0, 0.01, 0.02, 0.03])
        bx.set_xticklabels(["", "0", "", "0.02", ""])
        bx.spines["left"].set_visible(False)
        grid(bx, "x")
        sizes(bx)
        bx.set_title("after 2 updates" if k == 2 else "after 20", fontsize=NOTE, pad=3, color=GREY)
    to_figure = fig.transFigure.inverted()
    for (name, spec), row in zip(rows, ys):
        fy = to_figure.transform(columns[2].transData.transform((0, row)))[1]
        if spec is None:
            fig.text(2.92 / width, fy, name, fontsize=6.4, color=GREY, style="italic", va="center")
        else:
            fig.text(3.02 / width, fy, name, fontsize=NOTE, color=spec[1], va="center")
    fig.text((3.92 + 4.78 + 0.7) / 2 / width, 0.08 / height, "more likely by (nats per token)", fontsize=NOTE + 0.4,
             ha="center", va="center")
    save(fig, "figA3.pdf")
    return fig


def accuracy_interval(state):
    """Forced accuracy (%) of one evaluated state and its item-bootstrap 95% interval."""
    from common import stats
    out = stats.mean_interval(100 * counts()[state].to_numpy() / 4)
    return out["estimate"], out["interval"]


def fig_appendix_trajectory(math):
    """Forced accuracy (%, 95% intervals) of the drilled and once-trained 9B models through 60 updates of instruction
    tuning, one panel per learning rate. At 3e-4 the states after 2 and 5 updates come from a replay whose training
    losses match the stage's within 0.1%, and those after 40 and 60 from a run that replays its first 20 updates
    exactly (records/robustness.md R1, R3), so the points trace one trajectory."""
    run, pilot = "robustness-20260924", "pilot-20260924"
    states = {"fast": [(0, "{}-u140", "u0", pilot), (2, "{}-u140+it@2", "u2", run), (5, "{}-u140+it@5", "u5", run),
                       (20, "{}-u140+it", "u20", pilot), (40, "{}-u140+it-long@40", "u40", run),
                       (60, "{}-u140+it-long@60", "u60", run)],
              "slow": [(0, "{}-u140", "u0", pilot), (20, "{}-u140+it-lr1e-4@20", "u20", run),
                       (60, "{}-u140+it-lr1e-4@60", "u60", run)]}
    fig = plt.figure(figsize=(5.5, 2.15))
    for key, head, left in (("fast", "At three times the rate, 3×10$^{-4}$", 0.52),
                            ("slow", "At the reasoning rate, 10$^{-4}$", 3.12)):
        ax = place(fig, left, 0.46, 2.18, 1.42)
        updates = [k for k, *_ in states[key]]
        for arm, color in (("O", ONCE), ("D", DRILLED)):
            points = [accuracy_interval((source, label.format(arm), when)) for _, label, when, source in states[key]]
            ys = [y for y, _ in points]
            ax.plot(np.log1p(updates), ys, color=color, linewidth=1.3, zorder=3)
            ax.errorbar(np.log1p(updates), ys, yerr=[[y - i[0] for y, i in points], [i[1] - y for y, i in points]],
                        fmt="o", color=color, ms=3.4, elinewidth=0.9, capsize=0, zorder=4)
        ax.set(ylim=(0, 102), xlim=(-0.25, np.log1p(60) + 0.3), xticks=np.log1p(updates), yticks=[0, 25, 50, 75, 100],
               xlabel="updates of instruction tuning")
        ax.set_xticklabels([str(k) for k in updates])
        grid(ax, "y")
        sizes(ax)
        ax.set_title(head, fontsize=7.4, loc="left", pad=4)
        if key == "fast":
            ax.set_ylabel("forced accuracy (%)")
        else:
            ax.tick_params(labelleft=False)
        end = np.log1p(60) + 0.22
        ax.text(end, 97.5, "once-trained", color=ONCE, fontsize=NOTE, ha="right", va="bottom")
        ax.text(end, ys[-1] - 9, "drilled", color=DRILLED, fontsize=NOTE, ha="right", va="top")
    save(fig, "figA2.pdf")
    return fig


def fig_appendix_search(depth):
    """The arithmetic task at handoff, on M0's samples drawn at top-p 0.95 and scored to 4,096 tokens (hence slightly
    negative B for the lasting checkpoints): (a) the divergence B and (b) the share of M0's probability kept where M0
    sampled a less likely token. Left of each panel the checkpoints trained about once (M0, the RL teacher, its
    clone); right, the drilled ones against passes, F (correct samples, filled) and U (all samples, open) set apart,
    and U's repeat (diamond)."""
    states = depth["4096"]["states"]
    fig = plt.figure(figsize=(5.5, 2.1))
    panel(fig, 0.04, 0.13, "a", "Divergence $B$ from the starting model")
    panel(fig, 2.86, 0.13, "b", "Probability left on its less likely tokens")
    passes = [1.1, 3.3, 7.7]
    for left, key, ylim, ylabel in ((0.5, "B", (-0.012, 0.165), "nats per token"),
                                    (3.3, "alternative_probability_kept", (0, 1.05),
                                     "share of the start's probability")):
        strip, ax = place(fig, left, 0.5, 0.62, 1.34), place(fig, left + 0.7, 0.5, 1.42, 1.34)
        for x, label, color, filled in ((0, "M0", BASE, True), (1, "T-u30", ONCE, False), (2, "S-u230", ONCE, True)):
            strip.plot([x], [states[label][key]], "o", color=color, ms=3.8, mfc=color if filled else "white", mew=0.9)
        strip.set(xlim=(-0.6, 2.6), ylim=ylim, xticks=[0, 1, 2], xlabel="trained once", ylabel=ylabel)
        strip.set_xticklabels(["start", "RL", "clone"])
        strip.tick_params(axis="x", length=0)
        for arm, dx, filled in (("F", -0.16, True), ("U", 0.16, False)):
            ys = [states[f"{arm}-u{u}"][key] for u in (20, 60, 140)]
            ax.plot(np.add(passes, dx), ys, color=DRILLED, linewidth=1.3 if filled else 0.9,
                    linestyle="-" if filled else (0, (3, 2)), zorder=3)
            ax.plot(np.add(passes, dx), ys, "s", color=DRILLED, ms=3.6, mfc=DRILLED if filled else "white", mew=0.9,
                    zorder=4)
        ax.plot([7.7 + 0.48], [states["U2-u140"][key]], "D", color=DRILLED, ms=3.6, mfc="white", mew=0.9, zorder=4)
        ax.set(xlim=(0.4, 8.9), ylim=ylim, xticks=passes, xlabel="drilled: passes over the same texts")
        ax.set_xticklabels(["1.1", "3.3", "7.7"])
        ax.tick_params(labelleft=False)
        for axis in (strip, ax):
            grid(axis, "y")
            sizes(axis)
            if key == "B":
                axis.axhline(0, color="#9a9a9a", linewidth=0.8, zorder=1)
        note = {"fontsize": NOTE, "va": "center", "color": DRILLED, "linespacing": 0.9}
        f, u, u2 = (states[label][key] for label in ("F-u140", "U-u140", "U2-u140"))
        if key == "B":
            ax.text(7.2, f + 0.004, "correct\nsamples", ha="right", **note)
            ax.text(6.95, u - 0.024, "all samples", ha="center", **note)
            ax.text(8.2, u2 + 0.012, "repeat", ha="center", **note)
        else:
            ax.text(7.4, f - 0.07, "correct", ha="right", **note)
            ax.text(7.7, u + 0.09, "all", ha="center", **note)
            ax.text(8.2, u2 - 0.08, "repeat", ha="center", **note)
    save(fig, "figA1.pdf")
    return fig


# ---------------------------------------------------------------- README figures (assets/, light and dark)
README_STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Avenir Next", "Helvetica Neue", "Arial", "DejaVu Sans"],
                "svg.fonttype": "path", "svg.hashsalt": "readme"}
README_THEMES = {  # surface, text, secondary, muted, grid/track, drilled, fresh, once-trained, ink on a once-trained tile
    "light": dict(bg="#ffffff", text="#1f2328", sub="#59636e", muted="#8c959f", track="#eaeef2",
                  drilled="#b03a2e", fresh="#2a67a8", once="#2f3338", once_ink="#ffffff"),
    "dark": dict(bg="#0d1117", text="#e6edf3", sub="#9198a1", muted="#6e7681", track="#21262d",
                 drilled="#d65a4a", fresh="#3f86dc", once="#c9ced6", once_ink="#0d1117"),
}
def readme_canvas(w, h, t):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w), ax.set_ylim(h, 0), ax.axis("off")
    return fig, ax


def readme_tile(ax, x, y, s, color, label, ink):
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.045", fc=color, ec="none"))
    ax.text(x + s / 2, y + s / 2 + 0.005, label, color=ink, fontsize=8.2, ha="center", va="center", fontweight="demibold")


def readme_bar(ax, x, y, width, value, color, t, h=0.11):
    ax.add_patch(FancyBboxPatch((x, y), width, h, boxstyle="round,pad=0,rounding_size=0.03", fc=t["track"], ec="none"))
    ax.add_patch(FancyBboxPatch((x, y), width * value / 100, h, boxstyle="round,pad=0,rounding_size=0.03", fc=color, ec="none"))


def readme_hero(t, N, T):
    fig, ax = readme_canvas(8.0, 4.05, t)
    ax.text(0.28, 0.36, "Same accuracy at handoff. A different fate after more training.", fontsize=15, fontweight="bold",
            color=t["text"], va="center")
    ax.text(0.28, 0.66, f"Qwen3.5-9B-Base, fine-tuned on its own correct math solutions for {N['d.drill.updates']} updates, "
            "then trained further on something unrelated.", fontsize=9.3, color=t["sub"], va="center")
    ax.text(0.28, 0.87, f"Forced accuracy on {N['d.math.eval']} held-out MATH-500 problems.", fontsize=9.3, color=t["sub"],
            va="center")
    top = 1.33
    ax.text(0.28, top - 0.08, "WHAT EACH MODEL TRAINS ON", fontsize=8, color=t["muted"], fontweight="demibold", va="center")
    ax.text(0.28, top + 0.1, "number = problem, letter = solution text", fontsize=7.8, color=t["muted"], va="center")
    stages = (("At handoff", ""), ("After one", "ordinary epoch"), ("Stress test:", "instructions"), ("Stress test:", "answer-only"))
    x0, cw = 3.62, 1.08
    for i, (a, b) in enumerate(stages):
        ax.text(x0 + i * cw, top - (0.08 if b else 0), a.upper(), fontsize=8, color=t["muted"],
                fontweight="demibold", va="center")
        if b:
            ax.text(x0 + i * cw, top + 0.1, b.upper(), fontsize=8, color=t["muted"], fontweight="demibold", va="center")
    ax.plot([x0 - 0.12, x0 - 0.12], [top - 0.16, 3.9], color=t["track"], linewidth=1)
    rows = (
        ("Drilled", t["drilled"], ["1a"] * 8, "white",
         f"{T['d.drill.n']} problems, one solution each, seen about eight times",
         [N["m.h.D140"], N["rs.D.a"], N["m.it.D140"], N["m.b.D140"]]),
        ("Fresh solutions", t["fresh"], [f"1{c}" for c in "abcdefgh"], "white",
         "the same problems, a new correct solution at every visit",
         [N["mm.P.h"], N["rs.P.a"], N["mm.P.it"], N["mm.P.b"]]),
        ("Once-trained", t["once"], [f"{p}a" for p in range(1, 9)], t["once_ink"],
         f"{T['d.O.n']} problems, each solution seen once",
         [N["m.h.O140"], N["rs.O.a"], N["m.it.O140"], N["m.b.O140"]]),
    )
    s, gap = 0.3, 0.045
    for r, (name, color, labels, ink, caption, values) in enumerate(rows):
        y = 1.72 + r * 0.8
        ax.text(0.28, y + 0.02, name, fontsize=11, fontweight="bold", color=t["text"], va="center")
        for k, label in enumerate(labels):
            readme_tile(ax, 0.28 + k * (s + gap), y + 0.16, s, color, label, ink)
        ax.text(0.28, y + 0.61, caption, fontsize=7.9, color=t["sub"], va="center")
        for i, v in enumerate(values):
            x = x0 + i * cw
            story = name == "Drilled" and i > 0
            ax.text(x, y + 0.2, f"{v:.1f}%", fontsize=12.5 if story else 11, color=t["text"] if story or i == 0 else t["sub"],
                    fontweight="bold" if story else "normal", va="center")
            readme_bar(ax, x, y + 0.38, 0.9, v, color, t)
    return fig


def readme_recovery(t, N, T):
    fig, ax = readme_canvas(8.0, 3.5, t)
    ax.text(0.28, 0.36, "Suppressed, not lost.", fontsize=15, fontweight="bold", color=t["text"], va="center")
    ax.text(0.28, 0.66, "The broken drilled models, trained a little more on the base model's own solutions to new problems.",
            fontsize=9.3, color=t["sub"], va="center")
    ax.text(0.28, 0.87, f"Five updates ({N['d.RL.u5']} examples) bring almost all of the reasoning back.", fontsize=9.3,
            color=t["sub"], va="center")
    # plot area in inches: x from 0.75 to 5.55, y from 1.2 (100%) to 2.85 (0%)
    left, right, y100, y0 = 0.75, 5.55, 1.25, 2.9
    xs = [left + 0.3 + i * (right - left - 0.45) / 4 for i in range(5)]
    Y = lambda v: y0 - (y0 - y100) * v / 100  # noqa: E731
    for g in (0, 25, 50, 75, 100):
        ax.plot([left - 0.08, right + 0.08], [Y(g), Y(g)], color=t["track"], linewidth=0.9, zorder=0)
        ax.text(left - 0.16, Y(g), f"{g}%", fontsize=8, color=t["muted"], ha="right", va="center")
    for x, u in zip(xs, (0, 1, 2, 5, 20)):
        ax.text(x, y0 + 0.17, str(u), fontsize=8.5, color=t["muted"], ha="center", va="center")
    ax.text((left + right) / 2, y0 + 0.37, "updates of reasoning training (32 examples each)", fontsize=8.5, color=t["sub"],
            ha="center", va="center")
    series = (
        ("once-trained model, trained the same way", t["once"], "o", [N[f"rl.O.it.u{u}"] for u in (0, 1, 2, 5, 20)]),
        ("drilled, broken by the instruction stress test", t["drilled"], "o", [N[f"rl.D.it.u{u}"] for u in (0, 1, 2, 5, 20)]),
        ("drilled, broken by the answer-only stress test", t["drilled"], "s", [N[f"rl.D.b.u{u}"] for u in (0, 1, 2, 5, 20)]),
    )
    for name, color, marker, vals in series:
        ys = [Y(v) for v in vals]
        ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot(xs, ys, marker, color=color, markersize=6.5, markeredgecolor=t["bg"], markeredgewidth=1.6, zorder=3)
    # direct labels: starting values on the left, names on the right
    for key in ("rl.D.it.u0", "rl.D.b.u0"):
        ax.text(xs[0] + 0.08, Y(N[key]) + 0.19, f"{T[key]}%", fontsize=9.5, color=t["text"], va="center", fontweight="bold")
    ax.plot([xs[3], xs[3]], [Y(88.5), Y(66)], color=t["muted"], linewidth=0.8, zorder=1)
    ax.text(xs[3] - 0.08, Y(66), f"{T['rl.D.it.u5']}% and {T['rl.D.b.u5']}%", fontsize=9.5, ha="right", color=t["text"],
            va="center", fontweight="bold")
    ax.text(xs[3] - 0.08, Y(66) + 0.2, "after five updates", fontsize=8.5, color=t["sub"], ha="right", va="center")
    lx = 5.85
    key = ((t["drilled"], "o", "Drilled, broken by the\ninstruction stress test"),
           (t["drilled"], "s", "Drilled, broken by the\nanswer-only stress test"),
           (t["once"], "o", "Once-trained model,\ntrained the same way"))
    for i, (color, marker, text) in enumerate(key):
        y = 1.35 + i * 0.52
        ax.plot([lx, lx + 0.28], [y, y], color=color, linewidth=2)
        ax.plot([lx + 0.14], [y], marker, color=color, markersize=6.5, markeredgecolor=t["bg"], markeredgewidth=1.6)
        ax.text(lx + 0.4, y, text, fontsize=8.5, color=t["text"], va="center", linespacing=1.15)
    return fig


def readme_figures():
    """The README's two figures, from the number registry, as SVG in GitHub's light and dark themes."""
    registry = json.loads((SUMMARIES / "numbers.json").read_text())
    N, T = {k: v["value"] for k, v in registry.items()}, {k: v["text"] for k, v in registry.items()}
    (ROOT / "assets").mkdir(exist_ok=True)
    with plt.rc_context(README_STYLE):
        for name, make in (("hero", readme_hero), ("recovery", readme_recovery)):
            for mode, theme in README_THEMES.items():
                fig = make(theme, N, T)
                fig.savefig(ROOT / "assets" / f"{name}-{mode}.svg", facecolor=theme["bg"], metadata={"Date": None})
                plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    math, tces, depth, revision = load("math"), load("tces"), load("depth"), load("revision")
    for fig in (fig1(math, revision), fig2(math, tces, revision), fig3(math, tces),
                fig5(math, revision), fig_appendix_search(depth), fig_appendix_trajectory(math),
                fig_appendix_movement(math)):
        plt.close(fig)
    readme_figures()
    import explainer_gif  # the README's animated explainer, assets/explainer.gif
    explainer_gif.main()


if __name__ == "__main__":
    main()
