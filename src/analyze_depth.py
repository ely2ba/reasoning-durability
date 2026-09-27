"""TCES divergence from M0 (paper section 5 and appendix J): B, A, the kept/flipped split and the
probability kept on M0's alternatives, at 1,024 and 4,096 response tokens; B' on each state's own
texts; paired differences; the dose in passes; rank correlations with the forced accuracy lost
under Stage B; and the anchor arms' fit to the drilled texts. Reads outputs/; writes
outputs/summaries/depth.json.
"""

import json
from pathlib import Path

import pandas as pd

from common import divergence, stats

OUT = Path(__file__).resolve().parents[1] / "outputs"
PROFILE, FULL, ANCHOR = "depth-20260923", "depth-full-20260924", "anchor-20260924"
PASSES = {"M0": 0.0, "T-u30": 0.0, "S-u230": 1.15, "RP47-u140": 7.7, "RS47-u200": 11.1}
PASSES |= {f"{arm}-u{u}": round(u * 32 / 579, 1) for arm in "UF" for u in (20, 60, 140)}
SEARCH = ("M0", "T-u30", "S-u230", "RP47-u140", "U-u20", "U-u60", "U-u140", "F-u20", "F-u60", "F-u140")
GROUP = {label: 0 for label in ("M0", "T-u30", "S-u230", "U-u20", "F-u20")}  # the registered dose groups
GROUP |= {"U-u60": 1, "F-u60": 1, "U-u140": 2, "F-u140": 2, "RP47-u140": 2}
LOSS_RUNS = {"M0": "dc-20260923b", "T-u30": "dc-20260923b", "S-u230": "dc-20260923b", "RP47-u140": "dc-20260923b",
             "RS47-u200": "dc-20260923b", "U-u20": "rep-20260923-forced", "F-u20": "rep-20260923-forced",
             "U-u140": "rep-20260923-forced", "F-u140": "rep-20260923-forced", "U-u60": "rep-20260923-forced60",
             "F-u60": "rep-20260923-forced60"}


def states(texts, run, kind="common"):
    out, sums = {}, {}
    for label, rows in texts[(texts.run == run) & (texts.set == kind)].groupby("state"):
        rows = rows.sort_values("task_id")
        sign = -1 if kind == "own" else 1  # B' is the mean of lp_state - lp_base on the state's texts
        ratio = stats.ratio_interval(sign * rows.d, rows.n)
        out[label] = divergence.summary(rows) | {"B_interval": ratio["interval"]}
        if kind == "own":
            out[label] = {"B_own": ratio["estimate"], "B_own_interval": ratio["interval"]}
        sums[label] = rows.set_index("task_id")[["d", "n"]]
    return out, sums


def paired(sums, x, y):
    diff = sums[x].d - sums[y].d.loc[sums[x].index]
    return stats.ratio_interval(diff.to_numpy(), sums[x].n.to_numpy())


def forced_lost_4k(samples):
    """Forced accuracy (targeted items) lost by Stage-B update 20 in the 4,096-token probes, from exact
    counts of correct samples, so that equal losses tie in the rank correlations (R-S47-u200 and F-u60
    both lose 70 of 384)."""
    out = {}
    for label, run in LOSS_RUNS.items():
        rows = samples[(samples.run == run) & (samples.label == label) & (samples.kind == "think")
                       & (samples.role == "targeted")]
        count = rows.groupby("update").correct.agg(["sum", "size"])
        out[label] = 100 * float(count.loc[0, "sum"] - count.loc[20, "sum"]) / int(count.loc[0, "size"])
    return out


def main():
    texts = pd.read_csv(OUT / "depth_texts.csv.gz")
    samples = pd.read_csv(OUT / "tces_samples.csv.gz", usecols=["run", "label", "update", "kind", "role", "correct"])
    result = {}
    for run, length in ((PROFILE, 1024), (FULL, 4096)):
        common, sums = states(texts, run)
        result[str(length)] = {"states": common, "S-u230 minus": {x: paired(sums, "S-u230", x) for x in
                                                                   ("U-u140", "F-u140", "U-u60", "F-u60") if x in sums}}
        if run == PROFILE:
            result["1024"]["own"] = states(texts, run, "own")[0]
    b = {label: v["B"] for label, v in result["4096"]["states"].items()}
    lost = forced_lost_4k(samples)
    result["dose"] = {label: {"passes": PASSES[label], "B_4096": b[label], "lost_4k": lost[label]}
                      for label in PASSES if label in b}
    corr = result["rank_correlation_with_4k_loss"] = {}
    for name, members in (("search", SEARCH), ("with_solver", SEARCH + ("RS47-u200",))):
        y = [lost[x] for x in members]
        corr[name] = {"B_4096": stats.spearman([b[x] for x in members], y),
                      "B_1024": stats.spearman([result["1024"]["states"][x]["B"] for x in members], y),
                      "passes": stats.spearman([PASSES[x] for x in members], y)}
        if name == "search":
            corr[name]["group"] = stats.spearman([GROUP[x] for x in members], y)
    drilled = texts[(texts.run == ANCHOR) & (texts.set == "drilled")]
    if not drilled.empty:  # anchor.md's registered fit check: every scored token, the forced prefix included
        nll = divergence.text_nll(drilled)
        drop = {label: nll["M0"] - value for label, value in nll.items()}
        result["drilled_text_fit"] = {"nll": nll, "drop_share_of_F140": {k: v / drop["F-u140"] for k, v in drop.items()}}
    (OUT / "summaries").mkdir(exist_ok=True)
    (OUT / "summaries" / "depth.json").write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
