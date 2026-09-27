"""TCES results (paper sections 3-5 and appendix): the decision clock, forced accuracy before and
after a later stage at 4,096 and 16,384 tokens, how searches end, the anchor arms, and the
next-batch loss after the first later-stage update. Reads outputs/; writes outputs/summaries/tces.json.

Accuracy is over the 192 targeted items (two forced draws each; the decision probe draws four).
Paired intervals resample items. "Within k" counts a sample correct only if it answered within k
tokens, so the 16,384-token probes also give accuracy within 4,096.
"""

import json
from pathlib import Path

import pandas as pd

from common import stats

OUT = Path(__file__).resolve().parents[1] / "outputs"
CAP16 = {"u0": "cap16k-u0-20260924", "after": {"M0": "cap16k-stageb-20260924", "T-u30": "cap16k-stageb-20260924",
                                               "S-u230": "cap-20260924", "U-u140": "cap-20260924", "F-u140": "cap-20260924"}}
ANCHOR, ANCHOR_ARMS = "anchor-20260924", ("FA1-u140", "FA025-u140", "FN-u140", "U2-u140")
FORCED4K = {label: "dc-20260923b" for label in ("M0", "T-u30", "S-u230", "RP47-u140", "RS47-u200")}
FORCED4K |= {f"{arm}-u{u}": "rep-20260923-forced" for arm in "UF" for u in (20, 140)}
FORCED4K |= {f"{arm}-u60": "rep-20260923-forced60" for arm in "UF"}
LATER4K = {"instruction tuning": ("rep-20260923-instruct",),
           "Stage B at 1e-4": ("rep-20260923-gentle", "rep-20260923-gentle-F140")}
DECISION = {"dc-20260923b": ("M0", "T-u10", "T-u20", "T-u30", "S-u40", "S-u120", "S-u230", "RP47-u20", "RP47-u60",
                             "RP47-u140", "RS47-u200"),
            "rep-20260923": tuple(f"{arm}-u{u}" for arm in "UF" for u in (20, 60, 140))}
HALVING = ("M0", "T-u30", "S-u230", "RP47-u140", "RS47-u200")


def condition(rows, cap):
    """Summary of one probe file, over its targeted items (over held-out items for the probes that
    only have those)."""
    t = rows[rows.role == "targeted"] if (rows.role == "targeted").any() else rows
    out = {"role": t.role.iloc[0], "n": len(t)}
    if rows.kind.iloc[0] == "decision":
        return out | {"skip": 100 * float(t.skip.mean()), "skip_all_roles": 100 * float(rows.skip.mean())}
    capped = t.tokens >= cap
    out |= {"accuracy": 100 * float(t.correct.mean()), "strict": 100 * float(t.strict_correct.mean()),
            "capped": 100 * float(capped.mean()), "mean_tokens": float(t.tokens.mean()),
            "resumed": 100 * float((t.resumed > 0).mean()), "capped_with_solution": int(t.found[capped].sum()),
            "capped_count": int(capped.sum())}
    if cap > 4096:
        out |= {"within_4096": 100 * float((t.correct.astype(bool) & (t.tokens <= 4096)).mean()),
                "over_4096": 100 * float((t.tokens > 4096).mean())}
    return out


def per_item(rows, within=None):
    t = rows[rows.role == "targeted"]
    ok = t.correct.astype(bool) & (t.tokens <= within) if within else t.correct.astype(bool)
    return ok.groupby(t.task_id).mean()


def loss(before, after, other=None):
    ids = before.index
    other = None if other is None else (other[0].loc[ids].to_numpy(), other[1].loc[ids].to_numpy())
    return stats.loss(before.to_numpy(), after.loc[ids].to_numpy(), other)


def main():
    samples = pd.read_csv(OUT / "tces_samples.csv.gz")
    groups = dict(tuple(samples.groupby(["run", "label", "update", "kind"])))

    def get(run, label, update, kind="think"):
        return groups[(run, label, update, kind)]

    result = {"conditions": {f"{r}|{label}|u{u}|{k}": condition(g, int(g.cap.iloc[0]))
                             for (r, label, u, k), g in groups.items()}}
    result["half_skip"] = {}
    for run, labels in DECISION.items():
        for label in labels:
            points = sorted((u, g[g.role == "targeted"].skip.mean()) for (r, lab, u, k), g in groups.items()
                            if r == run and lab == label and k == "decision")
            result["half_skip"][f"{run}|{label}"] = stats.crossing(points)
    for label in ("T-u30", "S-u230"):  # the rerun probed updates 1-6; update 0 is the same state's
        points = [(0, get("dc-20260923b", label, 0, "decision").pipe(lambda g: g[g.role == "targeted"].skip.mean()))]
        points += sorted((u, g[g.role == "targeted"].skip.mean()) for (r, lab, u, k), g in groups.items()
                         if r == "dc-decision-rep" and lab == label and k == "decision")
        result["half_skip"][f"dc-decision-rep|{label}"] = stats.crossing(points)
    result["forced_halving"] = {}
    for label in HALVING:
        points = sorted((u, per_item(g).mean()) for (r, lab, u, k), g in groups.items()
                        if r in ("dc-20260923b", "dc-long-20260923") and lab == label and k == "think")
        result["forced_halving"][label] = {"points": points, "halves_at": stats.halving(points)}
    lost = result["loss"] = {}
    for label, after_run in CAP16["after"].items():
        before, after = get(CAP16["u0"], label, 0), get(after_run, label, 20)
        for within in (None, 4096):
            lost[f"16k|{label}|Stage B|{within or 16384}"] = loss(per_item(before, within), per_item(after, within))
    clone = (per_item(get(CAP16["u0"], "S-u230", 0)), per_item(get(CAP16["after"]["S-u230"], "S-u230", 20)))
    result["excess_over_clone"] = {label: loss(per_item(get(CAP16["u0"], label, 0)),
                                               per_item(get(CAP16["after"][label], label, 20)), other=clone)
                                   for label in ("U-u140", "F-u140")}
    f_after = per_item(get("cap-20260924", "F-u140", 20))
    for arm in ANCHOR_ARMS:
        before, after = get(ANCHOR, arm, 0), get(ANCHOR, arm, 20)
        lost[f"16k|{arm}|Stage B|16384"] = loss(per_item(before), per_item(after))
        gain = stats.mean_interval(100 * (per_item(after) - f_after.loc[per_item(after).index]).to_numpy())
        result.setdefault("anchor_minus_F_after", {})[arm] = gain
    for label, run in FORCED4K.items():
        lost[f"4k|{label}|Stage B"] = loss(per_item(get(run, label, 0)), per_item(get(run, label, 20)))
        for stage, later_runs in LATER4K.items():
            for later_run in later_runs:
                if (later_run, label, 20, "think") in groups:
                    lost[f"4k|{label}|{stage}"] = loss(per_item(get(run, label, 0)), per_item(get(later_run, label, 20)))
    train = pd.read_csv(OUT / "train_log.csv.gz")
    later = train[(train.stage == "later") & train.run.isin(["dc-20260923b", "rep-20260923-forced",
                                                              "rep-20260923-forced60"]) & (train["update"] <= 3)]
    result["next_batch_loss"] = {f"{r}|{label}": {int(u): float(v) for u, v in zip(g["update"], g.loss_sum)}
                                 for (r, label), g in later.groupby(["run", "label"])}
    (OUT / "summaries").mkdir(exist_ok=True)
    (OUT / "summaries" / "tces.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
