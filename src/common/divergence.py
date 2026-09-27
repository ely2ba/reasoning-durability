"""How far a trained state has moved from its base, measured on fixed texts.

Each text is scored under the base and under the state by teacher forcing. Per
response position a score file holds the log-probability of the text's token
(`lp`), the most likely next token (`top`) and its log-probability (`top_lp`).
On the base's own forced reasoning, sampled at top-p 1, the token-pooled mean of
lp_base - lp_state estimates KL(base || state) per token: this is B. On the
state's own texts, the mean of lp_state - lp_base is B'. A is the share of
positions where the two models' most likely tokens differ.

`aggregate` reduces a pair of score files to per-text sums, so that every
statistic and its item-bootstrap interval can be computed without the arrays.
"""

import numpy as np

SURE = np.log(0.95)  # a top token this likely is the whole top-p 0.95 nucleus
FIELDS = ("n", "d", "flips", "d_kept", "gain_kept", "branch", "branch_kept", "d_branch_kept", "top_p_branch_base",
          "top_p_branch_state", "base_lp", "sure_nll", "state_hits")


def load(path):
    data = np.load(path)
    return {key: data[key] for key in data.files}


def aggregate(base, state, body_start):
    """Per-text sums over response positions from `body_start` on (after the forced prefix)."""
    assert (base["tokens"] == state["tokens"]).all(), "the two score files hold different texts"
    body = ~np.isnan(base["lp"]) & (np.arange(base["lp"].shape[1]) >= body_start)
    d = base["lp"] - state["lp"]
    kept = body & (state["top"] == base["top"])  # the state keeps the base's most likely token
    branch = body & (base["tokens"] != base["top"])  # the text took one of the base's alternatives

    def total(values, mask):
        return np.where(mask, values, 0.0).sum(1)

    return {"n": body.sum(1), "d": total(d, body), "flips": (body & ~kept).sum(1), "d_kept": total(d, kept),
            "gain_kept": total(np.exp(state["top_lp"]) - np.exp(base["top_lp"]), kept),
            "branch": branch.sum(1), "branch_kept": (branch & kept).sum(1), "d_branch_kept": total(d, branch & kept),
            "top_p_branch_base": total(np.exp(base["top_lp"]), branch),
            "top_p_branch_state": total(np.exp(state["top_lp"]), branch),
            "base_lp": total(base["lp"], body), "sure_nll": total(-base["top_lp"], body & (base["top_lp"] >= SURE)),
            "state_hits": (body & (state["top"] == base["tokens"])).sum(1)}


def summary(agg):
    """The paper's divergence statistics from per-text sums (dicts or DataFrame columns)."""
    s = {key: float(np.sum(agg[key])) for key in FIELDS}
    n, kept_positions = s["n"], s["n"] - s["flips"]
    return {"B": s["d"] / n, "A": s["flips"] / n, "B_kept_top": s["d_kept"] / n,
            "kept_share": s["d_kept"] / s["d"] if s["d"] > 0 else None,
            "top_probability_gain_where_kept": s["gain_kept"] / kept_positions,
            "alternative_probability_kept": float(np.exp(-s["d_branch_kept"] / s["branch_kept"])),
            "branch_top_probability_base": s["top_p_branch_base"] / s["branch"],
            "branch_top_probability_state": s["top_p_branch_state"] / s["branch"],
            "base_nll": -s["base_lp"] / n, "top_p_offset_lower_bound": s["sure_nll"] / n, "positions": int(n),
            "token_accuracy_base": 1 - s["branch"] / n, "token_accuracy_state": s["state_hits"] / n}


def text_nll(rows):
    """Per-token NLL of a set of texts over every scored token, the forced prefix included, as the registered fit
    checks on the drilled texts compute it (main-phase.md N2, anchor.md): under each state, and under the base the
    rows share, keyed by its label. `rows` holds the per-text sums with the prefix columns (outputs/depth_texts)."""
    rows = rows.assign(n=rows.n + rows.n_prefix, base_lp=rows.base_lp + rows.base_lp_prefix, d=rows.d + rows.d_prefix)
    out = {state: float(-(g.base_lp.sum() - g.d.sum()) / g.n.sum()) for state, g in rows.groupby("state")}
    first = rows[rows.state == rows.state.iloc[0]]  # every row's base scores are the base's
    return out | {first.base.iloc[0]: float(-first.base_lp.sum() / first.n.sum())}
