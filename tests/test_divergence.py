"""Divergence (src/common/divergence.py) and bootstrap intervals (src/common/stats.py).

A tiny synthetic scoring, hand-computed:
- two texts; response positions 0-2 are the forced prefix and are ignored; text 1 is one token shorter
  (NaN padding);
- d = log p_base - log p_state at body positions: text 0 -> -0.05, 0.5, 0.2; text 1 -> 0.0, 0.2.

  B         = (-0.05 + 0.5 + 0.2 + 0.0 + 0.2) / 5          = 0.17
  A         = 2 flipped top tokens / 5 positions            = 0.4
  kept part = (-0.05 + 0.5 + 0.0) / 5                        = 0.09   (share 0.45 / 0.85)
  top gain  = mean over kept of exp(top_state) - exp(top_base) = (0.046392 + 0.077913 + 0) / 3
  branch    = text 0, position 4 (the base sampled a token other than its own top)
  alt kept  = exp(-0.5) (the state keeps the base's top there and gives the sampled token 0.5 nats less)
  base NLL  = (0.1 + 1.0 + 2.0 + 0.5 + 0.2) / 5              = 0.76

The reference implementation below is checked against these numbers first; the tests of
src/common/divergence.py (aggregate, summary) and src/common/stats.py then use the same numbers,
plus regression values audited on the DuraSeed-v1 runs.
"""

import math

import numpy as np
import pytest

from conftest import DURASEED, common, needs

NAN, P = float("nan"), 7  # P: a filler token id for the three prefix positions
BASE = {
    "tokens": np.array([[P, P, P, 10, 11, 12], [P, P, P, 20, 21, -1]], np.int32),
    "lp": np.array([[-5, -5, -5, -0.1, -1.0, -2.0], [-5, -5, -5, -0.5, -0.2, NAN]], np.float32),
    "top": np.array([[P, P, P, 10, 99, 12], [P, P, P, 20, 21, -1]], np.int32),
    "top_lp": np.array([[-5, -5, -5, -0.1, -0.3, -2.0], [-5, -5, -5, -0.5, -0.2, NAN]], np.float32),
    "task_ids": np.array(["a", "b"]),
}
STATE = {
    "tokens": BASE["tokens"].copy(),
    "lp": np.array([[-9, -9, -9, -0.05, -1.5, -2.2], [-9, -9, -9, -0.5, -0.4, NAN]], np.float32),
    "top": np.array([[P, P, P, 10, 99, 13], [P, P, P, 20, 22, -1]], np.int32),
    "top_lp": np.array([[-9, -9, -9, -0.05, -0.2, -1.9], [-9, -9, -9, -0.5, -0.3, NAN]], np.float32),
    "task_ids": BASE["task_ids"].copy(),
}
TOP_GAIN = ((math.exp(-0.05) - math.exp(-0.1)) + (math.exp(-0.2) - math.exp(-0.3)) + 0.0) / 3
EXPECTED = {"B": 0.17, "A": 0.4, "B_kept_top": 0.09, "share_kept_top": 0.45 / 0.85,
            "top_gain_kept": TOP_GAIN, "alt_kept": math.exp(-0.5), "base_nll": 0.76,
            "item_sums": [0.65, 0.2], "item_counts": [3, 2]}
TOL = 1e-6


def reference(base, state, body_start=3):
    """Plain re-statement of the registered definitions (DuraSeed-v1 analyze_math_pilot/depth-profile)."""
    body = ~np.isnan(base["lp"]) & (np.arange(base["lp"].shape[1]) >= body_start)
    n = body.sum()
    d = np.where(body, base["lp"].astype(float) - state["lp"].astype(float), 0.0)
    kept = body & (state["top"] == base["top"])
    alt = kept & (base["tokens"] != base["top"])
    return {"B": d.sum() / n, "A": (body & (state["top"] != base["top"])).sum() / n,
            "B_kept_top": np.where(kept, d, 0).sum() / n, "share_kept_top": np.where(kept, d, 0).sum() / d.sum(),
            "top_gain_kept": np.where(kept, np.exp(state["top_lp"]) - np.exp(base["top_lp"]), 0).sum() / kept.sum(),
            "alt_kept": math.exp(-np.where(alt, d, 0).sum() / alt.sum()),
            "base_nll": -np.where(body, base["lp"], 0).sum() / n,
            "item_sums": d.sum(1).tolist(), "item_counts": body.sum(1).tolist()}


def test_reference_matches_hand_values():
    got = reference(BASE, STATE)
    for key, want in EXPECTED.items():
        assert np.allclose(got[key], want, atol=TOL), (key, got[key], want)


def test_fixture_is_a_valid_scoring():
    """Top-1 identity holds, so the fixture is something a real scorer could return."""
    for arrays in (BASE, STATE):
        body = ~np.isnan(arrays["lp"])
        assert (arrays["top_lp"][body] >= arrays["lp"][body] - 1e-6).all()
        same = body & (arrays["top"] == arrays["tokens"])
        assert np.allclose(arrays["top_lp"][same], arrays["lp"][same])


# ---- src/common/divergence.py: aggregate(base, state, body_start) -> per-text sums; summary(agg) ---------

def api_summary(base=BASE, state=STATE, body_start=3):
    aggregate, summary = common("divergence", "aggregate", "summary")
    return aggregate(base, state, body_start), summary(aggregate(base, state, body_start))


def test_per_text_sums():
    agg, _ = api_summary()
    assert list(agg["n"]) == [3, 2]
    assert np.allclose(agg["d"], EXPECTED["item_sums"], atol=TOL)
    assert list(agg["flips"]) == [1, 1] and list(agg["branch"]) == [1, 0] and list(agg["branch_kept"]) == [1, 0]
    assert np.allclose(agg["d_kept"], [0.45, 0.0], atol=TOL) and np.allclose(agg["d_branch_kept"], [0.5, 0.0], atol=TOL)


def test_summary_matches_hand_values():
    _, got = api_summary()
    for mine, theirs in (("B", "B"), ("A", "A"), ("B_kept_top", "B_kept_top"), ("share_kept_top", "kept_share"),
                         ("top_gain_kept", "top_probability_gain_where_kept"),
                         ("alt_kept", "alternative_probability_kept"), ("base_nll", "base_nll")):
        assert abs(got[theirs] - EXPECTED[mine]) < TOL, (theirs, got[theirs], EXPECTED[mine])
    assert got["positions"] == 5
    # branch points (the base sampled an alternative): text 0, position 4; top probabilities there
    assert abs(got["branch_top_probability_base"] - math.exp(-0.3)) < TOL
    assert abs(got["branch_top_probability_state"] - math.exp(-0.2)) < TOL
    assert got["top_p_offset_lower_bound"] == 0.0  # no body position has a top token above 0.95


def test_prefix_positions_are_ignored():
    """Changing the three prefix positions changes nothing."""
    moved = {k: v.copy() for k, v in STATE.items()}
    moved["lp"][:, :3] = -50.0
    moved["top"][:, :3] = 12345
    assert api_summary(state=moved)[1] == api_summary()[1]


def test_identical_models_give_zero():
    _, got = api_summary(state=BASE)
    assert got["B"] == 0.0 and got["A"] == 0.0 and got["kept_share"] is None
    assert got["alternative_probability_kept"] == 1.0


def test_top_p_offset_lower_bound():
    """At a body position whose base top token has p >= 0.95 the nucleus is that token alone, so
    -log p_top lower-bounds the per-token offset of B on top-p 0.95 texts."""
    base = {k: v.copy() for k, v in BASE.items()}
    base["top_lp"][0, 3] = base["lp"][0, 3] = math.log(0.99)
    state = {k: v.copy() for k, v in STATE.items()}
    got = api_summary(base, state)[1]
    assert abs(got["top_p_offset_lower_bound"] - (-math.log(0.99)) / 5) < TOL


@pytest.mark.duraseed
def test_audited_depth_profile_numbers():
    """F-u140 against M0 on the TCES depth-profile common set (an independent recomputation, 2026-09-24)."""
    root = needs(DURASEED / "runs" / "depth-profile" / "depth-20260923", "DuraSeed-v1 depth-profile run")
    load = common("divergence", "load")
    _, got = api_summary(load(root / "M0" / "common.npz"), load(root / "F-u140" / "common.npz"))
    assert abs(got["B"] - 0.161075) < 1e-5 and abs(got["A"] - 0.08797) < 1e-4
    assert abs(got["base_nll"] - 0.42900) < 1e-4
    assert abs(got["kept_share"] - 0.692) < 0.002                 # two-thirds of B at kept positions
    assert abs(got["alternative_probability_kept"] - math.exp(-1.305)) < 0.002
    assert abs(got["top_p_offset_lower_bound"] - 0.00330) < 1e-4


# ---- src/common/stats.py -------------------------------------------------------------------------

def test_ratio_interval_is_degenerate_when_every_item_has_the_same_rate():
    ratio_interval = common("stats", "ratio_interval")
    counts = np.array([3.0, 5.0, 8.0, 2.0])
    out = ratio_interval(0.25 * counts, counts)
    assert abs(out["estimate"] - 0.25) < 1e-12 and np.allclose(out["interval"], [0.25, 0.25], atol=1e-12)


def test_ratio_interval_pools_tokens_not_items():
    """sum(sums) / sum(counts) = 0.17, not the mean of per-item rates (0.1583); the interval lies
    between the two item rates."""
    ratio_interval = common("stats", "ratio_interval")
    out = ratio_interval(EXPECTED["item_sums"], EXPECTED["item_counts"])
    lo, hi = out["interval"]
    assert abs(out["estimate"] - 0.17) < 1e-12 and 0.1 - 1e-9 <= lo <= 0.17 <= hi <= 0.65 / 3 + 1e-9


def test_loss_is_paired_by_problem():
    """Problems differ a lot in difficulty, but each loses exactly 5 points: the paired interval is [5, 5]."""
    loss = common("stats", "loss")
    before = np.random.default_rng(0).uniform(0.2, 1.0, 221)
    out = loss(before, before - 0.05)
    assert abs(out["points"] - 5) < 1e-9 and np.allclose(out["interval"], [5, 5], atol=1e-9)
    excess = loss(before, before - 0.3, other=(before, before - 0.1))
    assert abs(excess["points"] - 20) < 1e-9 and np.allclose(excess["interval"], [20, 20], atol=1e-9)


def test_mean_interval_is_reproducible_and_normal_sized():
    mean_interval = common("stats", "mean_interval")
    values = np.random.default_rng(3).normal(2.0, 10.0, 221)
    first, second = mean_interval(values), mean_interval(values)
    assert first == second
    lo, hi = first["interval"]
    half = 1.96 * values.std() / math.sqrt(len(values))
    assert lo < values.mean() < hi and 0.8 * 2 * half < hi - lo < 1.2 * 2 * half


def test_mean_interval_coverage_is_near_95_percent():
    mean_interval = common("stats", "mean_interval")
    rng, hits = np.random.default_rng(7), 0
    for _ in range(60):
        lo, hi = mean_interval(rng.normal(0.3, 1.0, 100))["interval"]
        hits += lo <= 0.3 <= hi
    assert 50 <= hits <= 60


def test_spearman_ties_and_registered_values():
    spearman = common("stats", "spearman")
    assert abs(spearman([1, 2, 2, 3], [1, 2, 3, 4]) - 4.5 / math.sqrt(4.5 * 5)) < 1e-12
    # depth-profile prediction 4: B against forced points lost by update 20 (lost in draws of 384;
    # ties kept exact as integers): M0, T-u30, S-u230, R-P47-u140, U-u20, U-u60, U-u140, F-u20, F-u60, F-u140
    b = [0.0, -0.00035246, -0.00350270, 0.06102172, -0.00272649, 0.01527942, 0.10717411, -0.00251517,
         0.01568342, 0.16107486]
    lost = [38, 11, -2, 225, 7, 71, 205, -15, 70, 225]
    assert abs(spearman(b, lost) - 0.93010) < 1e-4
    assert abs(spearman(b + [0.09331218], lost + [70]) - 0.89042) < 1e-4  # with the solver R-S47-u200


def test_crossing_reproduces_the_half_skip_updates():
    crossing = common("stats", "crossing")
    assert crossing([(0, 0.0), (2, 0.2), (4, 0.6)]) == 3.5
    assert crossing([(0, 0.6), (2, 0.9)]) == 0 and crossing([(0, 0.0), (2, 0.3)]) is None
    # decision-competence C3 (second run): T-u30 4.04, S-u230 4.81 from the skip shares after updates 1-6
    t = crossing(list(zip(range(1, 7), (0.01, 0.05, 0.12, 0.48, 0.90, 0.98))))
    s = crossing(list(zip(range(1, 7), (0.00, 0.02, 0.09, 0.25, 0.56, 0.68))))
    assert abs(t - 4.04) < 0.02 and abs(s - 4.81) < 0.02


def test_halving_reproduces_the_registered_half_lives():
    """decision-competence Part C, forced accuracy at updates 0, 20, 80, 160, 320: 120, 65, 148, 105."""
    halving = common("stats", "halving")
    curves = {"M0": (54.9, 45.1, 39.3, 15.6, 0.0), "T-u30": (62.2, 59.4, 21.4, 16.7, 0.0),
              "S-u230": (57.0, 57.6, 47.7, 25.0, 0.8), "R-S-u200": (58.9, 40.6, 35.4, 16.1, 1.0)}
    want = {"M0": 120, "T-u30": 65, "S-u230": 148, "R-S-u200": 105}
    for label, values in curves.items():
        got = halving(list(zip((0, 20, 80, 160, 320), values)))
        assert abs(got - want[label]) < 1.0, (label, got)
    assert halving([(0, 10.0), (20, 9.0)]) is None
