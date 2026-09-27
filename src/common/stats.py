"""Item-bootstrap intervals, rank correlation and crossing times.

Every interval resamples items (problems or TCES tasks) with replacement,
20,000 times, from one fixed seed. Samples of the same item stay together, and
two states compared on the same items share the resampling, so differences
are paired.
"""

import numpy as np

RESAMPLES, SEED = 20_000, 20260924


def _picks(n):
    return np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))


def _percentiles(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def mean_interval(values):
    """Mean over items and its 95% interval; `values` has one entry per item."""
    values = np.asarray(values, float)
    return {"estimate": float(values.mean()), "interval": _percentiles(values[_picks(len(values))].mean(1))}


def ratio_draws(num, den):
    """The pooled ratio in each resample; draws of different quantities over the same items pair up."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    picks = _picks(len(num))
    return num[picks].sum(1) / den[picks].sum(1)


def percentiles(values):
    return _percentiles(values)


def ratio_interval(num, den):
    """Pooled ratio sum(num) / sum(den) over items (a per-token mean) and its 95% interval."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    picks = _picks(len(num))
    return {"estimate": float(num.sum() / den.sum()), "interval": _percentiles(num[picks].sum(1) / den[picks].sum(1))}


def loss(before, after, other=None):
    """Accuracy lost between two evaluations of the same items, in points; with `other` (a second
    before/after pair on the same items), the excess of the first loss over the second."""
    diff = np.asarray(before, float) - np.asarray(after, float)
    if other is not None:
        diff = diff - (np.asarray(other[0], float) - np.asarray(other[1], float))
    out = mean_interval(100 * diff)
    return {"points": out["estimate"], "interval": out["interval"]}


def spearman(x, y):
    """Rank correlation with average ranks for ties."""
    def ranks(values):
        values = np.asarray(values, float)
        out = np.empty(len(values))
        out[np.argsort(values, kind="stable")] = np.arange(len(values), dtype=float)
        for value in np.unique(values):
            tied = values == value
            out[tied] = out[tied].mean()
        return out
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def crossing(points, level=0.5):
    """First update at which a rising curve [(update, share), ...] reaches `level`, interpolated."""
    for (u0, v0), (u1, v1) in zip(points, points[1:]):
        if v0 >= level:
            return u0
        if v1 >= level:
            return u0 + (level - v0) * (u1 - u0) / (v1 - v0)
    return points[-1][0] if points and points[-1][1] >= level else None


def halving(points):
    """First update at which a falling curve [(update, value), ...] drops below half its first value."""
    level = points[0][1] / 2
    for (a, va), (b, vb) in zip(points, points[1:]):
        if vb < level:
            return a + (va - level) * (b - a) / (va - vb)
    return None
