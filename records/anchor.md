# Anchor test: does keeping the base's reasoning in reach make drilled competence last?

Draft 2026-09-24 01:36 UTC, revised after an internal review. It is fixed
when the launch time is added at the end, before any training.

**Authorization and budget.** [redacted: private message] [redacted: account cap] This run has
its own limit of $250; its ledger stops beyond that.

## Why

On TCES, repeating 579 samples 7.7 times over-sharpens M0's reasoning: it keeps
M0's top choices and cuts its alternatives. After later training, such
checkpoints' forced searches stop finishing. With a 16,384-token budget they
solve 6–12%, against 76% for a clone that saw fresh completions
([depth-profile.md](depth-profile.md): corrections and cap test).

Two things rise together in every checkpoint so far: the number of passes, and
B, the per-token log-likelihood gap on M0's own reasoning. So B marks the
fragile group, but nothing yet shows that it matters.

This test keeps the drilling fixed and moves B. Fitting M0's own samples is the
sampled gradient of KL(M0‖state), the quantity B estimates. Adding such samples
to every drilling batch therefore pulls the state back toward M0's reasoning
without removing any drilling.

A control adds the same number of unrelated examples. It dilutes the drilling
in the same way but adds no pull toward M0's reasoning. It is not neutral:
instruction data can affect reasoning on their own, as Part D3 showed, so the
control gets the same checks.

## Arms

All arms start from M0's weights with a fresh optimizer and use the U/F recipe:
loss on the sampled tokens normalized by each stream's mean length, learning
rate 1e-4, Adam 0.9/0.95/1e-12, no clipping, 140 updates.

- **FA1.** F-579's exact 579 correct teacher samples, in F's order, 32 per
  update (7.7 passes, as in F-u140). Every batch also holds 32 anchor samples,
  loss weight 1.0.
- **FA025.** The same, with the anchor's loss weighted 0.25. The data are
  identical to FA1; only the pull changes.
- **FN (dilution control).** The same drilling plus 32 fresh single-turn No
  Robots examples per batch: 4,480, each seen once, disjoint from the 640 of
  Part D3.
- **U2.** U-579 repeated on a second random subset (namespace
  `anchor-20260924-u2`), 7.7 passes, no extra stream. It shares 57 of its 579
  texts with U, so it replicates the drilled collapse.

**Anchor samples.**
- Six forced samples of M0 on each of the clone corpus's 800 prompts:
  temperature 1, top-p 1, cap 4,096.
- They stop only at a new user turn, so they never end on a restated answer
  template and need no resumption.
- The first 4,480 in a fixed prompt order are kept, and each is seen once.
- The forced prefix `\n<think>\n` sits in the prompt with weight 0. The anchor
  therefore acts on the reasoning body, which is also what the probe and B
  measure.
- 541 of the 579 drilled prompts also receive anchor samples.

## Later stage and measures

Each arm runs the unchanged 20-update Stage B at 3e-4.

**Probes.**
- Forced `<think>` on the 192 targeted items, two draws, cap 16,384, template
  stops resumed; after updates 0 and 20.
- From the same samples, accuracy within 4,096 tokens, for comparison with Part
  D and the cap test.
- Unforced accuracy at update 0, scored within 4,096 tokens to match F-u140's
  55.7%. Decision probes at updates 0 and 20.

**Divergence.** B, A and the kept-top/flipped split of B, on the depth-profile
common set: the same 192 texts, cut at 1,024, against M0's scores from
`depth-20260923`.

**Fit to the drilled texts.** F-579's 579 texts are scored under M0, F-u20,
F-u140 and each drilled arm. Drilling is measured as the fall in per-token
negative log-likelihood from M0.

**Comparators.** The same day's cap test gives F-u140 after Stage B: 11.7%
within 16,384 tokens and 4.2% within 4,096, on the same items. Each arm's
difference from F-u140 is reported with paired item-bootstrap intervals at
both budgets.

## Checks

These apply to FA1, FA025 and FN, and are not predictions.

**Mechanism checks.** If any fails for FA1, prediction 1 is reported as not
interpretable.
- **The drilling happened.** The arm's fall in drilled-text NLL from M0 is at
  least 80% of F-u140's.
- **Forced parity.** Forced accuracy at update 0, within 4,096 tokens, is
  within 5 points of F-u140's 61.7%.
- **The anchor moved B** (FA arms only). B(FA1) is at most half of F-u140's
  0.1611, and B(FA025) is below 0.1611.

**For the practical claim only.** The claim that the anchor keeps what drilling
bought also requires unforced accuracy within 5 points of F-u140's 55.7%.

## Predictions, written before data

1. **The anchor protects the drilled competence.** After 20 Stage-B updates,
   FA1 solves at least 40% forced within 16,384 tokens (F-u140: 11.7%).
   - If the mechanism checks pass and FA1 solves at most 20%, over-sharpening
     is rejected as the cause of the collapse, and the paper reports the
     repetition findings as associations.
   - Between 20% and 40% reads as partial protection.
2. **Dose.** After Stage B, forced accuracy orders FA1 > FA025 > 11.7%. At
   handoff, B orders them the opposite way: 0.161 > FA025 > FA1.
3. **Replication.** U2 solves at most 20% after Stage B within 16,384 tokens,
   and B(U2) ≥ 0.06.
4. **Dilution does not protect.** FN solves at most 20% after Stage B within
   16,384 tokens. This is read only if FN passes the drilling and forced-parity
   checks.

**How the result will be worded.** The anchor rehearses M0's forced reasoning,
the mode the probe measures. A protected FA1 therefore shows that keeping the
base's reasoning in reach during drilling prevents the collapse, and that
dilution alone does not (if FN collapses). It does not separate drilled
competence that lasts from base competence that was rehearsed. The paper will
say so.

**Reported alongside:** cap-hit rates, strict and resumed scores.

## Projected cost

About $190, within the $250 limit:

- Anchor samples: $26.
- Training FA1 and FA025: $30 each.
- Training FN: $14.
- Training U2: $15.
- Probes: $68.
- Divergence and fit scoring: $5.

## Launch

Fixed and launched 2026-09-24 01:48 UTC as run `anchor-20260924`, with
`--prior [redacted: account balance] --budget 250`. [redacted: account balance] Code SHA-256 at launch:

- `tools/run_anchor.py`: `80aed9f8d3ed0972…`
- `tools/analyze_anchor.py`: `b53a3887020b8b1d…`
- `tools/run_repetition_probe.py`: `cbbfd4ac0cc3ce32…`
- `tools/run_instruction_continuation.py`: `fd35889684fa2696…`

An independent pre-launch audit and its re-check found no blockers.

## Results (written 2026-09-24 05:26 UTC; run `anchor-20260924`, $213.0677)

No job failed. Forced `<think>` accuracy on the 192 targeted items, two draws,
template stops resumed. The F-u140 references come from Part D (handoff, within
4,096 tokens), the 16k consistency probes (handoff, within 16,384) and the cap
test (after Stage B).

| Arm | B | Drilled-text NLL drop, share of F's | Handoff, within 4k | Handoff, within 16k | After Stage B, within 16k | After Stage B, within 4k |
|---|---|---|---|---|---|---|
| F-u140 (reference) | 0.161 | 100% | 61.7 | 81.5 | 11.7 | 4.2 |
| FA025 | 0.024 | 99% | 53.6 | 78.6 | 27.9 | 7.6 |
| FA1 | 0.009 | 85% | 51.0 | 75.3 | 48.2 | 25.8 |
| FN (dilution) | 0.015 | 36% | 58.3 | 85.7 | 25.3 | 4.9 |
| U2 (replicate) | 0.172 | | 45.3 | 70.6 | 6.5 | 0.3 |

After Stage B, within 16k, each arm minus F-u140, with paired item intervals:
FA1 +36.5 [30.5, 42.4], FA025 +16.1 [10.9, 21.4], FN +13.5 [8.1, 19.0].
Search reaching the 16k cap after Stage B: FA1 49%, FA025 71%, FN 72%, U2 92%,
F-u140 87.5%.

**Registered readings.**

1. **Prediction 1: not interpretable.** FA1 solves 48.2% after Stage B
   (threshold 40), but it fails the registered forced-parity check. At
   handoff, within 4,096 tokens, it solves 51.0% against F-u140's 61.7%, a gap
   of 10.7 points against a limit of 5. FA025 fails the same check (53.6).
2. **Prediction 2: dose holds.** Accuracy after Stage B orders FA1 48.2 >
   FA025 27.9 > F 11.7. B orders them the opposite way: 0.161 > 0.024 >
   0.009.
3. **Prediction 3: replication holds.** U2 solves 6.5% after Stage B, and B =
   0.172.
4. **Prediction 4: not readable.** FN fails its drilling check: its NLL drop on
   the drilled texts is 36% of F's, against a minimum of 80%. The No Robots
   stream kept most of the drilling from happening.

**What the failed check means (not registered).** The anchor texts are long M0
searches: 2,458 of the 4,480 (55%) reach the 4,096 cap. The anchored arms learned to search
longer, which lowers accuracy within 4,096 tokens at handoff.
- Within 16,384 tokens the handoff gap shrinks to 2.9 points for FA025 and 6.2
  for FA1.
- F-u140's 16k handoff value did not exist when the check was fixed, so the
  registered reading stands.
- FA025 is the cleanest comparison. It fits the drilled texts as well as F does
  (99%), is within 3 points of F at handoff within 16k, cuts B by 85%, and
  keeps 16 more points after Stage B.

**Limits of the account.** The anchor nearly removes B (0.009 for FA1), but FA1
still loses 27 points within 16k (75.3 → 48.2). The lasting states lose 2–4:
M0 −4.2, T-u30 −3.4, S-u230 −2.1. B measured on M0's forced reasoning is
therefore not a sufficient statistic for fragility.
- A likely reason is the rehearsal caveat registered above. The anchor repairs
  exactly the `<think>` mode that B measures, while the drilled texts use the
  teacher's own formats.
- So the anchor protects in proportion to its weight, but B does not capture
  everything drilling does.
