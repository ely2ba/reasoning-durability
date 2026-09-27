# Does durable post-training leave the base's reasoning in place?

Fixed 2026-09-23 20:22 UTC, before any trained state was scored. [redacted: private message] The design scores
existing checkpoints only, at about $5. **Cap $6**, in this experiment's own
ledger. [redacted: account balance]

## Why

The two earlier experiments of this session
([decision-competence.md](decision-competence.md),
[repetition-fragility.md](repetition-fragility.md)) found three things:

- Every search-trained checkpoint stops choosing to reason within about four
  Stage-B updates.
- Forced into a `<think>` search, M0 and the checkpoints trained about once keep
  their accuracy for tens of updates.
- Checkpoints drilled 7.7 passes over 579 teacher samples lose that accuracy by
  update 20 and end below M0. Yet at handoff they are as accurate as the rest,
  their samples vary as much between draws, and they share no more text with
  their training set.

No handoff measure taken so far separates the two groups.

Proposed account: post-training that lasts leaves the base's own reasoning in
place and changes which mode the model enters. A later stage then flips the
mode, the reasoning underneath survives, and forcing the mode brings it back.
Drilling rewrites the reasoning itself, so the competence lives in the update,
and the next stage damages it.

## Question

At handoff, before any later training, do the lasting and the drilled
checkpoints differ in how far their reasoning has moved from M0's? And does that
distance, measured with no further training, rank how much competence each
checkpoint later loses?

## Instrument

We use teacher-forced log-probabilities from Tinker: `include_prompt_logprobs`,
plus the most likely next token from `topk_prompt_logprobs=1`. There is no
training, and each call samples one token, which is discarded.

**Texts.**

- *Common set.* M0's own forced `<think>` completions before Stage B: targeted
  items from Part A (`dc-20260923b/M0/u0`) and sentinel items from E2
  (`dc-panels-20260923/E2-M0/u0`). We take draw 0 of 96 items per role, chosen
  by SHA-256 order (namespace `depth-20260923`), for 192 texts. Each text is a
  sample from M0 at temperature 1 and top-p 0.95; generation resumed after a
  restated template continues from M0's own tokens, so the whole text is M0's.
- *Own sets.* Each state's own forced `<think>` completion before Stage B (draw
  0, the same 96 targeted items), for the ten states that have one: T-u30,
  S-u230, R-P47-u140, R-S47-u200, and U and F after 20, 60 and 140 updates.

We rebuild each response from its stored text: the three-token prefix
`\n<think>\n` followed by the re-encoded completion, cut at 1,024 tokens to
limit cost. Each response follows its rendered monitor prompt.

**States** (weights as saved before Stage B, 17 in all):

- M0.
- The RL teacher, T-u10, T-u20 and T-u30.
- Its clone, S-u40, S-u120 and S-u230.
- Order-47 R-P after 20, 60 and 140 updates.
- Order-47 R-S after 200 updates.
- U-579 and F-579, each after 20, 60 and 140 updates.

**Measures.** All are in nats per token over response positions 4 to 1,024;
positions 1 to 3 are the forced prefix.

- *B(θ), body divergence.* The token-pooled mean of `log p_M0 − log p_θ` over
  the common set. On M0's own samples this estimates KL(M0‖θ) per token, up to
  nucleus truncation: how much of the base's reasoning θ no longer produces.
- *B′(θ), own-sample divergence.* The token-pooled mean of
  `log p_θ − log p_M0` over θ's own set, which estimates KL(θ‖M0).
- *A(θ), top-1 disagreement.* The share of common-set positions where θ's most
  likely next token differs from M0's.
- Profiles of B and A over position bins 4–7, 8–15, …, 512–1,024.

**Durability outcome.** This is already measured: Stage B at learning rate
3e-4, forced `<think>`, targeted items, two draws. It is the change in forced
accuracy from update 0 to update 20, read from the existing probe files, for
M0, T-u30, S-u230, R-P47-u140, R-S47-u200, and U and F after 20, 60 and 140
updates.

## Predictions, written before data

1. **Repetition, not the amount of training, moves the reasoning.**
   B(S-u230) < B(U-u140) and B(S-u230) < B(F-u140), with the 95% paired
   item-bootstrap interval of each difference below zero. S-u230 had 230
   updates of the same recipe and source at 1.15 passes; U-u140 and F-u140 had
   140 updates at 7.7 passes. Secondary: B(S-u230) < B(U-u60) and
   B(S-u230) < B(F-u60) (60 updates at 3.3 passes).
2. **Separation.** Every lasting state has a lower B than every drilled state.
   The lasting states are T-u30, S-u230, U-u20 and F-u20 (forced change over 20
   Stage-B updates between −3 and +4 points). The drilled states are U-u140,
   F-u140 and R-P47-u140 (−53 to −59 points).
3. **Dose.** Within U and within F, B rises with passes: u20 < u60 < u140.
4. **Prediction from handoff.** Across M0 and the nine search-trained states
   with Stage-B forced data, the rank correlation between B and the forced
   accuracy lost by update 20 is at least 0.7. Handoff forced and unforced
   accuracy are reported against it. R-S47-u200, trained on a different kind of
   solution, is reported separately.
5. B′ orders the states as B does for predictions 1–3 (secondary).

All predictions fail or hold as worded. The account above loses its support if
prediction 1 or 2 fails.

## Prechecks (M0 only, before the main run)

Every remote precheck scores only M0, so no trained state's divergence is seen
before the main run.

1. **Leakage.** No common-set or own-set item is among the training prompts of
   any state: the clone corpus's 800 prompts and the RL training manifest.
2. **Re-encoding.** At least 95% of re-encoded completions have exactly their
   stored token count.
3. **Alignment.** A fresh 64-token M0 sample, scored teacher-forced, returns the
   sampler's own log-probabilities for the same tokens. Mean absolute
   difference must be under 0.01 nats and the maximum under 0.1.
4. **Noise floor.** Two M0 samplers, created independently, score eight common
   texts. The pooled mean difference must be under 0.002 nats per token.
5. **Top-1.** The top-1 log-probability is at least the actual token's at every
   position (tolerance 1e-3), and equals it where the actual token is the top-1.
6. **Cost.** Projected cost of the main run at list price must be at most $5.

## Analysis

- Point estimates of B, B′ and A per state.
- Paired item-bootstrap intervals over the common set's 192 items (20,000
  resamples), and item-bootstrap intervals for B′.
- Spearman correlations for prediction 4.
- The position profiles, reported descriptively.

No other gate.

## Results (run `depth-20260923`, written 20:32 UTC)

**Prechecks** (`precheck.json`, all passed before the main run):

- **Leakage.** No overlap with the 2,304 training prompts.
- **Re-encoding.** 1,129 of 1,152 re-encoded completions match their stored
  token count exactly; the other 23 differ by 1 to 3 tokens.
- **Alignment.** Teacher-forced log-probabilities of a fresh M0 sample differ
  from the sampler's own by 0.004 nats on average, and by 0.056 at most.
- **Noise floor.** Two independently created M0 samplers differ by 0.000005
  nats per token (mean absolute difference 0.00008); the same sampler repeats
  exactly.
- **Top-1.** Consistent at all 8,031 positions checked.
- **Cost.** The projection was $3.78; the run spent $3.80 of its $6 cap,
  prechecks included.

All five predictions hold.

| State | Updates | Passes | B [95% interval] | A | B′ | Forced points lost by update 20 |
|---|---|---|---|---|---|---|
| M0 | 0 | 0 | 0 | 0 | 0 | 9.9 |
| T-u10 / u20 / u30 (RL) | 10 / 20 / 30 | | −0.0006 / −0.0007 / −0.0004 | 0.008–0.015 | u30: 0.004 | u30: 2.9 |
| S-u40 / u120 / u230 | 40 / 120 / 230 | 0.2 / 0.6 / 1.15 | −0.0033 / −0.0045 / −0.0035 | 0.020–0.024 | u230: 0.020 | u230: −0.5 |
| U-u20 / u60 / u140 | 20 / 60 / 140 | 1.1 / 3.3 / 7.7 | −0.0027 / 0.0153 / 0.1072 | 0.021 / 0.040 / 0.082 | 0.017 / 0.040 / 0.097 | 1.8 / 18.5 / 53.4 |
| F-u20 / u60 / u140 | 20 / 60 / 140 | 1.1 / 3.3 / 7.7 | −0.0025 / 0.0157 / 0.1611 | 0.020 / 0.041 / 0.088 | 0.015 / 0.040 / 0.126 | −3.9 / 18.2 / 58.6 |
| R-P47-u20 / u60 / u140 | 20 / 60 / 140 | 1.1 / 3.3 / 7.7 | −0.0009 / 0.0131 / 0.0610 | 0.022 / 0.041 / 0.078 | u140: 0.099 | u140: 58.6 |
| R-S47-u200 (solver) | 200 | 11 | 0.0933 | 0.095 | 0.116 | 18.2 |

B is in nats per token; the 95% item intervals of every B are narrower than
±0.006. For scale, M0's own negative log-likelihood on these texts is 0.429
nats per token.

**The lasting states have not moved away from M0's reasoning.**

- Every one has B ≤ 0: it gives M0's own reasoning at least as much probability
  as M0 does, up to 0.0045 nats per token more.
- Each agrees with M0's most likely next token at 97.6–98.5% of positions.

**The drilled states have moved away from it.**

- They give M0's reasoning 0.061 to 0.161 nats per token less probability,
  which is 14% to 38% of M0's own negative log-likelihood on these texts.
- They disagree with M0's top choice at 7.8–8.8% of positions.

**Prediction 1: repetition, not training amount, moves the reasoning.**

- S-u230 − U-u140 = −0.111 [−0.115, −0.106].
- S-u230 − F-u140 = −0.165 [−0.170, −0.159].
- Against the 60-update states, the difference is −0.019 for both.
- B does not grow with the clone's updates (−0.0033, −0.0045, −0.0035 at 40,
  120 and 230 updates), and the RL teacher's stays within 0.001 of zero.
- Cycling 579 samples moves B from −0.003 (1.1 passes) to 0.015 (3.3 passes),
  then to 0.107 (U) and 0.161 (F) at 7.7 passes.

**Prediction 2: separation.** The highest lasting B is −0.0004 (T-u30); the
lowest drilled B is 0.061 (R-P47-u140).

**Prediction 3: dose.** B is monotone in passes in both arms (table).

**Prediction 4: prediction from handoff.** Across M0 and the nine
search-trained states, the rank correlation between B and forced points lost by
update 20 is 0.93 (0.89 with R-S47-u200). Handoff accuracy does not rank the
loss: forced 0.23, unforced 0.05. Top-1 disagreement A ranks it less well
(0.76). The Stage-B loss on the second batch ranks it as well as B does (0.94),
but it needs the continuation to be run.

**Prediction 5.** B′ agrees with B:

- S-u230 − U-u140 = −0.077 [−0.080, −0.074].
- S-u230 − F-u140 = −0.106 [−0.110, −0.101].
- Separation and dose hold.

**Where the divergence sits.** The drilled states follow M0 over the first 31
positions, which mostly restate the problem, and depart in the search itself.
F-u140's B is 0.03 over positions 16–31 and 0.10 to 0.21 from position 32 on,
highest over 128–255. From position 8 on, every lasting state stays within
0.005 of zero.

**Post hoc (not registered).**

- Using only the reasoning before M0's first restated template changes no B by
  more than 0.003.
- Targeted and sentinel items agree within 0.004.
- Every post-trained state is at least as confident as M0: mean top-1
  log-probability of −0.148 to −0.234, against M0's −0.237. The drilled U and F
  states are the most confident.
- R-P47-u140 is not unusually confident (−0.199), yet it sits at B = 0.061. Its
  departure from M0 is therefore not sharpening alone.

**Scope.** The solver R-S47-u200 is as far from M0 as the drilled states
(B = 0.093, A = 9.5%) but lost 18.2 points, not 53 to 59. For a skill M0 lacks,
leaving M0's reasoning is the skill itself. The reading "far from M0 means
fragile" is shown here for competence M0 already had.

**What this shows.** The checkpoints that keep their competence under later
training have sharpened M0's reasoning without moving it. The ones that lose it
have replaced it. Repetition, not the amount of training, is what replaces it:
the RL teacher's own samples, seen once, leave M0's reasoning in place; the
same samples cycled 7.7 times do not.

The measure is available at handoff, costs one forward pass per text on the
base model's own samples, and ranks the later loss where accuracy does not.

**Limits.**

- One base model, one task family, LoRA rank 32.
- The ten states form three dose clusters, so the rank correlation mostly
  reflects the clusters; the S-u230 contrast is the sharper test.
- The evidence is correlational. Drilling moves both B and durability, and no
  intervention has yet moved one without the other.

## Corrections after the audit (2026-09-24, about 01:00 UTC)

An independent audit recomputed every number above and found them correct.
[redacted: internal path] It
found that several sentences go beyond the data. They are corrected here; the
results above stay as written.

1. **B does not show "sharpen versus replace".**
   - About two-thirds of each drilled state's B (U-u140 69%, F-u140 69%,
     R-P47-u140 67%) comes from positions where the state keeps M0's most
     likely token. There, it gives the alternatives M0 actually sampled far
     less probability: 0.90 nats (U-u140) and 1.31 nats (F-u140) less.
   - At M0's branch points, its probability on M0's top token rises from 0.58
     to 0.70–0.73. The lasting states do the same at about a tenth of the
     scale (0.56 to 0.60).
   - The drilled states are therefore **over-sharpened**: they keep M0's
     choices and cut away most of its alternatives, and also flip M0's top
     choice at about 4% more positions. "Replaced" and "displaced" are wrong
     words for this; "sharpened versus over-sharpened" is what B shows.
2. **B ≤ 0 is an artifact of top-p sampling.**
   - The common texts were drawn with top-p 0.95. On such texts, B equals the
     divergence from M0's truncated distribution minus an offset of at least
     0.0033. Without truncation, B's expected value is never negative.
   - The lasting states do move: B′ is 0.004–0.020 (intervals exclude zero), A
     is 1.5–2.4%, and they give M0's sampled alternatives 8–11% less
     probability.
   - "Not moved away" and "at least as much probability as M0" are wrong. The
     0.0045 figure is S-u120, which has no durability outcome.
3. **The rank correlation of 0.93 reflects the three dose groups.**
   - A three-level group label alone gives 0.92.
   - Within groups, B does not rank the loss: 0.4 in the low group without
     M0, 0.0 in the high group.
   - The fair comparison is the number of passes, which gives 0.78.
   - What B shows is that it separates the fragile states from the lasting
     ones without knowing their training history. It does not show a finer
     ranking.
4. **The loss is non-terminating search.**
   - The forced accuracy lost matches the rise in 4,096-token cap hits
     (correlation 0.998 without R-S). After 20 Stage-B updates, U-u140 and
     F-u140 hit the cap on 99.5% and 96.4% of samples, and no capped sample is
     correct.
   - The endings are long enumerations, not loops.
   - Whether a larger budget recovers the answers is untested. That test is
     registered as the next step.
5. **Truncation.** 87% of M0's texts are longer than 1,024 tokens, so B covers
   37% of M0's tokens.
6. **Repetition means repeating identical completions.**
   - S-u230 sees each prompt 9.2 times, with 8 distinct completions. U/F-u140
     see each prompt 7.7 times with one completion.
   - The contrast is therefore between repeating the same text and seeing the
     same problems solved in fresh ways, with one run per arm. F is not
     matched to S in composition (0% capped against 37%); U is.
7. **The solver sentence is withdrawn.** R-S47-u200 is the most-repeated state
   (11 passes) yet lost only 18.2 points. It is a counterexample to "far from
   M0 means fragile", not an explained exception.
8. **Smaller fixes.**
   - "Not sharpening alone" is withdrawn (mean confidence is the wrong test).
   - B′ agrees with B only for predictions 1–3: it puts S-u230 above U-u20
     and F-u20, and R-P47-u140 above U-u140.
   - The Stage-B loss after one update is enough to rank the loss.
   - The drilled states already differ from M0 at 0.01–0.03 nats per token
     over positions 4–31.
   - F-u20's B′ is 0.014.
9. **Provenance.** After the run, `tools/run_depth_profile.py` gained a
   `length` argument (default 1,024) for later use. The run's code path is
   unchanged.

## Cap test (fixed 2026-09-24 01:01 UTC, before any sample)

**Question.** Correction 4 found that the drilled states' loss is searches that
do not finish within 4,096 tokens. Do they finish, and succeed, with four times
the budget?

**Method.** F-u140 and U-u140 (drilled) and S-u230 (lasting) rerun the
unchanged 20-update Stage B at 3e-4. They are then probed forced with
`<think>` on the 192 targeted items, two draws, with a 16,384-token cap and
template stops resumed as before. The run is `runs/repetition-probe/cap-20260924`,
from `tools/run_cap_test.py` [redacted: account].

Accuracy within 4,096 tokens is read from the same samples: a sample counts if
it finished correct within 4,096 tokens. Its first 4,096 tokens are
distributed exactly as in a 4,096-cap probe.

**Predictions, written before data.**

- If the competence is lost rather than slowed, F-u140 and U-u140 solve under
  15% forced within 16,384 tokens (3.1% and 0.3% within 4,096 in Part D).
- S-u230 gains at most 10 points over its 4,096-token accuracy (57.6% in Part
  A).
- Either outcome is reported. If the drilled states recover more than half of
  their handoff accuracy (61.7% and 53.6%), the paper's claim becomes "later
  training slows drilled search", not "removes the competence".

**Cap test result** (`runs/repetition-probe/cap-20260924`, $29.72, written
01:18 UTC). Forced `<think>` accuracy after 20 Stage-B updates, 384 samples per
state, template stops resumed. The budgets are read from the same samples.

| State | Within 4,096 | Within 8,192 | Within 16,384 | Over 4,096 tokens | At the 16,384 cap |
|---|---|---|---|---|---|
| S-u230 | 54.2% | 65.6% | 75.8% | 41% | 16% |
| F-u140 | 4.2% | 7.3% | 11.7% | 95% | 87.5% |
| U-u140 | 1.8% | 2.9% | 6.0% | 98% | 93% |

**Predictions.**

- The first holds. The drilled states solve under 15% even with four times the
  budget. They recover a fifth or less of their handoff accuracy (61.7% and
  53.6% at 4,096), and almost all their searches run to the cap. Later training
  has removed their competence, not slowed it.
- The second fails. S-u230 gains 21.6 points from 4,096 to 16,384 tokens in the
  same run, and 18.2 against Part A's 57.6%.

**What changes.** The 4,096-token cap used throughout the TCES experiments
understates the lasting states' competence: 41% of S-u230's searches need
more. It barely touches the drilled states' failure. Every earlier TCES number
reads "within 4,096 tokens", and the lasting-versus-drilled contrast is larger
than those numbers showed.

The within-4,096 values of this run match Parts A and D within sampling error:
54.2 against 57.6, 4.2 against 3.1, and 1.8 against 0.3.

## 16k consistency probes (fixed 2026-09-24 02:02 UTC, before any sample)

**Purpose.** A review found that the paper would mix token budgets. The
handoff values are within 4,096 tokens, while the cap test's post-Stage-B values
are within 16,384. This is a measurement for consistency, not a new
hypothesis.

**Probes.** Forced `<think>` probes with a 16,384-token cap, the 192 targeted
items, two draws, template stops resumed; strict scores are kept.

- At handoff: M0, T-u30, S-u230, F-u140 and U-u140.
- After the unchanged 20-update Stage B at 3e-4: M0 and T-u30. S-u230, F-u140
  and U-u140 already have these values from the cap test.

**Runs.** `runs/repetition-probe/cap16k-{u0,stageb}-20260924` via
`tools/run_cap_test.py` [redacted: account balance].

**Expectations.**
- At handoff, within 16,384 tokens, every state solves at least its
  within-4,096 value.
- After Stage B, M0 and T-u30 keep at least half of their handoff value, as
  S-u230 did (75.8%).

Both runs report strict and resumed scores.

**16k consistency result** (written 2026-09-24 05:26 UTC; $23.61 + $10.92). Forced `<think>`, 192
targeted items, two draws, template stops resumed.
- Values are within 16,384 tokens, with within-4,096 and strict scores in
  brackets.
- After Stage B, S-u230, F-u140 and U-u140 come from the cap test.

| State | Handoff | After 20 Stage-B updates | Change |
|---|---|---|---|
| M0 | 77.6 (50.0; strict 19.3) | 73.4 (50.8; 18.8) | −4.2 |
| T-u30 | 78.1 (58.6; 22.9) | 74.7 (55.5; 34.6) | −3.4 |
| S-u230 | 77.9 (54.2; 26.3) | 75.8 (54.2; 10.9) | −2.1 |
| F-u140 | 81.5 (60.4; 68.5) | 11.7 (4.2; 4.9) | −69.8 |
| U-u140 | 75.5 (52.9; 17.4) | 6.0 (1.8; 0.8) | −69.5 |

Both expectations hold.

- **All five start level within 16k (75.5–81.5).** The later stage costs the
  lasting states 2–4 points and the drilled states about 70.
- **Strict scoring makes the drilled F-u140 look far best at handoff** (68.5%
  against 17.4–26.3%), because it rarely restates the answer template. It then
  collapses to 4.9%.
- **The within-4,096 values** of these new samples agree with Parts A and D to
  within about 5 points, by sampling: M0 50.0 against 54.9, T 58.6 against
  62.2, S 54.2 against 57.0, F 60.4 against 61.7, U 52.9 against 53.6.
