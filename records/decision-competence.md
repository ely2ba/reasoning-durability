# Decision or competence: what later training removes from an acquired skill

Fixed 2026-09-23 00:32 UTC, before any sample in this experiment. [redacted: private message] The experiment ran under a spend cap of **$250**, enforced by
the runner's ledger at current list prices. Completed evidence is unchanged.

## Why

Splitting every saved Stage-B arithmetic completion by length shows that the
RL checkpoints (B-G) and the trace-trained checkpoints (R-P) solve arithmetic
only inside a long search. Searches are right 35–76% of the time before
Stage B; answers written without one are right 1.6% of the time. Stage B turns
the search off: after one update the RL checkpoints still search as often as
before (59%→58%, 56%→57%) while the trace checkpoints do not (49%→34%,
57%→31%). When the original trace checkpoint's search returned at update 10,
accuracy inside it was back at its starting level (67–74% against 70.5%).
Solver-trained checkpoints lose arithmetic differently: they keep writing
derivations while those derivations go wrong.

## Question

When continued training on an unrelated task removes a newly acquired
reasoning skill, does it remove the **competence**, or the **decision** to use
it? Does the way the skill was acquired decide how robust that decision is,
and does a behavioural copy of an RL checkpoint inherit its robustness?

## Instruments (block-11 monitor panel: 192 targeted + 192 held-out items)

- **Decision probe.** Four sampled 8-token openings per item, temperature 1,
  top-p 0.95. A completion *skips* if its text, stripped of leading
  whitespace, begins with `<answer>`. Readout: skip share per role.
- **Competence probe.** The completion is prefilled with
  `" Let's think step by step.\n\n"`; the model writes the rest (cap 4,096,
  unchanged stops); the unchanged exact verifier scores prefix + completion.
- **Unforced baseline.** Ordinary full completions, same settings.

## A. Route map: does the decision survive the next task?

Origins, restored weights-only with a fresh optimizer, then the unchanged
Stage-B recipe (MAPS records and order, batch 32, LR 3e-4, Adam 0.9/0.95/1e-12,
no clipping) for 20 updates:

| Route | Origins |
|---|---|
| Base model | M0 |
| On-policy RL (fresh 30-update teacher, endpoint-clone run) | T-u10, T-u20, T-u30 |
| Unfiltered behavioural clone of T-u30 (all its samples, failures included) | S-u40, S-u120, S-u230 |
| Success-filtered SFT on RL training-history traces (order-47 run) | R-P-u20, R-P-u60, R-P-u140 |
| Solver-derivation SFT (order-47 run) | R-S-u200 |

Decision probes after updates 0–6, 8, 10, 12, 16, 20. Competence probes
(targeted, two draws) after updates 0, 5 and 20. Unforced baseline (targeted,
two draws) at update 0. Several acquisition doses per route guard against a
single-checkpoint accident; no checkpoint is chosen from outcomes.

## B. Long horizon: does competence outlive the decision?

Existing saved Stage-B samplers, no training: original-pair R-P and R-S and
order-47 R-P and R-S at updates 0, 20, 40, 80, 160, 320, 480 (R-S at 0, 20,
80, 480), and M0. Competence probe on all 384 items, two draws; decision probe
alongside.

## Predictions, written before data

1. The RL teacher's skip share rises more slowly under Stage B than the
   success-filtered trace checkpoints', as B-G's did against R-P's.
2. The unfiltered clone is the open case. If it skips like the trace
   checkpoints, copying an RL policy's behaviour, failures included, does not
   copy the robustness of its decision to reason. If it holds like its
   teacher, the replay's fragility came from filtering or training history.
3. Forced to reason, trace and RL checkpoints stay well above both their own
   unforced accuracy and M0's forced accuracy after the decision has flipped.
4. Forcing a search opening does not rescue solver checkpoints above M0's
   forced level; the prefix restores a skill only where one was learned.

## Analysis

Skip-share curves per origin and role; the first update at which skip share
reaches 50%; forced against unforced accuracy; paired item-bootstrap intervals
(items resampled with all draws). Descriptive; no gate, no equivalence claim,
no pooling across routes into a single test.

## Deviations, recorded 2026-09-23 00:51 UTC before any Part A readout

1. The first launch (`dc-20260923`, stopped at $13.72) passed one seed per
   multi-sample call. Tinker then returns identical copies (8 of 8 identical
   when seeded, 8 of 8 distinct unseeded, three items on M0). Its draws were
   not independent, so it is kept only as an aborted record. The relaunch
   (`dc-20260923b`) makes one unseeded multi-sample call per item.
2. In that launch, order-47 R-P at Stage-B update 80 answered 0 of 768 forced
   items: it wrote one sentence after the registered prefix and then answered.
   The registered prefix can therefore under-elicit a search, so a second
   forced probe prefills `"\n<think>\n"`, the checkpoints' own most common
   search opening (verified token-for-token against B-G completions). Both are
   reported; the registered prefix stays primary.
3. For budget, forced probes use the 192 targeted items with two draws
   everywhere, and Part B probes updates 0, 20, 80 and 480 for all four series.
4. Recorded 01:08 UTC, after reading only Part B forced probes. Forced
   completions often restate the task's answer template inside the search;
   the unchanged stop at `</answer>` then ends them on
   `<answer>EXPRESSION</answer>` (the `<think>` prefix: 47–54% of completions).
   The registered unforced records are not affected: their Stage-B losses are
   answers written without a search, not template stops. From here on, a
   generation that stops on an answer tag containing no digit is resumed from
   its own tokens (at most 16 times, same total cap). Every other completion is
   scored exactly as before; the strict score is stored beside it. The 22
   Part B forced files collected under strict stopping are kept in
   `strict-stop/` and regenerated.
5. To stay under the cap, forced and unforced probes in Part A run only on the
   highest dose of each route and on M0 (M0, T-u30, S-u230, R-P-u140,
   R-S-u200), at updates 0 and 20. Decision probes keep all eleven origins.
   Part B drops cells Part A already covers (M0; order-47 updates 0 and 20).
   The aborted launch's $13.72 counts against the same $250 cap.
6. 01:20 UTC: the cap was raised to **$500** before later launches [redacted: private message]. The running process keeps its $250 check; later launches use
   $500, still counting every earlier launch.

## C. How long does the competence outlast the decision? (added 01:32 UTC)

Fixed before any Part C training. Part A shows every checkpoint except the
solver and the lightest trace dose skipping reasoning almost always by update
10, while forced `<think>` accuracy at update 0 is 55% for M0 and 62–68% for the
trained checkpoints (M0 solves about 9% unforced). The Pilot samplers that
would give the RL arm's long horizon have expired, so Part C continues the five
Part A forced origins (M0, T-u30, S-u230, R-P-u140, R-S-u200) through the full
480-update Stage B in one pipeline. The Stage-B recipe and batch order are
unchanged (checked token-for-token against the original run in Part A). Probes
after updates 80, 160, 320 and 480: decision (both roles, four draws) and both
forced prefixes (targeted, two draws, template stops resumed). Budget: $90
within the $500 cap.

Predictions, written before data:

1. Forced `<think>` accuracy stays near its update-0 level through update 80
   for every origin, M0 included, while all but the solver skip reasoning.
2. The trained checkpoints stay above M0 under forcing by about their update-0
   margin; what the continuation spares is mostly competence M0 already had.
3. By update 480 forced accuracy is near zero for every origin, as in Part B,
   so the competence is finite. The update at which it halves is at least ten
   times the decision's half-skip update.

## E. Two stress tests of "the checkpoints differ in decisions" (added 01:45 UTC)

Fixed before any Part E sample.

- **E1, the paper's coverage result.** The matched original pair (R-S@220,
  R-P@20) differs in coverage on the 256 targeted problems of the profile
  panel (16 draws each: 86 against 9 problems never solved). Forced into
  `<think>` (template stops resumed), draw 16 completions per problem from both
  checkpoints and from M0. Prediction: the solver checkpoint's never-solved
  count falls to near the trace checkpoint's, and M0's coverage is close to
  both; the coverage gap is a difference in the mode each checkpoint chooses.
- **E2, held-out families.** The Part A forced probes used targeted items only.
  Forced `<think>` on the 192 held-out (sentinel) monitor items, two draws, for
  the five Part A origins before Stage B. Prediction: trained checkpoints are
  close to M0 here too.

Budget: $70 within the $500 cap.

E1's M0 cell stopped at the Part E ledger's $70 cap (worst-case holds for
16 draws of 4,096 tokens). Relaunched alone at 02:06 UTC with the run's cap
raised to $110, i.e. $40 more, before any of its samples were read.

## Results (written 02:14 UTC; Part C update 480 pending, update 320 is at zero)

Runs: `dc-20260923b` (A, B), `dc-long-20260923` (C), `dc-panels-20260923` (E).
Stage-B training in A reproduced the original batches token for token (124 of
124 updates checked) and the original losses within about 2%; Part C's batches
match the original continuation's through update 480 (2,400 of 2,400 updates).

**A. The decision.** Half-skip update (targeted): M0 4.9; T-u10, u20, u30 3.8,
4.0, 4.1; S-u40, u120, u230 4.4, 4.1, 5.0; R-P-u20 none (it moves into a
`<think>` opening, 99% of openings at update 12), R-P-u60 3.5, R-P-u140 1.7;
R-S-u200 none by update 20 (30% skip). Held-out families track targeted ones
within 5 points at 146 of 149 probes (largest gap 6.9, R-P-u60 at update 3). Mean skip over updates 1–6, targeted: T-u30 −
S-u230 +0.157 [+0.145, +0.169]; T-u30 − S-u120 −0.016 [−0.028, −0.004]; S-u230
− M0 −0.020 [−0.037, −0.004]; S-u230 − R-P-u140 −0.459 [−0.475, −0.442].
Prediction 1 holds against R-P-u60 and R-P-u140, not against R-P-u20.
Prediction 2: the clone keeps its teacher's timing (the second branch).

**Forcing.** Targeted accuracy with the `<think>` prefix (template stops
resumed), unforced accuracy before Stage B in brackets:

| Origin | Update 0 | 20 | 80 | 160 | 320 | Halves at |
|---|---|---|---|---|---|---|
| M0 | 54.9 (13.5) | 45.1 | 39.3 | 15.6 | 0.0 | 120 |
| T-u30 | 62.2 (44.0) | 59.4 | 21.4 | 16.7 | 0.0 | 65 |
| S-u230 | 57.0 (37.8) | 57.6 | 47.7 | 25.0 | 0.8 | 148 |
| R-P-u140 | 61.2 (43.2) | 2.6 | 0.3 | 0.0 | 0.0 | 10 |
| R-S-u200 | 58.9 (39.3) | 40.6 | 35.4 | 16.1 | 1.0 | 105 |

The registered prefix gives the same ordering with lower levels (M0 31.2,
T-u30 52.6, S-u230 43.2, R-P-u140 57.3, R-S-u200 22.7 at update 0). Against M0
at the same update, forced `<think>`, [95% item interval]: T-u30 +7.3 [+1.3,
+13.3] at 0, +14.3 [+8.6, +20.1] at 20, −18.0 [−24.0, −11.7] at 80; S-u230
+12.5 [+6.2, +18.8] at 20, +8.3 [+2.3, +14.3] at 80; R-P-u140 −42.4 [−47.9,
−37.0] at 20. Forced after 20 updates, R-P-u140 searches without converging
(90% of completions reach the 4,096-token cap, 31% before Stage B).
Prediction 3 holds for T-u30, S-u230 and the original R-P@20 (Part B: 68, 66,
63% forced at updates 0, 20, 80, with 100% skipping from update 20), fails for
R-P-u140. Unforced completions before Stage B: M0 searches in 46% of
completions and is right in 28% of those; T-u30 and S-u230 search in 67% and
59% and are right in 64% and 63%.

**B.** Original-run samplers: R-P@20 forced 68/66/63/0% and R-S@220 65/54/41/0%
at updates 0/20/80/480; order-47 R-P@140 already at 0–1% by update 20 in its
own original run (strict-stop files), as in the Part A rerun.

**C.** Predictions 1 and 2 fail as worded: by update 80 T-u30 has lost most of
its forced accuracy (21.4%) and falls below M0, while S-u230 stays above M0
through update 160. Prediction 3 holds except for R-P-u140: forced accuracy
halves after 65–148 updates (105 for R-S-u200), 16 to 30 times the half-skip
update for M0, T-u30 and S-u230, and is gone by 320 for every origin. One Stage-B realization per origin; T-u30's dip at 80
may be a transient of that realization (it is level with M0 again at 160).

**E1.** Forced `<think>`, 16 draws on the 256 targeted profile problems:

| | Unforced accuracy | never / always solved | Forced accuracy | never / always | pass@4 | pass@16 |
|---|---|---|---|---|---|---|
| R-S@220 | 35.7 | 86 / 35 | 66.8 | 2 / 18 | 92.9 | 99.2 |
| R-P@20 | 34.6 | 9 / 0 | 67.7 | 0 / 31 | 94.0 | 100.0 |
| M0 | (≈9, other panel) | | 53.5 | 1 / 8 | 87.8 | 99.6 |

The prediction holds: forced, the pair's coverage gap disappears, and M0
already reaches 255 of 256 problems. Per-problem forced success correlates
0.71 between the pair and 0.64–0.79 with M0.

**E2.** Forced `<think>` on held-out families before Stage B: M0 54.2, T-u30
61.5, S-u230 58.9, R-P-u140 60.9, R-S-u200 62.2 (targeted: 54.9, 62.2, 57.0,
61.2, 58.9). The prediction holds.

Spend through 02:14 UTC: $13.72 (aborted) + $140.43 (A, B) + $65.80+ (C) +
$88.55 (E), plus $81.38 + $20.34 for the repetition experiment.

**C, update 480.** Every origin is at 0.0% forced (both prefixes) and 100%
skipping. Part C spent $88.02. [redacted: account balance]

**Strict scores for E1 and M0.** Counting template stops as wrong, forced
`<think>` coverage of the 256 profile problems is R-S@220 28.3% accuracy, 14
never solved, pass@16 94.5%; R-P@20 38.1%, 8, 96.9%; M0 12.8%, 52, 79.7%. The
pair's coverage gap (86 against 9 unforced) still shrinks to 14 against 8. The
statements about M0 do not survive strict scoring: M0 restates the template in
52.5% of forced completions, so its strict forced accuracy (12.8% here, 13.8% on
the monitor panel) equals its unforced accuracy. Resuming grants no extra
attempt (a restated template holds no expression, and the first expression the
model then writes ends the completion), which is why the resumed score is used
for competence throughout.

## C2. Is the update-80 gap between teacher and clone reproducible? (added 02:48 UTC)

Fixed before any C2 sample. Part C has one continuation per origin, and one of
its comparisons carries weight: at update 80 the clone keeps 47.7% forced and
the RL teacher 21.4%. T-u30 and S-u230 are continued again to update 80 with
the same recipe, with decision and forced probes after update 80 only. Cap $10.
Prediction: the clone stays above the teacher at update 80 by more than 10
points.

**C2 result and rerun.** S-u230 after 80 updates of the second continuation:
41.9% forced with `<think>` (first run 47.7%), 33.6% with the registered prefix
(first run 0.3%), 100% skipping. The `<think>` measure repeats within 6 points;
the registered prefix does not repeat at late updates and is not used for
conclusions there. T-u30's cell stopped at this run's $10 ledger cap before
its forced probes ($9.10 spent). It is rerun alone (`dc-long80-rep-T`, 03:08
UTC, `<think>` probe only, cap $5) before any of its C2 data were read.

**T-u30 rerun result** ($2.84). After 80 updates of the second continuation
T-u30 solves 53.9% forced with `<think>` (first run 21.4%) and skips reasoning
on 100%. The prediction fails: the clone is not above the teacher at update 80
in the rerun (41.9% against 53.9%). Two runs of the same checkpoint on the same
batches can differ by 32 points this late, so single long-horizon values and
the half-lives built on them (65, 148, 120, 105 updates) are loose; the
teacher's dip at update 80 was a transient of the first run. What holds across
runs: the non-drilled checkpoints keep most of their forced accuracy for tens of
updates after the decision has gone, while the drilled checkpoints fall to
0–3% by update 20 (R-P-u140 in two independent continuation runs). [redacted: account balance]

## C3. Do the decision clocks repeat? (added 03:20 UTC)

Fixed before any C3 sample. The half-skip updates of T-u30 (4.1) and S-u230
(5.0) each come from one continuation run, and the update-80 rerun showed that
late values can move a lot between runs. Both origins run the first six
updates again with decision probes after updates 1–6 only. Cap $5.
Prediction: both half-skip updates fall between 3 and 6 again, within one
update of their first-run values.

**C3 result** (`dc-decision-rep`, $1.97). Half-skip update, targeted, second
run against first: T-u30 4.04 against 4.12; S-u230 4.81 against 4.95. Skip
shares after updates 1–6 in the second run: T-u30 1, 5, 12, 48, 90, 98%;
S-u230 0, 2, 9, 25, 56, 68%. The prediction holds. The early decision curves
repeat to within 0.15 update, while forced accuracy 80 updates in can move by
32 points between runs. [redacted: account balance]

## Correction (2026-09-24 02:03 UTC, from a fact-check)

Deviation 2 calls `\n<think>\n` "the checkpoints' own most common search
opening, verified token for token against B-G completions". That holds for the
original RL checkpoints (B-G). It does not hold for the checkpoints studied
here.

- In the decision probes, only 1.3–5.1% of openings start with `\n<think>`.
- T-u30 and S-u230 open with " Thought:", and M0 with " Derivation:".

The prefix is a three-token opening of a reasoning block, which these models
answer with a search. It is not their own habitual opening. No number changes.
