# Does repetition make an imitated reasoning skill fragile?

Fixed 2026-09-23 01:27 UTC, before any training in this experiment. Same
authorization as [decision-competence.md](decision-competence.md); this
experiment's limit is **$120** [redacted: account cap], enforced by
its own ledger.

## Why

In the decision-competence run (`dc-20260923b`, Part A, Stage-B updates 0–6),
the RL teacher T, its unfiltered clone S and M0 all switched reasoning off at
about the same point (first update with half of openings skipping: T 3.8–4.1,
S 4.1–5.0, M0 4.9). The replay's trace checkpoints did not share one timing:
order-47 R-P after 20, 60 and 140 updates reached half-skip at none (through
update 5), 3.5 and 1.7. R-P cycles 579 success-filtered traces, so those
doses are 1.1, 3.3 and 7.7 passes. S saw 6,400 unfiltered samples at most 1.15
times and was never fragile. Its corpus holds almost no direct answers (0.3% of
targets under 32 tokens), so R-P's fragility was not copied from its data.

## Question

With the source (the frozen teacher's own samples), loss, learning rate and
number of updates fixed, does training many passes over a small set make the
reasoning switch off sooner under Stage B than one pass over a large set? Does
success filtering matter at fixed repetition?

## Arms

Both start from M0 (weights only, fresh optimizer) and use the clone's recipe
unchanged: sampled-token NLL normalized by the set's mean response length,
learning rate 1e-4, batch 32, Adam 0.9/0.95/1e-12, no clipping, max length 4,231.

| Arm | Examples |
|---|---|
| U-579 | 579 of the corpus's 800 prompts, one sampled completion each, unfiltered |
| F-579 | 579 of the 655 prompts with a correct, uncapped completion, one such completion each |

Prompts, completions and the cycling order are chosen by fixed SHA-256 keys
(namespace `repetition-20260923`), without looking at length or style. Each arm
trains 140 updates (7.7 passes). At updates 20, 60 and 140 (1.1, 3.3 and 7.7
passes) the weights branch into the unchanged Stage-B protocol of Part A: 20
updates, decision probes after updates 0–6, 8, 10, 12, 16 and 20.

Comparison: S (6,400 samples; S-u40, S-u120, S-u230 at 0.2, 0.6 and 1.15
passes) from `dc-20260923b`.

## Predictions, written before data

1. If repetition makes the skill fragile, U-579 and F-579 after 140 updates
   switch off sooner than S at similar update counts (S-u120, S-u230): higher
   mean skip share over Stage-B updates 1–6 and an earlier half-skip update.
   After 20 updates (1.1 passes) they behave like S.
2. If only F-579 switches off early, repetition needs success filtering.
3. If neither does, R-P's fragility comes from what else differs: its source
   (an RL run's training history rather than a frozen endpoint) or its
   per-example loss.

## Analysis

As in Part A: skip-share curves, first half-skip update, and mean skip share
over Stage-B updates 1–6 with paired item-bootstrap intervals against S-u120,
S-u230 and between arms. Descriptive; no gate.

## D. Does repetition also cost the competence? (added 01:43 UTC)

Fixed after reading only the decision probes of the 20- and 60-update branches
and, from Part A of the decision-competence run, one forced result: order-47
R-P after 140 updates solves 61.2% of targeted items forced into `<think>`
before Stage B and 2.6% after 20 Stage-B updates (90% of its forced searches
hit the 4,096-token cap), while T-u30, S-u230 and M0 keep 59.4%, 57.6% and
45.1%. The registered branches above carry decision probes only.

The saved stage-A states of both 579-example arms after 20 and 140 updates
(1.1 and 7.7 passes) are restored again, weights only, and run through the same
20 Stage-B updates, with the Part A forced probes (both prefixes, targeted, two
draws, template stops resumed) and decision probes after updates 0 and 20, and
the unforced baseline at update 0. Budget within this experiment's $120.

Prediction: if repetition is what made R-P's skill fragile, the 140-update
states lose most of their forced accuracy by Stage-B update 20 and the
20-update states keep it, as S does.

**Extension, 02:07 UTC,** after reading Part D's results (both 140-update states
fall to 0.3% and 3.1% forced by Stage-B update 20; both 20-update states keep
58.1% and 62.5%). The saved 60-update states (3.3 passes) go through the same
protocol to show whether the loss grows gradually with passes or appears
between 3.3 and 7.7. Cap $25; the experiment then totals at most $106 of its
$120.

## Results (runs `rep-20260923`, `rep-20260923-forced`; written 02:09 UTC)

Spend $81.38 of this experiment's $120 before the extension. Both arms trained
140 updates without error; the U set holds 227 capped and 22 direct-answer
completions (39% and 3.8%), the F set none capped and 2 direct answers.

**Decision.** Half-skip update (targeted), with the reference origins of
`dc-20260923b`:

| Passes | U-579 | F-579 | R-P (order 47) | S (6,400) |
|---|---|---|---|---|
| 0.2 / 0.6 / 1.15 | | | | 4.4 / 4.1 / 5.0 |
| 1.1 | 4.2 | 3.8 | none by update 20 | |
| 3.3 | 3.6 | 3.4 | 3.5 | |
| 7.7 | 2.7 | 3.7 | 1.7 | |

Mean skip share over Stage-B updates 1–6, targeted, first minus second [95%
item interval]: U-u140 − S-u120 +0.173 [+0.159, +0.186]; F-u140 − S-u120
+0.035 [+0.022, +0.048]; U-u140 − F-u140 +0.138 [+0.124, +0.152]; U-u140 −
R-P-u140 −0.113 [−0.130, −0.097]. Held-out families agree within 0.011.
Prediction 1 holds for the unfiltered arm only; prediction 2 is reversed:
repeating unfiltered samples, not filtered ones, moves the decision earlier,
and only U-u140 shows R-P's rise from the first Stage-B update (3, 8, 23, 60%).

**Competence (Part D).** Forced `<think>` accuracy on targeted items, before
and after 20 Stage-B updates (unforced accuracy before Stage B in brackets):

| State | Update 0 | Update 20 |
|---|---|---|
| U after 20 updates (1.1 passes) | 59.9% (43.8%) | 58.1% |
| F after 20 updates (1.1 passes) | 58.6% (51.8%) | 62.5% |
| U after 140 updates (7.7 passes) | 53.6% (33.9%) | 0.3% |
| F after 140 updates (7.7 passes) | 61.7% (55.7%) | 3.1% |

The Part D prediction holds for both arms. Seen about once, the teacher's
samples leave a competence that survives 20 Stage-B updates; drilled 7.7
times, filtered or not, they leave one that is gone by then, as with R-P-u140
(61.2% to 2.6%). F-u140 has the highest unforced score of any checkpoint in
these runs and loses its forced competence fastest. One subset and one Stage-B
realization per state.

**Extension result** (`rep-20260923-forced60`, $20.34). After 60 updates (3.3
passes): U 64.8% forced before Stage B (46.1% unforced), 46.4% after 20
updates; F 66.1% (65.6% unforced), 47.9%. Change in forced accuracy over 20
Stage-B updates, by passes 1.1 / 3.3 / 7.7: U −1.8 / −18.4 / −53.3 points, F
+3.9 / −18.2 / −58.6. The loss grows steadily with repetition in both arms.
For reference: R-P-u140 (7.7 passes) −58.6, S-u230 (1.15) +0.6, T-u30 −2.8,
M0 −9.8.

Signs of memorization do not separate the states before Stage B: on new
problems, forced completions of the 140-update states differ between draws as
much as the 20-update states' (mean 4-gram Jaccard 0.031 and 0.023 against
0.026 and 0.024) and share as little text with their own training set (3.9%
and 2.6% of 8-grams against 3.4% and 2.3%). Mean training loss on the cycled
batches falls modestly with passes (U 0.27, 0.18, 0.14 per token at updates
20, 60, 140).

## D2. A gentler continuation (added 02:15 UTC)

Fixed before any D2 sample. Stage B's learning rate (3e-4) is aggressive; a
drilled state could collapse because it is hit hard rather than because it is
fragile. The four Part D states (U and F after 20 and 140 updates) run the same
20 Stage-B updates at learning rate 1e-4, all else unchanged, with forced
probes (both prefixes, targeted, two draws) and a decision probe after update
20; update-0 values are Part D's, same states. Cap $20.

Prediction: the 140-update states still lose most of their forced accuracy
(below half of their update-0 value), and the 20-update states keep theirs.

## D3. A different continuation: ordinary instruction tuning (added 02:19 UTC)

Fixed before any D3 sample. Stage B is answer-only program synthesis, a
continuation that directly rewards skipping reasoning. D3 replaces it with
human-written instruction data: single-turn examples of the No Robots
collection from the locally cached Tulu-3 SFT mixture (shard 0), rendered
length at most 1,024 tokens, 640 distinct examples in a fixed SHA-256 order
(namespace `repetition-20260923-d3`), so 20 updates of 32 with no repetition.
Same optimizer and learning rate (3e-4) as Stage B. Origins: M0 and the Part D
states U and F after 20 and 140 updates. After update 20: forced probes (both
prefixes, targeted, two draws) and the decision probe; update-0 values are
Part A's and Part D's. Cap $25.

Prediction: the 140-update states lose most of their forced accuracy (below
half of their update-0 value) under instruction tuning too, while M0 and the
20-update states keep most of theirs.

## D2 and D3 results (written 02:26 UTC)

Forced `<think>` accuracy on targeted items after 20 continuation updates
(update 0 from Part D, same states); registered prefix in brackets.

| State | Update 0 | D2: Stage B at 1e-4 | D3: instruction tuning at 3e-4 |
|---|---|---|---|
| M0 | 54.9 (31.2) | | 53.6 (2.3) |
| U, 20 updates | 59.9 (47.7) | 61.5 (47.9) | 58.9 (24.7) |
| F, 20 updates | 58.6 (53.9) | 60.9 (55.7) | 57.8 (26.6) |
| U, 140 updates | 53.6 (40.6) | 29.9 (27.1) | 34.9 (1.6) |
| F, 140 updates | 61.7 (55.2) | 37.2 (38.3); first run (33.6) | 36.7 (2.1) |

F-u140 in D2 hit that run's $20 ledger cap before its `<think>` probe; it was
rerun alone (`rep-20260923-gentle-F140`, $5.07), and the two realizations agree
on the registered prefix (33.6 and 38.3). Under instruction tuning no origin
learns to answer first (skip share 0.4–1.9% after update 20), but D3b below shows
that every origin stops searching all the same; the registered prefix no
longer elicits the answer format for any origin, M0 included; the `<think>`
prefix does.

Both predictions fail as worded: the 140-update states lose 24 and 25 points
(D2) and 19 and 25 points (D3), about 40% of their forced accuracy, not more
than half. The contrast they were written to test holds in both: the 20-update
states and M0 lose at most 1.3 points. Instruction tuning wears down drilled competence as well (on
its decision effect, see D3b).

Spend for this experiment: $39.17 + $42.21 + $20.34 + $18.32 + $5.07 + $17.44 =
$142.55, above its $120 share. [redacted: account balance]

**Intervals** (paired item bootstrap over the 192 targeted problems, 20,000
resamples; change in forced `<think>` accuracy over 20 updates, points).
Drilled minus once-trained, same data source: U −51.6 [−60.4, −42.7] (Stage B),
−25.3 [−34.4, −16.1] (Stage B at 1e-4), −17.7 [−26.6, −9.1] (instruction
tuning); F −62.5 [−70.1, −54.9], −26.8 [−34.9, −18.5], −24.2 [−33.1, −15.4].
Every once-trained change includes zero (largest +3.9 [−1.3, +8.9]); M0
changes −9.9 [−15.9, −3.9] under Stage B and −1.3 [−7.6, +4.9] under
instruction tuning; 3.3 passes gives −18.5 [−24.5, −12.2] (U) and −18.2
[−24.5, −12.0] (F) under Stage B.

**How drilled competence fails** (exact verifier failure codes, forced
`<think>`, after 20 updates of instruction tuning). The drilled states' capped
share barely moves (U 41.4% to 43.0%, F 37.0% to 39.6%); what grows is wrong
answers (4.9% to 21.1%, 1.3% to 22.7%). Most violate the operand rule: 43 of
U-u140's and 75 of F-u140's failed answers reuse a number (for example
`(19 - 3) + 2 * (5 / 5) * (17 / 17)` for target 18, with the text asserting that
it "uses all numbers exactly once"), against 1 for U-u20. After Stage B at 3e-4
the drilled states fail differently: 90% of forced searches reach the cap.

## D3b. What an ordinary evaluation would show (added 02:30 UTC)

Fixed before any D3b sample. Under instruction tuning the checkpoints keep
choosing to reason (skip share under 2%), so their own unforced accuracy should
show the drilled loss without any forcing. The D3 continuation is rerun for U
and F after 20 and 140 updates with the unforced probe (targeted, two draws)
after update 20; unforced accuracy before is Part D's. Cap $10.

Prediction: the 140-update states lose at least 10 points of unforced
accuracy; the 20-update states lose at most 3.

**D3b result** (`rep-20260923-instruct-unforced`, $3.30). The premise was wrong.
Unforced accuracy after 20 instruction-tuning updates: U-u20 6.0% (43.8%
before), F-u20 10.2% (51.8%), U-u140 0.5% (33.9%), F-u140 0.5% (55.7%). The
prediction fails for the 20-update states. They do not answer first (skip
0.5–1.3%), but they stop searching: unforced completions shrink to a median
of 119 and 139 tokens (56 and 59 for the 140-update states) and mostly read as
one-line derivations such as `2+14+19+21+17=73` followed by the answer, which is
what the prompt's "concise derivation" asks for. The skip probe detects only
answer-first openings and missed this switch. Read with D3, instruction tuning
flips the decision for every origin within 20 updates while leaving the
competence of M0 and the once-trained states in place under forcing (57.8–58.9%)
and wearing down the drilled states' (34.9–36.7%). [redacted: account balance]

**Strict scores** (template stops counted wrong, stored beside every row).
Under Stage B the drilled collapse holds strictly as well: F-u140 52.9% to 2.3%,
R-P-u140 51.8% to 0.5%, U-u140 12.8% to 0.3%, while U-u20 and F-u20 go from
10.4% to 25.3% and 22.9% to 22.1%. Under instruction tuning the strict score
moves with how often a model restates the template, which the tuning reduces
for every state: U-u20 +16.4, F-u20 +14.1, U-u140 +14.8, F-u140 −21.9 points.
F-u140's loss survives strict scoring; U-u140's appears only once template
restatements are resumed (53.6% to 34.9%).

**Loss on the continuation's own batches** (same batches in the same order for
every origin, so values compare directly; `loss:sum` over 32 examples). Under
Stage B at 3e-4, after one update the 140-update states' loss on the next batch
jumps: U-u140 10.35, 19.50, 19.44 at updates 1–3 and F-u140 11.77, 26.91, 16.19,
while every non-drilled origin falls to 7.7–8.7 at update 2 (M0 8.06, T-u30 7.89,
S-u230 7.74, U-u20 8.05, F-u20 7.86, U-u60 8.52, F-u60 8.69). R-P-u140 does not
fall (12.90, 13.12); R-P-u60 (11.54) and R-S-u200 (12.32), also trained for
many passes, stay high as well. At 1e-4 the jump is smaller (12.38 and 15.93 against 8.37
and 8.41), and so is the competence loss. Under instruction tuning there is no
jump: the 140-update states stay within 2% of the others at every update (for
example 52.56 and 52.82 against 52.05–52.38 at update 2), and two independent
runs agree to 0.1. The drilled states are thrown off by the first steps of a
continuation that shares their task's answer format; under one that does not,
their arithmetic erodes without any sign in the new task's loss.

Across the eleven origins with forced probes under Stage B at 3e-4 (M0, T-u30,
S-u230, R-P-u140, R-S-u200, U and F after 20, 60 and 140 updates), the loss on
the continuation's second batch correlates with the forced accuracy lost by
update 20: Pearson r = −0.84 (rank correlation of the update-1-to-2 change
−0.77). The three origins whose loss did not fall lost 53–59 points; the rest
lost between −3.9 and 18.5. It is driven by those three, and under instruction
tuning the signal is absent while drilled competence still erodes, so it is a
lead for this continuation, not a general early warning.

## D4. Cliff or slope? (added 03:26 UTC)

Fixed before any D4 sample. The loss jump after the first Stage-B update
suggests the drilled states are thrown off by the continuation's first steps.
F-u140 (the correct-only set, 7.7 passes) runs Stage B again at 3e-4 with the
forced `<think>` probe after updates 1, 2 and 4 only (update 0: 61.7%, update
20: 3.1%). Cap $7. [redacted: account balance]
Prediction: most of the loss happens by update 2 (forced accuracy below 30%).
D4 was launched as `run_repetition_forced.py --run-id rep-20260923-cliff
--labels F-u140 --cap 7 --forced 1 2 4 --prefixes think`, continuing to update 4
with no decision probes; the script's defaults were restored afterwards and the
same run is now `... --last 4 --no-decision`.
The first D4 launch stopped at its $7 cap before any probe finished: three
concurrent forced probes hold about $9.40 of worst-case cost between them. It
spent $2.94 and wrote no samples. Relaunched at 03:28 UTC with the update-2
probe alone (`--forced 2 --last 2 --no-decision --prefixes think`, cap $3.50),
which answers the same question. [redacted: account balance]

**D4 result** (`rep-20260923-cliff2`, $2.44). F-u140 solves 45.1% forced with
`<think>` after two Stage-B updates (61.7% before, 3.1% after twenty). The
prediction fails: about a quarter of the loss has happened by update 2, when the
new task's loss jumps, and the rest drains over the following updates. The
jump marks the start of the damage, not the damage itself. [redacted: account balance]
No further Tinker work.
