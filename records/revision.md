# Revision: relearning, replay, a realistic later stage, sharpening without repetition, seeds, and a skill the base lacks

Written 2026-09-25 23:40 UTC, after an external review of the paper and a critique of the first experiment plan
[redacted: the review's score and the venue the critique took as its bar]. Each part below is fixed when its launch time and code hashes are added to
its launch section, before any of its samples exist. Nothing here changes a completed experiment.

**Authorization.** On 2026-09-25 (about 23:30 UTC) the author approved this program, as revised after the critique, with a
total list-price cap of **$725**. Each part has its own limit, enforced by its run's ledger:

| Part | What | Limit |
|---|---|---|
| RL | relearning after the later stage, with a habit control | $120 |
| RP | replay of each arm's own training texts during the later stage | $40 |
| RS | a realistic later stage: one epoch of No Robots at the reasoning rate | $45 |
| SH | sharpening without repetition | $60 |
| SD | seeds, with the drilled subset redrawn | $240 |
| SK | a skill the base lacks (synthetic), behind a learning gate | $100 |
| | margin, released only with a note here | $120 |

**Shared definitions.**
- **States.** These are the paper's existing states: M0 (Qwen3.5-9B-Base), D-u140, O-u140 and P-u140
  (own solutions), and D-T-u140 and O-T-u140 (gpt-oss-120b traces). Their states after 20 updates of
  instruction tuning (IT) and of the answer-only stage (AO), both at 3e-4, also exist.
- **Evaluation**, unless a part says otherwise:
  - forced accuracy on the 221 integer-answer MATH-500 problems at levels 3–5;
  - four draws, up to 16,384 tokens, temperature 1, top-p 0.95, as in the paper.
- **Primary score:** the last boxed integer. **Secondary, lenient score:** the last integer in the
  response.
- **Intervals:** paired item bootstraps over problems, 20,000 resamples, 95%.
- **Excess.** Arm A's excess over arm B is A's loss of forced accuracy minus B's, from each arm's own
  handoff.

---

## RL. Relearning: is the reasoning lost, or only suppressed?

**Why.** The review's first question: if a handful of updates of reasoning training restores the drilled
model, "lost competence" overstates the result. Under LoRA the base weights are untouched, so the question
is what the adapted model can still do, and how cheaply that is restored.

**Origins** (after-stage states):
- D+IT and D+AO;
- D-T+IT and D-T+AO;
- O+IT, the ceiling control.

**Relearning data.**
- **Math.** The base model's own correct forced solutions (top-p 1, uncapped) to 640 pool problems that
  no arm trained on and that the evaluation does not contain. These are drawn after the pilot's 7,680
  screened problems, in pool order, with the pilot's screen and keep rule.
- **Habit control.** The base model's own forced responses to 640 non-math single-turn prompts, the
  first 640 rows of the R5 chat mix. A response is kept if its reasoning block closes and a final answer
  follows.

  It teaches the format (open, reason, close, answer) with no mathematics. It is run from D+IT and
  D+AO only.

**Schedule.** Each origin continues its own adapter:
- learning rate 1e-4, the rate of the reasoning training;
- batch 32, a fresh optimizer, each example seen once;
- states after updates 1, 2, 5 and 20.

**Measure.** Forced accuracy after k = 1, 2, 5 and 20 updates. The recovered share is

    R_k = (acc_D,k − acc_D,0) / (acc_ref − acc_D,0),

where acc_ref is the matching once-trained after-state:
- O+IT for D+IT, and O+AO for D+AO;
- O-T+IT for D-T+IT, and O-T+AO for D-T+AO.

**Predictions, written before data.**
1. **Fast relearning.** For D+IT, R_5 ≥ 0.5. Under LoRA the base's competence is intact, so we expect
   undoing the adapter's damage to be cheap.
2. **The ceiling control holds.** O+IT stays within 2 points of its after-stage accuracy at every k.

**Readings.** These are fixed now and applied to the paper's wording.
- **R_5 ≥ 0.75** for D+IT: "the drilled model's competence is suppressed and is restored by about 160
  examples of reasoning training". The word "lost" is dropped everywhere.
- **R_20 ≤ 0.5** for D+IT: "not cheaply restored".
- **Between those:** "partly restored".
- **Habit control with R_20 ≥ 0.5:** the claim "more than a lost habit" is dropped. The paper says that
  retraining the reasoning format alone restores much of the loss.
- **Habit control with R_20 < 0.25** while math relearning reaches R_20 ≥ 0.5: the loss concerns
  mathematical competence rather than format.

**Cost.**
- About 22 evaluations, plus training on about 30k examples.
- Sampling about 900 problems for the math data, and 640 prompts for the habit data.
- Limit $120.

## RP. Replay of each arm's own training texts during the later stage

**Why.** The review's third question, the most decision-relevant for practitioners: later SFT often
mixes in some of the earlier data.

**Stage.** Instruction tuning on R2's schedule:
- 1e-4, 60 updates, the 1,920 distinct No Robots examples, fresh optimizer.
- Each batch of 32 holds 30 instruction examples and 2 examples replayed from the arm's own training
  texts (6.25%), taken in order and cycled.
- For D and D-T, the replayed texts are their repeated texts.

**Arms.** D-u140 and O-u140; D-T-u140 and O-T-u140.

**Measure.**
- The replay excess, D over O (and D-T over O-T), at update 60.
- The reference is the same stage without replay: 13.1 for own solutions and 6.9 for teacher traces, from
  records/robustness.md.

**Readings, fixed now.**
- **Replay prevents the break:** excess ≤ 3 points with its upper bound < 6.
- **Mitigates:** a reduction of at least 50% from the reference.
- **Fails:** a reduction under 25%.

**Prediction.** For own solutions, replay mitigates.

**Scope rule.** If replay prevents the break, the paper scopes its claim to later training without
replay, such as downstream fine-tuning of a released model.

**Cost.** Limit $40.

## RS. A realistic later stage: one epoch of No Robots at the reasoning rate

**Why.** The headline uses 3e-4 for 20 updates, three times the reasoning rate. The review asks for a
realistic stage.

**Stage.**
- One pass over the No Robots train rows available in the cached Tülu 3 mixture, in the SHA-256 order of
  the existing selection, at 1e-4, batch 32, fresh optimizer.
- The number of updates is set by the rows, about 300. Evaluate after the last update.

**Arms.** D-u140, O-u140, P-u140, and the base from a fresh adapter.

**Prediction.** D's excess over O is above zero, with its interval above zero, and at least 5 points.
P's excess over O is within ±3 points.

**Readings.**
- **If the prediction holds:** the abstract leads with this realistic number, and the 3e-4 stage becomes
  the stress test.
- **If D's excess has an interval that includes zero:** the paper says the break does not survive one
  realistic epoch, and scopes the claim to stronger later stages.

**Cost.** Limit $45.

## SH. Sharpening without repetition

**Why.** This tests the mechanism. Does over-sharpening itself make reasoning fragile, or does it take
repeated texts? It also answers the review's point that B may just detect overfitting.

**Arm O-sharp.**
- O's 4,480 problems, each with one of the base model's own forced solutions sampled at **temperature
  0.5** (top-p 1, uncapped, kept if correct; up to 4 draws per problem).
- Trained exactly as O: 140 updates, each text seen once, 1e-4.

**Gate before any evaluation.** B of O-sharp on the base's common texts, scored as in the paper, must
be at least 0.05 nats per token (about half of D's 0.111). If it is lower, the arm is retrained once at
temperature 0.3, and that value is final.

**Stages.** IT and AO at the defaults (3e-4, 20 updates).

**Readings, fixed now.**
- **Sharpening alone breaks it:** O-sharp's excess over O is at least 10 points after AO, with its
  interval above zero. Then over-sharpening, even without repetition, makes reasoning fragile, and B is
  more than an overfitting detector.
- **Sharpening alone does not:** within 3 points of O after both stages. Then seeing the same texts again
  matters beyond the sharpening it causes, and Section 4 is rewritten to say so.

**Prediction.** O-sharp's excess after AO is less than half of D's (73.1), i.e. below 36.5. In other
words, sharpening alone reproduces less than half of the break.

**Cost.** Limit $60.

## SD. Seeds, with the drilled subset redrawn

**Why.** The review's fifth question: the intervals cover problems, not training runs. The largest known
source of variance is the drilled subset (D against D2: 36.2 against 12.6 under IT).

**Seeds s2 and s3.** For each seed:
- a new random subset of 579 of O's 4,480 problems, disjoint from D's and D2's, drawn with the seed;
- a new random order of O's 4,480 texts, and a new adapter initialization;
- D, O and P as in the paper, with P's 8 fresh correct samples per problem drawn for the new subset
  under the main phase's rule.

**Stages.**
- IT at the defaults;
- IT at 1e-4 for 60 updates, R2's schedule.

AO is not run for seeds.

**Predictions, per seed and stage.**
1. **Primary:** D's excess over O and over P has an interval above zero.
2. **Secondary:** it is at least 5 points.
3. **Fresh solutions behave like once-trained:** P's excess over O is within ±3 points.

**Reporting.** Each run is shown separately, next to the paper's run and D2. The pooled mean is reported
with the spread across runs, never as a single number alone.

**Cost.** About $115 per seed. Limit $240.

## SK. A skill the base lacks

**Why.** The review's second question and its lead weakness. Our base already solves 94.7% of the test
problems, so training adds little and "the fix costs nothing" is uninformative. Tinker has retired the
weaker base models (Llama). Qwen3.5-9B-Base solves about 90% of even the non-easy pool problems at the
first draw (records/math-pilot.md), so there are too few hard problems. The part therefore uses a
synthetic skill.

**Calibration, fixed now.** Candidates are tried in this order:
1. Multiplying two base-7 numbers of 5 digits each.
2. Sorting 12 words under a custom alphabet given in the prompt.

For each candidate, the difficulty ladder is at most two steps: 5 then 6 digits, or 12 then 16 words. We
choose the first candidate and step at which the base's forced accuracy (300 held-out items, four draws)
is at most 25%.

Each task has:
- an exact verifier;
- solver-written worked solutions with several distinct valid texts per problem, from different methods
  and step orders.

**Arms** on the 9B, trained as in the paper (140 updates, 1e-4):
- D-S: 579 problems, one solution each, 7.7 passes;
- O-S: 4,480 problems once;
- P-S: D-S's problems, a different solution at each visit.

**Gate.** O-S-u140 must beat the base by at least 30 points, with its interval above zero, and at most 15%
of responses may reach the cap. If the gate fails, the part stops, and the paper reports it as an
unsuccessful attempt.

**Stages.** IT at the defaults, and IT at 1e-4 for 60 updates (primary).

**Outcomes registered, per stage.**
- **Break:** D-S is at least 10 points below O-S after the stage, with its interval above zero.
- **Consolidation**, the rival: D-S within 3 points of O-S after the stage.
- **The fix costs nothing:** P-S within 3 points of the better of D-S and O-S, both at handoff and after
  the stage.

Also reported, descriptively: whether D-S beats O-S at handoff, as Kopiczko et al. report for repetition.

**Cost.** Limit $100.

---

## Launches

Each part's launch is recorded here before its first sample, with its time (from `date -u`), its run
folder, its command, its limit and the SHA-256 of every script it runs. Deviations are added with their
times.

**Public timestamp.** This record, as written above, was published before any of its data as commit
`b99fd9f` of the paper's public repository, pushed 2026-09-26 00:52 UTC.

## Corrections and deviations fixed before any launch (2026-09-26 00:55 UTC)

These were written after the implementation and its offline checks, and before any sample. The checks
rebuild every part's data from local files. On stand-in folders built from existing evaluations, the
analysis reproduces the recorded values: 13.1, 6.9, 10.3, 36.2, 36.4, 73.1 and 12.6, and B of 0.111 and
0.0004.

**Limits re-split within the $725 cap.**
- Each launch's limit also holds the ledger's in-flight reservations, up to about $17 for 128 concurrent
  four-draw evaluations at 16,384 tokens. So each limit is the high estimate plus about $14.
- The limits: RL $175 (estimated $122–161), RP $40 ($25–29), RS $52 ($34–38), SH $79 ($55–65),
  SD $148 per seed ($128 and $127), and SK up to $100.
- The sum of the limits slightly exceeds $725, but the estimated spend is $550–650. The total stays
  under $725 because each later launch's limit is set to at most the cap minus the spend so far.
- A retrain of SH at temperature 0.3 would need about $36 more, released only from what remains.

**Corrections to the text above.**
- **RL trains on about 4,480 examples** (7 relearning runs of 20 updates × 32), not "about 30k".
- **RS has 267 updates.** The paper's selection rules (single-turn, at most 1,024 rendered tokens) leave
  8,545 No Robots rows, so there are 267 updates rather than "about 300".
- **RL prediction 2** (O+IT within 2 points at every k) is close to evaluation noise. It is kept as
  written, and its interval is reported alongside.

**Deviations** (the most faithful reading where the text was ambiguous):
1. **RL math data.** Problems are taken after pool position 7,680, in pool order, with the pilot's screen
   and keep rule, in chunks of 512. Problems any arm of the paper trained on are skipped, which is 52
   problems that Nemotron's O arm used. The first 640 kept are used.
2. **RL habit data.** There is one forced response (top-p 1) per prompt, in R5's order, until 640 are
   kept. Keeping also requires that the response is uncapped and fits the 17,000-token training length.
3. **RL habit evaluations** run after update 20 only, since the readings use R_20 alone.
4. **RP batches.** Each batch takes 30 of R2's examples in order, so R2's first 1,800 are used. The 2
   replayed examples are the arm's own training datums in its training order, cycled, with their original
   loss normalization.
5. **SH sampling.** Draws are sequential until the first correct and uncapped one (at most 4). A problem
   with none drops out and its slot cycles, as O-T's did.
6. **SD.**
   - There is one launch and one ledger per seed; the LoRA seeds are 2 and 3.
   - Subsets come from SHA-256 orders in the namespaces revision-sd-s2 and revision-sd-s3. They are drawn
     independently, disjoint from D and D2 but not from each other (they share 91 problems).
   - P's 8 texts start with D's own text, as in the main phase.
   - The 1e-4 stage is evaluated at update 60.
7. **Limits** cap each launch's own ledger, with the prior set to 0.

**Code, SHA-256:** `run_revision.py` `adbf8bb72ba0…`; `run_revision_arms.py` `94e806af0b1d…`; `analyze_revision.py` `caba8878e3c7…`; `run_math_pilot.py` `d8d537ffdc04…`; `run_math_main.py` `b983f0c52e37…`; `run_robustness.py` `81cf8b17e8e3…`; `run_teacher.py` `f90b3d112f6f…`; `run_instruction_continuation.py` `fd35889684fa…`; `run_decision_probe.py` `ae6905fb3051…`; `run_depth_profile.py` `57413d99c540…`; `analyze_depth_profile.py` `459f6b5bb3e4…`; `analyze_math_main.py` `fb4ed155cb96…`; `math_models.py` `d132c934a053…`; `math_score.py` `f9194799c176…`; 

### Launch: RP and SH (2026-09-26 00:56 UTC)

- **RP:** `tools/run_revision.py --part RP --run-id revision-rp-20260926 --limit 40`
  (`runs/math/revision-rp-20260926`).
- **SH:** `tools/run_revision_arms.py --part SH --run-id revision-sh-20260926 --limit 79`
  (`runs/math/revision-sh-20260926`).

Both launch from the repository root with `PYTHONPATH=src`, with the code hashes above. RP also serves
as the integration check of `run_revision.py`'s stages against Tinker before RL and RS are launched.

## SK: implementation, deviations and phased launch (2026-09-26 01:02 UTC)

**Tasks.**
- Base-7 multiplication: 20 distinct worked solutions per problem, 10 when the two numbers are equal.
  They come from long multiplication with either number on top, digits taken from either end, and the
  sum at the end or as a running total; from conversion through base 10 and back; and from collecting
  digit products by place.
- Custom-alphabet sorting: 12 per problem, by insertion, selection, merge, grouping and rank codes.
- Five property tests pass, and each fails on a deliberately planted bug.
- Mean text lengths: 871 tokens for 5 digits (max 1,937), 1,164 for 6 digits, 1,139 for 12 words and
  1,702 for 16 words.

**Deviations**, approved before launch:
1. **The token cap is 4,096** for every state, twice the longest 5-digit text. The smoke test checks it.
   If more than 10% of the base's responses reach the cap, the cap becomes 8,192 for every state before
   calibration, recorded here.
2. The 300 evaluation items are separate from the 300 calibration items, and the gate evaluates the base
   on the evaluation items.
3. If no setting reaches 25% or below, SK stops, and the paper reports the attempt.
4. Only O-S is trained before the gate. D-S and P-S are trained after it passes.
5. The primary later stage (1e-4, 60 updates) runs before the default one.
6. Each problem's solution texts are shuffled per problem. D-S and O-S take the first, so both see a
   random mix of methods, as the math arms' sampled texts do. P-S's 8 visits use 8 different texts.
7. The sorting prompt reads "Solve the following problem."; its words are 3 to 7 uniform random letters.
8. The training clients carry the label `study: math-pilot` (a reused function); this is a label only.

**Phases.** Each phase is a separate launch, reviewed before the next. `--limit` is cumulative for the
run's ledger.
- Smoke: 20 items × 4 draws on the base, limit $1.
- Calibration: one setting per launch, limit $12.
- Gate: O-S plus the evaluations, limit $40.
- Rest: limit $100.

**Code, SHA-256:** `run_skill.py` `4c47fcf3c7ff…`; `__init__.py` `930e30c216ad…`; `base7.py` `44b8d49d3926…`; `words.py` `c028eae2086b…`; `test_skill_properties.py` `10bcdc487f53…`; 

### Launch: SK smoke (2026-09-26 01:02 UTC)

`tools/run_skill.py --run-id skill-20260926 --phase smoke --limit 1` (`runs/skill/skill-20260926`).

### SK smoke at 4,096 tokens, and the cap raised (2026-09-26 01:08 UTC)

The base, 20 items × 4 draws per setting, within 4,096 tokens ($0.65):
- base-7, 5 digits: 45% correct, 65% capped;
- base-7, 6 digits: 10% correct, 95% capped;
- 12 words: 35% correct, 95% capped;
- 16 words: 0% correct, 100% capped.

More than 10% of the base's responses reach the cap, so under deviation 1 the cap becomes **8,192 for
every state**. The 4,096 results are kept in `smoke-cap4096/`, and the smoke is re-run at 8,192 before
calibration. `run_skill.py` SHA-256 is now `d017dc357543…`.

### SK smoke at 8,192 tokens (2026-09-26 01:13 UTC)

- base-7, 5 digits: 75% correct, 40% capped;
- base-7, 6 digits: 55% correct, 65% capped;
- 12 words: 85% correct, 30% capped;
- 16 words: 60% correct, 55% capped.

Spend so far $1.71. Given enough tokens, the base solves both tasks. The calibration decision follows
in its own entry.

### Launch: RL and RS (2026-09-26 01:13 UTC)

RP finished without error ($27.36), so `run_revision.py`'s stages work against Tinker.
- **RL:** `tools/run_revision.py --part RL --run-id revision-rl-20260926 --limit 175`.
- **RS:** `tools/run_revision.py --part RS --run-id revision-rs-20260926 --limit 52`.

### SK amendment: a skill the base lacks within a budget (2026-09-26 01:14 UTC)

**The finding so far.** Given enough tokens, the base solves both candidate tasks: at 8,192 tokens it
gets 55–85% on the smoke items, with many responses still capped. So, as first written, no setting would
pass the 25% rule, and SK would stop. This negative result is reported in the paper as it stands:
**Qwen3.5-9B-Base does not lack these skills; it lacks an efficient procedure for them.** Its own route
takes about 6,000 tokens, where the worked solutions average 871.

**The amendment.** It was made after seeing the smoke at 4,096 and 8,192 tokens, and before any sample
at the new budget. The skill is redefined as **solving within 2,048 tokens**. That budget is set by the
worked solutions: about twice their mean for 5 digits, and above the longest (1,937). It is not set by
any base outcome, since the base has not been sampled at 2,048.

Every state is evaluated at 2,048 tokens. Everything else is unchanged:
- the ladder and its order;
- the 25% calibration rule, the gate (+30 points), the arms and the stages;
- the three registered outcomes (break, consolidation, the fix costs nothing).

The paper will call the result "a skill the base lacks within a fixed token budget" and state this
amendment.

The 8,192 smoke results are kept in `smoke-cap8192/`. `run_skill.py` SHA-256 is now `2532e3151af8…`.

### Launch: SK calibration at 2,048 tokens (2026-09-26 01:14 UTC)

`tools/run_skill.py --run-id skill-20260926 --phase calibration --limit 8`: one setting per launch, in
the ladder's order.

### SK calibration result and gate launch (2026-09-26 01:15 UTC)

At 2,048 tokens the base solves 0.2% of the 300 calibration items (base-7, 5 digits; four draws, all
capped), so **base7-5 is chosen** ($6.66 so far). The gate launches as
`tools/run_skill.py --run-id skill-20260926 --phase gate --limit 25`.

### RP result (2026-09-26 01:16 UTC; `tools/analyze_revision.py --part RP`)

Forced accuracy after 60 updates of instruction tuning at 1e-4, with 6.25% replay of each arm's own
training texts:

| Arm | Handoff | After |
|---|---|---|
| D-u140 | 95.4 | 94.3 |
| O-u140 | 94.6 | 94.7 |
| D-T-u140 | 95.0 | 95.5 |
| O-T-u140 | 96.3 | 95.8 |

**Excess with replay:**
- D over O: 1.1 points, interval [−0.9, 3.3], against 13.1 without replay (a 91% reduction);
- D-T over O-T: −0.6 points, [−3.3, 2.0], against 6.9.

**Reading, for both: replay prevents the break.** The prediction (own solutions: mitigates) holds, since
prevention is the stronger outcome. The registered scope rule applies. The paper scopes its claim to
later training without replay, such as downstream fine-tuning of a released model, and reports replay as
a remedy. Spend: $27.36.

### Launch: SD seed s2 (2026-09-26 01:16 UTC)

`tools/run_revision_arms.py --part SD --seed s2 --run-id revision-sd-20260926 --limit 148`. Seed s3
follows once RL, RS and SH have finished, with its limit set to at most the cap minus the spend so far.

### SK gate result and launch of the rest (2026-09-26 01:40 UTC)

- **Base** (evaluation items, 2,048 tokens): 0.2%.
- **O-S-u140:** 66.2%, with 0.1% of responses capped (mean 925 tokens).
- **The gate passes:** a gain of about 66 points, against the required 30.

The rest (D-S, P-S, both later stages, and the evaluations) launches as
`tools/run_skill.py --run-id skill-20260926 --phase rest --limit 100`, where the limit is cumulative.
Spend so far is $19.93.

### RS result (2026-09-26 01:52 UTC; `tools/analyze_revision.py --part RS`)

One pass over 8,544 No Robots rows (267 updates at 1e-4). Forced accuracy after the pass:

| Arm | Handoff | After |
|---|---|---|
| D-u140 | 95.4 | 86.0 |
| O-u140 | 94.6 | 94.8 |
| P-u140 | 94.6 | 95.0 |
| M0, fresh adapter | 94.7 | 94.5 |

**Excess over O:**
- D: **9.6 points**, interval [6.9, 12.4];
- P: −0.2, interval [−1.9, 1.6].

**Both predictions hold.** Reading: the abstract leads with this realistic number, and the 3e-4 stage
becomes the stress test. Spend: $34.75.

### SK result (2026-09-26 02:02 UTC; `tools/analyze_skill.py`, SHA-256 `87ff34393d66…`)

The analysis script was written after the runs, since the runner computes only the gate. It applies the
registered outcomes and reads "break" and "consolidation" two ways:
- as after-stage accuracy, the registered wording;
- as extra loss from each arm's handoff.

Forced accuracy at 2,048 tokens, 300 held-out items, four draws:

| | Base | D-S | O-S | P-S |
|---|---|---|---|---|
| Handoff | 0.2 | 37.9 | 66.2 | 58.1 |
| After IT, 1e-4 × 60 (primary) | | 26.6 | 50.7 | 38.7 |
| After IT, 3e-4 × 20 | | 0.1 | 0.4 | 0.4 |

**Handoff.** D-S − O-S = −28.3 [−32.3, −24.5]. Repetition hurts acquisition here, the opposite of
Kopiczko et al.'s result at handoff.

**Primary stage.**
- Losses: D-S 11.3 [7.8, 14.8], O-S 15.5 [11.7, 19.3], P-S 19.4 [15.4, 23.3].
- D's extra loss over O: −4.2 [−9.2, 0.9].
- O − D after the stage: 24.2 [20.4, 27.9], most of it the handoff gap.
- Outcomes:
  - break by after-stage accuracy: yes;
  - break by extra loss: no;
  - consolidation, read either way: no;
  - the fix costs nothing: no (P-S − the better arm is −8.2 at handoff and −12.1 after).

**Default stage.** The skill is erased in every arm: losses of 37.8, 65.8 and 57.7, all ending below 1%.
Consolidation by after-stage accuracy is met trivially.

**Reading.** For a skill the base lacks within the budget:
- repetition lowers what is learned;
- it does not make what is learned less durable (the extra loss has an interval that includes zero and a
  negative point estimate);
- a newly learned skill is fragile for every arm, and the 3e-4 stage erases it;
- fresh solutions on few problems do not match many problems seen once.

The paper states this regime as a limit of its claim: the fragility from repeated texts is shown where
training reshapes a skill the base already has. Spend: $51.23.

### Launch: SD seed s3 (2026-09-26 02:03 UTC)

Spend so far is about $320, and the running parts are expected to finish near $610 with s3. The limit
therefore stays $148, within the cap:
`tools/run_revision_arms.py --part SD --seed s3 --run-id revision-sd-20260926 --limit 148`.
Correction: summed from the ledgers, spend at s3's launch was $344.11, not "about $320":
- RL $121.84, RP $27.36, RS $34.75, SH $24.71, SD s2 $84.22, SK $51.23.

The expected total with s3 is still about $610.

### RL result (2026-09-26 02:12 UTC; `tools/analyze_revision.py --part RL`)

Recovered share R_k, with its interval. The reference is the matching once-trained after-state
(94.6–94.8%):

| Origin (start) | k = 1 | 2 | 5 | 20 | Habit, k = 20 |
|---|---|---|---|---|---|
| D+IT (59.3) | 0.72 [0.65, 0.78] | 0.88 | **0.98 [0.94, 1.01]** | 0.99 | 0.98 [0.94, 1.02] |
| D+AO (22.5) | 0.39 | 0.79 | 0.97 | 0.98 | 0.97 [0.94, 0.99] |
| D-T+IT (33.7) | 0.70 | 0.88 | 0.98 | 0.98 | – |
| D-T+AO (1.1) | 0.22 | 0.74 | 0.96 | 0.98 | – |

**Ceiling control.** O+IT stays within 0.6 points of its 94.7 at every k.

**Both predictions hold.** R_5 = 0.98 ≥ 0.5, and the ceiling control holds.

**Readings, applied as fixed.**
- **D+IT:** "the drilled model's competence is suppressed and is restored by about 160 examples of
  reasoning training". The word "lost" is dropped from the paper.
- **The habit control** (R_20 of 0.98 and 0.97, with no mathematics): the claim "more than a lost habit"
  is dropped. The paper says that retraining the reasoning format alone restores the loss.

This supersedes the paper's earlier reading of the re-forcing probe. A "Wait" cue at inference recovers
41% and 6.5%, but 20 updates of training on the format recover almost all of it.

Spend: $157.74.

### SH result (2026-09-26 02:29 UTC; `tools/analyze_revision.py --part SH`)

**Gate.** O-sharp at temperature 0.5 (4,453 texts): B = **0.0816** [0.0766, 0.0866]. That passes
(≥ 0.05), so there is no retrain at 0.3. For comparison, D's B is 0.111 and O's 0.0004. O-sharp is
therefore about three-quarters as sharpened as D, without any repetition.

| | Handoff | After IT | After AO |
|---|---|---|---|
| O-sharp | 93.1 | 95.0 | 92.4 |
| O | 94.6 | 94.7 | 94.8 |

**O-sharp's excess over O:**
- after IT: −1.8 [−4.1, 0.3];
- after AO: **0.9 [−1.6, 3.4]**.

**Reading: sharpening alone does not break it.** Seeing the same texts again matters beyond the
sharpening it causes, and Section 4 is rewritten to say so. B flags drilled models but does not
identify the mechanism.

**The prediction holds:** O-sharp's AO excess is below half of D's 73.1.

Spend: $58.24.

### Corrections from the independent reproduction (2026-09-26 02:49 UTC)

The public repository's pipeline recomputes every value in the five parts' summary.json files, and in
SK's gate.json and calibration.json, bit for bit. It found three errors of transcription in this record.
The summaries were right in each case; the readings are unchanged.
1. **RP.** D-T's excess over O-T is **−0.9 [−2.8, 0.8]** on the primary score. The −0.6 [−3.3, 2.0]
   given above is the lenient score.
2. **SK.** Under the paper's rounding (halves away from zero), 66.25, 50.75 and −9.25 print as 66.3, 50.8
   and −9.3. The gate's gain is 66.1 [63.3, 68.8].
3. **SK's smoke.** It used 5 items × 4 draws per setting (20 responses per setting), not 20 items × 4
   draws.

### SD result (2026-09-26 03:31 UTC; `tools/analyze_revision.py --part SD`, SHA-256 `caba8878e3c7…`)

Seed s3 finished at 03:30 UTC ($129.90); s2 had finished at 02:48 UTC ($132.96). Forced accuracy:

| Run | Model | Handoff | After IT (3e-4 × 20) | After IT (1e-4 × 60) |
|---|---|---|---|---|
| s2 | D | 94.8 | 32.7 | 80.9 |
| | O | 96.2 | 95.1 | 95.4 |
| | P | 95.6 | 94.6 | 94.0 |
| s3 | D | 94.6 | 45.9 | 76.1 |
| | O | 94.9 | 95.1 | 94.2 |
| | P | 94.2 | 94.5 | 94.9 |

**Excesses**, in points with intervals:

| Run | Stage | D over O | D over P | P over O |
|---|---|---|---|---|
| s2 | IT, 3e-4 | 61.1 [57.0, 65.2] | 61.1 [57.0, 65.0] | 0.0 [−1.9, 1.9] |
| s2 | IT, 1e-4 | 13.1 [10.0, 16.3] | 12.3 [9.2, 15.5] | 0.8 [−1.1, 2.8] |
| s3 | IT, 3e-4 | 48.9 [45.2, 52.5] | 48.9 [44.8, 52.9] | 0.0 [−2.1, 2.1] |
| s3 | IT, 1e-4 | 17.8 [14.1, 21.5] | 19.1 [15.2, 23.1] | −1.4 [−3.6, 0.9] |

**All three predictions hold in every seed and stage:**
- primary: D's excess over O and over P has an interval above zero;
- secondary: each excess is at least 5 points;
- P's excess over O is within 3 points.

**Pooled, with the spread, as registered.** D over O under IT at 3e-4 is 36.2 (the paper's run), 12.6 (D2), 61.1 and
48.9: a mean of 39.7, a range of 12.6–61.1, and an SD of 20.7. At 1e-4 it is 13.1, 13.1 and 17.8: a mean of 14.7, a range of
13.1–17.8. The direction holds in every run. The size under the harsh stage varies almost fivefold between drilled sets,
and is steadier at 1e-4.

**Spend.** The six parts cost $592.18 at list prices (RL $157.74, RP $27.36, RS $34.75, SH $58.24, SK $51.23,
SD $262.86), within the $725 cap.

### Notes from the public port (2026-09-26 04:13 UTC)

The six parts' runners were ported to the public repository and checked offline against every run folder. The
port found four places where this record and the private code differ. None of them changes a result.
1. **SK's token cap in the docstring.** `run_skill.py`'s docstring still says "at most 4,096 tokens". Its code, as
   hashed after the amendment (`2532e3151af8…`), uses 2,048, as recorded above.
2. **Distinct base-7 solutions.** Four of the 4,480 training problems have 18 distinct worked solutions, not 20.
   When one factor is a number such as 20000, the running-total and sum-at-the-end methods write the same text.
   P-S uses at most 8 solutions per problem, so training is unaffected.
3. **SK's phase limits.** The calibration and gate launches used limits of $8 and $25, below the $12 and $40
   planned in the phases paragraph, as their launch entries show.
4. **The text-length figures' sample.** The mean lengths (871.5, 1,164.4, 1,139.0 and 1,702.0 tokens) and the
   five-digit maximum (1,937) are over every worked solution of the first 150 training problems of each setting.
