# Robustness: when the drilled model breaks, and whether the step size matters

Adopted 2026-09-24 11:18 UTC from a reviewed draft,
unchanged except for this line and the launch section. It is fixed when its launch time and code
hashes are added, before any sample.

**Authorization.** [redacted: account cap]

**Budget.** R1 about $30–38 (limit $45); R2 about $20–30 (limit $40).

## Why

Two questions a reviewer will ask of the main result are still open.

**R1, timing and mechanism.** Under instruction tuning (IT), every 9B state fits the new task
equally well. The per-example training loss at update 20 is 1.64–1.65 for all of them. Yet the
drilled state loses 36 points more forced accuracy than the others. Two things are unknown:
- whether the drilled reasoning is displaced more by the same updates;
- whether that displacement comes before the accuracy falls, and how many examples it takes.

**R2, step size.** The later stages use learning rate 3e-4, and at that rate instruction tuning
breaks even the untrained 35B base, which starts from a fresh adapter. Does the drilled model's
excess survive a gentler rate, with the total movement matched?

## R1. Early dynamics under instruction tuning

**States.** All are re-saved with no expiry:
- the 9B base (fresh rank-32 LoRA);
- D-u140 and O-u140 (pilot);
- P-u140 (main phase).

**Stage.** The pilot's instruction tuning, unchanged:
- the same 640 No Robots examples in the same order;
- batch 32, learning rate 3e-4, Adam 0.9/0.95/1e-12, no clipping, fresh optimizer.

It runs for 5 updates, and weights are saved after updates 1, 2 and 5. The update-20 states are the
paper's existing ones.

**Trajectory check.** Before any measure is read, the replayed training losses at updates 1–5 must
match the logged losses of the paper's IT runs within 2% for each state. If a state fails, its early
states are reported as a separate realization.

**Measures.**
1. **Displacement.**
   - Definition: δ_k(s) = the mean over positions after the prefix of [log p_handoff(y_t) −
     log p_after-k(y_t)].
   - Texts: state s's own handoff forced texts, one top-p-1 sample per MATH-500 level 3–5 problem
     (221 texts), cut at 2,048 tokens. For the base these are the pilot's common set; for D and O,
     the pilot's own sets. For P, draw one if the main run has none (about $1.5).
   - It estimates KL(handoff ‖ after-k) per token.
   - Points: k = 1, 2, 5 and 20.
   - Also reported, as a secondary: δ_k divided by the state's own handoff NLL on the same texts.
2. **Forced accuracy.** MATH-500 levels 3–5, four draws, 16,384 tokens, for D-u140 and O-u140 at
   k = 2 and 5. The values at k = 0 and 20 exist.
3. **Training loss.** The stage's own per-update loss, from the logs (free).

**Predictions, written before data.** Intervals are paired item bootstraps over the 221 texts or
problems (20,000 resamples).
1. **More displacement for the drilled state.** At k = 2, δ_2(D) is at least 2 × max(δ_2(O),
   δ_2(P)), with the interval of δ_2(D) − 2·max(δ_2(O), δ_2(P)) above zero.
2. **Displacement before the collapse.** At k = 2, D's forced accuracy has fallen by less than half
   its k = 20 loss, i.e. it is above 77.4% (95.4 at handoff; 59.3 at update 20).
3. **Fresh texts behave like once-trained.** δ_k(P) ≤ 1.5 × δ_k(O) at k = 1, 2 and 5.
4. **Equal fit to the new task.** The per-update training loss at updates 2–5 differs across the four
   states by at most 10%.

**Readings.**
- **1 and 2 hold:** "the same updates that fit the new task equally well move the drilled reasoning
  further, and do so before its accuracy falls."
- **1 holds and 2 fails:** the displacement and the collapse come together.
- **1 fails:** no mechanism reading. The paper keeps the timing only.
- The base's δ is descriptive: it starts from a fresh adapter, not a trained one.

**Limits.**
- One realization per state.
- Each state's δ is measured on its own texts, so it compares how far each state moves from itself,
  not a shared yardstick. The NLL-normalized version is the check.
- Early states come from replaying the first updates, and the trajectory check guards this.

**Cost.**

| Part | Cost |
|---|---|
| Training, 4 states × 5 updates | about $1 |
| P's own texts | about $1.5 |
| Displacement scoring (4 states × 3 points × 221 texts × up to 2,048 tokens, prefill only) | about $4–5 |
| Forced evaluations (4 state-points) | about $24–32 |
| **Total** | **about $30–38** (limit $45) |

If money is short, drop the forced evaluations at k = 5 (saves about $12).

## R2. A gentler learning rate at matched movement

**States.** D-u140 and O-u140 (9B pilot).

**Stage.** Instruction tuning at learning rate 1e-4 (batch 32, same optimizer settings, fresh
optimizer).
- 60 updates, 1,920 distinct examples: the pilot's 640 in the same order, then the first 1,280 of the
  4,480 single-turn No Robots examples used by FN (disjoint from the 640), in their fixed order.
- No example repeats.
- Weights are saved at updates 20 and 60.

**Why 60.** 1e-4 × 60 matches 3e-4 × 20 in total step size, a first-order match under Adam; it is
stated as a heuristic.

**Measures.**
- Forced MATH-500 levels 3–5 (four draws, 16,384 tokens) at update 60 (primary) and update 20
  (secondary).
- The stage's per-update loss.

**Predictions, written before data.**
1. **Primary.** At 1e-4 × 60, D-u140's excess forced loss over O-u140 is at least 10 points, with the
   interval above zero.
2. **Secondary, descriptive.** At 1e-4 × 20 the excess is reported as it falls. On TCES, the
   gentler rate roughly halved the drilled loss within 4,096 tokens.

**Readings.**
- **1 holds:** the break does not depend on the harsher step size at matched movement.
- **1 fails:** the paper scopes the claim to learning rates typical of LoRA fine-tuning, and the
  limits give both numbers.

**Limits.**
- One run per condition.
- Updates 21–60 see new examples, as a longer instruction-tuning stage would.
- The 35B base's collapse at 3e-4 is not re-tested here.

**Cost.**

| Part | Cost |
|---|---|
| Training, 2 × 60 updates | about $6 |
| Forced evaluations, 4 state-points | about $14–24 |
| **Total** | **about $20–30** (limit $40) |

If money is short, evaluate update 60 only (about $15).

## Launch (R1 and R2)

Fixed and launched 2026-09-24 11:18 UTC as `tools/run_robustness.py` (`runs/math/robustness-20260924`),
one run for both parts with a limit of $80 (R1 $45 + R2 $40). [redacted: account balance] The trajectory check of R1 is read from the run's training logs
against the paper's instruction-tuning logs before any R1 measure.

Code SHA-256: `run_robustness.py` `97f6469f2fa3…`; `run_math_pilot.py` `d8d537ffdc04…`; `math_models.py` `d132c934a053…`; `run_instruction_continuation.py` `fd35889684fa…`; `run_decision_probe.py` `ae6905fb3051…`; `run_depth_profile.py` `57413d99c540…`; 

## R1 trajectory check (read 11:25 UTC, before any R1 measure)

The replayed training losses at updates 1–5 match the logged losses of the
paper's instruction-tuning runs within 0.1% for all four states, well inside the
2% tolerance: the base, D-u140, O-u140 and P-u140. The early states are faithful
replicas of the paper's instruction-tuning trajectories.

## R1 result (written 11:35 UTC; `tools/analyze_robustness.py`, `runs/math/robustness-20260924/summary.json`)

**Displacement δ_k.** Nats per token of KL(handoff ‖ after k updates), on each
state's own handoff texts, after k updates of instruction tuning.

| State | k = 1 | k = 2 | k = 5 | k = 20 |
|---|---|---|---|---|
| D-u140 (drilled) | 0.0205 | 0.0666 | 0.1185 | 0.1221 |
| O-u140 (once) | 0.0001 | 0.0005 | 0.0018 | 0.0013 |
| P-u140 (fresh texts) | 0.0001 | 0.0004 | 0.0021 | 0.0019 |
| Base (fresh adapter) | 0.0008 | 0.0034 | 0.0087 | 0.0082 |

As a share of each state's own handoff NLL on the same texts, D moves 24% by
k = 2 and 42% by k = 5. O and P move 0.1–0.5%.

**Forced accuracy (MATH-500 levels 3–5, four draws).**

| State | k = 0 | k = 2 | k = 5 | k = 20 |
|---|---|---|---|---|
| D-u140 | 95.4 | 81.8 | 46.0 | 59.3 |
| O-u140 | 94.6 | 93.9 | 94.8 | 94.7 |

**All four predictions hold.**
1. δ_2(D) = 0.0666 is 133 times O's (0.0005) and more than twice max(O, P):
   δ_2(D) − 2·max(δ_2(O), δ_2(P)) = 0.066 [0.060, 0.072].
2. At k = 2 D has lost 13.6 points, less than half its k = 20 loss of 36.1: the
   displacement comes before the collapse.
3. δ_k(P)/δ_k(O) is 0.74, 0.84 and 1.18 at k = 1, 2 and 5 (at most 1.5).
4. The training loss at updates 2–5 differs across the four states by 1.7–4.0%
   (at most 10%).

**Reading.** The same updates that fit the new task equally well move the
drilled reasoning about a hundred times further, and its forced accuracy falls
with it.

*Correction after a review.* "Before its accuracy falls" overstated
the lead. At update 2 the displacement has reached 55% of its update-20 value,
and the accuracy loss 38% of its update-20 value, so the displacement leads only
mildly. Prediction 2 holds as registered; the reviewed wording:

> In the first two updates, which fit the new task about equally well in every
> model (training losses within 4%), the drilled model's reasoning moves 0.067
> nats per token, over a hundred times as far as the once-trained and
> fresh-solution models (0.0005 and 0.0004). Its forced accuracy falls with it:
> by 14 points at update 2, and by 49 at its lowest, at update 5.

**Not predicted.** D's forced accuracy is lowest at k = 5 (46.0) and partly
recovers by k = 20 (59.3), while its displacement keeps growing slightly.

## R2 result (written 11:37 UTC; the same summary; R1 and R2 spent $44.53 together)

Forced accuracy on MATH-500 levels 3–5 (four draws) under instruction tuning at
learning rate 1e-4, on 1,920 distinct examples:

| State | Handoff | After 20 updates | After 60 updates |
|---|---|---|---|
| D-u140 | 95.4 | 82.8 | 82.2 |
| O-u140 | 94.6 | 94.6 | 94.6 |

**Prediction 1 holds.** At 1e-4 × 60, D-u140's excess forced loss over
O-u140's is 13.1 points [9.8, 16.5]; after 20 updates it is 12.6 [9.3, 16.0].

**Reading.** The break does not depend on the harsher step: at a third of the
step size, matched in total movement, the drilled model still loses about 13
points more, and the once-trained model none. The size depends on the step:
36.2 at 3e-4 × 20 against 13.1 at 1e-4 × 60. The drilled model's loss stops
growing after update 20 (82.8, then 82.2).

## R3: persistence and a shared yardstick (draft 2026-09-24 11:37 UTC, after a review)

**Why.** R1 found D's forced accuracy lowest at update 5 (46.0) and partly back
by update 20 (59.3). Two questions follow:
- Does it recover fully with more instruction tuning?
- R1 measured displacement on each state's own texts. Do sharper texts give a
  larger KL by construction?

**R3a, persistence.**
- **States:** D-u140 and O-u140.
- **Stage:** instruction tuning at 3e-4 for 60 updates on R2's 1,920 distinct
  examples: the paper's 640 in order, then the first 1,280 of the FN stream. The
  first 20 updates replay the paper's stage exactly.
- **Measure:** forced MATH-500 levels 3–5 (four draws) after updates 40 and 60.
- **Prediction:** after update 60, D's excess forced loss over O's is at least
  20 points, with the interval above zero.
- **Readings:**
  - If it holds, the loss persists with further training.
  - If it fails, the loss is deep but transient, which is still a hazard for a
    pipeline that stops early. The paper reports it as such.

**R3b, a shared yardstick.**
- **Texts:** the base's common texts (the pilot's M0 set, 221 texts, 2,048
  tokens).
- **Scoring:** under D, O and P after instruction-tuning updates 2 and 20,
  against each state's own handoff scores on the same texts.
- **Measure:** δ^common_k, the mean of log p_handoff − log p_after-k over
  positions after the prefix.
- **Prediction:** δ^common_2(D) is at least 2·max(δ^common_2(O),
  δ^common_2(P)), with the paired interval above zero.

**Cost.**

| Item | Estimate | Limit |
|---|---|---|
| R3a | about $16 | $25 |
| R3b | about $2 | $5 |

**Launch (R3).** Fixed and launched 2026-09-24 11:37 UTC as `tools/run_robustness.py --parts R3`, in the same
run folder, whose ledger continues R1 and R2's $44.53. It stops at $30 beyond that. [redacted: account balance] `run_robustness.py` SHA-256 is `e2956f457a92…`.

## R3 result (written 12:03 UTC; the same summary; R3 spent $19.73)

**R3a, persistence: the prediction holds.** Forced accuracy under instruction
tuning at 3e-4, continued on distinct examples:

| State | Update 0 | Update 20 | Update 40 | Update 60 |
|---|---|---|---|---|
| D-u140 | 95.4 | 59.3 | 41.1 | 41.6 |
| O-u140 | 94.6 | 94.7 | 95.0 | 96.0 |

D's excess over O is 54.8 [50.7, 58.8] after update 40 and 55.2 [50.9, 59.5]
after update 60, well above 20. The loss does not recover with more training: it
deepens, and the partial rise at update 20 was not a trend.

**R3b, a shared yardstick: the prediction fails.**
- On the base's texts, the drilled model's displacement is negative:
  δ^common_2(D) = −0.0068 and δ^common_20(D) = −0.0057. After the updates it
  gives the base's reasoning more probability than it did at handoff.
- For O the values are 0.0003 and 0.0014; for P, 0.0002 and 0.0018.
- The registered test gives −0.0073 [−0.0112, −0.0034], against a required
  value above zero.

**Reading.** R1's hundredfold displacement is specific to the drilled model's own
texts. It is not a larger movement on every text. What the later stage erases is
the drilled model's own sharpened trajectories. On the base's reasoning it moves
slightly back toward the base, yet its forced accuracy falls far below the base's
under the same stage (the base keeps 95.6).

The mechanism wording must say "erases the drilled model's own reasoning," not
"moves its reasoning further". R1's own prediction 1 stands as registered: it
was about each state's own texts.

## Corrections after an independent recomputation (written 12:37 UTC)

Every R1–R3 number was recomputed independently from the raw files, and all of them
reproduce: the intervals within bootstrap noise, and the R2/R3a training data at
all 60 updates. Four wordings above are corrected here.
1. **"Training losses within 4%"** holds from update 2 on. At update 1 the spread
   across the four states is 8.9%.
2. **"Matched in total movement"** (R2) should read "matched in total step size".
   Movement was not measured.
3. **"About a hundred times further"** holds at updates 1–2 (133× at update 2).
   At update 5 the ratio is 56–66×.
4. **The replays are close, not exact.**
   - R1's losses match the paper's run within 0.1%.
   - R3a matches the paper's stage in data and settings, but its losses differ
     by up to 0.18% from update 1 on, and its update-20 column is the paper's own
     run: R3a did not evaluate update 20.

**A further observation from the same files (no cost).** On the drilled model's
own solutions, two instruction-tuning updates remove 88% of the log-likelihood
drilling had added. From update 5 on, those solutions are less likely under the
drilled model than under the untrained base: 0.403 against 0.357 nats per token.

## R4 and R5: the teacher arms at the gentler step, and a broad later stage (registered 2026-09-24 13:10 UTC, before launch)

**Why.** After the teacher result (teacher.md), two objections remain (two review points):
- The self-data break shrank from 36.2 to 13.1 points at the gentler step (R2). Does the break on a
  stronger model's traces survive it too?
- The later stages are narrow: one chat dataset (No Robots) and one answer-only set. Does the break
  recur under a broad instruction mix of the kind general post-training uses?

**R4, the teacher arms at 1e-4.**
- **States:** D-T-u140 and O-T-u140 (teacher.md).
- **Stage:** R2's, unchanged: instruction tuning at 1e-4 for 60 updates of 32 on R2's 1,920 distinct
  examples (the paper's 640 in order, then the first 1,280 of the FN stream).
- **Measure:** forced MATH-500 levels 3–5 (four draws) after updates 20 and 60.
- **Prediction R4.1 (R2.1's criterion):** after update 60, D-T's excess forced loss over O-T's is at
  least 10 points, with the paired interval above zero. No P-T arm: P-T did not break at 3e-4.

**R5, a broad later stage.**
- **Data:** 1,920 distinct single-turn rows of the cached Tülu-3 SFT mixture, in the SHA-256 order
  of the namespace `robustness-20260924-chat`, from its non-math sources.
  - Left out: the five math sources (PersonaHub math, NuminaMath-TIR, OpenMath-2 GSM8K, the grade-school
    personas and the intermediate-algebra set), No Robots (the IT data) and the 240 hard-coded
    identity rows.
  - The same role-colon rendering as IT. Rows over 1,024 rendered tokens are left out, not truncated.
- **Realized mix** (computed offline before launch; ids and counts in `r5/examples.json`):
  - Aya 361, FLAN 356, Evol-CodeAlpaca 344;
  - WildJailbreak 203, WildGuardMix 182, WildChat 164;
  - PersonaHub code 119, PersonaHub instruction-following 98;
  - CoCoNot 35, SciRIFF 28, OASST 21, TableGPT 9.
  - Mean 316 rendered tokens (the IT stage's 640: 284).
- **Content check.**
  - 10.4% of the rows (200) match a broad math pattern: \\boxed, \\frac, inline $…$, \\( or \\[, "solve",
    "calculate". None has \\boxed.
  - Most matches are code tasks that "calculate" something (125) and FLAN's grade-school word problems
    (61).
  - The paper says "non-math sources", not "no math".
- **States:** D-u140 and O-u140 (the model's own solutions); D-T-u140 and O-T-u140 (the teacher's).
- **Stage:** R2's schedule (1e-4, 60 updates of 32). Only the data differ from R2 and R4.
- **Measure:** forced after update 60; the training loss at every update.
- **Predictions.**
  - **R5.1:** D's excess forced loss over O's has its paired interval above zero. Secondary: at
    least 10 points.
  - **R5.2:** the same for D-T over O-T.
  - **R5.3:** O and O-T each lose at most 5 points (forced at handoff minus forced after update 60).
    If R5.3 fails, the excesses are still read, but the paper calls the stage harsh for every model,
    not routine.
- **Readings.**
  - If R5.1 and R5.2 hold, the break recurs under broad instruction data.
  - If either fails, the break depends on the later stage's data, and the paper says for which
    solutions.

**Analysis.** `tools/analyze_robustness.py` (its `followup`): paired resamples over problems
(20,000), as in R2. Adding it leaves R1–R3's summary unchanged (checked).

**Cost and limit.**

| Item | Estimate |
|---|---|
| R4: 4 forced evaluations at about $5.9, training about $1.5 | about $25 |
| R5: 2 self-data evaluations at about $4.6, 2 teacher at about $5.9, training about $3.6 | about $25 |

- **Run:** `tools/run_robustness.py --run-id robustness-followup-20260924 --parts R4 R5 --prior [redacted: account balance] --budget 80`,
  one run with its own ledger.
- **Limit:** $80 [redacted: account cap], which includes up to about $17 of worst cases held in
  flight. The expected spend is about $50.
- [redacted: account balance]

**Correction before launch (13:17 UTC, from an independent pre-launch check).**
- The content check counts keywords, not math. Of the 200 matches, 125 are code tasks that
  "calculate". Of FLAN's 61, about 15 are grade-school word problems; the rest match "solve" in task
  templates or "$" in tag lists.
- A blind reading of 100 random rows found one worked word problem. Over the 1,920 rows,
  worked math word problems come to about 15–30 (1–1.5%).
- R4 and R5's analysis writes into `runs/math/robustness-20260924/summary.json` (keys `r4`, `r5`),
  beside R1–R3. It reads R4 and R5's outputs from the follow-up run folder.

**Launch (R4 and R5).** Launched 13:17 UTC as `tools/run_robustness.py --run-id robustness-followup-20260924
--parts R4 R5 --prior [redacted: account balance] --budget 80` (limit $80).
- [redacted: account balance]
- SHA-256: `run_robustness.py` `81cf8b17e8e3…`, `analyze_robustness.py` `5881e667a5c4…`.

## R4 and R5 result (written 17:02 UTC; `tools/analyze_robustness.py`, keys `r4` and `r5` of `runs/math/robustness-20260924/summary.json`; R4 and R5 spent $42.71)

The run finished at 13:33 UTC with no failed call. [redacted: account balance]

Forced accuracy (%). Handoff values are those of the pilot (D-u140, O-u140) and the teacher run
(D-T-u140, O-T-u140).

| State | Handoff | R4: IT at 1e-4, u20 | R4: u60 | R5: broad mix at 1e-4, u60 |
|---|---|---|---|---|
| D-u140 | 95.4 | (R2: 82.8) | (R2: 82.2) | 85.4 |
| O-u140 | 94.6 | (R2: 94.6) | (R2: 94.6) | 94.9 |
| D-T-u140 | 95.0 | 89.7 | 86.8 | 85.3 |
| O-T-u140 | 96.3 | 96.6 | 94.9 | 95.8 |

Excess forced loss of the drilled model over the once-trained model, in points with 95% paired
intervals:

| Stage | Own solutions (D over O) | Teacher traces (D-T over O-T) |
|---|---|---|
| IT at 1e-4, update 20 | 12.6 [9.3, 16.0] (R2) | 5.7 [3.3, 7.9] |
| IT at 1e-4, update 60 | 13.1 [9.8, 16.5] (R2) | 6.9 [4.0, 10.0] |
| Broad mix at 1e-4, update 60 | 10.3 [7.5, 13.2] | 9.3 [6.2, 12.3] |

**Predictions.**
- **R4.1 fails as registered.** D-T's excess after 60 updates is 6.9 points, below the 10
  required. Its interval lies above zero.
- **R5.1 holds.** D's excess is 10.3, with its interval above zero. The secondary check (at least
  10 points) also holds.
- **R5.2 holds.** D-T's excess is 9.3, with its interval above zero. The secondary check (at least
  10 points) fails, by 0.7.
- **R5.3 holds.** O gains 0.3 points and O-T loses 0.5; both are within the 5 allowed.

**Training loss** (mean per example):
- **R5.** By updates 56–60 every state fits the broad mix about equally well (0.956–0.968, a 1.3%
  spread). They start apart: over updates 1–5 the drilled states sit at 1.30 against 1.18 for O and
  1.24 for O-T.
- **R4.** D-T 1.846 → 1.744, O-T 1.801 → 1.740.

**Failure form** (update 60; descriptive):
- **Drilled states** write shorter responses: median 596–748 tokens, against 1,166–1,407 for the
  once-trained ones.
- **They seldom close the reasoning block:** 29–39% of responses, against 96%. The answer usually
  stays inside the block and the turn ends there.
- **Unboxed answers** are 4–7%, against 2–3%, so most of the gap is wrong answers, not missing ones.
- **Capped responses:** 2–3% in every state.

**Reading.**
- **A broad, routine later stage breaks the drilled models.**
  - Sixty updates at a third of the default step, on 1,920 rows of non-math instruction data (code,
    safety, multilingual, FLAN, WildChat), cost the drilled models about 10 points more than the
    once-trained ones.
  - This holds whether the repeated solutions were the model's own or a stronger model's.
  - The once-trained models lose nothing.
- **R4 failed its registered magnitude.**
  - At the gentler step on instruction data, the break on the teacher's traces is smaller than on
    the model's own solutions: 6.9 against 13.1 points. That reverses their order at the default
    step (59.6 against 36.2).
  - So "larger than with the model's own solutions" holds only at the default step, and the paper
    must say so.
- **Across the gentle stages the excess is 6–13 points,** against 36–60 at the default step and
  73–92 after the answer-only stage.

**Corrections after an independent recomputation (17:09 UTC).** Every
accuracy, excess, prediction outcome, training loss and the $42.71 spend reproduce from the raw
files. The starting states are confirmed by the update-1 losses. Two descriptive phrases change:
- **Capped responses** are 2–5% in every state, not 2–3%: R5's O is at 4.5%.
- **"6–13 points at the gentle stages, against 36–60 at the default step and 73–92 after the
  answer-only stage"** holds for the 9B's own and teacher solutions, not for every cell. The second
  drilled subset's default-step IT excess is 12.6, and Nemotron's answer-only excess is 40.0.
