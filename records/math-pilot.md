# Math pilot: does drilling a base model's own reasoning over-sharpen it and make it fragile?

Draft revised 2026-09-24 01:19 UTC after an independent pre-launch audit. It is
fixed when the launch time is added at the end, before any trained state exists.

**Authorization.** [redacted: private message]

**Budget.**
- [redacted: account cap]
- This pilot has its own limit of $450. Its ledger stops the run beyond that,
  and the work already done is kept for a relaunch.
- [redacted: account balance]

## Why

On TCES arithmetic with Qwen3.5-9B-Base (records:
[depth-profile.md](depth-profile.md), with its audit corrections and cap test;
[repetition-fragility.md](repetition-fragility.md)):

- Checkpoints drilled 7.7 passes over 579 samples keep the base's top next-token
  choices but cut most of its alternatives. They are over-sharpened: B of 0.06
  to 0.16 nats per token, where the base's own negative log-likelihood is 0.43.
- After 20 updates of a later stage, the drilled checkpoints' forced searches
  stop finishing. They solve 6–12% even with a 16,384-token budget. A clone that
  saw each prompt with fresh completions solves 76%.
- B at handoff separates the fragile checkpoints from the lasting ones without
  knowing how they were trained. It does not rank finer differences within dose
  groups.

The pilot asks whether this happens on competition math, with the base model's
own correct reasoning as the training data. This is rejection-sampling
fine-tuning, the commonest self-training recipe.

## Design choices made before registration

**Exploration** (8 easy pool problems, $0.05). The base solved 14 of 16
unforced and 15 of 16 forced. Forced into `\n<think>\n`, it reasons, closes
`</think>` and boxes its answer.

**Calibration** (one draw per problem and mode, $1.85;
the calibration script is not released). "Direct" means an empty think block with a
256-token cap; in 83% of those samples the model still writes working until
the cap.

| Source | Forced | Unforced | Direct | Median forced tokens |
|---|---|---|---|---|
| MATH-500 levels 1–2 | 1.00 | 1.00 | 0.58 | 850 |
| MATH-500 level 3 | 0.92 | 0.83 | 0.42 | 1,326 |
| MATH-500 level 4 | 0.88 | 0.71 | 0.17 | 954 |
| MATH-500 level 5 | 0.62 | 0.58 | 0.08 | 2,928 |
| OlympiadBench (integer) | 0.46 | 0.42 | 0.00 | 6,144 (cap) |
| NuminaMath-TIR (integer) | 0.73 | 0.67 | 0.19 | 1,410 |

- MATH-500 levels 1–2 have no headroom, so the evaluation uses levels 3–5.
- MATH-500 is saturated and may be contaminated for Qwen models. It is the
  pilot's precise screen, and any contamination is shared by every arm, since
  all start from the same base.
- AIME 2025 and 2026 are the headline benchmarks ($1.48 calibration, forced,
  one draw, cap 16,384). The base scored 15 of 30 and 18 of 30. Token deciles
  were 3,205, 7,930, 15,620, 16,384 and 16,384; 29 of 60 runs hit the cap.
  AIME therefore gets a 32,768 cap.
- OlympiadBench is left for the main experiments.

## Data

**Training pool.** Built by a data-preparation script (not released; `src/build_inputs.py` rebuilds the
frozen pool) from Tulu-3's `numinamath_tir_math_decontaminated` problems whose reference
answer (the last `\boxed{}` of the reference solution) is one integer.

- The source has 64,312 rows and 58,674 distinct problems.
- Removed:
  - 12,493 problems with non-integer answers;
  - 2,614 whose solution boxes several different answers;
  - 107 whose duplicate copies disagree;
  - 331 with figures;
  - 28 with prompts over 512 tokens;
  - 91 contamination hits against MATH-500, OlympiadBench and AIME 2025/2026
    (exact, 13-gram, punctuation-stripped 13-gram, and a hand-reviewed
    paraphrase search). These include AIME 2026 problem 1 word for word and two
    MATH-500 paraphrases;
  - 6,563 problems whose answers can be guessed or that are not single-answer:
    multiple choice, answer in the statement, proofs, multi-part, "find all",
    numbered statements.
- The remaining 36,447 are in SHA-256 order (namespace `math-pilot-20260924`):
  `runs/math/data/pool.jsonl`, SHA-256 `43a48d75…d044`.
- Reference answers agree with the solutions' own code output in 93% of a
  200-row sample; the rest were checked by hand.

**Screen.** In pool order, a problem is skipped if the base solves it within
256 tokens after an empty think block. Only the rest supply training samples.

**Evaluation.**
- The MATH-500 problems at levels 3–5 with integer answers: 221 of the 311 in
  `runs/math/data/eval_math500.jsonl` (SHA-256 `82d7be62…4c42`).
- AIME 2025 and 2026 from the cached MathArena files, less 2026 problem 1,
  which is in the training source: 59 problems.

**Scoring.** `tools/math_score.py` (SHA-256 `f9194799…77e`):
- It takes the last `\boxed{}`, with nested and escaped braces handled; an
  unclosed box means no answer.
- It normalizes units, degrees, currency, percent, thousands separators and
  `x = 5`, then parses an integer.
- Its 30 unit examples pass, and it recovers all 311 MATH-500 reference
  answers.

## Arms

All arms start from Qwen3.5-9B-Base with a fresh LoRA of rank 32 on attention,
MLP and output layers, seed 20260924.

**Training data.**
- Each screened problem gets one forced sample from the base: temperature 1,
  top-p 1 so the training texts carry the base's full distribution rather than
  a truncated one, cap 16,384.
- A sample is kept if it is correct and uncapped, until 4,480 are kept. The cap
  is generous, but keeping only correct samples still favours shorter
  solutions.
- The target is the forced opening `\n<think>\n` followed by the sampled
  tokens, so a trained model learns to open a reasoning block itself.
- The recipe is that of the arithmetic U and F arms: loss on the sampled
  tokens, normalized by the set's mean length; learning rate 1e-4; batch 32;
  Adam 0.9/0.95/1e-12; no clipping.

**The two arms.**
- **D (drilled).** The first 579 kept examples, cycled for 140 updates. States
  are saved after 20, 60 and 140 updates (1.1, 3.3 and 7.7 passes).
- **O (once).** All 4,480 kept examples, one pass in 140 updates. D's 579 are a
  random subset of them. States are saved after 60 and 140 updates.

D-u140 and O-u140 share the recipe, the source and the number of updates. They
differ in repetition, and in how many problems they cover (579 against 4,480).
The main experiments add an arm that separates these two.

## Later stages

Each stage runs 20 updates of 32 at learning rate 3e-4 with a fresh optimizer,
from the base and from D-u20, D-u60, D-u140 and O-u140. The stages are the
same as in the arithmetic experiments:
- **Instruction tuning:** the 640 No Robots examples of Part D3.
- **Stage B:** the answer-only program-synthesis continuation (MAPS records,
  same batch order). Its targets use an `<answer>…</answer>` tag, so math
  answers may drift into that format. A secondary score also accepts
  `<answer>N</answer>`, and the share of unboxed answers is reported.

The base continues from a fresh LoRA, while the trained states continue from
their trained LoRA. Comparisons among trained states are symmetric;
comparisons with the base (prediction 5) carry this difference.

Each stage's trained weights are saved, so a relaunch evaluates the same
realization.

## Measures

- **Accuracy.** Forced accuracy (`\n<think>\n` prefilled) with four draws per
  problem, and unforced accuracy with two draws: cap 16,384, temperature 1,
  top-p 0.95. A sample is correct if its last `\boxed{}` equals the answer.
- **Accuracy against budget.** From the same samples, accuracy within 2,048,
  4,096, 8,192 and 16,384 tokens. The failure modes are also reported: cap
  hits, loops (few distinct 16-grams in the last 2,000 tokens), and unboxed
  answers.
- **AIME** (descriptive in this pilot). Forced accuracy on 59 problems, four
  draws each, cap 32,768, for the base, D-u140 and O-u140, at handoff and after
  instruction tuning. With 59 problems its interval is about ±13 to 16 points.
  Its score also depends on when the model stops, since a capped run is scored
  on its last box.
- **Decision.** The share of unforced completions that open a `<think>` block.
- **Divergence.** B, B′ and A are measured as in
  [depth-profile.md](depth-profile.md).
  - The common set is one forced sample of the base per problem, drawn at top-p
    1 so that B is an unbiased estimate of KL(base‖state). It is stored as
    exact tokens and cut at 2,048.
  - Each own set is one top-p 1 forced sample of the state per problem.
  - B is scored for the base, D-u20, D-u60, D-u140, O-u60 and O-u140; B′ for
    the four evaluated trained states.
  - B is split into positions where the state keeps the base's top token and
    those where it flips it. The top token's probability gain is reported at
    the kept positions.
- **Durability.** Forced accuracy lost over the 20 updates of each stage.

## Predictions, written before data

1. **Repetition over-sharpens.** B(D-u140) − B(O-u140) is at least 5% of the
   base's own per-token negative log-likelihood on the common set, and its 95%
   paired item-bootstrap interval is above zero.
2. **Dose.** B(D-u20) < B(D-u60) < B(D-u140).
3. **Over-sharpened competence is fragile.** Tested separately for each stage:
   D-u140 loses more forced accuracy than D-u20 and than O-u140, with the 95%
   paired item-bootstrap intervals of both excesses above zero.
4. **Handoff diagnostic.** B ranks the forced loss of the five evaluated states
   (Spearman; descriptive with five points).
5. **Two clocks.** For the base and O-u140, each stage lowers unforced accuracy
   by more than forced accuracy.

**Parity condition.** At handoff, D-u140 and O-u140 are within 5 points of each
other in forced accuracy.

## Go/no-go for the main experiments

For each stage, "the excess" is the smaller of D-u140's two excess losses, over
D-u20 and over O-u140.

- **Stop and rethink** if prediction 1 fails: drilling a base's own math
  reasoning does not over-sharpen it.
- **Ambiguous** if parity fails; the author decides.
- **Go** if prediction 3 holds under at least one stage.
- **Go with more subsets** if the excess is at least 5 points under at least
  one stage but an interval includes zero. The main experiments then add
  training subsets.
- **Rethink before spending more** if the excess is under 3 points under both
  stages.
- **Ambiguous** otherwise; the author decides.

`tools/analyze_math_pilot.py` implements these rules exactly.

## Prechecks

1. **Leakage: passed.** Exact, 13-gram and paraphrase checks found
   no hits left in the pool; the hits were removed. An independent re-run of exact and
   13-gram matching against MATH-500 levels 3–5 and AIME found zero.
2. **Scorer: passed.** The 30 unit examples pass. In 20 base solutions checked
   by hand (`runs/math/pilot-20260924/precheck_samples.jsonl`), the scorer read
   the final box correctly in all 20, including three wrong answers (one of
   them √2, which is not an integer).
3. **Fresh-LoRA identity: passed.** A fresh rank-32 LoRA scores the base's four
   test texts with a mean absolute difference of 0.0 over 1,036 positions.
4. **Cost.** Projected $300–400: training data $40–70, training $30, MATH-500
   evaluations $80–150, AIME $50, divergence texts and scoring $12, later
   stages $3. States whose search stops finishing cost the most. The $450
   limit covers the projection.

## Launch

Fixed and launched 2026-09-24 01:34 UTC as run `pilot-20260924`, with
`--prior [redacted: account balance] --budget 450`. Code SHA-256 at launch:

- `tools/run_math_pilot.py`: `20d3ac118227c388…`
- `tools/analyze_math_pilot.py`: `f018be395450c2cc…`
- `tools/math_score.py`: `f9194799c17619ea…`

An independent pre-launch audit and its re-check found no blockers.

## Results (written 2026-09-24 05:26 UTC; run `pilot-20260924`, $230.55)

**Training data.**
- 7,680 problems screened; 2,330 (30%) were solved within 256 tokens after an
  empty think block and skipped.
- Of the other 5,350, 4,810 (90%) were kept: correct and uncapped. 356 hit
  the 16,384 cap and 184 were wrong.
- D's 579 average 2,362 tokens (median 1,351); O's 4,480 average 2,608 (median
  1,458).

No job failed.

**Decision: go.** Every registered prediction holds, and parity holds (a
handoff gap of 0.8 points).

**Forced accuracy** on the 221 MATH-500 problems (levels 3–5), four draws,
within 16,384 tokens. Unforced accuracy (two draws) in brackets.

| State | Handoff | After instruction tuning | After Stage B |
|---|---|---|---|
| Base | 94.7 (74.7) | 95.6 (33.5) | 91.7 (12.2) |
| D-u20, 1.1 passes | 95.4 (95.0) | 94.5 (32.8) | 94.3 (44.8) |
| D-u60, 3.3 passes | 95.1 (97.3) | 94.0 (29.9) | 92.8 (36.7) |
| D-u140, 7.7 passes | 95.4 (95.2) | **59.3** (22.4) | **22.5** (13.3) |
| O-u140, 1 pass over 4,480 | 94.6 (94.1) | 94.7 (32.1) | 94.8 (74.2) |

**Prediction 1.** B(D-u140) − B(O-u140) = 0.110 [0.104, 0.116]. That is 24% of
the base's per-token NLL on the common set (0.459), against a registered floor
of 5%.
- B: D-u20 0.0007, D-u60 0.0158, D-u140 0.1106, O-u60 0.0003, O-u140 0.0004.
- B′: D-u140 0.076, O-u140 0.0004.
- A: D-u140 8.3%, O-u140 0.8%.
- 63% of D-u140's B comes from positions where it keeps the base's top token.
- 140 updates on fresh samples leave the base's reasoning where it was; 140
  updates cycling 579 of the same samples move it.

**Prediction 2.** B rises with passes (0.0007, 0.0158, 0.1106).

**Prediction 3.** D-u140's excess loss, with 95% paired intervals:
- over D-u20: +35.2 [31.0, 39.3] under instruction tuning and +71.8 [68.1,
  75.7] under Stage B;
- over O-u140: +36.2 [31.9, 40.5] and +73.1 [69.3, 76.8].

Every other state loses at most 2.9 points forced, the base under Stage B.

**Prediction 4.** Spearman 1.0 under instruction tuning and 0.4 under Stage B
(five points each, descriptive).

**Prediction 5.** The two clocks.
- Unforced accuracy falls more than forced accuracy for the base and O-u140
  under both stages.
- After instruction tuning, 0% of unforced answers from any state open a think
  block (base 39% and trained states 100% before).

**How the drilled state fails.** After Stage B, D-u140's forced outputs are
short and often unboxed, not endless:
- mean 867 tokens, against about 3,000 for the other states;
- 40% have no `\boxed{}`, against 3–4%;
- only 2.6% reach the cap.

Reading `<answer>N</answer>` tags recovers nothing. After instruction tuning:
2,007 tokens on average, 20% unboxed, 5.9% capped. The TCES drilled states
instead searched to the cap, so the surface failure differs by task.

**AIME 2025–26** (59 problems, forced, four draws, within 32,768; descriptive):

| State | Handoff | After instruction tuning |
|---|---|---|
| Base | 75.8 | 71.6 |
| D-u140 | 69.9 | 22.0 |
| O-u140 | 69.5 | 70.8 |

The 32k budget matters for AIME: within 16,384, the base's handoff score is
51.7.

**Reading.** On competition math with the base model's own correct solutions,
drilling reproduces the TCES result, and more sharply:
- the drilled model is as accurate as the others at handoff;
- it has moved the base's reasoning off its alternatives;
- it loses 36 and 73 points under two unrelated later stages, while one pass
  over fresh solutions for the same number of updates loses nothing.

Unlike TCES, the 3.3-pass state does not lose.
