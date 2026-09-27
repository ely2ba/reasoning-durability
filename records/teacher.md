# Teacher traces: does repetition make distilled reasoning fragile too?

Draft 2026-09-24 11:09 UTC. Items are fixed when their launch time and code hashes are
added below, before any of their samples. The design follows an internal
review.

## Why

Every result so far trains on reasoning the model could already produce:
- on math, its own correct forced samples;
- on arithmetic search (TCES), its RL teacher's samples.

The widely used small-set recipes (s1, LIMO) are different: a few epochs on
about a thousand long traces from a stronger model. On TCES, repetition made a
skill the model lacked more durable, not less (the solver arm R-S). So in the
teacher regime the outcome could go either way, and both answers matter for
the paper's scope.

## Design

**Teacher: gpt-oss-120b on Tinker, at high reasoning effort.**
- Sampling: temperature 1, top-p 1, at most 16,384 tokens, with the pilot's
  instruction to box the final answer.
- A trace is kept if it meets all of these:
  - it is a clean two-part reply (analysis, then final), with no other channel,
    tool call or stray control text;
  - the last boxed integer of its final part equals the answer;
  - it is under 16,384 tokens.
- Calibration on the first 50 of the pilot D's problems: 47 correct; median 1,511
  tokens, mean 2,209, one capped. ($0.22 at the assumed prices below; 16 pool
  problems before that: 14 correct.)

**Student format.** The pilot's role-colon prompt, then "\n<think>\n" + the
teacher's analysis + "\n</think>\n\n" + the teacher's final part + the
end-of-text token, in Qwen3.5-9B-Base's tokens. It has the same shape as the
base's own forced traces, which close their reasoning block before answering.

**Arms.** All use exactly the pilot's problems and orders (`runs/math/pilot-20260924/arms.json`)
and the pilot's recipe: a fresh rank-32 LoRA, 140 updates of 32 at 1e-4, the
same loss normalization.

| Arm | Problems | Traces | Passes |
|---|---|---|---|
| D-T | D's 579 | one trace each (the first kept) | 7.7; also saved at update 20 (D-T-u20, 1.1 passes) |
| O-T | O's 4,480 | one trace each | 1 (D's traces are among O's, as in the pilot) |
| P-T | D's 579 | a different kept trace at each visit (up to 8 per problem, drawn as P) | |

The teacher gets at most 24 draws for each of D's problems and 4 for each of O's
others. A problem left without a trace drops out of its arms and is counted.

**Later stages, as in the pilot.** Instruction tuning (640 No Robots) and Stage B
(answer-only program synthesis), 20 updates each at 3e-4, from each of D-T-u20,
D-T-u140, O-T-u140 and P-T-u140.

**Measures.**
- Forced accuracy on MATH-500 levels 3–5 (221 problems, four draws, 16,384
  tokens) at handoff and after each stage.
- Free accuracy (two draws) at handoff.
- AIME (59 problems, four draws, 32,768 tokens) for D-T and O-T, at handoff and
  after instruction tuning.
- B of each state on the pilot's base texts. It is descriptive only: foreign
  traces move every arm away from the base.

## Predictions, written before data

An excess is one state's forced loss minus another's, with paired item-bootstrap
intervals (20,000 resamples).

**Parity.** D-T-u140 and O-T-u140 are within 5 points at handoff. If not, the
reading is "ambiguous". The excess over D-T-u20 and the relative losses are still
reported.

1. **Repetition breaks distilled reasoning.** Under each stage, D-T-u140's excess
   over both O-T-u140 and D-T-u20 is at least 10 points, with the interval above
   zero.
2. **Consolidation.** Under both stages, D-T-u140's excess over O-T-u140 is under
   3 points, with parity.
3. **Otherwise: partial.** It is reported as it falls.
4. **The fix carries over.** Under each stage, P-T-u140's excess over O-T-u140 is
   at most 5 points.

**Meaning.**
- If 1 holds, the paper's claim extends to the s1/LIMO distillation regime.
- If 2 holds, the claim narrows to repeating reasoning the model could already
  produce. The title and abstract say so.

## Cost and limits

The teacher's list price is not in our records. Its ledger assumes $0.70 per
million prefill tokens and $2.00 per million sampled, which is probably high;
the actual price can be reconciled from the provider's console.

| Phase | Estimate | Limit |
|---|---|---|
| Teacher (about 20M sampled tokens) | $45 | $60 |
| Student at the 9B's list prices (training about $50, evaluations about $80, the rest under $5) | $135 | $165 |

[redacted: account cap]

## Launch: teacher phase

Fixed and launched 2026-09-24 11:10 UTC as `tools/run_teacher.py --phase teacher`
(`runs/math/teacher-20260924/teacher/`), limit $60 at the assumed prices. [redacted: account balance] The two teacher calibrations cost $0.40
at the assumed prices. The student phase and its predictions are fixed
separately, before any student sample.

Code SHA-256: `run_teacher.py` `1abaa27d1917…`; `run_math_pilot.py` `d8d537ffdc04…`; `math_models.py` `d132c934a053…`; `math_score.py` `f9194799c176…`; `run_decision_probe.py` `ae6905fb3051…`; 

## Relaunch of the teacher phase (2026-09-24 11:14 UTC)

The first launch was stopped at 11:13 UTC after $10.18 at the assumed prices,
before any chunk was written, to add one field. Each kept trace now records the
draw that produced it. This allows the inclusion rule requested in review:
a problem enters D-T and O-T only if the teacher solved it within its first 4
draws, for D's problems as for O's. P-T's further traces may come from up to 24
draws. The limit is $70, counting the $10.18. `run_teacher.py` SHA-256 is now
`7d395733ca49…`.

## Changes before the student phase (written 11:15 UTC, after review)

1. **Symmetric inclusion,** already in the relaunch. A problem enters D-T and O-T
   only if the teacher solved it within its first 4 draws.
2. **O-T keeps the included O problems.** No problem from outside the pilot's O
   is added, since those were never screened by the base. If fewer than 4,480
   remain, the last slots cycle from the start. The counts are reported.
3. **Prediction 2, tightened.** Under both stages the excess is under 3 points
   and its interval's upper bound is under 10, with parity.
4. **Two conditions for reading a negative result (prediction 2).**
   - **Drilling happened.** From updates 16–20 to updates 136–140, D-T's
     per-example training loss falls by at least 80% as much as the pilot D's did
     over the same updates (training logs, free).
   - **The student learned the teacher.** On held-out teacher traces, the
     per-token NLL under O-T-u140 is at least 20% below the base's. The held-out
     traces are each D-T problem's second kept trace, which neither D-T nor O-T
     trained on, capped at 200. Opening-style markers are not usable: the
     teacher's openings ("We need to …") are the base's own.
5. **Wording.**
   - The solver: the one TCES arm trained on a skill the model lacked (solver
     derivations, 11 passes) lost 18 points, against 53–59 for the drilled arms.
   - If prediction 1 holds, the claim extends to repeating a stronger model's
     long traces for several epochs, the data regime of s1 and LIMO. Ours uses
     LoRA on a base model, not their full fine-tuning of an instruct model.
6. **Also reported:** the kept-trace length distribution per arm, and the
   distinct texts per P-T problem.

## Second change before the student phase (written 11:27 UTC, after an independent review)

1. **Usable traces.** The Qwen tokenizer spends about 6% more tokens than
   gpt-oss's on these texts, so a trace under 16,384 teacher tokens can exceed
   the pilot's training length (17,000) as a student row. In the first chunk, 3
   of 2,038 kept traces do. A kept trace is usable only if its student row fits.
   - A problem enters D-T and O-T only if its first usable trace came within the
     teacher's first 4 draws.
   - P-T uses every usable trace.
   - The held-out traces for the learned-the-teacher check are the second usable
     traces.
   - The count of unusable traces is reported.
2. **No hidden channels.** A review noted that parse() would accept, and
   drop, a commentary message between the analysis and the final part. Every
   kept trace so far holds 9–11 tokens more than its analysis and final parts:
   the channel headers only, so no kept trace carried such a message.

## Third change before the student phase (written 11:42 UTC, after its re-check)

1. **An exactly symmetric rule.** An O-only problem is sampled until its first
   kept trace. The rule of the second change would let a D problem enter through
   a later usable trace when its first kept trace is too long, while the same O
   problem would be out. A problem now enters only if its first kept trace is
   itself usable and came within the first 4 draws. This affects a handful of
   problems.
2. **A completeness check.** The student phase refuses to start unless every O
   problem is present in the traces.

## Prediction 5, an out-of-sample test of the narrowing account (written 12:08 UTC)

The robustness record (R1, R3b) found the following:
- The first instruction-tuning updates drive the self-trained drilled model off
  its own solutions (0.067 nats per token after two updates, against 0.0005 for
  O).
- On the base's texts it does not move away (−0.007).

The student phase now tests this account on data it has not seen. D-T-u140,
O-T-u140 and P-T-u140 each draw their own forced texts at handoff (one top-p-1
sample per MATH-500 level 3–5 problem, 2,048 tokens). Those texts and the base's
common texts are scored at handoff and after update 2 of the instruction-tuning
stage (the same run as its update 20). Cost: about $5.

**Prediction 5,** read only if prediction 1 holds. After two updates:
- D-T's displacement on its own texts is at least 10 times O-T's;
- on the base's texts, D-T's displacement lies within ±0.01 nats per token.

P-T is reported.

## Teacher phase result and student launch (2026-09-24 12:12 UTC)

**The teacher phase finished.**
- 4,480 problems, 8,480 kept traces, $53.77 at the assumed prices ($10.18 of it
  from the stopped first launch).
- The arms, as the student phase builds them:
  - D-T: 575 problems (4 dropped);
  - O-T: 4,441 problems (39 dropped);
  - P-T: 4,480 slots over D-T's 575 problems, 7.99 kept traces per problem on
    average;
  - 13 traces unusable (too long as student rows).
- Median student length: D-T 1,473 tokens, O-T 1,652, P-T 1,492.

**Student phase.** Fixed and launched as `tools/run_teacher.py --phase student`,
with limit $165 at the 9B's list prices. [redacted: account balance] The
registered analysis is `tools/analyze_teacher.py`.

Code SHA-256: `run_teacher.py` `20d857844cbc…`; `analyze_teacher.py` `ff7a207a0dc4…`; `run_math_pilot.py` `d8d537ffdc04…`; `math_models.py` `d132c934a053…`; `run_decision_probe.py` `ae6905fb3051…`; `run_depth_profile.py` `57413d99c540…`; 

## Student phase relaunch (2026-09-24 12:36 UTC)

A pre-launch check projected that the student phase would pass its $165 limit
(about $176), because the students write about 3,200 tokens per answer and the
record's evaluation estimate was too low. It also warned that calls are refused
before the limit, which aborts whole evaluations. The run was stopped at
$91.0857.

By then everything was trained, and five evaluations had finished: the forced
handoff evaluation of all four states, and AIME for O-T at handoff. It was
relaunched unchanged with a limit of $210 [redacted: account cap].
Finished outputs are skipped, and the evaluations that were in flight at the
stop are redrawn. No result had been read.

## Prediction 5 by replay (2026-09-24 12:57 UTC)

The two-update probe was to run inside the instruction-tuning stage. The first
student launch trained every later stage, but it was stopped (see "Student phase
relaunch") before the probe's scoring finished. The relaunch found the stage
states saved and did not retrain, so the update-2 scores were never written.

The probe now replays the first two updates of the same stage (the same
examples, order and settings) from each u140 state, then scores the same texts:
`tools/run_teacher.py --phase probe`, in the same run folder and ledger. R1
showed such replays match the logged losses within 0.1%, and the replayed losses
are checked against this stage's own logs. This is a deviation in how the
update-2 state is obtained. The prediction and its texts are unchanged, and no
update-2 score had been read. SHA-256: `f90b3d112f6f…`.

## Result (written 12:57 UTC; `tools/analyze_teacher.py`, `runs/math/teacher-20260924/summary.json`; student phase $156.67)

Forced accuracy on MATH-500 levels 3–5 (221 problems, four draws, 16,384
tokens), with free accuracy at handoff:

| State | Handoff (free) | Handoff (forced) | After IT | After B |
|---|---|---|---|---|
| D-T-u140 (drilled) | 93.9 | 95.0 | 33.7 | 1.1 |
| D-T-u20 (1.1 passes) | 95.0 | 96.2 | 95.4 | 94.3 |
| O-T-u140 (once) | 95.7 | 96.3 | 94.6 | 94.8 |
| P-T-u140 (fresh texts) | 95.5 | 96.0 | 95.4 | 95.4 |

AIME (forced, four draws): D-T 73.7 → 11.0 after IT; O-T 72.9 → 78.0.

**Registered reading: "repetition breaks distilled reasoning".**
- **Parity holds:** 1.2 points.
- **Prediction 1 holds.** D-T's excess is at least 10 points over both
  comparators under each stage, with intervals above zero:
  - over O-T: 59.6 [55.5, 63.7] (IT) and 92.4 [89.4, 95.1] (B);
  - over D-T-u20: 60.5 [56.4, 64.5] and 92.1 [88.9, 94.9].
- **Prediction 4 holds.** P-T's excess over O-T is −1.0 [−2.6, 0.5] and −0.8
  [−2.7, 1.1]. Fresh traces for the same problems remove the effect in the
  distillation regime too.
- **Prediction 2 fails**, and the conditions for reading it also fail:
  - drilling ratio 0.69, against 0.8 required;
  - held-out NLL under O-T only 1.2% below the base's (0.591 against 0.598),
    against 20% required.

  Neither matters here, since prediction 1 holds.

**Notes.**
- Under this recipe the student moved only a little toward the teacher on
  held-out traces: O-T's NLL fell 1.2%.
- D-T-u140 finds held-out teacher traces harder than the base does (0.724
  against 0.598). This is the over-sharpening signature again.
- The break is larger than with the model's own solutions: 59.6 and 92.4,
  against 36.2 and 73.1.
- Prediction 5 is pending its replayed probe.

## Prediction 5 result (written 12:58 UTC; replay probe $1.37; student run folder $158.05 in all)

The replayed first two updates match the stage's logged losses within 0.07% for
all three states.

Displacement after two instruction-tuning updates, in nats per token of
log-likelihood lost since handoff:

| State | Own handoff texts | The base's texts |
|---|---|---|
| D-T-u140 | 0.0583 | −0.0257 |
| O-T-u140 | 0.0004 | 0.0056 |
| P-T-u140 | 0.0006 | 0.0054 |

**Prediction 5 fails as registered.**
- **Part (a) holds.** D-T's own-text displacement is 140 times O-T's (at least
  10 required).
- **Part (b) fails.** On the base's texts D-T lies outside the ±0.01 band. It
  moves toward the base, by 0.026 nats per token: its handoff B on these texts
  goes down, not up.

**Reading.** The pattern of R1 and R3b recurs on data it had not seen, and more
strongly. The later stage drives the drilled model off its own solutions (140×)
while moving it toward the base's reasoning, not away. The band in part (b) was
set too tight for the size of that movement. The paper reports prediction 5 as
failed and states this pattern.

## Corrections after an independent recomputation (written 13:10 UTC)

Every number above was recomputed independently from the raw files [redacted: internal path].
The accuracies, excesses, AIME scores, held-out NLLs and displacements all reproduce; the excess
intervals agree to Monte Carlo precision (one bootstrap grid step, 0.11 points). Four wordings change:
- **"Within 0.07%"** becomes **within 0.08%**: the largest difference is 0.0706% (D-T, update 2).
  Update 1 is not bit-exact for D-T and P-T, although state and batch are the same, so the probe
  is a faithful replay, not the stage's own update-2 state.
- **"More strongly"** holds only on the base's texts (D-T −0.026 against R3b's −0.007). On its
  own texts D-T's 0.058 is below R1's 0.067 for D.
- **"The band in part (b) was set too tight"** is a judgement made after the result, and is marked
  as such wherever it appears.
- **"140 times"** is 140.5 (0.0583 / 0.000415); rounded, 141, as the paper prints it.

**Failure form** (an independent count; the pilot's D-u140 for comparison):

| Condition | Median tokens | Under 100 tokens | Ends by opening a user turn | Closes </think> | No \\boxed | At cap |
|---|---|---|---|---|---|---|
| D-T after IT | 277 | 18.8% | 93.1% | 6.4% | 39.6% | 2.6% |
| D-T after B | 22 | 93.7% | 99.1% | 0% | 35.1% | 0.9% |
| D-u140 after IT | 451 | 7.5% | 77.7% | 29.2% | 20.1% | 5.9% |
| D-u140 after B | 103 | 49.2% | 96.6% | 0.9% | 40.3% | 2.6% |

The failure is the pilot's, stronger: the turn ends from inside the reasoning block, the block rarely
closes, and responses rarely reach the cap. At handoff D-T closes the block in 96% of responses (median
position 1,064 tokens).
