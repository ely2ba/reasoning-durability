# Re-forcing: is the drilled model's math loss lost competence, or a failure to keep reasoning?

Draft 2026-09-24 06:22 UTC. It is fixed when the launch time and code hashes are added below,
before any sample. It answers a point raised in review.

## Why

After Stage B, D-u140's forced math answers are short: median 103 tokens, against
1,230 at handoff. 89.7% of them end the turn within 1,024 tokens, and those
short answers are right 21.2% of the time. 39% (346 of 884) end the turn with no boxed
answer at all. On TCES, the arithmetic task, forced samples that stop on a restated
template are resumed. Math samples get no such retry. So the math loss may be a
failure to keep reasoning, not a failure to solve.

On TCES the question does not arise in the same way. After Stage B, the drilled
searches run to the 16,384-token cap (F-u140 336 of 384, U-u140 358). A scan of
their text for any expression the verifier accepts finds one in 9 of F-u140's
capped searches and 7 of U-u140's. They search and do not find.

## Design

**Conditions.** The pilot's own forced samples (`runs/math/pilot-20260924`, 221
problems, 4 draws) of M0, D-u140 and O-u140, each at handoff and after 20 updates
of each later stage (nine conditions). No new first-pass samples are drawn.

**Continuation.** A sample that ended its turn before the 16,384-token cap is
continued as follows.
- Its end-of-turn is removed: a final end-of-text token, or a trailing
  "\n\nUser:" or "\nUser:".
- "\n\nWait" is appended, as in s1's budget forcing (Muennighoff et al., 2025).
- Sampling resumes from the whole sequence with the evaluation's settings
  (temperature 1, top-p 0.95, the same stops), with the remaining budget.
- This repeats at most eight times, and never past 16,384 tokens in all.
- The final text is scored by the registered scorer (last boxed integer).

**Two rules** decide which samples are continued.
- **Budget rule (primary).** A sample is continued while it has fewer than 1,024
  tokens or no `\boxed`. So every answer rests on at least 1,024 tokens of
  reasoning, unless the eight continuations run out. The base's median forced
  solution is 1,402 tokens.
- **Answer rule (secondary).** A sample is continued while it has no
  `\boxed`, whatever its length.

The rules look only at length and at whether an answer was boxed, never at
correctness. Both apply to every condition, so the lasting states' short, correct
answers are continued under the budget rule too.

**Other models and arms.** The same code and rules apply to the 35B and Nemotron
replications and to the 9B main arms (P, D2, rank 128) once those runs finish.
Nemotron's end-of-turn is its `<|im_end|>` or `</s>` token.

## Predictions, written before data

Let E be the registered excess loss of D-u140 over O-u140 under a stage: 36.2
points under instruction tuning and 73.1 under Stage B. Let E_b be the same
quantity with every condition scored under the budget rule. Intervals are paired
item bootstraps (20,000 resamples).

1. **Competence is lost.** Under both stages, E_b is at least E/2 and its
   interval excludes zero.
2. If E_b is below E/2 under a stage, that stage's math loss is mostly a failure
   to keep reasoning. The paper then says the drilled model stops reasoning early,
   and that kept reasoning, it recovers the stated points.

The answer rule is read the same way and reported as secondary. Other outcomes
are reported as they fall.

Also reported:
- the number of continuations used;
- final lengths;
- the share still unboxed;
- the lasting states' accuracy change under each rule (a cost of the probe itself).

## Cost

About $20–35 for the pilot's conditions at the 9B's list prices:
- continuations of 4,023 samples under the budget rule, 1,852 of them
  D-u140's (counted by a dry run);
- 493 under the answer rule.

Limit $45. The main-phase runs are added under their own limits when they finish.

## Launch (pilot conditions)

Fixed and launched 2026-09-24 06:24 UTC as `tools/run_reforce.py --run-id pilot-20260924`,
output in `runs/math/pilot-20260924/reforce/`, limit $45. [redacted: account balance]

Code SHA-256 at launch: `run_reforce.py` `1c43b04a23f0…`; `run_math_pilot.py` `d8d537ffdc04…`; `math_models.py` `d132c934a053…`; `math_score.py` `f9194799c176…`; `run_decision_probe.py` `ae6905fb3051…`; 

## Relaunch (2026-09-24 10:38 UTC)

The first launch was stopped at 06:40 UTC after spending $20.78 without finishing
any condition. Its outputs were written only when a whole condition was done, so
at the $45 limit all unfinished work would have been lost. None of its samples was
written or read. The relaunch changes only the bookkeeping:
- each finished sample is appended to a partial file, so a stop loses nothing;
- the limit is $100, counting the $20.78 already spent.

The design, rules and predictions are unchanged. `run_reforce.py` SHA-256 is now
`24ec99ce984a…`. [redacted: account balance]

## Result (written 11:15 UTC; `tools/analyze_reforce.py`, `runs/math/pilot-20260924/reforce/summary.json`)

The spend was $68.47, including the first launch's $20.78. The table gives
forced accuracy on MATH-500 levels 3–5 (221 problems, four draws), first pass
and under each rule.

| State | First pass | Budget rule | Answer rule |
|---|---|---|---|
| M0 | 94.7 | 92.8 | 94.7 |
| M0 + IT | 95.6 | 93.3 | 95.7 |
| M0 + B | 91.7 | 82.5 | 92.3 |
| D-u140 | 95.4 | 93.7 | 95.4 |
| D-u140 + IT | 59.3 | 71.7 | 69.0 |
| D-u140 + B | 22.5 | 26.1 | 37.6 |
| O-u140 | 94.6 | 92.9 | 94.6 |
| O-u140 + IT | 94.7 | 92.2 | 94.7 |
| O-u140 + B | 94.8 | 93.7 | 94.8 |

**Excess of D-u140's loss over O-u140's, with every condition under the same
rule:**

| Stage | Registered (first pass) | Budget rule | Answer rule |
|---|---|---|---|
| IT | 36.2 [31.9, 40.4] | 21.3 [16.7, 25.7] | 26.5 [22.3, 30.7] |
| B | 73.1 [69.3, 76.8] | 68.3 [64.4, 72.3] | 58.0 [53.7, 62.2] |

**Prediction 1 holds under both stages and both rules.** Each rule's excess is
at least half the registered one, with its interval above zero. Most of the
drilled model's loss is lost competence, not a failure to keep reasoning.
- After instruction tuning, keeping D-u140 reasoning recovers 12.4 of its lost
  points (59.3 → 71.7), which removes 41% of the excess. The other 59% stays.
- After Stage B it recovers 3.6 points (22.5 → 26.1), 7% of the excess.
- The answer rule, which continues only unboxed answers, recovers more after
  Stage B (to 37.6). The budget rule also continues short boxed answers, and
  these often run to the cap.

**The probe's cost to the other states.** The cue changes finished answers.
Under the budget rule the lasting states lose 1.1–2.5 points, and M0 after Stage
B loses 9.3. This biases the test against the competence reading, which makes
its result stronger.
