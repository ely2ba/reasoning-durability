# Main phase: the cause, scale and family of the break in drilled reasoning

Draft 2026-09-24 05:38 UTC. It adopts a reviewed plan
with the changes listed under
"Changes from the plan". Each item is fixed when its launch time and code hashes
are added at the end, before any of its samples.

**Authorization.** [redacted: private message]

**Budget.**
- [redacted: account balance]
- The items project about $530 at their own list prices. Each run has its own
  limit of about 1.25 times its projection.
- [redacted: account cap]

## Why

These are the verified facts this phase builds on:
[math-pilot.md](math-pilot.md) (decision "go"), [anchor.md](anchor.md), and
[depth-profile.md](depth-profile.md) (the 16k consistency probes).

**Math pilot, Qwen3.5-9B-Base.**
- Every state starts at 94.6–95.4% forced on MATH-500 levels 3–5.
- After instruction tuning or Stage B, the drilled D-u140 is at 59.3% and
  22.5%; every other state is at 91.7–95.6%.
- Its excess loss over O-u140, which saw fresh problems once in the same 140
  updates, is 36.2 [31.9, 40.5] and 73.1 [69.3, 76.8] points.
- B(D-u140) is 0.111 nats per token and B(O-u140) is 0.0004.
- On AIME, D-u140 goes from 69.9% to 22.0% after instruction tuning, and O-u140
  from 69.5% to 70.8%.

**TCES at 16k.** Every state starts at 75.5–81.5%. After Stage B the lasting
states lose 2–4 points and the drilled states about 70.

**Anchor.**
- Prediction 1 is not interpretable (the parity check failed); the dose and
  the replication hold.
- FA1's B is 0.009, yet it loses 27 points. So B is not sufficient.

**Open questions.**
1. Is the cause the repeated text, or the narrow set of problems? The pilot's D
   and O differ in both.
2. Does the break hold at a second scale and in a second model family?
3. Does it hold on a second drilled subset?
4. Is "B is not sufficient" an artefact of scoring only the first 1,024 tokens
   of the TCES texts?
5. Is the break an artefact of rank-32 adapters?

## Items

Every math item uses the pilot's recipe, pool, screen, evaluation sets, scorer
and both later stages: instruction tuning and Stage B, 20 updates each.

- Forced accuracy: four draws, 16,384-token cap.
- Free accuracy: two draws.
- B: scored against the model's own base on a common set of its base's top-p-1
  forced samples.

**M3: P arm on the 9B** (`tools/run_math_main.py --arms P`, about $90).
- The pilot D's 579 problems, in D's cycling order, for 140 updates.
- Every visit uses a fresh correct forced sample of the base (top-p 1, cap
  16,384), with the pilot's D sample first.
- Sampling stops at 8 kept samples or 24 draws per problem. A problem that runs
  out cycles its own samples; distinct texts per problem are reported.
- P-u140 is evaluated at handoff and after each stage, with AIME at handoff and
  after instruction tuning.
- B is scored on the pilot's common set.

**M1: Qwen3.5-35B-A3B-Base, full pilot replication**
(`tools/run_math_pilot.py --base-model Qwen/Qwen3.5-35B-A3B-Base`, about
$163).
- D and O come from the 35B's own screened samples.
- States: the base, D-u20, D-u60, D-u140, O-u60 and O-u140, at handoff and
  after each stage.
- AIME for the base, D-u140 and O-u140, at handoff and after instruction tuning.

**M2: second drilled subset D2 on the 9B** (`--arms D2`, about $55).
- 579 examples of the pilot's O outside D, drawn by hash (namespace
  `main-20260924`, key `s2`), for 140 updates.
- D2-u140 is evaluated at handoff and after each stage.
- The comparator is the pilot's O-u140.

**N1: full-length B on TCES** (`tools/run_depth_full.py`, about $10).
- The depth-profile common set is rescored to 4,096 tokens instead of 1,024.
- States: M0, T-u30, S-u230, U-u140, F-u140, FA1, FA025, FN and U2.
- It also reports the share of B at kept-top positions and the probability
  kept on M0's sampled alternatives.

**N2: rank-128 drilled arm on the 9B** (`--arms R128`, about $50).
- The pilot D's data and order, with LoRA rank 128, for 140 updates.
- It is evaluated at handoff and after each stage, with B.

**M5: Nemotron-3 Nano, a second family** (about $100).
- This needs a chat-template renderer and a check that Tinker trains LoRA on
  Nemotron. Its code, calibration and launch time are added separately, before
  its first sample.
- If Tinker cannot train Nemotron, gpt-oss-20b takes the slot.
- The predictions are M1's, measured against Nemotron itself.

**M3b: P arm on the 35B** (about $60, conditional). It runs only if M3's
result is decisive in either direction.

## Changes from the plan

- **D2.** The plan drew it from new problems after the pilot's pool position.
  Drawing it from O's other 3,901 examples keeps the pilot's structure (D
  inside O), needs no new sampling, and keeps O-u140 an exact comparator.
- **35B identity precheck.** A fresh LoRA scored the base's texts with a mean
  absolute difference of 0.0034 nats per token. The registered criterion was
  under 0.001, so it failed.
  - The model is nondeterministic by itself: two independently created base
    samplers differ by 0.0030 on average (signed mean −0.0002, maximum 0.35),
    and the same sampler scoring twice differs by 0.0049. This is the
    mixture-of-experts model's own numerical noise.
  - The fresh LoRA is therefore at the base's own noise floor.
  - For the 35B, B carries this noise: about ±0.003 per position with no
    bias. The effects predicted are about 0.1.
- **35B training samples.** They are drawn in the pilot's way, until 4,480 are
  kept.

## Predictions, written before data

Intervals are 95% paired item bootstraps. An "excess" is one state's loss minus
another's, in points of forced accuracy.

**M3 (P arm).**
1. B(P-u140) is at most half of B(D-u140). P-u140's handoff forced accuracy is
   within 5 points of D-u140's.
2. Under each stage, D-u140's loss exceeds P-u140's by at least 20 points, with
   the interval above zero. The repeated text, not the narrow set of problems,
   is then the cause. If handoff parity (item 1) fails, the reading is
   "ambiguous", whatever the losses.
   - Opposite reading: if P-u140's excess over O-u140 is at least 80% of
     D-u140's under both stages, the narrow set is the cause.
   - Otherwise: partial.

**M1 (35B).** The pilot's registered predictions and decision rules, applied by
`tools/analyze_math_pilot.py`.
- The 35B's own noise floor for B, A and the kept/flipped split is measured by
  scoring the common set with a second base sampler (about $1). "Half of B at
  kept positions" is read against that floor.

In addition:
- at least half of B(D-u140) comes from positions where D-u140 keeps the base's
  top token;
- on AIME (descriptive), D-u140 loses more after instruction tuning than O-u140
  does.

**M2 (D2).**
1. Under each stage, D2-u140's excess over O-u140 is at least 10 points, with
   the interval above zero.
2. B(D2-u140) is at least 5% of the base's per-token NLL.

**N1** (read either way). The TCES texts were drawn at top-p 0.95, which
shifts every state's B down by 0.003–0.05, so N1 is read relative to the
lasting states. Let L be the mean B of M0, T-u30 and S-u230 at 4,096 tokens.
- If FA1's B minus L is at least 0.05, the residual loss lies past 1,024
  tokens, and full-length B is the flag.
- If it is at most 0.02, "B is not sufficient" stands.
- The share of B at kept-top positions is reported only for states whose B is
  positive.

**N2.** Under each stage, D-r128-u140 loses at least half of what D-u140 loses.
- If not, the break depends on adapter capacity, and this goes into the
  Limitations with the numbers.
- Rank 128 at the same learning rate can change the effective step. So the
  reading also reports each rank's training loss at update 140 and its
  negative log-likelihood on D's 579 drilled texts, scored under the base,
  D-u140 and D-r128-u140. "Depends on capacity" is claimed only if both ranks
  drilled comparably (NLL drop from the base within 20% of each other).

`tools/analyze_math_main.py` implements M3, M2 and N2 exactly.

## Launch (M3, M2, N2, M1, N1)

Fixed and launched 2026-09-24 05:53 UTC. [redacted: account balance] An independent
pre-launch audit found no blockers; its six suspected issues are addressed
above.

| Run | Items | Command | Limit |
|---|---|---|---|
| `runs/math/main-9b-20260924` | M3 (P), M2 (D2), N2 (rank 128) | `tools/run_math_main.py` | $260 |
| `runs/math/main-35b-20260924` | M1 | `tools/run_math_pilot.py --base-model Qwen/Qwen3.5-35B-A3B-Base` | $210 |
| `runs/depth-profile/depth-full-20260924` | N1 | `tools/run_depth_full.py` | $15 |

Code SHA-256 at launch: `run_math_main.py` `c3074af6fd6f…`; `analyze_math_main.py` `fb4ed155cb96…`; `run_math_pilot.py` `2e69faa3b7a0…`; `analyze_math_pilot.py` `f018be395450…`; `run_depth_full.py` `25ed3cca67ad…`; `run_depth_profile.py` `57413d99c540…`; `run_decision_probe.py` `d4f2bb90f160…`; `math_score.py` `f9194799c176…`; 

## Launch (M5, Nemotron-3 Nano)

Fixed and launched 2026-09-24 05:58 UTC as `runs/math/main-nemotron-20260924`
(`tools/run_math_pilot.py --base-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`), limit $180 at
Nemotron's list prices. [redacted: account balance]

- **Format.** `tools/nemotron_format.py` (via `tools/math_models.py`) uses Nemotron's own chat
  template.
  - It is token-identical to the template on 7,047 prompts.
  - Forced reasoning prefills `<think>\n`, and direct answers `<think></think>`. Turns end on
    `<|im_end|>`.
  - The later stages use the same 640 No Robots and 4,096 Stage-B examples as the Qwen runs, as
    answer-only turns (`<think></think>` + answer, trimmed as the template does).
- **Precheck.** A fresh LoRA reproduces the base exactly (0.0 mean absolute difference over 1,032
  positions). The scorer reads all 20 hand-checked solutions correctly.
- **Differences from the Qwen runs.** Nemotron is post-trained and reasons by default. Its
  "unforced" prompt ends at the assistant header, which its own template never does.
- **Design.** The pilot's, run on Nemotron itself: the states and AIME as in M1.

## N1 result (written 2026-09-24 05:59 UTC; `runs/depth-profile/depth-full-20260924`, $3.18)

B on M0's 192 forced texts, scored to 4,096 tokens (509,143 positions; M0's own NLL 0.390).

| State | B at 4,096 | A | Share of B at kept-top positions | Probability kept on M0's sampled alternatives |
|---|---|---|---|---|
| T-u30 | −0.0001 | 0.013 | n/a (B ≤ 0) | 0.98 |
| S-u230 | −0.0034 | 0.022 | n/a | 0.91 |
| FA1 | 0.0085 | 0.031 | 0.83 | 0.95 |
| FN | 0.0132 | 0.041 | 0.75 | 0.78 |
| FA025 | 0.0222 | 0.049 | 0.75 | 0.87 |
| U-u140 | 0.0969 | 0.076 | 0.69 | 0.41 |
| F-u140 | 0.1387 | 0.081 | 0.68 | 0.30 |
| U2-u140 | 0.1538 | 0.082 | 0.71 | 0.25 |

**Registered reading.** L, the mean of M0 (0), T-u30 and S-u230, is −0.0012. FA1
minus L is 0.0097, at most 0.02, so **"B is not sufficient" stands**.

- At full length, FA1 looks like a lasting state on B and on the alternatives it
  keeps: 0.95, against 0.91–0.98 for the lasting states.
- Yet it loses 27 points within 16k after Stage B, where the lasting states lose
  2–4.
- The drilled states' figures at 1,024 tokens hold at full length: about
  two-thirds of their B sits at positions where they keep M0's top token, and
  they keep 25–41% of M0's probability on the alternatives it sampled.

## N1 extension: the dose states at 4,096 tokens (fixed 06:14 UTC)

N1 left the dose states' B at 1,024 tokens. For one scale across the paper, the
states of depth-profile prediction 4 that N1 did not cover are rescored the same
way: U-u20, U-u60, F-u20, F-u60, R-P47-u140 and the solver R-S47-u200
(`tools/run_depth_full.py --extra`, same run folder, results in
`summary-extra.json`; limit $6). The paths are those of the 1,024-token profile;
for the five states both runs share, they are identical. Nothing is predicted:
this only replaces the 1,024-token B in the paper's dose figure and
correlations. The registered N1 reading also averaged M0, whose B is zero by
construction, into L. Without M0, L is −0.0018 and FA1 minus L is 0.0103;
the reading is unchanged.

## Addendum: code hashes of the Nemotron launch (written 06:24 UTC)

The M5 launch above gave no code hashes. `run_math_pilot.py`, `math_models.py`,
`nemotron_format.py` and `run_decision_probe.py` were last changed at 05:54 UTC,
after the 05:53 launch of the 9B and 35B runs and before the 05:58 Nemotron
launch. The change split the model-family hooks into `math_models.py` for
Nemotron. The 9B and 35B processes run the 05:53 code, whose hashes are listed
under the first launch. The Nemotron run uses these: `run_math_pilot.py` `d8d537ffdc04…`; `math_models.py` `d132c934a053…`; `nemotron_format.py` `d8a7528ae6f8…`; `run_decision_probe.py` `ae6905fb3051…`; `analyze_math_pilot.py` `4d3a2cde6210…`; `math_score.py` `f9194799c176…`; 

## N1 extension result (written 06:25 UTC; `summary-extra.json`, $2.12)

B at 4,096 tokens, against the 1,024-token value of the depth profile:

| State | Passes | B at 4,096 | B at 1,024 | A | Probability kept on M0's alternatives |
|---|---|---|---|---|---|
| U-u20 | 1.1 | −0.0027 | −0.0027 | 0.019 | 0.89 |
| F-u20 | 1.1 | −0.0024 | −0.0025 | 0.019 | 0.90 |
| U-u60 | 3.3 | 0.0125 | 0.0153 | 0.037 | 0.74 |
| F-u60 | 3.3 | 0.0134 | 0.0157 | 0.038 | 0.74 |
| R-P47-u140 | 7.7 | 0.0465 | 0.061 | 0.068 | 0.68 |
| R-S47-u200 (solver) | 11 | 0.0684 | 0.093 | 0.078 | 0.67 |

The states with positive B lose 10–27% of it at full length. Their order is
unchanged.

## Results: M3, M2, N2, M1, M5 (written 10:39 UTC)

All runs finished without errors: 9B main $151.72, 35B $149.65, Nemotron $160.14
(at their own list prices). [redacted: account balance] Forced accuracy is on MATH-500 levels 3–5 (221 problems,
four draws, 16,384 tokens). Excess losses are against O-u140 of the same run,
except the 9B main arms, which use the pilot's O-u140 and D-u140.

**M3, the P arm (`tools/analyze_math_main.py`).** 577 of D's 579 problems got
eight distinct correct samples; only 6 of P's 4,480 training slots repeat a text.

| State | Handoff | After IT | After B | B |
|---|---|---|---|---|
| D-u140 (pilot) | 95.4 | 59.3 | 22.5 | 0.111 |
| P-u140 | 94.6 | 94.9 | 94.2 | 0.0004 |
| O-u140 (pilot) | 94.6 | 94.7 | 94.8 | 0.0004 |

- **Prediction 1 holds:** B(P) is 0.35% of B(D), and handoff parity holds (0.8
  points).
- **Prediction 2 holds:** D's loss exceeds P's by 36.4 [32.5, 40.5] points under
  instruction tuning and 72.5 [68.8, 76.1] under Stage B.
- **Registered reading:** the repeated text, not the narrow set of problems, is
  the cause. P's excess over O-u140 is −0.2 [−2.0, 1.5] and 0.6 [−1.7, 2.9].

**M2, a second drilled subset.** D2-u140 goes 93.9 → 81.4 (IT) and → 5.5 (B).
- **Prediction 1 holds:** the excess over O-u140 is 12.6 [8.8, 16.3] and 88.6
  [85.5, 91.5].
- **Prediction 2 holds:** B(D2) is 0.102, 22% of the base's NLL of 0.459.
- The size of the break varies with the subset: IT takes less from D2 than from D
  (12 against 36), and Stage B takes more (88 against 73).

**N2, LoRA rank 128.** D-r128-u140 goes 95.6 → 56.8 (IT) and → 12.9 (B).
- **Prediction 1 holds:** it loses 1.08 and 1.14 times what rank 32 loses.
- Both ranks drilled comparably: their NLL on D's texts falls from the base's
  0.453 to 0.277 (rank 32) and 0.270 (rank 128), drops within 4% of each other.
- **Reading:** the break does not depend on adapter capacity between ranks 32 and
  128.

**M1, Qwen3.5-35B-A3B-Base (`tools/analyze_math_pilot.py`).** Decision "go".
- **Registered predictions:**
  - 1, 2 and 3 hold;
  - 4 (the rank correlation of B with the loss) gives −0.1 under IT and 0.7 under B;
  - 5 fails for M0 under IT;
  - handoff parity holds (1.8 points).
- **Excess of D-u140 over O-u140:** 38.3 [33.7, 43.0] under IT and 72.1 [68.0,
  76.0] under B.
- **B:** D-u140 0.160, O-u140 0.0009 (base NLL 0.499). 57% of D-u140's B sits at
  positions where it keeps the base's top token (predicted: at least half). The
  noise-floor measurement for the 35B's B is still to run.
- **AIME** (descriptive, as predicted): D-u140 72.0 → 11.4 and O-u140 78.8 → 77.5.
- **Not predicted.** The untrained 35B base also breaks under instruction
  tuning: 96.0 → 48.1 forced, and 79.7 → 5.9 on AIME. Trained once (O-u140) it
  keeps 91.0. Under Stage B the base holds (93.7). This is confounded: the base
  continued from a fresh LoRA, and every trained state from its own adapter.
  D-u20, also a trained adapter, loses only 2.6. So the protection may come from
  starting the stage on a trained adapter, not from the training itself.
- **Failure form.** After Stage B both forms occur on the 35B. The median
  drilled sample is 270 tokens and 62% end within 1,024 tokens (9B: 103 and 90%),
  while 26.6% run to the cap and 98% of those loop.

**M5, Nemotron-3 Nano.** Decision "go".
- **Registered predictions:**
  - 1, 2 and 3 hold;
  - 4 gives 0.7 and 0.9;
  - 5 fails for M0 under B;
  - parity holds (0.0).
- **Excess over O-u140:** 59.2 [54.5, 63.8] under IT and 40.0 [35.6, 44.5] under B.
- **B:** D-u140 0.114, O-u140 0.0002 (base NLL 0.640). 61% of B sits at kept-top
  positions.
- **Forced accuracy:**

  | State | Handoff | After IT | After B |
  |---|---|---|---|
  | D-u140 | 93.4 | 22.3 | 54.2 |
  | O-u140 | 93.4 | 81.4 | 94.2 |
  | Base | 93.6 | 83.6 | 94.1 |

- **AIME:** D-u140 70.8 → 2.5, O-u140 74.6 → 64.4 and the base 72.9 → 67.8.

**M1 noise floor (written 10:39 UTC; `tools/run_noise_floor.py`, $0.17).** A second,
independently created 35B base sampler scored the run's common set. Its B against
the first scoring is 0.0000 [−0.00002, 0.00002], A is 0.0001, and the mean absolute
per-token difference is 0.0003 nats: a tenth of the precheck's 0.0030 on short
texts. Every state's B lies above this floor, O-u140's 0.0009 included.

## Additions (2026-09-24 11:04 UTC)

[redacted: account cap] Added: the
teacher-trace experiment (drilled, fresh and once-trained arms on a stronger
model's traces) and two robustness checks (early dynamics; instruction tuning at
learning rate 1e-4). Each gets its own record and limit before any sample.
