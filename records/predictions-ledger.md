# Ledger of registered predictions behind the paper

Compiled 2026-09-24, 17:20 UTC, by an independent reviewer.
The revision parts (revision.md) were added on 2026-09-26, 03:09 UTC, and SD's outcomes at 03:48 UTC.

Sources:
- The records in this folder. Outcomes are taken from each record's own result
  sections and checked against independent recomputations where they were made (depth profile,
  math pilot, main phase, re-forcing, anchor, teacher, R1-R5).
- [redacted: citations of a draft of the paper by line number]

## Counting rule (proposed)

- **Counted.** Every item a record lists under "Predictions, written before data", or as
  "Prediction:" in a later fixed-before-data part, that has a pass/fail criterion.
  - An item holds only if every registered part holds.
  - "Holds for some arms, not others" counts as failed and is marked *partial*.
  - Replications that re-apply a design's predictions count once per model: 35B = M1, Nemotron = M5.
- **Not counted as predictions** (listed in section 3):
  - conditions: parity, mechanism and reading checks, trajectory checks;
  - two- or three-way readings, where the record states the alternative outcomes in advance;
  - items marked descriptive with no threshold;
  - "expectations" that a record marks as not a hypothesis;
  - secondary predictions, which are counted separately.
- **Teacher prediction 2 ("consolidation")** is the registered rival to prediction 1. The record
  numbers it as a prediction and reports it as failed, so it is counted as failed here. Section 4
  also gives the count without it.

Outcomes: H = holds; F = fails (F* = fails in part); NR = not readable, or never evaluated.

## 1. Mathematics

### math-pilot.md ("Predictions, written before data"; results "Decision: go")

| ID | Registered (one line) | Threshold | Observed | Out |
|---|---|---|---|---|
| MP1 | Repetition over-sharpens | B(D-u140)-B(O-u140) >= 5% of base NLL, CI > 0 | 0.110 [0.104, 0.116] = 24% | H |
| MP2 | Dose | B(D-u20) < B(D-u60) < B(D-u140) | 0.0007 < 0.0158 < 0.1106 | H |
| MP3 | Over-sharpened competence is fragile, per stage | D-u140 loses more than D-u20 and O-u140, both CIs > 0 | over O 36.2 [31.9, 40.5], 73.1 [69.3, 76.8]; over D-u20 35.2, 71.8 | H |
| MP5 | Two clocks: unforced falls more than forced for base and O-u140, each stage | 4 cells | all 4 | H |

### main-phase.md ("Predictions, written before data"; "Results: M3, M2, N2, M1, M5")

| ID | Registered | Threshold | Observed | Out |
|---|---|---|---|---|
| M3.1 | P keeps B low, with parity | B(P) <= B(D)/2 and handoff within 5 | B(P) = 0.35% of B(D); gap 0.8 | H |
| M3.2 | Repeated text is the cause | D - P >= 20 each stage, CI > 0 | 36.4 [32.5, 40.5], 72.5 [68.8, 76.1] | H |
| M1.1 | 35B: pilot P1 | as MP1 | 0.160 - 0.0009 = 32% of 0.499 | H |
| M1.2 | 35B: dose | as MP2 | holds (record) | H |
| M1.3 | 35B: fragile | as MP3 | over O 38.3, 72.1; over D-u20 41.5, 73.4 | H |
| M1.5 | 35B: two clocks | as MP5 | fails for base under IT (free -36.9, forced -48.0) | F* |
| M1.6 | 35B: at least half of B at kept-top positions | >= 50% | 57% | H |
| M1.7 | 35B: on AIME, D-u140 loses more after IT than O-u140 (marked descriptive) | direction | D 72.0 -> 11.4, O 78.8 -> 77.5 | H |
| M5.1-M5.3 | Nemotron: pilot P1-P3 | as MP1-MP3 | B 17.8% of NLL; dose holds; over O 59.2, 40.0 | H (3) |
| M5.5 | Nemotron: two clocks | as MP5 | fails for the start model under B (free +2.3, forced +0.6) | F* |
| M5.6 | Nemotron: kept-top >= half | >= 50% | 61% | H |
| M5.7 | Nemotron: AIME direction | direction | D 70.8 -> 2.5, O 74.6 -> 64.4 | H |
| M2.1 | Second subset breaks | D2 - O >= 10 each stage, CI > 0 | 12.6 [8.8, 16.3], 88.6 [85.5, 91.5] | H |
| M2.2 | Second subset over-sharpens | B(D2) >= 5% NLL | 22% | H |
| N2 | Rank 128 breaks too | loses >= half of rank 32's, each stage | 1.08x, 1.14x | H |

### reforcing.md

| ID | Registered | Threshold | Observed | Out |
|---|---|---|---|---|
| RF1 | Competence is lost | E_b >= E/2 and CI excludes 0, both stages (budget rule) | 21.3 [16.7, 25.7] >= 18.1; 68.3 [64.4, 72.3] >= 36.6 | H |

### teacher.md

| ID | Registered | Threshold | Observed | Out |
|---|---|---|---|---|
| T1 | Repetition breaks distilled reasoning | excess >= 10 over O-T and D-T-u20, each stage, CI > 0 | 59.6, 92.4; 60.5, 92.1 (lower bounds >= 55) | H |
| T2 | Consolidation (the rival to T1; tightened 11:15) | excess < 3 and upper bound < 10, each stage, with parity | 59.6, 92.4 | F |
| T4 | The fix carries over | P-T - O-T <= 5, each stage | -1.0, -0.8 | H |
| T5 | Out-of-sample narrowing: (a) own-text displacement at u2 >= 10x O-T's; (b) \|base-text displacement\| <= 0.01 | both | (a) 140.5x; (b) -0.0257 | F* |

### robustness.md

| ID | Registered | Threshold | Observed | Out |
|---|---|---|---|---|
| R1.1 | More displacement for D (own texts, u2) | δ2(D) >= 2 max(O, P), CI > 0 | 0.0656 [0.060, 0.072] | H |
| R1.2 | Displacement before collapse | D forced at u2 > 77.4 | 81.8 | H |
| R1.3 | Fresh texts like once-trained | δk(P) <= 1.5 δk(O), k = 1, 2, 5 | 0.74, 0.84, 1.18 | H |
| R1.4 | Equal fit to the new task | loss spread <= 10% at u2-5 | 1.7-4.0% | H |
| R2.1 | Break survives 1e-4 x 60 | excess >= 10, CI > 0 | 13.1 [9.8, 16.5] | H |
| R3a | Loss persists to u60 | excess >= 20, CI > 0 | 55.2 [50.9, 59.5] | H |
| R3b | More displacement for D on base texts | δ^common_2(D) >= 2 max(O, P), CI > 0 | -0.0073 [-0.0112, -0.0034] | F |
| R4.1 | Teacher break survives 1e-4 x 60 | D-T - O-T >= 10, CI > 0 | 6.9 [4.0, 10.0] | F |
| R5.1 | Broad chat mix breaks D | D - O CI > 0 | 10.3 [7.5, 13.2] | H |
| R5.2 | Broad chat mix breaks D-T | D-T - O-T CI > 0 | 9.3 [6.2, 12.3] | H |
| R5.3 | The broad stage is routine | O and O-T lose <= 5 | -0.3, 0.5 | H |

### revision.md (each part's predictions, written before data; results in its dated entries)

| ID | Registered | Threshold | Observed | Out |
|---|---|---|---|---|
| RL1 | Fast relearning | R_5(D+IT) >= 0.5 | 0.98 [0.94, 1.01] | H |
| RL2 | The ceiling control holds | O+IT within 2 points of its after-stage accuracy at every k | largest change 0.6 (k = 1, 2, 5, 20) | H |
| RP1 | Replay mitigates, own solutions | a reduction of the no-replay excess (13.1) >= 50% | 1.1 [-0.9, 3.3]: 91%, and replay prevents the break | H |
| RS1 | The break survives one realistic epoch | D - O >= 5, CI > 0 | 9.6 [6.9, 12.4] | H |
| RS2 | Fresh solutions stay like once-trained there | P - O within +/-3 | -0.2 [-1.9, 1.6] | H |
| SH1 | Sharpening alone gives less than half the break | O-sharp - O after AO < 73.1 / 2 = 36.5 | 0.9 [-1.6, 3.4] | H |
| SD1 | Seeds: the break survives a redrawn subset (primary) | D - O and D - P CIs > 0, each seed and stage | D - O: s2 61.1 [57.0, 65.2] and 13.1 [10.0, 16.3], s3 48.9 [45.2, 52.5] and 17.8 [14.1, 21.5] (3e-4, 1e-4); lower bounds of D - P 57.0, 9.2, 44.8, 15.2 | H |
| SD3 | Seeds: fresh solutions like once-trained | P - O within +/-3, each seed and stage | 0.0 and 0.8 (s2), 0.0 and -1.4 (s3) | H |

SD2 (secondary: each excess at least 5 points) is with the secondary predictions in section 3.

## 2. The arithmetic task

### depth-profile.md (profile, cap test)

| ID | Registered | Threshold | Observed | Out |
|---|---|---|---|---|
| DP1 | Repetition, not training amount, moves the reasoning | B(S-u230) < B(U-u140), B(F-u140), CIs < 0 | -0.111, -0.165 (CIs < 0) | H |
| DP2 | Separation | every lasting B < every drilled B | -0.0004 < 0.061 | H |
| DP3 | Dose | u20 < u60 < u140 in U and F | monotone | H |
| DP4 | Prediction from handoff | Spearman(B, loss) >= 0.7 | 0.93 (group label alone 0.92) | H |
| CT1 | Competence lost, not slowed | F-u140, U-u140 < 15% within 16k | 11.7, 6.0 | H |
| CT2 | Lasting state gains little from budget | S-u230 gains <= 10 over 4k | +21.6 | F |

### anchor.md

| ID | Registered | Threshold | Observed | Out |
|---|---|---|---|---|
| AN1 | The anchor protects | FA1 >= 40% after B within 16k, if checks pass | 48.2, but the forced-parity check failed (gap 10.7 > 5) | NR |
| AN2 | Dose | FA1 > FA025 > 11.7 after B; B in reverse order | 48.2 > 27.9 > 11.7; 0.161 > 0.024 > 0.009 | H |
| AN3 | Replication | U2 <= 20% and B(U2) >= 0.06 | 6.5; 0.172 | H |
| AN4 | Dilution does not protect | FN <= 20% after B, if checks pass | drilling check failed (36% < 80%); FN 25.3 | NR |

### decision-competence.md

| ID | Registered | Threshold | Observed | Out |
|---|---|---|---|---|
| DC1 | RL teacher's skip rises more slowly than the trace checkpoints' | slower than R-P-u20, u60, u140 | holds against u60 and u140, not u20 | F* |
| DC3 | Forced, trace and RL checkpoints stay well above their own unforced and M0's forced after the flip | all such origins | holds for T-u30, S-u230, R-P@20; fails for R-P-u140 | F* |
| DC4 | Forcing does not rescue solver checkpoints above M0 | R-S <= M0 forced after the flip | never evaluated; R-S never flips (30% skip at u20); forced 58.9 vs 54.9 at u0, then below M0 | NR |
| C-1 | Forced accuracy near its u0 level through u80 for every origin | all origins | T-u30 falls to 21.4 by u80 | F |
| C-2 | Trained checkpoints stay above M0 by about their u0 margin | all | T-u30 below M0 at u80 (-18.0) | F |
| C-3 | Near zero by u480; halving >= 10x the half-skip update | all origins | holds except R-P-u140 (halves at 10, 5.9x) | F* |
| E1 | Coverage gap is a choice of mode | solver's never-solved count falls to near the trace's; M0 close to both | 2 vs 0 (M0 1) | H |
| E2 | Held-out families forced: trained close to M0 | close | 58.9-62.2 vs 54.2 | H |
| C2 | Clone above teacher at u80 in a rerun | > 10 points | 41.9 vs 53.9 | F |
| C3 | Decision clocks repeat | both half-skip updates in 3-6, within 1 of the first run | 4.04, 4.81 | H |

### repetition-fragility.md

| ID | Registered | Threshold | Observed | Out |
|---|---|---|---|---|
| RR1 | U-579 and F-579 at 140 switch off sooner than S; at 20 they behave like S | both arms | holds for U only | F* |
| RR-D | Competence lost at 140, kept at 20 | "most" lost / kept | U -53.3, F -58.6; u20 +/- 2 | H |
| RR-D2 | Same at Stage B 1e-4 | below half of u0 | lose 24 and 25 (about 40%) | F |
| RR-D3 | Same under IT at 3e-4 | below half of u0 | lose 19 and 25 (about 40%) | F |
| RR-D3b | IT: 140-update states lose >= 10 unforced, 20-update states <= 3 | both parts | 20-update states lose 38-42 (premise wrong) | F |
| RR-D4 | Cliff: most of the loss by u2 | F-u140 < 30% at u2 | 45.1% | F |

## 3. Registered items that are not predictions, and unregistered results the paper reports

**Conditions.**
- Math parity: pilot 0.8, 35B 1.8, Nemotron 0.0, teacher 1.2. All within 5.
- M3's parity: 0.8.
- The 35B identity precheck failed (0.0034 against 0.001); a noise-floor measurement replaced it.
- Anchor mechanism checks: FA1 and FA025 failed parity; FN failed the drilling check.
- R1's trajectory check held (within 0.1%).
- **Teacher T2's reading conditions** failed: drilling ratio 0.69 against 0.8; held-out NLL drop 1.2%
  against 20%.
  - They gate only the reading of a null result. T2 failed on its own numbers, so the conditions
    change no count.
  - Count them as conditions (failed), not as predictions. Report them because they bound the
    teacher cell: the student moved little toward the teacher.
- Revision SK: the calibration condition was met (base-7 multiplication with 5 digits: the base solves
  0.2% of 300 items within 2,048 tokens, at most 25%), and the gate passed (O-S-u140 66.1 [63.3, 68.8]
  points above the base, at least 30 with CI > 0; 0.1% of its responses capped, at most 15%).
- Revision SH: the gate passed (B of O-sharp, sampled at temperature 0.5, is 0.082 [0.077, 0.087], at
  least 0.05), so there was no retrain at 0.3.

**Two- or three-way readings** (alternatives fixed in advance; the one that occurred is given):
- M3's "narrow set is the cause" alternative: did not occur.
- N1: "B is not sufficient" branch (FA1 - L = 0.0097 <= 0.02).
- RF2: the failure-to-keep-reasoning branch was not triggered.
- DC2: the clone keeps its teacher's timing, the second branch.
- RR2 and RR3: reversed; unfiltered, not filtered, repetition moves the clock.
- Teacher T3 ("otherwise partial"): did not occur.
- Anchor P1's 20/40 bands: not reached (P1 not interpretable).
- RL, D+IT: R_5 = 0.98, at least 0.75: "suppressed, restored by about 160 examples of reasoning
  training"; the word "lost" is dropped.
- RL, habit control: R_20 = 0.98 (D+IT) and 0.97 (D+AO), at least 0.5: the claim "more than a lost habit"
  is dropped; retraining the reasoning format alone restores the loss.
- RP: replay prevents the break for own solutions (1.1 [-0.9, 3.3]) and for teacher traces (-0.9
  [-2.8, 0.8]; the record's -0.6 [-3.3, 2.0] is the lenient score), so the scope rule applies.
- RS: the prediction held, so the abstract leads with the one-epoch number and the 3e-4 stage becomes the
  stress test.
- SH: "sharpening alone does not break it" (O-sharp within 3 points of O after both stages: -1.8, 0.9).
- SK outcomes, per stage, read as after-stage accuracy (the registered wording) and as extra loss from
  each arm's handoff:
  - IT at 1e-4 x 60 (primary): a break by after-stage accuracy (O-S - D-S 24.2 [20.4, 27.9]) but not by
    extra loss (-4.2 [-9.3, 0.9]); no consolidation either way; "the fix costs nothing" fails (P-S is
    8.2 below the better arm at handoff and 12.1 after).
  - IT at the defaults: the skill is erased in every arm (all below 1%), so consolidation by after-stage
    accuracy is met trivially (0.3 [0.0, 0.8]); no break either way; "the fix costs nothing" fails at
    handoff (-8.2).

**Descriptive, with no threshold.**
- Spearman of B with the loss: MP4 (1.0 / 0.4), M1.4 (-0.1 / 0.7), M5.4 (0.7 / 0.9).
- R2's secondary: 12.6 at u20.
- SK: D-S - O-S at handoff, -28.3 [-32.3, -24.5] (registered as descriptive).

**Expectations marked "not a new hypothesis".** Both 16k consistency expectations held.

**Secondary predictions** (counted separately):

| ID | Threshold | Observed | Out |
|---|---|---|---|
| DP1-sec | S-u230 below U-u60, F-u60 | -0.019 | H |
| DP5 | B' orders states as B for DP1-DP3 | agrees | H (not in the paper) |
| RF1 answer rule | E/2 | 26.5, 58.0 | H |
| R5.1-sec | at least 10 | 10.3 | H |
| R5.2-sec | at least 10 | 9.3 | F |
| SD2 | seeds: D - O and D - P at least 5 points, each seed and stage | smallest 12.3 (D - P, s2, 1e-4) | H |

**Unregistered or exploratory results in the paper.** [redacted: editing note]
- "Nor is it memorization" (90.7% against 85.1%). Not registered.
- The 35B base collapse under IT (the record's "Not predicted"). It is
  confounded by the fresh adapter; the paper says so.
- The own-text overshoot below the base (88% of the gain gone by u2; below the base from u5). An independent check, post hoc. [redacted: editing note]
- The movement toward the base in both settings. Labelled "post hoc".
- The Stage-B loss jump and its r = -0.84. Observed, not registered.
- The anchor section is labelled exploratory.
- Failure forms (medians, loops, unboxed), AIME and strict-score comparisons: descriptive.
- The teacher held-out result "D-T finds held-out traces harder than the base":
  a by-product of a registered check, itself descriptive.
- SK: P-S - D-S at handoff, 20.2 [16.2, 24.1]. Post hoc, descriptive.

## 4. Counts

| | Registered | Held | Failed (of which partial) | Not readable |
|---|---|---|---|---|
| Mathematics (math pilot, main phase, re-forcing, teacher, robustness, revision) | 45 | 39 | 6 (3) | 0 |
| Arithmetic task (depth profile, anchor, decision-competence, repetition-fragility) | 26 | 11 | 12 (4) | 3 |
| **All** | **71** | **50** | **18 (7)** | **3** |

- **The 18 failures:**
  - mathematics: M1.5, M5.5, T2, T5, R3b, R4.1;
  - arithmetic: CT2, DC1*, DC3*, C-1, C-2, C-3*, C2, RR1*, RR-D2, RR-D3, RR-D3b, RR-D4.
  - (* partial.) In mathematics, M1.5, M5.5 and T5 are the partial failures.
- **If T2 is treated as the rival alternative and not counted:** 70 registered, 17 failed.
- **The 3 not readable:** AN1 and AN4 (a registered check failed) and DC4 (never evaluated).
- **Secondary:** 6 registered, 5 held, 1 failed (R5.2-sec).
- **Not counted:** 15 conditions (parity x4, M3 parity, 35B precheck, anchor checks x3, R1
  trajectory, T2 conditions x2, SK calibration and gate, SH gate); 16 readings (8, and RL x2, RP, RS, SH
  and SK x3 from the revision); 5 descriptive; 2 expectations.

## 5. Notes on a draft of the paper

[redacted: editing notes on a draft of the paper]

## 6. Proposed appendix table rows ("Registered predictions and outcomes")

Columns: Record | ID | Prediction | Threshold | Observed | Outcome.

```
Math pilot | MP1 | Drilling over-sharpens | B(D)-B(O) >= 5% of NLL | 24% | holds
Math pilot | MP2 | Dose | B rises with passes | 0.0007 < 0.016 < 0.111 | holds
Math pilot | MP3 | Drilled loses more | > D-u20 and O, CIs > 0 | 36.2, 73.1 over O | holds
Math pilot | MP5 | Free falls more than forced | base and O, both stages | all four | holds
Main | M3.1 | Fresh texts keep B low | B(P) <= B(D)/2 | 0.35% | holds
Main | M3.2 | Repeated text is the cause | D-P >= 20, CI > 0 | 36.4, 72.5 | holds
Main 35B | M1.1-3 | Pilot P1-P3 at 35B | as pilot | 32%; dose; 38.3, 72.1 | hold
Main 35B | M1.5 | Free falls more than forced | as pilot | base under IT | fails
Main 35B | M1.6 | Half of B at kept-top | >= 50% | 57% | holds
Main 35B | M1.7 | AIME: D loses more after IT | direction | 60.6 vs 1.3 | holds
Main Nemotron | M5.1-3 | Pilot P1-P3 on Nemotron | as pilot | 18%; dose; 59.2, 40.0 | hold
Main Nemotron | M5.5 | Free falls more than forced | as pilot | start model under B | fails
Main Nemotron | M5.6-7 | Kept-top >= half; AIME | as M1 | 61%; 68.3 vs 10.2 | hold
Main | M2.1 | Second subset breaks | >= 10, CI > 0 | 12.6, 88.6 | holds
Main | M2.2 | Second subset over-sharpens | >= 5% of NLL | 22% | holds
Main | N2 | Rank 128 breaks | >= 1/2 of rank 32 | 1.08x, 1.14x | holds
Re-forcing | RF1 | Competence is lost | E_b >= E/2, CI > 0 | 21.3, 68.3 | holds
Teacher | T1 | Drilled traces break | >= 10 over O-T and D-T-u20 | 59.6, 92.4 | holds
Teacher | T2 | Rival: consolidation | < 3 | 59.6, 92.4 | fails
Teacher | T4 | Fresh traces fix it | P-T - O-T <= 5 | -1.0, -0.8 | holds
Teacher | T5 | Narrowing out of sample | own >= 10x; base within 0.01 | 140x; -0.026 | fails (second part)
Robustness | R1.1 | D moves more on own texts | >= 2x max, CI > 0 | 0.066 vs 0.001 | holds
Robustness | R1.2 | Displacement before collapse | D u2 > 77.4 | 81.8 | holds
Robustness | R1.3 | P like O | <= 1.5x | 0.74-1.18 | holds
Robustness | R1.4 | Equal fit | spread <= 10% | 1.7-4.0% | holds
Robustness | R2.1 | Survives 1e-4 x 60 | >= 10, CI > 0 | 13.1 | holds
Robustness | R3a | Persists to 60 updates | >= 20, CI > 0 | 55.2 | holds
Robustness | R3b | D moves more on base texts | >= 2x max, CI > 0 | -0.007 | fails
Robustness | R4.1 | Teacher break at 1e-4 x 60 | >= 10, CI > 0 | 6.9 [4.0, 10.0] | fails
Robustness | R5.1 | Broad mix breaks D | CI > 0 | 10.3 | holds
Robustness | R5.2 | Broad mix breaks D-T | CI > 0 | 9.3 | holds
Robustness | R5.3 | Broad mix is routine | O, O-T lose <= 5 | -0.3, 0.5 | holds
Revision | RL1 | Relearning is fast | R_5(D+IT) >= 0.5 | 0.98 | holds
Revision | RL2 | Ceiling control | O+IT within 2 at every k | at most 0.6 | holds
Revision | RP1 | Replay mitigates | reduction >= 50% | 91% (prevents) | holds
Revision | RS1 | Survives one epoch | D-O >= 5, CI > 0 | 9.6 [6.9, 12.4] | holds
Revision | RS2 | Fresh like once | P-O within 3 | -0.2 | holds
Revision | SH1 | Sharpening alone: under half | O-sharp - O < 36.5 after AO | 0.9 | holds
Revision | SD1 | Survives redrawn subsets | D-O, D-P CIs > 0 | 61.1, 13.1; 48.9, 17.8 | holds
Revision | SD3 | Fresh like once, seeds | P-O within 3 | 0.0, 0.8; 0.0, -1.4 | holds
Depth profile | DP1 | Repetition, not updates | S < U, F, CIs < 0 | -0.111, -0.165 | holds
Depth profile | DP2 | Separation | lasting < drilled | -0.0004 < 0.061 | holds
Depth profile | DP3 | Dose | monotone | monotone | holds
Depth profile | DP4 | B ranks the loss | Spearman >= 0.7 | 0.93 | holds
Cap test | CT1 | Lost, not slowed | < 15% within 16k | 11.7, 6.0 | holds
Cap test | CT2 | Budget adds little for S | <= +10 | +21.6 | fails
Anchor | AN1 | Anchor protects | FA1 >= 40% | 48.2; parity check failed | not readable
Anchor | AN2 | Dose | FA1 > FA025 > F | 48.2 > 27.9 > 11.7 | holds
Anchor | AN3 | Replication | U2 <= 20%, B >= 0.06 | 6.5, 0.172 | holds
Anchor | AN4 | Dilution does not protect | FN <= 20% | drilling check failed | not readable
Decision | DC1 | RL skips later than traces | vs R-P-u20/u60/u140 | not vs u20 | fails (in part)
Decision | DC3 | Forcing restores competence | all origins | not R-P-u140 | fails (in part)
Decision | DC4 | Forcing does not rescue the solver | <= M0 after flip | not evaluated | not read
Decision | C-1 | Forced level holds to u80 | all origins | T-u30 21.4 | fails
Decision | C-2 | Margin over M0 kept | all | T-u30 -18.0 | fails
Decision | C-3 | Halving >= 10x half-skip | all origins | R-P-u140 5.9x | fails (in part)
Decision | E1 | Coverage gap is a mode | never-solved near | 2 vs 0 | holds
Decision | E2 | Held-out forced near M0 | close | 58.9-62.2 vs 54.2 | holds
Decision | C2 | Clone > teacher at u80 (rerun) | > 10 | 41.9 vs 53.9 | fails
Decision | C3 | Clocks repeat | within 1 update | 4.04, 4.81 | holds
Repetition | RR1 | Drilled switch off sooner | U and F | U only | fails (in part)
Repetition | RR-D | Competence lost at 7.7 passes | most lost; 1.1 kept | -53, -59; +/-2 | holds
Repetition | RR-D2 | Same at 1e-4 | below half | -24, -25 (40%) | fails
Repetition | RR-D3 | Same under IT | below half | -19, -25 (40%) | fails
Repetition | RR-D3b | Unforced shows it under IT | >= 10 vs <= 3 | 1.1-pass states -38 to -42 | fails
Repetition | RR-D4 | A cliff by update 2 | < 30% | 45.1% | fails
```

Footer: "71 registered predictions (45 on mathematics, 26 on the arithmetic task): 50 held, 18 failed
(7 in part), 3 could not be read. Registered conditions, alternative readings, descriptive items and
secondary predictions are listed in the records."
