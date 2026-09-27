# Compute behind the paper

Compiled 2026-09-24, 18:00 UTC, from each run's own ledger (`billing.json` in the original runs tree,
which is not public), for the paper's statement of compute. Amounts are US dollars at Tinker list
prices, the prices each ledger used. Totals are sums of unrounded values. The revision parts
(`revision.md`) were added on 2026-09-26, 03:49 UTC, from their ledgers.

**Total.**
- The runs the paper reports cost **$2,437.28**: $1,845.11 before the revision and $592.17 for its
  parts. This is the total the paper states.
- With calibrations ($4.54) and aborted launches ($16.66), all work done for the paper comes to
  $2,458.48.
- The earlier studies whose states and results the paper reuses add $638.93 (token cost only), for
  $3,097.41 in all.

## Runs the paper reports

| Experiment | Record | Spend | Ledgers (run folders) |
|---|---|---|---|
| Math pilot, with its prechecks | `math-pilot.md` | $230.55 | `math/pilot-20260924` |
| Main phase, 9B: P, D2, rank 128 | `main-phase.md` | $151.72 | `math/main-9b-20260924` |
| Main phase, 35B, with its noise floor ($0.17) | `main-phase.md` | $149.82 | `math/main-35b-20260924` |
| Main phase, Nemotron | `main-phase.md` | $160.14 | `math/main-nemotron-20260924` |
| Re-forcing, with its stopped first launch ($20.78) | `reforcing.md` | $68.47 | `math/pilot-20260924/reforce` |
| Teacher traces from gpt-oss-120b, at assumed prices, with a stopped first launch ($10.18) | `teacher.md` | $53.77 | `math/teacher-20260924/teacher` |
| Teacher student phase ($156.67) and its replayed probe ($1.37) | `teacher.md` | $158.05 | `math/teacher-20260924` |
| Robustness R1–R3 | `robustness.md` | $64.26 | `math/robustness-20260924` |
| Robustness R4 and R5 | `robustness.md` | $42.71 | `math/robustness-followup-20260924` |
| **Mathematics** | | **$1,079.50** | |
| Decision or competence: A–C, E, C2, C3 | `decision-competence.md` | $330.90 | `decision-probe/dc-20260923b`, `dc-long-20260923`, `dc-panels-20260923`, `dc-long80-rep`, `dc-long80-rep-T`, `dc-decision-rep` |
| Repetition and fragility: A, D, D2, D3, D3b, D4 | `repetition-fragility.md` | $148.29 | `repetition-probe/rep-20260923` and its `-forced`, `-forced60`, `-gentle`, `-gentle-F140`, `-instruct`, `-instruct-unforced` and `-cliff2` runs |
| Cap test and 16k consistency probes | `depth-profile.md` | $64.25 | `repetition-probe/cap-20260924`, `cap16k-u0-20260924`, `cap16k-stageb-20260924` |
| Depth profile (1,024 tokens), and N1 at full length with its extension | `depth-profile.md`, `main-phase.md` | $9.10 | `depth-profile/depth-20260923`, `depth-full-20260924` |
| Anchor: FA1, FA025, FN, U2 | `anchor.md` | $213.07 | `anchor/anchor-20260924` |
| **Arithmetic task (TCES)** | | **$765.60** | |
| Revision RL: relearning, with its habit control | `revision.md` | $157.74 | `math/revision-rl-20260926` |
| Revision RP: replay during the later stage | `revision.md` | $27.36 | `math/revision-rp-20260926` |
| Revision RS: one epoch of instructions at the reasoning rate | `revision.md` | $34.75 | `math/revision-rs-20260926` |
| Revision SH: sharpening without repetition | `revision.md` | $58.24 | `math/revision-sh-20260926` |
| Revision SD: two more runs, seed s2 | `revision.md` | $132.96 | `math/revision-sd-20260926/s2` |
| Revision SD: seed s3 | `revision.md` | $129.90 | `math/revision-sd-20260926/s3` |
| Revision SK: a skill the base lacks, with its smoke and calibration | `revision.md` | $51.23 | `skill/skill-20260926` |
| **Revision** | | **$592.17** | |
| **Runs the paper reports** | | **$2,437.28** | |

The subtotals are sums of the unrounded ledger values ($1,079.5015 + $765.6048 + $592.1740 =
$2,437.2803), so the rounded rows can differ from them by a cent: the mathematics rows add to
$1,079.49, the arithmetic rows to $765.61 and the revision rows to $592.18.

These totals include parts that produced no reported result: re-forcing's first launch ($20.78), the
teacher phase's first launch ($10.18), the student phase's evaluations in flight at its 12:36 UTC
stop, the C2 run whose T-u30 cell stopped at its limit ($9.10; its S-u230 half is reported), and
E1's M0 cell, relaunched inside `dc-panels-20260923`.

## Listed separately

| Item | Spend | Where stated |
|---|---|---|
| Math pilot: exploration and calibration | $3.38 | the pilot's budget note (redacted in `math-pilot.md` with an account balance); its calibration ($1.85) and AIME calibration ($1.48) are in `math-pilot.md` |
| 35B calibration | $0.76 | the main-phase launch note (redacted in `main-phase.md` with an account balance) |
| Two teacher calibrations, at assumed prices | $0.40 | `teacher.md`, launch of the teacher phase |
| **Calibrations** | **$4.54** | |
| Aborted launch: decision or competence, `dc-20260923` | $13.72 | its ledger; `decision-competence.md`, Deviations |
| Aborted launch: repetition and fragility D4, first launch, no samples (`rep-20260923-cliff`) | $2.94 | its ledger; `repetition-fragility.md`, D4 |
| **Aborted launches** | **$16.66** | |

## Earlier studies the paper reuses

Token cost at the 9B's list prices ($0.66, $1.995 and $1.463 per million prefill, sampled and trained
tokens). The storage reservations in these studies' ledgers ($63 in all) are
bounds on reserved storage, not token charges, and are left out.

| Study | Spend | What the paper reuses |
|---|---|---|
| Endpoint clone (`endpoint-clone.md`) | $93.57 | the RL teacher T, the clone S and the clone corpus |
| Replay-v1, original | $236.36 | the matched pair's acquisition |
| Replay-v1, continuation | $105.13 | R-P@20 and R-S@220 (the original study, reinterpreted in the appendix) |
| Replay-v1, order-47 replication | $203.87 | R-P-u140 and R-S-u200 |
| **Total** | **$638.93** | |

## Not included

- Replay-v1 runs the paper does not use: dense retention ($51.78) and the u10 recheck ($4.44).
- The original study's earlier work that produced M0 and the task panels (August 2026). Its ledgers
  are marked "reconciliation required", so it has no reliable total.
- Other studies in the original repository that the paper does not use.

## Prices and checks

- The teacher phase's prices are assumed: $0.70 and $2.00 per million prefill and sampled tokens for
  gpt-oss-120b, as its ledger used. The true list price may be lower; the total moves by at most $54.
- Where a record states a spend, the ledgers match it: `repetition-fragility.md` ($142.55 over six
  ledgers), `reforcing.md` ($68.47), `teacher.md` ($156.67 and $1.37), `robustness.md` ($44.53 and
  $19.73; $42.71), `main-phase.md` ($151.72, $149.65 and $160.14), `anchor.md` ($213.07) and `revision.md` (RL $157.74,
  RP $27.36, RS $34.75, SH $58.24, SK $51.23 and SD $262.86; its total of $592.18 adds the rounded parts).
