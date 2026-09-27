# records/

Each record is an experiment's pre-registration and lab notebook, in the order it was written: the
design and predictions, fixed before the experiment's data existed; every deviation, with its time;
the results; and the corrections made after independent recomputation. A result is read against the
predictions of its own record. The last two files below are compiled from the records and from the
runs' ledgers. Times are UTC.

| Record | Experiment |
|---|---|
| `endpoint-clone.md` | the RL teacher and its behavioural clone on the arithmetic task (TCES) |
| `decision-competence.md` | what a later stage removes: the decision to reason, or the competence |
| `repetition-fragility.md` | whether repeating the clone's samples makes the skill fragile |
| `depth-profile.md` | the divergence B at handoff, the dose and the cap test |
| `anchor.md` | whether keeping the base's reasoning in reach protects drilled competence |
| `math-pilot.md` | drilling a base model's own math solutions (9B) |
| `main-phase.md` | fresh solutions, a second subset, rank 128, the 35B and Nemotron replications |
| `reforcing.md` | continuing the drilled model's forced searches |
| `teacher.md` | the design on a stronger model's (gpt-oss-120b) traces |
| `robustness.md` | timing, step size, longer training, the teacher arms at a smaller step, broad chat data |
| `revision.md` | after review: relearning, replay, a realistic later stage, sharpening without repetition, seeds, a skill the base lacks |
| `predictions-ledger.md` | every registered prediction and its outcome, with the counts the paper reports |
| `compute.md` | the list-price compute of every run the paper reports, from the runs' own ledgers |

**Code.** The records name the scripts that ran (`tools/*.py`) and their SHA-256 hashes. Those are
the original DuraSeed-v1 code. `src/` reimplements it, and `tests/test_runners.py` checks that the
reimplementation builds the same prompts, training rows and datums, token for token, as the code that
produced the runs.

**Redactions.** The records are verbatim except for three kinds of text, which were not scientific
content:
- the research account's balances, caps and reserve, including the `--prior` launch argument (a
  balance), and the author's private messages: removed, each marked `[redacted: …]`;
- internal role names of reviewers: replaced by neutral words ("an independent audit", "a review",
  "the author");
- paths to internal scratch files, and one link to an unreleased status file: removed or neutralized.
Per-run costs and limits, designs, predictions, timestamps, deviations, results and code hashes are
unchanged. The ledger also drops its citations of a draft of the paper by line number, and its notes
on that draft.

**Intervals.** Bootstrap intervals in the paper come from this repository's analysis, which draws a
fresh stream (seed 20260924) for each comparison, so an interval end can differ from the one in a
record by about 0.1 points; point estimates are identical.
