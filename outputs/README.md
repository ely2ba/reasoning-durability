# outputs/

`src/collect.py --runs <DuraSeed-v1>/runs` builds these compact tables from the raw runs. They hold
no text, and every number in the paper is computed from them. The format is gzip CSV. Booleans are
0/1, and an empty cell means the column does not apply. The per-sample texts are in the
Hugging Face dataset (data/ARCHIVE.md). `summaries/` holds what the analysis scripts write.

## tces_samples.csv.gz: one row per TCES probe sample

- `run`: probe run, a directory of runs/decision-probe, runs/repetition-probe or runs/anchor. The aborted dc-20260923 is left out.
- `label`: state probed (M0, T-u30, S-u230, RP47-u140, U-u140, F-u140, FA1-u140, …).
- `update`: later-stage update at which the state was probed; 0 is the handoff.
- `kind`: `decision` (an 8-token reply), `think` (forced `\n<think>\n`), `competence` (forced "Let's think step by step.") or `unforced`.
- `cap`: the run's forced and unforced token budget. It is 16,384 for the cap-, cap16k- and anchor runs, else 4,096 (checked: every sample stopped by length sits at the cap). Decision replies are 8 tokens.
- `task_id`: TCES item (data/tces/panel.jsonl, or the E1 panel of dc-panels-20260923).
- `role`: `targeted` or `sentinel` (held out).
- `draw`: draw index for the item.
- `tokens`: sampled tokens, after any resumption.
- `stop`: `stop` (a stop sequence) or `length` (the budget).
- `skip`: decision rows only: the reply opens with an answer tag, so it skips reasoning.
- `correct`: the exact verifier accepts the answer once restated templates are removed (the resumed score).
- `strict_correct`: correct with no resumption.
- `resumed`: times a stop on a restated answer template was resumed.
- `found`: think, competence and unforced rows on panel items: a verifier-accepted expression appears anywhere in the text (`common.tces.found`).

## math_samples.csv.gz: one row per math sample

- `run`: `pilot-20260924`, `main-9b-20260924`, `main-35b-20260924`, `main-nemotron-20260924`,
  `teacher-20260924` (the D-T, O-T and P-T students of gpt-oss-120b's traces), `robustness-20260924` (R1-R3),
  `robustness-followup-20260924` (R4 and R5), or a revision part (records/revision.md): `revision-rp-20260926`
  (RP), `revision-rs-20260926` (RS), `revision-rl-20260926` (RL), `revision-sh-20260926` (SH), or
  `revision-sd-20260926/s2` and `revision-sd-20260926/s3` (SD: two more runs of the 9B design, each with a
  redrawn drilled subset, a new order of O's texts and a new initialization).
- `bench`: `math500` (the 221 problems at levels 3–5) or `aime` (59).
- `label`: state evaluated. `+it` and `+b` mean after 20 updates of instruction tuning or Stage B. The robustness
  run's `+it@k` is after *k* updates of instruction tuning, `+it-lr1e-4@k` at learning rate 1e-4 on 1,920 distinct
  examples (R2; R4 from the teacher states), `+it-long@k` at 3e-4 on the same examples, and `+chat-lr1e-4@60` after
  60 updates at 1e-4 on R5's 1,920 broad chat rows (data/chat_ids.json). The revision's: `+it-replay` is R2's
  stage with 2 of each batch of 32 replayed from the arm's own training texts (RP); `+nr-epoch` one pass over
  8,544 No Robots rows at 1e-4 (RS); `+relearn` and `+habit` 1e-4 training of an after-stage state on the base's
  own solutions to 640 new pool problems, or its responses to 640 chat prompts (RL; data/revision_ids.json);
  `O-sharp-t05-u140` O trained on the base's solutions sampled at temperature 0.5 (SH); in SD's runs,
  `+it-lr1e-4` (at `u60`) is R2's stage, 60 updates at 1e-4.
- `when`: `u0` (handoff), or `uk` after *k* updates of the later stage (`u20` for the standard stages).
- `mode`: `forced` (reasoning prefilled) or `unforced`.
- `rule`: `first` (the evaluation's own sample), or `budget` or `answer` (the re-forcing probe's continued sample).
- `task_id`: MATH-500 `unique_id` or AIME id (data/math/).
- `level`: MATH level (3–5), or `aime2025` or `aime2026`.
- `draw`: draw index.
- `tokens`: sampled tokens; for re-forced rows, after the continuations.
- `capped`: reached the cap (16,384 for MATH-500, 32,768 for AIME).
- `end`: `cap`, `eos` (the family's end token: Qwen 248044, Nemotron 11 or 2), `user` (the text ends with a new `\nUser:` turn) or `other`.
- `boxed`: the text contains `\boxed`.
- `correct`: the last `\boxed{}` parses to the reference answer.
- `tagged`: the integer inside `<answer>…</answer>` (Stage B's format), if any.
- `think`: the sample opens a `<think>` block. Forced rows are always 1.
- `closed_think`: the text contains `</think>`.
- `within_2048`, `within_4096`, `within_8192`, `within_16384`: correct when the sample is cut at *k* tokens. Empty when *k* is not below the cap (so `within_16384` is filled for AIME only).
- `lenient_correct`: correct, or no `\boxed` at all and the text's last integer (thousands commas removed) is the answer.
- `last_integer_correct`: the text's last integer is the answer (thousands commas allowed; none if it runs to 100
  characters): the revision's secondary, lenient score.
- `looping`: capped samples only: the last 2,000 tokens hold fewer distinct 16-grams than half the windows.
- `continuations`: re-forcing rows: how many times the sample was continued.
- `first_correct`: re-forcing rows: whether the original sample was correct.

## depth_texts.csv.gz: one row per scored text and pair of models

- `run`: depth-20260923, depth-full-20260924, anchor-20260924 or a math run.
- `length`: scored response tokens: 1,024, 4,096, 2,048 (math), or the drilled training texts' 4,231 (TCES) and 17,000 (math).
- `set`: `common` (M0's own forced texts), `own` (the state's own texts), `drilled` (the drilled training texts),
  `heldout` (the teacher run's 200 held-out teacher traces), or, for the robustness and teacher runs, `displacement` (a state's own handoff
  texts) and `displacement_common` (M0's texts), each scored at handoff and again after *k* later-stage updates.
- `base`: the reference model: M0, or for the two displacement sets the state at handoff.
- `state`: the state compared with it (`label@uk`: after *k* updates). `M0` is the base against itself; the 35B run's
  `M0-second` is its noise floor.
- `task_id`: the text's item or sample id.
- `n`: body positions scored, after the forced prefix (3 tokens for Qwen, 2 for Nemotron).
- `d`: Σ (log p_base − log p_state) of the text's tokens.
- `flips`: positions where the state's most likely token differs from the base's.
- `d_kept`: Σ d where the state keeps the base's most likely token.
- `gain_kept`: Σ (p_state(top) − p_base(top)) where the top token is kept.
- `branch`: positions where the text took one of the base's alternatives (its token is not the base's top token).
- `branch_kept`: branch positions where the top token is kept.
- `d_branch_kept`: Σ d over branch positions where the top token is kept.
- `top_p_branch_base`: Σ p_base(top) at branch positions.
- `top_p_branch_state`: Σ p_state(top) at branch positions.
- `base_lp`: Σ log p_base of the text's tokens.
- `sure_nll`: Σ −log p_base(top) where p_base(top) ≥ 0.95.
- `state_hits`: positions where the state's most likely token is the text's token.
- `d_4_7`, `n_4_7`, …, `d_512_1024`, `n_512_1024`: depth-20260923's common texts only: Σ d and the positions scored
  within each bin of 1-indexed response positions (4–7, 8–15, …, 512–1,024), for B's position profile.
- `n_prefix`, `base_lp_prefix`, `d_prefix`: drilled texts only: the positions of the forced prefix, which every
  other column leaves out, and Σ log p_base and Σ d over them. The registered fit checks on the drilled texts
  (main-phase.md N2, anchor.md) average the NLL over every scored token, prefix included (`common.divergence.text_nll`).

`common.divergence.summary` turns these per-text sums into B (Σd/Σn), A (Σflips/Σn) and the other statistics.

## train_log.csv.gz: one row per training update

- `run`: training run (TCES acquisition, TCES later-stage probes, anchor, or a math run).
- `label`: arm or state trained.
- `stage`: `acquisition` (stage_a/ or train/), `later` (u*/train.json or later/), or the robustness runs' parts `r1`
  to `r5` (r*/<label>/train/); the teacher run's `r1` rows are its probe's two replayed updates.
- `update`: update number.
- `train_tokens`: tokens in the update's batch.
- `loss_sum`: Tinker's summed token loss (`loss:sum`).
- `batch`: examples per batch (anchor arms only).

## math_rft.csv.gz: one row per screened pool problem

- `run`: pilot-20260924, main-35b-20260924, main-nemotron-20260924 or revision-rl-20260926 (RL's relearning data:
  the pool after the pilot's 7,680 screened problems, less those an arm trained on; its first 640 correct).
- `task_id`: pool problem (data/math/pool_ids.jsonl).
- `easy`: solved within 256 tokens after an empty think block, so screened out and not sampled.
- `tokens`: length of the forced sample.
- `capped`: the sample reached the 16,384-token cap.
- `correct`: correct and uncapped, so eligible as training data.

## math_p_samples.csv.gz: the main-9b P arm's samples

- `task_id`: one of the D arm's 579 problems.
- `distinct`: distinct correct, uncapped forced samples kept (up to 8, the pilot's D sample first).
- `draws`: sampling draws made (at most 24).

## skill_samples.csv.gz: one row per sample of the revision's skill part (SK)

- `run`: skill-20260926 (runs/skill).
- `set`: `smoke` (the base on the first 5 calibration items of each setting), `calibration` (the base on 300
  calibration items) or `eval` (every state on the 300 held-out items of the chosen setting).
- `setting`: `base7-5` or `base7-6` (multiplying two base-7 numbers of 5 or 6 digits), `words-12` or `words-16`
  (sorting words under a custom alphabet).
- `label`: state evaluated: `M0`, `D-S-u140`, `O-S-u140` or `P-S-u140`, and `+it-lr1e-4` (instruction tuning at
  1e-4 for 60 updates, the primary stage) or `+it` (at the defaults, 3e-4 for 20) after them.
- `cap`: the token budget: 4,096 or 8,192 (the smoke), then 2,048 (checked: a sample is capped exactly when it
  reaches it). A re-run with src/run_skill.py runs its smoke at 2,048 too (smoke-cap2048/).
- `task_id`, `draw`, `tokens`, `capped`, `boxed`, `correct` (the exact verifier accepts the answer), as above.
