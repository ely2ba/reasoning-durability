# data/

These are the frozen inputs of the paper's experiments. Third-party datasets are stored by id
only; their texts are rebuilt from the public sources. The TCES and MAPS data are the project's
own synthetic tasks and are stored in full; so are the ids of SK's problems, which `src/common/skill`
rebuilds from its seed. `SHA256SUMS` covers every file here.

| File | What it is | Source | Licence |
|---|---|---|---|
| `tces/panel.jsonl` | The 384 TCES evaluation items: 192 targeted and 192 held-out (sentinel). Fields: `task_id`, `role`, `prompt`, `prompt_token_ids`, `task` (see below). | a_monitor manifest of the Pilot-0 source run (seed 11), read through DuraSeed-v1 `read_source` | ours |
| `tces/maps_stage_b.jsonl` | The 4,096 Stage-B records (answer-only program synthesis), in training order: `index`, `task_id`, `prompt_text`, `completion_text` = `<answer>shortest program</answer>`. Update *u* trains on records ((*u*−1)·32 + *j*) mod 4,096, for *j* = 0..31. | DuraSeed-v1 `stage_b_sources`; manifest `sha256:45edb9cb…` | ours |
| `tces/clone_prompts.jsonl` | The 800 prompts of the clone corpus: `task_id`, corpus `group`, `prompt_token_ids`. | `runs/endpoint-clone/endpoint-clone-20260910T162817Z/acquisition/corpus` | ours |
| `tces/subsets.json` | Clone-corpus sample ids, in training order, for U and F (namespace `repetition-20260923`) and U2 (`anchor-20260924-u2`). Also the 4,480 anchor items as (task_id, draw). FA1, FA025 and FN drill F. | Recomputed, and equal to the run manifests | ours |
| `tces/common_texts.jsonl` | The 192 texts B is scored on: M0's draw-0 forced texts on 96 targeted items (dc-20260923b) and 96 sentinel items (dc-panels-20260923, E2), chosen by SHA-256 of `depth-20260923:<task_id>`. Fields: `task_id`, `role`, `prompt` (prompt token ids), `response` (the prefix [198, 248068, 198], then the re-encoded text, cut at 4,096 tokens). Rows are in scoring order. | DuraSeed-v1 `run_depth_profile.chosen` and `texts`. Equal token for token to depth-full-20260924's M0 scores, and to depth-20260923's over the first 1,024. | ours |
| `math/pool_ids.jsonl` | The 36,447 training problems in pool order: `task_id` (SHA-256 of the lowercased, whitespace-collapsed problem), `source_row_id` (the Tulu-3 row whose solution gave the answer), `answer`. | Tulu-3 SFT mixture, `ai2-adapt-dev/numinamath_tir_math_decontaminated` | NuminaMath-TIR: Apache 2.0; mixture: ODC-BY-1.0 |
| `math/pool.sha256` | SHA-256 of the full pool file with problem texts (`runs/math/data/pool.jsonl`). A rebuild must match it. | | |
| `math/pool_stats.json` | How the pool was built: source rows, distinct problems, the problems dropped at each filter (non-integer or several answers, disagreeing duplicates, contamination, figures, prompts over 512 tokens, guessable), and the reference-answer check against code output on 200 rows. | The pool build, 2026-09-24 | |
| `math/eval_math500_ids.json` | The 221 MATH-500 problems with integer answers at levels 3–5: `unique_id`, `level`, `answer`. | HuggingFaceH4/MATH-500 | MATH (Hendrycks et al., 2021): MIT |
| `math/aime_ids.json` | 59 AIME problems: all 30 of 2025, and 2026 without problem 1, which appears word for word in NuminaMath-TIR. `id`, `answer`. | MathArena/aime_2025, MathArena/aime_2026 | CC BY-NC-SA 4.0: ids and answers only |
| `no_robots_ids.json` | The 640 instruction-tuning ids (Part D3) and the 4,480 ids of the anchor's FN control, which excludes D3. Selection rule inside. The FN list reproduces the FN arm's logged training tokens at all 140 updates. | Tulu-3 SFT mixture, `ai2-adapt-dev/no_robots_converted` | No Robots: CC BY-NC 4.0, ids only |
| `chat_ids.json` | R5's 1,920 broad chat rows in training order: `id`, `source` and `mathlike` (the registered keyword pattern matches the row). The mixture's non-math sources; selection rule and pattern inside. Equal to the run's `r5/examples.json` and reproduces R4/R5's logged training tokens. | Tulu-3 SFT mixture, 12 sources | ODC-BY-1.0 (mixture); each source's own licence; ids only |
| `revision_ids.json` | The training data of the revision parts (records/revision.md), in training order: RL's 640 relearning problems (`rl_relearn`, pool `task_id`s) and 640 habit prompts (`rl_habit`, R5's chat rows), RS's 8,544 No Robots rows (`rs_no_robots`, the D3 selection continued), SH's 4,453 problems (`sh_sharp`, O's problems with a solution at temperature 0.5), and SD's drilled subsets (`sd_s2`, `sd_s3`: 579 of the pilot's O problems each, in D's order, with each run's LoRA seed). Also the 52 pool problems RL's screen skips because an arm of the paper trained on them (`rl_relearn.skipped`, all Nemotron's O arm), which `src/run_revision.py` reads. Selection rules inside. Equal to the runs' `rft.jsonl` and `habit.jsonl` (RL), `examples.json` (RS), `sharp-t05.jsonl` (SH) and `arms.json` (SD). | Tulu-3 SFT mixture: `ai2-adapt-dev/numinamath_tir_math_decontaminated`, `no_robots_converted` and R5's sources | NuminaMath-TIR: Apache 2.0; No Robots: CC BY-NC 4.0; mixture: ODC-BY-1.0; ids only |
| `skill_ids.json` | SK's problems (records/revision.md), in order: the 300 calibration problems of each setting of the ladder (base-7 products of 5 and 6 digits, 12 and 16 words under a custom alphabet), and base7-5's 300 evaluation and 4,480 train problems. `src/common/skill` rebuilds them from its seed (a test checks it); a base-7 id spells out its two numbers. Equal to the run's smoke, calibration and evaluation files and its arms.json. | `src/common/skill`; runs/skill/skill-20260926 | ours |
| `checkpoints.json` | The Tinker states the paper uses: label, origin run, creation time and whether it is kept. The 53 kept, stored without expiry and available on request, are the base M0, every trained model at handoff (the 9B drilled, once-trained, fresh-solution and sharpened models of every run, the second drilled set, rank 128, the 35B and Nemotron models, the students of gpt-oss-120b's traces and SK's arms) and the arithmetic task's states. The revision's later-stage states are listed but not kept; they are re-trainable from the kept states. A runner reads a state from a state file in its run folder (README.md). | Run records; the creation time is the state file's | |
| `ARCHIVE.md` | The model outputs (probe and evaluation texts, training texts, depth arrays), published as a Hugging Face dataset: what it holds, how to download it and what is withheld. | | |

**The `task` field in `panel.jsonl`.** It has this form:
`{"operands": [ints], "target": [numerator, denominator], "allowed_ops": [...], "constraints": {...}}`.
`constraints` lists only fields that differ from the defaults:
- `use_each_once` true;
- `max_abs_intermediate` 10,000;
- `max_denominator` 1,000;
- `max_tree_depth` 5;
- `max_ast_nodes` 31;
- `max_answer_length` 1,024.

All 384 items use the defaults.

**Prompt tokens.** `prompt_token_ids` is the plain role-colon rendering, `User: …\n\nAssistant:`, with the
Qwen3.5-9B-Base tokenizer loaded through `AutoTokenizer`, which puts `<think>` at id 248068. The forced
prefix `\n<think>\n` = [198, 248068, 198] is appended at sampling time.
