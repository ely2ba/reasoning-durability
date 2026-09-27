# Model outputs

**Published** as the Hugging Face dataset
[Ely2ba/reasoning-durability](https://huggingface.co/datasets/Ely2ba/reasoning-durability) (1.97 GB
compressed, 1,722 files; 6.96 GB as the runs wrote them). It holds what the paper's runs produced that is
too large, or too textual, for this repository:

- every evaluation and probe response: MATH-500 and AIME, "Wait" continuations, the TCES probes and the SK
  skill samples;
- every training text the models wrote: the base model's solutions (`rft`), the fresh-solution samples,
  gpt-oss-120b's traces, the sharpened solutions, the habit control's responses, the clone corpus and the
  anchor samples, each with its token ids and the decoded text;
- the texts scored for the divergence B, and the divergence arrays.

The release this repository describes is revision `1c1eef011ac400464cbb0ad03f159578a6b56092` (add
`--revision` to the download below to fetch exactly it).

Paths mirror the runs tree that `src/collect.py` reads, `runs/<family>/<run>/...`, and the dataset's
`MANIFEST.csv` lists every file with its rows, bytes and SHA-256. Its row counts equal those of `outputs/`.
The repository needs none of it to reproduce the paper's numbers, tables and figures; reading what the
models wrote, re-scoring it, or re-running an arm on the same texts does.

```
hf download Ely2ba/reasoning-durability --repo-type dataset --local-dir release/hf     # or --include "runs/math/pilot-20260924/*"
```

To re-run from it, decompress a run folder into `runs/`. The TCES runner reads the clone corpus from
`release/clone-corpus/group-NNNN/samples.json`:

```
for f in release/hf/runs/endpoint-clone/*/acquisition/corpus/group-*/samples.json.gz; do
  g=release/clone-corpus/$(basename "$(dirname "$f")"); mkdir -p "$g" && gunzip -c "$f" > "$g/samples.json"; done
```

**Removed or withheld.** The tokenized third-party prompts (NuminaMath-TIR problems, Tülu 3 chat prompts) are
removed from the training files; `make inputs` rebuilds them from their public sources, and the dataset card
shows how to restore the removed tokens exactly. The habit control's responses to the 131 WildJailbreak and
WildGuardMix prompts are withheld and available on request; the dataset lists them by id. Tinker checkpoint
paths, which carry account session ids, are removed.

**Not included, available on request.** The LoRA adapters of the endpoint-clone run (T-u10/u20/u30,
S-u10..u160: 19 files, 7.19 GB, tree SHA-256
`ef145442cb70fb1626f6845da6143e3218e88ed96b73342d4b59c52515401892`); the 86 Tinker states in
`checkpoints.json`, archived on the author's Tinker account; the RL teacher's training rollouts and the
clone's in-training probes; the runs of the original replay study (`replay-v1`), which trained the RP and
RS states that the TCES probes use; the aborted probe run `dc-20260923`; and the per-update training logs,
summarized in `outputs/train_log.csv.gz`.

**Terms.** Each model's outputs are under that model's terms: Qwen3.5 (Apache-2.0), NVIDIA Nemotron 3 Nano
(NVIDIA Nemotron Open Model License) and gpt-oss-120b (Apache-2.0, with OpenAI's gpt-oss usage policy).
Completions may restate the problems they solve; those problems stay under their own licences (MATH-500:
MIT; NuminaMath-TIR: Apache-2.0; AIME 2025–2026: CC BY-NC-SA 4.0). Our synthetic tasks, ids, scores and
arrays are Apache-2.0, like this repository.
