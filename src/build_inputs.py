"""Rebuild the full-text inputs the runners read from the public datasets in the local Hugging Face
cache, never downloading: `python src/build_inputs.py [--duraseed /path/to/DuraSeed-v1]`.

data/ stores third-party data by id only. This writes the texts to inputs/ (kept out of git):
inputs/math/pool.jsonl (the 36,447 training problems; must match data/math/pool.sha256),
inputs/math/eval_math500.jsonl (the 311 integer-answer problems), inputs/math/aime.jsonl (59),
inputs/no_robots.jsonl (the 640 instruction-tuning rows in training order, then the 4,480 FN rows),
inputs/chat.jsonl (R5's 1,920 broad chat rows, in training order) and inputs/no_robots_epoch.jsonl (the
revision's RS: 8,544 No Robots rows in training order, the first 640 those of instruction tuning). With
--duraseed, each file is also checked against the DuraSeed-v1 file it replaces.
"""

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
DATA, INPUTS = REPO / "data", REPO / "inputs"
HUB = Path("~/.cache/huggingface/hub").expanduser()
TULU = "datasets--allenai--tulu-3-sft-mixture/snapshots/*/data/train-*.parquet"
MATH500 = "datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl"
AIME = "datasets--MathArena--aime_{}/snapshots/*/data/*.parquet"
NAMESPACE = "math-pilot-20260924"


def cached(pattern):
    paths = sorted(glob.glob(str(HUB / pattern)))
    assert paths, f"not in the local cache: {pattern}"
    return paths


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def order_key(problem):
    """The pool and evaluation order: SHA-256 of the namespace and the problem's task id."""
    task_id = hashlib.sha256(" ".join(problem.lower().split()).encode()).hexdigest()
    return hashlib.sha256(f"{NAMESPACE}:{task_id}".encode()).hexdigest()


def tulu_rows(ids):
    """Tulu-3 SFT mixture rows by id, from the cached shards."""
    wanted, out = set(ids), {}
    for path in cached(TULU):
        table = pq.read_table(path, columns=["id", "messages"])
        out |= {r["id"]: r["messages"] for r in table.to_pylist() if r["id"] in wanted}
    missing = wanted - set(out)
    assert not missing, f"{len(missing)} ids are missing from the cached Tulu-3 shards"
    return out


def build_pool(tok):
    ids = [json.loads(line) for line in (DATA / "math/pool_ids.jsonl").open()]
    messages = tulu_rows(r["source_row_id"] for r in ids)
    rows = []
    for r in ids:
        problem = messages[r["source_row_id"]][0]["content"]
        rows.append({"task_id": r["task_id"], "problem": problem, "answer": r["answer"],
                     "tokens": len(tok.encode(problem, add_special_tokens=False))})
    return write(INPUTS / "math/pool.jsonl", rows)


def build_math500():
    rows = [json.loads(line) for line in open(cached(MATH500)[0])]
    rows = [{"unique_id": r["unique_id"], "problem": r["problem"], "answer": int(r["answer"]), "level": r["level"]}
            for r in rows if re.fullmatch(r"-?\d+", r["answer"])]
    return write(INPUTS / "math/eval_math500.jsonl", sorted(rows, key=lambda r: order_key(r["problem"])))


def build_aime():
    rows = []
    for year in (2025, 2026):
        rows += [{"unique_id": f"aime{year}-{r['problem_idx']}", "problem": r["problem"], "answer": int(r["answer"]),
                  "level": f"aime{year}"} for r in pq.read_table(cached(AIME.format(year))[0]).to_pylist()
                 if (year, r["problem_idx"]) != (2026, 1)]  # 2026 problem 1 is word for word in NuminaMath-TIR
    return write(INPUTS / "math/aime.jsonl", rows)


def conversations(order, name):
    """inputs/<name>.jsonl: the single-turn Tulu-3 rows `order` names, in that order, as prompt and completion."""
    messages = tulu_rows(order)
    return write(INPUTS / f"{name}.jsonl", [{"id": i, "prompt": messages[i][0]["content"],
                                             "completion": messages[i][1]["content"]} for i in order])


def build_no_robots():
    ids = json.loads((DATA / "no_robots_ids.json").read_text())
    return conversations(ids["d3_instruction_tuning"]["ids"] + ids["fn_anchor_control"]["ids"], "no_robots")


def build_chat():
    return conversations([r["id"] for r in json.loads((DATA / "chat_ids.json").read_text())["rows"]], "chat")


def build_epoch():
    return conversations(json.loads((DATA / "revision_ids.json").read_text())["rs_no_robots"]["ids"], "no_robots_epoch")


def jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def check(paths, duraseed):
    pool_sha = hashlib.sha256(paths["pool"].read_bytes()).hexdigest()
    results = {"pool.jsonl matches data/math/pool.sha256": pool_sha == (DATA / "math/pool.sha256").read_text().split()[0]}
    aime = jsonl(paths["aime"])
    frozen = json.loads((DATA / "math/aime_ids.json").read_text())["items"]
    results["aime.jsonl matches data/math/aime_ids.json"] = [(r["unique_id"], r["answer"]) for r in aime] == [
        (r["id"], r["answer"]) for r in frozen]
    results["eval_math500.jsonl has 311 rows"] = len(jsonl(paths["math500"])) == 311
    rows = jsonl(paths["no_robots"])
    ids = json.loads((DATA / "no_robots_ids.json").read_text())
    results["no_robots.jsonl holds 640 + 4,480 rows in the frozen order"] = [r["id"] for r in rows] == (
        ids["d3_instruction_tuning"]["ids"] + ids["fn_anchor_control"]["ids"])
    chat, frozen = jsonl(paths["chat"]), json.loads((DATA / "chat_ids.json").read_text())
    pattern = re.compile(frozen["mathlike_pattern"], re.IGNORECASE)
    results["chat.jsonl holds the 1,920 rows in the frozen order, with the frozen keyword flags"] = [
        (r["id"], int(bool(pattern.search(r["prompt"] + r["completion"])))) for r in chat] == [
        (r["id"], r["mathlike"]) for r in frozen["rows"]]
    epoch = [r["id"] for r in jsonl(paths["epoch"])]
    results["no_robots_epoch.jsonl holds RS's 8,544 rows in the frozen order, the 640 of instruction tuning first"] = (
        epoch == json.loads((DATA / "revision_ids.json").read_text())["rs_no_robots"]["ids"]
        and epoch[:640] == ids["d3_instruction_tuning"]["ids"])
    if duraseed:
        old = Path(duraseed)
        results["pool.jsonl is byte-identical to DuraSeed-v1 runs/math/data/pool.jsonl"] = (
            paths["pool"].read_bytes() == (old / "runs/math/data/pool.jsonl").read_bytes())
        fields = ("unique_id", "problem", "answer", "level")
        results["eval_math500.jsonl equals DuraSeed-v1 runs/math/data/eval_math500.jsonl (fields, order)"] = [
            [r[k] for k in fields] for r in jsonl(paths["math500"])] == [
            [r[k] for k in fields] for r in jsonl(old / "runs/math/data/eval_math500.jsonl")]
        d3 = json.loads((old / "runs/repetition-probe/rep-20260923-instruct/config.json").read_text())["examples"]
        results["no_robots.jsonl opens with the D3 ids in DuraSeed-v1's training order"] = [r["id"] for r in rows[:640]] == d3
        r5 = json.loads((old / "runs/math/robustness-followup-20260924/r5/examples.json").read_text())["examples"]
        results["chat.jsonl holds R5's ids in DuraSeed-v1's training order"] = [r["id"] for r in chat] == r5
        rs = json.loads((old / "runs/math/revision-rs-20260926/examples.json").read_text())["examples"]
        results["no_robots_epoch.jsonl holds RS's ids in DuraSeed-v1's training order"] = epoch == rs
    for name, ok in results.items():
        print(("ok    " if ok else "FAILED"), name)
    return all(results.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--duraseed", help="a DuraSeed-v1 checkout, to check each file against the one it replaces")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"  # the tokenizer from the local cache only
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B-Base")
    paths = {"pool": build_pool(tok), "math500": build_math500(), "aime": build_aime(), "no_robots": build_no_robots(),
             "chat": build_chat(), "epoch": build_epoch()}
    for path in paths.values():
        print(f"{path.relative_to(REPO)}: {sum(1 for _ in path.open()):,} rows")
    raise SystemExit(0 if check(paths, args.duraseed) else 1)


if __name__ == "__main__":
    main()
