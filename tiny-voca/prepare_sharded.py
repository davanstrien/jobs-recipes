"""Run tiny_voca.prepare on shards of the input files in parallel, then merge the shards into one features directory.

tiny_voca.prepare tracks pitch with librosa.pyin on one core. Each shard is a separate `python -m tiny_voca.prepare` process over
a contiguous slice of the input files; the merge keeps the input-file order, so the manifest lists the clips in the same order as
one serial run. Feature paths (<split>/<sha256(id)>.pt) do not depend on the shard.

    python prepare_sharded.py --split train --inputs data/train-*.parquet --output features --workers 8
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--inputs", nargs="+", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

inputs = list(a.inputs)
k = min(a.workers, len(inputs))
cuts = [round(i * len(inputs) / k) for i in range(k + 1)]
shards = [inputs[cuts[i]:cuts[i + 1]] for i in range(k)]
env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
work = a.output / f".shards-{a.split}"
work.mkdir(parents=True, exist_ok=True)
procs = []
for i, files in enumerate(shards):
    log = open(a.output / f"prepare-{a.split}-{i}.log", "a")
    cmd = [sys.executable, "-m", "tiny_voca.prepare", "--split", a.split, "--output", str(work / str(i)), "--inputs", *map(str, files)]
    if (work / str(i) / f"{a.split}.preparation.json").exists():
        cmd.append("--resume")
    procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]
assert not any(codes), f"prepare shards failed: {codes} (see {a.output}/prepare-{a.split}-*.log)"

(a.output / a.split).mkdir(parents=True, exist_ok=True)
accepted = rejected = 0
schema = None
seen = set()
man_tmp, rej_tmp = a.output / f"{a.split}.jsonl.tmp", a.output / f"{a.split}.rejected.jsonl.tmp"
with open(man_tmp, "w") as man, open(rej_tmp, "w") as rej:
    for i in range(k):
        d = work / str(i)
        meta = json.loads((d / f"{a.split}.preparation.json").read_text())
        assert meta["complete"], f"shard {i} incomplete"
        schema = schema or (d / "features.json").read_text()
        assert (d / "features.json").read_text() == schema, "feature schema differs between shards"
        # physical LF lines, as tiny_voca reads its manifests (text may hold U+2028 and other line separators)
        for line in (d / f"{a.split}.jsonl").read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            row["record_key"] = f"s{i}:{row['record_key']}"   # unique across shards
            if row["id"] in seen:   # as one serial run: the first accepted id wins, a later one is a reject
                rej.write(json.dumps({"record_key": row["record_key"], "id": row["id"], "source": row.get("source_file"),
                                      "reason": "ValueError: duplicate id"}, ensure_ascii=False) + "\n")
                rejected += 1
                continue
            seen.add(row["id"])
            man.write(json.dumps(row, ensure_ascii=False) + "\n")
            dst = a.output / row["path"]
            if not dst.exists():
                os.link(d / row["path"], dst)   # hard link: the shard keeps its file until the merge is complete
            accepted += 1
        r = d / f"{a.split}.rejected.jsonl"
        if r.exists():
            for line in r.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    rej.write(line + "\n"); rejected += 1
(a.output / "features.json").write_text(schema)
os.replace(man_tmp, a.output / f"{a.split}.jsonl")
os.replace(rej_tmp, a.output / f"{a.split}.rejected.jsonl")
(a.output / f"{a.split}.preparation.json").write_text(json.dumps(
    {"identity": {"sharded": True, "shards": [[str(p) for p in s] for s in shards]}, "complete": True, "accepted": accepted, "rejected": rejected}))
shutil.rmtree(work)
print(json.dumps({"split": a.split, "accepted": accepted, "rejected": rejected, "shards": k}), flush=True)
