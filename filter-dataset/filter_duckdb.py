# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.5", "pyarrow>=18", "huggingface_hub>=1.0"]
# ///
"""Filter a large Hugging Face dataset with DuckDB on a Job, and push the result as a new dataset.

The filter is one SQL WHERE expression, for example "int_score >= 4 AND token_count >= 4000". The input parquet files
are read over hf:// at one commit with your token (so private and gated datasets work). DuckDB runs out-of-core: it
spills to disk when memory is full, so it also works when the data does not fit in memory. Slower than
filter_polars.py.

To adapt it, change the SQL in run_duckdb(): joins, GROUP BY, window functions and DuckDB's text functions all work.

Output: <output-repo> (private unless --public) with data/train-*.parquet and a README that records the source, the
commit, the filter and the row counts.

    hf jobs uv run --flavor cpu-upgrade --timeout 1h --secrets HF_TOKEN filter-dataset/filter_duckdb.py \\
        --dataset HuggingFaceFW/fineweb-edu --include "sample/10BT/*.parquet" \\
        --where "int_score >= 4 AND token_count >= 4000 AND language_score >= 0.95" \\
        --output-repo <you>/fineweb-edu-long-hq
"""
import argparse
import fnmatch
import json
import os
import re
import resource
import shutil
import sys
import time
from pathlib import Path

ENGINE = "duckdb"
OUT = Path("/tmp/filter-out")
SHARD_BYTES = 500 * 1024**2  # output shards of about this size


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def peak_gb():
    """peak resident memory of this process, in GB (Linux reports KB, macOS bytes)"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2)


def lit(s):
    """a SQL string literal"""
    return "'" + str(s).replace("'", "''") + "'"


def ident(c):
    """a SQL identifier"""
    return '"' + str(c).replace('"', '""') + '"'


def input_paths(api, dataset, sha, include):
    """hf:// paths at one commit; DuckDB reads them with the token, so private and gated datasets work"""
    files = sorted(f for f in api.list_repo_files(dataset, repo_type="dataset", revision=sha)
                   if f.endswith(".parquet") and any(fnmatch.fnmatch(f, p) for p in include))
    if not files:
        raise SystemExit(f"no parquet files in {dataset}@{sha} match {include}")
    return files, [f"hf://datasets/{dataset}@{sha}/{f}" for f in files]


def run_duckdb(urls, where, columns, out, memory_limit, threads, token):
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if token:
        con.execute(f"CREATE SECRET hf (TYPE huggingface, TOKEN {lit(token)})")
    con.execute("SET preserve_insertion_order=false;")   # lets COPY stream
    con.execute("SET temp_directory='/tmp/duckdb-spill';")
    if memory_limit:
        con.execute(f"SET memory_limit={lit(memory_limit)};")
    if threads:
        con.execute(f"SET threads={int(threads)};")
    src = "read_parquet([" + ", ".join(lit(u) for u in urls) + "], union_by_name = true)"   # files may differ in columns
    n_in = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]   # from the parquet footers
    cols = ", ".join(ident(c) for c in columns) if columns else "*"
    con.execute(f"COPY (SELECT {cols} FROM {src} WHERE {where}) TO {lit(out)} (FORMAT parquet, COMPRESSION zstd)")
    return n_in, duckdb.__version__


def shard(path, out_dir):
    """split the filtered file into data/train-XXXXX-of-YYYYY.parquet files of about SHARD_BYTES each (on disk).

    It streams record batches, so a large row group never has to fit in memory. While it writes, the disk holds the
    filtered file and the shards; when the free space is too small for that, the file is uploaded as one shard."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    n_out, size = pf.metadata.num_rows, os.path.getsize(path)
    n_shards = max(1, round(size / SHARD_BYTES))
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_shards > 1 and shutil.disk_usage(out_dir).free < 1.2 * size:
        log(f"not enough free disk to split {size / 1024**3:.1f} GB into shards: one shard")
        n_shards = 1
    if n_shards == 1:
        shutil.move(path, out_dir / "train-00000-of-00001.parquet")
        return n_out, 1
    per = -(-n_out // n_shards)   # rows per shard
    k, w, rows = 0, None, 0
    for batch in pf.iter_batches(batch_size=10_000):
        while batch.num_rows:
            if w is None:
                w = pq.ParquetWriter(out_dir / f"train-{k:05d}-of-{n_shards:05d}.parquet", pf.schema_arrow, compression="zstd")
            take = min(per - rows, batch.num_rows)
            w.write_batch(batch.slice(0, take)); batch = batch.slice(take); rows += take
            if rows == per:
                w.close(); w, rows, k = None, 0, k + 1
    if w is not None:
        w.close(); k += 1
    os.remove(path)
    if k != n_shards:   # rename to the true count (the last shards can be empty when rows are few)
        for i in range(k):
            os.replace(out_dir / f"train-{i:05d}-of-{n_shards:05d}.parquet", out_dir / f"train-{i:05d}-of-{k:05d}.parquet")
    return n_out, k


def made_with(seconds):
    recipe = "[filter-dataset recipe](https://github.com/davanstrien/jobs-recipes/tree/main/filter-dataset) (`filter_duckdb.py`)"
    if os.environ.get("JOB_ID"):   # set inside a Hugging Face Job
        return (f"Made on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) with the {recipe}, "
                f"in {round(seconds)} s of filtering on one CPU Job.")
    return f"Made with the {recipe}."


def card(a, revision, files, n_in, n_out, engine_version, seconds):
    meta = {"source": a.dataset, "revision": revision, "include": a.include, "where": a.where, "columns": a.columns,
            "engine": f"{ENGINE} {engine_version}", "input_files": len(files), "input_rows": n_in, "output_rows": n_out,
            "seconds": round(seconds)}
    return f"""---
configs:
- config_name: default
  data_files: data/*.parquet
tags:
- filtered
---

# {a.output_repo.split('/')[-1]}

A filtered subset of [{a.dataset}](https://huggingface.co/datasets/{a.dataset}) at revision `{revision}`.
Read the source dataset's card for its licence and terms; they apply to this subset.

| | |
|---|---|
| input files | `{', '.join(a.include)}` ({len(files)} files) |
| filter | `{a.where}` |
| columns | {', '.join(f'`{c}`' for c in a.columns) if a.columns else 'all'} |
| rows | {n_out:,} of {n_in:,} ({100 * n_out / max(n_in, 1):.3f} %) |
| engine | {ENGINE} {engine_version} |

{made_with(seconds)}

```json
{json.dumps(meta, indent=1)}
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="source dataset repo id")
    ap.add_argument("--revision", default=None, help="commit to read (default: the current head, resolved and logged)")
    ap.add_argument("--include", nargs="+", default=["*.parquet", "**/*.parquet"], help="glob(s) of parquet files in the repo")
    ap.add_argument("--where", required=True, help="SQL filter expression")
    ap.add_argument("--columns", nargs="*", default=None, help="columns to keep (default: all)")
    ap.add_argument("--memory-limit", default="", help="e.g. 24GB (default: DuckDB's own, 80 %% of RAM)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("CPU_CORES") or 0),
                    help="threads (default: $CPU_CORES on Jobs, else DuckDB's own)")
    ap.add_argument("--output-repo", required=True, help="dataset repo to create or update")
    ap.add_argument("--public", action="store_true", help="create the output repo public (default: private)")
    ap.add_argument("--no-push", action="store_true", help="filter and report, but do not push")
    a = ap.parse_args()

    from huggingface_hub import HfApi, get_token
    api = HfApi()
    if a.memory_limit and not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", a.memory_limit):
        raise SystemExit(f"--memory-limit must look like 24GB, not {a.memory_limit!r}")
    token = get_token()
    # an existing repo keeps its visibility: never push into a public one without --public
    if (not a.no_push and not a.public and api.repo_exists(a.output_repo, repo_type="dataset")
            and not api.dataset_info(a.output_repo).private):
        raise SystemExit(f"{a.output_repo} exists and is public: pass --public to push into it, or choose another name")
    revision = api.dataset_info(a.dataset, revision=a.revision).sha   # a branch or tag becomes one commit
    files, urls = input_paths(api, a.dataset, revision, a.include)
    log(f"{a.dataset}@{revision}: {len(files)} files ({files[0]} ... {files[-1]})")
    log(f"engine {ENGINE}, filter: {a.where}")

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    t0 = time.time()
    tmp = str(OUT / "filtered.parquet")
    n_in, version = run_duckdb(urls, a.where, a.columns, tmp, a.memory_limit, a.threads, token)
    seconds = time.time() - t0
    n_out, n_shards = shard(tmp, OUT / "data")
    log(f"RESULT engine {ENGINE} {version} input_rows {n_in} output_rows {n_out} ({100 * n_out / max(n_in, 1):.3f} %) "
        f"shards {n_shards} filter_seconds {seconds:.0f} peak_memory_gb {peak_gb():.1f}")
    (OUT / "README.md").write_text(card(a, revision, files, n_in, n_out, version, seconds))
    if a.no_push:
        log("--no-push: not pushed")
        return
    api.create_repo(a.output_repo, repo_type="dataset", private=not a.public, exist_ok=True)
    api.upload_folder(repo_id=a.output_repo, repo_type="dataset", folder_path=OUT, allow_patterns=["README.md", "data/*"],
                      delete_patterns=["data/*"], commit_message=f"filter {a.dataset}@{revision[:8]}: {a.where}")
    log(f"pushed https://huggingface.co/datasets/{a.output_repo}")


if __name__ == "__main__":
    main()
