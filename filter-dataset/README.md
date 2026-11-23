# Filter a large dataset on Hugging Face Jobs

Filter a large dataset on the Hub with one SQL expression, on a CPU Job, and push the result as a new dataset. The input
parquet files are read in the Job over `hf://` at one commit, with your token, so private and gated datasets work too.
Nothing is downloaded to your laptop.

```bash
hf jobs uv run --flavor cpu-upgrade --timeout 1h --secrets HF_TOKEN filter-dataset/filter_polars.py \
    --dataset HuggingFaceFW/fineweb-edu --include "sample/10BT/*.parquet" \
    --where "int_score >= 4 AND token_count >= 4000 AND language_score >= 0.95" \
    --output-repo <you>/fineweb-edu-long-hq
```

This example keeps the long (4,000 tokens or more), high-quality, confidently English documents of FineWeb-Edu. A new
output repo is private unless you add `--public`. The script refuses to push into an existing public repo without
`--public`.

## Results

FineWeb-Edu at revision `87f09149`, the filter above, on `cpu-upgrade` (8 vCPU, 32 GB, $0.03 per hour):

| input | engine | input rows | output rows | filter time | Job time | peak memory | cost |
|---|---|---|---|---|---|---|---|
| `sample/10BT` (14 files) | polars | 9,672,101 | 17,423 (0.18 %) | 35-46 s | 44 s | 2.1 GB | under $0.01 |
| `sample/10BT` (14 files) | duckdb | 9,672,101 | 17,423 (0.18 %) | 110-150 s | 2 min | 27.2 GB* | under $0.01 |
| `sample/100BT` (140 files) | polars | 97,270,686 | 174,063 (0.18 %) | 257-457 s | 6 min | 10.9-11.5 GB | under $0.01 |

- Two or three runs each; the filter time and peak memory are the range. "Job time" is from start to push.
- Both engines run the same filter expression and return the same rows.
- Peak memory is the highest resident memory of the script's process, from the start to the end of the shard step.
- \*DuckDB keeps data in its buffer pool up to its memory limit (80 % of the Job's memory by default) and spills to disk
  after that. Its peak memory therefore says little about the memory the filter needs.

## Choose an engine

There is one script per engine. They take the same options (DuckDB has two more) and write the same output, so you
can switch between them. Each script is short and has no code for the other engine, so it is easy to adapt.

| script | how it works | use it when | to adapt it |
|---|---|---|---|
| `filter_polars.py` | streaming scan and sink | the Job's memory is about 2-3 times the peak memory of a small test run. Fastest. | change `run_polars()`: any LazyFrame code works |
| `filter_duckdb.py` | out-of-core; spills to disk | the data does not fit in memory, or a Polars run is killed for memory. Slower. | change the SQL in `run_duckdb()` |

Start with a small part of the data (one folder or a few files, with `--include`) and read the `peak_memory_gb` value in
the `RESULT` line of a Polars run. The Polars peak grows with the input, but less than the input: 2.1 GB for 10BT and
11.5 GB for 100BT in the runs above. Run a second, larger test before you choose a small flavor for a very large input.

| flavor | vCPU | memory | disk | price per hour |
|---|---|---|---|---|
| `cpu-upgrade` | 8 | 32 GB | 50 GB | $0.03 |
| `cpu-xl` | 16 | 124 GB | 1,000 GB | $1.00 |
| `cpu-performance` | 32 | 256 GB | 1,024 GB | $1.90 |

Check the current list with `hf jobs hardware`. The disk holds the filtered output (and DuckDB's spill files) before the
upload. Splitting the output into shards needs about 2 times its size on disk; when the free space is too small, the
output is pushed as one file.

## Options

| option | default | meaning |
|---|---|---|
| `--dataset` | required | source dataset repo id |
| `--revision` | current head | a commit, branch or tag; it is resolved to one commit, which is read and written to the output card |
| `--include` | all parquet files | one or more globs of parquet paths in the repo, e.g. `"data/fra_Latn/train/*.parquet"`. A `*` also matches `/`, so `"data/*.parquet"` takes every subfolder. |
| `--where` | required | the SQL filter expression. It must work in both Polars (`pl.sql_expr`) and DuckDB |
| `--columns` | all | columns to keep |
| `--memory-limit` | DuckDB's own (80 % of memory) | `filter_duckdb.py` only, e.g. `24GB` |
| `--threads` | `$CPU_CORES` on Jobs | `filter_duckdb.py` only. `filter_polars.py` also uses `$CPU_CORES` (unless `POLARS_MAX_THREADS` is set). |
| `--output-repo` | required | the dataset repo to create or update; its `data/` files are replaced |
| `--public` | off | create the output repo public. It does not change an existing repo's visibility. |
| `--no-push` | off | filter and print the `RESULT` line, but do not push |

The files that `--include` selects must have the same columns for `filter_polars.py`. If they do not (for example a
`train` and a `test` split with different columns), it stops with a message: narrow `--include`, or use
`filter_duckdb.py`, which merges the columns by name and fills the missing ones with null.

## Output

The output repo has `data/train-XXXXX-of-YYYYY.parquet` files of about 500 MB each and a README that records the
source dataset, the commit, the globs, the filter, the engine and the row counts. A new run into the same repo replaces
the `data/` files and the README. A large upload can take more than one commit, so a stopped re-run can leave a mix of
old and new files: run it again, or use a new repo name. The source dataset's licence and
terms apply to the subset: read its card before you publish.

The log ends with one line:

```
RESULT engine polars <version> input_rows <n> output_rows <n> (<percent> %) shards <n> filter_seconds <s> peak_memory_gb <gb>
```

## Write a filter

The expression is SQL. These work in both scripts:

- comparisons and logic: `score >= 3 AND (lang = 'fr' OR lang = 'de')`
- lists: `dump IN ('CC-MAIN-2024-10', 'CC-MAIN-2024-18')`
- text: `length(text) > 2000`, `url LIKE '%.ac.uk/%'`
- nulls: `title IS NOT NULL`

Test a new filter on a small `--include` with `--no-push` first. A column or function that an engine does not know
makes the run fail early, with the engine's error.

## Notes

- **Memory.** In earlier tests on FineWeb-Edu, Polars 2.0 and older ran out of memory on large filters; Polars 2.1 fixed
  this. The script requires Polars 2.1 or newer.
- **CPU count.** A Job reports the host's CPUs. The script sets Polars' and DuckDB's thread counts to `$CPU_CORES`, the
  CPUs the Job may use, by default.
- **Very large inputs.** One Job reads at about 110-135 MB/s in earlier DuckDB tests, so the Job timeout limits the input
  size (about 2.6 TB in 6 hours). For more, run several Jobs on parts of the data (`--include` per part) into separate
  output repos.
- Learn more about the approach in [Filtering FineWeb2 using Polars](https://danielvanstrien.xyz/posts/2024/12/23/fineweb-filter-polars.html).
