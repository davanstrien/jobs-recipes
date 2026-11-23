# Filter a large dataset on Hugging Face Jobs

Filter a large dataset on the Hub with one SQL expression, on a CPU Job, and push the result as a new dataset. The input
parquet files are read over HTTP at a pinned revision, so nothing is downloaded first and your laptop does no work.

```bash
hf jobs uv run --flavor cpu-upgrade --timeout 1h --secrets HF_TOKEN filter-dataset/filter.py \
    --dataset HuggingFaceFW/fineweb-edu --include "sample/10BT/*.parquet" \
    --where "int_score >= 4 AND token_count >= 4000 AND language_score >= 0.95" \
    --output-repo <you>/fineweb-edu-long-hq
```

This example keeps the long (4,000 tokens or more), high-quality, confidently English documents of FineWeb-Edu. The output
repo is private unless you add `--public`.

## Results

FineWeb-Edu at revision `87f09149`, the filter above, on `cpu-upgrade` (8 vCPU, 32 GB, $0.03 per hour):

| input | engine | input rows | output rows | filter time | Job time | peak memory | cost |
|---|---|---|---|---|---|---|---|
| `sample/10BT` (14 files) | polars | 9,672,101 | 17,423 (0.18 %) | 35-46 s | 44 s | 2.1 GB | under $0.01 |
| `sample/10BT` (14 files) | duckdb | 9,672,101 | 17,423 (0.18 %) | 110-150 s | 2 min | 27.2 GB* | under $0.01 |
| `sample/100BT` (140 files) | polars | 97,270,686 | 174,063 (0.18 %) | 311-457 s | 6 min | 10.9 GB | under $0.01 |

- Two runs each; the filter time is the range. "Job time" is the second run, from start to push.
- Both engines run the same filter expression and return the same rows.
- \*DuckDB uses memory up to its limit (80 % of the Job's memory by default) and spills to disk after that, so its peak
  shows the limit, not what the filter needs.

## Choose an engine

| engine | how it works | use it when |
|---|---|---|
| `polars` (default) | streaming scan and sink | the Job's memory is about 2-3 times the peak memory of a small test run. Fastest. |
| `duckdb` | out-of-core; spills to disk | the data does not fit in memory, or a Polars run is killed for memory. Slower. |

Start with a small part of the data (one folder or a few files, with `--include`) and read the `peak_memory_gb` value in
the `RESULT` line of a Polars run. The Polars peak grows with the input, but less than the input: 2.1 GB for 10BT and
10.9 GB for 100BT in the runs above. Run a second, larger test before you choose a small flavor for a very large input.

| flavor | vCPU | memory | disk | price per hour |
|---|---|---|---|---|
| `cpu-upgrade` | 8 | 32 GB | 50 GB | $0.03 |
| `cpu-xl` | 16 | 124 GB | 1,000 GB | $1.00 |
| `cpu-performance` | 32 | 256 GB | 1,024 GB | $1.90 |

Check the current list with `hf jobs hardware`. The disk holds the filtered output (and DuckDB's spill files) before the
upload, so the output must fit on it.

## Options

| option | default | meaning |
|---|---|---|
| `--dataset` | required | source dataset repo id |
| `--revision` | current head | the commit to read; the default is resolved and written to the output card |
| `--include` | all parquet files | one or more globs of parquet paths in the repo, e.g. `"data/fra_Latn/*.parquet"` |
| `--where` | required | the SQL filter expression. It must work in both Polars (`pl.sql_expr`) and DuckDB |
| `--columns` | all | columns to keep |
| `--engine` | `polars` | `polars` or `duckdb` |
| `--memory-limit` | DuckDB's own (80 % of memory) | DuckDB only, e.g. `24GB` |
| `--threads` | `$CPU_CORES` on Jobs | DuckDB only. Polars uses `$CPU_CORES` too. |
| `--output-repo` | required | the dataset repo to create or update; its `data/` files are replaced |
| `--public` | off | create the output repo public |
| `--dry-run` | off | filter and print the `RESULT` line, but do not push |

## Output

The output repo has `data/train-XXXXX-of-YYYYY.parquet` files of about 500 MB each and a README that records the
source dataset, the revision, the globs, the filter, the engine and the row counts. The source dataset's licence and
terms apply to the subset: read its card before you publish.

The log ends with one line:

```
RESULT engine polars <version> input_rows <n> output_rows <n> (<percent> %) shards <n> filter_seconds <s> peak_memory_gb <gb>
```

## Write a filter

The expression is SQL. These work in both engines:

- comparisons and logic: `score >= 3 AND (lang = 'fr' OR lang = 'de')`
- lists: `dump IN ('CC-MAIN-2024-10', 'CC-MAIN-2024-18')`
- text: `length(text) > 2000`, `url LIKE '%.ac.uk/%'`
- nulls: `title IS NOT NULL`

Test a new filter on a small `--include` with `--dry-run` first. A column or function that an engine does not know
makes the run fail early, with the engine's error.

## Notes

- **Memory.** In earlier tests on FineWeb-Edu, Polars 2.0 and older ran out of memory on large filters; Polars 2.1 fixed
  this. The script requires Polars 2.1 or newer.
- **CPU count.** A Job reports the host's CPUs. The script limits Polars and DuckDB to `$CPU_CORES`, the CPUs the Job may
  use.
- **Very large inputs.** One Job reads at about 110-135 MB/s in earlier DuckDB tests, so the Job timeout limits the input
  size (about 2.6 TB in 6 hours). For more, run several Jobs on parts of the data (`--include` per part) into separate
  output repos.
- Learn more about the approach in [Filtering FineWeb2 using Polars](https://danielvanstrien.xyz/posts/2024/12/23/fineweb-filter-polars.html).
