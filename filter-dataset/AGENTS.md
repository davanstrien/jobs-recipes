# Agent guide: filter a large dataset on Hugging Face Jobs

Read `README.md` in this folder first; it has the command and the options. This file is the procedure, the checks and
the stop rules for an agent that runs the recipe for a user.

## Ask the user before you start

1. The source dataset and which part of it (a config, a folder, a language), so you can set `--include`.
2. What to keep, in their words. Write the `--where` expression from it and show it to them.
3. The output repo name, and whether it may be public (default: private).

## Procedure

1. **Look at the data.** Read the dataset card and list the parquet files
   (`uv run --with huggingface_hub python -c "from huggingface_hub import HfApi; print(HfApi().list_repo_files('<id>', repo_type='dataset'))"`). Many large
   datasets have one folder per config (language, dump, ...) with split folders inside, and some have folders of
   removed or rejected rows (for example FineWeb-2's `data/<lang>_removed/`). Use the card to choose the folders that
   the user means, and say which ones you chose. Check the column names and types of one file, for example
   `uv run --with polars python -c "import polars as pl; print(pl.scan_parquet('hf://datasets/<id>/<file>').head(3).collect())"`
   (the dataset viewer API can fail for very large datasets). If the dataset has no parquet files, stop: this recipe
   reads parquet only.
2. **Test the filter.** Run `filter_polars.py` with `--no-push`, `--flavor cpu-upgrade` and a small `--include` (one file or one
   folder). Check the `RESULT` line: `output_rows` is more than 0 and the share is what the user expects. Show the user
   the share. To show them example rows, run the same small test without `--no-push` into a private test repo and read
   3-5 rows from it.
3. **Choose the flavor and script** for the full run from the Polars test's `peak_memory_gb` and the size of the full
   input (README, "Choose an engine"). The Polars peak grows with the input, but less than the input (2.1 GB at 10BT,
   11.5 GB at 100BT). If the full input is more than about 10 times the test, run a second, larger test first. Do not
   use DuckDB's `peak_memory_gb` for this: DuckDB fills memory up to its limit.
4. **Run the full filter** with a `--timeout` that fits: start from the test's `filter_seconds`, scaled by the input
   size, and double it.
5. **Check the result.** The log ends with `RESULT` and `pushed`. The output repo's README has the counts. Report the
   input rows, output rows, share, time and cost to the user.

If the user needs more than one filter expression (a new column, a join, a sample), copy the script for the engine and
change its `run_polars()` or `run_duckdb()`. Keep the rest (inputs, shards, card, push) as it is, and test the copy with
`--no-push` on a small `--include`.

## Stop and ask the user when

- the filter expression fails (a column name or function that does not exist): show the error and propose a fix
- the test keeps 0 rows, or a share very different from what the user expects
- a Job is killed for memory (`Killed`, exit code 137, or the log stops with no `RESULT`): offer `filter_duckdb.py` or a
  larger flavor
- the full run would cost more than the user agreed (CPU flavors: `hf jobs hardware`)
- a Job waits in SCHEDULING for more than 10 minutes (offer another flavor)

## Rules

- Keep the output repo private unless the user says otherwise.
- Pin `--revision` for a run that the user wants to reproduce; the default (current head) is written to the output card.
- The source dataset's licence and terms apply to the subset. Tell the user before they make the output public.
- Launch with `--detach` and follow the Job with `hf jobs inspect` and `hf jobs logs` (root README, "Watch, cost and
  stop a Job").
- Log what you launched (Job ids, flavor, time, cost) for the user. Cost = running time × the flavor's price.
