# Jobs recipes

Tested recipes for things you can do with [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs):
train or reproduce a model, build a dataset, run batch inference, render 3D assets, build a container image, run a
service against a bucket, and more. Anything that runs in a container on a CPU or GPU can be a recipe.

Each recipe is one folder with:

- `README.md`: what it does, the commands, the cost and the results of a tested run;
- `AGENTS.md`: the procedure, the checks and the stop rules for a coding agent that runs the recipe for a user;
- the scripts. They pin the upstream code and data, and keep their state in a
  [Hugging Face bucket](https://huggingface.co/docs/hub/storage-buckets) when a recipe has more than one step, so a
  stopped Job can resume.

## Recipes

| recipe | what you get | hardware, time, cost |
|---|---|---|
| [decision-model](decision-model/) | a fine-tuned Kev-4B decision model that answers one question about a text with your labels, pushed with a model card | 1 × A100 (`a100-large`), 78 min for 2,000 records, about $3.25 |
| [filter-dataset](filter-dataset/) | a filtered subset of a large Hub dataset, pushed as a new dataset; one SQL expression, Polars or DuckDB | 1 CPU Job (`cpu-upgrade`), 6 min for 97M FineWeb-Edu rows, under $0.01 |
| [tiny-voca](tiny-voca/) | a 4.5M-parameter English TTS model with two voices, trained from scratch; WER close to the released checkpoint in one tested run | 1 × RTX PRO 6000, about 5.5 h, about $15 |

## Requirements

- A Hugging Face account that can run Jobs (a paid plan or credits).
- The `hf` CLI (`pip install -U huggingface_hub`), logged in with a token that can write.
- For recipes that keep state: a bucket, `hf buckets create <you>/<bucket> --private`.

Run the commands from the root of this repository. A recipe made of one uv script runs it with `hf jobs uv run
<recipe>/<script>.py`, which uploads the local script for you. A recipe with several files mounts its folder into the
Job with `-v ./<recipe>:/scripts:ro`. Each recipe's README gives its exact commands.

`hf jobs uv run` reads its own options (`--dry-run`, `--timeout`, `--name`, `--label`, ...) even after the script path, so
a script option with the same name never reaches the script. Put `--` before the script's options to pass them through
(`hf jobs uv run ... script.py -- --timeout 5`). The scripts in this repository do not use option names that the CLI uses.

## Watch, cost and stop a Job

- `hf jobs run` and `hf jobs uv run` stream the log and wait. Add `--detach` to get the Job id at once; an agent should
  do this, so it can check the Job and stop it.
- `hf jobs ps -a` lists your Jobs, `hf jobs logs <id>` prints the log, `hf jobs inspect <id>` gives the status and the
  running time, and `hf jobs cancel <id>` stops a Job.
- Cost = running time × the flavor's price per hour (`hf jobs hardware`).
- `--secrets HF_TOKEN` passes your token to the Job: from the `HF_TOKEN` variable, or else from `hf auth login`.
- A local folder in `-v` (for example `-v ./tiny-voca:/scripts:ro`) and a local script in `hf jobs uv run` are first
  copied to a bucket in your namespace (`<you>/jobs-artifacts`), which the Job mounts.

## Add a recipe

A recipe is ready when it has a tested full run (with its cost and results in the README), a cheap smoke test and an
`AGENTS.md`. Use the [tiny-voca](tiny-voca/) folder as the template.
