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
| [filter-dataset](filter-dataset/) | a filtered subset of a large Hub dataset, pushed as a new dataset; one SQL expression, Polars or DuckDB | 1 CPU Job (`cpu-upgrade`), 6 min for 97M FineWeb-Edu rows, under $0.01 |
| [tiny-voca](tiny-voca/) | a 4.5M-parameter English TTS model with two voices, trained from scratch; WER close to the released checkpoint in one tested run | 1 × RTX PRO 6000, about 5.5 h, about $15 |

## Requirements

- A Hugging Face account that can run Jobs (a paid plan or credits).
- The `hf` CLI (`pip install -U huggingface_hub`), logged in with a token that can write.
- For recipes that keep state: a bucket, `hf buckets create <you>/<bucket> --private`.

Run the commands from the root of this repository: each command mounts its recipe folder into the Job with
`-v ./<recipe>:/scripts:ro`.

## Add a recipe

A recipe is ready when it has a tested full run (with its cost and results in the README), a cheap smoke test and an
`AGENTS.md`. Use the [tiny-voca](tiny-voca/) folder as the template.
