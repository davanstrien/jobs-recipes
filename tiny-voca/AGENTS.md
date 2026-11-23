# Agent guide: train tiny-voca on Hugging Face Jobs

Read `README.md` in this folder first; it has the commands. This file is the procedure, the checks and the stop rules for
an agent that runs the recipe for a user.

## Ask the user before you start

1. Their bucket (`hf://buckets/<user>/<bucket>`). If they have none, offer `hf buckets create <user>/<bucket> --private`.
   Pass `-e TV_BUCKET=hf://buckets/<user>/<bucket>/tiny-voca` to every Job.
2. A budget. Typical: under $1 for the smoke test, about $15 for a full run on `rtx-pro-6000`.
3. A run name (`RUN`), if they want one other than `repro`.

## Procedure

Run the commands from the root of this repository. Run one step, check its output, then run the next.

| step | run | check before going on |
|---|---|---|
| smoke | `run.sh all` with `-e SMOKE=1`, `--timeout 1h` | the log ends with `STAGE-DONE all`; `$TV_BUCKET/smoke/export/tiny-voca-smoke.onnx` exists |
| data | `run.sh data` (or `run.sh all` for the whole chain) | the log lines `annotated train: 16018 clips`, `validation: 714`, `test: 594`, then `data done` |
| train | `run.sh train`, `--timeout 8h` | `train done`; `runs/<run>/inference.pt` is in the bucket |
| eval | `run.sh eval` | `eval done`; two `"summary"` blocks in the log, the first for the user's run, the second for the released model |

With the default pins, other annotation counts mean the data stage did not run as tested. Stop and report the counts.

## Is the model good?

Compare the two `"summary"` blocks of the eval log, `test` → `all`:

- `generated_wer` of the user's run should be within about 0.3 points of the released model's (tested run: 1.26 % against
  1.33 %).
- `duration_ratio` should be between about 0.93 and 1.03.
- Report these numbers. Download 2-3 clips from `$TV_BUCKET/<run>/eval/<run>/` and the same clips from `eval/released/`
  for the user to compare by ear.

## Stop and ask the user when

- the spend would pass the agreed budget
- the smoke test fails: report the last 30 log lines
- the annotation counts differ from 16,018 / 714 / 594 with the default pins
- a Job waits in SCHEDULING for more than 10 minutes (offer another GPU flavor)
- a stage fails twice for the same reason

## Rules

- Keep the user's bucket private unless they say otherwise.
- Use the upstream code at the pinned commits; `run.sh` installs it. Do not copy it into the repository.
- To continue a stopped run, use the same `RUN`. The train stage resumes from `last.pt` in the bucket. Never reuse a run
  name for a different experiment: the train stage would resume the old run.
- Log what you launched (Job ids, flavor, time, cost) for the user.
