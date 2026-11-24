# Agent guide: train tiny-voca on Hugging Face Jobs

Read `README.md` in this folder first; it has the commands. This file is the procedure, the checks and the stop rules for
an agent that runs the recipe for a user.

## Ask the user before you start

1. Their bucket (`hf://buckets/<user>/<bucket>`). If they have none, offer `hf buckets create <user>/<bucket> --private`.
   Pass `-e TV_BUCKET=hf://buckets/<user>/<bucket>/tiny-voca` to every Job.
2. A budget. `rtx-pro-6000` costs $2.75 per hour (check with `hf jobs hardware`). Typical: under $1 for the smoke test,
   about $15 for a full run. The 8 h train timeout limits a full run to about $22.
3. A run name (`RUN`), if they want one other than `repro`. Use a new name for each new experiment.

## Procedure

Run the commands from the root of this repository, with `--detach`, and follow each Job with `hf jobs inspect` and
`hf jobs logs` (root README, "Watch, cost and stop a Job"). Run each stage as its own Job, with the timeout below. Check its
output before you start the next one. Do not use `run.sh all` for a full run: it goes on to training without your check.

| step | run | timeout | check before going on |
|---|---|---|---|
| smoke | `run.sh all` with `-e SMOKE=1` (and a new `-e RUN=smoke-<date>` for each smoke test) | 1h | the log ends with `STAGE-DONE all`; `$TV_BUCKET/$RUN/export/tiny-voca-$RUN.onnx` exists. The smoke data has 1,231 / 16 / 10 annotated clips, not the full counts below. |
| data | `run.sh data` | 1h | the log lines `annotated train: 16018 clips`, `validation: 714`, `test: 594`, then `data done` |
| train | `run.sh train` | 8h | `train done`; `runs/<run>/inference.pt` is in the bucket |
| eval | `run.sh eval` | 2h | `eval done`; two `RESULT` lines, the first for the user's run, the second for `released` |

The data stage fails by itself if a count differs by more than 5 clips from the counts above (full data, default pins). A
difference of 1-5 clips can come from newer Python packages: report it to the user and go on.

If a stage stops, start the same stage again with the same `RUN`. Never run `run.sh data` with `FORCE_DATA=1` for a run
that has started training: the training cannot resume on new annotations.

## Is the model good?

Read the two `RESULT` lines of the eval log (or `eval/<run>/report.json` and `eval/released/report.json`, `summary` →
`test` → `all`):

- `step` of the user's run must be 104000. A lower step means the train stage did not finish; run `run.sh train` again.
- `token mismatches` must be 0 for both models.
- `test wer` of the user's run should be not more than about 0.3 points above the released model's (tested run: 1.26 %
  against 1.33 %). This limit comes from one run.
- `duration` should be between about 0.93 and 1.03.
- Report these numbers. Download `eval/<run>/audio.tar` and `eval/released/audio.tar`, and give the user 2-3 clips with
  the same id from both (`<id>-generated.wav`) to compare by ear.

## Stop and ask the user when

- the spend would pass the agreed budget
- the smoke test fails: report the `FAILED` line and the Python traceback above it
- the data stage fails on the clip counts
- a Job waits in SCHEDULING for more than 10 minutes (offer another GPU flavor)
- a stage fails twice for the same reason

## Rules

- Keep the user's bucket private unless they say otherwise.
- Use the upstream code at the pinned commits; `run.sh` installs it. Do not copy it into the repository.
- A model that the user publishes from this recipe uses the tiny-voca dataset (CC BY 4.0): tell them to credit it.
- Log what you launched (Job ids, flavor, time, cost) for the user. Cost = running time × the flavor's price.
- A smoke test that reuses a `RUN` name reuses its data in the bucket; use a new name to test the data stage again.

## New voices (experimental)

`README.md`, section "A new voice (experimental)", has the steps. Before a full render (about 3.7 h, about $5.50 per voice
on `a10g-large`), run `render_voice.py` with `--limit 50` and give the user 2-3 clips to check by ear. The Whisper screen
checks the words, not the voice or the accent. Training with `VOICES` is not tested: tell the user before you start it.
