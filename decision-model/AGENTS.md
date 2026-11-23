# Agent guide: fine-tune a decision model (Kev) on Hugging Face Jobs

Read `README.md` in this folder first; it has the commands. This file is the procedure, the checks and the stop rules for
an agent that runs the recipe for a user.

## Ask the user before you start

1. The decision: what question the model answers, and which labels it chooses from. Write the `--question` with them.
2. The data: a Hub dataset with a text column and a label column. If their data is not on the Hub, help them push it
   first (a private dataset is fine).
3. Their bucket (`hf://buckets/<user>/<bucket>`), the model repo name, and a budget. Typical: under $1 for prepare and
   the smoke test, about $3.25 for a full run of 2,000 records × 2 epochs on `a100-large` ($2.50 per hour; check with
   `hf jobs hardware`). Training takes about 1 s per record per epoch.

## Procedure

Launch every Job with `--detach` and follow it with `hf jobs inspect` and `hf jobs logs` (root README). Run one step,
check it, then run the next.

| step | run | timeout | check before going on |
|---|---|---|---|
| prepare | `decision-model/prepare.py` on `cpu-basic` | 30m | the `RESULT` line: `train` and `development` counts, `labels`, and `labels_without_train_records` is empty. If `train_per_label_min` is below about 5, tell the user that rare labels will be weak. |
| smoke | `train-kev.py ... --max-steps 5` on `a100-large`, into a new output folder | 1h | the last line is a JSON summary with `"status": "completed"`; `rejected_records` in the benchmark output is small. Report the released model's accuracy: it is the baseline to beat. |
| train | `train-kev.py` without `--max-steps`, into another new output folder | records × epochs × 1.1 s + 15 min (2h for the default 2,000 × 2) | `run.json` has `"status": "completed"`; compare `finetuned` and `released` accuracy |
| publish | `decision-model/publish.py` on `cpu-basic` | 30m | `pushed https://huggingface.co/<repo>`; the repo is private unless the user asked for public |

Use the pinned `train-kev.py` URL from the README. Each `train-kev.py` run needs a new or empty output folder.

## Is the model good?

- `finetuned.acc` should be above `released.acc`. If it is not, report it: the released model may already fit the task
  (on banking77 the released Kev-4B already scores 82.8 %), or the data may be too small or noisy.
- ECE above about 0.05: if the out-of-fold row in the model card shows a lower error, tell the user to use the fitted
  temperature (`KEV_TEMPERATURE`, in the card) before they set confidence thresholds.
- It is one run with one seed and one development set. Say so when you report the numbers.

## Stop and ask the user when

- the spend would pass the agreed budget
- `prepare.py` stops (more than 255 labels, development labels not in the training labels, a column that does not
  exist): show the message
- the smoke test fails: report the last Python error
- many records are rejected or truncated by Kev's 384-token limit: offer a lower `--max-chars`, or shorter texts
- a Job waits in SCHEDULING for more than 10 minutes (offer another GPU flavor; only `a100-large` is tested)

## Rules

- Keep the bucket and the model repo private unless the user says otherwise.
- Do not copy `train-kev.py` into the repository; use the pinned URL.
- Check the licence of the training data before the model is made public, and credit the data as its licence asks.
- Log what you launched (Job ids, flavor, time, cost) for the user. Cost = running time × the flavor's price.
