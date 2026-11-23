# Fine-tune a decision model on Hugging Face Jobs

Fine-tune [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b), an open decision model by Jared Palmer, to answer one
question about a text: for example "Which team should handle this message?" with your own labels. Kev gives a
probability for each answer, so you can act on the confident cases and send the rest to a person.

Three Jobs, from a labelled Hub dataset to a model on the Hub:

| step | script | flavor | what it does |
|---|---|---|---|
| 1 prepare | `prepare.py` (this folder) | `cpu-basic` | turns a labelled dataset into Kev records (`train.jsonl`, `development.jsonl`) in your bucket |
| 2 train | [`train-kev.py`](https://huggingface.co/datasets/uv-scripts/classification) from uv-scripts, pinned | `a100-large` | Kev's own training, then Kev's benchmark of the fine-tuned and the released model, and a calibration report |
| 3 publish | `publish.py` (this folder) | `cpu-basic` | pushes the checkpoint, the reports and a model card to a model repo (private by default) |

Step 2 runs the uv-scripts training script by URL at a pinned commit; this recipe does not copy it.

## Results

[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) (CC BY 4.0): 77 customer-support intents of a
bank. 2,000 training records and 500 development records (from the test split), question "Which customer-support intent
does this message express?".

| model | accuracy | calibration error (ECE) |
|---|---|---|
| released Kev-4B | 82.8 % | 0.123 |
| fine-tuned | **87.6 %** | 0.077 |
| fine-tuned, temperature fitted on other folds (out-of-fold check) | 87.6 % | 0.022 |

- Fine-tuned minus released, on the same 500 records: +4.8 points (95 % bootstrap interval +2.2 to +7.4); 35 answers
  became correct and 11 became wrong. One seed.
- The released model is already good at this task; fine-tuning adds 4.8 points. Kev's calibration step fits one
  temperature (1.91 here) on the development records; the out-of-fold check (row 3) shows that it lowers the
  calibration error from 0.077 to 0.022.
- Full run: 78 min on `a100-large` (training 2 epochs at about 1.06 s per record, then both benchmarks), about $3.25.
  Smoke test (`--max-steps 5`): 8.5 min, about $0.35; most of it is the two benchmarks, so it scales with the
  development records (4 min for 300 records in another test). Prepare and publish: under a minute each on `cpu-basic`.
- The model is [davanstrien/kev-banking77](https://huggingface.co/davanstrien/kev-banking77) (private for now).

## Before you start

- A Hugging Face account that can run Jobs, and the `hf` CLI, logged in with a token that can write.
- A bucket for the records and the results: `hf buckets create <you>/<bucket> --private`.
- A labelled dataset on the Hub: one text column and one label column (a `ClassLabel` or text), up to 255 labels.

## Run it

Run the commands from the root of this repository.

```bash
B=hf://buckets/<you>/<bucket>
TRAIN=https://huggingface.co/datasets/uv-scripts/classification/resolve/6ce57c6e690113f49fc28dfdd46007425fa97146/train-kev.py

# 1. prepare (about 1 min, under $0.01)
hf jobs uv run --flavor cpu-basic --timeout 30m --secrets HF_TOKEN -v $B:/mnt/kev decision-model/prepare.py \
    --dataset PolyAI/banking77 --text-column text --label-column label \
    --question "Which customer-support intent does this message express?" --out /mnt/kev/banking77

# 2a. smoke test: 5 training steps, then both benchmarks (about 9 min, about $0.35)
hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN -v $B:/mnt/kev $TRAIN \
    /mnt/kev/banking77 /mnt/kev/runs/banking77-smoke --max-steps 5

# 2b. full training (78 min for 2,000 records × 2 epochs, about $3.25)
hf jobs uv run --flavor a100-large --timeout 2h --secrets HF_TOKEN -v $B:/mnt/kev $TRAIN \
    /mnt/kev/banking77 /mnt/kev/runs/banking77-v1

# 3. publish (under a minute)
hf jobs uv run --flavor cpu-basic --timeout 30m --secrets HF_TOKEN -v $B:/mnt/kev decision-model/publish.py \
    /mnt/kev/runs/banking77-v1 --data /mnt/kev/banking77 --repo <you>/kev-banking77
```

Add `--detach` to get the Job id at once, then follow it with `hf jobs logs <id>` (root README, "Watch, cost and stop a
Job").

### prepare.py options

| option | default | meaning |
|---|---|---|
| `--dataset`, `--revision`, `--config` | required, current head, none | the source dataset; the revision is resolved to one commit and recorded |
| `--text-column`, `--label-column` | required | the text Kev reads (the "state") and the answer |
| `--question` | required | the instruction Kev answers. Write it as you would ask a person. |
| `--descriptions` | none | a JSON file `{label: description}`; Kev reads the descriptions with the labels |
| `--train-split`, `--dev-split` | `train`, `test` | `--dev-split none` takes the development records from the training split |
| `--n-train`, `--n-dev` | 2000, 500 | records of each (0 = all). Training takes about 1 s per record per epoch on `a100-large`; set `--timeout` from it. |
| `--max-chars` | 2000 | cut longer texts; Kev drops states longer than 384 tokens |
| `--out` | required | output folder, on the mounted bucket |

`prepare.json` in the output folder records the source, the commit, the question, the question key, the labels with
their descriptions, the counts and the smallest and largest number of training records per label. Ask the trained
model with the same question and labels. `--dev-split` must differ from `--train-split`; use `--dev-split none
--n-dev <n>` to hold records out of the training split. The label column must hold one label per row.

### train-kev.py

The output folder must be new or empty, on the mounted bucket. It gets `checkpoint/`, `finetuned-development/` and
`baseline-development/` (Kev's benchmark reports for both models), `finetuned-development/calibration.json` and
`run.json`. Options: `--epochs` (default 2), `--lr`, `--max-steps`, `--init-from`. Its full documentation is in the
[uv-scripts classification README](https://huggingface.co/datasets/uv-scripts/classification).

### publish.py options

| option | default | meaning |
|---|---|---|
| `run_dir` | required | train-kev.py's output folder |
| `--data` | required | prepare.py's output folder |
| `--repo` | required | the model repo to create or update |
| `--license` | `apache-2.0` (Kev-4B's) | the licence id in the card. Check the training data's licence too. |
| `--public` | off | create the repo public. The script refuses to push into an existing public repo without it. |

## Use the model

Serve it with the [Kev repository](https://github.com/jaredpalmer/kev), as Kev-4B:

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run <you>/kev-banking77 --port 8008
```

Ask with the same instruction and the same labels as in `prepare.json`. The model card gives the fitted temperature
(`KEV_TEMPERATURE`) for calibrated probabilities.

## Credits and licences

Kev and Kev-4B are by Jared Palmer (Apache-2.0); Kev-4B is a LoRA adapter on Qwen3.5-4B-Base. The training and
evaluation code is Kev's; `train-kev.py` only installs it at a pinned commit and runs it. Banking77 is by PolyAI, under
CC BY 4.0: a model trained on it must credit it.
