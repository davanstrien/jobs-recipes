# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.0"]
# ///
"""Push a fine-tuned Kev checkpoint from train-kev.py's output folder to a model repo, with a model card.

The card records the data (from prepare.py's prepare.json), the pinned Kev and Kev-4B versions (from run.json), and
the development results of the fine-tuned and the released model, with the calibration report. The checkpoint has the
same layout as jaredpalmer/kev-4b (LoRA adapter, pointer head, tokenizer), so Kev's own commands load it.

    hf jobs uv run --flavor cpu-basic --timeout 30m --secrets HF_TOKEN -v hf://buckets/<you>/<bucket>:/mnt/kev \\
        decision-model/publish.py /mnt/kev/runs/banking77-v1 --data /mnt/kev/banking77 --repo <you>/kev-banking77
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path


def pct(x):
    return f"{100 * x:.1f} %"


def paired(run_dir, data_dir, seed=0, n_boot=10_000):
    """accuracy difference fine-tuned minus released on the same development records, with a 95 % bootstrap interval;
    None when the prediction files are missing. Kev ids are custom/<line number in development.jsonl>."""
    import random
    files = [run_dir / d / "predictions.jsonl" for d in ("finetuned-development", "baseline-development")]
    if not all(f.exists() for f in files):
        return None
    gold = {}
    for i, line in enumerate((data_dir / "development.jsonl").read_text().splitlines()):
        questions = json.loads(line)["questions"]
        if len(questions) != 1:   # prepare.py writes one question per record
            return None
        gold[i] = next(iter(questions.values()))["label"]
    correct = []
    for f in files:
        c = {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if len(r["prediction"]["probabilities"]) != 1:
                return None
            probs = next(iter(r["prediction"]["probabilities"].values()))
            i = int(r["id"].rsplit("/", 1)[1])
            c[i] = max(probs, key=probs.get) == gold[i]
        correct.append(c)
    ids = sorted(set(correct[0]) & set(correct[1]))
    d = [int(correct[0][i]) - int(correct[1][i]) for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return {"n": len(d), "diff": sum(d) / len(d), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot)],
            "better": d.count(1), "worse": d.count(-1)}


def card(a, run, prep, cal, pair):
    arms = (cal or {}).get("arms", {})
    oof = arms.get("workload_oof")
    t = (cal or {}).get("workload_temperature")
    ft, rel = run["finetuned"], run["released"]
    rows = [f"| fine-tuned (this model) | {pct(ft['acc'])} | {ft['ece']:.3f} |",
            f"| released Kev-4B | {pct(rel['acc'])} | {rel['ece']:.3f} |"]
    temp = ""
    if oof and t:
        rows.append(f"| fine-tuned, temperature fitted on other folds (out-of-fold check) | {pct(oof['acc'])} | {oof['ece']:.3f} |")
        if oof["ece"] < ft["ece"]:
            temp = (f"\nThe checkpoint runs at temperature 1.0. Kev's calibration step fitted one temperature, {t:.2f}, on all "
                    "development records. Row 3 checks the method out-of-fold: each fold's temperature is fitted on the other "
                    f"folds. It lowers the calibration error from {ft['ece']:.3f} to {oof['ece']:.3f}. To use the fitted value, "
                    f"set `KEV_TEMPERATURE={t:.2f}` when you serve the model with Kev's commands.\n")
        else:
            temp = ("\nThe checkpoint runs at temperature 1.0. A fitted temperature did not lower the calibration error "
                    "out-of-fold (row 3), so keep the default.\n")
    diff = (f"Fine-tuned minus released, on the same {pair['n']} records: {100 * pair['diff']:+.1f} points (95 % bootstrap "
            f"interval {100 * pair['lo']:+.1f} to {100 * pair['hi']:+.1f}); {pair['better']} answers became correct and "
            f"{pair['worse']} became wrong.\n\n") if pair else ""
    init = run["init_from"]
    return f"""---
base_model: {init.split('@')[0]}
library_name: peft
license: {a.license}
datasets:
- {prep['dataset']}
pipeline_tag: text-classification
tags:
- kev
- decision-model
- jobs-recipes
---

# {a.repo.split('/')[-1]}

[Kev-4B]({'https://huggingface.co/' + init.split('@')[0]}) fine-tuned to answer one question about a text:

> {prep['question']}

It chooses one of {prep['labels']} labels from [{prep['dataset']}](https://huggingface.co/datasets/{prep['dataset']}) and
gives a probability for each.

Trained on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) with the
[decision-model recipe](https://github.com/davanstrien/jobs-recipes/tree/main/decision-model): {run['seconds'] // 60} min
on one GPU Job, from a labelled Hub dataset to this repo.

## Results

On {ft['n']} development records (`{prep['dataset']}`, held out from training):

| model | accuracy | calibration error (ECE) |
|---|---|---|
{chr(10).join(rows)}

{diff}One training run, one seed.
{temp}
## Training

| | |
|---|---|
| data | [{prep['dataset']}](https://huggingface.co/datasets/{prep['dataset']}) at `{prep['revision']}`; {prep['train']:,} training and {prep['development']:,} development records; texts cut at {prep['max_chars']:,} characters |
| started from | `{init}` |
| Kev code | commit `{run['kev_commit']}` |
| base model | Qwen/Qwen3.5-4B-Base at `{run['base_revision']}` |
| run time | {run['seconds'] // 60} min on one GPU Job (training and both evaluations) |

Trained with `train-kev.py` from [uv-scripts/classification](https://huggingface.co/datasets/uv-scripts/classification)
on Hugging Face Jobs. `prepare.json`, `run.json` and the Kev reports in `reports/` are in this repo.

## Use

Serve it with the [Kev repository](https://github.com/jaredpalmer/kev), as Kev-4B:

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run {a.repo} --port 8008
```

Then ask the question with the same instruction and the same labels as in training (see `prepare.json`).

## Licence

Read the licences of the base model ({init.split('@')[0]}) and of the training data ({prep['dataset']}) before you
use or share this model. The training data's licence can ask for attribution.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="train-kev.py's output folder (has run.json and checkpoint/)")
    ap.add_argument("--data", type=Path, required=True, help="prepare.py's output folder (has prepare.json)")
    ap.add_argument("--repo", required=True, help="model repo to create or update")
    ap.add_argument("--license", default="apache-2.0", help="licence id for the card (default: Kev-4B's, apache-2.0)")
    ap.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    a = ap.parse_args()

    run = json.loads((a.run_dir / "run.json").read_text())
    # the --data folder must hold the development records that were benchmarked (Kev records their SHA-256)
    import hashlib
    dev_sha = hashlib.sha256((a.data / "development.jsonl").read_bytes()).hexdigest()
    for d in ("finetuned-development", "baseline-development"):
        report = a.run_dir / d / "report.json"
        if report.exists() and json.loads(report.read_text()).get("suite_sha256") != dev_sha:
            raise SystemExit(f"{a.data}/development.jsonl is not the file that {d} was benchmarked on: "
                             "use the prepare.py folder of this training run")
    if run.get("status") != "completed":
        raise SystemExit(f"{a.run_dir}/run.json has status {run.get('status')!r}: the training run did not finish")
    prep = json.loads((a.data / "prepare.json").read_text())
    cal_path = a.run_dir / "finetuned-development" / "calibration.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else None

    from huggingface_hub import HfApi
    api = HfApi()
    if not a.public and api.repo_exists(a.repo) and not api.model_info(a.repo).private:
        raise SystemExit(f"{a.repo} exists and is public: pass --public to push into it, or choose another name")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        shutil.copytree(a.run_dir / "checkpoint", out, dirs_exist_ok=True)
        (out / "reports").mkdir()
        for name in ("finetuned-development/report.json", "finetuned-development/calibration.json",
                     "baseline-development/report.json"):
            if (a.run_dir / name).exists():
                shutil.copy(a.run_dir / name, out / "reports" / name.replace("/", "-"))
        shutil.copy(a.run_dir / "run.json", out / "run.json")
        shutil.copy(a.data / "prepare.json", out / "prepare.json")
        (out / "README.md").write_text(card(a, run, prep, cal, paired(a.run_dir, a.data)))
        api.create_repo(a.repo, private=not a.public, exist_ok=True)
        api.upload_folder(repo_id=a.repo, folder_path=out, commit_message="Kev fine-tune from jobs-recipes/decision-model")
    print("RESULT", json.dumps({"repo": a.repo, "finetuned_acc": run["finetuned"]["acc"], "released_acc": run["released"]["acc"]}))
    print(f"pushed https://huggingface.co/{a.repo}", flush=True)


if __name__ == "__main__":
    main()
