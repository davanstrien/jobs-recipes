# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=4.0", "huggingface_hub>=1.0"]
# ///
"""Turn a labelled Hub dataset into Kev records: one choice question per row, for train.py (Kev's trainer).

Each row becomes {"state": <text>, "questions": {<name>: {"type": "choice", "instructions": <question>,
"criteria": {<label>: <description or null>, ...}, "label": <the row's label>}}}. Kev's records format and the checks
are in train-kev.py (uv-scripts/classification).

Output: <out>/train.jsonl, <out>/development.jsonl and <out>/prepare.json (the source, the commit and the counts).
Run it on a CPU Job with a bucket mounted at the output, or locally and then `hf buckets sync`.

    hf jobs uv run --flavor cpu-basic --timeout 30m --secrets HF_TOKEN -v hf://buckets/<you>/<bucket>:/mnt/kev \\
        decision-model/prepare.py --dataset PolyAI/banking77 --text-column text --label-column label \\
        --question "Which customer-support intent does this message express?" --out /mnt/kev/banking77
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--revision", default=None, help="commit, branch or tag (default: the current head; resolved and recorded)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--text-column", required=True)
    ap.add_argument("--label-column", required=True, help="a ClassLabel or a string column")
    ap.add_argument("--question", required=True, help='the instruction Kev answers, e.g. "Which team should handle this?"')
    ap.add_argument("--question-name", default="label", help="the key of the question in each record")
    ap.add_argument("--descriptions", default=None, help="optional JSON file {label: description}; default: no descriptions")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--dev-split", default="test", help="split for development records; 'none' takes them from train")
    ap.add_argument("--n-train", type=int, default=2000, help="training records (0 = all)")
    ap.add_argument("--n-dev", type=int, default=500, help="development records (0 = all)")
    ap.add_argument("--max-chars", type=int, default=2000, help="cut longer texts (Kev drops states over 384 tokens)")
    ap.add_argument("--min-chars", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from datasets import ClassLabel, load_dataset
    from huggingface_hub import HfApi
    sha = HfApi().dataset_info(a.dataset, revision=a.revision).sha
    ds = load_dataset(a.dataset, a.config, revision=sha)
    feature = ds[a.train_split].features[a.label_column]
    name = feature.int2str if isinstance(feature, ClassLabel) else str

    def rows(split):
        out = []
        for r in ds[split]:
            text = (r[a.text_column] or "").strip()
            if len(text) >= a.min_chars and r[a.label_column] is not None:
                out.append((text[:a.max_chars], name(r[a.label_column])))
        return out

    rng = random.Random(a.seed)
    train = rows(a.train_split)
    rng.shuffle(train)
    if a.dev_split == "none":
        n_dev = a.n_dev or len(train) // 5
        dev, train = train[:n_dev], train[n_dev:]
    else:
        dev = rows(a.dev_split)
        rng.shuffle(dev)
    train = train[:a.n_train] if a.n_train else train
    dev = dev[:a.n_dev] if a.n_dev else dev

    labels = feature.names if isinstance(feature, ClassLabel) else sorted({lab for _, lab in train})
    if len(labels) > 255:
        raise SystemExit(f"{len(labels)} labels: Kev's choice questions take at most 255 options")
    desc = json.loads(Path(a.descriptions).read_text()) if a.descriptions else {}
    criteria = {lab: desc.get(lab) for lab in labels}
    unseen = sorted({lab for _, lab in dev} - set(labels))
    if unseen:
        raise SystemExit(f"development labels not in the label set: {unseen[:10]}")

    a.out.mkdir(parents=True, exist_ok=True)
    for fname, recs in (("train.jsonl", train), ("development.jsonl", dev)):
        with (a.out / fname).open("w") as f:
            for text, lab in recs:
                q = {"type": "choice", "instructions": a.question, "criteria": criteria, "label": lab}
                f.write(json.dumps({"state": text, "questions": {a.question_name: q}}) + "\n")
    counts = Counter(lab for _, lab in train)
    meta = {"dataset": a.dataset, "revision": sha, "config": a.config, "text_column": a.text_column,
            "label_column": a.label_column, "question": a.question, "labels": len(labels), "train": len(train),
            "development": len(dev), "max_chars": a.max_chars, "seed": a.seed,
            "train_per_label_min": min(counts.values()), "train_per_label_max": max(counts.values()),
            "labels_without_train_records": [lab for lab in labels if lab not in counts]}
    (a.out / "prepare.json").write_text(json.dumps(meta, indent=1))
    print("RESULT", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
