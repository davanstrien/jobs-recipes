"""Render the tiny-voca sentences in one new voice: the training audio for a third voice.

A port of scripts/generate_audio.py from tiny-voca (github.com/n4ze3m/tiny-voca, Apache-2.0, by Muhammed Nazeem), the script
that rendered the dataset's second batch. It keeps the upstream method: Qwen3-TTS 1.7B Base clones a VoiceDesign reference
clip, batches of eight, Whisper small.en reads each clip back, a clip fails above 15 % normalized WER or with an implausible
duration, speaking rate or silence, a failed clip gets one more try, and accepted clips are peak-normalised to 0.95.
Differences: one voice, given on the command line; the sentences are those of an existing voice in the pinned dataset
(the same ids, texts and splits); the work folder can be mirrored to a bucket folder, so a stopped Job can resume.

Resumable: every clip is recorded in a SQLite file; run the same command again.
Output, under --root: audio/<text_id>_<voice>.flac, manifest.jsonl (accepted clips, in the JSONL form tiny_voca.prepare reads),
review.jsonl (rejected clips), summary.json (counts, speed, WER of the new voice and of the source voice on the same sentences).

    python render_voice.py --voice S1 --reference refs/S1.wav --root /work/S1 --mirror /out/S1 --limit 50
"""

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DATASET, DS_REV = "n4ze3m/tiny-voca", "63b6b89622e984d7c7a0799ca4b4c443dd5f48b2"   # the revision run.sh uses
COLUMNS = ["id", "text_id", "voice_id", "text", "split", "category", "origin", "source_id", "source_license", "source_url",
           "normalized_wer"]


def load_texts(source_voice, revision, cache):
    """Rows of one voice in the dataset, without the audio column (Parquet reads only the columns it needs)."""
    if cache.exists():
        return [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = [f for pattern in ("data/*.parquet", "batch2/data/*.parquet") for f in fs.glob(f"datasets/{DATASET}@{revision}/{pattern}")]

    def read(path):
        names = pq.ParquetFile(path, filesystem=fs).schema_arrow.names
        return pq.read_table(path, columns=[c for c in COLUMNS if c in names], filesystem=fs).to_pylist()

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        rows = [r for part in pool.map(read, files) for r in part if r["voice_id"] == source_voice]
    rows.sort(key=lambda r: r["id"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", required=True, help="Id of the new voice, e.g. S1. Clip ids are <text_id>_<voice>.")
    parser.add_argument("--reference", type=Path, required=True, help="Reference clip of the new voice (from design_voice.py)")
    parser.add_argument("--reference-text", help="Text of the reference clip (default: from voices.json next to the clip)")
    parser.add_argument("--root", type=Path, required=True, help="Local work folder")
    parser.add_argument("--mirror", type=Path, help="Folder (e.g. a mounted bucket) to copy the work to, and to resume from")
    parser.add_argument("--mirror-every", type=float, default=600, help="Seconds between copies to --mirror")
    parser.add_argument("--source-voice", default="F2", help="Render the sentences of this dataset voice")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, help="Only N sentences, drawn at random with --seed (smoke test)")
    parser.add_argument("--max-minutes", type=float, help="Stop generating after this time (keep it below the Job timeout)")
    parser.add_argument("--batch", type=int, default=8, help="Sentences per generation call (upstream: 8)")
    parser.add_argument("--wer-threshold", type=float, default=0.15, help="Read-back WER above which a clip fails (upstream: 0.15)")
    parser.add_argument("--revision", default=DS_REV)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    from faster_whisper import WhisperModel
    from jiwer import wer
    from qwen_tts import Qwen3TTSModel
    from scipy.signal import resample_poly
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    if args.reference_text is None:
        args.reference_text = json.loads((args.reference.parent / "voices.json").read_text())[args.reference.stem]["text"]
    if args.mirror and (args.mirror / "state.sqlite").exists() and not (args.root / "state.sqlite").exists():
        shutil.copytree(args.mirror, args.root, dirs_exist_ok=True)   # resume a stopped Job
        print(f"resumed from {args.mirror}", flush=True)
    audio = args.root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.root / "state.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS clips(id TEXT PRIMARY KEY, status TEXT, attempts INTEGER, payload TEXT)")
    db.commit()

    rows = [r for r in load_texts(args.source_voice, args.revision, args.root / f"texts-{args.source_voice}.jsonl") if r["split"] in args.splits]
    if args.limit:
        rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))
    rows.sort(key=lambda r: (len(r["text"]), r["id"]))   # similar lengths in a batch, as upstream
    jobs = [{"id": f"{r['text_id']}_{args.voice}", "row": r} for r in rows]
    started, mirrored, counter = time.time(), 0.0, {"clips": 0, "since": None}

    def summary(stage):
        payloads = [(s, json.loads(p)) for s, p in db.execute("SELECT status, payload FROM clips")]
        done = [p for s, p in payloads]
        reasons = {}
        for p in done:
            for reason in p["review_reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        source_wer = {r["text_id"]: r.get("normalized_wer") for r in rows}
        pairs = [(p["normalized_wer"], source_wer[p["text_id"]]) for p in done if p["asr_transcript"] and source_wer.get(p["text_id"]) is not None]
        quantiles = lambda xs: {q: round(float(np.quantile(xs, q / 100)), 4) for q in (50, 90, 95)} if xs else {}
        gen = sum(p["generation_seconds"] / p["batch_size"] for p in done)
        value = {"stage": stage, "voice": args.voice, "target_clips": len(jobs),
                 "counts": {s: sum(1 for t, _ in payloads if t == s) for s in ("accepted", "review")}, "review_reasons": reasons,
                 "accepted_audio_hours": round(sum(p["duration_seconds"] for s, p in payloads if s == "accepted") / 3600, 3),
                 "elapsed_seconds_this_run": round(time.time() - started), "generation_gpu_seconds": round(gen),
                 "clip_attempts_this_run": counter["clips"],
                 "clips_per_minute": round(60 * counter["clips"] / (time.time() - counter["since"]), 1) if counter["clips"] else None,
                 "wer_new_voice": quantiles([a for a, _ in pairs]), "wer_source_voice_same_texts": quantiles([b for _, b in pairs]),
                 "source_voice": args.source_voice, "wer_threshold": args.wer_threshold, "updated": time.time()}
        (args.root / "summary.json").write_text(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)

    def mirror(force=False):
        nonlocal mirrored
        if not args.mirror or not (force or time.time() - mirrored > args.mirror_every):
            return
        since, mirrored = mirrored, time.time()
        (args.mirror / "audio").mkdir(parents=True, exist_ok=True)
        for f in audio.iterdir():   # copy the clips written (or written again by a retry) since the last copy
            if f.stat().st_mtime >= since - 5 or not (args.mirror / "audio" / f.name).exists():
                shutil.copyfile(f, args.mirror / "audio" / f.name)
        backup = sqlite3.connect(args.root / "state.backup.sqlite")   # a consistent copy, written in one piece
        db.backup(backup)
        backup.close()
        shutil.copyfile(args.root / "state.backup.sqlite", args.mirror / "state.sqlite")
        for name in ("manifest.jsonl", "review.jsonl", "summary.json", f"texts-{args.source_voice}.jsonl"):
            if (args.root / name).exists():
                shutil.copyfile(args.root / name, args.mirror / name)

    cpus = int(os.environ.get("CPU_CORES") or os.cpu_count())
    torch.set_num_threads(max(1, cpus - 1))
    model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    prompt = model.create_voice_clone_prompt(ref_audio=str(args.reference), ref_text=args.reference_text, x_vector_only_mode=False)[0]
    asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=max(1, cpus - 1))   # the read-back runs on the CPU, beside the GPU
    normalize = EnglishTextNormalizer({"cancelled": "canceled", "colour": "color", "favourite": "favorite", "centre": "center"})
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    futures = []

    def analyze(batch, waves, rate, attempt, seed, batch_id, seconds):
        records = []
        for job, wave in zip(batch, waves):
            row, w = job["row"], np.asarray(wave)
            duration, peak, reasons = len(w) / rate, float(np.max(np.abs(w))) if len(w) else 0.0, []
            if not np.isfinite(w).all() or peak < 1e-5:
                reasons.append("invalid_or_silent_audio")
            if duration < 0.5 or duration > 35:
                reasons.append("duration")
            if duration > 0 and not 0.5 <= len(row["text"].split()) / duration <= 7:
                reasons.append("speaking_rate")
            if len(w) and float(np.mean(np.abs(w) < 1e-4)) > 0.8:
                reasons.append("excessive_silence")
            transcript, error = "", 1.0
            if not reasons:
                segments, _ = asr.transcribe(resample_poly(w.astype("float32"), 2, 3), language="en", beam_size=3, condition_on_previous_text=False)
                transcript = " ".join(s.text.strip() for s in segments)
                error = wer(normalize(row["text"]), normalize(transcript) or "<empty>")
                if error > args.wer_threshold:
                    reasons.append("asr_mismatch")
            if np.isfinite(w).all() and peak > 0:
                sf.write(audio / (job["id"] + ".flac"), w * (0.95 / peak), rate, subtype="PCM_16")
            payload = {"id": job["id"], "text_id": row["text_id"], "voice_id": args.voice, "text": row["text"], "split": row["split"],
                       "category": row.get("category"), "origin": row.get("origin"), "source_id": row.get("source_id"),
                       "source_license": row.get("source_license"), "source_url": row.get("source_url"), "teacher_model": MODEL,
                       "reference_audio": args.reference.name, "audio": f"audio/{job['id']}.flac", "duration_seconds": round(duration, 3),
                       "generation_seed": seed, "generation_batch": batch_id, "attempt": attempt, "generation_seconds": seconds,
                       "batch_size": len(batch), "asr_transcript": transcript, "normalized_wer": round(float(error), 4),
                       "review_reasons": reasons}
            records.append((job["id"], "review" if reasons else "accepted", attempt, json.dumps(payload)))
        return records

    def collect(future):
        records = future.result()
        counter["clips"] += len(records)
        db.executemany("INSERT INTO clips(id, status, attempts, payload) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                       "status=excluded.status, attempts=excluded.attempts, payload=excluded.payload", records)
        db.commit()

    def generate(batch, attempt, batch_id):
        seed = args.seed + batch_id + attempt * 100000
        torch.manual_seed(seed)
        start = time.time()
        try:
            waves, rate = model.generate_voice_clone(text=[j["row"]["text"] for j in batch], language=["English"] * len(batch),
                                                     voice_clone_prompt=[prompt] * len(batch), max_new_tokens=600)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            generate(batch[:middle], attempt, batch_id)
            generate(batch[middle:], attempt, batch_id + 50000)
            return
        futures.append(pool.submit(analyze, batch, waves, rate, attempt, seed, batch_id, round(time.time() - start, 3)))
        if len(futures) >= 2:
            collect(futures.pop(0))

    stopped, counter["since"] = False, time.time()
    for attempt in (1, 2):
        existing = {uid: (status, n) for uid, status, n in db.execute("SELECT id, status, attempts FROM clips")}
        pending = [j for j in jobs if (attempt == 1 and j["id"] not in existing)
                   or (attempt == 2 and j["id"] in existing and existing[j["id"]][0] == "review" and existing[j["id"]][1] < 2)]
        size = args.batch if attempt == 1 else 1
        for start in range(0, len(pending), size):
            if (args.root / "STOP").exists() or (args.max_minutes and time.time() - started > 60 * args.max_minutes):
                stopped = True
                break
            generate(pending[start:start + size], attempt, start // args.batch if attempt == 1 else start + 20000)
            if (start // size) % 16 == 15:
                summary("generating")
                mirror()
        while futures:
            collect(futures.pop(0))
        if stopped:
            break
    pool.shutdown(wait=True)
    accepted = [json.loads(p) for (p,) in db.execute("SELECT payload FROM clips WHERE status='accepted' ORDER BY id")]
    (args.root / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted), encoding="utf-8")
    (args.root / "review.jsonl").write_text("".join(p + "\n" for (p,) in db.execute("SELECT payload FROM clips WHERE status!='accepted' ORDER BY id")), encoding="utf-8")
    summary("paused" if stopped else "complete")
    mirror(force=True)


if __name__ == "__main__":
    main()
