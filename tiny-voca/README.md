# tiny-voca on Hugging Face Jobs

Train [tiny-voca](https://github.com/n4ze3m/tiny-voca), a 4.5M-parameter English text-to-speech model with two voices by
Muhammed Nazeem, from scratch on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs). One script
runs the upstream README's training steps: data preparation, alignment, training, evaluation and ONNX export. The state is
kept in a [Hugging Face bucket](https://huggingface.co/docs/hub/storage-buckets), so a stopped run can resume.

Cost: about $15 for a full run (5.5 h on one `rtx-pro-6000` at $2.75 per hour; `hf jobs hardware` has current prices).
The upstream project's training run cost $2.89 on a rented RTX 4090; "Cost compared with upstream" in Notes explains the difference.

## Results

One full run (`README recipe, one phase`), compared with the released checkpoint. Both models were scored by the same
`tiny_voca.evaluate` call, on the same clips.

| split | model | WER | duration ratio | pause ratio | mel DTW |
|---|---|---|---|---|---|
| test (594 clips) | ours | 1.26 % | 0.986 | 1.13 | 0.733 |
| | released | 1.36 % | 0.956 | 1.05 | 0.731 |
| | source recordings | 1.14 % | | | |
| validation (600 draws) | ours | 0.75 % | 1.002 | 1.15 | 0.741 |
| | released | 0.82 % | 0.970 | 1.06 | 0.741 |
| | source recordings | 0.64 % | | | |

Per voice, on the test split:

| voice | WER ours | WER released | WER source |
|---|---|---|---|
| F2 (296 clips) | 1.27 % | 1.31 % | 1.18 % |
| M2 (298 clips) | 1.26 % | 1.41 % | 1.10 % |

What the numbers show:

- WER and mel distance are close to the released checkpoint's. The differences are within the evaluation noise: a first
  evaluation of the same two checkpoints, in another Job, gave WERs up to 0.17 points different (for example 1.33 % for
  the released model and 0.98 % for the source recordings on the test split).
- Our model's pauses are about 8 % longer than the released model's, on every split. Its total duration is closer to the
  source recordings.
- This is one run with one seed. The upstream evaluation notes that these numbers support a listening test and do not
  replace one. The evaluation audio of both models is in the bucket (see [Output](#output)), so you can listen to the same
  clips.

How the metrics are measured (upstream `src/tiny_voca/evaluate.py`):

- WER: Whisper small.en (faster-whisper) reads the audio back. The WER is the mean over clips, after Whisper's English text
  normaliser. "Source recordings" is the same read-back of the dataset's own audio.
- Duration ratio: length of the model's audio divided by the length of the source recording. Pause ratio: the model's
  predicted pause frames divided by the pause frames in the source alignment. 1.0 is the same.
- Mel DTW: distance between the model's mel spectrogram and the source recording's, after time alignment. Lower is closer.
- Validation draws 300 clips per voice from 359 (F2) and 355 (M2), so some clips are scored twice. Both models get the
  same draws.

### How this run differs from the released checkpoint

The released checkpoint was trained in two phases: 66,000 steps on the first data batch with eSpeak annotations, then
38,737 steps on the current data and annotations. This recipe runs the single 104,000-step command from the upstream
README on the current data. Our data stage keeps the same clips as the released annotations (dataset `annotations/v4`):
the same 16,018 train, 714 validation and 594 test clip ids.

The tested run used `RUN=repro-rtx-pro-6000`. In that run, the upload of the evaluation audio (about 7,200 small files)
failed with a rate-limit error (HTTP 429). A second `run.sh eval` Job finished and uploaded everything. The tables above are from that second Job: they match the
reports in the bucket. The script now
retries every bucket transfer and packs the evaluation audio into one tar file per model. A smoke test of the current
script passed (`STAGE-DONE all`; the evaluation upload took 15 s).

## Before you start

- A Hugging Face account that can run Jobs (a paid plan or credits), and the `hf` CLI (`pip install -U huggingface_hub`),
  logged in with a token that can write.
- A bucket for the state:

  ```bash
  hf buckets create <you>/<bucket> --private
  ```

## Run it

Run the commands from the root of this repository. The `-v ./tiny-voca:/scripts:ro` flag puts this folder into the Job.
Give every Job a `--timeout`: the default is 30 minutes.

**1. Smoke test** (about 15 min, under $1). It uses 2 input files from each data batch and trains 300 steps from step 0.
It checks the install, the bucket and every stage.

```bash
hf jobs run --flavor rtx-pro-6000 --timeout 1h --secrets HF_TOKEN \
    -e TV_BUCKET=hf://buckets/<you>/<bucket>/tiny-voca -e SMOKE=1 \
    -v ./tiny-voca:/scripts:ro ghcr.io/astral-sh/uv:python3.12-bookworm bash /scripts/run.sh all
```

The log must end with `STAGE-DONE all`.

**2. Full run**, one stage per Job (recommended: you can check each stage before you pay for the next one).

```bash
job() {  # job <timeout> <stage>
  hf jobs run --flavor rtx-pro-6000 --timeout "$1" --secrets HF_TOKEN -e TV_BUCKET=hf://buckets/<you>/<bucket>/tiny-voca \
      -v ./tiny-voca:/scripts:ro ghcr.io/astral-sh/uv:python3.12-bookworm bash /scripts/run.sh "$2"
}
job 1h data
job 8h train
job 2h eval
```

`hf jobs run` streams the log and waits for the Job to finish. Check the log before you start the next stage.

Or the whole chain in one Job, with `--timeout 8h` and `run.sh all`.

### Stages

| stage | what it does | time on RTX PRO 6000 | timeout |
|---|---|---|---|
| data | downloads the dataset (pinned revision), computes features with one process per CPU, aligns phonemes with wav2vec2, checks the clip counts, saves `features/` and `annotations/` | 21 min | 1h |
| train | `tiny_voca.train`, 104,000 steps (GAN from step 8,000); saves `runs/<run>/` to the bucket every 10 minutes | 3 h 50 min | 8h (caps the cost at about $22) |
| eval | `tiny_voca.evaluate` for our checkpoint and for the released one, the ONNX export, then the upload | 1 h 10 min, plus the upload | 2h |

### Settings

| variable | default | meaning |
|---|---|---|
| `TV_BUCKET` | required | bucket folder for the state, e.g. `hf://buckets/<you>/<bucket>/tiny-voca` |
| `RUN` | `repro` (`smoke` with `SMOKE=1`) | run name; the state is in `$TV_BUCKET/$RUN`. `released` is reserved. |
| `SMOKE` | `0` | `1`: 2 input files per data batch, `STEPS=300`, `PER_VOICE=4` |
| `STEPS` | `104000` (`300` with `SMOKE=1`) | training steps |
| `PER_VOICE` | `300` (`4` with `SMOKE=1`) | evaluation clips per voice and split |
| `FORCE_DATA` | `0` | `1`: prepare the data again even if the bucket has it |
| `TV_COMMIT`, `DIZIONE_COMMIT`, `DS_REV`, `MODEL_REV` | pinned | code, phonemizer, dataset and released-weights revisions |

### Resume

If a Job stops (for example at its timeout), start the stage again with the same `RUN`:

- `run.sh train` resumes from `runs/<run>/last.pt` in the bucket. If training is already done, it skips to the end.
- `run.sh data` and `run.sh all` reuse the data in the bucket. Do not prepare the data again for a run that has started
  training: new annotations have a new fingerprint, and the training cannot resume from them.
- A smoke test (`SMOKE=1`) always trains from step 0.

Use a new `RUN` for each new experiment.

### Output

Everything is in `$TV_BUCKET/$RUN`:

| path | content |
|---|---|
| `runs/<run>/inference.pt` | the trained model (18 MB) |
| `runs/<run>/last.pt`, `step-*.pt` | training checkpoints |
| `runs/<run>/metrics.jsonl` | training metrics |
| `eval/<run>/report.json`, `eval/released/report.json` | evaluation reports of both models |
| `eval/<run>/audio.tar`, `eval/released/audio.tar` | the evaluation audio of both models (generated and source clips) |
| `export/tiny-voca-<run>.onnx` | the ONNX model |

The eval log also prints one `RESULT` line per model, with the step, the test WER and the ratios.

### Use the model

```bash
hf buckets cp hf://buckets/<you>/<bucket>/tiny-voca/repro/runs/repro/inference.pt .
pip install "tiny-voca[own] @ git+https://github.com/n4ze3m/tiny-voca@903f940" \
    "dizione @ git+https://github.com/n4ze3m/dizione@8cd889e"
hf download n4ze3m/tiny-voca --revision 5fe06641d0afecb3c04bc9faca2cf64d9b2b1600 --include "phonemizer/*" --local-dir weights
DIZIONE_DATA=weights/phonemizer python -m tiny_voca.infer --checkpoint inference.pt --voice M2 \
    --text "Hello! This is tiny voca." --output hello.wav
```

The phonemizer must be the one the model was trained with: `tiny_voca.infer` refuses a checkpoint with another front-end
fingerprint. The ONNX file works in the same way as `onnx/model.onnx` in the
[model repository](https://huggingface.co/n4ze3m/tiny-voca).

## Notes

- **Cost compared with upstream.** The tiny-voca README gives $9.56 of GPU time for the whole project on one rented
  RTX 4090, of which $2.89 was the 7.7 h training run (about $0.38 per GPU hour). Here the training takes 3 h 50 min on
  `rtx-pro-6000`, about $10.50. In 10-minute training tests on other flavors, the GPU was 70-88 % busy on each, so a
  cheaper flavor was slower by about as much as it was cheaper:

  | flavor | $/h | training steps/s | full training (estimated) |
  |---|---|---|---|
  | t4-small | 0.40 | 0.87 | 32 h, $13 |
  | l4x1 | 0.80 | 1.46 | 19 h, $15 |
  | a10g-small | 1.00 | 2.12 | 13 h, $13 |
  | l40sx1 | 1.80 | 4.06 | 6.8 h, $12 |
  | rtx-pro-6000 | 2.75 | 7.18 | 3.8 h, $10.50 (measured) |
  | h200 | 5.00 | 7.74 | 3.5 h, $18 |

  bf16 autocast made training slower (3.8 against 6.8 steps/s), and `torch.compile` compiled again for each batch length.
- **Other GPUs.** The full run was tested on `rtx-pro-6000` only. The other flavors in the table above ran 10 minutes of
  training each, not a full run. The evaluation runs Whisper on the CPU.
- **Python packages.** The tiny-voca code, the phonemizer, the dataset and the released weights are pinned. torch and the
  other packages are not, so a later install can move the clip counts or the scores a little.
- **Experimental: new voices.** `design_voice.py` makes voice designs with Qwen3-TTS VoiceDesign, the method that made the
  two released voices. It writes one clip per design and seed:

  ```bash
  hf jobs run --flavor l4x1 --timeout 1h --secrets HF_TOKEN -v ./tiny-voca:/scripts:ro \
      -v hf://buckets/<you>/<bucket>/tiny-voca/design:/out ghcr.io/astral-sh/uv:python3.12-bookworm \
      bash -c 'uv pip install -q --system qwen-tts soundfile torch && python /scripts/design_voice.py --designs /scripts/designs.example.json --out /out/try1'
  ```

  The steps that render training audio in a new voice and train with it are not part of this recipe yet.

## Credits and licences

- tiny-voca is by Muhammed Nazeem ([n4ze3m](https://github.com/n4ze3m)). Code and weights: Apache-2.0. Dataset
  ([n4ze3m/tiny-voca](https://huggingface.co/datasets/n4ze3m/tiny-voca)): CC BY 4.0. A model that you train and publish
  with this recipe uses the dataset, so credit it as its card asks.
- dizione, the phonemizer: Apache-2.0; its dictionary comes from CMUdict.
- The recipe also downloads [facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)
  (alignment), Whisper small.en (evaluation) and, for `design_voice.py`,
  [Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign). Each has its own licence.

Cite tiny-voca as:

```bibtex
@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```
