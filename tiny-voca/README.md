# tiny-voca on Hugging Face Jobs

Train [tiny-voca](https://github.com/n4ze3m/tiny-voca), a 4.5M-parameter English text-to-speech model with two voices, from
scratch on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs). One script runs the upstream
README's training steps: data preparation, alignment, training, evaluation and ONNX export. The state is kept in a
[Hugging Face bucket](https://huggingface.co/docs/hub/storage-buckets), so a stage can stop and resume.

A full run takes about 5 h 20 min on one RTX PRO 6000 and costs about $15.

## Results

Our run (`README recipe, one phase`) compared with the released checkpoint. Both models were scored by the same
`tiny_voca.evaluate` call: 300 clips per voice and split (the test split has 296 F2 and 298 M2 clips).

| split | model | WER | duration ratio | pause ratio | mel DTW |
|---|---|---|---|---|---|
| test | ours | 1.26 % | 0.986 | 1.13 | 0.733 |
| test | released | 1.33 % | 0.956 | 1.05 | 0.731 |
| test | source recordings | 0.98 % | | | |
| validation | ours | 0.74 % | 1.002 | 1.15 | 0.741 |
| validation | released | 0.81 % | 0.970 | 1.06 | 0.741 |
| validation | source recordings | 0.72 % | | | |

- WER: Whisper small.en (faster-whisper) reads the model's audio back; the WER is against the split's text. "Source
  recordings" is the same read-back of the dataset's own audio.
- Duration ratio: length of the model's audio divided by the length of the source recording. Pause ratio: the model's
  predicted pause frames divided by the pause frames in the source alignment. 1.0 is the same.
- Mel DTW: distance between the model's mel spectrogram and the source recording's, after time alignment. Lower is closer.

Per voice, on the test split:

| voice | WER ours | WER released | WER source |
|---|---|---|---|
| F2 | 1.28 % | 1.25 % | 1.01 % |
| M2 | 1.24 % | 1.41 % | 0.94 % |

### How this run differs from the released checkpoint

The released checkpoint was trained in two phases: 66,000 steps on the first data batch with eSpeak annotations, then
38,737 steps on the current data and annotations. This recipe runs the single 104,000-step command from the upstream
README on the current data. Our data stage keeps the same clips as the released annotations (16,018 train, 714 validation,
594 test).

## Before you start

- A Hugging Face account with [Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) access and the `hf` CLI
  (`pip install -U huggingface_hub`), logged in with a token that can write.
- A bucket for the state:

  ```bash
  hf buckets create <you>/<bucket> --private
  ```

## Run it

Run the commands from the root of this repository. The `-v ./tiny-voca:/scripts:ro` flag puts this folder into the Job.

**1. Smoke test** (about 15 min, under $1). Two input files per split, 300 training steps. It checks the install, the
bucket and every stage.

```bash
hf jobs run --flavor rtx-pro-6000 --timeout 1h --secrets HF_TOKEN \
    -e TV_BUCKET=hf://buckets/<you>/<bucket>/tiny-voca -e SMOKE=1 \
    -v ./tiny-voca:/scripts:ro ghcr.io/astral-sh/uv:python3.12-bookworm bash /scripts/run.sh all
```

The log must end with `STAGE-DONE all`.

**2. Full run** (about 5 h 20 min, about $15).

```bash
hf jobs run --flavor rtx-pro-6000 --timeout 8h --secrets HF_TOKEN \
    -e TV_BUCKET=hf://buckets/<you>/<bucket>/tiny-voca \
    -v ./tiny-voca:/scripts:ro ghcr.io/astral-sh/uv:python3.12-bookworm bash /scripts/run.sh all
```

You can also run the stages as separate Jobs: `run.sh data`, then `run.sh train`, then `run.sh eval`. Each stage loads
what it needs from the bucket.

### Stages

| stage | what it does | time on RTX PRO 6000 |
|---|---|---|
| data | downloads the dataset (pinned revision), computes features with one process per CPU, aligns phonemes with wav2vec2, saves `features/` and `annotations/` | 21 min |
| train | `tiny_voca.train`, 104,000 steps (GAN from step 8,000); saves `runs/<run>/` to the bucket every 10 minutes | 3 h 50 min |
| eval | `tiny_voca.evaluate` for our checkpoint and for the released one, then the ONNX export | 1 h 10 min |

### Settings

| variable | default | meaning |
|---|---|---|
| `TV_BUCKET` | required | bucket folder for the state, e.g. `hf://buckets/<you>/<bucket>/tiny-voca` |
| `RUN` | `repro` (`smoke` with `SMOKE=1`) | run name; the state is in `$TV_BUCKET/$RUN` |
| `SMOKE` | `0` | `1`: small data and 300 steps |
| `STEPS` | `104000` | training steps |
| `PER_VOICE` | `300` | evaluation clips per voice and split |
| `TV_COMMIT`, `DIZIONE_COMMIT`, `DS_REV`, `MODEL_REV` | pinned | code, phonemizer, dataset and released-weights revisions |

### Resume

If a Job stops (for example at its timeout), start the same command again with the same `RUN`. The train stage resumes
from `runs/<run>/last.pt` in the bucket. To run only the missing stages, call `run.sh train` or `run.sh eval`.

### Output

Everything is in `$TV_BUCKET/$RUN`:

| path | content |
|---|---|
| `runs/<run>/inference.pt` | the trained model (18 MB) |
| `runs/<run>/last.pt`, `step-*.pt` | training checkpoints |
| `runs/<run>/metrics.jsonl` | training metrics |
| `eval/<run>/`, `eval/released/` | evaluation summaries and audio for both models |
| `export/tiny-voca-<run>.onnx` | the ONNX model |

### Use the model

```bash
hf buckets cp hf://buckets/<you>/<bucket>/tiny-voca/repro/runs/repro/inference.pt .
pip install "tiny-voca[own] @ git+https://github.com/n4ze3m/tiny-voca@903f940"
hf download n4ze3m/tiny-voca --include "phonemizer/*" --local-dir weights
DIZIONE_DATA=weights/phonemizer python -m tiny_voca.infer --checkpoint inference.pt --voice M2 \
    --text "Hello! This is tiny voca." --output hello.wav
```

The ONNX file works in the same way as `onnx/model.onnx` in the [model repository](https://huggingface.co/n4ze3m/tiny-voca).

## Notes

- **CPU count.** A Job reports the host's CPU count, which can be higher than the CPUs it may use. `run.sh` reads the
  container's CPU quota and starts one feature process per CPU (`prepare_sharded.py`).
- **Other GPUs.** The full run was tested on `rtx-pro-6000` (about 7.2 training steps per second). Other flavors should
  work, but they are not tested; the train stage time depends on the GPU.
- **Uploads.** The bucket upload retries up to five times, with a longer wait each time.
- **Experimental:** `design_voice.py` makes new voice designs with Qwen3-TTS VoiceDesign, the method that made the two
  released voices. The steps that render training audio in a new voice and train with it are not part of this recipe yet.

## Credits and licences

tiny-voca, its dataset and its phonemizer (dizione) are by [n4ze3m](https://github.com/n4ze3m), under Apache-2.0. Read the
[model card](https://huggingface.co/n4ze3m/tiny-voca) and the [dataset card](https://huggingface.co/datasets/n4ze3m/tiny-voca)
for the data sources and their terms.
