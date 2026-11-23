#!/usr/bin/env bash
# tiny-voca (github.com/n4ze3m/tiny-voca) on Hugging Face Jobs: the README's training steps, one stage per call, state in a bucket.
#
#   hf jobs run --flavor rtx-pro-6000 --timeout 8h --secrets HF_TOKEN -v ./tiny-voca:/scripts:ro \
#       -e TV_BUCKET=hf://buckets/<you>/<bucket>/tiny-voca ghcr.io/astral-sh/uv:python3.12-bookworm bash /scripts/run.sh all
#
# Stages: data (download, features, alignment) | train | eval (scores + ONNX) | all. SMOKE=1: two input files per split and
# 300 training steps, to check the whole chain in a few minutes. Each stage saves to $TV_BUCKET/$RUN before it ends; train also
# saves every 10 minutes and resumes from the bucket.
set -uo pipefail
STAGE=${1:?stage: data | train | eval | all}
TV_REPO=https://github.com/n4ze3m/tiny-voca; TV_COMMIT=${TV_COMMIT:-903f940}
DIZIONE_COMMIT=${DIZIONE_COMMIT:-8cd889e}   # the phonemizer; same front-end fingerprint as the released checkpoint
DS_REV=${DS_REV:-63b6b89622e984d7c7a0799ca4b4c443dd5f48b2}       # n4ze3m/tiny-voca dataset
MODEL_REV=${MODEL_REV:-5fe06641d0afecb3c04bc9faca2cf64d9b2b1600}  # n4ze3m/tiny-voca weights (phonemizer data, released checkpoint)
SMOKE=${SMOKE:-0}
RUN=${RUN:-$([ "$SMOKE" = 1 ] && echo smoke || echo repro)}
STEPS=${STEPS:-$([ "$SMOKE" = 1 ] && echo 300 || echo 104000)}
GAN_START=$([ "$SMOKE" = 1 ] && echo 100 || echo 8000)
PER_VOICE=${PER_VOICE:-$([ "$SMOKE" = 1 ] && echo 4 || echo 300)}   # evaluation clips per voice and split
BK=${TV_BUCKET:?set TV_BUCKET, e.g. hf://buckets/<you>/<bucket>/tiny-voca}/$RUN
W=/work; HERE=$(dirname "$0"); HF=/opt/hfcli/bin/hf
export TINY_VOCA_FRONTEND=own DIZIONE_DATA=$W/weights/phonemizer HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_PROGRESS_BARS=1
log() { echo "$(date +%T) $*"; }
cpus() { echo "${CPU_CORES:-$(nproc)}"; }   # the CPUs this Job may use (nproc in older images reports the host)
fail() { log "FAILED: $*"; exit 1; }
[ "$RUN" = released ] && fail "RUN=released is reserved for the released checkpoint's evaluation"
# bucket transfers are idempotent, so one that fails (e.g. a 429 rate limit) is retried with a growing wait
retry() { for t in 1 2 3 4 5; do "$@" >/dev/null && return 0; [ $t = 5 ] && return 1; log "try $t failed: ${*:1:4}; retry in $((t * 60)) s"; sleep $((t * 60)); done; }

setup() {
  log "setup"
  uv venv -q /opt/hfcli && uv pip install -q --python /opt/hfcli/bin/python huggingface_hub || fail "hf cli"
  uv pip install -q --system torch torchaudio onnx onnxscript "tiny-voca[prepare,annotate,evaluate] @ git+$TV_REPO@$TV_COMMIT" \
      "dizione @ git+https://github.com/n4ze3m/dizione@$DIZIONE_COMMIT" || fail "install"
  # faster-whisper (evaluate) finds the CUDA libraries that came with torch
  export LD_LIBRARY_PATH=$(python -c "import os,nvidia.cublas,nvidia.cudnn as c; print(':'.join(os.path.join(os.path.dirname(m.__file__),'lib') for m in (nvidia.cublas,c)))" 2>/dev/null):${LD_LIBRARY_PATH:-}
  mkdir -p $W && cd $W
  echo probe > $W/probe.txt && $HF buckets cp $W/probe.txt $BK/probe.txt >/dev/null && $HF buckets cp $BK/probe.txt - >/dev/null || fail "cannot write to $BK"
  $HF download n4ze3m/tiny-voca --revision $MODEL_REV --include "phonemizer/*" --include tiny-voca.pt --local-dir $W/weights -q >/dev/null || fail "weights"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; log "setup done"
}
save() { log "save $*"; for d in "$@"; do retry $HF buckets sync $W/$d $BK/$d --exclude "*.tmp" || fail "save $d"; done; }
load() { log "load $*"; for d in "$@"; do mkdir -p $W/$d; retry $HF buckets sync $BK/$d $W/$d || fail "load $d"; done; }

data() {
  # a finished data stage is reused: preparing again would change the annotation fingerprint, and training could not resume
  if [ "${FORCE_DATA:-0}" != 1 ] && $HF buckets cp $BK/annotations/annotations.json - >/dev/null 2>&1; then
    log "data is in the bucket: skip (FORCE_DATA=1 prepares it again)"; return; fi
  log "prepare workers: $(cpus)"
  $HF download n4ze3m/tiny-voca --repo-type dataset --revision $DS_REV --include "data/*" --include "batch2/data/*" --local-dir $W/ds -q >/dev/null || fail "dataset"
  pick() { if [ "$SMOKE" = 1 ]; then ls $1 | head -2; else ls $1; fi; }
  python $HERE/prepare_sharded.py --split train --inputs $(pick "$W/ds/data/train-*.parquet") $(pick "$W/ds/batch2/data/train-*.parquet") \
      --output $W/features --workers $(cpus) || fail "prepare train"
  for s in validation test; do python $HERE/prepare_sharded.py --split $s --inputs $(pick "$W/ds/data/$s-*.parquet") --output $W/features \
      --workers $(cpus) || fail "prepare $s"; done
  # annotate loads its aligner from the local cache only
  python -c "from huggingface_hub import snapshot_download as d; d('facebook/wav2vec2-lv-60-espeak-cv-ft')" >/dev/null || fail "aligner download"
  python -m tiny_voca.annotate --features $W/features --output $W/annotations --dataset-revision $DS_REV || fail "annotate"
  # the released model's annotations (dataset annotations/v4) kept 16,018 / 714 / 594 clips
  for s in train validation test; do log "annotated $s: $(grep -c . $W/annotations/$s.jsonl) clips"; done
  if [ "$SMOKE" != 1 ] && [ "$DS_REV" = 63b6b89622e984d7c7a0799ca4b4c443dd5f48b2 ]; then
    for e in train:16018 validation:714 test:594; do n=$(grep -c . $W/annotations/${e%%:*}.jsonl); d=$(( n - ${e##*:} ))
      [ ${d#-} -le 5 ] || fail "annotated ${e%%:*}: $n clips, expected about ${e##*:}"; done
  fi
  save features annotations; log "data done"
}

train() {
  # a smoke test always trains from step 0; a full run resumes from the bucket
  if [ "$SMOKE" = 1 ]; then rm -rf $W/runs/$RUN; else load runs/$RUN; fi
  R=""
  if [ -f $W/runs/$RUN/last.pt ]; then
    done_steps=$(python -c "import torch; print(torch.load('$W/runs/$RUN/last.pt', map_location='cpu', weights_only=False)['step'])")
    [ "$done_steps" -ge "$STEPS" ] && { log "train already done ($done_steps steps): skip"; return; }
    R="--resume $W/runs/$RUN/last.pt"; log "resuming from the bucket at step $done_steps"
  fi
  [ -f $W/annotations/train.jsonl ] || load features annotations
  # periodic backup every 10 min; the stop file ends it, and `wait` lets an upload in flight finish before the final save
  rm -f $W/.stop-sync
  ( while true; do
      for _ in $(seq 60); do [ -f $W/.stop-sync ] && exit 0; sleep 10; done
      $HF buckets sync $W/runs/$RUN $BK/runs/$RUN --exclude "*.tmp" >/dev/null 2>&1 || log "periodic save failed"
    done ) & SYNC=$!
  python -m tiny_voca.train --train $W/annotations/train.jsonl --validation $W/annotations/validation.jsonl --output $W/runs/$RUN \
      --steps $STEPS --gan-start $GAN_START --gan-ratio 0.7 --teacher-final 0 --max-seconds 172800 $R
  rc=$?; touch $W/.stop-sync; wait $SYNC; save runs/$RUN
  [ $rc = 0 ] || fail "train"; log "train done"
}

evaluate() {
  [ -f $W/runs/$RUN/inference.pt ] || load runs/$RUN
  [ -f $W/annotations/test.jsonl ] || load features annotations
  python -m tiny_voca.evaluate --checkpoint $W/runs/$RUN/inference.pt --annotations $W/annotations --output $W/eval/$RUN --tag $RUN --per-voice $PER_VOICE || fail "evaluate"
  python -m tiny_voca.evaluate --checkpoint $W/weights/tiny-voca.pt --annotations $W/annotations --output $W/eval/released --tag released --per-voice $PER_VOICE || fail "evaluate released"
  mkdir -p $W/export && python -m tiny_voca.export --checkpoint $W/runs/$RUN/inference.pt --output $W/export/tiny-voca-$RUN.onnx || fail "export"
  # one line per model for the log; the full reports are eval/<run>/report.json and eval/released/report.json
  for t in $RUN released; do python -c "import json,sys; r=json.load(open(sys.argv[1])); a=r['summary']['test']['all']
print('RESULT', r['tag'], 'step', r['step'], 'test wer', a['generated_wer'], 'source wer', a['reference_wer'], 'duration', a['duration_ratio'],
      'pause', a['pause_ratio'], 'mel_dtw', a['mel_dtw'], 'token mismatches', len(r.get('frontend_token_mismatches') or []))" $W/eval/$t/report.json; done
  # thousands of small WAV files make many upload requests: one tar per model
  for t in $RUN released; do tar -cf $W/eval/$t/audio.tar -C $W/eval/$t audio && rm -rf $W/eval/$t/audio; done
  save eval export; log "eval done"
}

setup
case $STAGE in
  data) data;; train) train;; eval) evaluate;;
  all) data; train; evaluate;;
  *) fail "unknown stage $STAGE";;
esac
log "STAGE-DONE $STAGE"
