#!/usr/bin/env bash
# tiny-voca (github.com/n4ze3m/tiny-voca) on Hugging Face Jobs: the README's training steps, one stage per call, state in a bucket.
#
#   hf jobs run --flavor l40sx1 --timeout 12h --secrets HF_TOKEN -v ./tiny-voca:/scripts:ro \
#       -e TV_BUCKET=hf://buckets/<you>/<bucket>/tiny-voca ghcr.io/astral-sh/uv:python3.12-bookworm bash /scripts/run.sh all
#
# Stages: data (download, features, alignment) | train | eval (scores + ONNX) | all. SMOKE=1: two input files per split and
# 300 training steps, to check the whole chain in a few minutes. Each stage saves to $TV_BUCKET/$RUN before it ends; train also
# saves every 10 minutes and resumes from the bucket.
set -uo pipefail
STAGE=${1:?stage: data | train | eval | all}
TV_REPO=https://github.com/n4ze3m/tiny-voca; TV_COMMIT=${TV_COMMIT:-903f940}
DS_REV=${DS_REV:-63b6b89622e984d7c7a0799ca4b4c443dd5f48b2}       # n4ze3m/tiny-voca dataset
MODEL_REV=${MODEL_REV:-5fe06641d0afecb3c04bc9faca2cf64d9b2b1600}  # n4ze3m/tiny-voca weights (phonemizer data, released checkpoint)
SMOKE=${SMOKE:-0}
RUN=${RUN:-$([ "$SMOKE" = 1 ] && echo smoke || echo repro)}
STEPS=${STEPS:-$([ "$SMOKE" = 1 ] && echo 300 || echo 104000)}
GAN_START=$([ "$SMOKE" = 1 ] && echo 100 || echo 8000)
BK=${TV_BUCKET:?set TV_BUCKET, e.g. hf://buckets/<you>/<bucket>/tiny-voca}/$RUN
W=/work; HERE=$(dirname "$0"); HF=/opt/hfcli/bin/hf
export TINY_VOCA_FRONTEND=own DIZIONE_DATA=$W/weights/phonemizer HF_HUB_ENABLE_HF_TRANSFER=0
log() { echo "$(date +%T) $*"; }
fail() { log "FAILED: $*"; exit 1; }

setup() {
  log "setup"
  uv venv -q /opt/hfcli && uv pip install -q --python /opt/hfcli/bin/python huggingface_hub || fail "hf cli"
  uv pip install -q --system torch torchaudio onnx onnxscript "tiny-voca[own,prepare,annotate,evaluate] @ git+$TV_REPO@$TV_COMMIT" || fail "install"
  # faster-whisper (evaluate) finds the CUDA libraries that came with torch
  export LD_LIBRARY_PATH=$(python -c "import os,nvidia.cublas,nvidia.cudnn as c; print(':'.join(os.path.join(os.path.dirname(m.__file__),'lib') for m in (nvidia.cublas,c)))" 2>/dev/null):${LD_LIBRARY_PATH:-}
  mkdir -p $W && cd $W
  echo probe > $W/probe.txt && $HF buckets cp $W/probe.txt $BK/probe.txt >/dev/null && $HF buckets cp $BK/probe.txt - >/dev/null || fail "cannot write to $BK"
  $HF download n4ze3m/tiny-voca --revision $MODEL_REV --include "phonemizer/*" --include tiny-voca.pt --local-dir $W/weights -q >/dev/null || fail "weights"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; log "setup done"
}
save() { log "save $*"; for d in "$@"; do $HF buckets sync $W/$d $BK/$d >/dev/null || fail "save $d"; done; }
load() { log "load $*"; for d in "$@"; do mkdir -p $W/$d; $HF buckets sync $BK/$d $W/$d >/dev/null || fail "load $d"; done; }

data() {
  $HF download n4ze3m/tiny-voca --repo-type dataset --revision $DS_REV --include "data/*" --include "batch2/data/*" --local-dir $W/ds -q >/dev/null || fail "dataset"
  pick() { if [ "$SMOKE" = 1 ]; then ls $1 | head -2; else ls $1; fi; }
  python $HERE/prepare_sharded.py --split train --inputs $(pick "$W/ds/data/train-*.parquet") $(pick "$W/ds/batch2/data/train-*.parquet") \
      --output $W/features --workers $(nproc) || fail "prepare train"
  for s in validation test; do python $HERE/prepare_sharded.py --split $s --inputs $(pick "$W/ds/data/$s-*.parquet") --output $W/features \
      --workers $(nproc) || fail "prepare $s"; done
  # annotate loads its aligner from the local cache only
  python -c "from huggingface_hub import snapshot_download as d; d('facebook/wav2vec2-lv-60-espeak-cv-ft')" >/dev/null || fail "aligner download"
  python -m tiny_voca.annotate --features $W/features --output $W/annotations --dataset-revision $DS_REV || fail "annotate"
  save features annotations; log "data done"
}

train() {
  [ -f $W/annotations/train.jsonl ] || load features annotations
  load runs/$RUN
  R=""; [ -f $W/runs/$RUN/last.pt ] && R="--resume $W/runs/$RUN/last.pt" && log "resuming from the bucket"
  ( while sleep 600; do $HF buckets sync $W/runs/$RUN $BK/runs/$RUN >/dev/null 2>&1; done ) & SYNC=$!
  python -m tiny_voca.train --train $W/annotations/train.jsonl --validation $W/annotations/validation.jsonl --output $W/runs/$RUN \
      --steps $STEPS --gan-start $GAN_START --gan-ratio 0.7 --teacher-final 0 --max-seconds 172800 $R || { kill $SYNC; save runs/$RUN; fail "train"; }
  kill $SYNC; save runs/$RUN; log "train done"
}

evaluate() {
  [ -f $W/runs/$RUN/inference.pt ] || load runs/$RUN
  [ -f $W/annotations/test.jsonl ] || load features annotations
  python -m tiny_voca.evaluate --checkpoint $W/runs/$RUN/inference.pt --annotations $W/annotations --output $W/eval/$RUN --tag $RUN || fail "evaluate"
  python -m tiny_voca.evaluate --checkpoint $W/weights/tiny-voca.pt --annotations $W/annotations --output $W/eval/released --tag released || log "released eval failed"
  mkdir -p $W/export && python -m tiny_voca.export --checkpoint $W/runs/$RUN/inference.pt --output $W/export/tiny-voca-$RUN.onnx || fail "export"
  save eval export; log "eval done"
}

setup
case $STAGE in
  data) data;; train) train;; eval) evaluate;;
  all) data; train; evaluate;;
  *) fail "unknown stage $STAGE";;
esac
log "STAGE-DONE $STAGE"
