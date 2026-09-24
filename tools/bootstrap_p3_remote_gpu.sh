#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${1:-$HOME/vn97-p3-gpu}"
REPO_SLUG="${VN97_REPOSITORY:-dathito108-hue/VN97}"
TRAINING_COMMIT="${VN97_TRAINING_COMMIT:-b68428ea9e16f8fd0b42bdd6a76b633456695201}"
CORPUS_RUN_ID="${VN97_P3_CORPUS_RUN_ID:-35992987070}"
CORPUS_ARTIFACT="${VN97_P3_CORPUS_ARTIFACT:-VN97-P3-Corpus-1623f2c187d043918d681a7dc9e9cf025a58d26f}"

REPO_DIR="$WORK_ROOT/VN97"
CORPUS_DIR="$WORK_ROOT/p3-corpus-artifact"
OUTPUT_DIR="$WORK_ROOT/p3-language-output"
LOG_DIR="$WORK_ROOT/logs"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 10
  }
}

require_command git
require_command gh
require_command python
require_command sha256sum

mkdir -p "$WORK_ROOT" "$LOG_DIR"

if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run: gh auth login" >&2
  exit 11
fi

if [[ -e "$REPO_DIR" && ! -d "$REPO_DIR/.git" ]]; then
  echo "repository target exists but is not a Git checkout: $REPO_DIR" >&2
  exit 12
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  gh repo clone "$REPO_SLUG" "$REPO_DIR"
fi

git -C "$REPO_DIR" fetch --force --tags origin
git -C "$REPO_DIR" checkout --detach "$TRAINING_COMMIT"

ACTUAL_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$TRAINING_COMMIT" ]]; then
  echo "training commit mismatch after checkout" >&2
  exit 13
fi

if ! git -C "$REPO_DIR" diff --quiet   || ! git -C "$REPO_DIR" diff --cached --quiet; then
  echo "tracked repository files are not clean" >&2
  exit 14
fi

if [[ -e "$CORPUS_DIR" ]]; then
  if [[ ! -d "$CORPUS_DIR" ]]; then
    echo "corpus target exists but is not a directory" >&2
    exit 15
  fi
  if [[ -n "$(find "$CORPUS_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "corpus target must be new or empty: $CORPUS_DIR" >&2
    exit 16
  fi
else
  mkdir -p "$CORPUS_DIR"
fi

gh run download "$CORPUS_RUN_ID"   -R "$REPO_SLUG"   -n "$CORPUS_ARTIFACT"   -D "$CORPUS_DIR"

for required in   "$CORPUS_DIR/SHA256SUMS"   "$CORPUS_DIR/corpus/corpus.vn97corpus1.json"   "$CORPUS_DIR/corpus/training.jsonl"   "$CORPUS_DIR/corpus/validation.jsonl"   "$CORPUS_DIR/corpus/release.jsonl"
do
  [[ -f "$required" ]] || {
    echo "missing corpus artifact file: $required" >&2
    exit 17
  }
done

(
  cd "$CORPUS_DIR"
  sha256sum -c SHA256SUMS
)

EXPECTED_CORPUS_ID="77e9142054f82b78c8ca0ff108c5e5297d43292841f9fe1ea69c5dc97a7c303f"
ACTUAL_CORPUS_ID="$(
  python - "$CORPUS_DIR/corpus/corpus.vn97corpus1.json" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["manifest_id"])
PY
)"
if [[ "$ACTUAL_CORPUS_ID" != "$EXPECTED_CORPUS_ID" ]]; then
  echo "VN97CORPUS1 identity mismatch" >&2
  exit 18
fi

if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "output directory must be new or empty: $OUTPUT_DIR" >&2
  exit 19
fi

cd "$REPO_DIR"

export VN97_EXPECTED_REPOSITORY_COMMIT="$TRAINING_COMMIT"

echo "VN97 P3 remote GPU bootstrap ready"
echo "repository_commit=$TRAINING_COMMIT"
echo "corpus_run_id=$CORPUS_RUN_ID"
echo "corpus_artifact=$CORPUS_ARTIFACT"
echo "corpus_manifest_id=$ACTUAL_CORPUS_ID"
echo "output_dir=$OUTPUT_DIR"

tools/run_p3_gpu.sh   "$CORPUS_DIR/corpus"   "$OUTPUT_DIR"   2>&1 | tee "$LOG_DIR/p3-language-campaign.log"

echo "VN97 P3 remote GPU run completed"
echo "evidence=$OUTPUT_DIR/p3-run.vn97p3run1.json"
echo "log=$LOG_DIR/p3-language-campaign.log"
