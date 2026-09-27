#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-mamba2-2.7b-source}"
MODEL_REV="99b226cc377d131cccc610ed4346db564f381f1e"
MODEL_SHA="254d89bfef6dd9f44c8e3f54e0730af3f10c482969117cb37fb33486348b00be"
MODEL_SIZE="5405424282"
TOKENIZER_REV="364ae95407723fadd1d47b023c1efb92a4d891c3"

mkdir -p "$OUT/tokenizer"

available_kb="$(df -Pk "$OUT" | awk 'NR==2 {print $4}')"
required_kb="$((12 * 1024 * 1024))"
if [[ "$available_kb" -lt "$required_kb" ]]; then
  echo "R2-G0.2 source fetch requires at least 12 GiB free space." >&2
  echo "available_kb=$available_kb required_kb=$required_kb" >&2
  exit 2
fi

model_base="https://huggingface.co/state-spaces/mamba2-2.7b/resolve/${MODEL_REV}"
tokenizer_base="https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/${TOKENIZER_REV}"

curl_common=(
  --fail
  --location
  --retry 5
  --retry-delay 3
  --retry-all-errors
)

curl "${curl_common[@]}"   "$model_base/config.json?download=true"   -o "$OUT/config.json"

curl "${curl_common[@]}"   --continue-at -   "$model_base/pytorch_model.bin?download=true"   -o "$OUT/pytorch_model.bin"

actual_size="$(wc -c < "$OUT/pytorch_model.bin" | tr -d ' ')"
if [[ "$actual_size" != "$MODEL_SIZE" ]]; then
  echo "Mamba-2 weight size mismatch: $actual_size != $MODEL_SIZE" >&2
  exit 3
fi

actual_sha="$(sha256sum "$OUT/pytorch_model.bin" | awk '{print $1}')"
if [[ "$actual_sha" != "$MODEL_SHA" ]]; then
  echo "Mamba-2 weight SHA-256 mismatch." >&2
  exit 4
fi

for file in   tokenizer.json   tokenizer_config.json   special_tokens_map.json   vocab.json   merges.txt
do
  curl "${curl_common[@]}"     "$tokenizer_base/$file?download=true"     -o "$OUT/tokenizer/$file"
done

(
  cd "$OUT"
  sha256sum config.json pytorch_model.bin tokenizer/* > SHA256SUMS
)

cat > "$OUT/SOURCE_IDENTITY" <<EOF
model=state-spaces/mamba2-2.7b
model_revision=$MODEL_REV
model_weight_sha256=$MODEL_SHA
model_weight_size_bytes=$MODEL_SIZE
tokenizer=EleutherAI/gpt-neox-20b
tokenizer_revision=$TOKENIZER_REV
EOF

echo "R2-G0.2 pinned source download complete:"
echo "  root=$OUT"
echo "  model_revision=$MODEL_REV"
echo "  weight_sha256=$MODEL_SHA"
echo "  tokenizer_revision=$TOKENIZER_REV"
