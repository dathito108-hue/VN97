#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
OUTPUT="${2:-}"

cd "$(git rev-parse --show-toplevel)"
python -m pip install -e .

if [[ -n "${OUTPUT}" ]]; then
  python -m vn97.r2.cpu_pilot_cli     --profile "${PROFILE}"     --output "${OUTPUT}"
else
  python -m vn97.r2.cpu_pilot_cli     --profile "${PROFILE}"
fi
