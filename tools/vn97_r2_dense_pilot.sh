#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
python -m pip install -e .
exec python -m vn97.r2.dense_pilot_cli "$@"
