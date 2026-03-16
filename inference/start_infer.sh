#!/usr/bin/env bash
# coding=utf-8
# Copyright 2026 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<EOF
Usage: bash inference/start_infer.sh [options]

Options:
  --model-url <url(s)>   Model base URL(s), comma-separated for multiple (default: http://0.0.0.0:1/v1)
  --max-steps <int>      Max reasoning steps (default: 100)
  --benchmark <name>     Benchmark name (default: browsecomp)
  --parallel <int>       Parallel workers (default: 4)
  --web-topk <int>       web_search top-k (default: 20)
  --log-file <path>      Log file path (default: inference/logs/infer_log.log)
  -h, --help             Show this help

Available benchmarks:
  browsecomp, gaia, xbench, webwalker, frames, seal_0

Example (single URL):
  bash inference/start_infer.sh \
    --model-url http://0.0.0.0:1/v1 \
    --max-steps 100 \
    --benchmark browsecomp \
    --parallel 4 \
    --web-topk 20

Example (multiple URLs, round-robin):
  bash inference/start_infer.sh \
    --model-url "http://0.0.0.0:8000/v1,http://0.0.0.0:8001/v1" \
    --parallel 8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-url)
      MODEL_URL="$2"; shift 2 ;;
    --max-steps)
      MAX_STEPS="$2"; shift 2 ;;
    --benchmark)
      BENCHMARK="$2"; shift 2 ;;
    --parallel)
      PARALLEL="$2"; shift 2 ;;
    --web-topk)
      WEB_TOPK="$2"; shift 2 ;;
    --log-file)
      LOG_FILE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

# Apply defaults for any options not provided
MODEL_URL="${MODEL_URL:-http://0.0.0.0:1/v1}"
MAX_STEPS="${MAX_STEPS:-100}"
BENCHMARK="${BENCHMARK:-browsecomp}"
PARALLEL="${PARALLEL:-4}"
WEB_TOPK="${WEB_TOPK:-20}"
LOG_FILE="${LOG_FILE:-inference/logs/infer_log.log}"

echo ""
echo "Available benchmarks: browsecomp | gaia | xbench | webwalker | frames | seal_0"
echo ""

mkdir -p "$(dirname "${LOG_FILE}")"

# Load service endpoints and keys
set -a
source "${PROJECT_ROOT}/.env"
set +a

# Export runtime overrides used by inference_web_agent.py
export MODEL_URL
export MAX_STEPS
export BENCHMARK
export PARALLEL
export WEB_TOPK

echo "Launching inference with params:"
echo "  MODEL_URL=${MODEL_URL}"
echo "  MAX_STEPS=${MAX_STEPS}"
echo "  BENCHMARK=${BENCHMARK}"
echo "  PARALLEL=${PARALLEL}"
echo "  WEB_TOPK=${WEB_TOPK}"
echo "  LOG_FILE=${LOG_FILE}"

nohup python3 inference/inference_web_agent.py > "${LOG_FILE}" 2>&1 &
echo "Started. PID=$!"
