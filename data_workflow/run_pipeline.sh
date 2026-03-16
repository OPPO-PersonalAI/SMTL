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

# ============================================================
# data_workflow/run_pipeline.sh
#
# End-to-end trajectory generation workflow runner.
# Run from the project root (SMTL-main):
#   bash data_workflow/run_pipeline.sh
#
# Steps:
#   1. Run search agent and capture trajectories
#   2. Evaluate predicted answers against ground truth
#   3. Filter correct trajectories
#   4. Convert filtered trajectories to training conversations
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env file not found at $ENV_FILE"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------- configurable parameters ----------
INPUT_FILE="${INPUT_FILE:-}"
SAMPLE_NUM="${SAMPLE_NUM:-}"
SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-8}"
PROMPTS_TYPE="${PROMPTS_TYPE:-generation}"
PARALLEL="${PARALLEL:-100}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_RETRIES="${MAX_RETRIES:-3}"

MAX_WORKERS="${MAX_WORKERS:-50}"
RESUME="${RESUME:-true}"
SKIP_DETAILED_STATS="${SKIP_DETAILED_STATS:-false}"

DATE_TAG="${DATE_TAG:-$(date +%m%d)}"
# ---------------------------------------------

if [[ ! -d "$SCRIPT_DIR" ]]; then
    echo "ERROR: data_workflow directory not found."
    exit 1
fi

if [[ -z "$INPUT_FILE" ]]; then
    echo "ERROR: INPUT_FILE is not set."
    echo "Usage: INPUT_FILE=./data_workflow/input/sample.jsonl bash data_workflow/run_pipeline.sh"
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

STEM="$(basename "$INPUT_FILE" .jsonl)"
STEM="${STEM%.json}"

RESULTS1_DIR="$SCRIPT_DIR/results1"
RESULTS2_DIR="$SCRIPT_DIR/results2"
RESULTS3_DIR="$SCRIPT_DIR/results3"
RESULTS4_DIR="$SCRIPT_DIR/results_final"

mkdir -p "$RESULTS1_DIR" "$RESULTS2_DIR" "$RESULTS3_DIR" "$RESULTS4_DIR"

STEP1_OUT="$RESULTS1_DIR/${STEM}_traj.jsonl"
STEP2_OUT="$RESULTS2_DIR/${STEM}_traj_evaluated.jsonl"

echo "======================================================"
echo "  Data Workflow Pipeline"
echo "======================================================"
echo "  Input file         : $INPUT_FILE"
echo "  Sample num         : ${SAMPLE_NUM:-all}"
echo "  Summary interval   : $SUMMARY_INTERVAL"
echo "  Prompts type       : $PROMPTS_TYPE"
echo "  Parallel workers   : $PARALLEL"
echo "  Max steps          : $MAX_STEPS"
echo "  Max retries        : $MAX_RETRIES"
echo "  Eval workers       : $MAX_WORKERS"
echo "  Resume             : $RESUME"
echo "  Skip stats         : $SKIP_DETAILED_STATS"
echo "  Date tag           : $DATE_TAG"
echo "======================================================"

echo ""
echo ">>> Step 1: Run search agent and generate trajectories"
STEP1_ARGS=(
    --infile "$INPUT_FILE"
    --outfile "$STEP1_OUT"
    --summary_interval "$SUMMARY_INTERVAL"
    --prompts_type "$PROMPTS_TYPE"
    --parallel "$PARALLEL"
    --max_steps "$MAX_STEPS"
    --max_retries "$MAX_RETRIES"
)
if [[ -n "$SAMPLE_NUM" ]]; then
    STEP1_ARGS+=(--sample_num "$SAMPLE_NUM")
fi
"$PYTHON_BIN" "$SCRIPT_DIR/step1_run_parallelized_agent.py" "${STEP1_ARGS[@]}"

echo ""
echo ">>> Step 2: Evaluate trajectories"
STEP2_ARGS=(
    --input_file "$STEP1_OUT"
    --output_dir "$RESULTS2_DIR"
    --max_workers "$MAX_WORKERS"
)
if [[ -n "$SAMPLE_NUM" ]]; then
    STEP2_ARGS+=(--sample_num "$SAMPLE_NUM")
fi
if [[ "$RESUME" == "true" ]]; then
    STEP2_ARGS+=(--resume)
fi
if [[ "$SKIP_DETAILED_STATS" == "true" ]]; then
    STEP2_ARGS+=(--skip_detailed_stats)
fi
"$PYTHON_BIN" "$SCRIPT_DIR/step2_postprocess_eval_results.py" "${STEP2_ARGS[@]}"

echo ""
echo ">>> Step 3: Filter correct trajectories"
"$PYTHON_BIN" "$SCRIPT_DIR/step3_postprocess_filter_defeat.py" \
    --input_dir "$RESULTS2_DIR" \
    --output_dir "$RESULTS3_DIR"

echo ""
echo ">>> Step 4: Convert filtered trajectories to training format"
"$PYTHON_BIN" "$SCRIPT_DIR/step4_postprocess_clean_and_transform_format.py" \
    --input_dir "$RESULTS3_DIR" \
    --output_dir "$RESULTS4_DIR" \
    --date "$DATE_TAG"

echo ""
echo "======================================================"
echo "✅ Workflow complete!"
echo "   Step 1 trajectories : $STEP1_OUT"
echo "   Step 2 evaluations  : $STEP2_OUT"
echo "   Step 3 filtered dir : $RESULTS3_DIR"
echo "   Step 4 final dir    : $RESULTS4_DIR"
echo "======================================================"
