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
# data_synthesis/run_pipeline.sh
#
# End-to-end data synthesis pipeline runner.
# Run from the project root (SMTL-main):
#   bash data_synthesis/run_pipeline.sh
#
# Directory layout expected / created automatically:
#   ./data_synthesis/input/          <- put input file here
#   ./data_synthesis/cache/cache_1/ <- step 1 intermediate output
#   ./data_synthesis/cache/cache_2/ <- step 2 intermediate output
#   ./data_synthesis/cache/cache_3a/<hashes>/ <- LightRAG KG files (step 3a)
#   ./data_synthesis/cache/cache_3b/ <- step 3b intermediate output
#   ./data_synthesis/cache/cache_5/ <- step 5 intermediate output
#   ./data_synthesis/cache/cache_7/ <- step 7 intermediate output
#   ./data_synthesis/result/        <- final QA pairs and statistics
#
# Environment variables (set via export or on the command line):
#   INPUT_FILE       - raw trajectory JSONL inside ./data_synthesis/input/ (required)
#   GENERATE_MODEL   - LLM for generation          (default: deepseek-v3.2)
#   VERIFY_MODEL     - LLM for verification        (default: gpt-5-mini)
#   PARALLEL         - concurrency level            (default: 1)
#   MAX_OBF_ITER     - max obfuscation iterations   (default: 5)
# ============================================================

set -euo pipefail

# ---------- configurable parameters ----------
INPUT_FILE="${INPUT_FILE:-}"
GENERATE_MODEL="${GENERATE_MODEL:-deepseek-v3.2}"
VERIFY_MODEL="${VERIFY_MODEL:-gpt-5-mini}"
PARALLEL="${PARALLEL:-1}"
MAX_OBF_ITER="${MAX_OBF_ITER:-5}"
# ---------------------------------------------

# Ensure the script is run from SMTL-main
if [[ ! -d "./data_synthesis" ]]; then
    echo "❌ ERROR: Please run this script from the SMTL-main project root."
    echo "   cd /path/to/SMTL-main && bash data_synthesis/run_pipeline.sh"
    exit 1
fi

# Ensure INPUT_FILE is provided
if [[ -z "$INPUT_FILE" ]]; then
    echo "❌ ERROR: INPUT_FILE is not set."
    echo "   Usage: INPUT_FILE=./data_synthesis/input/my_data.jsonl bash data_synthesis/run_pipeline.sh"
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "❌ ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

STEM="$(basename "$INPUT_FILE" .jsonl)"
STEM="${STEM%.json}"   # also strip .json if used

# Create all cache/result directories up front
mkdir -p \
    ./data_synthesis/cache/cache_1 \
    ./data_synthesis/cache/cache_2 \
    ./data_synthesis/cache/cache_3a \
    ./data_synthesis/cache/cache_3b \
    ./data_synthesis/cache/cache_5 \
    ./data_synthesis/cache/cache_7 \
    ./data_synthesis/result

echo "======================================================"
echo "  Data Synthesis Pipeline"
echo "======================================================"
echo "  Input file      : $INPUT_FILE"
echo "  Generate model  : $GENERATE_MODEL"
echo "  Verify model    : $VERIFY_MODEL"
echo "  Parallelism     : $PARALLEL"
echo "  Max obf. iters  : $MAX_OBF_ITER"
echo "======================================================"

# ---- Step 1: Process trajectories ----
echo ""
echo ">>> Step 1: Process trajectories"
STEP1_OUT="./data_synthesis/cache/cache_1/${STEM}_step1.jsonl"
python ./data_synthesis/step1_extract_urls_from_traj.py \
    --input "$INPUT_FILE" \
    --output "$STEP1_OUT"

# ---- Step 2: Broaden information ----
echo ""
echo ">>> Step 2: Broaden information"
STEP2_OUT="./data_synthesis/cache/cache_2/${STEM}_step2.jsonl"
python ./data_synthesis/step2_broaden_information.py \
    --input "$STEP1_OUT" \
    --output "$STEP2_OUT"

# ---- Step 3a: Construct graph (step 1 – build LightRAG KGs) ----
echo ""
echo ">>> Step 3a: Construct graph (build LightRAG knowledge graphs)"
python ./data_synthesis/step3_construct_graph_stage_a.py \
    --input "$STEP2_OUT" \
    --working-dir "./data_synthesis/cache/cache_3a" \
    --model "$GENERATE_MODEL"

# ---- Step 3b: Construct graph (step 2 – extract entities & relations) ----
echo ""
echo ">>> Step 3b: Construct graph (extract entities and relations)"
STEP3B_OUT="./data_synthesis/cache/cache_3b/${STEM}_step3b.jsonl"
python ./data_synthesis/step3_construct_graph_stage_b.py \
    --input "$STEP2_OUT" \
    --output "$STEP3B_OUT" \
    --working-dir "./data_synthesis/cache/cache_3a" \
    --model "$GENERATE_MODEL"

# ---- Step 4: Visualize graph (optional, comment out if not needed) ----
# echo ""
# echo ">>> Step 4: Visualize graph"
# python ./data_synthesis/step4_visualize_graph.py \
#     --input "$STEP3B_OUT" \
#     --mode summary

# ---- Step 5: Extract subgraphs ----
echo ""
echo ">>> Step 5: Extract subgraphs"
STEP5_OUT="./data_synthesis/cache/cache_5/${STEM}_subgraphs.jsonl"
python ./data_synthesis/step5_extract_subgraph.py \
    --input "$STEP3B_OUT" \
    --output "$STEP5_OUT" \
    --working-dir "./data_synthesis/cache/cache_3a" \
    --llm-model "$GENERATE_MODEL"

# ---- Step 6: Visualize subgraphs (optional, comment out if not needed) ----
# echo ""
# echo ">>> Step 6: Visualize subgraphs"
# python ./data_synthesis/step6_visualize_subgraph.py \
#     --input "$STEP5_OUT" \
#     --mode summary

# ---- Step 7: Layer-wise description generation ----
echo ""
echo ">>> Step 7: Layer-wise description generation"
STEP7_OUT="./data_synthesis/cache/cache_7/${STEM}_descriptions.jsonl"
python ./data_synthesis/step7_layerwise_description_generator.py \
    --graph_file "$STEP5_OUT" \
    --output_file "$STEP7_OUT" \
    --generate-model "$GENERATE_MODEL" \
    --verify-model "$VERIFY_MODEL" \
    --max-obfuscation-iterations "$MAX_OBF_ITER" \
    --parallel "$PARALLEL"

# ---- Step 8: Extract QA pairs ----
echo ""
echo ">>> Step 8: Extract QA pairs"
STEP8_OUT="./data_synthesis/result/${STEM}_qa.jsonl"
STEP8_STATS="./data_synthesis/result/${STEM}_qa_stats.json"
python ./data_synthesis/step8_get_qa.py \
    --input_file "$STEP7_OUT" \
    --output_file "$STEP8_OUT" \
    --stats_file "$STEP8_STATS"

echo ""
echo "======================================================"
echo "✅ Pipeline complete!"
echo "   Final QA pairs : $STEP8_OUT"
echo "   Statistics      : $STEP8_STATS"
echo "======================================================"
