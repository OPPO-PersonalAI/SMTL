#!/usr/bin/env python
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

"""
Quick single-question inference test.

Usage:
    cd /path/to/SMTL-main
    python inference/main.py --question "Who won the 2024 Nobel Prize in Physics?"

Or interactively (no --question argument):
    python inference/main.py
"""

import sys
import os
import logging
import argparse

# Ensure imports resolve relative to the inference/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)

# ── Minimal logging setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # suppress verbose inference logs
    format="%(levelname)s - %(message)s",
)

# ── Import the core inference function and shared config ───────────────────────
from inference_web_agent import process_single_data, MODEL_URLS, INFER_KWARGS


def run_single_question(question: str) -> str:
    """
    Run inference for a single question and return the final answer string.
    """
    # Use the first model URL (same default as the main script)
    fixed_url = MODEL_URLS[0]

    print(f"\n[Model URL]  {fixed_url}")
    print(f"[Question]   {question}\n")
    print("Running inference ... (this may take a while)\n")

    # A lightweight copy of INFER_KWARGS – reduce parallelism to 1 for single query
    kwargs = dict(INFER_KWARGS)
    kwargs["parallel"] = 1

    conversation_history, failed_reason, prediction, timing_stats = process_single_data(
        question, fixed_url=fixed_url, **kwargs
    )

    if failed_reason:
        print(f"[Error] Inference failed: {failed_reason}")
        return ""

    total_time = timing_stats.get("total_time", 0)
    total_steps = timing_stats.get("total_steps", 0)

    print("=" * 60)
    print(f"[Answer]     {prediction}")
    print("-" * 60)
    print(f"[Steps]      {total_steps}")
    print(f"[Total time] {total_time:.1f}s")
    print("=" * 60)

    return prediction


def main():
    parser = argparse.ArgumentParser(description="Single-question inference test for SMTL.")
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="The question to ask the model.",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help="Override the model base URL (e.g. http://0.0.0.0:8000/v1).",
    )
    args = parser.parse_args()

    # Override MODEL_URLS if user provides a URL
    if args.model_url:
        import inference_web_agent as _agent
        _agent.MODEL_URLS = [args.model_url]
        print(f"[Override] MODEL_URL -> {args.model_url}")

    # Get question from CLI arg or interactive prompt
    question = args.question
    if not question:
        question = input("Enter your question: ").strip()
    if not question:
        print("No question provided. Exiting.")
        sys.exit(1)

    run_single_question(question)


if __name__ == "__main__":
    main()