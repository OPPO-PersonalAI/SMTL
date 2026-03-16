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
Statistics script for inference outputs.
It reports overall accuracy/error rates, level-wise breakdown,
step distribution, and tool-call usage statistics.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import re


def read_jsonl(file_path: str) -> List[Dict]:
    """Read a JSONL file."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as exc:
        print(f"Failed to read file: {exc}")
        return []
    return data


def count_tool_calls_in_trajectory(trajectory: List[Dict]) -> Tuple[int, Dict[str, int]]:
    """
    Count tool calls and tool-type distribution from a trajectory.

    Args:
        trajectory: Conversation trajectory list.

    Returns:
        (total_tool_calls, tool_type_counter)
    """
    total_calls = 0
    tool_types = defaultdict(int)

    for turn in trajectory:
        if turn.get("role") != "assistant":
            continue

        content = turn.get("content", "")
        tool_call_matches = re.findall(r"<tool_call>.*?</tool_call>", content, re.DOTALL)
        total_calls += len(tool_call_matches)

        for tool_call in tool_call_matches:
            try:
                json_match = re.search(r"\{.*\}", tool_call, re.DOTALL)
                if json_match:
                    tool_data = json.loads(json_match.group())
                    tool_name = tool_data.get("name", "unknown")
                    tool_types[tool_name] += 1
            except Exception:
                tool_types["unknown"] += 1

    return total_calls, dict(tool_types)


def get_percentile(sorted_data: List[float], p: float) -> float:
    """Compute percentile using linear interpolation."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d = k - f
    return sorted_data[f] * (1 - d) + sorted_data[c] * d


def calculate_step_stats(steps: List[int]) -> Dict[str, Any]:
    """Compute detailed step statistics."""
    if not steps:
        return None

    sorted_steps = sorted(steps)
    avg_steps = sum(steps) / len(steps)
    max_step = sorted_steps[-1]

    return {
        "avg_steps": f"{avg_steps:.2f}",
        "min_steps": sorted_steps[0],
        "max_steps": max_step,
        "max_steps_count": steps.count(max_step),
        "p10": f"{get_percentile(sorted_steps, 10):.2f}",
        "p25": f"{get_percentile(sorted_steps, 25):.2f}",
        "p50": f"{get_percentile(sorted_steps, 50):.2f}",
        "p75": f"{get_percentile(sorted_steps, 75):.2f}",
        "p90": f"{get_percentile(sorted_steps, 90):.2f}",
    }


def analyze_data(data: List[Dict]) -> Dict[str, Any]:
    """Analyze records and produce aggregated statistics."""
    if not data:
        return {}

    total_count = len(data)
    correct_count = 0
    incorrect_count = 0

    level_stats = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0})

    correct_steps = []
    incorrect_steps = []
    correct_tool_calls = []
    incorrect_tool_calls = []
    correct_tool_types = defaultdict(int)
    incorrect_tool_types = defaultdict(int)

    for item in data:
        level = item.get("Level", "-1")
        llm_judge = item.get("llm_judge", 0)
        trajectory = item.get("trajectory", [])

        is_correct = llm_judge == 1
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        level_stats[level]["total"] += 1
        if is_correct:
            level_stats[level]["correct"] += 1
        else:
            level_stats[level]["incorrect"] += 1

        steps = len([turn for turn in trajectory if turn.get("role") == "assistant"])
        tool_calls, tool_types = count_tool_calls_in_trajectory(trajectory)

        if is_correct:
            correct_steps.append(steps)
            correct_tool_calls.append(tool_calls)
            for tool_type, count in tool_types.items():
                correct_tool_types[tool_type] += count
        else:
            incorrect_steps.append(steps)
            incorrect_tool_calls.append(tool_calls)
            for tool_type, count in tool_types.items():
                incorrect_tool_types[tool_type] += count

    results = {
        "overall": {
            "total_count": total_count,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "accuracy": f"{correct_count / total_count * 100:.2f}%" if total_count > 0 else "0%",
            "error_rate": f"{incorrect_count / total_count * 100:.2f}%" if total_count > 0 else "0%",
        },
        "level_stats": {},
        "step_stats": {},
        "tool_call_stats": {},
    }

    for level, stats in level_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        incorrect = stats["incorrect"]
        results["level_stats"][f"Level_{level}"] = {
            "total_count": total,
            "correct_count": correct,
            "incorrect_count": incorrect,
            "accuracy": f"{correct / total * 100:.2f}%" if total > 0 else "0%",
            "error_rate": f"{incorrect / total * 100:.2f}%" if total > 0 else "0%",
        }

    if correct_steps:
        results["step_stats"]["correct_cases"] = calculate_step_stats(correct_steps)
    if incorrect_steps:
        results["step_stats"]["incorrect_cases"] = calculate_step_stats(incorrect_steps)

    if correct_tool_calls:
        avg_tool_calls = sum(correct_tool_calls) / len(correct_tool_calls)
        avg_tools_per_step = avg_tool_calls / (sum(correct_steps) / len(correct_steps)) if correct_steps else 0
        results["tool_call_stats"]["correct_cases"] = {
            "avg_tool_calls": f"{avg_tool_calls:.2f}",
            "avg_tools_per_step": f"{avg_tools_per_step:.2f}",
            "tool_type_distribution": dict(correct_tool_types),
        }

    if incorrect_tool_calls:
        avg_tool_calls = sum(incorrect_tool_calls) / len(incorrect_tool_calls)
        avg_tools_per_step = avg_tool_calls / (sum(incorrect_steps) / len(incorrect_steps)) if incorrect_steps else 0
        results["tool_call_stats"]["incorrect_cases"] = {
            "avg_tool_calls": f"{avg_tool_calls:.2f}",
            "avg_tools_per_step": f"{avg_tools_per_step:.2f}",
            "tool_type_distribution": dict(incorrect_tool_types),
        }

    return results


def format_stats_report(results: Dict[str, Any]) -> str:
    """Format analysis result into a plain-text report."""
    report = []
    report.append("=" * 60)
    report.append("Inference Statistics Report")
    report.append("=" * 60)
    report.append("")

    if "overall" in results:
        stats = results["overall"]
        report.append("[Overall]")
        report.append(f"Total count: {stats['total_count']}")
        report.append(f"Correct count: {stats['correct_count']}")
        report.append(f"Incorrect count: {stats['incorrect_count']}")
        report.append(f"Accuracy: {stats['accuracy']}")
        report.append(f"Error rate: {stats['error_rate']}")
        report.append("")

    if "level_stats" in results and results["level_stats"]:
        report.append("[Level Stats]")
        for level, stats in results["level_stats"].items():
            report.append(f"{level}:")
            report.append(f"  Total count: {stats['total_count']}")
            report.append(f"  Correct count: {stats['correct_count']}")
            report.append(f"  Incorrect count: {stats['incorrect_count']}")
            report.append(f"  Accuracy: {stats['accuracy']}")
            report.append(f"  Error rate: {stats['error_rate']}")
        report.append("")

    if "step_stats" in results and results["step_stats"]:
        report.append("[Step Stats]")
        for case_type, stats in results["step_stats"].items():
            report.append(f"{case_type}:")
            report.append(f"  Avg steps: {stats['avg_steps']}")
            report.append(f"  Min steps: {stats['min_steps']}")
            report.append(f"  Max steps: {stats['max_steps']} (count {stats['max_steps_count']})")
            report.append("  Percentiles:")
            report.append(f"    10%: {stats['p10']}")
            report.append(f"    25%: {stats['p25']}")
            report.append(f"    50% (median): {stats['p50']}")
            report.append(f"    75%: {stats['p75']}")
            report.append(f"    90%: {stats['p90']}")
        report.append("")

    if "tool_call_stats" in results and results["tool_call_stats"]:
        report.append("[Tool Call Stats]")
        for case_type, stats in results["tool_call_stats"].items():
            report.append(f"{case_type}:")
            report.append(f"  Avg tool calls: {stats['avg_tool_calls']}")
            report.append(f"  Avg tools per step: {stats['avg_tools_per_step']}")
            report.append("  Tool type distribution:")
            for tool_type, count in stats["tool_type_distribution"].items():
                report.append(f"    {tool_type}: {count}")
        report.append("")

    return "\n".join(report)


def main():
    """Entry point."""
    if len(sys.argv) != 2:
        print("Usage:")
        print("  Single file: python cal_stats.py <input_jsonl_file>")
        print("Example:")
        print("  python cal_stats.py sample.round_0.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: file does not exist: {input_file}")
        sys.exit(1)

    print(f"Analyzing file: {input_file}")

    data = read_jsonl(input_file)
    if not data:
        print("Error: failed to read data or file is empty")
        sys.exit(1)

    print(f"Loaded {len(data)} records")

    results = analyze_data(data)
    report = format_stats_report(results)

    output_file = input_file.replace(".jsonl", ".output_stats.txt")
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(report)
        print(f"Statistics report saved to: {output_file}")
    except Exception as exc:
        print(f"Failed to save report: {exc}")
        sys.exit(1)

    print("\n" + report)


if __name__ == "__main__":
    main()
