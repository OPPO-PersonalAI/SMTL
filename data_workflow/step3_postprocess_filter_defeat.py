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
Filter evaluation results, keeping only data where judgement is 'correct'.
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter

from utils import read_jsonl, write_jsonl, read_json, write_json


def filter_correct_data(data):
    """Filter data, keeping only entries where judgement is 'correct'."""
    filtered_data = []
    
    for item in data:
        # Check if evaluation field exists
        evaluation = item.get("evaluation", {})
        judgement = evaluation.get("judgement", "")
        
        # Keep only data where judgement is 'correct'
        if judgement == "correct":
            # Ensure model_id field is preserved (set to 'unknown' if not present)
            if "model_id" not in item:
                item["model_id"] = "unknown"
            filtered_data.append(item)
    
    return filtered_data


def calculate_detailed_statistics(results, filter_stats=None, total_processed=None):
    """Calculate detailed statistics."""
    total = len(results)
    
    # Trajectory statistics
    trajectory_steps = []
    tool_usage = {}
    step_types = {}
    tool_usage_by_question = {}
    model_ids = {}
    
    for result in results:
        # Count model_id
        model_id = result.get("model_id", "unknown")
        model_ids[model_id] = model_ids.get(model_id, 0) + 1
        # Trajectory statistics
        agent_trajectory = result.get("agent_trajectory", [])
        question = result.get("question", "")
        
        if agent_trajectory:
            # Count steps in agent_trajectory
            step_count = len(agent_trajectory)
            trajectory_steps.append(step_count)
            
            # Count tools used per question
            question_tools = set()
            
            # Count step types and tool usage
            for step in agent_trajectory:
                step_name = step.get("name", "unknown")
                # Ensure step_name is a string
                if isinstance(step_name, dict):
                    step_name = str(step_name)
                elif not isinstance(step_name, str):
                    step_name = str(step_name)
                step_types[step_name] = step_types.get(step_name, 0) + 1
                
                # Count tool usage (from tool_calls in action steps)
                if step_name == "action" and "tool_calls" in step:
                    tool_calls = step["tool_calls"]
                    if isinstance(tool_calls, list):
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get("name", "unknown")
                                # Ensure tool_name is a string; convert if dict
                                if isinstance(tool_name, dict):
                                    tool_name = str(tool_name)
                                elif not isinstance(tool_name, str):
                                    tool_name = str(tool_name)
                                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                                question_tools.add(tool_name)
            
            # Record tools used per question
            if question_tools:
                tool_usage_by_question[question] = list(question_tools)
    
    # Calculate trajectory statistics
    trajectory_stats = {}
    if trajectory_steps:
        trajectory_stats = {
            "total_trajectories": len(trajectory_steps),
            "avg_steps": sum(trajectory_steps) / len(trajectory_steps),
            "median_steps": sorted(trajectory_steps)[len(trajectory_steps) // 2],
            "min_steps": min(trajectory_steps),
            "max_steps": max(trajectory_steps),
            "agent_step_distribution": {
                "1-5": sum(1 for s in trajectory_steps if 1 <= s <= 5),
                "6-10": sum(1 for s in trajectory_steps if 6 <= s <= 10),
                "11-15": sum(1 for s in trajectory_steps if 11 <= s <= 15),
                "16-20": sum(1 for s in trajectory_steps if 16 <= s <= 20),
                "21-25": sum(1 for s in trajectory_steps if 21 <= s <= 25),
                "26-30": sum(1 for s in trajectory_steps if 26 <= s <= 30),
                "31-40": sum(1 for s in trajectory_steps if 31 <= s <= 40),
                "41-50": sum(1 for s in trajectory_steps if 41 <= s <= 50),
                "51-60": sum(1 for s in trajectory_steps if 51 <= s <= 60),
                "61-80": sum(1 for s in trajectory_steps if 61 <= s <= 80),
                "81-100": sum(1 for s in trajectory_steps if 81 <= s <= 100),
                "101+": sum(1 for s in trajectory_steps if s > 100)
            }
        }
    
    # Calculate tool usage percentages
    total_tool_calls = sum(tool_usage.values())
    tool_usage_percentages = {}
    if total_tool_calls > 0:
        for tool, count in tool_usage.items():
            tool_usage_percentages[tool] = {
                "count": count,
                "percentage": count / total_tool_calls * 100
            }
    
    # Calculate step type percentages
    total_steps = sum(step_types.values())
    step_types_percentages = {}
    if total_steps > 0:
        for step_type, count in step_types.items():
            step_types_percentages[step_type] = {
                "count": count,
                "percentage": count / total_steps * 100
            }
    
    # Tool combination analysis
    tool_combinations = defaultdict(int)
    for question, tools in tool_usage_by_question.items():
        if len(tools) > 1:
            tool_combinations[tuple(sorted(tools))] += 1
    
    # Convert tuples to strings for JSON serialization
    tool_combinations_str = {}
    for combination, count in tool_combinations.items():
        combination_str = " + ".join(combination)
        tool_combinations_str[combination_str] = count
    
    # Build base statistics
    stats_result = {
        "filtering": {
            "filtered_count": total,
            "filter_rate": filter_stats.get('filter_rate', 0) if filter_stats else 0
        },
        "trajectory": trajectory_stats,
        "tool_usage": {
            "total_calls": total_tool_calls,
            "tools": tool_usage_percentages,
            "tool_combinations": tool_combinations_str
        },
        "step_types": {
            "total_steps": total_steps,
            "types": step_types_percentages
        },
        "model_distribution": model_ids
    }
    
    # If filter stats exist, merge them in
    if filter_stats is not None and total_processed is not None:
        stats_result["original_data"] = {
            "total_processed": total_processed,
            "filtered_count": filter_stats.get('filtered_count', 0),
            "filter_rate": filter_stats.get('filter_rate', 0),
            "judgement_distribution": filter_stats.get('judgement_distribution', {})
        }
    
    return stats_result

def print_detailed_report(stats):
    """Print detailed statistics report."""
    print("\n" + "="*80)
    print("📊 Detailed Filter Statistics Report")
    print("="*80)
    
    # Filter result statistics
    filter_stats = stats.get('filtering', {})
    print(f"🎯 Filter Results:")
    print(f"  Retained count: {filter_stats.get('filtered_count', 0)}")
    print(f"  Retention rate: {filter_stats.get('filter_rate', 0):.2f}%")
    
    # Original data statistics
    original_stats = stats.get('original_data', {})
    if original_stats:
        print(f"\n📥 Original Data Statistics:")
        print(f"  Total processed: {original_stats.get('total_processed', 0)}")
        print(f"  After filter: {original_stats.get('filtered_count', 0)}")
        print(f"  Retention rate: {original_stats.get('filter_rate', 0):.2f}%")
        
        judgement_dist = original_stats.get('judgement_distribution', {})
        if judgement_dist:
            print(f"  Evaluation result distribution:")
            for judgement, count in judgement_dist.items():
                print(f"    {judgement}: {count}")
    
    # Trajectory statistics
    trajectory_stats = stats.get('trajectory', {})
    if trajectory_stats:
        print(f"\n🔄 Trajectory Statistics:")
        print(f"  Total trajectories: {trajectory_stats.get('total_trajectories', 0)}")
        print(f"  Avg steps: {trajectory_stats.get('avg_steps', 0):.2f}")
        print(f"  Median steps: {trajectory_stats.get('median_steps', 0)}")
        print(f"  Min steps: {trajectory_stats.get('min_steps', 0)}")
        print(f"  Max steps: {trajectory_stats.get('max_steps', 0)}")
        
        step_dist = trajectory_stats.get('step_distribution', {})
        if step_dist:
            print(f"  Step distribution:")
            for range_name, count in step_dist.items():
                percentage = count / trajectory_stats.get('total_trajectories', 1) * 100
                print(f"    {range_name} steps: {count} ({percentage:.1f}%)")
    
    # Tool usage statistics
    tool_usage = stats.get('tool_usage', {})
    if tool_usage.get('tools'):
        print(f"\n🔧 Tool Usage Statistics:")
        print(f"  Total tool calls: {tool_usage.get('total_calls', 0)}")
        print(f"  Tool usage percentages:")
        # Sort by usage count
        sorted_tools = sorted(tool_usage['tools'].items(), 
                            key=lambda x: x[1]['count'], reverse=True)
        for tool_name, tool_stats in sorted_tools:
            print(f"    {tool_name}: {tool_stats['count']} ({tool_stats['percentage']:.1f}%)")
        
        # Tool combination analysis
        tool_combinations = tool_usage.get('tool_combinations', {})
        if tool_combinations:
            print(f"  Common tool combinations (top 10):")
            sorted_combinations = sorted(tool_combinations.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
            for combination, count in sorted_combinations:
                print(f"    {combination}: {count} times")
    
    # Step type statistics
    step_types = stats.get('step_types', {})
    if step_types.get('types'):
        print(f"\n📝 Step Type Statistics:")
        print(f"  Total steps: {step_types.get('total_steps', 0)}")
        print(f"  Step type distribution:")
        # Sort by usage count
        sorted_step_types = sorted(step_types['types'].items(), 
                                 key=lambda x: x[1]['count'], reverse=True)
        for step_type, type_stats in sorted_step_types:
            print(f"    {step_type}: {type_stats['count']} ({type_stats['percentage']:.1f}%)")
    
    # Model distribution statistics
    model_distribution = stats.get('model_distribution', {})
    if model_distribution:
        print(f"\n🤖 Model Distribution:")
        total_records = filter_stats.get('filtered_count', 0)
        for model_id, count in model_distribution.items():
            percentage = round(count / total_records * 100, 2) if total_records > 0 else 0
            print(f"  {model_id}: {count} records ({percentage}%)")
    
    print("="*80)


def process_directory(input_dir, output_dir):
    """Process all jsonl files in the directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Find all jsonl files
    jsonl_files = list(input_path.rglob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No jsonl files found in directory {input_dir}")
        return
    
    print(f"Found {len(jsonl_files)} jsonl files")
    
    total_processed = 0
    total_filtered = 0
    all_filtered_data = []  # Collect all filtered data for statistics
    judgement_distribution = defaultdict(int)  # Track judgement result distribution
    
    for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
        # Compute relative path
        relative_path = jsonl_file.relative_to(input_path)
        
        # Build output file path
        output_file = output_path / relative_path
        
        # Load data
        if str(jsonl_file).endswith('.jsonl'):
            data = read_jsonl(str(jsonl_file))
        else:
            data = read_json(str(jsonl_file))
        
        if not data:
            print(f"Skipping empty file: {jsonl_file}")
            continue
        
        # Track judgement result distribution
        for item in data:
            evaluation = item.get("evaluation", {})
            judgement = evaluation.get("judgement", "unknown")
            judgement_distribution[judgement] += 1
        
        # Filter data
        filtered_data = filter_correct_data(data)
        
        # Save filtered data
        if filtered_data:
            write_jsonl(output_file, filtered_data)
            print(f"✅ {relative_path}: {len(data)} -> {len(filtered_data)} (retained {len(filtered_data)/len(data)*100:.1f}%)")
            
            # Collect filtered data for statistics
            all_filtered_data.extend(filtered_data)
        else:
            print(f"⚠️  {relative_path}: {len(data)} -> 0 (no correct data)")
        
        total_processed += len(data)
        total_filtered += len(filtered_data)
    
    print(f"\n📊 Processing complete:")
    print(f"  Total processed: {total_processed}")
    print(f"  Retained: {total_filtered}")
    print(f"  Retention rate: {total_filtered/total_processed*100:.2f}%")
    print(f"  Output directory: {output_dir}")
    
    # Generate individual statistics for each processed file
    print(f"\n📊 Generating detailed statistics report...")
    
    # Reprocess each file to generate independent statistics
    for jsonl_file in jsonl_files:
        relative_path = jsonl_file.relative_to(input_path)
        output_file = output_path / relative_path
        
        # Only generate statistics for files that actually have data
        if not (output_path / relative_path).exists():
            continue
            
        # Load current file data
        if str(jsonl_file).endswith('.jsonl'):
            file_data = read_jsonl(str(jsonl_file))
        else:
            file_data = read_json(str(jsonl_file))
        
        if not file_data:
            continue
        
        # Track current file judgement result distribution
        file_judgement_distribution = defaultdict(int)
        for item in file_data:
            evaluation = item.get("evaluation", {})
            judgement = evaluation.get("judgement", "unknown")
            file_judgement_distribution[judgement] += 1
        
        # Filter current file data
        file_filtered_data = filter_correct_data(file_data)
        
        # Build current file filter statistics
        file_filter_stats = {
            'filtered_count': len(file_filtered_data),
            'filter_rate': len(file_filtered_data)/len(file_data)*100 if len(file_data) > 0 else 0,
            'judgement_distribution': dict(file_judgement_distribution)
        }
        
        # Calculate current file detailed statistics
        file_detailed_stats = calculate_detailed_statistics(file_filtered_data, file_filter_stats, len(file_data))
        
        # Generate statistics file path
        if str(output_file).endswith('.jsonl'):
            stats_file = str(output_file).replace('.jsonl', '_stats.json')
        else:
            stats_file = str(output_file) + '_stats.json'
        
        # Save current file statistics
        write_json(stats_file, file_detailed_stats)
        print(f"📊 Statistics report saved to: {stats_file}")
    
    # Generate overall statistics report
    if all_filtered_data:
        # Build overall filter statistics
        overall_filter_stats = {
            'filtered_count': total_filtered,
            'filter_rate': total_filtered/total_processed*100 if total_processed > 0 else 0,
            'judgement_distribution': dict(judgement_distribution)
        }
        
        # Calculate overall detailed statistics
        overall_detailed_stats = calculate_detailed_statistics(all_filtered_data, overall_filter_stats, total_processed)
        
        # Print overall statistics report
        print_detailed_report(overall_detailed_stats)


def main():
    parser = argparse.ArgumentParser(description='Filter evaluation results, retaining only entries with judgement=correct')
    parser.add_argument('--input_dir', type=str, 
                       default='./data_workflow/results2',
                       help='Input directory path')
    parser.add_argument('--output_dir', type=str,
                       default='./data_workflow/results3',
                       help='Output directory path')
    parser.add_argument('--dry_run', action='store_true', help='Dry run mode, no files will be written')
    
    args = parser.parse_args()

    print("="*80)
    print("Filter Evaluation Results - Retain Correct Data Only")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    if args.dry_run:
        print("🔍 Dry run mode - no files will be written")
    
    # Process directory
    process_directory(args.input_dir, args.output_dir)

    print("="*80)
    print("Processing complete")
    print("="*80)


if __name__ == "__main__":
    main()
