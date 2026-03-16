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
Evaluate consistency between predicted answers and ground truth using LLM.
"""

import json
import os
import time
import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from FlashOAgents import OpenAIServerModel, custom_role_conversions
from collections import defaultdict, Counter


from utils import read_json, read_jsonl, write_jsonl, write_json


# Load environment variables
load_dotenv(override=True)


# Evaluation prompt template
LLM_EVALUATION_PROMPT = """
Please determine whether the Answer 1 is semantically equivalent to the Answer 2, given the Question.

Question: {question}  
Answer 1: {gt_answer}  
Answer 2: {pred_answer}  

**Evaluation Principles**:
1. The judgement is based on whether the Answer 1 conveys the same meaning as the Answer 2.
2. Only semantic meaning matters. Differences in capitalization, punctuation, grammar (including prepositions), expression order, wording style, or measurement units do NOT matter.
3. There may be noise in the Answer 1, but if the Answer 2 semantically contains the answer in Answer 1 or matches the meaning of the Answer 1, regardless of the noise, the judgement is **correct**.
4. Otherwise, if the answer 1 does not contain the answer in Answer 2 or does not match the meaning of the Answer 2, the judgement is **incorrect**.

**Output Format**:
{{
  "rationale": "your rationale for the judgement",
  "judgement": "correct or incorrect"
}}
""".strip()


def evaluate_single_item(item, model, max_retries=3):
    """Evaluate answer consistency for a single item."""
    
    # Extract required fields
    question = item.get("question", "")
    gt_answer = item.get("golden_answer", "")
    pred_answer = item.get("agent_result", "")
    
    # Return error if required fields are missing
    if not question or not gt_answer or not pred_answer:
        return {
            **item,
            "evaluation": {
                "rationale": "Missing required fields (question, golden_answer, or agent_result)",
                "judgement": "error",
                "error": "Missing required fields"
            }
        }
    
    # Build evaluation prompt
    prompt = LLM_EVALUATION_PROMPT.format(
        question=question,
        gt_answer=gt_answer,
        pred_answer=pred_answer
    )
    
    # Retry mechanism
    for attempt in range(max_retries):
        try:
            # Call LLM for evaluation
            response = model([{"role": "user", "content": prompt}])
            response_content = response.content.strip()
            
            # Try to parse JSON response
            try:
                # Extract JSON portion
                if "{" in response_content and "}" in response_content:
                    start_idx = response_content.find("{")
                    end_idx = response_content.rfind("}") + 1
                    json_str = response_content[start_idx:end_idx]
                    evaluation_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                
                # Validate required fields
                if "judgement" not in evaluation_result:
                    raise ValueError("Missing 'judgement' field")
                
                if "rationale" not in evaluation_result:
                    evaluation_result["rationale"] = "No rationale provided"
                
                # Validate judgement value
                if evaluation_result["judgement"] not in ["correct", "incorrect"]:
                    evaluation_result["judgement"] = "incorrect"
                    evaluation_result["rationale"] += " (Invalid judgement value)"
                
                return {
                    **item,
                    "evaluation": evaluation_result
                }
                
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    return {
                        **item,
                        "evaluation": {
                            "rationale": f"Failed to parse model response: {str(e)}",
                            "judgement": "error",
                            "error": f"JSON parsing error: {str(e)}",
                            "raw_response": response_content
                        }
                    }
                else:
                    time.sleep(1)  # Wait before retry
                    continue
                    
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    **item,
                    "evaluation": {
                        "rationale": f"Model evaluation failed: {str(e)}",
                        "judgement": "error",
                        "error": str(e)
                    }
                }
            else:
                time.sleep(1)  # Wait before retry
                continue
    
    # If all retries fail
    return {
        **item,
        "evaluation": {
            "rationale": "All retry attempts failed",
            "judgement": "error",
            "error": "Max retries exceeded"
        }
    }


def is_tool_call_error(observations, tool_name):
    """Determine if a tool call failed based on tool name and observations"""
    if not observations:
        return False
    
    observations_lower = observations.lower()
    
    # Check error patterns by tool type
    if tool_name == "crawl_page":
        # Error patterns for crawl_page tool (including internal read_page errors)
        error_patterns = [
            "error reading page after",  # read_page error
            "unexpected error in page reading",  # read_page error
            "invalid url format",  # crawl_page URL validation error
            "content extraction failed"  # crawl_page content extraction error
        ]
        return any(pattern in observations_lower for pattern in error_patterns)
    
    elif tool_name == "web_search":
        # Error patterns for web_search tool
        error_patterns = [
            "search failed after",
            "unexpected error in web search",
            "query is empty"
        ]
        return any(pattern in observations_lower for pattern in error_patterns)
    
    elif tool_name == "wiki_search":
        # Error patterns for wiki_search tool
        error_patterns = [
            "wikipedia api error",
            "request to wikipedia api timed out",
            "network error occurred",
            "unexpected error"
        ]
        return any(pattern in observations_lower for pattern in error_patterns)
    
    else:
        # Generic error patterns
        error_patterns = [
            "error",
            "failed",
            "timeout",
            "network error",
            "api error",
            "unexpected error"
        ]
        return any(pattern in observations_lower for pattern in error_patterns)

def analyze_tool_errors(agent_trajectory):
    """Analyze tool call errors in the agent trajectory"""
    if not agent_trajectory:
        return {"total_tool_calls": 0, "error_tool_calls": 0, "error_rate": 0.0, "tool_errors": {}}
    
    total_tool_calls = 0
    error_tool_calls = 0
    tool_errors = {}
    
    for step in agent_trajectory:
        if step.get("name") == "action" and "tool_calls" in step:
            tool_calls = step.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "unknown")
                        total_tool_calls += 1
                        
                        # Check if tool call succeeded
                        observations = step.get("obs", "")
                        if is_tool_call_error(observations, tool_name):
                            error_tool_calls += 1
                            tool_errors[tool_name] = tool_errors.get(tool_name, 0) + 1
    
    error_rate = (error_tool_calls / total_tool_calls * 100) if total_tool_calls > 0 else 0.0
    
    return {
        "total_tool_calls": total_tool_calls,
        "error_tool_calls": error_tool_calls,
        "error_rate": error_rate,
        "tool_errors": tool_errors
    }


def should_skip_due_to_tool_errors(result, max_error_rate=3.0, min_tool_calls=2):
    """Determine whether to skip saving due to excessive tool call errors"""
    agent_trajectory = result.get("agent_trajectory", [])
    tool_analysis = analyze_tool_errors(agent_trajectory)
    
    # If too few tool calls, do not skip
    if tool_analysis["total_tool_calls"] < min_tool_calls:
        return False, "Insufficient tool calls"
    
    # If error rate exceeds threshold, skip
    if tool_analysis["error_rate"] > max_error_rate:
        return True, f"Tool call error rate too high: {tool_analysis['error_rate']:.1f}% (threshold: {max_error_rate}%)"
    
    # Specifically check critical tool errors
    critical_tools = ["web_search", "crawl_page", "wiki_search"]
    critical_errors = 0
    critical_total = 0
    
    for step in agent_trajectory:
        if step.get("name") == "action" and "tool_calls" in step:
            tool_calls = step.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "unknown")
                        if tool_name in critical_tools:
                            critical_total += 1
                            observations = step.get("obs", "")
                            if is_tool_call_error(observations, tool_name):
                                critical_errors += 1
    
    # If critical tool error rate is too high, also skip
    if critical_total > 0:
        critical_error_rate = (critical_errors / critical_total * 100)
        if critical_error_rate > max_error_rate:
            return True, f"Critical tool ({', '.join(critical_tools)}) error rate too high: {critical_error_rate:.1f}% ({critical_errors}/{critical_total}) (threshold: {max_error_rate}%)"
    
    return False, f"Tool calls normal (overall: {tool_analysis['error_tool_calls']}/{tool_analysis['total_tool_calls']}, critical tools: {critical_errors}/{critical_total})"


def process_batch(items, model, output_file, stats_file, existing_results, max_workers=10, max_tool_error_rate=50.0, min_tool_calls=2, verbose_tool_errors=False):
    """Concurrently process a batch of data, saving each result immediately"""
    results = []
    skipped_errors = 0  # Count skipped error records
    skipped_tool_errors = 0  # Count records skipped due to tool errors
    file_lock = threading.Lock()
    
    def safe_append_and_save(result):
        nonlocal skipped_errors, skipped_tool_errors
        with file_lock:
            # Check if judgement is error; skip saving if so
            judgement = result.get("evaluation", {}).get("judgement", "")
            if judgement == "error":
                skipped_errors += 1
                print(f"\rSkipping evaluation error record: {result.get('question', 'Unknown')[:50]}... (skipped: {skipped_errors})", end='', flush=True)
                return
            
            # Check tool call errors
            should_skip, skip_reason = should_skip_due_to_tool_errors(result, max_tool_error_rate, min_tool_calls)
            if should_skip:
                skipped_tool_errors += 1
                if verbose_tool_errors:
                    print(f"\rSkipping tool error record: {result.get('question', 'Unknown')[:50]}... ({skip_reason}) (skipped: {skipped_tool_errors})", end='', flush=True)
                else:
                    print(f"\rSkipping tool error record: {result.get('question', 'Unknown')[:50]}... (skipped: {skipped_tool_errors})", end='', flush=True)
                return
            
            results.append(result)
            # Save result immediately
            write_jsonl(output_file, [result], mode='a')
            
            # Show real-time progress every 10 records
            if len(results) % 10 == 0:
                all_results = existing_results + results
                stats = calculate_detailed_statistics(all_results)
                
                # Display real-time progress
                eval_stats = stats.get('evaluation', {})
                print(f"\rProcessed: {len(results)}/{len(items)} | "
                      f"Correct: {eval_stats.get('correct', 0)} ({eval_stats.get('accuracy', 0):.1f}%) | "
                      f"Incorrect: {eval_stats.get('incorrect', 0)} | "
                      f"Failed: {eval_stats.get('errors', 0)} | "
                      f"Skipped: {skipped_errors + skipped_tool_errors}", end='', flush=True)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(evaluate_single_item, item, model) 
            for item in items
        ]
        
        # Collect results and save immediately
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = future.result()
                safe_append_and_save(result)
            except Exception as e:
                print(f"Error processing task: {e}")
                error_result = {
                    "evaluation": {
                        "rationale": f"Task processing failed: {str(e)}",
                        "judgement": "error",
                        "error": str(e)
                    }
                }
                safe_append_and_save(error_result)
    
    # Newline after processing
    print()  # Newline to avoid mixing progress info with subsequent output
    print(f"This batch skipped {skipped_errors} evaluation error records")
    print(f"This batch skipped {skipped_tool_errors} tool error records")
    return results, skipped_errors, skipped_tool_errors


def load_existing_results(output_file):
    """Load existing results file, return set of already-processed questions"""
    if not os.path.exists(output_file):
        return set(), []
    
    processed_questions = set()
    existing_results = []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if 'question' in item:
                            processed_questions.add(item['question'])
                            existing_results.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {e}")
    except Exception as e:
        print(f"Warning: Error reading existing results file: {e}")
    
    return processed_questions, existing_results


def calculate_detailed_statistics(results):
    """Calculate detailed statistics"""
    total = len(results)
    correct = 0
    incorrect = 0
    errors = 0
    
    # Trajectory statistics
    trajectory_steps = []
    tool_usage = {}
    step_types = {}
    tool_usage_by_question = {}
    step_count_by_accuracy = {"correct": [], "incorrect": [], "error": []}
    trajectory_lengths = []
    
    for result in results:
        # Evaluation result statistics
        judgement = result.get("evaluation", {}).get("judgement", "error")
        if judgement == "correct":
            correct += 1
        elif judgement == "incorrect":
            incorrect += 1
        else:
            errors += 1
        
        # Trajectory statistics
        agent_trajectory = result.get("agent_trajectory", [])

        question = result.get("question", "")
        
        if agent_trajectory:
            # Count steps
            step_count = len(agent_trajectory)
            trajectory_steps.append(step_count)
            # Categorize steps by accuracy
            step_count_by_accuracy[judgement].append(step_count)
            
            # Count tools used per question
            question_tools = set()
            
            # Count step types
            for step in agent_trajectory:
                step_name = step.get("name", "unknown")
                step_types[step_name] = step_types.get(step_name, 0) + 1
                
                # Calculate trajectory length
                obs = step.get("obs", "")
                obs = obs if isinstance(obs, str) else str(obs)
                think = step.get("think", "")
                think = think if isinstance(think, str) else str(think)
                tool_calls = step.get("tool_calls", "")
                tool_calls = tool_calls if isinstance(tool_calls, str) else str(tool_calls)
                trajectory_length = len(obs) + len(think) + len(tool_calls)
                trajectory_lengths.append(trajectory_length)

                # Count tool usage
                if step_name == "action" and "tool_calls" in step:
                    tool_calls = step["tool_calls"]
                    if isinstance(tool_calls, list):
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get("name", "unknown")
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
            "avg_length": sum(trajectory_lengths) / total,
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
            },
            "step_count_by_accuracy": {
                "correct": {
                    "avg": sum(step_count_by_accuracy["correct"]) / len(step_count_by_accuracy["correct"]) if step_count_by_accuracy["correct"] else 0,
                    "count": len(step_count_by_accuracy["correct"])
                },
                "incorrect": {
                    "avg": sum(step_count_by_accuracy["incorrect"]) / len(step_count_by_accuracy["incorrect"]) if step_count_by_accuracy["incorrect"] else 0,
                    "count": len(step_count_by_accuracy["incorrect"])
                },
                "error": {
                    "avg": sum(step_count_by_accuracy["error"]) / len(step_count_by_accuracy["error"]) if step_count_by_accuracy["error"] else 0,
                    "count": len(step_count_by_accuracy["error"])
                }
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
    
    return {
        "evaluation": {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "errors": errors,
            "accuracy": correct / total * 100 if total > 0 else 0,
            "error_rate": errors / total * 100 if total > 0 else 0
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
        }
    }


def print_detailed_report(stats):
    """Print detailed statistics report"""
    print("\n" + "="*80)
    print("📊 Detailed Evaluation Statistics Report")
    print("="*80)
    
    # Evaluation result statistics
    eval_stats = stats.get('evaluation', {})
    print(f"🎯 Evaluation Results:")
    print(f"  Total: {eval_stats.get('total', 0)}")
    print(f"  Correct: {eval_stats.get('correct', 0)} ({eval_stats.get('accuracy', 0):.2f}%)")
    print(f"  Incorrect: {eval_stats.get('incorrect', 0)} ({100-eval_stats.get('accuracy', 0)-eval_stats.get('error_rate', 0):.2f}%)")
    print(f"  Evaluation Failed: {eval_stats.get('errors', 0)} ({eval_stats.get('error_rate', 0):.2f}%)")
    print(f"  Accuracy: {eval_stats.get('accuracy', 0):.2f}%")
    
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
        
        # Steps by accuracy category
        step_by_accuracy = trajectory_stats.get('step_count_by_accuracy', {})
        if step_by_accuracy:
            print(f"  Avg steps by accuracy category:")
            for accuracy_type, data in step_by_accuracy.items():
                if data['count'] > 0:
                    print(f"    {accuracy_type}: {data['avg']:.2f} steps (samples: {data['count']})")
    
    # Tool usage statistics
    tool_usage = stats.get('tool_usage', {})
    if tool_usage.get('tools'):
        print(f"\n🔧 Tool Usage Statistics:")
        print(f"  Total tool calls: {tool_usage.get('total_calls', 0)}")
        print(f"  Tool distribution:")
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
                tools_str = " + ".join(combination)
                print(f"    {tools_str}: {count} times")
    
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
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate consistency between predicted and ground-truth answers')
    parser.add_argument('--input_file', type=str, required=True, help='Input file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--max_workers', type=int, default=50, help='Concurrency')
    parser.add_argument('--sample_num', type=int, default=None, help='Sample size limit')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint, skip already-processed data')
    parser.add_argument('--skip_detailed_stats', action='store_true', help='Skip detailed statistics analysis')
    parser.add_argument('--max_tool_error_rate', type=float, default=100.0, help='Tool call error rate threshold; records above this will be skipped (default: 100.0%%)')
    parser.add_argument('--min_tool_calls', type=int, default=0, help='Minimum tool calls; below this value error rate check is skipped (default: 0)')
    parser.add_argument('--verbose_tool_errors', action='store_true', help='Show detailed tool error information')
    
    args = parser.parse_args()
    
    # Check required environment variables
    model_name = os.environ.get("DEFAULT_MODEL")
    if not model_name:
        print("Error: DEFAULT_MODEL environment variable is not set")
        exit(1)

    # Initialize model
    print("Initializing evaluation model...")
    model = OpenAIServerModel(
        model_name,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=2048,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
        temperature=0.1
    )
    
    # Determine output file path
    input_filename = Path(args.input_file).stem
    output_file = Path(args.output_dir) / f"{input_filename}_evaluated.jsonl"
    
    # Load data
    print(f"Loading data file: {args.input_file}")
    if args.input_file.endswith('.jsonl'):
        data = read_jsonl(args.input_file)
    else:
        data = read_json(args.input_file)
    
    if not data:
        print("No valid data found")
        return
    
    # Limit sample size
    if args.sample_num:
        data = data[:args.sample_num]
        print(f"Limiting sample size to: {args.sample_num}")
    
    # Resume from checkpoint
    existing_results = []
    if args.resume:
        print(f"Checking existing results file: {output_file}")
        processed_questions, existing_results = load_existing_results(output_file)
        
        if processed_questions:
            print(f"Found {len(processed_questions)} already-processed questions")
            # Filter out already-processed data
            original_count = len(data)
            data = [item for item in data if item.get('question', '') not in processed_questions]
            filtered_count = len(data)
            print(f"Remaining after filtering: {filtered_count} items (original: {original_count})")
        else:
            print("No existing results found, starting from scratch")
    else:
        print(f"Starting from scratch, will overwrite existing results file: {output_file}")
    
    if not data:
        print("All data already processed, nothing to do")
        return
    
    print(f"Total items to evaluate: {len(data)}")
    print(f"Tool error filter config: error rate threshold={args.max_tool_error_rate}%, min tool calls={args.min_tool_calls}")
    if args.verbose_tool_errors:
        print("Verbose tool error information: enabled")
    
    # Determine statistics file path
    stats_file = Path(args.output_dir) / f"{input_filename}_stats.json"
    
    # Concurrent processing (each result saved immediately)
    print("Starting evaluation...")
    print(f"Results will be saved in real-time to: {output_file}")
    print(f"Statistics will be periodically updated to: {stats_file}")
    
    # If resume mode and existing results, append new results
    if args.resume and existing_results:
        print(f"Resume mode: appending new results to existing {len(existing_results)} records")
        new_results, skipped_errors, skipped_tool_errors = process_batch(data, model, output_file, stats_file, existing_results, args.max_workers, args.max_tool_error_rate, args.min_tool_calls, args.verbose_tool_errors)
        all_results = existing_results + new_results
    else:
        # Start fresh: clear file and begin saving
        print("Start fresh mode: clearing output file and beginning to save")
        # Clear output file
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
        new_results, skipped_errors, skipped_tool_errors = process_batch(data, model, output_file, stats_file, [], args.max_workers, args.max_tool_error_rate, args.min_tool_calls, args.verbose_tool_errors)
        all_results = new_results
    
    # Generate and save detailed statistics
    print(f"\n📊 Generating detailed statistics report...")
    detailed_stats = calculate_detailed_statistics(all_results)
    
    # Save detailed statistics
    print(f"Saving detailed statistics to: {stats_file}")
    write_json(stats_file, detailed_stats)
    
    # Print detailed statistics report
    print_detailed_report(detailed_stats)
    
    # Print processing summary
    print("\n" + "="*80)
    print("📋 Processing Summary")
    print("="*80)
    eval_stats = detailed_stats.get('evaluation', {})
    print(f"📊 Evaluation Results:")
    print(f"  Total: {eval_stats.get('total', 0)}")
    print(f"    - Existing: {len(existing_results)}")
    print(f"    - Newly processed: {len(new_results)}")
    print(f"  Correct: {eval_stats.get('correct', 0)} ({eval_stats.get('accuracy', 0):.2f}%)")
    print(f"  Incorrect: {eval_stats.get('incorrect', 0)} ({100-eval_stats.get('accuracy', 0)-eval_stats.get('error_rate', 0):.2f}%)")
    print(f"  Evaluation Failed: {eval_stats.get('errors', 0)} ({eval_stats.get('error_rate', 0):.2f}%)")
    print(f"  Accuracy: {eval_stats.get('accuracy', 0):.2f}%")
    print(f"  Skipped evaluation error records: {skipped_errors}")
    print(f"  Skipped tool error records: {skipped_tool_errors}")
    print(f"  Total skipped: {skipped_errors + skipped_tool_errors}")
    
    # Trajectory statistics
    trajectory_stats = eval_stats.get('trajectory', {})
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
    tool_usage = eval_stats.get('tool_usage', {})
    if tool_usage.get('tools'):
        print(f"\n🔧 Tool Usage Statistics:")
        print(f"  Total tool calls: {tool_usage.get('total_calls', 0)}")
        print(f"  Tool distribution:")
        # Sort by usage count
        sorted_tools = sorted(tool_usage['tools'].items(), 
                            key=lambda x: x[1]['count'], reverse=True)
        for tool_name, tool_stats in sorted_tools:
            print(f"    {tool_name}: {tool_stats['count']} ({tool_stats['percentage']:.1f}%)")
    
    # Step type statistics
    step_types = eval_stats.get('step_types', {})
    if step_types.get('types'):
        print(f"\n📝 Step Type Statistics:")
        print(f"  Total steps: {step_types.get('total_steps', 0)}")
        print(f"  Step type distribution:")
        # Sort by usage count
        sorted_step_types = sorted(step_types['types'].items(), 
                                 key=lambda x: x[1]['count'], reverse=True)
        for step_type, type_stats in sorted_step_types:
            print(f"    {step_type}: {type_stats['count']} ({type_stats['percentage']:.1f}%)")
    
    print("="*80)
    
    print("\n⏭️  Detailed statistics analysis complete")


if __name__ == "__main__":
    main()
