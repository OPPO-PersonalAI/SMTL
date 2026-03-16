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
Data format conversion script - with data cleaning and statistics functionality
1. Summary prompt requires replies of no more than 500 words
2. Use query in crawl
"""

import argparse
import json
import re
import os
from datetime import datetime
from collections import defaultdict, Counter
import json_repair
from pathlib import Path

from utils import write_json


def format_tool_calls(tool_calls):
    """
    Format tool_calls into the specified format, wrapping each tool call with <tool_call> tags.
    
    Args:
        tool_calls: List of tool calls
        
    Returns:
        Formatted string
    """
    if not tool_calls:
        return ""
    
    formatted_calls = []
    for tool_call in tool_calls:
        # Create new dict containing only name and arguments
        formatted_call = {
            "name": tool_call.get("name", ""),
            "arguments": tool_call.get("arguments", {})
        }
        
        # Convert to JSON string
        call_str = json.dumps(formatted_call, ensure_ascii=False, separators=(',', ':'))
        
        # Wrap with <tool_call> tags
        wrapped_call = f"<tool_call>\n{call_str}\n</tool_call>"
        formatted_calls.append(wrapped_call)
    
    # Join all tool calls with newlines
    return "\n".join(formatted_calls)

def convert_obs_to_tool_response(obs, tool_calls):
    # pattern = r"Results for tool call '(.*?)' with arguments '(\{.*?\})':(.*?)"
    pattern = r"Results for tool call '(.*?)' with arguments '(\{.*?\})':(.*?)(?=\nResults for tool call |$)"
    matches = re.findall(pattern, obs, re.DOTALL)
    tool_responses = []
    assert len(matches) == len(tool_calls), f"Tool call count ({len(tool_calls)}) does not match response count ({len(matches)})"
    for idx, match in enumerate(matches):
        tool_name, arguments, observation = match
        assert tool_name == tool_calls[idx]["name"], f"Tool call name mismatch: {tool_name} != {tool_calls[idx]['name']}"
        assert tool_name, f"Tool call name cannot be empty: {tool_name}"
        assert arguments == str(tool_calls[idx]["arguments"]), f"Tool call arguments mismatch: {arguments} != {tool_calls[idx]['arguments']}"
        assert arguments, f"Tool call arguments cannot be empty: {arguments}"
        arguments_dict = json_repair.loads(arguments)
        tool_response = {
            "name": tool_name,
            "arguments": arguments_dict,
            "observation": observation.strip()
        }
        
        tool_responses.append(f"<tool_response>\n{json.dumps(tool_response, ensure_ascii=False, indent=2)}\n</tool_response>")
    
    return "\n".join(tool_responses)


def clean_data(data):
    """Clean data, remove None values and invalid fields."""
    if not isinstance(data, dict):
        return None
    
    # Check required fields
    required_fields = ["question", "agent_trajectory"]
    for field in required_fields:
        if field not in data or data[field] is None:
            return None
    
    # Clean agent_trajectory
    if not isinstance(data["agent_trajectory"], list):
        return None
    
    cleaned_trajectory = []
    for action in data["agent_trajectory"]:
        if not isinstance(action, dict):
            continue
        
        # Check required fields
        if "name" not in action or action["name"] is None:
            continue
        
        # Clean fields, replace None with empty string
        cleaned_action = {}
        for key, value in action.items():
            if value is None:
                cleaned_action[key] = ""
            else:
                cleaned_action[key] = value
        
        cleaned_trajectory.append(cleaned_action)
    
    if not cleaned_trajectory:
        return None
    
    # Create cleaned data
    cleaned_data = {}
    for key, value in data.items():
        if value is None:
            cleaned_data[key] = ""
        else:
            cleaned_data[key] = value
    
    cleaned_data["agent_trajectory"] = cleaned_trajectory
    return cleaned_data


def jsonl_to_multi_turn_dialogue(data, file_stats=None) -> dict:
    global stats
    # Use file-level stats if provided; otherwise use global stats
    current_stats = file_stats if file_stats is not None else stats
    
    # First clean the data
    cleaned_data = clean_data(data)
    if cleaned_data is None:
        current_stats["cleaned_data_failed"] += 1
        return None
    
    conversations = [
        {
            "role": "system",
            "content": 
'''You are an expert assistant who solves tasks through structured tool calls, following a step-by-step process. Each step (action) involves analyzing needs, selecting tools, and executing calls to achieve the task goal. You are required to solve the task by formulating your thinking and reasoning process as described below:

1. Objective:
    1.1 Your core goal is to systematically solve user-assigned tasks by:
        - Decomposing the task into clear goals & paths.
        - Executing tools purposefully and efficiently.
        - Advancing all goals in parallel, while keeping each goal’s paths sequential.
        - Tracking progress with summaries.
        - Delivering a final confirmed answer only when all goals are resolved.
2. Execution Requirements:
    2.1 Follow a logical order of functions/tools.
    2.2 Parallelize independent goals; within each goal, execute paths sequentially as fallbacks.
    2.3 Each step must include:
        - Reasoning process (before you execute tools, why this tool/path is chosen).
        - <tool_call> execution (with correct parameters).
        - After executing the tools, you will receive observations (results of tool calls), which can be used as input for subsequent actions. This Action/Observation cycle may repeat as needed.
        - Use observations to refine next actions.
        - Ensure no redundant tool calls (don’t repeat identical queries).
        - Never assume a goal is completed without explicit verification.
        - Continue advancing all goals until they are resolved.
3. Functions:
    3.1 <plan> Function:
        - Role: Decompose the original task into goals and execution paths.
        - Rules:
            - 1–5 parallelizable goals.
            - Each goal has 1–5 paths, executed sequentially as fallback options.
            - Define success criteria for each path.
        - Timing: Only the first step.
        - Format Example:
            <plan>
            ## Goal 1: [Name]
            - Path 1.1: [Approach]  
            - Success: [Criteria]
            - Path 1.2: [Approach]  
            - Success: [Criteria]
            ## Goal 2: [Name]
            - Path 2.1: [Approach]  
            - Success: [Criteria]
            </plan>
    3.2 <summary> Function:
        - Role: Recap execution status and decide next actions.
        - Content:
            - Plan summary (original goals/paths).
            - Execution status for each goal: Completed / In Progress / Blocked.
            - Path analysis (which worked, which failed).
            - Next steps: specify which sub-paths to run in parallel.
        - Timing: Every several steps, occurs when there are enough actions to summarize;
        - Example:
            <summary>
            ## Plan Summary
            [Brief recap of goals]
            ## Execution Status
            ### Goal 1: [Status]
            - Path Analysis: [...]
            ### Goal 2: [Status]
            - Path Analysis: [...]
            ## Next Parallel Sub-Paths
            - Goal 1: Path 1.2
            - Goal 2: Path 2.1
            </summary>
    3.3 <tool_call> Tool:
        - Role: Execute tools to advance goals.
            - web_search: it has only one parameter: query (search statement). For example, {'name': 'web_search', 'arguments': {'query': 'xxx'}}.
            - crawl_page: it has two parameters: url (valid link) and query (info to extract). For example, {'name': 'crawl_page', 'arguments': {'url': 'xxx', 'query': 'xxx'}}.
        - Rules:
            - Use **1–5** tools per step (each targeting a distinct task part).
            - Each tool call must have complete, valid parameters.
        - Tool Usage Strategy:
            **web_search Strategy Adjustment (MANDATORY when results are insufficient):**
            - CRITICAL LIMITATION: web_search results may NOT always contain the exact information you need. The returned snippets are often incomplete or may not directly address your query requirements.
            - Official Source Priority: When precise information is needed, PRIORITIZE searching for official sources (Wikipedia, Hugging Face, .gov/.edu domains, international organizations, official technical docs, academic sources, etc.) using site-specific searches when appropriate.
            - When web_search results are insufficient or irrelevant, you MUST adjust your search strategy:
              * Strategy 1 - Strategic Major Adjustment: Re-read the original query carefully to identify ALL conditions, requirements, and constraints. Analyze what information you've already found, exclude ineffective approaches, and find new breakthrough directions (different aspects, keywords, or information sources). Prioritize official sources when appropriate.
              * Strategy 2 - Search Query Minor Adjustment: If searches return no/few results, your query might be TOO STRICT. Identify strict constraints (site restrictions, year restrictions, quoted phrases, multiple AND conditions) and consider relaxing them or using alternative search terms/synonyms. Consider targeting official sources with appropriate site-specific searches.
            **crawl_page Best Practice (MANDATORY):**
            - When web_search returns URLs, CAREFULLY ANALYZE each URL's title, snippet, and source to determine which ones are potentially relevant to your query requirements and reasoning needs.
            - Official Source Priority: When MULTIPLE URLs show potential relevance, PRIORITIZE crawling official and authoritative sources first (Wikipedia, Hugging Face, .gov/.edu domains, international organizations, official technical docs, academic sources, official news/statistics/standards/regulatory agencies, etc.) as they typically provide more accurate and reliable information.
            - For URLs that show promise (match query conditions or align with your reasoning), you MUST use crawl_page to verify EACH promising URL IN PARALLEL. Prioritize official sources, but do not skip any URLs with genuine potential.
            - Do NOT miss any URL that genuinely appears relevant to your query requirements or reasoning needs. The snippets from web_search are incomplete - crawl_page provides the full context. However, be selective and only crawl URLs that show real promise, giving priority to official channels when multiple sources are available.
            - Workflow: web_search (broad discovery) → careful analysis of URLs → prioritize official sources → crawl_page (parallel deep verification for promising URLs, official sources prioritized)
        - Timing: All steps except <plan>, <summary>, and <answer>.
        - Example:
            <tool_call>
            {'name': 'web_search', 'arguments': {'query': 'Ths highest mountain in the world'}}
            </tool_call>
            <tool_call>
            {'name': 'crawl_page', 'arguments': {'url': 'xxx', 'query': 'xxx'}}
            </tool_call>
    3.4 <answer> Function:
        - Role: Deliver the final confirmed answer.
        - Rules:
            - Only after all goals are resolved.
            - Must consolidate results across all goals.
            - Answer language must match task language.
        - Format Example:
            <answer>
            [Final Answer Content]
            </answer>
4. Execution Rules (Critical)
    4.1 Parallel Goals, Sequential Paths
        - Advance all goals concurrently.
        - Within a goal, execute paths sequentially as fallbacks.
    4.2 No Early Termination
        - Do not assume a goal is complete until explicitly verified.
        - Always continue advancing other goals in parallel.
    4.3 Result Verification
        - After web_search returns URLs, carefully analyze each URL to identify promising ones that match query conditions or align with reasoning needs.
        - Use crawl_page to verify promising search results IN PARALLEL (do not skip URLs with genuine potential).
        - Do not consider a goal "completed" until verified through crawl_page for all promising URLs.
    4.4 Parallel Functions with Limited workers
        - Use no more than 10 tools per step.
    4.5 Final Answer Condition
        - Only produce <answer> when all goals are complete.
        - Consolidated results must be accurate and fully solve the original task.

** Important Tips **:
1. Do not give an answer easily unless you are absolutely sure. The answer should be as concise as possible and avoid detailed descriptions. For example, <answer>Beijing</answer>.
'''.strip()
        },
        {
            "role": "user",
            "content": f"Your task is: {cleaned_data['question']}\nNow Begin! Solve the task!"
        }
    ]

    trajectory = cleaned_data["agent_trajectory"]
    if len(trajectory) < 1:
        current_stats["trajectory_too_short"] += 1
        return None
    
    if len(trajectory) > 100:
        current_stats["trajectory_too_long"] += 1
        return None
    
    steps = len(trajectory)
    for i in range(steps):
        action = trajectory[i]
        ans = False
    
        if action["name"] == "plan":
            plan_content = action["value"]
            assistant_content = (
                f"Now, Let's break down this problem into manageable goals and identify multiple solution paths for each goal.\n\n"
                f"<plan>{plan_content}</plan>"
            )
        elif action["name"] == "summary":
            summary_content = action["value"]
            conversations[-1]["content"] += "\n\n# Note: Now, you should summarize the task completion status and provide recommendations for next steps."
            assistant_content = (
                f"Let me summarize the completion status of the plan based on the conversation before."
                f"<summary>{summary_content}</summary>"
            )
        else:
            # Check if this is a final_answer first
            if action["name"] != "summary" and action["name"] != "plan":
                if not action["tool_calls"] or len(action["tool_calls"]) == 0:
                    current_stats["empty_tool_calls"] += 1
                    return None
                for item in action["tool_calls"]:
                    if item["name"] == "final_answer":
                        ans = True
                        # Process final_answer
                        think_content = action["think"]
                        try:
                            ans_result = item["arguments"]["answer"]
                        except Exception:
                            current_stats["final_answer_parse_error"] += 1
                            return None
                        
                        # Create assistant message containing final_answer
                        assistant_content = (
                            f"{think_content}\n\n"
                            f"<answer>\n{ans_result}\n</answer>"
                        )
                        conversations.append({
                            "role": "assistant",
                            "content": assistant_content
                        })
                        break
            
            # If final_answer was found, break out of the loop
            if ans:
                break
            
            # Only process as regular tool_call if not final_answer
            # Format tool_calls into specified format
            tool_calls_formatted = format_tool_calls(action["tool_calls"])
            if len(action["tool_calls"]) >= 10:
                current_stats["too_many_tool_calls"] += 1
                return None
            think_content = action["think"]
            assistant_content = (
                f"{think_content}\n\n"
                f"{tool_calls_formatted}"
            )
        
        conversations.append({
            "role": "assistant",
            "content": assistant_content
        })
        
        if not ans:
            if action["name"] == "plan" or action["name"] == "summary":
                conversations.append({
                    "role": "user",
                    "content": "Based on the plan/summary and previous conversations, continue solving the task!"
                })
            else:
                # Format tool response results and validate correspondence with tool_calls
                formatted_responses = convert_obs_to_tool_response(action["obs"], action["tool_calls"])
                conversations.append({
                    "role": "user",
                    "content": formatted_responses
                })

    # Check if final_answer was found
    has_final_answer = False
    for action in trajectory:
        if action["name"] != "summary" and action["name"] != "plan":
            for item in action["tool_calls"]:
                if item["name"] == "final_answer":
                    has_final_answer = True
                    break
        if has_final_answer:
            break
    
    if not has_final_answer:
        current_stats["no_final_answer"] += 1
        return None
    
    if len(trajectory[-1]["tool_calls"]) != 1:
        current_stats["final_tool_calls_not_one"] += 1
        return None
    if len(trajectory[-1]["tool_calls"]) >= 6:
        current_stats["final_tool_calls_too_many"] += 1
        return None
    if conversations[-1]["role"] == "user":
        current_stats["ends_with_user"] += 1
        return None
    else:
        current_stats["successful_conversions"] += 1
        # Ensure model_id field is preserved (default to 'unknown' if not present)
        result = {"conversations": conversations}
        if "model_id" in cleaned_data:
            result["model_id"] = cleaned_data["model_id"]
        else:
            result["model_id"] = "unknown"
        # Save question and golden_answer fields
        if "question" in cleaned_data:
            result["question"] = cleaned_data["question"]
        if "golden_answer" in cleaned_data:
            result["golden_answer"] = cleaned_data["golden_answer"]
        return result


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
        # Trajectory statistics (use conversations field)
        conversations = result.get("conversations", [])
        question = ""
        
        # Extract question
        for conv in conversations:
            if conv.get("role") == "user":
                content = conv.get("content", "")
                if "Your task is:" in content:
                    task_start = content.find("Your task is:") + len("Your task is:")
                    task_end = content.find(".", task_start)
                    if task_end == -1:
                        task_end = len(content)
                    question = content[task_start:task_end].strip()
                else:
                    question = content.strip()
                break

        if conversations:
            # Count steps (number of messages in conversations)
            step_count = len(conversations)
            trajectory_steps.append(step_count)
            
            # Count tools used per question
            question_tools = set()
            
            # Count step types and tool usage
            for conv in conversations:
                role = conv.get("role", "unknown")
                step_types[role] = step_types.get(role, 0) + 1
                
                # Count tool usage (from <tool_call> tags in assistant content)
                if role == "assistant":
                    content = conv.get("content", "")
                    # Find <tool_call> tags
                    if "<tool_call>" in content:
                        tool_call_start = content.find("<tool_call>")
                        tool_call_end = content.find("</tool_call>")
                        if tool_call_start != -1 and tool_call_end != -1:
                            tool_call_content = content[tool_call_start + 11:tool_call_end].strip()
                            try:
                                # Parse JSON format tool calls
                                import json
                                tool_calls = json.loads(tool_call_content)
                                if isinstance(tool_calls, list):
                                    for tool_call in tool_calls:
                                        if isinstance(tool_call, dict):
                                            tool_name = tool_call.get("name", "unknown")
                                            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                                            question_tools.add(tool_name)
                            except (json.JSONDecodeError, ValueError):
                                # If JSON parse fails, try simple text matching
                                import re
                                # Match "name": "tool_name" pattern
                                name_matches = re.findall(r'"name":\s*"([^"]+)"', tool_call_content)
                                for tool_name in name_matches:
                                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                                    question_tools.add(tool_name)
            
            # Record tools used per question
            if question_tools:
                tool_usage_by_question[question] = list(question_tools)
    
    # Compute trajectory statistics
    trajectory_stats = {}
    if trajectory_steps:
        trajectory_stats = {
            "total_trajectories": len(trajectory_steps),
            "avg_steps": sum(trajectory_steps) / len(trajectory_steps),
            "median_steps": sorted(trajectory_steps)[len(trajectory_steps) // 2],
            "min_steps": min(trajectory_steps),
            "max_steps": max(trajectory_steps),
            "conversation_step_distribution": {
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
    
    # Compute tool usage percentages
    total_tool_calls = sum(tool_usage.values())
    tool_usage_percentages = {}
    if total_tool_calls > 0:
        for tool, count in tool_usage.items():
            tool_usage_percentages[tool] = {
                "count": count,
                "percentage": count / total_tool_calls * 100
            }
    
    # Compute step type percentages
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
        "conversion": {
            "total": total,
            "successful": total,  # Assumes all results are successful conversions
            "success_rate": 100.0 if total > 0 else 0
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
    
    # Merge filter statistics if provided
    if filter_stats is not None and total_processed is not None:
        stats_result["filtering"] = {
            "total_processed": total_processed,
            "successful_conversions": filter_stats.get('successful_conversions', 0),
            "overall_success_rate": filter_stats.get('successful_conversions', 0) / total_processed * 100 if total_processed > 0 else 0,
            "filter_failures": {
                "cleaned_data_failed": filter_stats.get('cleaned_data_failed', 0),
                "trajectory_too_short": filter_stats.get('trajectory_too_short', 0),
                "trajectory_too_long": filter_stats.get('trajectory_too_long', 0),
                "too_many_tool_calls": filter_stats.get('too_many_tool_calls', 0),
                "empty_tool_calls": filter_stats.get('empty_tool_calls', 0),
                "no_final_answer": filter_stats.get('no_final_answer', 0),
                "final_tool_calls_not_one": filter_stats.get('final_tool_calls_not_one', 0),
                "final_tool_calls_too_many": filter_stats.get('final_tool_calls_too_many', 0),
                "ends_with_user": filter_stats.get('ends_with_user', 0),
                "final_answer_parse_error": filter_stats.get('final_answer_parse_error', 0)
            }
        }
    
    return stats_result


def print_detailed_report(stats):
    """Print detailed statistics report."""
    print("\n" + "="*80)
    print("📊 Detailed Conversion Statistics Report")
    print("="*80)
    
    # Conversion result statistics
    conv_stats = stats.get('conversion', {})
    print(f"🎯 Conversion Results:")
    print(f"  Total: {conv_stats.get('total', 0)}")
    print(f"  Successful: {conv_stats.get('successful', 0)}")
    print(f"  Success rate: {conv_stats.get('success_rate', 0):.2f}%")
    
    # Filter statistics
    filter_stats = stats.get('filtering', {})
    if filter_stats:
        print(f"\n🔍 Filter Statistics:")
        print(f"  Total processed: {filter_stats.get('total_processed', 0)}")
        print(f"  Successful conversions: {filter_stats.get('successful_conversions', 0)}")
        print(f"  Overall success rate: {filter_stats.get('overall_success_rate', 0):.2f}%")
        
        filter_failures = filter_stats.get('filter_failures', {})
        if filter_failures:
            print(f"  ❌ Failure reason statistics:")
            for failure_type, count in filter_failures.items():
                if count > 0:
                    print(f"    {failure_type}: {count}")
    
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
        total_records = conv_stats.get('total', 0)
        for model_id, count in model_distribution.items():
            percentage = round(count / total_records * 100, 2) if total_records > 0 else 0
            print(f"  {model_id}: {count} records ({percentage}%)")
    
    print("="*80)

def save_detailed_stats(stats, output_file):
    """Save detailed statistics to file."""
    # Generate stats file path
    if output_file.endswith('.json'):
        stats_file = output_file.replace('.json', '_stats.json')
    else:
        stats_file = output_file + '_stats.json'
    
    try:
        write_json(stats_file, stats)
        print(f"📊 Detailed statistics saved to: {stats_file}")
    except Exception as e:
        print(f"❌ Error saving statistics report: {e}")


def print_statistics(stats, total_processed):
    """Print statistics."""
    print("\n" + "="*80)
    print("📊 Data Conversion Statistics Report")
    print("="*80)
    print(f"Total processed: {total_processed}")
    print(f"Successful conversions: {stats['successful_conversions']}")
    if total_processed > 0:
        print(f"Success rate: {stats['successful_conversions']/total_processed*100:.2f}%")
    else:
        print("Success rate: 0.00%")
    
    print(f"\n❌ Failure reason statistics:")
    print(f"  Data cleaning failed: {stats['cleaned_data_failed']}")
    print(f"  Trajectory too short: {stats['trajectory_too_short']}")
    print(f"  Trajectory too long: {stats['trajectory_too_long']}")
    print(f"  Too many tool calls: {stats['too_many_tool_calls']}")
    print(f"  Empty tool calls: {stats['empty_tool_calls']}")
    print(f"  No final_answer: {stats['no_final_answer']}")
    print(f"  Final tool calls not one: {stats['final_tool_calls_not_one']}")
    print(f"  Final tool calls too many: {stats['final_tool_calls_too_many']}")
    print(f"  Ends with user message: {stats['ends_with_user']}")
    print(f"  final_answer parse error: {stats['final_answer_parse_error']}")
    
    print("="*80)


def find_all_jsonl_files(input_dir):
    """Find all .jsonl files under the input directory."""
    from pathlib import Path
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return []
    
    # Recursively find all .jsonl files
    jsonl_files = list(input_path.rglob("*.jsonl"))
    return jsonl_files

def generate_output_path(input_file, input_dir, output_dir, date):
    """Generate output file path while preserving relative directory structure."""
    input_path = Path(input_file)
    input_root = Path(input_dir)
    output_root = Path(output_dir)

    relative_path = input_path.relative_to(input_root)
    relative_parent = relative_path.parent
    normalized_stem = relative_path.stem.replace("-", "_").replace("3.2", "3_2")
    output_file = output_root / relative_parent / f"afm2_{date}_{normalized_stem}.json"

    return str(output_file.parent), str(output_file)


def process_single_file(infile, date, stats, input_dir, output_dir):
    """Process a single file."""
    print(f"\n🔄 Processing file: {infile}")
    
    # Generate output path
    prefix, outfile = generate_output_path(infile, input_dir, output_dir, date)
    os.makedirs(prefix, exist_ok=True)
    
    file_processed = 0
    file_successful = 0
    res = []
    
    # Create independent stats object for current file
    file_stats = {
        "cleaned_data_failed": 0,
        "trajectory_too_short": 0,
        "trajectory_too_long": 0,
        "too_many_tool_calls": 0,
        "empty_tool_calls": 0,
        "no_final_answer": 0,
        "final_tool_calls_not_one": 0,
        "final_tool_calls_too_many": 0,
        "ends_with_user": 0,
        "final_answer_parse_error": 0,
        "successful_conversions": 0
    }
    
    try:
        with open(infile, 'r', encoding='utf-8') as f: 
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    file_processed += 1
                    
                    tk = jsonl_to_multi_turn_dialogue(sample, file_stats)
                    if isinstance(tk, dict):
                        res.append(tk)
                        file_successful += 1
                        
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON decode error at line {line_num}: {e}")
                except Exception as e:
                    print(f"  Warning: Processing error at line {line_num}: {e}")
        
        print(f"  ✅ File processing complete: {file_processed} records, {file_successful} successful conversions")
        
    except FileNotFoundError:
        print(f"  ❌ File not found: {infile}")
        return 0, 0
    except Exception as e:
        print(f"  ❌ File processing failed: {e}")
        return 0, 0
    
    # Save current file results
    if res:
        write_json(outfile, res)

        # Generate detailed statistics report for current file
        print(f"\n📊 Generating detailed statistics report...")
        
        # Use file-level statistics
        file_filter_stats = file_stats.copy()
        
        # Compute detailed statistics for current file
        file_detailed_stats = calculate_detailed_statistics(res, file_filter_stats, file_processed)
        
        # Save detailed statistics for current file
        save_detailed_stats(file_detailed_stats, outfile)
        
        # Print detailed statistics report for current file
        print_detailed_report(file_detailed_stats)
    
    return file_processed, file_successful

def generate_stats_for_other_files(output_dir):
    """Generate statistics for other files in the results_final_v1 directory."""
    from utils import read_json, read_jsonl
    
    results_final_v1_path = Path(output_dir)
    if not results_final_v1_path.exists():
        print(f"{results_final_v1_path} directory does not exist, skipping stats generation for other files")
        return
    
    # Find all .json and .jsonl files, excluding _stats.json files
    all_files = []
    for pattern in ["*.json", "*.jsonl"]:
        files = list(results_final_v1_path.rglob(pattern))
        # Filter out stats files
        files = [f for f in files if not f.name.endswith("_stats.json")]
        all_files.extend(files)
    
    if not all_files:
        print("No files found for stats generation")
        return
    
    print(f"Found {len(all_files)} files for stats generation")
    
    for file_path in all_files:
        print(f"\n📊 Generating stats for file: {file_path}")
        
        try:
            # Read file data
            if str(file_path).endswith('.jsonl'):
                data = read_jsonl(str(file_path))
            else:
                data = read_json(str(file_path))
            
            if not data:
                print(f"  ⚠️ File is empty, skipping stats generation")
                continue
            
            # Check data format
            if isinstance(data, list) and len(data) > 0:
                # Check if this is conversations format
                if "conversations" in data[0]:
                    # Converted data, use conversations statistics
                    stats_result = calculate_detailed_statistics(data)
                else:
                    # Raw data, use agent_trajectory statistics
                    stats_result = calculate_detailed_statistics_for_agent_trajectory(data)
            else:
                print(f"  ⚠️ Unsupported data format, skipping stats generation")
                continue
            
            # Generate stats file path
            if str(file_path).endswith('.jsonl'):
                stats_file = str(file_path).replace('.jsonl', '_stats.json')
            else:
                stats_file = str(file_path).replace('.json', '_stats.json')
            
            # Save stats file
            write_json(stats_file, stats_result)
            print(f"  ✅ Stats file saved: {stats_file}")
            
        except Exception as e:
            print(f"  ❌ Error generating stats: {e}")

def calculate_detailed_statistics_for_agent_trajectory(results):
    """Calculate statistics for agent_trajectory format data."""
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
            # Count steps (number of steps in agent_trajectory)
            step_count = len(agent_trajectory)
            trajectory_steps.append(step_count)
            
            # Count tools used per question
            question_tools = set()
            
            # Count step types and tool usage
            for step in agent_trajectory:
                step_name = step.get("name", "unknown")
                step_types[step_name] = step_types.get(step_name, 0) + 1
                
                # Count tool usage (from tool_calls in action steps)
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
    
    # Compute trajectory statistics
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
    
    # Compute tool usage percentages
    total_tool_calls = sum(tool_usage.values())
    tool_usage_percentages = {}
    if total_tool_calls > 0:
        for tool, count in tool_usage.items():
            tool_usage_percentages[tool] = {
                "count": count,
                "percentage": count / total_tool_calls * 100
            }
    
    # Compute step type percentages
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
    
    # Build statistics
    stats_result = {
        "data_analysis": {
            "total": total,
            "format": "agent_trajectory"
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
    
    return stats_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean filtered trajectory data and convert it to multi-turn training conversations."
    )
    parser.add_argument("--input_dir", type=str, default="results3", help="Input directory containing filtered JSONL files.")
    parser.add_argument("--output_dir", type=str, default="results_final_v1", help="Output directory for converted JSON files.")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%m%d"), help="Date tag used in output filenames.")
    args = parser.parse_args()

    # Initialize statistics
    stats = {
        "cleaned_data_failed": 0,
        "trajectory_too_short": 0,
        "trajectory_too_long": 0,
        "too_many_tool_calls": 0,
        "empty_tool_calls": 0,
        "no_final_answer": 0,
        "final_tool_calls_not_one": 0,
        "final_tool_calls_too_many": 0,
        "ends_with_user": 0,
        "final_answer_parse_error": 0,
        "successful_conversions": 0,
    }

    # Find all input files
    input_dir = args.input_dir
    output_dir = args.output_dir
    date = args.date
    jsonl_files = find_all_jsonl_files(input_dir)

    if not jsonl_files:
        print(f"No .jsonl files found in directory {input_dir}")
        exit(1)

    print(f"Found {len(jsonl_files)} .jsonl files")

    total_processed = 0
    total_successful = 0

    # Process all files
    for infile in jsonl_files:
        file_processed, file_successful = process_single_file(infile, date, stats, input_dir, output_dir)
        total_processed += file_processed
        total_successful += file_successful

    # Print final statistics
    print_statistics(stats, total_processed)

    # Generate statistics for other files in output directory
    print(f"\n🔍 Checking other files in {output_dir} directory...")
    generate_stats_for_other_files(output_dir)
