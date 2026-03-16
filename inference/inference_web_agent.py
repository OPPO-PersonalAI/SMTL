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

import json
import re
import time
import copy
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import logging
from openai import OpenAI, APIError, APITimeoutError
import os
import subprocess
import random
from queue import Queue, Empty
from threading import Thread, Lock
from transformers import AutoTokenizer
from datetime import datetime

from utils import (
    read_jsonl,
    write_jsonl, 
    read_json, 
    write_json, 
    extract_specific_tag,
)
# Import truncation functions
import sys
import os
from utils import truncate_conversation_history, should_truncate
from tools import (
    WebSearchTool as RemoteWebSearchTool, 
    CrawlPageTool as RemoteCrawlPageTool
)
from em_metric import em_check, subem_check
from prompts import judge_prompt, system_prompt


# Initialize with default configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), 
    ]
)

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv(override=True)

###############################################################################
# Common configuration
KEY = "empty"

SYSTEM_PROMPT = system_prompt
MODEL = 'smtl'


def get_env_value(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning(f"Invalid integer for {name}={value}, fallback to {default}")
        return default


def get_next_model_url() -> str:
    global MODEL_URL_INDEX
    with MODEL_URL_LOCK:
        url = MODEL_URLS[MODEL_URL_INDEX % len(MODEL_URLS)]
        MODEL_URL_INDEX += 1
        return url


MODEL_URLS = [u.strip() for u in os.getenv("MODEL_URL", "http://0.0.0.0:1/v1").split(",") if u.strip()]
MODEL_URL_INDEX = 0
MODEL_URL_LOCK = Lock()


## External API ##
#### llm_judge ###
judge_model_config = {
    "model_id": None,   # 占位，INFER_KWARGS 定义后回填
    "config": [
        [os.getenv("BASE_URL"), os.getenv("API_KEY", "")],  
    ],
    "pointer": 0,
}
#### llm_replan ###
replan_model_config = {
    "model_id": None,   # 占位，INFER_KWARGS 定义后回填
    "config": [
        [os.getenv("BASE_URL"), os.getenv("API_KEY", "")],  
    ],
    "pointer": 0
}
### web_search ###
web_search_config = {
    "config": [
        [f"{os.getenv('WEBSEARCH_URL', '')}"],  
    ],
    "pointer": 0,
}
### crawl_page ###
crawl_page_config = {
    "config": [
        [f"{os.getenv('CRAWL_PAGE_URL', '')}"],  
    ],
    "pointer": 0,
}

INFER_KWARGS = {
    "temperature": 1.0,
    "top_p": 0.9,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "total_tokens": 131072,
    "max_tokens": 8192,
    "web_topk": get_env_value("WEB_TOPK", 20),
    "max_steps": get_env_value("MAX_STEPS", 100),
    "max_retry": 1,
    "api_retry": 3,
    "api_timeout": 300.0,
    "tool_retry": 3,
    "parallel": get_env_value("PARALLEL", 4),
    "benchmark": os.getenv("BENCHMARK", "browsecomp_sample"),
    "judge_model_id": "deepseek-v3.2",
    "replan_model_id": "deepseek-v3.2",
    "judge_model_config": judge_model_config,
    "replan_model_config": replan_model_config,
    "web_search_config": web_search_config,
    "crawl_page_config": crawl_page_config,
    "replan_interval": 5,
}

# INFER_KWARGS 定义完后，将 model_id 回填到 config（dict 为引用类型，同步生效）
judge_model_config["model_id"]  = INFER_KWARGS["judge_model_id"]
replan_model_config["model_id"] = INFER_KWARGS["replan_model_id"]

SHOW_KWARGS = {
    "key": KEY,
    "model": MODEL,
    "model_urls": MODEL_URLS,
    "system_prompt": SYSTEM_PROMPT,
    **INFER_KWARGS
}

tokenizer = AutoTokenizer.from_pretrained("tokenizer/qwen3/Qwen3-4B", trust_remote_code=True)


def api_client(
    url: str,
    key: str,
    model: str,
    conversation: List[Dict[str, str]] = None,
    system: str = None,
    prompt: str = None,
    api_retry: int = None,
    api_timeout: float = None,
    enable_retry: bool = True,
    **kwargs
) -> tuple[str, str, dict]:
    """
    Unified API client function using non-streaming calls
    
    Args:
        url: API base URL
        key: API key
        model: Model name
        conversation: Conversation history (mutually exclusive with system/prompt)
        system: System prompt (mutually exclusive with conversation)
        prompt: User prompt (mutually exclusive with conversation)
        api_retry: Maximum retry count (if None, priority: api_retry in kwargs > api_retry in INFER_KWARGS > default 1)
        api_timeout: Request timeout (if None, priority: api_timeout in kwargs > api_timeout in INFER_KWARGS > default 120.0)
        enable_retry: Whether to enable retry mechanism
        **kwargs: Other parameters (temperature, top_p, max_tokens, etc.)
    
    Returns:
        tuple[str, str, dict]: (Model response content, stop reason, timing statistics)
    """
    # If api_retry is not specified, get from kwargs first, then INFER_KWARGS, finally default to 1
    if api_retry is None:
        api_retry = kwargs.get("api_retry") or INFER_KWARGS.get("api_retry", 1)
    
    # If api_timeout is not specified, get from kwargs first, then INFER_KWARGS, finally default to 120.0
    if api_timeout is None:
        api_timeout = kwargs.get("api_timeout") or INFER_KWARGS.get("api_timeout", 120.0)
    
    # Initialize timing statistics
    timing_stats = {
        "total_time": 0,
        "api_call_time": 0,
        "retry_count": 0,
        "start_time": time.time(),
        "end_time": 0
    }
    
    client = OpenAI(
        base_url=url,
        api_key=key,
        timeout=api_timeout
    )
    
    # Build messages
    if conversation is not None:
        messages = conversation
    elif system is not None and prompt is not None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    else:
        raise ValueError("Must provide either conversation or system+prompt parameters")
    
    # Build request parameters
    request_params = {
        "model": model,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 4096),
    }
    
    # Add optional parameters
    extra_body = {}
    if "temperature" in kwargs:
        request_params["temperature"] = kwargs.get("temperature", 1.0)
    if "top_p" in kwargs:
        request_params["top_p"] = kwargs.get("top_p", 1.0)
    if "presence_penalty" in kwargs:
        request_params["presence_penalty"] = kwargs.get("presence_penalty", 0)
    if "frequency_penalty" in kwargs:
        request_params["frequency_penalty"] = kwargs.get("frequency_penalty", 0)
    if "repetition_penalty" in kwargs:
        extra_body["repetition_penalty"] = kwargs.get("repetition_penalty", 1.1)
    if extra_body:
        request_params["extra_body"] = extra_body
    
    retry_count = 0
    while retry_count < (api_retry if enable_retry else 1):
        try:
            # Record API call start time
            api_start_time = time.time()
            logging.info(f"Starting API call - URL: {url}, Model: {model}, Retry count: {retry_count}")
            
            # Non-streaming processing
            response = client.chat.completions.create(**request_params)
            
            # Record API call end time
            api_end_time = time.time()
            api_call_duration = api_end_time - api_start_time
            timing_stats["api_call_time"] = api_call_duration
            timing_stats["retry_count"] = retry_count
            
            logging.info(f"API call completed - Duration: {api_call_duration:.3f}s, Response length: {len(tokenizer.encode(response.choices[0].message.content)) if response.choices else 0}")

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason or "completed"
                
                # Calculate total time
                timing_stats["end_time"] = time.time()
                timing_stats["total_time"] = timing_stats["end_time"] - timing_stats["start_time"]
                
                logging.info(f"Model inference completed - Total duration: {timing_stats['total_time']:.3f}s")
                return content, finish_reason, timing_stats
            else:
                logging.warning(f"API returned no available response options - URL: {url}, Model: {model}")
                timing_stats["end_time"] = time.time()
                timing_stats["total_time"] = timing_stats["end_time"] - timing_stats["start_time"]
                return "", "no_choices", timing_stats

        except APITimeoutError:
            if enable_retry and retry_count < api_retry - 1:
                retry_count += 1
                # Reduce wait time for quick retry
                wait_time = min(1, retry_count * 0.5)  # Maximum wait 1 second
                logging.warning(f"API request timeout, will retry after {wait_time}s (attempt {retry_count+1}) - URL: {url}, Model: {model}, Current retry count: {retry_count}")
                time.sleep(wait_time)
            else:
                logging.error(f"API request timeout, reached maximum retry count ({api_retry}) - URL: {url}, Model: {model}, Total retry count: {retry_count}")
                timing_stats["end_time"] = time.time()
                timing_stats["total_time"] = timing_stats["end_time"] - timing_stats["start_time"]
                return "", "timeout", timing_stats
        except APIError as e:
            logging.error(f"API error: {str(e)} - URL: {url}, Model: {model}, Retry count: {retry_count}")
            timing_stats["end_time"] = time.time()
            timing_stats["total_time"] = timing_stats["end_time"] - timing_stats["start_time"]
            return "", "api_error", timing_stats
        except Exception as e:
            logging.error(f"Error calling API: {str(e)} - URL: {url}, Model: {model}, Retry count: {retry_count}")
            timing_stats["end_time"] = time.time()
            timing_stats["total_time"] = timing_stats["end_time"] - timing_stats["start_time"]
            return "", "error", timing_stats
    
    logging.error(f"All {api_retry} retries failed - URL: {url}, Model: {model}, Final retry count: {retry_count}")
    timing_stats["end_time"] = time.time()
    timing_stats["total_time"] = timing_stats["end_time"] - timing_stats["start_time"]
    return "", "api_retry_exceeded", timing_stats


def analyze_tool_failure(tool_type, tool_params, error_result):
    """
    Analyze tool call failure and provide detailed error message in English
    
    Args:
        tool_type: Type of tool (web_search, crawl_page)
        tool_params: Parameters passed to the tool (dict)
        error_result: Error result from tool call
    
    Returns:
        str: Detailed error message in English
    """
    error_str = str(error_result) if error_result else ""
    
    # Check for parameter errors
    if tool_type == 'web_search':
        query = tool_params.get('query', '')
        if not query or not query.strip():
            return "Tool call failed: Invalid parameter - query is empty or missing. Please provide a valid search query."
        if len(query.strip()) < 2:
            return "Tool call failed: Invalid parameter - query is too short (less than 2 characters). Please provide a more specific search query."
    
    elif tool_type == 'crawl_page':
        url = tool_params.get('url', '')
        query = tool_params.get('query', '')
        if not url or not url.strip():
            return "Tool call failed: Invalid parameter - URL is empty or missing. Please provide a valid URL to crawl."
        if not url.strip().startswith(('http://', 'https://')):
            return "Tool call failed: Invalid parameter - URL format is incorrect. URL must start with 'http://' or 'https://'."
        if not query or not query.strip():
            return "Tool call failed: Invalid parameter - query is empty or missing. Please provide a valid query for page content extraction."
    
    # Check for API errors
    error_lower = error_str.lower()
    if "budgetexceedederror" in error_lower or "budget exceeded" in error_lower:
        return f"Tool call failed: API budget exceeded. The service has reached its usage limit. Please try again later or use a different approach."
    elif "400 bad request" in error_lower or "400" in error_str:
        return f"Tool call failed: Invalid request (400 Bad Request). The request parameters may be malformed or invalid. Please check the tool parameters and try again."
    elif "an error occurred:" in error_lower or "error occurred" in error_lower:
        return f"Tool call failed: API service error occurred. The external service returned an error. Please try again or use an alternative method."
    elif "timeout" in error_lower:
        return f"Tool call failed: Request timeout. The service did not respond within the expected time. Please try again or use a simpler query."
    elif "connection" in error_lower or "network" in error_lower or "connection refused" in error_lower:
        return f"Tool call failed: Network connection error. Unable to connect to the service. Please check your network connection and try again."
    elif "404" in error_str or "not found" in error_lower:
        return f"Tool call failed: Resource not found (404). The requested URL or endpoint does not exist. Please verify the URL or query and try again."
    elif "403" in error_str or "forbidden" in error_lower:
        return f"Tool call failed: Access forbidden (403). You may not have permission to access this resource. Please check your credentials or try a different approach."
    elif "500" in error_str or "internal server error" in error_lower:
        return f"Tool call failed: Internal server error (500). The service encountered an unexpected error. Please try again later."
    elif "rate limit" in error_lower or "too many requests" in error_lower:
        return f"Tool call failed: Rate limit exceeded. Too many requests in a short time. Please wait a moment and try again."
    
    # Check for tool selection errors
    if tool_type not in ['web_search', 'crawl_page']:
        return f"Tool call failed: Invalid tool selection. Tool type '{tool_type}' is not recognized or not available. Please use one of the supported tools: web_search, crawl_page."
    
    # Generic error message with error preview
    error_preview = error_str[:150] if len(error_str) > 150 else error_str
    if error_preview:
        return f"Tool call failed: Unknown error occurred. Error details: {error_preview}. Please review the tool parameters and try again."
    else:
        return f"Tool call failed: No response received from the tool. The tool may be unavailable or the request may have been rejected. Please try again or use an alternative approach."


def get_search_results_with_format(response, **kwargs):
    # Initialize tool call timing statistics
    tool_timing_stats = {
        "total_tool_time": 0,
        "web_search_time": 0,
        "crawl_page_time": 0,
        "tool_count": 0,
        "start_time": time.time()
    }
    
    def format_tool_responses(tool_calls_obs):
        """
        Convert tool call results to standard format
        
        Args:
            tool_calls_obs: List containing tool call results
            
        Returns:
            str: Formatted tool response string
        """
        formatted_responses = []
        
        for tool in tool_calls_obs:
            tool_name = tool['type']
            tool_obs = tool['obs']
            
            # Build response_dict based on tool type
            if tool_name == 'web_search':
                response_dict = {
                    "name": "web_search",
                    "arguments": {
                        "query": tool.get('query', '')
                    },
                    "observation": tool_obs
                }
            elif tool_name == 'crawl_page':
                response_dict = {
                    "name": "crawl_page", 
                    "arguments": {
                        "url": tool.get('url', ''),
                        "query": tool.get('query', '')
                    },
                    "observation": tool_obs
                }
            else:
                return False, f"Unknown tool name: {tool_name}"
            
            # Convert to JSON string
            response_str = json.dumps(response_dict, ensure_ascii=False, separators=(',', ':'))
            
            # Wrap with <tool_response> tags
            wrapped_response = f"<tool_response>\n{response_str}\n</tool_response>"
            formatted_responses.append(wrapped_response)
        
        # Join all tool responses with newlines
        return "\n".join(formatted_responses)
    
    try:
        think_content, tool_calls, parsed_content = extract_specific_tag(response)
        # Check if tool_calls is "tool_call"
        if tool_calls != "tool_call":
            return False, f"Expected tool_calls to be 'tool_call', but got '{tool_calls}'"
        
        web_topk = kwargs.get("web_topk", 10)
        tool_retry_count = kwargs.get("tool_retry", 1)
        
        # Get configuration information
        cur_web_search_config = web_search_config["config"][
            web_search_config["pointer"] % len(web_search_config["config"])]
        cur_crawl_page_config = crawl_page_config["config"][
            crawl_page_config["pointer"] % len(crawl_page_config["config"])]
        cur_replan_model_config = replan_model_config["config"][
            replan_model_config["pointer"] % len(replan_model_config["config"])]
        replan_model_id = replan_model_config['model_id']

        # Update configuration pointer
        web_search_config["pointer"] += 1
        crawl_page_config["pointer"] += 1
        replan_model_config["pointer"] += 1

        tool_calls_obs = []
        # Since we've already asserted tool_calls == "tool_call", directly process parsed_content
        logging.info(f"Starting to process {len(parsed_content)} tool calls")
        tool_timing_stats["tool_count"] = len(parsed_content)
        
        for tool in parsed_content:
            if tool['type'] == 'web_search':
                # Process web search, support multiple queries
                logging.info(f"Starting tool call: {tool['type']}, query: {tool['query'][:50]}...")
                query = tool['query']
                num = web_topk
                web_results = ""
                
                # Record web_search start time
                web_search_start_time = time.time()

                for _ in range(tool_retry_count):
                    web_results = RemoteWebSearchTool(
                        cur_web_search_config[0],
                        query=query,
                        topk=num
                    )
                    if web_results:
                        logging.info(f"web_search call succeeded")
                        break
                
                # Record web_search end time
                web_search_end_time = time.time()
                web_search_duration = web_search_end_time - web_search_start_time
                tool_timing_stats["web_search_time"] += web_search_duration
                logging.info(f"web_search tool call completed - duration: {web_search_duration:.3f}s")
                
                if web_results:
                    tool["obs"] = web_results
                    tool_calls_obs.append(tool)
                    logging.info(f"Tool call completed: {tool['type']}, result length: {len(tokenizer.encode(str(web_results)))}")
                else:
                    # Normal mode: skip the tool if it fails, don't add to result list
                    logging.error(f"websearch failed after {tool_retry_count} retries - query: {query[:50]}, final error: {str(web_results)[:100]}")
                    continue  # Skip this tool, don't add to tool_calls_obs
                
            elif tool['type']  == "crawl_page":
                logging.info(f"Starting tool call: {tool['type']}, URL: {tool['url'][:50]}..., query: {tool['query'][:50]}...")
                url = tool['url']
                query = tool['query']
                crawl_results = ""
                
                # Record crawl_page start time
                crawl_page_start_time = time.time()

                for _ in range(tool_retry_count):
                    crawl_results = RemoteCrawlPageTool(
                        crawl_page_url=cur_crawl_page_config[0],
                        api_key=cur_replan_model_config[1],
                        api_url=cur_replan_model_config[0],
                        model=replan_model_id,
                        query=query,
                        url=url,
                        content_max_len=200000,
                    )
                    if crawl_results:
                        logging.info(f"crawl_page call succeeded")
                        break
                
                # Record crawl_page end time
                crawl_page_end_time = time.time()
                crawl_page_duration = crawl_page_end_time - crawl_page_start_time
                tool_timing_stats["crawl_page_time"] += crawl_page_duration
                logging.info(f"crawl_page tool call completed - duration: {crawl_page_duration:.3f}s")
                
                if crawl_results: 
                    tool["obs"] = crawl_results
                    tool_calls_obs.append(tool)
                    logging.info(f"Tool call completed: {tool['type']}, result length: {len(tokenizer.encode(str(crawl_results)))}")
                else:
                    # Normal mode: skip the tool if it fails, don't add to result list
                    logging.error(f"crawl_page failed after {tool_retry_count} retries - URL: {url[:50]}, query: {query[:50]}, final error: {str(crawl_results)[:100]}")
                    continue  # Skip this tool, don't add to tool_calls_obs

            else:
                unknown_tool_type = tool.get('type', 'unknown')
                logging.warning(f"Model toolcalls output unknown toolcall type: {unknown_tool_type}, tool details: {str(tool)[:200]}")
            
        # Calculate total tool call time
        tool_timing_stats["total_tool_time"] = time.time() - tool_timing_stats["start_time"]
        
        logging.info(f"Tool call statistics: requested {len(parsed_content)} tools, actually returned {len(tool_calls_obs)} tool results")
        logging.info(f"Tool call timing statistics: total time={tool_timing_stats['total_tool_time']:.3f}s, "
                    f"web_search={tool_timing_stats['web_search_time']:.3f}s, "
                    f"crawl_page={tool_timing_stats['crawl_page_time']:.3f}s")
        
        if not tool_calls_obs:
            logging.error(f"All tool calls failed - requested {len(parsed_content)} tools, actually returned {len(tool_calls_obs)} tool results")
            return False, "All tool calls failed", tool_timing_stats

        # Use the new formatting function
        formatted_tool_obs = format_tool_responses(tool_calls_obs)
        logging.info(f"Tool response formatting completed, length: {len(tokenizer.encode(formatted_tool_obs))}")
        return True, formatted_tool_obs, tool_timing_stats
    except Exception as err:
        logging.error(f"get_search_results_with_format exception: {str(err)} - response content length: {len(tokenizer.encode(response)) if response else 0}")
        tool_timing_stats["total_tool_time"] = time.time() - tool_timing_stats["start_time"]
        return False, str(err), tool_timing_stats


def process_single_data(query, fixed_url: str, **kwargs):
    # Initialize timing statistics for single inference
    step_timing_stats = {
        "total_steps": 0,
        "total_model_time": 0,
        "total_tool_time": 0,
        "step_details": [],
        "start_time": time.time()
    }
    
    def validate_response_format(response):
        """
        Check if response format is correct: tags appear in pairs and end with correct end_tag
        
        Args:
            response (str): API response content
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Define tags to check
        tags_to_check = ['tool_call', 'summary', 'plan', 'answer']
        
        # Check if each tag appears in pairs
        for tag in tags_to_check:
            start_tag = f'<{tag}>'
            end_tag = f'</{tag}>'
            
            start_count = response.count(start_tag)
            end_count = response.count(end_tag)
            
            if start_count != end_count:
                return False, f"Tag {tag} not paired: {start_count} start tags, {end_count} end tags"
        
        # Check if response ends with correct end_tag
        valid_end_tags = [f'</{tag}>' for tag in tags_to_check]
        response_trimmed = response.strip()
        
        # Check if response ends with any valid end_tag
        ends_with_valid_tag = any(response_trimmed.endswith(end_tag) for end_tag in valid_end_tags)
        
        if not ends_with_valid_tag:
            return False, f"Response does not end with valid end_tag. Valid end tags: {valid_end_tags}"
        
        return True, "Format validation passed"
    

    max_steps = kwargs.get("max_steps", 16)

    initial_conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Your task is: {query}.\nNow Begin! Solve the task!"}
    ]
    
    # Save complete conversation history for final saving
    conversation_history = copy.deepcopy(initial_conversation_history)
    # Conversation history for API calls
    api_conversation_history = copy.deepcopy(initial_conversation_history)
    
    error_count = 0
    for attempt in range(max_steps):
        step_start_time = time.time()
        logging.info(f"Starting step {attempt+1} inference")
        if error_count >= 10:
            step_timing_stats["total_steps"] = attempt
            step_timing_stats["end_time"] = time.time()
            step_timing_stats["total_time"] = step_timing_stats["end_time"] - step_timing_stats["start_time"]
            return conversation_history, "error_count >= 10", "", step_timing_stats

        tmp_answer = ""
        # Truncate context before API call (based on api_conversation_history)
        # Calculate tokens to reserve for output (max_tokens + safety margin)
        max_tokens = kwargs.get("max_tokens", 8192)
        total_tokens = kwargs.get("total_tokens", 131072)
        # Add larger safety margin, considering token calculation deviation and format overhead (role markers, separators, etc.)
        # Empirical value: each message has approximately 5-10 additional tokens, plus other overhead, reserve 1024 as safety margin
        safety_margin = 1024
        reserve_tokens = max_tokens + safety_margin
        max_input_tokens = total_tokens - reserve_tokens
        
        if should_truncate(
            conversation=api_conversation_history,
            tokenizer=tokenizer,
            max_history_tokens=max_input_tokens,
            reserve_output_tokens=reserve_tokens
        ):
            logging.info(f"Current context length: {len(api_conversation_history)}, exceeds limit, truncating")
            truncate_message = [
                {"role": "user", "content": "messgaes are truncated to save context tokens..."}
            ]
            replan_message = [
                {"role": "user", "content": f"You have reached your context limit and have not finished the task. I will only keep the last several thinking trajectories. You should reflect on the past thinking history and create a new plan. Your task is: {query}.\nNow Begin! Solve the task!"}
            ]
            api_conversation_history = copy.deepcopy(initial_conversation_history[:1])
            api_conversation_history += truncate_message
            api_conversation_history += api_conversation_history[-2:]
            api_conversation_history += replan_message
            conversation_history += replan_message

        # Use original max_tokens directly, no adjustment
        adjusted_kwargs = kwargs.copy()

        try:
            cur_result, stop_reason, model_timing = api_client(
                url=fixed_url, 
                key=KEY, 
                model=MODEL,
                conversation=api_conversation_history,
                **kwargs
            )
            
            # Record model inference time
            step_timing_stats["total_model_time"] += model_timing.get("api_call_time", 0)
            if not cur_result:
                logging.error(f"API call failed: empty response, stop reason: {stop_reason} - round: {attempt + 1}, error count: {error_count}")
                error_count += 1
                continue
            
            # Validate response format
            is_valid, validation_error = validate_response_format(cur_result)
            if not is_valid:
                logging.error(f"Response format validation failed: {validation_error} - round: {attempt + 1}, error count: {error_count}")
                logging.error(f"Response content: {cur_result[:200]}...")
                error_count += 1
                continue

            logging.info(f"Current reply round: {attempt + 1}, current reply length: {len(tokenizer.encode(cur_result))}")
            # Append response to both complete history and API history
            assistant_response = {
                "role": "assistant", 
                "content": cur_result
            }
            conversation_history.append(assistant_response)
            api_conversation_history.append(assistant_response)
            
            answer_match = re.findall(r'<answer>(.*?)</answer>', cur_result, re.DOTALL)
            if answer_match:
                tmp_answer = answer_match[0].strip()
                logging.info(f"Found final answer: {tmp_answer[:100]}...")
                break

            match_start = re.search(r'<tool_call>', cur_result, re.DOTALL) 
            match_end = re.search(r'</tool_call>', cur_result, re.DOTALL) 
            if match_start and match_end:
                logging.info(f"Detected tool call, starting to process tool call...")
                try:
                    is_success, tools_result_str, tool_timing = get_search_results_with_format(cur_result, **kwargs)
                    
                    # Record tool call time
                    step_timing_stats["total_tool_time"] += tool_timing.get("total_tool_time", 0)
                    
                    if not is_success:
                        logging.error(f"Tool call failed: {tools_result_str} - round: {attempt + 1}, error count: {error_count}")
                        error_count += 1
                        continue
                    logging.info(f"Tool call succeeded, result length: {len(tokenizer.encode(tools_result_str))}")

                    if attempt % kwargs.get("replan_interval", 5) == 0 and attempt != 0:
                        tools_result_str += "\n\n# Note: Now, you should summarize the task completion status and provide recommendations for next steps."
                        logging.info(f"Round {attempt}, adding replan prompt")

                    # Append tool call results to both complete history and API history
                    user_tool_response = {
                        "role": "user",
                        "content": tools_result_str
                    }
                    conversation_history.append(user_tool_response)
                    api_conversation_history.append(user_tool_response)
                        
                except Exception as parse_err:
                    logging.error(f"Failed to parse tool call content: {str(parse_err)} - round: {attempt + 1}, error count: {error_count}")
                    logging.error(f"Unparseable content: {cur_result[:200]}...")
            else:
                match_plan = re.search(r'<plan>', cur_result, re.DOTALL) and re.search(r'</plan>', cur_result, re.DOTALL)
                match_replan = re.search(r'<summary>', cur_result, re.DOTALL) and re.search(r'</summary>', cur_result, re.DOTALL)
                if not match_plan and not match_replan:
                    logging.error(f"Model reply did not find plan, replan or tool_call tag - round: {attempt + 1}, error count: {error_count}")
                    error_count += 1
                    continue
                elif match_plan and match_replan:
                    logging.error(f"Model reply found both plan and replan tags - round: {attempt + 1}, error count: {error_count}")
                    error_count += 1
                    continue
                elif match_plan:
                    logging.info(f"Model reply found plan tag, adding special suffix to user conversation")
                elif match_replan:
                    logging.info(f"Model reply found replan tag, adding special suffix to user conversation")
                # Append plan/summary response to both complete history and API history
                user_plan_response = {
                    "role": "user",
                    "content": "Based on the plan/summary and previous conversations, continue solving the task!"
                }
                conversation_history.append(user_plan_response)
                api_conversation_history.append(user_plan_response)
            
            # Record step time
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            step_timing_stats["step_details"].append({
                "step": attempt + 1,
                "duration": step_duration,
                "model_time": model_timing.get("api_call_time", 0),
                "tool_time": tool_timing.get("total_tool_time", 0) if 'tool_timing' in locals() else 0
            })
            logging.info(f"Step {attempt+1} completed - total time: {step_duration:.3f}s, model inference: {model_timing.get('api_call_time', 0):.3f}s, tool call: {tool_timing.get('total_tool_time', 0) if 'tool_timing' in locals() else 0:.3f}s")
            
        except Exception as e:
            logging.error(f"Round {attempt + 1} error: {str(e)} - error count: {error_count}, max rounds: {max_steps}")
            if attempt < max_steps - 1:
                wait_time = 1.0
                logging.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"Reached maximum retry count: {max_steps} - error count: {error_count}, final error: {str(e)}")
                step_timing_stats["total_steps"] = attempt + 1
                step_timing_stats["end_time"] = time.time()
                step_timing_stats["total_time"] = step_timing_stats["end_time"] - step_timing_stats["start_time"]
                return conversation_history, f"Round {attempt + 1} error: {str(e)}, reached maximum retry count: {max_steps}", "", step_timing_stats

    if tmp_answer:
        logging.info(f"Task completed, got answer: {tmp_answer[:100]}...")
        conversation_history.append({
            "role": "assistant",
            "content": tmp_answer
        })
        
        # Calculate final timing statistics
        step_timing_stats["total_steps"] = max_steps
        step_timing_stats["end_time"] = time.time()
        step_timing_stats["total_time"] = step_timing_stats["end_time"] - step_timing_stats["start_time"]
        
        logging.info(f"Inference completed - total steps: {step_timing_stats['total_steps']}, total time: {step_timing_stats['total_time']:.3f}s, "
                    f"total model inference time: {step_timing_stats['total_model_time']:.3f}s, "
                    f"total tool call time: {step_timing_stats['total_tool_time']:.3f}s")
        
        return conversation_history, None, tmp_answer, step_timing_stats
    else:
        logging.info(f"Answer not found, starting final summary step...")
        final_task = {
            "role": "user",
            "content": f'''Based on the above agent memory, please provide a brief answer to the following user task. You answer should be included within <answer> and </answer>'''
        }
        
        # Truncate context before API call (based on api_conversation_history)
        # Reserve output space for final summary step
        judge_max_tokens = 4096
        total_tokens = kwargs.get("total_tokens", 131072)
        # Add larger safety margin, considering token calculation deviation and format overhead
        safety_margin = 1024
        reserve_tokens = judge_max_tokens + safety_margin
        max_input_tokens = total_tokens - reserve_tokens
        
        truncated_history_summary = truncate_conversation_history(
            conversation=api_conversation_history,
            tokenizer=tokenizer,
            max_history_tokens=max_input_tokens,
            reserve_output_tokens=0  # Already subtracted in max_history_tokens
        )
        
        # Conversation for API call (based on truncated history)
        final_conversation = truncated_history_summary + [final_task]
        
        # Validate final conversation token count, ensure input tokens + judge_max_tokens <= total_tokens
        actual_final_input_tokens = sum(len(tokenizer.encode(msg.get('content', ''))) for msg in final_conversation)
        # Check if truncation was triggered
        is_truncated_summary = len(truncated_history_summary) < len(api_conversation_history)
        if is_truncated_summary:
            logging.info(f"Context truncated: from {len(api_conversation_history)} messages to {len(truncated_history_summary)} messages, "
                        f"input tokens: {actual_final_input_tokens}, reserved output tokens: {reserve_tokens}, "
                        f"total tokens: {actual_final_input_tokens + judge_max_tokens} <= {total_tokens}")
        # Append final_task to complete conversation_history (for saving)
        conversation_history.append(final_task)
        logging.info(f"After exceeding max rounds, final summary conversation length: {len(tokenizer.encode(str(final_conversation)))}")

        cur_judge_model_config_list = judge_model_config["config"][
            judge_model_config["pointer"] % len(judge_model_config["config"])]
        judge_model_id = judge_model_config["model_id"]
        judge_model_config["pointer"] += 1
        cur_result, stop_reason, final_model_timing = api_client(
            url=cur_judge_model_config_list[0],
            key=cur_judge_model_config_list[1],
            model=judge_model_id,
            conversation=final_conversation,
            max_tokens=judge_max_tokens,
        )
        
        # Record final model inference time
        step_timing_stats["total_model_time"] += final_model_timing.get("api_call_time", 0)
        conversation_history.append({
            "role": "assistant",
            "content": cur_result
        })
        answer_match = re.findall(r'<answer>(.*?)</answer>', cur_result, re.DOTALL)
        if answer_match:
            tmp_answer = answer_match[0].strip()
            logging.info(f"After exceeding max rounds, successfully parsed <answer> from model output: {tmp_answer}")
        else:
            tmp_answer = cur_result
            logging.error(f"After exceeding max rounds, failed to parse <answer> from model output: {tmp_answer}")
        
        # Calculate final timing statistics
        step_timing_stats["total_steps"] = max_steps
        step_timing_stats["end_time"] = time.time()
        step_timing_stats["total_time"] = step_timing_stats["end_time"] - step_timing_stats["start_time"]
        
        logging.info(f"Inference completed (exceeded max rounds) - total steps: {step_timing_stats['total_steps']}, total time: {step_timing_stats['total_time']:.3f}s, "
                    f"total model inference time: {step_timing_stats['total_model_time']:.3f}s, "
                    f"total tool call time: {step_timing_stats['total_tool_time']:.3f}s")
        
        return conversation_history, None, tmp_answer, step_timing_stats


def process_queries(infile, outfile, q_key, a_key, **kwargs):
    def decode_response(response):
        """
        Parse API response, support multiple output formats
        
        Supported formats:
        1. JSON format: {"judgement": "correct/incorrect"} or {"judgement": "yes/no"} (webthinker, refuse)
        2. Single letter: A, B, C (qa, browsecomp, etc.)
        3. Text format: Correct, Incorrect (gaia, etc.)
        4. Contains keywords: text containing "correct" or "incorrect" keywords
        
        Returns:
            dict: Dictionary containing judgement field, values are "correct", "incorrect", "yes" or "no"
        """
        if not isinstance(response, str):
            response = str(response)
        
        response = response.strip()
        
        # 1. Try to parse JSON format (webthinker, refuse)
        try:
            json_data = json.loads(response)
            if isinstance(json_data, dict) and "judgement" in json_data:
                judgement = json_data["judgement"].lower().strip()
                if judgement in ["correct", "incorrect", "yes", "no"]:
                    return {"judgement": judgement}
        except:
            pass
        
        # 2. Check if it's a single letter A/B/C (qa, browsecomp, etc.)
        response_upper = response.upper().strip()
        # Remove possible whitespace and punctuation
        response_clean = response_upper.replace(".", "").replace(" ", "").replace("\n", "").replace("\r", "")
        
        if response_clean == "A":
            # A usually means CORRECT (for qa may be CORRECT, for browsecomp is [CORRECT])
            return {"judgement": "correct"}
        elif response_clean == "B":
            # B usually means INCORRECT
            return {"judgement": "incorrect"}
        elif response_clean == "C":
            # C usually means NOT_ATTEMPTED (qa benchmark)
            return {"judgement": "incorrect"}  # NOT_ATTEMPTED treated as incorrect
    
        # 3. Check if contains "correct" or "incorrect" keywords (case insensitive)
        response_lower = response.lower()
        
        # Check yes/no related keywords (for refuse judgment)
        # Prioritize checking cases containing "no", but exclude cases with explicit negative words
        if any(keyword in response_lower for keyword in ["no", "不", "不是"]):
            # If also contains explicit negative words, prioritize returning incorrect
            if any(keyword in response_lower for keyword in ["not correct", "is not correct", "not accurate", "is incorrect", "incorrect", "【错误】", "[incorrect]", "错误", "不对", "不正确"]):
                return {"judgement": "incorrect"}
            return {"judgement": "no"}
        
        if any(keyword in response_lower for keyword in ["yes", "是", "对的"]):
            # Exclude "no" cases
            if "no" not in response_lower and "不" not in response_lower and "错误" not in response_lower:
                return {"judgement": "yes"}
        
        # Check explicit negative words
        if any(keyword in response_lower for keyword in ["not correct", "is not correct", "not accurate", "is incorrect", "incorrect", "【错误】", "[incorrect]", "错误", "不对", "不正确"]):
            return {"judgement": "incorrect"}
        
        # Check correct-related keywords (placed last to avoid conflict with yes/no)
        if any(keyword in response_lower for keyword in ["correct", "【正确】", "[correct]", "正确"]):
            # Exclude "incorrect" cases
            if "incorrect" not in response_lower and "错误" not in response_lower and "not correct" not in response_lower:
                return {"judgement": "correct"}
        
        # 4. Default to incorrect
        logging.warning(f"Unable to parse judge response, using default value incorrect. Response content: {response[:100]}")
        return {"judgement": "incorrect"}
    
    current_judge_prompt = judge_prompt
    logging.info(f"Using fixed judge_prompt: {current_judge_prompt[:100]}...")

    # Read input data
    if infile.endswith(".json"):
        questions_data = read_json(infile)
    elif infile.endswith(".jsonl"):
        questions_data = read_jsonl(infile)
    else:
        raise ValueError(f"Unsupported file format: {infile}")

    # Check if output file exists and deduplicate
    out_data = []
    out_set = set()
    if os.path.exists(outfile):
        out_data = read_jsonl(outfile)
        unique_out_data = []
        unique_out_set = set()
        for item in out_data:
            if item["question"] in unique_out_set:
                continue
            unique_out_data.append(item)
            unique_out_set.add(item["question"])
        out_data = unique_out_data
        write_jsonl(out_data, outfile)
        out_set = set([item["question"] for item in out_data])

    logging.info(outfile)
    new_questions_data = [item for item in questions_data if item[q_key] not in out_set]
    logging.info(f"Initial data: {len(questions_data)}, filtered data: {len(new_questions_data)}")
    questions_data = new_questions_data

    # Initialize statistics and shared queues
    stats = {"total": len(new_questions_data), "success": 0, "failed": 0}

    task_queue = Queue()
    result_queue = Queue()
    write_lock = Lock()  # Lock for file writing
    
    # Initialize processed questions set, load from existing output file
    processed_questions = set()
    if os.path.exists(outfile):
        existing_data = read_jsonl(outfile)
        for item in existing_data:
            if item.get("status") == "completed" and item.get("question"):
                processed_questions.add(item["question"])
        logging.info(f"Loaded {len(processed_questions)} processed questions from existing file")

    # Producer function - put tasks into queue
    def producer():
        # Use original data index, not filtered data index
        for original_idx, question_data in enumerate(questions_data):
            question = question_data[q_key]
            # Check if already processed at task assignment stage
            if question in processed_questions:
                logging.info(f"Skipping already processed question: {question[:50]}...")
                continue
            # Find corresponding index in original data
            original_data_idx = next((i for i, item in enumerate(questions_data) if item[q_key] == question), original_idx)
            task_queue.put((original_data_idx, question_data))
        # Put end markers
        for _ in range(kwargs.get("parallel", 4)):
            task_queue.put(None)

    # Consumer function - get tasks from queue and process
    def consumer():
        # Get judge model configuration using round-robin
        cur_judge_model_config_list = judge_model_config["config"][
            judge_model_config["pointer"] % len(judge_model_config["config"])]
        judge_model_id = judge_model_config["model_id"]
        judge_model_config["pointer"] += 1
        # Use model URL pool for inference
        fixed_url = get_next_model_url()

        nonlocal stats, processed_questions
        while True:
            task = task_queue.get()
            if task is None:  # End marker
                break

            idx, question_data = task
            question = question_data[q_key]
            golden_answer = question_data[a_key]
            level = question_data.get('Level', '-1')
            
            # Mark question as being processed (prevent concurrent processing of same question)
            with write_lock:  # Use lock to protect shared state
                processed_questions.add(question)

            max_retry = kwargs.get("max_retry", INFER_KWARGS.get("max_retry", 3))
            result = 0  # Default to failure
            trace = None

            for retry in range(max_retry):
                trace = {
                    "question_id": str(idx),
                    "question": question,
                    "Level": level,
                    "golden_answer": golden_answer,
                    "prediction": None,
                    "llm_judge": 0,
                    "tag": None,
                    "steps": [],
                    "status": None,
                    "error": None,
                    "trajectory":None,
                }

                logging.info(f"Starting to process question {idx}: {question[:100]}...")
                conversation_history, failed_reason, prediction, timing_stats = process_single_data(question, fixed_url=fixed_url, **kwargs)
                trace["prediction"] = prediction
                trace['trajectory'] = conversation_history
                trace['timing_stats'] = timing_stats

                if failed_reason:
                    logging.error(f"Question {idx} processing failed: {failed_reason} - retry count: {retry + 1}/{max_retry}")
                    trace["error"] = failed_reason
                    trace["status"] = "failed"
                    
                    # Output timing statistics on failure
                    if timing_stats:
                        logging.info(f"Question {idx} failure timing statistics - total steps: {timing_stats.get('total_steps', 0)}, "
                                    f"total time: {timing_stats.get('total_time', 0):.3f}s, "
                                    f"model inference: {timing_stats.get('total_model_time', 0):.3f}s, "
                                    f"tool call: {timing_stats.get('total_tool_time', 0):.3f}s")
                else:
                    # Check if prediction exists
                    if trace["prediction"]:
                        trace["prediction"] = prediction
                        trace["status"] = "completed"
                    else:
                        logging.error(f"Prediction is empty - question ID: {idx}, question: {question[:50]}...")
                        trace["status"] = "invalid_prediction"
                        trace["error"] = f"prediction not found: {trace['prediction']}"

                # Only perform LLM evaluation when status is completed and no error
                if not trace.get("error") and trace.get("status") == "completed":
                    logging.info(f"Starting LLM evaluation...")
                    # LLM_JUDGE - use corresponding benchmark's judge_prompt
                    llm_evaluation_prompt = current_judge_prompt.format(
                        question=question,
                        gt_answer=golden_answer,
                        pred_answer=trace["prediction"]
                    )
                    output, stop_reason, evaluation_timing = api_client(
                        url=cur_judge_model_config_list[0],
                        key=cur_judge_model_config_list[1],
                        model=judge_model_id,
                        system="You are an evaluation assistant.",
                        prompt=llm_evaluation_prompt,
                        temperature=0.1,
                    )
                    json_output = decode_response(output)
                    if (json_output and isinstance(json_output, dict) and
                            "judgement" in json_output and
                            json_output['judgement'].lower() == "correct"):
                        trace['llm_judge'] = 1
                        logging.info(f"LLM evaluation result: correct")
                    else:
                        trace['llm_judge'] = 0
                        logging.info(f"LLM evaluation result: incorrect")

                    # EM_JUDGE
                    logging.info(f"Starting EM evaluation...")
                    final_em_judge = 0
                    final_subem_judge = 0
                    prediction_list = trace["prediction"].split("|")
                    golden_answers=[golden_answer]
                    assert isinstance(golden_answers, list)
                    for prediction in prediction_list:
                        em_judge = em_check(prediction, golden_answers)
                        subem_judge = subem_check(prediction, golden_answers)
                        final_em_judge = max(final_em_judge, em_judge)
                        final_subem_judge = max(final_subem_judge, subem_judge)
                    trace["em_judge"] = final_em_judge
                    trace["subem_judge"] = final_subem_judge
                    logging.info(f"EM evaluation result: EM={final_em_judge}, SubEM={final_subem_judge}")

                    # Use result level to display important result information
                    logging.info("#" * 50)
                    logging.info(f"Question {idx} processing completed:")
                    logging.info(f"-- Level: {trace['Level']}")
                    logging.info(f"-- question: {question[:100]}...")
                    logging.info(f"-- predicted_answer: {trace['prediction'][:100]}...")
                    logging.info(f"-- golden_answer: {golden_answer[:100]}...")
                    logging.info(f"-- llm_judge: {trace['llm_judge']}")
                    logging.info(f"-- em_judge: {trace['em_judge']}")
                    logging.info(f"-- subem_judge: {trace['subem_judge']}")
                    logging.info(f"-- status: {trace['status']}")
                    
                    # Display timing statistics
                    if timing_stats:
                        logging.info(f"-- Timing statistics:")
                        logging.info(f"  - Total steps: {timing_stats.get('total_steps', 0)}")
                        logging.info(f"  - Total time: {timing_stats.get('total_time', 0):.3f}s")
                        logging.info(f"  - Model inference time: {timing_stats.get('total_model_time', 0):.3f}s")
                        logging.info(f"  - Tool call time: {timing_stats.get('total_tool_time', 0):.3f}s")
                        
                        # Display detailed time for each step
                        if timing_stats.get('step_details'):
                            logging.info(f"  - Detailed time per step:")
                            for step_detail in timing_stats['step_details']:
                                logging.info(f"    Step {step_detail['step']}: total time {step_detail['duration']:.3f}s, "
                                          f"model inference {step_detail['model_time']:.3f}s, "
                                          f"tool call {step_detail['tool_time']:.3f}s")
                    
                    logging.info("#" * 50)

                    trace["raw"] = question_data
                    result = 1  # Success count
                    break  # Exit retry loop on success
                else:
                    # Record error information when there's an error
                    logging.error(f"Question {idx} processing failed, reason: {trace['error']} - retry count: {retry + 1}/{max_retry}, status: {trace.get('status', 'unknown')}")

            # Put result into result queue
            result_queue.put((result, trace))
            task_queue.task_done()
        return 

    def result_writer():
        nonlocal stats
        # Read existing data first
        existing_data = []
        if os.path.exists(outfile):
            existing_data = read_jsonl(outfile)
        
        # Get statistics script path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cal_stats_script = os.path.join(current_dir, "cal_stats.py")
        
        while True:
            result_item = result_queue.get()
            if result_item is None:  # End marker
                break
            result, trace = result_item
            # Only process when prediction is not empty
            if trace.get("prediction"):
                if result == 1:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                # Append to existing data
                existing_data.append(trace)
                # Only lock when writing, ensure atomicity of each write
                with write_lock:
                    # Write to file (append mode)
                    write_jsonl([trace], outfile, "a")
                    
                    # Immediately run statistics script to update statistics
                    try:
                        if os.path.exists(cal_stats_script):
                            abs_outfile = os.path.abspath(outfile)
                            # Run single file statistics script
                            result = subprocess.run([
                                "python", cal_stats_script, abs_outfile
                            ], capture_output=True, text=True, cwd=current_dir, timeout=30)
                            
                            if result.returncode == 0:
                                logging.info(f"Statistics updated: {outfile.replace('.jsonl', '.output_stats.txt')}")
                            else:
                                logging.warning(f"Statistics script execution failed: {result.stderr}")
                        else:
                            logging.warning(f"Statistics script does not exist: {cal_stats_script}")
                    except subprocess.TimeoutExpired:
                        logging.warning("Statistics script execution timeout, skipping this statistics update")
                    except Exception as e:
                        logging.warning(f"Error running statistics script: {str(e)}")
            else:
                # Don't count cases where prediction is empty
                logging.warning(f"Skipping write: prediction is empty (question ID: {trace.get('question_id', 'unknown')}, question: {trace.get('question', '')[:50]}..., status: {trace.get('status', 'unknown')})")
            result_queue.task_done()
        return 

    # Create thread pool
    num_workers = kwargs.get("parallel", 4)
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        # Start producer thread
        executor.submit(producer)
        # Start consumer threads
        consumer_futures = [executor.submit(consumer) for _ in range(num_workers)]
        # Start result writer thread
        writer_future = executor.submit(result_writer)
        # Wait for all tasks to complete
        for future in as_completed(consumer_futures):
            future.result()
        # After all consumers complete, send end signal to writer thread
        result_queue.put(None)
        writer_future.result()

    # Save statistics
    stats_file = outfile.replace(".jsonl", ".param_stats.json")
    write_json({**SHOW_KWARGS, **stats}, stats_file)
    # Use result level to display final statistics
    logging.info(f"Processing completed! Success: {stats['success']}, Failed: {stats['failed']}, Total: {len(new_questions_data)}")
    return outfile


def process_single_dataset(infile, outfile_base, q_key, a_key, **infer_kwargs):
    def ensure_directory_exists(file_path):
        """Ensure directory for file exists, create if it doesn't"""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")

    """Function to process a single dataset"""
    start_time = time.time()
    # Ensure output directory exists
    ensure_directory_exists(outfile_base)
    
    # Process a single output file (no round slicing)
    process_queries(infile, outfile_base, q_key, a_key, **infer_kwargs)
    
    # Analyze results
    stats_file = outfile_base.replace(".jsonl", ".output_stats.txt")
    bad_case_file = outfile_base.replace(".jsonl", ".bad_case.jsonl")
    
    # Ensure directories for statistics file and bad case file exist
    ensure_directory_exists(stats_file)
    ensure_directory_exists(bad_case_file)
    
    # Note: Statistics script now runs automatically on each data write, no need for batch run
    
    cost_time = time.time() - start_time
    # Use result level to display dataset processing completion information
    logging.info(f"Dataset {infile} processing completed, time: {cost_time:.2f}s")
    return infile, cost_time


def main():
    # Enable logging by default (INFO level)
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Detailed logging enabled, showing all log levels")
    
    # Display configuration information
    for key, value in SHOW_KWARGS.items():
        logging.info(f">>>> {key}: {value}")
    
    # Get single benchmark from INFER_KWARGS
    benchmark = INFER_KWARGS.get("benchmark")
    
    # Define dataset configuration based on benchmark
    dataset_config = None
    if benchmark == "gaia":
        dataset_config = {
            "infile": "./inference/benchmarks/gaia/gaia_dev_103.json",
            "q_key": "question",
            "a_key": "answer",
        }
    elif benchmark == "browsecomp":
        dataset_config = {
            "infile": "./inference/benchmarks/browsecomp/browsecomp.json",
            "q_key": "question",
            "a_key": "answer",
        }
    elif benchmark == "xbench":
        dataset_config = {
            "infile": "./inference/benchmarks/xbench_deepsearch/DeepSearch_qa.jsonl",
            "q_key": "question",
            "a_key": "answer",
        }
    elif benchmark == "webwalker":
        dataset_config = {
            "infile": "./inference/benchmarks/webwalker/webwalker_main.jsonl",
            "q_key": "question",
            "a_key": "answer",
        }
    elif benchmark == "frames":
        dataset_config = {
            "infile": "./inference/benchmarks/frames/frames.jsonl",
            "q_key": "Prompt",
            "a_key": "Answer",
        }
    elif benchmark == "seal_0":
        dataset_config = {
            "infile": "./inference/benchmarks/seal0/seal_0.jsonl",
            "q_key": "question",
            "a_key": "answer",
        }
    else:
        logging.error(f"Unknown benchmark: {benchmark}")
        return
    
    if dataset_config is None:
        logging.error(f"Failed to configure dataset for benchmark: {benchmark}")
        return
    
    filename = dataset_config["infile"].split("/")[-1].replace(".json", "").replace(".jsonl", "")
    outfile_base = f"./results/{MODEL}/{filename}.summary_model_{replan_model_config['model_id']}.replan_interval_{INFER_KWARGS['replan_interval']}.total_tokens_{INFER_KWARGS['total_tokens']}.max_steps_{INFER_KWARGS['max_steps']}.t_{INFER_KWARGS['temperature']}.search_topk_{INFER_KWARGS['web_topk']}.jsonl"

    # Record start time
    total_start_time = time.time()
    
    # Process single dataset
    try:
        dataset_path, cost_time = process_single_dataset(
            dataset_config["infile"],
            outfile_base,
            dataset_config["q_key"],
            dataset_config["a_key"],
            **INFER_KWARGS,
        )
        logging.info(f"Dataset {dataset_path} processing completed, time: {cost_time:.2f}s")
    except Exception as e:
        logging.error(f"Error processing dataset: {str(e)}")
    
    # Calculate total time
    total_cost_time = time.time() - total_start_time
    logging.info(f"Dataset processing completed, total time: {total_cost_time:.2f}s")
    
    # Generate final statistics report (single file mode)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cal_stats_script = os.path.join(current_dir, "cal_stats.py")
        
        if os.path.exists(cal_stats_script):
            logging.info("Starting final statistics report generation...")
            logging.info(f"Generating statistics for dataset: {outfile_base}")
            abs_outfile_base = os.path.abspath(outfile_base)
            result = subprocess.run(
                ["python", cal_stats_script, abs_outfile_base],
                capture_output=True,
                text=True,
                cwd=current_dir
            )

            if result.returncode == 0:
                stats_file = outfile_base.replace('.jsonl', '.output_stats.txt')
                logging.info(f"Statistics report generated successfully: {stats_file}")
            else:
                logging.error(f"Statistics script execution failed: {result.stderr}")
        else:
            logging.warning(f"Statistics script does not exist: {cal_stats_script}")
    except Exception as e:
        logging.error(f"Error running statistics script: {str(e)}")

    return 


if __name__ == "__main__":
    main()