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

import time
import os
import random
import argparse
import json
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from FlashOAgents import SearchAgent, OpenAIServerModel, custom_role_conversions
from utils import read_jsonl, write_jsonl, read_json


# Load environment variables
load_dotenv(override=True)

# Global API key rotator
class APIKeyRotator:
    def __init__(self, keys_string, key_type):
        self.keys = [key.strip() for key in keys_string.split('|') if key.strip()]
        self.key_type = key_type
        self.current_index = 0
        self.lock = threading.Lock()
        
        if not self.keys:
            raise ValueError(f"No valid {key_type} keys provided")
        
        print(f"Initialized {key_type} key rotator with {len(self.keys)} key(s)")
    
    def get_current_key(self):
        with self.lock:
            return self.keys[self.current_index]
    
    def rotate_key(self):
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.keys)
            print(f"Rotated to {self.key_type} key #{self.current_index + 1}")
    
    def get_all_keys(self):
        return self.keys.copy()

# Global key rotator instances
serper_rotator = None
jina_rotator = None


def _get_serper_keys_from_env():
    """Read Serper keys from env, supporting both legacy and current names."""
    return os.environ.get("SERPER_API_KEY", "") or os.environ.get("WEB_SEARCH_SERPER_API_KEY", "")

def init_api_key_rotators():
    """Initialize API key rotators."""
    global serper_rotator, jina_rotator
    
    try:
        serper_keys = _get_serper_keys_from_env()
        if serper_keys:
            serper_rotator = APIKeyRotator(serper_keys, "SERPER")
    except Exception as e:
        print(f"Warning: Failed to initialize SERPER key rotator: {e}")
    
    try:
        jina_keys = os.environ.get("JINA_API_KEY", "")
        if jina_keys:
            jina_rotator = APIKeyRotator(jina_keys, "JINA")
    except Exception as e:
        print(f"Warning: Failed to initialize JINA key rotator: {e}")

def get_current_serper_key():
    """Get current SERPER API key."""
    if serper_rotator:
        return serper_rotator.get_current_key()
    return _get_serper_keys_from_env()

def get_current_jina_key():
    """Get current JINA API key."""
    if jina_rotator:
        return jina_rotator.get_current_key()
    return os.environ.get("JINA_API_KEY", "")

def rotate_serper_key():
    """Rotate SERPER API key."""
    if serper_rotator and len(serper_rotator.keys) > 1:
        serper_rotator.rotate_key()

def rotate_jina_key():
    """Rotate JINA API key."""
    if jina_rotator and len(jina_rotator.keys) > 1:
        jina_rotator.rotate_key()

def process_item(item, model, summary_interval, prompts_type, max_steps, max_retries=3):
    question = item["question"]
    golden_answer = item["answer"]
    model_id = str(model)
    
    for attempt in range(max_retries):
        try:
            # Update API keys in environment variables
            current_serper_key = get_current_serper_key()
            os.environ["WEB_SEARCH_SERPER_API_KEY"] = current_serper_key
            os.environ["SERPER_API_KEY"] = current_serper_key
            os.environ["JINA_API_KEY"] = get_current_jina_key()
            
            search_agent = SearchAgent(model, summary_interval=summary_interval, prompts_type=prompts_type, max_steps=max_steps)
            result = search_agent(question)
            
            # Check if result contains an error
            if "error" in result:
                error_msg = result["error"]
                print(f"Agent returned error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # Analyze error type
                if "reasoning_content" in error_msg and "NoneType" in error_msg:
                    print("Detected reasoning_content NoneType error, likely due to model response format issue")
                    # For this specific error, try adjusting parameters
                    if attempt < max_retries - 1:
                        # Try reducing max_steps or adjusting other parameters
                        adjusted_max_steps = max(1, max_steps - 2)
                        print(f"Adjusting max_steps from {max_steps} to {adjusted_max_steps}")
                        search_agent = SearchAgent(model, summary_interval=summary_interval, prompts_type=prompts_type, max_steps=adjusted_max_steps)
                        result = search_agent(question)
                        if "error" not in result:
                            return {
                                "question": question,
                                "golden_answer": golden_answer,
                                "model_id": model_id,
                                **result,
                            }
                
                if attempt == max_retries - 1:
                    # Last attempt failed, return error result
                    print(f"Question '{question[:50]}...' processing failed: {error_msg}")
                    return {
                        "question": question,
                        "golden_answer": golden_answer,
                        "model_id": model_id,
                        "error": error_msg,
                        "agent_result": "Processing failed",
                        "agent_trajectory": []
                    }
                
                # Wait before retrying
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            
            return {
                "question": question,
                "golden_answer": golden_answer,
                "model_id": model_id,
                **result,
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"Exception occurred when calling multi_agent (attempt {attempt + 1}/{max_retries}): {error_msg}")
            
            # Analyze error type
            if "reasoning_content" in error_msg and "NoneType" in error_msg:
                print("Detected reasoning_content NoneType error, likely due to model response format issue")
            
            # Check if it's an API key-related error
            if any(keyword in error_msg.lower() for keyword in ['403', '401', 'forbidden', 'unauthorized', 'quota', 'limit']):
                print("Detected API key-related error, attempting to rotate key...")
                
                # Rotate the appropriate key based on error type
                if 'serper' in error_msg.lower() or 'search' in error_msg.lower():
                    rotate_serper_key()
                elif 'jina' in error_msg.lower() or 'crawl' in error_msg.lower():
                    rotate_jina_key()
                else:
                    # If unable to determine which API failed, rotate all keys
                    rotate_serper_key()
                    rotate_jina_key()
            
            if attempt == max_retries - 1:
                # Last attempt failed, return error result
                print(f"Question '{question[:50]}...' processing failed: {error_msg}")
                return {
                    "question": question,
                    "golden_answer": golden_answer,
                    "model_id": model_id,
                    "error": error_msg,
                    "agent_result": "Processing failed",
                    "agent_trajectory": []
                }
            
            # Wait before retrying
            time.sleep(2 ** attempt)  # Exponential backoff


def main(args):
    # Initialize API key rotators
    print("Initializing API key rotators...")
    init_api_key_rotators()

    # Check required environment variables
    model_name = os.environ.get("DEFAULT_MODEL")
    if not model_name:
        print("Error: DEFAULT_MODEL environment variable is not set")
        exit(1)
    
    model = OpenAIServerModel(
        model_name,
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=32768,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE"),
    )

    if args.infile.lower().endswith('.json'):
        with open(args.infile, 'r') as f:
            data = json.load(f)
    else:
        data = read_jsonl(args.infile)

    if args.sample_num is not None:
        data = data[:args.sample_num]
    try:
        out_data = read_jsonl(args.outfile)
    except Exception:
        out_data = []
    done_questions = set([item.get("question") for item in out_data])
    data_to_run = [item for item in data if item.get("question") not in done_questions]
    print(f"Total data: {len(data)}, completed: {len(done_questions)}, remaining: {len(data_to_run)}")

    results = []
    file_lock = threading.Lock()

    def safe_write(result):
        with file_lock:
            # Only save results without errors
            if "error" not in result:
                write_jsonl(args.outfile, [result], "a")
            else:
                print(f"Skipping error result: {result.get('error', 'Unknown error')}")

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        summary_interval = random.randint(args.summary_interval - 1, args.summary_interval + 1)

        futures = [
            executor.submit(process_item, item, model, summary_interval, args.prompts_type, args.max_steps, args.max_retries) for item in data_to_run
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            results.append(result)
            safe_write(result)

    # Summarize successful and failed results
    successful_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]
    
    print(f"Processing complete:")
    print(f"  This batch: {len(results)} item(s)")
    print(f"  Successful: {len(successful_results)} item(s)")
    print(f"  Failed: {len(failed_results)} item(s)")
    print(f"  Total completed: {len(done_questions) + len(successful_results)} item(s)")
    
    if failed_results:
        print(f"Failure reason summary:")
        error_counts = {}
        for result in failed_results:
            error = result.get("error", "Unknown error")
            error_counts[error] = error_counts.get(error, 0) + 1
        for error, count in error_counts.items():
            print(f"  {error}: {count} time(s)")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data generation')

    parser.add_argument('--infile', type=str, default="./data/hle.json", help='input path')
    parser.add_argument('--outfile', type=str, default="./output/hle_gpt_5mini_summary.jsonl", help='output path')
    parser.add_argument('--sample_num', type=int, default=None, help='sample num')
    parser.add_argument('--summary_interval', type=int, default=8, help='summary interval')
    parser.add_argument('--prompts_type', type=str, default="generation", help='prompts type')
    parser.add_argument('--parallel', type=int, default=100, help='parallel steps')
    parser.add_argument('--max_steps', type=int, default=50, help='max steps')
    parser.add_argument('--max_retries', type=int, default=3, help='max retries for API failures')

    args = parser.parse_args()
    
    main(args)
