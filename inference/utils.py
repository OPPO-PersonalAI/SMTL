#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from typing import List, Dict, Any, Union

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error in read_jsonl: {str(e)}")
    return data


def write_jsonl(
    data: List[Dict[str, Any]], 
    file_path: str, 
    append: bool = False, 
    ensure_ascii: bool = False
) -> bool:
    try:
        mode = 'a' if append else 'w'
        with open(file_path, mode, encoding='utf-8') as file:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=ensure_ascii) + '\n'
                file.write(json_line)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def read_json(file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def write_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]], 
    file_path: str, 
    indent: int = 2, 
    ensure_ascii: bool = False,
    sort_keys: bool = False
) -> bool:
    try:
        with open(file_path, "w", encoding='utf-8') as file:
            json.dump(
                data, 
                file, 
                indent=indent, 
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys
            )
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def count_tokens(text, tokenizer):
    if not text:
        return 0
    return len(tokenizer.encode(text))

def extract_specific_tag(text):
    """
    Parse text containing standard tool call format and extract all tool_call information
    
    Args:
        text (str): Text containing tool calls
    
    Returns:
        tuple: (think_content, tool, parsed_content)
            - think_content: Thinking content before tool calls
            - tool: Tool type identifier
            - parsed_content: List of parsed tool call content
    """
    # Extract all tool_call tags directly from the entire text
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    tool_calls = re.findall(tool_call_pattern, text, re.DOTALL)
    
    if tool_calls:
        # If tool_call tags are found, parse them
        parsed_content = []
        
        for call_idx, call in enumerate(tool_calls, start=1):
            try:
                # 1. Try to parse tool call JSON
                try:
                    call_dict = json.loads(call)
                except Exception as e:
                    # JSON parsing may throw various exceptions, including JSONDecodeError, ValueError, TypeError, etc.
                    parsed_item = {
                        'type': "dummy_tool",
                        'id': call_idx,  # Use index to generate unique error ID
                        'raw': call,
                        'error_msg': f"JSON parsing failed with error message: {str(e)}, please output arguments in json format string."
                    }
                    parsed_content.append(parsed_item)
                    continue

                # 2. Check if required dictionary keys exist
                required_keys = ['name', 'arguments']
                missing_keys = [k for k in required_keys if k not in call_dict]
                if missing_keys:
                    parsed_item = {
                        'type': "dummy_tool",
                        'id': call_idx,
                        'raw': call,
                        'error_msg': f"Tool call has no {', '.join(missing_keys)}"
                    }
                    parsed_content.append(parsed_item)
                    continue

                # 3. Extract basic information
                tool_type = call_dict['name']
                arguments = call_dict['arguments']

                # 4. Build basic parsed item (with default error field value)
                parsed_item = {
                    'type': tool_type,
                    'id': call_idx,
                    'raw': call,
                    'error_msg': None  # None when there's no error
                }

                # 5. Validate and add parameters based on tool type, record error when parameters are missing
                if tool_type == 'web_search':
                    query = arguments.get('query', '')
                    parsed_item['query'] = query

                elif tool_type == 'crawl_page':
                    url = arguments.get('url', '')
                    query = arguments.get('query', '')
                    parsed_item['url'] = url
                    parsed_item['query'] = query

                else:
                    parsed_item['type'] = "dummy_tool"
                    parsed_item['error_msg'] = "Unknown tool name."

                # 6. Add parsed item to result list
                parsed_content.append(parsed_item)

            # Catch other unexpected errors
            except Exception as e:
                parsed_item = {
                    'type': "dummy_tool",
                    'id': call_idx,
                    'raw': call,
                    'error_msg': f"Parsing arguments failed with error message: {str(e)}, please output arguments in json format string."
                }
                parsed_content.append(parsed_item)

        # Extract thinking content (content before tool_call)
        think_content = text.split('<tool_call>')[0].strip() if '<tool_call>' in text else None
        return think_content, "tool_call", parsed_content
    
    else:
        # If no tool_call is found, return empty result
        return None, "tool_call", []

def truncate_conversation_history(
    conversation: List[Dict[str, str]], 
    tokenizer,
    max_history_tokens: int = 100000,
    reserve_output_tokens: int = 0
) -> List[Dict[str, str]]:
    """
    Truncate conversation history based on maximum token count
    
    Args:
        conversation: Conversation history list
        tokenizer: Tokenizer instance
        max_history_tokens: Maximum allowed tokens for conversation history
        reserve_output_tokens: Tokens to reserve for output (used to calculate actual available input space)
    
    Returns:
        Truncated conversation history list
        
    Notes:
        1. Always keep the first two messages: system prompt and first user prompt
        2. If total tokens exceed the limit, delete messages starting from the 3rd message (oldest to newest)
        3. Continue until total tokens meet the limit or only the first two messages remain
        4. If reserve_output_tokens is set, actual limit is max_history_tokens - reserve_output_tokens
    """
    if not conversation:
        return conversation
    
    def calculate_total_tokens(msgs):
        """Calculate total tokens in message list"""
        return sum(count_tokens(msg.get('content', ''), tokenizer) for msg in msgs)
    
    # Calculate actual available input tokens (need to reserve space for output)
    actual_limit = max_history_tokens - reserve_output_tokens
    
    # If conversation history has no more than 2 messages, return directly
    if len(conversation) <= 2:
        return conversation
    
    # Calculate current total tokens
    total_tokens = calculate_total_tokens(conversation)
    
    # If not exceeding limit, return directly
    if total_tokens <= actual_limit:
        return conversation
    
    # Always keep the first two messages
    fixed_messages = conversation[:2]
    dynamic_messages = list(conversation[2:])  # Variable part, from oldest to newest
    
    # Delete from oldest message until limit is met
    while dynamic_messages:
        current_conversation = fixed_messages + dynamic_messages
        current_tokens = calculate_total_tokens(current_conversation)
        
        if current_tokens <= actual_limit:
            return current_conversation
        
        # Delete the oldest message (first dynamic message)
        dynamic_messages.pop(0)
    
    # If all dynamic messages are deleted and still exceeds limit, return only the fixed first two messages
    return fixed_messages

def should_truncate(
    conversation: List[Dict[str, str]], 
    tokenizer,
    max_history_tokens: int = 100000,
    reserve_output_tokens: int = 0
) -> List[Dict[str, str]]:
    if not conversation:
        return conversation
    
    def calculate_total_tokens(msgs):
        """Calculate total tokens in message list"""
        return sum(count_tokens(msg.get('content', ''), tokenizer) for msg in msgs)
    
    # Calculate actual available input tokens (need to reserve space for output)
    actual_limit = max_history_tokens - reserve_output_tokens

    # Calculate current total tokens
    total_tokens = calculate_total_tokens(conversation)
    
    # If not exceeding limit, return False (no truncation needed)
    if total_tokens <= actual_limit:
        return False
    else:
        return True