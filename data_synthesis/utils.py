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
from typing import Any, Dict, List, Union


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL (JSON Lines) file and return a list of records.
    Each line is parsed as a JSON object.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Warning: failed to parse line {line_num}, skipping.")
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
    return data


def write_jsonl(
    data: List[Dict[str, Any]],
    file_path: str,
    append: bool = False,
    ensure_ascii: bool = False,
) -> bool:
    """
    Write a list of records to a JSONL file (one JSON object per line).

    Args:
        data: List of JSON-serialisable objects.
        file_path: Destination file path.
        append: If True, append to an existing file; otherwise overwrite.
        ensure_ascii: If True, escape non-ASCII characters.

    Returns:
        True on success, False on failure.
    """
    try:
        mode = "a" if append else "w"
        with open(file_path, mode, encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False


def read_json(file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """
    Read a standard JSON file and return the parsed object.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed Python object, or None on failure.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
    except Exception as e:
        print(f"Error reading file: {e}")
    return None


def write_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    file_path: str,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> bool:
    """
    Write a Python object to a JSON file.

    Args:
        data: Dict or list to serialise.
        file_path: Destination file path.
        indent: Number of spaces for indentation.
        ensure_ascii: If True, escape non-ASCII characters.
        sort_keys: If True, sort dictionary keys.

    Returns:
        True on success, False on failure.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False


def safe_json_loads(s: str) -> Any:
    """
    Attempt to parse a JSON string, with minimal error recovery
    (trailing commas before } or ]).

    Returns the parsed object, or None if parsing fails.
    """
    try:
        return json.loads(s)
    except Exception:
        s2 = re.sub(r",\s*}", "}", s)
        s2 = re.sub(r",\s*]", "]", s2)
        try:
            return json.loads(s2)
        except Exception:
            return None
