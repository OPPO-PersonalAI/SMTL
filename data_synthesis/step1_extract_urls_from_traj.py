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
Step 1: Process trajectory messages.

- Input : Records containing fields such as question / answer / messages
          (messages is a list[dict] inference trajectory).
- Processing:
    1. visited_urls  – Extract all <tool_call>…</tool_call> JSON blocks from
       assistant messages.  For tool_call.name == "visit" or "crawl_page",
       collect url and goal/query.  Deduplicate by URL, keeping the last
       occurrence; output ordered by last-occurrence position.
    2. related_urls  – Extract URLs from the user message that follows each
       "search" tool call; associate each URL with the corresponding query.
       Exclude URLs already present in visited_urls.
- Output: Original record + new fields visited_urls and related_urls,
          written to data_synthesis/cache/cache_1/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


def safe_json_loads(s: str) -> Any:
    """Try to parse JSON; on failure attempt minimal recovery (trailing commas)."""
    try:
        return json.loads(s)
    except Exception:
        s2 = re.sub(r",\s*}", "}", s)
        s2 = re.sub(r",\s*]", "]", s2)
        try:
            return json.loads(s2)
        except Exception:
            return None


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    # Handle ```json ... ``` or ``` ... ``` wrapping
    if s.startswith("```"):
        s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _extract_tool_call_blocks(text: str) -> List[str]:
    """Extract all <tool_call>…</tool_call> inner strings in order."""
    if not text:
        return []
    blocks: List[str] = []
    i = 0
    while True:
        s = text.find(TOOL_CALL_START, i)
        if s == -1:
            break
        e = text.find(TOOL_CALL_END, s + len(TOOL_CALL_START))
        if e == -1:
            break
        inner = text[s + len(TOOL_CALL_START): e].strip()
        if inner:
            blocks.append(inner)
        i = e + len(TOOL_CALL_END)
    return blocks


def _parse_tool_call_payload(block: str) -> Optional[Dict[str, Any]]:
    """Parse the inner string of a <tool_call> block into a dict."""
    if not block:
        return None
    raw = _strip_code_fence(block)
    obj = safe_json_loads(raw)
    if isinstance(obj, dict):
        return obj
    # Fallback: try to extract first {...} span
    l = raw.find("{")
    r = raw.rfind("}")
    if l != -1 and r != -1 and r > l:
        obj = safe_json_loads(raw[l:r + 1])
        if isinstance(obj, dict):
            return obj
    return None


def _normalize_urls(x: Any) -> List[str]:
    """
    Normalise a url field to List[str].  Handles:
    - str
    - list (recursively flattened)
    - dict with "url" or "urls" key
    """
    out: List[str] = []
    if x is None:
        return out
    if isinstance(x, str):
        u = x.strip()
        return [u] if u else []
    if isinstance(x, (list, tuple)):
        for it in x:
            out.extend(_normalize_urls(it))
        return out
    if isinstance(x, dict):
        if "url" in x:
            out.extend(_normalize_urls(x.get("url")))
        elif "urls" in x:
            out.extend(_normalize_urls(x.get("urls")))
        return out
    return out


def _get_messages_from_record(record: Dict[str, Any]) -> Any:
    """
    Return the message list from a record, preferring 'messages' over
    'conversations'.
    """
    messages = record.get("messages")
    if isinstance(messages, list) and len(messages) > 0:
        return messages
    conversations = record.get("conversations")
    if isinstance(conversations, list) and len(conversations) > 0:
        return conversations
    return messages if messages is not None else conversations


def extract_visited_urls_from_messages(messages: Any) -> List[Dict[str, str]]:
    """
    Extract visited URLs and their goals from a messages list.

    Supports two tool types:
    - "visit"      : extracts arguments.url  and arguments.goal
    - "crawl_page" : extracts arguments.url  and arguments.query (used as goal)

    Deduplication: if the same URL appears multiple times, keep only the last
    occurrence and its goal.  Output is ordered by last-occurrence position.

    Returns a list of dicts, each with "url" and "goal".
    """
    url_to_info: Dict[str, tuple] = {}

    if not isinstance(messages, list):
        return []

    current_index = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            continue

        for block in _extract_tool_call_blocks(content):
            payload = _parse_tool_call_payload(block)
            if not payload:
                continue
            name = str(payload.get("name") or "").strip()
            if name not in ("visit", "crawl_page"):
                continue

            args = payload.get("arguments")
            if not isinstance(args, dict):
                continue

            if name == "visit":
                goal = str(args.get("goal") or "").strip()
            else:  # crawl_page
                goal = str(args.get("query") or "").strip()

            url_field = args.get("url") if "url" in args else args.get("urls")
            urls = _normalize_urls(url_field)

            for url in urls:
                if url:
                    url_to_info[url] = (goal, current_index)
                    current_index += 1

    sorted_items = sorted(url_to_info.items(), key=lambda x: x[1][1])
    return [{"url": url, "goal": goal} for url, (goal, _) in sorted_items]


def _calculate_string_overlap(s1: str, s2: str) -> float:
    """
    Calculate the word-level overlap percentage between two strings (based on
    the shorter string), using the Longest Common Subsequence (LCS) method.

    overlap = LCS_word_count / min(word_count_1, word_count_2) * 100
    """
    if not s1 or not s2:
        return 0.0
    words1 = [w.lower().strip() for w in s1.split() if w.strip()]
    words2 = [w.lower().strip() for w in s2.split() if w.strip()]
    if not words1 or not words2:
        return 0.0

    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    min_length = min(m, n)
    return 0.0 if min_length == 0 else (lcs_length / min_length) * 100.0


def _extract_urls_from_text(text: str) -> List[str]:
    """Extract all http/https URLs from a text string."""
    if not text:
        return []
    url_pattern = re.compile(r'https?://[^\s\)\]\}\"\'<>]+')
    urls = url_pattern.findall(text)
    cleaned = []
    for url in urls:
        url = url.rstrip('.,;:!?)')
        if url:
            cleaned.append(url)
    return cleaned


def extract_related_urls_from_messages(
    messages: Any,
    visited_urls: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Extract related URLs from the user message that follows each "search"
    tool call.

    Logic:
    1. Find all "search" tool calls and collect their queries.
    2. Find the next user message (containing <tool_response>).
    3. Extract URLs from that message and associate each with the query.
    4. Exclude URLs already present in visited_urls.

    Returns a list of dicts, each with "url" and "query".
    """
    if not isinstance(messages, list):
        return []

    visited_url_set = {item.get("url", "") for item in visited_urls if item.get("url")}
    url_to_info: Dict[str, tuple] = {}
    current_index = 0

    for i in range(len(messages)):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            continue

        search_queries = []
        for block in _extract_tool_call_blocks(content):
            payload = _parse_tool_call_payload(block)
            if not payload:
                continue
            name = str(payload.get("name") or "").strip()
            if name != "search":
                continue
            args = payload.get("arguments")
            if not isinstance(args, dict):
                continue
            query_field = args.get("query")
            if isinstance(query_field, list):
                search_queries.extend([str(q).strip() for q in query_field if q])
            elif isinstance(query_field, str):
                search_queries.append(query_field.strip())

        if search_queries and i + 1 < len(messages):
            next_msg = messages[i + 1]
            if isinstance(next_msg, dict) and next_msg.get("role") == "user":
                user_content = next_msg.get("content", "")
                if isinstance(user_content, str) and user_content:
                    for query in search_queries:
                        escaped_query = re.escape(query)
                        pattern = (
                            rf"(?:A\s+Google\s+search\s+for\s+['\"]?{escaped_query}['\"]?"
                            rf"|Google\s+search\s+for\s+['\"]?{escaped_query}['\"]?"
                            rf"|search\s+for\s+['\"]?{escaped_query}['\"]?)"
                        )
                        match = re.search(pattern, user_content, re.IGNORECASE)
                        if match:
                            start_pos = match.start()
                            next_query_pattern = (
                                r"(?:A\s+Google\s+search\s+for|Google\s+search\s+for"
                                r"|search\s+for)\s+['\"]"
                            )
                            next_match = re.search(
                                next_query_pattern,
                                user_content[start_pos + len(match.group()):],
                                re.IGNORECASE,
                            )
                            if next_match:
                                section_end = start_pos + len(match.group()) + next_match.start()
                                section_content = user_content[start_pos:section_end]
                            else:
                                section_content = user_content[start_pos:]

                            for url in _extract_urls_from_text(section_content):
                                if url and url not in visited_url_set:
                                    url_to_info[url] = (query, current_index)
                                    current_index += 1

    sorted_items = sorted(url_to_info.items(), key=lambda x: x[1][1])
    return [{"url": url, "query": query} for url, (query, _) in sorted_items]


def _read_text_peek(path: Path, n: int = 4096) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read(n)


def load_records(input_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load records from a .json (top-level list) or .jsonl file.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    peek = _read_text_peek(p)
    first_nonspace = next((ch for ch in peek if not ch.isspace()), "")

    if first_nonspace == "[":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected top-level JSON list, got: {type(data)}")
        return [x for x in data if isinstance(x, dict)]

    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = safe_json_loads(s)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def write_records(records: Sequence[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".jsonl":
        with p.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with p.open("w", encoding="utf-8") as f:
            json.dump(list(records), f, ensure_ascii=False, indent=2)


def process_records(
    records: Sequence[Dict[str, Any]],
    start: int = 0,
    max_samples: Optional[int] = None,
    min_visited_urls: int = 3,
    max_qa_overlap: float = 50.0,
) -> tuple:
    """
    Add visited_urls and related_urls fields to each record.

    Args:
        records          : Input record list.
        start            : Starting index.
        max_samples      : Maximum number of records to process.
        min_visited_urls : Minimum number of visited URLs required (records
                           below this threshold are filtered out).
        max_qa_overlap   : Maximum allowed word overlap (%) between question
                           and answer; records above this threshold are filtered.

    Returns:
        (processed_records, filtered_by_visited_urls, filtered_by_overlap)
    """
    n = len(records)
    if start < 0:
        start = 0
    end = n if max_samples is None else min(n, start + max_samples)

    out: List[Dict[str, Any]] = []
    filtered_visited_count = 0
    filtered_overlap_count = 0

    for i in range(start, end):
        row = records[i]
        if not isinstance(row, dict):
            continue
        new_row = dict(row)

        messages = _get_messages_from_record(new_row)
        visited_urls = extract_visited_urls_from_messages(messages)
        new_row["visited_urls"] = visited_urls

        if len(visited_urls) < min_visited_urls:
            filtered_visited_count += 1
            continue

        question = str(new_row.get("question", "")).strip()
        answer = str(new_row.get("answer", "")).strip()
        if question and answer:
            if _calculate_string_overlap(question, answer) >= max_qa_overlap:
                filtered_overlap_count += 1
                continue

        new_row["related_urls"] = extract_related_urls_from_messages(messages, visited_urls)
        out.append(new_row)

    return out, filtered_visited_count, filtered_overlap_count


def _default_output_path(input_path: Union[str, Path]) -> Path:
    inp = Path(input_path)
    return Path("./data_synthesis/cache/cache_1") / f"{inp.stem}_step1.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 1: Extract visited_urls and related_urls from inference trajectories."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to input file (.json top-level list or .jsonl).",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Path to output file (default: data_synthesis/cache/cache_1/<input_stem>_step1.jsonl).",
    )
    ap.add_argument("--start", type=int, default=0, help="Start index (for debugging).")
    ap.add_argument("--max-samples", type=int, default=None, help="Max records to process.")
    ap.add_argument(
        "--min-visited-urls",
        type=int,
        default=1,
        help="Minimum number of visited URLs required per record (default: 1).",
    )
    ap.add_argument(
        "--max-question-answer-overlap",
        type=float,
        default=50.0,
        help="Max word-overlap %% between question and answer (default: 50.0).",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)

    records = load_records(input_path)
    processed, filtered_visited, filtered_overlap = process_records(
        records,
        start=args.start,
        max_samples=args.max_samples,
        min_visited_urls=args.min_visited_urls,
        max_qa_overlap=args.max_question_answer_overlap,
    )
    write_records(processed, output_path)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Processed: {len(processed)} / {len(records)} records")
    total_filtered = filtered_visited + filtered_overlap
    if total_filtered > 0:
        print(f"Filtered : {total_filtered} records total")
        if filtered_visited:
            print(f"  - too few visited_urls : {filtered_visited} (< {args.min_visited_urls})")
        if filtered_overlap:
            print(f"  - Q/A overlap too high : {filtered_overlap} (>= {args.max_question_answer_overlap}%)")
    if processed:
        print(f"Sample visited_urls (first 2): {processed[0].get('visited_urls', [])[:2]}")
        print(f"Sample related_urls (first 2): {processed[0].get('related_urls', [])[:2]}")


if __name__ == "__main__":
    main()
