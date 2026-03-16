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
Step 2: Broaden information by crawling visited URLs.

- Input : Records with a visited_urls field (output of step 1).
- Processing: Crawl each URL in visited_urls using the Jina API and store the
              page content in an "info" field.
- Output: Original fields + info added to each visited_urls entry,
          written to results/.

Requires environment variable:
    JINA_API_KEY  – Jina Reader API key
"""

import argparse
import asyncio
import hashlib
import json
import os
from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env (searches CWD and parent dirs)

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from asyncio import Semaphore

import aiohttp


JINA_API_URL = "https://r.jina.ai/"
JINA_API_KEY = os.getenv("JINA_API_KEY", "")


async def _jina_crawl_page_async(url: str, session: aiohttp.ClientSession) -> str:
    """Crawl a single page via the Jina Reader API (async)."""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with session.get(
            f"{JINA_API_URL}{url}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            response.raise_for_status()
            content = await response.text()
        if not content or content.strip() == "":
            return f"⚠️ Crawl error: No content extracted from {url}"
        return content
    except Exception as e:
        return f"⚠️ Crawl error: {e}"


async def _crawl_single_url(
    url: str, semaphore: Semaphore, session: aiohttp.ClientSession
) -> Dict[str, Any]:
    """Crawl one URL and return a dict with url, info, and status."""
    async with semaphore:
        try:
            info = await _jina_crawl_page_async(url, session)
            return {
                "url": url,
                "info": info,
                "status": "success" if not info.startswith("⚠️") else "failed",
            }
        except Exception as e:
            return {"url": url, "info": f"⚠️ Crawl error: {e}", "status": "failed"}


async def _batch_crawl_urls(
    urls: List[str], session: aiohttp.ClientSession, url_concurrency: int = 5
) -> List[Dict[str, Any]]:
    """Crawl a list of URLs concurrently (within-record concurrency)."""
    if not urls:
        return []
    print(f"    Crawling {len(urls)} URL(s)...")
    url_semaphore = Semaphore(url_concurrency)
    tasks = [_crawl_single_url(url, url_semaphore, session) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append({"url": urls[i], "info": f"⚠️ Crawl error: {result}", "status": "failed"})
        else:
            processed.append(result)

    successful = sum(1 for r in processed if r.get("status") == "success")
    print(f"    Done: {successful}/{len(urls)} succeeded.")
    return processed


def load_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return a list of records."""
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON line: {e}")
    return records


def append_jsonl(record: Dict[str, Any], output_path: Path) -> None:
    """Append a single record to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _get_record_key(record: Dict[str, Any]) -> str:
    """Generate a unique key for a record (used for resume logic)."""
    question = record.get("question", "")
    answer = record.get("answer", "") or record.get("golden_answer", "")
    if question:
        return question.strip()
    if answer:
        return answer.strip()
    record_str = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(record_str.encode("utf-8")).hexdigest()


def load_existing_results(output_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load already-processed records from the output file for resume support."""
    if not output_path.exists():
        return {}
    existing: Dict[str, Dict[str, Any]] = {}
    try:
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    existing[_get_record_key(rec)] = rec
                except json.JSONDecodeError:
                    continue
        print(f"Resume: loaded {len(existing)} already-processed records.")
    except Exception as e:
        print(f"Warning: failed to load existing results: {e}")
    return existing


async def process_single_record(
    record: Dict[str, Any],
    session: aiohttp.ClientSession,
    record_semaphore: Semaphore,
    url_concurrency: int = 5,
    record_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Process one record: crawl all visited_urls and add an info field."""
    async with record_semaphore:
        question = record.get("question", "N/A")[:50]
        idx_str = f"[{record_index + 1}]" if record_index is not None else ""
        print(f"  Processing record {idx_str}: {question}...")

        new_record = dict(record)
        visited_urls = record.get("visited_urls", [])
        if not visited_urls:
            print("    No visited_urls – skipping.")
            return new_record

        urls = [item.get("url", "") for item in visited_urls if item.get("url")]
        if not urls:
            print("    No valid URLs in visited_urls – skipping.")
            return new_record

        print(f"    Found {len(urls)} URL(s) to crawl.")
        crawl_results = await _batch_crawl_urls(urls, session, url_concurrency)
        url_to_info = {r["url"]: r["info"] for r in crawl_results}

        enhanced = []
        for item in visited_urls:
            url = item.get("url", "")
            enhanced_item = dict(item)
            enhanced_item["info"] = url_to_info.get(url, "⚠️ Crawl error: URL not found in results")
            enhanced.append(enhanced_item)

        new_record["visited_urls"] = enhanced
        return new_record


async def process_records(
    records: List[Dict[str, Any]],
    concurrency: int = 1,
    start: int = 0,
    max_samples: Optional[int] = None,
    url_concurrency: int = 5,
    output_path: Optional[Path] = None,
    enable_resume: bool = True,
) -> List[Dict[str, Any]]:
    """Process all records with record-level concurrency and resume support."""
    n = len(records)
    if start < 0:
        start = 0
    end = n if max_samples is None else min(n, start + max_samples)
    print(f"Processing records {start}–{end - 1} ({end - start} total)")
    print(f"Record concurrency: {concurrency}, URL concurrency per record: {url_concurrency}")

    existing_records: Dict[str, Dict[str, Any]] = {}
    if enable_resume and output_path:
        existing_records = load_existing_results(output_path)
        if existing_records:
            print(f"Resume mode: {len(existing_records)} records will be skipped.")

    record_semaphore = Semaphore(concurrency)
    file_lock = asyncio.Lock()

    async def process_and_save(
        record: Dict[str, Any], record_idx: int, session: aiohttp.ClientSession
    ) -> Optional[Dict[str, Any]]:
        record_key = _get_record_key(record)
        if enable_resume and record_key in existing_records:
            print(f"  Skipping record [{record_idx + 1}] (already processed).")
            return existing_records[record_key]
        try:
            processed = await process_single_record(
                record, session, record_semaphore, url_concurrency, record_index=record_idx
            )
            if output_path:
                async with file_lock:
                    append_jsonl(processed, output_path)
                print(f"  Saved record [{record_idx + 1}].")
            return processed
        except Exception as e:
            print(f"  ERROR processing record [{record_idx + 1}]: {e}")
            failed = dict(record)
            failed["_processing_error"] = str(e)
            if output_path:
                async with file_lock:
                    append_jsonl(failed, output_path)
            return failed

    async with aiohttp.ClientSession() as session:
        tasks = [(i, process_and_save(records[i], i, session)) for i in range(start, end)]
        print(f"\nStarting parallel processing of {len(tasks)} records...")
        results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

    processed_records = []
    for idx, (original_idx, _) in enumerate(tasks):
        result = results[idx]
        if isinstance(result, Exception):
            print(f"  ERROR: record {original_idx + 1} raised an exception: {result}")
            failed = dict(records[original_idx])
            failed["_processing_error"] = str(result)
            processed_records.append(failed)
        elif result is not None:
            processed_records.append(result)

    return processed_records


def _default_output_path(input_path: Path) -> Path:
    return Path("./data_synthesis/cache/cache_2") / f"{input_path.stem}_step2.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 2: Crawl visited URLs and add page content (info field)."
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file (output of step 1).",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Path to output JSONL file (default: results/<input_name>).",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Record-level concurrency (number of records processed simultaneously, default: 10).",
    )
    ap.add_argument(
        "--url-concurrency",
        type=int,
        default=10,
        help="URL-level concurrency within each record (default: 10).",
    )
    ap.add_argument("--start", type=int, default=0, help="Start index (for debugging).")
    ap.add_argument("--max-samples", type=int, default=None, help="Max records to process.")
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode (overwrite existing output file).",
    )
    args = ap.parse_args()

    if not JINA_API_KEY:
        print("Warning: JINA_API_KEY is not set. Crawling will likely fail.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    enable_resume = not args.no_resume

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Record concurrency: {args.parallel}, URL concurrency: {args.url_concurrency}")

    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records.")

    if not enable_resume and output_path.exists():
        print("Resume disabled – removing existing output file.")
        output_path.unlink()

    processed_records = asyncio.run(
        process_records(
            records,
            concurrency=args.parallel,
            start=args.start,
            max_samples=args.max_samples,
            url_concurrency=args.url_concurrency,
            output_path=output_path,
            enable_resume=enable_resume,
        )
    )

    print(f"\nDone. Processed {len(processed_records)} records → {output_path}")
    if processed_records:
        sample = processed_records[0].get("visited_urls", [])
        if sample:
            first = sample[0]
            print(f"Sample visited_urls[0]:")
            print(f"  url : {str(first.get('url', ''))[:80]}")
            print(f"  goal: {str(first.get('goal', ''))[:80]}")
            print(f"  info: {str(first.get('info', ''))[:100]}")


if __name__ == "__main__":
    main()
