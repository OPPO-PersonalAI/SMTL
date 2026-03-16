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
Step 3a: Build knowledge graphs (graph construction only, no data extraction).

- Input : Records with visited_urls + info fields (output of step 2).
- Processing: Use the golden_answer as a seed entity and build a knowledge
              graph for each question via LightRAG.
- Output: Graphs stored under the knowledge_graphs working directory.
          Supports resume (checks graph completeness).

Requires environment variables:
    OPENAI_API_KEY   – API key for the LLM provider
    OPENAI_BASE_URL  – Base URL for the LLM provider
    JINA_API_KEY     – Jina API key (required if reranker is enabled)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env (searches CWD and parent dirs)

import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import json_repair
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    print("Warning: json_repair not installed. LLM JSON repair will be unavailable.")
    print("         Install with: pip install json-repair")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from lightrag.rerank import jina_rerank


# ============================================================================
# Constants
# ============================================================================

# Supported entity types (8 categories)
ENTITY_TYPES = [
    "Person",        # Specific individuals
    "Organization",  # Companies / organisations
    "Event",         # Specific events
    "Location",      # Places / locations
    "Date",          # Dates / times
    "Product",       # Products / works
    "Currency",      # Monetary amounts
    "Title",         # Positions / titles / awards
]


# ============================================================================
# Utility functions
# ============================================================================

def safe_json_loads(s: str) -> Any:
    """Try to parse a JSON string; return None on failure."""
    try:
        return json.loads(s)
    except Exception:
        return None


def get_question_hash(question: str) -> str:
    """Return a short MD5 hash of the question string as a unique identifier."""
    return hashlib.md5(question.strip().encode("utf-8")).hexdigest()[:16]


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


def is_graph_complete(working_dir: Path, question_hash: str) -> bool:
    """
    Check whether the graph for a given question has been fully built.

    A complete graph must have a non-empty
    graph_chunk_entity_relation.graphml file.
    """
    graph_dir = working_dir / question_hash
    if not graph_dir.exists():
        return False
    graphml_file = graph_dir / "graph_chunk_entity_relation.graphml"
    return graphml_file.exists() and graphml_file.stat().st_size > 0


def has_partial_data(working_dir: Path, question_hash: str) -> bool:
    """
    Check whether partial LightRAG data exists for a question
    (i.e. the build was interrupted mid-way).
    """
    graph_dir = working_dir / question_hash
    if not graph_dir.exists():
        return False
    data_files = [
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
        "full_docs.json",
        "doc_status.json",
        "graph_chunk_entity_relation.graphml",
    ]
    for fname in data_files:
        fp = graph_dir / fname
        if fp.exists() and fp.stat().st_size > 0:
            return True
    return False


def cleanup_incomplete_graph(working_dir: Path, question_hash: str) -> bool:
    """Delete the graph directory for a question (used when cleanup_incomplete=True)."""
    graph_dir = working_dir / question_hash
    if not graph_dir.exists():
        return False
    try:
        shutil.rmtree(graph_dir)
        print(f"  Cleaned up incomplete graph directory: {graph_dir}")
        return True
    except Exception as e:
        print(f"  Warning: failed to clean up graph directory: {e}")
        return False


def load_complete_graphs(working_dir: Path) -> Set[str]:
    """
    Return the set of question hashes for graphs that have been fully built.
    Used for resume support.
    """
    complete_hashes: Set[str] = set()
    if not working_dir.exists():
        return complete_hashes
    for graph_dir in working_dir.iterdir():
        if graph_dir.is_dir():
            qhash = graph_dir.name
            if is_graph_complete(working_dir, qhash):
                complete_hashes.add(qhash)
    return complete_hashes


# ============================================================================
# LLM and embedding helpers
# ============================================================================

async def gpt_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "deepseek-v3.2",
    **kwargs,
) -> str:
    """Wrapper around openai_complete_if_cache using env-configured credentials."""
    return await openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs,
    )


def create_local_embedding_func(model_name: str = "BAAI/bge-m3") -> EmbeddingFunc:
    """
    Create a local sentence-transformer embedding function for LightRAG.

    Recommended models:
    - BAAI/bge-m3         : multilingual, high performance
    - all-MiniLM-L6-v2   : lightweight, English-only
    """
    print(f"Loading embedding model: {model_name} (this may take a moment)...")
    if "bge-m3" in model_name.lower():
        model = SentenceTransformer(model_name, device="cpu")
    else:
        model = SentenceTransformer(model_name)

    async def async_embed(texts: List[str]) -> np.ndarray:
        return await asyncio.to_thread(
            model.encode,
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )

    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")
    return EmbeddingFunc(embedding_dim=embedding_dim, max_token_size=8192, func=async_embed)


def create_jina_rerank_func(
    model: str = "jina-reranker-v2-base-multilingual",
    api_key: Optional[str] = None,
):
    """Create a Jina reranker function for LightRAG (improves retrieval quality)."""
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        print("Warning: JINA_API_KEY not set – reranker will be disabled.")
        return None
    print(f"Jina reranker configured: {model}")
    return partial(
        jina_rerank,
        model=model,
        api_key=api_key,
        base_url="https://api.jina.ai/v1/rerank",
    )


# ============================================================================
# GraphBuilder (build only)
# ============================================================================

class GraphBuilder:
    """Builds a LightRAG knowledge graph per question (no data extraction)."""

    def __init__(
        self,
        working_dir: str = "./data_synthesis/cache/cache_3a",
        model: str = "deepseek-v3.2",
        embedding_model: str = "BAAI/bge-m3",
        rerank_model: str = "jina-reranker-v2-base-multilingual",
        enable_rerank: bool = True,
        llm_max_async: int = 4,
        embedding_func_max_async: int = 8,
        embedding_batch_num: int = 10,
        max_parallel_insert: int = 2,
        chunk_token_size: int = 512,
        max_visited_urls: int = 10,
        min_visited_urls: int = 1,
        max_content_per_url: int = 1500,
        max_total_content: int = 15000,
        cleanup_incomplete: bool = False,
        default_llm_timeout: int = 300,
        default_embedding_timeout: int = 60,
        enable_llm_cache: bool = True,
        enable_llm_cache_for_entity_extract: bool = True,
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self._embedding_model_name = embedding_model
        self._rerank_model_name = rerank_model
        self.enable_rerank = enable_rerank
        self.max_visited_urls = max_visited_urls
        self.min_visited_urls = min_visited_urls
        self.max_content_per_url = max_content_per_url
        self.max_total_content = max_total_content
        self.cleanup_incomplete = cleanup_incomplete
        self.embedding_batch_num = embedding_batch_num
        self.default_llm_timeout = default_llm_timeout
        self.default_embedding_timeout = default_embedding_timeout
        self.enable_llm_cache = enable_llm_cache
        self.enable_llm_cache_for_entity_extract = enable_llm_cache_for_entity_extract

        print("\nConfiguring embedding model...")
        self.local_embed_func = create_local_embedding_func(embedding_model)

        print("\nConfiguring reranker...")
        if enable_rerank:
            self.rerank_func = create_jina_rerank_func(model=rerank_model)
            if self.rerank_func:
                print("Reranker enabled (recommended: use mix-mode queries).")
            else:
                print("Reranker disabled (JINA_API_KEY not configured).")
                self.enable_rerank = False
        else:
            self.rerank_func = None
            print("Reranker disabled.")

        self.rag_config: Dict[str, Any] = {
            "llm_model_max_async": llm_max_async,
            "embedding_func_max_async": embedding_func_max_async,
            "embedding_batch_num": embedding_batch_num,
            "max_parallel_insert": max_parallel_insert,
            "chunk_token_size": chunk_token_size,
            "summary_max_tokens": 1200,
            "default_llm_timeout": default_llm_timeout,
            "default_embedding_timeout": default_embedding_timeout,
            "enable_llm_cache": enable_llm_cache,
            "enable_llm_cache_for_entity_extract": enable_llm_cache_for_entity_extract,
            "addon_params": {"entity_types": ENTITY_TYPES},
        }
        if self.rerank_func:
            self.rag_config["rerank_model_func"] = self.rerank_func
            self.rag_config["min_rerank_score"] = 0.0

        self.logger = logging.getLogger(__name__)

    async def _create_rag_instance(self, question_hash: str) -> LightRAG:
        """Create a LightRAG instance with an isolated working directory per question."""
        entity_dir = str(self.working_dir / question_hash)
        os.makedirs(entity_dir, exist_ok=True)

        async def llm_func(prompt, system_prompt=None, **kwargs):
            return await gpt_complete(prompt, system_prompt, self.model, **kwargs)

        rag = LightRAG(
            working_dir=entity_dir,
            embedding_func=self.local_embed_func,
            llm_model_func=llm_func,
            **self.rag_config,
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()
        return rag

    def _organize_content_for_lightrag(
        self,
        question: str,
        golden_answer: str,
        seed_entities: List[Dict[str, str]],
        visited_urls: List[Dict[str, Any]],
    ) -> str:
        """
        Organise page content for insertion into LightRAG.

        Strategy:
        1. Limit each URL's content to max_content_per_url characters.
        2. Greedily select URLs from last to first until max_total_content is reached.
        3. Always include at least one URL even if it exceeds the budget.
        """
        content_parts = [
            "## Core Question and Answer",
            f"Question: {question}",
            f"Answer: {golden_answer}",
        ]
        if golden_answer:
            content_parts.append(f"Core entity (answer): {golden_answer}")

        url_contents = []
        for visited_url in visited_urls:
            url = visited_url.get("url", "")
            goal = visited_url.get("goal", "")
            info_str = visited_url.get("info", "")
            if not info_str:
                continue
            info = safe_json_loads(info_str) if isinstance(info_str, str) else info_str
            if not info:
                continue
            content = ""
            if isinstance(info, dict):
                data = info.get("data", {})
                if isinstance(data, dict):
                    content = data.get("content", "")
            if not content or content.startswith("⚠️"):
                continue
            if len(content) > self.max_content_per_url:
                content = content[: self.max_content_per_url]
            url_contents.append({"url": url, "goal": goal, "content": content, "char_count": len(content)})

        if not url_contents:
            print("  Warning: no valid URL content found.")
            return "\n\n".join(content_parts)

        print(
            f"  URLs available: {len(url_contents)}, "
            f"total chars: {sum(u['char_count'] for u in url_contents):,}"
        )

        # Greedy selection from last to first
        selected_urls: List[Dict[str, Any]] = []
        total_chars = 0
        for url_data in reversed(url_contents):
            if total_chars + url_data["char_count"] > self.max_total_content:
                if selected_urls:
                    break
                # Always keep at least one URL
                print(
                    f"  Single URL ({url_data['char_count']:,} chars) exceeds budget "
                    f"– keeping it anyway."
                )
                selected_urls.insert(0, url_data)
                total_chars += url_data["char_count"]
                break
            selected_urls.insert(0, url_data)
            total_chars += url_data["char_count"]

        content_parts.append("\n## Reference Sources")
        for idx, url_data in enumerate(selected_urls, 1):
            content_parts.append(f"\n### Source {idx}")
            content_parts.append(f"URL: {url_data['url']}")
            content_parts.append(f"Goal: {url_data['goal']}")
            content_parts.append(f"Content:\n{url_data['content']}")

        dropped = len(url_contents) - len(selected_urls)
        avg = total_chars // len(selected_urls) if selected_urls else 0
        print(
            f"  Selected: {len(selected_urls)}/{len(url_contents)} URLs "
            f"({dropped} dropped), total chars: {total_chars:,}/{self.max_total_content:,}, "
            f"avg per URL: {avg:,}"
        )
        return "\n\n".join(content_parts)

    async def build_graph_for_question(
        self,
        record: Dict[str, Any],
        question_hash: str,
        current_idx: int = 0,
        total_count: int = 0,
    ) -> bool:
        """
        Build a LightRAG knowledge graph for a single question.

        Returns True on success, False on failure.
        """
        question = record.get("question", "")
        golden_answer = record.get("golden_answer", "") or record.get("answer", "")
        visited_urls_raw = record.get("visited_urls", [])

        if len(visited_urls_raw) < self.min_visited_urls:
            print(
                f"  Skipping: only {len(visited_urls_raw)} visited_urls "
                f"(min required: {self.min_visited_urls})."
            )
            return False

        if len(visited_urls_raw) > self.max_visited_urls:
            visited_urls = visited_urls_raw[-self.max_visited_urls:]
            print(
                f"  Trimmed visited_urls: {len(visited_urls_raw)} → {len(visited_urls)} "
                f"(using last {self.max_visited_urls})."
            )
        else:
            visited_urls = visited_urls_raw

        pfx = f"[{current_idx}/{total_count}]" if total_count > 0 else ""
        print(f"{pfx} Question: {question[:100]}...")
        print(f"{pfx} Answer  : {golden_answer[:100] if golden_answer else '(empty)'}...")
        print(f"{pfx} URLs    : {len(visited_urls)}")

        if not golden_answer or not golden_answer.strip():
            print(f"{pfx} Skipping: golden_answer / answer field is empty.")
            return False

        seed_entities = [{"name": golden_answer.strip(), "type": "Other"}]
        print(f"{pfx} Seed entity: {golden_answer[:100]}...")

        # Resume check
        if is_graph_complete(self.working_dir, question_hash):
            print(f"{pfx} Graph already complete – skipping.")
            return True

        if has_partial_data(self.working_dir, question_hash):
            if self.cleanup_incomplete:
                print(f"{pfx} Incomplete graph detected – cleaning up and rebuilding...")
                cleanup_incomplete_graph(self.working_dir, question_hash)
            else:
                print(
                    f"{pfx} Incomplete graph detected – continuing (LightRAG will skip "
                    f"already-processed documents). Use --cleanup-incomplete to rebuild from scratch."
                )
        else:
            print(f"{pfx} Building graph...")

        rag = await self._create_rag_instance(question_hash)
        print(f"{pfx} Organising content...")
        content = self._organize_content_for_lightrag(question, golden_answer, seed_entities, visited_urls)

        print(f"  {pfx} Inserting content into LightRAG ({len(content):,} chars)...")
        try:
            await rag.ainsert(content)
            print(f"  {pfx} Content queued. Extracting entities...")

            from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock
            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_lock = get_pipeline_status_lock()
            async with pipeline_lock:
                is_busy = pipeline_status.get("busy", False)
                print(f"  {pfx} Pipeline busy: {is_busy}")

            t0 = time.time()
            await rag.apipeline_process_enqueue_documents()
            elapsed = time.time() - t0
            print(f"  {pfx} Entity extraction complete ({elapsed:.1f}s).")

            # Poll until the graph is confirmed complete (up to 5 minutes)
            max_wait = 300
            interval = 2
            last_size = 0
            no_progress = 0

            print(
                f"  {pfx} Waiting for graph to be written "
                f"(max {max_wait}s / {max_wait // 60}min)..."
            )
            for retry in range(max_wait // interval):
                await asyncio.sleep(interval)
                if is_graph_complete(self.working_dir, question_hash):
                    print(f"  {pfx} Graph confirmed complete (after {(retry + 1) * interval}s).")
                    return True

                graph_dir = self.working_dir / question_hash
                graphml = graph_dir / "graph_chunk_entity_relation.graphml"
                cur_size = graphml.stat().st_size if graphml.exists() else 0
                if cur_size > last_size:
                    no_progress = 0
                    last_size = cur_size
                else:
                    no_progress += 1

                if (retry + 1) % 10 == 0:
                    waited = (retry + 1) * interval
                    size_info = f", graphml size: {cur_size}" if cur_size > 0 else ""
                    print(f"  {pfx} Still waiting... ({waited}s elapsed{size_info})")

                if no_progress >= 30 and (retry + 1) * interval >= 120:
                    print(
                        f"  {pfx} No progress for 60s – still waiting "
                        f"(may be delayed by concurrent builds)."
                    )
                    no_progress = 0

            # Timed out
            if has_partial_data(self.working_dir, question_hash):
                print(
                    f"{pfx} Timed out after {max_wait}s – partial data found. "
                    f"Assuming build will complete (resume will verify next run)."
                )
                return True
            print(
                f"{pfx} Timed out after {max_wait}s – no partial data. "
                f"Build likely failed. Re-run to retry."
            )
            return False

        except Exception as e:
            print(f"{pfx} ERROR during content insertion / processing: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_records(
        self,
        records: List[Dict[str, Any]],
        enable_resume: bool = True,
        start: int = 0,
        max_samples: Optional[int] = None,
        parallel_questions: int = 1,
    ) -> int:
        """
        Process all records using multi-process graph building.

        Args:
            parallel_questions: Number of questions to process concurrently
                                (one process per question, default: 1 = sequential).
        """
        n = len(records)
        if start < 0:
            start = 0
        end = n if max_samples is None else min(n, start + max_samples)

        processed_hashes: Set[str] = set()
        if enable_resume:
            processed_hashes = load_complete_graphs(self.working_dir)

        processed_count = skipped_count = failed_count = 0
        overall_start = time.time()

        print(f"\n{'=' * 80}")
        print(f"Starting batch graph building (multi-process mode).")
        print(f"Range: [{start + 1}] – [{end}]  ({end - start} records)")
        print(f"Processes: {parallel_questions}")
        if enable_resume:
            print(f"Resume: enabled ({len(processed_hashes)} graphs already complete).")
        print(f"{'=' * 80}\n")

        builder_config = {
            "working_dir": str(self.working_dir),
            "model": self.model,
            "embedding_model": self._embedding_model_name,
            "rerank_model": self._rerank_model_name,
            "enable_rerank": self.enable_rerank,
            "llm_max_async": self.rag_config["llm_model_max_async"],
            "embedding_func_max_async": self.rag_config["embedding_func_max_async"],
            "embedding_batch_num": self.embedding_batch_num,
            "max_parallel_insert": self.rag_config["max_parallel_insert"],
            "chunk_token_size": self.rag_config["chunk_token_size"],
            "max_visited_urls": self.max_visited_urls,
            "min_visited_urls": self.min_visited_urls,
            "max_content_per_url": self.max_content_per_url,
            "max_total_content": self.max_total_content,
            "cleanup_incomplete": self.cleanup_incomplete,
            "default_llm_timeout": self.default_llm_timeout,
            "default_embedding_timeout": self.default_embedding_timeout,
            "enable_llm_cache": self.enable_llm_cache,
            "enable_llm_cache_for_entity_extract": self.enable_llm_cache_for_entity_extract,
        }

        records_to_process = []
        for i in range(start, end):
            rec = records[i]
            question = rec.get("question", "")
            if not question:
                continue
            qhash = get_question_hash(question)
            if enable_resume and qhash in processed_hashes:
                if is_graph_complete(self.working_dir, qhash):
                    skipped_count += 1
                    continue
            records_to_process.append((i, rec, qhash))

        print(f"To process: {len(records_to_process)} questions (skipped: {skipped_count}).")
        if not records_to_process:
            print("All questions already processed.")
            return 0

        with ProcessPoolExecutor(max_workers=parallel_questions) as executor:
            futures = []
            for i, rec, qhash in records_to_process:
                future = executor.submit(
                    _process_single_question_worker,
                    json.dumps(rec, ensure_ascii=False),
                    qhash,
                    str(self.working_dir),
                    builder_config,
                    i + 1,
                    end,
                )
                futures.append((i, future))

            results = []
            for i, future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"  [{i + 1}/{end}] Process raised an exception: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append((i, False, True))

        for result in results:
            if result is None or isinstance(result, Exception):
                failed_count += 1
                continue
            _, success, failed = result
            if failed:
                failed_count += 1
            elif success:
                processed_count += 1
            else:
                skipped_count += 1

        elapsed = time.time() - overall_start
        print(f"\n{'=' * 80}")
        print("Batch graph building complete.")
        print(f"  Built   : {processed_count}")
        print(f"  Skipped : {skipped_count}")
        print(f"  Failed  : {failed_count}")
        print(f"  Total   : {processed_count + skipped_count + failed_count}")
        print(f"  Elapsed : {elapsed:.1f}s ({elapsed / 60:.1f}min)")
        if processed_count > 0:
            avg = elapsed / processed_count
            print(f"  Avg/graph: {avg:.1f}s")
        print(f"  Graphs saved to: {self.working_dir}")
        print(f"{'=' * 80}\n")
        return processed_count


# ============================================================================
# Multi-process worker
# ============================================================================

def _process_single_question_worker(
    record_json: str,
    question_hash: str,
    working_dir: str,
    builder_config: Dict[str, Any],
    current_idx: int,
    total_count: int,
) -> tuple:
    """
    Worker function that runs in a separate process, building one graph.

    Each process gets its own pipeline_status, enabling true parallelism.
    """
    import asyncio as _asyncio
    import json as _json

    record = _json.loads(record_json)
    builder = GraphBuilder(**builder_config)
    try:
        success = _asyncio.run(
            builder.build_graph_for_question(
                record, question_hash, current_idx=current_idx, total_count=total_count
            )
        )
        return (current_idx - 1, True, False) if success else (current_idx - 1, False, False)
    except Exception as e:
        print(f"  [{current_idx}/{total_count}] Worker error: {e}")
        import traceback
        traceback.print_exc()
        return (current_idx - 1, False, True)


# ============================================================================
# Entry point
# ============================================================================

def main_sync(args) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Input         : {input_path}")
    print(f"Model         : {args.model}")
    print(f"Embedding     : {args.embedding_model}")
    print(f"Working dir   : {args.working_dir}")
    print(f"Resume        : {'disabled' if args.no_resume else 'enabled'}")
    print(f"Cleanup incpl.: {args.cleanup_incomplete}")
    print(f"\nConcurrency:")
    print(f"  Processes (questions) : {args.parallel_questions}")
    print(f"  LLM async per process : {args.llm_max_async}")
    print(f"  Embed async per proc  : {args.embedding_func_max_async}")
    print(f"  Insert async per proc : {args.max_parallel_insert}")
    print(f"\nEffective totals:")
    print(f"  Total LLM async  : {args.parallel_questions * args.llm_max_async}")
    print(f"  Total embed async: {args.parallel_questions * args.embedding_func_max_async}")
    print(f"  Total insert     : {args.parallel_questions * args.max_parallel_insert}")

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set.")
        return

    records = load_jsonl(input_path)
    print(f"\nLoaded {len(records)} records.")

    builder = GraphBuilder(
        working_dir=args.working_dir,
        model=args.model,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
        enable_rerank=not args.disable_rerank,
        llm_max_async=args.llm_max_async,
        embedding_func_max_async=args.embedding_func_max_async,
        embedding_batch_num=args.embedding_batch_num,
        max_parallel_insert=args.max_parallel_insert,
        chunk_token_size=args.chunk_token_size,
        max_visited_urls=args.max_visited_urls,
        min_visited_urls=args.min_visited_urls,
        max_content_per_url=args.max_content_per_url,
        max_total_content=args.max_total_content,
        cleanup_incomplete=args.cleanup_incomplete,
        default_llm_timeout=args.default_llm_timeout,
        default_embedding_timeout=args.default_embedding_timeout,
        enable_llm_cache=not args.disable_llm_cache,
        enable_llm_cache_for_entity_extract=not args.disable_entity_extract_cache,
    )

    builder.process_records(
        records,
        enable_resume=not args.no_resume,
        start=args.start,
        max_samples=args.max_samples,
        parallel_questions=args.parallel_questions,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 3a: Build LightRAG knowledge graphs (one per question)."
    )
    ap.add_argument("--input", required=True, help="Path to input JSONL file (output of step 2).")
    ap.add_argument(
        "--working-dir",
        default="./data_synthesis/cache/cache_3a",
        help="LightRAG working directory (default: ./data_synthesis/cache/cache_3a).",
    )
    ap.add_argument("--model", default="deepseek-v3.2", help="LLM model name.")
    ap.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Local sentence-transformer embedding model.",
    )
    ap.add_argument(
        "--rerank-model",
        default="jina-reranker-v3",
        help="Jina reranker model name.",
    )
    ap.add_argument("--disable-rerank", action="store_true", help="Disable the reranker.")
    ap.add_argument("--llm-max-async", type=int, default=4, help="LLM async concurrency per process.")
    ap.add_argument("--embedding-func-max-async", type=int, default=10, help="Embedding async concurrency per process.")
    ap.add_argument("--embedding-batch-num", type=int, default=10, help="Embedding batch size.")
    ap.add_argument("--max-parallel-insert", type=int, default=1, help="Parallel inserts per process.")
    ap.add_argument("--chunk-token-size", type=int, default=8192, help="LightRAG chunk token size.")
    ap.add_argument("--max-visited-urls", type=int, default=5, help="Max visited URLs per record (uses last N).")
    ap.add_argument("--min-visited-urls", type=int, default=5, help="Min visited URLs required; records below this are skipped.")
    ap.add_argument("--max-content-per-url", type=int, default=30000, help="Max chars per URL content.")
    ap.add_argument("--max-total-content", type=int, default=150000, help="Max total chars across all URLs per record.")
    ap.add_argument(
        "--parallel-questions",
        type=int,
        default=1,
        help="Number of questions to process concurrently (default: 1 = sequential).",
    )
    ap.add_argument("--start", type=int, default=0, help="Start index.")
    ap.add_argument("--max-samples", type=int, default=1000000, help="Max records to process.")
    ap.add_argument("--no-resume", action="store_true", help="Disable resume mode.")
    ap.add_argument(
        "--cleanup-incomplete",
        default=True,
        help="Delete and rebuild incomplete graphs (default: True).",
    )
    ap.add_argument("--default-llm-timeout", type=int, default=300, help="LLM timeout in seconds.")
    ap.add_argument("--default-embedding-timeout", type=int, default=60, help="Embedding timeout in seconds.")
    ap.add_argument("--disable-llm-cache", action="store_true", help="Disable LLM response cache.")
    ap.add_argument("--disable-entity-extract-cache", action="store_true", help="Disable entity extraction cache.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main_sync(args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
