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
Step 3b: Extract data from pre-built knowledge graphs.

- Input : Records with visited_urls + info (output of step 2).
- Processing: Load the LightRAG graphs built in step 3a and extract entities,
              relations, and LLM-enriched fields.
- Output: JSONL file containing entities, relations, graph statistics, and
          extended fields.
- Prerequisite: Graphs must already be built under the knowledge_graphs
                directory (run step 3a first).

Requires environment variables:
    OPENAI_API_KEY   – API key for the LLM provider
    OPENAI_BASE_URL  – Base URL of the LLM provider
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
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    import json_repair
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    print("Warning: json_repair not installed – LLM JSON repair unavailable.")
    print("         Install with: pip install json-repair")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LightRAG imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, GPTKeywordExtractionFormat
from lightrag.kg.shared_storage import initialize_pipeline_status, get_data_init_lock
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

# Pre-defined core relation types
CORE_RELATIONS = {
    # Person-related
    "worked_at", "born_in", "died_in", "educated_at", "created",
    "awarded", "member_of",
    # Organization-related
    "located_in", "founded_by", "owns", "partner_with",
    # Event-related
    "occurred_in", "participated_by", "caused_by",
    # Product-related
    "created_by", "published_by", "released_in",
    # Generic
    "related_to", "part_of", "mentioned_in",
}


# ============================================================================
# Utility functions
# ============================================================================

def safe_json_loads(s: str) -> Any:
    """Try to parse a JSON string; return None on failure."""
    try:
        return json.loads(s)
    except Exception:
        return None


def safe_json_loads_with_repair(s: str, use_repair: bool = True) -> Any:
    """Parse JSON with optional json_repair fallback (for LLM output).

    Args:
        s          : JSON string to parse.
        use_repair : If True, attempt json_repair before giving up.

    Returns:
        Parsed Python object, or None on failure.
    """
    if not s or not isinstance(s, str):
        return None

    # Direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Try json_repair
    if use_repair and JSON_REPAIR_AVAILABLE:
        try:
            return json.loads(json_repair.repair_json(s))
        except Exception:
            pass

    # Strip markdown code fences and retry
    try:
        cleaned = re.sub(r'^```(?:json)?\s*', '', s.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if use_repair and JSON_REPAIR_AVAILABLE:
                return json.loads(json_repair.repair_json(cleaned))
    except Exception:
        pass

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


def append_jsonl(record: Dict[str, Any], output_path: Path) -> None:
    """Append a single record to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_graph_complete(working_dir: Path, question_hash: str) -> bool:
    """Return True if the graph for the given question has been fully built."""
    graph_dir = working_dir / question_hash
    if not graph_dir.exists():
        return False
    graphml_file = graph_dir / "graph_chunk_entity_relation.graphml"
    return graphml_file.exists() and graphml_file.stat().st_size > 0


def load_existing_results(output_path: Path) -> Set[str]:
    """
    Load question hashes of already-processed records from the output file.
    Only records with status="completed" are considered valid.
    """
    processed_hashes: Set[str] = set()
    if output_path.exists():
        try:
            for record in load_jsonl(output_path):
                question = record.get("question", "")
                if question and record.get("status") == "completed":
                    processed_hashes.add(get_question_hash(question))
        except Exception as e:
            print(f"Warning: failed to load existing results: {e}")
    print(f"Resume: loaded {len(processed_hashes)} already-processed records.")
    return processed_hashes


def remove_records_from_output(output_path: Path, hashes_to_remove: Set[str]) -> int:
    """Remove records matching the given question hashes from the output file."""
    if not output_path.exists():
        return 0
    try:
        records = load_jsonl(output_path)
        kept = [
            r for r in records
            if get_question_hash(r.get("question", "")) not in hashes_to_remove
        ]
        removed = len(records) - len(kept)
        if removed > 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Removed {removed} records from output file.")
        return removed
    except Exception as e:
        print(f"Warning: failed to remove records: {e}")
        return 0


# ============================================================================
# LLM / embedding helpers
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
    """Create a local sentence-transformer embedding function for LightRAG."""
    print(f"Loading embedding model: {model_name}...")
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


# ============================================================================
# Reranker
# ============================================================================

def create_jina_rerank_func(
    model: str = "jina-reranker-v2-base-multilingual",
    api_key: Optional[str] = None,
):
    """Create a Jina reranker function for LightRAG."""
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        print("Warning: JINA_API_KEY not set – reranker disabled.")
        return None
    print(f"Jina reranker configured: {model}")
    return partial(
        jina_rerank,
        model=model,
        api_key=api_key,
        base_url="https://api.jina.ai/v1/rerank",
    )


# ============================================================================
# Seed entity extraction
# ============================================================================

async def extract_seed_entities(
    golden_answer: str,
    client: AsyncOpenAI,
    model: str = "deepseek-v3.2",
) -> List[Dict[str, str]]:
    """Extract 1-3 core seed entities from the golden_answer."""
    prompt = f"""Extract 1-3 core entities from the following answer to use as seed nodes in a knowledge graph.

Answer: {golden_answer}

Requirements:
1. Extract only the most essential entities.
2. Entities must be concrete and verifiable.
3. Return a JSON list.

Allowed entity types: Person, Organization, Event, Location, Date, Product, Currency, Title

Output format:
[
  {{"name": "entity name", "type": "entity type"}},
  ...
]
"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional knowledge graph construction assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        result_text = response.choices[0].message.content.strip()
        entities = safe_json_loads_with_repair(result_text, use_repair=True)
        if not isinstance(entities, list):
            print(f"Warning: seed entity extraction returned unexpected type {type(entities)}. Raw: {result_text[:200]}...")
            return []
        return [
            e for e in entities
            if isinstance(e, dict) and "name" in e and e.get("type") in ENTITY_TYPES
        ]
    except Exception as e:
        print(f"ERROR: seed entity extraction failed: {e}")
        return []


# ============================================================================
# Rule-based Field Extraction
# ============================================================================

def extract_rule_based_fields(
    entity_name: str,
    visited_urls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Rule-based extraction: source_urls, discovery_queries, source_spans, retrieval_trace
    
    Returned fields are automatically deduplicated.
    """
    source_urls = []
    discovery_queries = []
    source_spans = []
    
    # For deduplication
    seen_urls = set()
    seen_spans = set()
    
    for visited_url in visited_urls:
        info_str = visited_url.get("info", "")
        if not info_str:
            continue
        
        # Parse info (may be a JSON string)
        info = safe_json_loads(info_str) if isinstance(info_str, str) else info_str
        if not info:
            continue
        
        # Extract content
        content = ""
        if isinstance(info, dict):
            data = info.get("data", {})
            if isinstance(data, dict):
                content = data.get("content", "")
        
        if not content:
            continue
        
        # Check if entity appears in content (case-insensitive)
        content_lower = content.lower()
        entity_name_lower = entity_name.lower()
        
        if entity_name_lower in content_lower:
            url = visited_url.get("url", "")
            goal = visited_url.get("goal", "")
            
            # URL deduplication
            if url and url not in seen_urls:
                source_urls.append(url)
                discovery_queries.append(goal)
                seen_urls.add(url)
            
            # Extract all text spans containing the entity (±512 chars)
            # Find all occurrence positions (case-insensitive)
            start_pos = 0
            found_positions = []
            
            while True:
                idx = content_lower.find(entity_name_lower, start_pos)
                if idx == -1:
                    break
                found_positions.append(idx)
                start_pos = idx + 1
            
            # Extract text span for each occurrence position
            for idx in found_positions:
                start = max(0, idx - 512)
                end = min(len(content), idx + len(entity_name) + 512)
                span = content[start:end].strip()
                
                # Span deduplication (use first 512 chars as signature)
                span_signature = span[:512].lower() if len(span) > 512 else span.lower()
                if span_signature not in seen_spans:
                    source_spans.append(span)
                    seen_spans.add(span_signature)
    
    return {
        "source_urls": source_urls,
        "discovery_queries": discovery_queries,
        "source_spans": source_spans,
    }


# ============================================================================
# LLM-based Field Extraction
# ============================================================================

async def extract_llm_based_fields(
    entity_name: str,
    entity_type: str,
    description: str,
    source_content: str,
    client: AsyncOpenAI,
    model: str = "deepseek-v3.2",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """LLM extraction: key_attributes, surface_forms, aliases
    
    Returns:
        Tuple[extraction_result, status_info]
        extraction_result: {"key_attributes": {}, "surface_forms": [], "aliases": []}
        status_info: {"status": "success|error|empty", "message": "details"}
    """
    
    # Attribute templates customized per entity type (~10 attributes each)
    attribute_templates = {
        "Person": "birth_year, death_year, nationality, occupation, education, awards, works, family, achievements, residenc, others",
        "Organization": "founded_year, location, industry, size, employees, revenue, products, services, leadership, partnerships, others",
        "Event": "start_date, end_date, location, participants, organizers, impact, cause, outcome, duration, significance, others",
        "Location": "country, region, type, coordinates, population, area, climate, landmarks, economy, history, others",
        "Date": "year, month, day, era, significance, events, context, calendar, timezone, related_periods, others",
        "Product": "release_year, creator, type, category, price, features, specifications, reviews, sales, competitors, others",
        "Currency": "amount, currency_code, year, exchange_rate, country, denomination, value, inflation, market, trends, others",
        "Title": "position, organization, period, responsibilities, requirements, holder, history, significance, hierarchy, related_titles, others",
    }
    
    template = attribute_templates.get(entity_type, "relevant_attributes")
    
    # Check if source_content is empty
    if not source_content or not source_content.strip():
        status_info = {
            "status": "empty",
            "message": "source_content is empty, cannot extract information"
        }
        return {"key_attributes": {}, "surface_forms": [], "aliases": []}, status_info
    
    prompt = f"""Given entity information:
- Name: {entity_name}
- Type: {entity_type}
- Description: {description}

Source content excerpt:
{source_content[:2000]}

Extract the following information (JSON format):
{{
  "key_attributes": {{
    // Extract relevant attributes based on entity type ({template})
    // Only include information explicitly present in the text; do not infer.
  }},
  "surface_forms": [
    // Common ways this entity is mentioned in the text (abbreviations, short forms)
  ],
  "aliases": [
    // Fully equivalent alternative names (full name, pen name, former name)
  ]
}}

Requirements:
1. Only extract information explicitly present in the text.
2. Do not speculate or fabricate.
3. Return empty object / empty list if a field cannot be extracted.
"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional entity information extraction assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Use json_repair to fix and parse JSON (LLM output may be incomplete)
        result = safe_json_loads_with_repair(result_text, use_repair=True)
        
        if not isinstance(result, dict):
            status_info = {
                "status": "error",
                "message": f"LLM returned invalid format, type: {type(result)}, raw output: {result_text[:200]}..."
            }
            return {"key_attributes": {}, "surface_forms": [], "aliases": []}, status_info
        
        # Extract fields and deduplicate
        key_attributes = result.get("key_attributes", {})
        surface_forms_raw = result.get("surface_forms", [])
        aliases_raw = result.get("aliases", [])
        
        # Deduplicate list fields (preserve order)
        surface_forms = []
        seen_surface = set()
        for item in surface_forms_raw:
            if isinstance(item, str):
                item_clean = item.strip()
                if item_clean and item_clean.lower() not in seen_surface:
                    surface_forms.append(item_clean)
                    seen_surface.add(item_clean.lower())
        
        aliases = []
        seen_aliases = set()
        for item in aliases_raw:
            if isinstance(item, str):
                item_clean = item.strip()
                if item_clean and item_clean.lower() not in seen_aliases:
                    aliases.append(item_clean)
                    seen_aliases.add(item_clean.lower())
        
        # Check if result is empty
        has_key_attributes = bool(key_attributes)
        has_surface_forms = len(surface_forms) > 0
        has_aliases = len(aliases) > 0
        
        if not (has_key_attributes or has_surface_forms or has_aliases):
            status_info = {
                "status": "empty",
                "message": f"All fields are empty (key_attributes: {len(key_attributes)}, surface_forms: {len(surface_forms)}, aliases: {len(aliases)}), entity may not be found in text"
            }
        else:
            status_info = {
                "status": "success",
                "message": f"Extraction successful (key_attributes: {len(key_attributes)}, surface_forms: {len(surface_forms)}, aliases: {len(aliases)})"
            }
        
        return {
            "key_attributes": key_attributes,
            "surface_forms": surface_forms,
            "aliases": aliases,
        }, status_info
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        status_info = {
            "status": "error",
            "message": f"LLM call failed: {str(e)}, details: {error_detail[:300]}"
        }
        return {"key_attributes": {}, "surface_forms": [], "aliases": []}, status_info


# ============================================================================
# Relation Normalization
# ============================================================================

def normalize_relation_type(relation_keyword: str) -> str:
    """Normalize relation keywords to predefined relation types."""
    keyword_lower = relation_keyword.lower().strip()
    
    # Direct match
    if keyword_lower in CORE_RELATIONS:
        return keyword_lower
    
    # Synonym mapping
    synonym_map = {
        "work": "worked_at",
        "employment": "worked_at",
        "employed": "worked_at",
        "birth": "born_in",
        "birthplace": "born_in",
        "death": "died_in",
        "died": "died_in",
        "education": "educated_at",
        "studied": "educated_at",
        "author": "created_by",
        "wrote": "created",
        "published": "published_by",
        "release": "released_in",
        "award": "awarded",
        "prize": "awarded",
        "location": "located_in",
        "based": "located_in",
        "founded": "founded_by",
        "established": "founded_by",
        "happen": "occurred_in",
        "took_place": "occurred_in",
        "participate": "participated_by",
        "involve": "participated_by",
    }
    
    # Try synonym matching
    for key, core_relation in synonym_map.items():
        if key in keyword_lower:
            return core_relation
    
    # If no match, return original (preserve verb-phrase form)
    return keyword_lower.replace(" ", "_")


# ============================================================================
# Entity Validation
# ============================================================================

def verify_entities_in_content(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    content: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Verify that entities appear in the input content.
    
    Requirement: entity canonical_name must appear as a continuous string in
    the input content (case-insensitive), to ensure all entities are extracted
    from the content rather than inferred by the LLM.
    
    Args:
        entities: List of entities
        relations: List of relations
        content: Raw input content (for validation)
    
    Returns:
        Tuple[validated_entities, validated_relations]
    """
    if not content:
        return [], []
    
    content_lower = content.lower()
    
    # Validate entities: check if entity name appears in content (case-insensitive, must be continuous)
    valid_entities = []
    entity_name_to_entity = {}
    
    for entity in entities:
        entity_name = entity.get("canonical_name", "").strip()
        if not entity_name:
            continue
        
        # Check if entity name appears in content (must appear as continuous string)
        entity_name_lower = entity_name.lower()
        
        # Direct string match (case-insensitive)
        # Requires the entity name to appear as a complete string in the content
        if entity_name_lower in content_lower:
            valid_entities.append(entity)
            entity_name_to_entity[entity_name] = entity
    
    print(f"    🔍 Entity validation: {len(entities)} -> {len(valid_entities)} (filtered {len(entities) - len(valid_entities)} entities not found in content)")
    
    # Validate relations: only keep relations where both entities appear in content
    valid_entity_names = set(entity_name_to_entity.keys())
    valid_relations = []
    
    for relation in relations:
        source_entity = relation.get("source_entity", "").strip()
        target_entity = relation.get("target_entity", "").strip()
        
        # Check if both source and target entities are in the valid entity list
        if source_entity in valid_entity_names and target_entity in valid_entity_names:
            valid_relations.append(relation)
    
    print(f"    🔍 Relation validation: {len(relations)} -> {len(valid_relations)} (filtered {len(relations) - len(valid_relations)} invalid relations)")
    
    return valid_entities, valid_relations


# ============================================================================
# String Utility Functions
# ============================================================================

def contains_chinese(text: str) -> bool:
    """Check if a string contains Chinese characters."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def is_single_word_or_char(text: str) -> bool:
    """Check if text is a single word (English) or a single character (Chinese).
    
    Args:
        text: Text to check
    
    Returns:
        True if it is a single English word or single Chinese character, False otherwise.
    """
    text = text.strip()
    if not text:
        return True
    
    # Check if text contains Chinese
    if contains_chinese(text):
        # Chinese: filter if only 1 character
        return len(text) == 1
    else:
        # English: split by space, filter if only 1 word
        words = text.split()
        return len(words) == 1


def calculate_string_similarity(s1: str, s2: str) -> float:
    """Calculate the similarity percentage between two strings (based on the shorter one).
    Splits strings into words by space, then computes word-level similarity using LCS.
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Similarity percentage (0-100)
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
    max_length = min(m, n)
    
    if max_length == 0:
        return 0.0
    
    return (lcs_length / max_length) * 100.0


# ============================================================================
# Graph Structure Statistics
# ============================================================================

def calculate_depth_from_seeds(
    G: nx.DiGraph,
    seed_nodes: List[str],
) -> Dict[str, int]:
    """Calculate the shortest distance from each node to the nearest seed node."""
    depths = {}
    
    # Optimization: convert to undirected graph only once
    G_undirected = G.to_undirected()
    
    # Optimization: pre-compute shortest path trees for all seed nodes
    seed_paths = {}
    for seed in seed_nodes:
        if seed in G.nodes():
            try:
                seed_paths[seed] = nx.single_source_shortest_path_length(G_undirected, seed)
            except:
                seed_paths[seed] = {}
    
    # For each node, find the shortest distance from any seed node
    for node in G.nodes():
        min_depth = float('inf')
        for seed, paths in seed_paths.items():
            if node in paths:
                min_depth = min(min_depth, paths[node])
        
        depths[node] = min_depth if min_depth != float('inf') else -1
    
    return depths


def calculate_graph_metrics(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    seed_entity_ids: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Calculate graph structure statistics using NetworkX."""
    
    # Build directed graph
    G = nx.DiGraph()
    
    # Add nodes
    entity_id_map = {}
    for entity in entities:
        entity_id = entity["entity_id"]
        entity_id_map[entity["canonical_name"]] = entity_id
        G.add_node(entity_id, name=entity["canonical_name"], type=entity["entity_type"])
    
    # Add edges
    for relation in relations:
        source_name = relation.get("source_entity", "")
        target_name = relation.get("target_entity", "")
        
        source_id = entity_id_map.get(source_name)
        target_id = entity_id_map.get(target_name)
        
        if source_id and target_id:
            G.add_edge(source_id, target_id, type=relation.get("relationship_type", ""))
    
    if G.number_of_nodes() == 0:
        return {}, {
            "num_entities": 0,
            "num_relations": 0,
            "num_components": 0,
            "has_cycles": False,
            "cycle_count": 0,
            "avg_degree": 0.0,
            "graph_density": 0.0,
            "max_depth": 0,
        }
    
    # Compute node metrics
    node_metrics = {}
    
    # PageRank (limit iterations to speed up)
    try:
        max_iter = 50  # Limit max iterations (default is 100)
        pagerank = nx.pagerank(G, max_iter=max_iter)
    except:
        pagerank = {node: 0.0 for node in G.nodes()}
    
    # Betweenness (use sampling to speed up for large graphs)
    try:
        sample_size = min(100, len(G))  # Sample at most 100 nodes
        betweenness = nx.betweenness_centrality(G, k=sample_size)
    except:
        betweenness = {node: 0.0 for node in G.nodes()}
    
    # Articulation points (cut vertices)
    try:
        G_undirected = G.to_undirected()
        articulation_points = set(nx.articulation_points(G_undirected))
    except:
        articulation_points = set()
    
    # Connected components
    try:
        weakly_connected = list(nx.weakly_connected_components(G))
        component_map = {}
        for comp_id, component in enumerate(weakly_connected):
            for node in component:
                component_map[node] = comp_id
    except:
        component_map = {node: 0 for node in G.nodes()}
    
    # Depth from seeds
    depths = calculate_depth_from_seeds(G, seed_entity_ids)
    
    # Assemble node metrics
    for node in G.nodes():
        node_metrics[node] = {
            "degree": G.degree(node),
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
            "pagerank": float(pagerank.get(node, 0.0)),
            "betweenness": float(betweenness.get(node, 0.0)),
            "is_cut_vertex": node in articulation_points,
            "component_id": component_map.get(node, 0),
            "depth_from_seed": depths.get(node, -1),
            "is_seed": node in seed_entity_ids,
        }
    
    # Compute cycles (using undirected graph to ignore direction)
    try:
        # Method 1: directed cycles (strict)
        # Note: simple_cycles can be very slow for large graphs (exponential complexity)
        # For graphs with >50 nodes, use fast check instead of full enumeration
        if len(G) <= 50:
            directed_cycles = list(nx.simple_cycles(G))
            directed_cycle_count = len(directed_cycles)
        else:
            # For large graphs, only check existence, do not enumerate
            try:
                # Use strongly connected components to quickly check for cycles
                scc = list(nx.strongly_connected_components(G))
                has_directed_cycles = any(len(comp) > 1 for comp in scc)
                directed_cycle_count = 1 if has_directed_cycles else 0
            except:
                directed_cycle_count = 0
        
        # Method 2: undirected cycles (direction-agnostic)
        G_undirected = G.to_undirected()
        
        # Use cycle_basis to find all fundamental cycles
        undirected_cycles = nx.cycle_basis(G_undirected)
        undirected_cycle_count = len(undirected_cycles)
        
        # Use undirected cycle count as primary metric
        cycle_count = undirected_cycle_count
        has_cycles = cycle_count > 0
        
        print(f"    Cycle stats: directed={directed_cycle_count}, undirected={undirected_cycle_count}")
    except Exception as e:
        print(f"    ⚠️ Cycle computation failed: {e}")
        cycle_count = 0
        has_cycles = False
        directed_cycle_count = 0
        undirected_cycle_count = 0
    
    # Graph-level statistics
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0.0
    
    graph_statistics = {
        "num_entities": G.number_of_nodes(),
        "num_relations": G.number_of_edges(),
        "num_components": len(weakly_connected) if 'weakly_connected' in locals() else 0,
        "has_cycles": has_cycles,
        "cycle_count": cycle_count,  # undirected cycle count
        "directed_cycle_count": directed_cycle_count,  # additionally record directed cycle count
        "undirected_cycle_count": undirected_cycle_count,  # additionally record undirected cycle count
        "avg_degree": float(avg_degree),
        "graph_density": float(nx.density(G)),
        "max_depth": max([m["depth_from_seed"] for m in node_metrics.values()]) if node_metrics else 0,
    }
    
    return node_metrics, graph_statistics


# ============================================================================
# Multiprocessing Worker Function (module-level for pickling)
# ============================================================================

def _process_single_record_worker(
    args_tuple: Tuple[int, Dict[str, Any], str, str, int, int, str, bool, int, Set[str]]
) -> Tuple[int, bool, bool]:
    """Multiprocessing worker: processes a single record.
    
    Args:
        args_tuple: (index, record, working_dir, model, top_entities, llm_max_async,
                     output_path_str, enable_resume, end, processed_hashes)
        processed_hashes: set of already-processed question hashes (pre-loaded in the
                          main process to avoid re-reading the file for every task)
    
    Returns:
        (index, success, failed)
    
    Note: In multiprocessing mode each worker has its own event loop and memory space,
          avoiding lock conflicts in the lightrag library.
    """
    i, record, working_dir, model, top_entities, llm_max_async, output_path_str, enable_resume, end, processed_hashes = args_tuple
    output_path = Path(output_path_str)
    
    question = record.get("question", "")
    
    if not question:
        print(f"\n[{i+1}/{end}] ⚠️ Record missing 'question' field, skipping")
        return (i, False, False)
    
    question_hash = get_question_hash(question)
    
    # Check if already processed (use pre-loaded hash set to avoid reading file)
    if enable_resume:
        if question_hash in processed_hashes:
            print(f"\n[{i+1}/{end}] ⏭️  Already processed (result exists), skipping")
            return (i, False, False)
    
        print(f"\n{'='*80}")
        print(f"[{i+1}/{end}] 🚀 Starting data extraction")
        print(f"[{i+1}/{end}] 📝 Question: {question[:100]}...")
        print(f"[{i+1}/{end}] 🔷 Process ID: {os.getpid()}")
        print(f"{'='*80}")
    
    import time
    question_start_time = time.time()
    
    try:
        # Create an independent builder instance in each process
        builder = QuestionCentricGraphBuilder(
            working_dir=working_dir,
            model=model,
            llm_max_async=llm_max_async,
            top_entities=top_entities,
        )
        
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        try:
            result = new_loop.run_until_complete(
                builder.extract_data_from_graph(
                    record,
                    question_hash,
                    current_idx=i+1,
                    total_count=end
                )
            )
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)

        # Check if result is valid
        if result is None:
            print(f"  [{i+1}/{end}] ⏭️  Data extraction failed, skipping save")
            return (i, False, False)
        
        # Only save results with status == "completed"
        if result.get("status") != "completed":
            print(f"  [{i+1}/{end}] ⏭️  Status is '{result.get('status')}', skipping save")
            return (i, False, False)
        
        # Compute elapsed time
        question_elapsed_time = time.time() - question_start_time
        
        # Save result (use file lock to protect concurrent writes on Mac/Linux)
        import fcntl
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        print(f"  [{i+1}/{end}] ✅ Data extraction succeeded and saved | ⏱️ {question_elapsed_time:.1f}s | 📊 {result['graph_statistics']['num_entities']} entities / {result['graph_statistics']['num_relations']} relations")
        
        return (i, True, False)  # success
    
    except Exception as e:
        print(f"  [{i+1}/{end}] ❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return (i, False, True)  # failure


# ============================================================================
# Knowledge Graph Builder
# ============================================================================

class QuestionCentricGraphBuilder:
    """Question-centric knowledge graph builder."""
    
    def __init__(
        self,
        working_dir: str = "./data_synthesis/cache/cache_3a",
        model: str = "deepseek-v3.2",
        llm_max_async: int = 4,
        top_entities: int = 20,  # Only run LLM extraction on the top-N most important entities
    ):
        """
        Args:
            working_dir: LightRAG working directory (knowledge_graphs); graphs must have been
                         built in step1 before calling step2.
            model: LLM model (used for entity field extraction)
            llm_max_async: Maximum concurrent LLM calls per question
            top_entities: Only run LLM extraction on the top-N most important entities
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.top_entities = top_entities  # Only run LLM extraction on top-N entities
        
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )
        
        # LightRAG config (read-only mode — graphs were built in step1)
        self.rag_config = {
            "llm_model_max_async": llm_max_async,
            "summary_max_tokens": 1200,
            # Restrict entity types to precise categories, excluding ambiguous ones
            "addon_params": {
                "entity_types": ENTITY_TYPES,  # Person, Organization, Event, Location, Date, Product, Currency, Title
            },
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Note: in multiprocessing mode file writes are protected by fcntl file locks
    
    async def _create_rag_instance(self, question_hash: str) -> LightRAG:
        """Create a LightRAG instance for a question (read-only mode).
        
        Note: graphs are already built in step1; here we only read the data and
        do not need to configure an embedding model or reranker.
        
        Key fix: in multiprocessing mode each process has its own event loop and
        memory space, avoiding lock conflicts in the lightrag library.
        """
        entity_dir = str(self.working_dir / question_hash)
        os.makedirs(entity_dir, exist_ok=True)
        
        # Wrap LLM function (LightRAG initialization may require it even though
        # we won't actually use it for graph construction here)
        async def llm_func(prompt, system_prompt=None, **kwargs):
            return await gpt_complete(prompt, system_prompt, self.model, **kwargs)
        
        # Create a minimal dummy embedding function (only for initialization;
        # will not be called during data reading)
        async def dummy_embed(texts: List[str]) -> np.ndarray:
            return np.zeros((len(texts), 384))
        
        dummy_embed_func = EmbeddingFunc(
            embedding_dim=384,
            max_token_size=512,
            func=dummy_embed
        )
        
        rag = LightRAG(
            working_dir=entity_dir,
            embedding_func=dummy_embed_func,  # dummy — not actually used
            llm_model_func=llm_func,
            **self.rag_config,
        )
        
        # Initialize storages (read-only; pipeline status init not needed for step2)
        await rag.initialize_storages()
        # await initialize_pipeline_status()  # intentionally omitted for step2
        
        return rag
    
    def _organize_content_for_validation(
        self,
        question: str,
        golden_answer: str,
        seed_entities: List[Dict[str, str]],
        visited_urls: List[Dict[str, Any]],
    ) -> str:
        """Organize content for entity validation (simplified version).
        
        Note: In step2 this is only used to verify that entities appear in the
        input content; no complex URL-selection logic is needed.
        """
        content_parts = []
        
        # 1. Core context
        content_parts.append("## Core Question and Answer")
        content_parts.append(f"Question: {question}")
        content_parts.append(f"Answer: {golden_answer}")
        
        # Use golden_answer as core entity
        if golden_answer:
            content_parts.append(f"Core entity (answer): {golden_answer}")
        
        # 2. Parse all visited URL contents (for entity validation)
        for visited_url in visited_urls:
            url = visited_url.get("url", "")
            goal = visited_url.get("goal", "")
            info_str = visited_url.get("info", "")
            
            if not info_str:
                continue
            
            # Parse info
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
            
            # Append to content (for validation)
            content_parts.append("\n## Source")
            content_parts.append(f"URL: {url}")
            content_parts.append(f"Goal: {goal}")
            content_parts.append(f"Content:\n{content}")
        
        final_content = "\n\n".join(content_parts)
        
        return final_content
    
    async def _extract_entities_and_relations_from_rag(
        self,
        rag: LightRAG,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract entities and relations from LightRAG."""
        entities = []
        relations = []
        
        try:
            # Method 1: get nodes and edges from chunk_entity_relation_graph
            if hasattr(rag, 'chunk_entity_relation_graph') and rag.chunk_entity_relation_graph is not None:
                try:
                    # Get all nodes
                    nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
                    print(f"    🔍 Retrieved {len(nodes) if nodes else 0} nodes from graph storage")
                    
                    for node in nodes:
                        if isinstance(node, dict):
                            entity_name = node.get("entity_name", "") or node.get("id", "")
                            entity_type = node.get("entity_type", "Other")
                            description = node.get("description", "") or node.get("content", "")
                            
                            entities.append({
                                "entity_id": str(uuid.uuid4()),
                                "canonical_name": entity_name,
                                "entity_type": entity_type,
                                "description": description,
                            })
                    
                    # Get all edges
                    edges = await rag.chunk_entity_relation_graph.get_all_edges()
                    print(f"    🔍 Retrieved {len(edges) if edges else 0} edges from graph storage")
                    
                    for edge in edges:
                        if isinstance(edge, dict):
                            source = edge.get("source", "") or edge.get("source_node_id", "")
                            target = edge.get("target", "") or edge.get("target_node_id", "")
                            keywords = edge.get("keywords", "") or edge.get("relationship_type", "")
                            description = edge.get("description", "")
                            
                            if source and target:
                                # Normalize relation type
                                normalized_type = normalize_relation_type(keywords)
                                
                                relations.append({
                                    "relation_id": str(uuid.uuid4()),
                                    "source_entity": source,
                                    "target_entity": target,
                                    "relationship_type": normalized_type,
                                    "description": description,
                                })
                    
                    if entities:
                        print(f"    ✅ Extracted {len(entities)} entities and {len(relations)} relations from graph storage")
                        return entities, relations
                
                except Exception as e:
                    print(f"    ⚠️ Extraction from graph storage failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"    ⚠️ chunk_entity_relation_graph not available")
            
            # Method 2: get from full_entities and full_relations
            if hasattr(rag, 'full_entities') and rag.full_entities is not None:
                try:
                    # Get all entities
                    entities_dict = await rag.full_entities.get_all()
                    
                    # Extract entity names (assumed storage format: {doc_id: {entity_names: [...]}})
                    entity_names = set()
                    if isinstance(entities_dict, dict):
                        for doc_id, entity_data in entities_dict.items():
                            if isinstance(entity_data, dict):
                                names = entity_data.get("entity_names", [])
                                if isinstance(names, list):
                                    entity_names.update(names)
                    
                    # Create entity objects for each entity name
                    for entity_name in entity_names:
                        entities.append({
                            "entity_id": str(uuid.uuid4()),
                            "canonical_name": entity_name,
                            "entity_type": "Other",  # type needs to be obtained from elsewhere
                            "description": "",
                        })
                    
                    # Get all relations
                    relations_dict = await rag.full_relations.get_all()
                    
                    if isinstance(relations_dict, dict):
                        for doc_id, relation_data in relations_dict.items():
                            if isinstance(relation_data, dict):
                                relation_pairs = relation_data.get("relation_pairs", [])
                                if isinstance(relation_pairs, list):
                                    for pair in relation_pairs:
                                        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                                            source, target = pair[0], pair[1]
                                            relations.append({
                                                "relation_id": str(uuid.uuid4()),
                                                "source_entity": source,
                                                "target_entity": target,
                                                "relationship_type": "related_to",
                                                "description": "",
                                            })
                    
                    if entities:
                        print(f"    ✅ Extracted {len(entities)} entities and {len(relations)} relations from full_entities")
                        return entities, relations
                
                except Exception as e:
                    print(f"    ⚠️ Extraction from full_entities failed: {e}")
        
        except Exception as e:
            print(f"  ⚠️ Extraction of entities and relations from LightRAG failed: {e}")
        
        return entities, relations
    
    async def extract_data_from_graph(
        self,
        record: Dict[str, Any],
        question_hash: str,
        current_idx: int = 0,
        total_count: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Extract data from an already-built graph (no graph construction).
        
        Args:
            record: Input record
            question_hash: Hash of the question
            current_idx: Current processing index (for progress display)
            total_count: Total record count (for progress display)
        
        Returns:
            Dict[str, Any]: Full graph result on success
            None: On failure or invalid result (should skip saving)
        """
        question = record.get("question", "")
        # answer may be in 'answer' or 'golden_answer' field
        golden_answer = record.get("golden_answer", "") or record.get("answer", "")
        visited_urls_raw = record.get("visited_urls", [])
        
        progress_prefix = f"[{current_idx}/{total_count}]" if total_count > 0 else ""
        
        print(f"{progress_prefix} 📝 Question: {question[:100]}...")
        print(f"{progress_prefix} 🎯 Answer: {golden_answer[:100] if golden_answer else '(empty)'}...")
        
        # Step 1: Check if the graph is complete (must be complete before extracting data)
        graph_complete = is_graph_complete(Path(self.working_dir), question_hash)
        
        if not graph_complete:
            print(f"{progress_prefix} ⚠️ Graph is incomplete, cannot extract data (run step1 first to build the graph)")
            return None
        
        print(f"{progress_prefix} ✅ Graph is complete, starting data extraction...")
        
        # Step 2: Use golden_answer directly as the seed entity
        if not golden_answer or not golden_answer.strip():
            print(f"{progress_prefix} ⚠️ answer field is empty (checked both golden_answer and answer), skipping")
            return None
        
        seed_entities = [{
            "name": golden_answer.strip(),
            "type": "Other"
        }]
        seed_entity_names = [e["name"] for e in seed_entities]
        
        # In step2 we do not need to limit the number of visited_urls
        # (they are only used for entity validation, not graph construction)
        visited_urls = visited_urls_raw
        
        # Step 3: Create LightRAG instance (read-only mode)
        print(f"{progress_prefix} 🔧 Creating LightRAG instance (reading built graph)...")
        rag = await self._create_rag_instance(question_hash)
        
        # Step 4: Organize content for entity validation
        print(f"{progress_prefix} 📄 Organizing content for entity validation...")
        content = self._organize_content_for_validation(
                question, golden_answer, seed_entities, visited_urls
            )
        
        # Step 4: Extract entities and relations from LightRAG
        print(f"{progress_prefix} 📊 Extracting entities and relations...")
        entities, relations = await self._extract_entities_and_relations_from_rag(rag)
        print(f"{progress_prefix} ✅ Extracted {len(entities)} entities and {len(relations)} relations")
        
        if not entities:
            print(f"{progress_prefix} ⚠️ No entities extracted, skipping")
            return None
        
        # Step 4.5: Verify that entities appear in the input content
        print(f"{progress_prefix} 🔍 Verifying entities appear in input content...")
        entities, relations = verify_entities_in_content(entities, relations, content)
        
        if not entities:
            print(f"{progress_prefix} ⚠️ No valid entities after validation, skipping")
            return None
        
        print(f"{progress_prefix} ✅ Validation complete: {len(entities)} valid entities, {len(relations)} valid relations")
        
        # Step 4.6: Filter isolated nodes (entities with no edges)
        print(f"{progress_prefix} 🔍 Filtering isolated nodes (entities with no edges)...")
        # Collect all entity names that appear in relations
        entities_in_relations = set()
        for relation in relations:
            source_entity = relation.get("source_entity", "").strip()
            target_entity = relation.get("target_entity", "").strip()
            if source_entity:
                entities_in_relations.add(source_entity)
            if target_entity:
                entities_in_relations.add(target_entity)
        
        # Filter entities: only keep entities that appear in at least one relation
        filtered_entities = []
        isolated_count = 0
        for entity in entities:
            entity_name = entity.get("canonical_name", "").strip()
            if entity_name in entities_in_relations:
                filtered_entities.append(entity)
            else:
                isolated_count += 1
        
        entities = filtered_entities
        
        print(f"{progress_prefix}   🔍 Isolated node filter: {len(entities) + isolated_count} -> {len(entities)} (removed {isolated_count} isolated nodes)")
        
        if not entities:
            print(f"{progress_prefix} ⚠️ No valid entities after filtering isolated nodes, skipping")
            return None
        
        # Step 4.7: Compute graph structure first for selecting the most important entities
        print(f"{progress_prefix} 📈 Computing graph structure statistics (for top entity selection)...")
        seed_entity_names = [e["name"] for e in seed_entities]
        seed_entity_ids = [e["entity_id"] for e in entities if e["canonical_name"] in seed_entity_names]
        
        # Temporarily compute node metrics (for ranking)
        node_metrics_temp, _ = calculate_graph_metrics(entities, relations, seed_entity_ids)
        
        # Compute importance score for each entity
        def calculate_importance_score(entity_id: str, metrics: Dict[str, Any]) -> float:
            """Compute entity importance score based on graph structure metrics."""
            if entity_id not in metrics:
                return 0.0
            
            m = metrics[entity_id]
            # Combine multiple metrics: degree, pagerank, betweenness, seed status, cut vertex
            score = (
                m.get("degree", 0) * 0.3 +          # degree weight 30%
                m.get("pagerank", 0.0) * 1000 * 0.3 +  # pagerank weight 30% (scaled by 1000)
                m.get("betweenness", 0.0) * 100 * 0.2 +  # betweenness weight 20% (scaled by 100)
                (1.0 if m.get("is_seed", False) else 0.0) * 0.15 +  # seed entity weight 15%
                (1.0 if m.get("is_cut_vertex", False) else 0.0) * 0.05  # cut vertex weight 5%
            )
            return score
        
        # Compute importance scores for all entities
        entity_scores = []
        for entity in entities:
            entity_id = entity["entity_id"]
            score = calculate_importance_score(entity_id, node_metrics_temp)
            entity_scores.append((entity, score))
        
        # Sort by importance score
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter and select top-N entities (with filtering conditions)
        # Goal: find exactly top_entities entities that pass all filters (cost-efficient)
        print(f"{progress_prefix} 🔍 Filtering and selecting top entities...")
        print(f"{progress_prefix}   Target: select exactly {self.top_entities} entities that pass all filters for LLM extraction (cost-efficient)")
        print(f"{progress_prefix}   Filter conditions:")
        print(f"{progress_prefix}   1. Single-char/word filter: skip if entity name has only one word (English) or one character (Chinese)")
        print(f"{progress_prefix}   2. Similarity filter: skip if string similarity with already-selected entities >= 50% (avoid duplicates)")
        print(f"{progress_prefix}   3. Source URL count filter: skip if entity has >= 5 source_urls (likely too generic)")
        top_entities_list = []
        filtered_count = 0
        filtered_reasons = {"single_char": 0, "similarity": 0, "too_many_urls": 0}
        
        # Traverse entities (sorted by importance); stop as soon as enough pass filters
        for entity, score in entity_scores:
            # Stop immediately once we have enough entities (cost-efficient)
            if len(top_entities_list) >= self.top_entities:
                break
            
            entity_name = entity["canonical_name"]
            
            # Filter 1: skip if only a single word (English) or single character (Chinese)
            if is_single_word_or_char(entity_name):
                filtered_count += 1
                filtered_reasons["single_char"] += 1
                continue
            
            # Filter 2: skip if string similarity with already-selected entities > 66.7%
            is_similar = False
            for selected_entity, _ in top_entities_list:
                selected_name = selected_entity["canonical_name"]
                similarity = calculate_string_similarity(entity_name, selected_name)
                if similarity > 66.7:
                    is_similar = True
                    filtered_count += 1
                    filtered_reasons["similarity"] += 1
                    break
            
            if is_similar:
                continue
            
            # Filter 3: skip if entity has >= 5 source_urls (likely too generic)
            rule_fields_temp = extract_rule_based_fields(entity_name, visited_urls)
            source_urls_count = len(rule_fields_temp.get("source_urls", []))
            if source_urls_count >= 5:
                filtered_count += 1
                filtered_reasons["too_many_urls"] += 1
                continue
            
            # Passed all filters — add to top list
            top_entities_list.append((entity, score))
        
        print(f"{progress_prefix}   Filter stats: single_char={filtered_reasons['single_char']}, similarity={filtered_reasons['similarity']}, too_many_urls={filtered_reasons['too_many_urls']}")
        
        # Check if we found enough entities
        if len(top_entities_list) < self.top_entities:
            print(f"{progress_prefix} ⚠️  Warning: only {len(top_entities_list)} entities passed all filters, fewer than target {self.top_entities}")
            print(f"{progress_prefix}    Will use all {len(top_entities_list)} qualifying entities for LLM extraction")
        else:
            print(f"{progress_prefix} ✅ Found {len(top_entities_list)} qualifying entities (target: {self.top_entities}), stopped traversal (cost-efficient)")
        
        # Get remaining entities (not selected), keeping (entity, score) tuple format
        selected_entity_ids = {e["entity_id"] for e, _ in top_entities_list}
        other_entities_list = [(e, score) for e, score in entity_scores if e["entity_id"] not in selected_entity_ids]
        
        print(f"{progress_prefix} ✅ Selected {len(top_entities_list)} most important entities for LLM extraction (filtered {filtered_count} entities)")
        print(f"{progress_prefix} ℹ️  Remaining {len(other_entities_list)} entities will use rule-based extraction only")
        
        # Step 5: Add extended fields to each entity
        print(f"{progress_prefix} 🔍 Extracting extended fields (LLM: {len(top_entities_list)}, rule: {len(other_entities_list)})...")
        enhanced_entities = []
        
        # Process top-N entities first (using LLM extraction)
        for entity, importance_score in top_entities_list:
            entity_name = entity["canonical_name"]
            entity_type = entity["entity_type"]
            description = entity["description"]
            
            # Rule-based extraction
            rule_fields = extract_rule_based_fields(entity_name, visited_urls)
            
            # LLM extraction
            source_content = "\n\n".join(rule_fields.get("source_spans", []))
            llm_fields, status_info = await extract_llm_based_fields(
                entity_name, entity_type, description, source_content,
                self.client, self.model
            )
            
            # Print detailed status info
            if status_info["status"] != "success":
                print(f"      ⚠️ Entity '{entity_name}' LLM extraction status: {status_info['status']}")
                print(f"         Reason: {status_info['message']}")
            elif not llm_fields.get("key_attributes") and not llm_fields.get("surface_forms") and not llm_fields.get("aliases"):
                print(f"      ℹ️  Entity '{entity_name}' LLM extraction complete but all fields are empty")
                print(f"         Details: {status_info['message']}")
            
            # Compute retrieval_trace
            is_seed = entity_name in [e["name"] for e in seed_entities]
            source_urls = rule_fields.get("source_urls", [])
            discovered_via_url = None if is_seed else (source_urls[0] if source_urls else None)
            retrieval_trace = {
                "path": "seed_entity" if is_seed else "expanded_from_seed",
                "depth": 0 if is_seed else -1,  # will be computed later
                "parent_entity": None if is_seed else "TBD",
                "discovered_via_url": discovered_via_url,
            }
            
            # Get node metrics (for saving importance info)
            entity_id = entity["entity_id"]
            metrics = node_metrics_temp.get(entity_id, {})
            
            # Merge fields; add LLM-used flag and importance info
            enhanced_entity = {
                **entity,
                **llm_fields,
                **rule_fields,
                "retrieval_trace": retrieval_trace,
                "used_llm": True,  # mark as LLM-extracted
                "importance_score": float(importance_score),
                "importance_metrics": {  # detailed importance metrics
                    "degree": metrics.get("degree", 0),
                    "pagerank": float(metrics.get("pagerank", 0.0)),
                    "betweenness": float(metrics.get("betweenness", 0.0)),
                    "is_seed": metrics.get("is_seed", False),
                    "is_cut_vertex": metrics.get("is_cut_vertex", False),
                },
            }
            
            enhanced_entities.append(enhanced_entity)
        
        # Process remaining entities (rule-based extraction only, no LLM)
        for entity, importance_score in other_entities_list:
            entity_name = entity["canonical_name"]
            entity_type = entity["entity_type"]
            description = entity["description"]
            
            # Rule-based extraction only
            rule_fields = extract_rule_based_fields(entity_name, visited_urls)
            
            # No LLM call — use empty values
            llm_fields = {
                "key_attributes": {},
                "surface_forms": [],
                "aliases": [],
            }
            
            # Compute retrieval_trace
            is_seed = entity_name in seed_entity_names
            source_urls = rule_fields.get("source_urls", [])
            discovered_via_url = None if is_seed else (source_urls[0] if source_urls else None)
            retrieval_trace = {
                "path": "seed_entity" if is_seed else "expanded_from_seed",
                "depth": 0 if is_seed else -1,  # will be computed later
                "parent_entity": None if is_seed else "TBD",
                "discovered_via_url": discovered_via_url,
            }
            
            # Get node metrics
            entity_id = entity["entity_id"]
            metrics = node_metrics_temp.get(entity_id, {})
            
            # Merge fields; add LLM-used flag and importance info
            enhanced_entity = {
                **entity,
                **llm_fields,
                **rule_fields,
                "retrieval_trace": retrieval_trace,
                "used_llm": False,  # mark as rule-based only
                "importance_score": float(importance_score),
                "importance_metrics": {  # detailed importance metrics
                    "degree": metrics.get("degree", 0),
                    "pagerank": float(metrics.get("pagerank", 0.0)),
                    "betweenness": float(metrics.get("betweenness", 0.0)),
                    "is_seed": metrics.get("is_seed", False),
                    "is_cut_vertex": metrics.get("is_cut_vertex", False),
                },
            }
            
            enhanced_entities.append(enhanced_entity)
        
        print(f"{progress_prefix} ✅ Extended field extraction complete ({len(top_entities_list)} LLM, {len(other_entities_list)} rule-based)")
        
        # Step 6: Add evidence and importance info to relations
        print(f"{progress_prefix} 🔗 Processing relations...")
        enhanced_relations = []
        
        # Create entity name → importance score mapping
        entity_name_to_importance = {
            e["canonical_name"]: e.get("importance_score", 0.0)
            for e in enhanced_entities
        }
        
        for relation in relations:
            # Add source_urls and evidence_spans to each relation
            source_entity = relation["source_entity"]
            target_entity = relation["target_entity"]
            
            # Find URLs that contain both entities
            source_urls = []
            evidence_spans = []
            
            for visited_url in visited_urls:
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
                
                if source_entity.lower() in content.lower() and target_entity.lower() in content.lower():
                    source_urls.append(visited_url.get("url", ""))
                    
                    # Extract evidence span
                    idx1 = content.lower().find(source_entity.lower())
                    idx2 = content.lower().find(target_entity.lower())
                    if idx1 != -1 and idx2 != -1:
                        start = min(idx1, idx2)
                        end = max(idx1 + len(source_entity), idx2 + len(target_entity))
                        # Expand context
                        start = max(0, start - 50)
                        end = min(len(content), end + 50)
                        span = content[start:end].strip()
                        evidence_spans.append(span)
            
            # Compute relation importance score (sum of both entity importance scores)
            source_importance = entity_name_to_importance.get(source_entity, 0.0)
            target_importance = entity_name_to_importance.get(target_entity, 0.0)
            relation_importance = source_importance + target_importance
            
            enhanced_relation = {
                **relation,
                "source_urls": source_urls[:5],
                "evidence_spans": evidence_spans[:3],
                "importance_score": float(relation_importance),
            }
            
            enhanced_relations.append(enhanced_relation)
        
        print(f"{progress_prefix} ✅ Relation processing complete (original relations: {len(enhanced_relations)})")
        
        # Step 6.5: Add reverse relations to increase cycle count
        print(f"{progress_prefix} 🔄 Adding reverse relations (to increase cycle count)...")
        
        # Mark all original relations with relation_aug = 0
        for rel in enhanced_relations:
            rel["relation_aug"] = 0  # original relation
        
        # Create reverse relations for each original relation
        reverse_relations = []
        for rel in enhanced_relations:
            reverse_rel = {
                "relation_id": str(uuid.uuid4()),
                "source_entity": rel["target_entity"],  # reversed: target becomes source
                "target_entity": rel["source_entity"],  # reversed: source becomes target
                "relationship_type": f"reverse_{rel['relationship_type']}",  # add reverse_ prefix
                "description": f"Reverse relation: {rel.get('description', '')}",
                "source_urls": rel.get("source_urls", []),
                "evidence_spans": rel.get("evidence_spans", []),
                "importance_score": rel.get("importance_score", 0.0),
                "relation_aug": 1,  # mark as augmented (reverse) relation
            }
            reverse_relations.append(reverse_rel)
        
        # Merge original and reverse relations
        print(f"{progress_prefix}   Added {len(reverse_relations)} reverse relations")
        enhanced_relations.extend(reverse_relations)
        print(f"{progress_prefix} ✅ Relation augmentation complete (total: {len(enhanced_relations)} = {len(enhanced_relations) - len(reverse_relations)} original + {len(reverse_relations)} reverse)")
        
        # Step 7: Recompute graph structure statistics using the full enhanced_entities
        print(f"{progress_prefix} 📈 Computing final graph structure statistics...")
        seed_entity_ids_final = [e["entity_id"] for e in enhanced_entities if e["canonical_name"] in seed_entity_names]
        
        node_metrics, graph_statistics = calculate_graph_metrics(
            enhanced_entities, enhanced_relations, seed_entity_ids_final
        )
        
        # Update entity depth info
        for entity in enhanced_entities:
            entity_id = entity["entity_id"]
            if entity_id in node_metrics:
                entity["retrieval_trace"]["depth"] = node_metrics[entity_id]["depth_from_seed"]
        
        print(f"{progress_prefix} ✅ Graph structure statistics complete")
        print(f"{progress_prefix}   - Entities: {graph_statistics['num_entities']}")
        print(f"{progress_prefix}   - Relations: {graph_statistics['num_relations']}")
        print(f"{progress_prefix}   - Has cycles: {graph_statistics['has_cycles']}")
        print(f"{progress_prefix}   - Cycles (undirected): {graph_statistics['undirected_cycle_count']}")
        print(f"{progress_prefix}   - Cycles (directed): {graph_statistics['directed_cycle_count']}")
        
        # Sort entities and relations by importance
        print(f"{progress_prefix} 📊 Sorting entities and relations by importance...")
        enhanced_entities.sort(key=lambda x: x.get("importance_score", 0.0), reverse=True)
        enhanced_relations.sort(key=lambda x: x.get("importance_score", 0.0), reverse=True)
        
        # Assemble final result
        result = {
            "question": question,
            "golden_answer": golden_answer,
            "seed_entities": [e["name"] for e in seed_entities],
            "graph_statistics": graph_statistics,
            "entities": enhanced_entities,  # sorted by importance
            "relations": enhanced_relations,  # sorted by importance
            "node_metrics": node_metrics,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
        
        # Preserve 'answer' field from the original record if present
        if "answer" in record:
            result["answer"] = record["answer"]
        
        return result
    
    def process_records(
        self,
        records: List[Dict[str, Any]],
        output_path: Path,
        enable_resume: bool = True,
        start: int = 0,
        max_samples: Optional[int] = None,
        parallel_questions: int = 1,
        rerun_existing: bool = False,
    ) -> int:
        """Process all records using multiprocessing concurrency.
        
        Args:
            parallel_questions: Number of questions to process concurrently (default 1 = sequential)
            rerun_existing: If True, re-process records already in the output file (default False)
        """
        n = len(records)
        if start < 0:
            start = 0
        end = n if max_samples is None else min(n, start + max_samples)
        
        # Load already-processed records (step2: only check cache_3b output)
        processed_hashes = set()
        if enable_resume:
            processed_hashes = load_existing_results(output_path)
        
        # If rerun_existing is enabled, remove those records from the output file first
        if rerun_existing and processed_hashes:
            records_to_rerun = set()
            for i in range(start, end):
                record = records[i]
                question = record.get("question", "")
                if question:
                    question_hash = get_question_hash(question)
                    if question_hash in processed_hashes:
                        records_to_rerun.add(question_hash)
            
            if records_to_rerun:
                print(f"\n🔄 Preparing to re-process {len(records_to_rerun)} existing records")
                remove_records_from_output(output_path, records_to_rerun)
                # Remove from processed_hashes so they will be re-processed
                processed_hashes -= records_to_rerun
        
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        import time
        overall_start_time = time.time()
        
        # Prepare task list for multiprocessing
        tasks = []
        already_processed_in_range = 0
        need_process_in_range = 0
        
        for i in range(start, end):
            record = records[i]
            question = record.get("question", "")
            
            # Check if already processed (for statistics)
            if enable_resume and question:
                question_hash = get_question_hash(question)
                if question_hash in processed_hashes:
                    already_processed_in_range += 1
                else:
                    need_process_in_range += 1
            else:
                need_process_in_range += 1
            
            # Prepare argument tuple (pass processed hash set to avoid re-reading file)
            task_args = (
                i,
                record,
                str(self.working_dir),
                self.model,
                self.top_entities,
                self.rag_config["llm_model_max_async"],
                str(output_path),
                enable_resume,
                end,
                processed_hashes,
            )
            tasks.append(task_args)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"🚀 Starting batch data extraction")
        print(f"📊 Processing range: [{start+1}] to [{end}], total {end - start} records")
        print(f"⚡ Concurrency: {parallel_questions} questions in parallel (multiprocessing)")
        if enable_resume:
            print(f"♻️  Resume mode: enabled")
            print(f"   📁 Globally processed: {len(processed_hashes)} records")
            print(f"   📋 Current range statistics ([{start+1}-{end}]):")
            print(f"      ✅ Already processed: {already_processed_in_range} (will be skipped)")
            print(f"      🔄 Pending: {need_process_in_range} (will be executed)")
            print(f"      📊 Total: {len(tasks)}")
            if already_processed_in_range > 0:
                skip_percentage = (already_processed_in_range / len(tasks)) * 100
                print(f"      💡 Skip ratio: {skip_percentage:.1f}%")
        print(f"⚠️  Note: graphs must have been built in {self.working_dir} (via step1)")
        print(f"{'='*80}\n")
        
        # Use ProcessPoolExecutor for multiprocessing concurrency
        # Multiprocessing avoids event-loop lock conflicts in the lightrag library
        if parallel_questions > 1:
            print(f"\n🚀 Using multiprocessing: {parallel_questions} questions in parallel")
            print(f"   Total tasks: {len(tasks)}")
            if enable_resume:
                print(f"   - Already processed (skip): {already_processed_in_range}")
                print(f"   - Pending (execute): {need_process_in_range}")
            print(f"   Concurrency: {parallel_questions} (at most {parallel_questions} processes simultaneously)")
            print(f"   Submitting tasks...\n")
            
            with ProcessPoolExecutor(max_workers=parallel_questions) as executor:
                future_to_task = {}
                for task_args in tasks:
                    future = executor.submit(_process_single_record_worker, task_args)
                    future_to_task[future] = task_args
                
                results = []
                completed_count = 0
                failed_count = 0
                success_count = 0
                
                for future in as_completed(future_to_task):
                    completed_count += 1
                    task_info = future_to_task.get(future, None)
                    task_idx = task_info[0] if task_info else None
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result:
                            i, success, failed = result
                            if success:
                                success_count += 1
                                print(f"   ✅ [{i+1}/{end}] Task completed and saved | Progress: {completed_count}/{len(tasks)} ({completed_count*100//len(tasks)}%) | Success: {success_count} | Failed: {failed_count}")
                            elif failed:
                                failed_count += 1
                                print(f"   ❌ [{i+1}/{end}] Task failed (not saved) | Progress: {completed_count}/{len(tasks)} ({completed_count*100//len(tasks)}%) | Success: {success_count} | Failed: {failed_count}")
                            else:
                                print(f"   ⏭️  [{i+1}/{end}] Task skipped (not saved) | Progress: {completed_count}/{len(tasks)} ({completed_count*100//len(tasks)}%) | Success: {success_count} | Failed: {failed_count}")
                        else:
                            failed_count += 1
                            print(f"   ❌ [{task_idx+1 if task_idx is not None else '?'}/{end}] Task returned None | Progress: {completed_count}/{len(tasks)} | Success: {success_count} | Failed: {failed_count}")
                    except Exception as e:
                        failed_count += 1
                        print(f"   ❌ [{task_idx+1 if task_idx is not None else '?'}/{end}] Task exception: {e} | Progress: {completed_count}/{len(tasks)} | Success: {success_count} | Failed: {failed_count}")
                        import traceback
                        traceback.print_exc()
                        if task_idx is not None:
                            results.append((task_idx, False, True))
                        else:
                            results.append((None, False, True))
                    
                    # Print summary every 50 completions
                    if completed_count % 50 == 0:
                        print(f"\n   📊 Progress summary: {completed_count}/{len(tasks)} ({completed_count*100//len(tasks)}%) | Success: {success_count} | Failed: {failed_count} | Skipped: {completed_count - success_count - failed_count}\n")
        else:
            # Sequential processing
            print(f"\n📝 Sequential processing: processing questions one by one")
            results = []
            for task_args in tasks:
                try:
                    result = _process_single_record_worker(task_args)
                    results.append(result)
                except Exception as e:
                    print(f"  ❌ Task exception: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append((task_args[0], False, True))
        
        # Tally results
        for result in results:
            if result is None:
                failed_count += 1
                continue
            
            i, success, failed = result
            if i is None:
                failed_count += 1
                continue
            
            if failed:
                failed_count += 1
            elif success:
                processed_count += 1
                total_processed = processed_count + len(processed_hashes)
                total_target = len(records)
                progress_percentage = (total_processed / total_target) * 100 if total_target > 0 else 0
                print(f"📈 Live progress: {total_processed}/{total_target} ({progress_percentage:.1f}%)")
            else:
                skipped_count += 1
        
        overall_elapsed_time = time.time() - overall_start_time
        
        print(f"\n" + "="*80)
        print(f"🎉 Batch data extraction complete!")
        print(f"{'='*80}")
        print(f"📊 Processing statistics:")
        print(f"  ✅ Successfully extracted: {processed_count}")
        print(f"  ⏭️  Skipped (invalid/already processed): {skipped_count}")
        print(f"  ❌ Extraction failed: {failed_count}")
        print(f"  📝 Total attempted: {processed_count + skipped_count + failed_count}")
        print(f"\n⏱️  Timing:")
        print(f"  🕐 Total elapsed: {overall_elapsed_time:.1f}s ({overall_elapsed_time/60:.1f}min)")
        if processed_count > 0:
            avg_time = overall_elapsed_time / processed_count
            print(f"  📈 Average per record: {avg_time:.1f}s")
            remaining = len(records) - end
            if remaining > 0:
                estimated_remaining = remaining * avg_time
                print(f"  🔮 Estimated remaining: {estimated_remaining/60:.1f}min ({remaining} records left)")
        print(f"\n💾 Output file: {output_path}")
        print(f"{'='*80}\n")
        
        return processed_count


# ============================================================================
# Main
# ============================================================================

def _default_output_path(input_path: Path) -> Path:
    """Generates the default output path"""
    return Path("./data_synthesis/cache/cache_3b") / f"{input_path.stem}_step3b.jsonl"


async def main_async(args) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else _default_output_path(input_path)

    print(f"Input       : {input_path}")
    print(f"Output      : {output_path}")
    print(f"Working dir : {args.working_dir} (graphs must be built by step 3a)")
    print(f"Model       : {args.model}")
    print(f"Resume      : {'disabled' if args.no_resume else 'enabled'}")
    print(f"Rerun existing: {bool(args.rerun_existing)}")
    print(f"Concurrency : {args.parallel_questions} processes × {args.llm_max_async} LLM async = "
          f"{args.parallel_questions * args.llm_max_async} total LLM calls")

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set.")
        return

    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records.")

    builder = QuestionCentricGraphBuilder(
        working_dir=args.working_dir,
        model=args.model,
        llm_max_async=args.llm_max_async,
        top_entities=args.top_entities,
    )

    mode = "parallel" if args.parallel_questions > 1 else "sequential"
    print(f"\nStarting data extraction ({mode}, {args.parallel_questions} process(es))...")
    builder.process_records(
        records,
        output_path,
        enable_resume=not args.no_resume,
        start=args.start,
        max_samples=args.max_samples,
        parallel_questions=args.parallel_questions,
        rerun_existing=args.rerun_existing,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 3b: Extract entities and relations from pre-built LightRAG graphs."
    )
    ap.add_argument("--input", required=True, help="Path to input JSONL file (output of step 2).")
    ap.add_argument("--output", default=None, help="Path to output JSONL file (default: data_synthesis/cache/cache_3b/<input_stem>_step3b.jsonl).")
    ap.add_argument("--working-dir", default="./data_synthesis/cache/cache_3a", help="LightRAG working directory (graphs built by step 3a).")
    ap.add_argument("--model", default="deepseek-v3.2", help="LLM model for entity field extraction.")
    ap.add_argument("--llm-max-async", type=int, default=10, help="LLM async concurrency per question.")
    ap.add_argument("--top-entities", type=int, default=20, help="Number of top-importance entities to enrich via LLM (default: 20).")
    ap.add_argument(
        "--parallel-questions",
        type=int,
        default=3,
        help="Number of questions to process concurrently (multi-process, default: 1 = sequential).",
    )
    ap.add_argument("--start", type=int, default=0, help="Start index.")
    ap.add_argument("--max-samples", type=int, default=None, help="Max records to process.")
    ap.add_argument("--no-resume", default=False, help="Disable resume mode.")
    ap.add_argument("--rerun-existing", default=False, help="Re-process records already present in the output file.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
