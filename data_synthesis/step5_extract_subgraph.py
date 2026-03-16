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
v2 Step-5: Extract Subgraphs from Knowledge Graphs

Objective:
- Input: JSONL file containing complete knowledge graphs (from 3_construct_graph.py)
- Processing: For each graph, select N target_entities and construct subgraphs for each target_entity
- Output: JSONL file containing subgraph structures and path information
"""

import argparse
import asyncio
import hashlib
import json
import os
from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env (searches CWD and parent dirs)

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from openai import AsyncOpenAI
from json_repair import repair_json


# ============================================================================
# Utility Functions
# ============================================================================

def load_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    """Loads a JSONL file"""
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid JSON line: {e}")
                continue
    return records


def append_jsonl(record: Dict[str, Any], output_path: Path) -> None:
    """Appends a single record to a JSONL file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_subgraphs(output_path: Path) -> Dict[Tuple[str, str], int]:
    """Loads existing subgraphs, returning processed (target_entity, question_hash) combinations and their counts
    
    Returns:
        Dict[Tuple[str, str], int]: Processed (target_entity, question_hash) combinations -> count
    """
    processed = defaultdict(int)
    
    if not output_path.exists():
        return processed
    
    try:
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    target_entity = record.get("target_entity", "")
                    question_hash = record.get("parent_graph", {}).get("question_hash", "")
                    if target_entity and question_hash:
                        key = (target_entity, question_hash)
                        processed[key] += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"⚠️ Failed to read existing output file: {e}")
    
    return dict(processed)


def is_graph_complete(working_dir: Path, question_hash: str) -> bool:
    """Checks if the graph is complete
    
    A complete graph must contain the graph_chunk_entity_relation.graphml file (graph structure file).
    This file is generated after LightRAG processing. If it exists and is not empty, the graph is considered complete.
    """
    graph_dir = working_dir / question_hash
    if not graph_dir.exists():
        return False
    
    # Only check for graph_chunk_entity_relation.graphml
    graphml_file = graph_dir / "graph_chunk_entity_relation.graphml"
    if not graphml_file.exists() or graphml_file.stat().st_size == 0:
        return False
    
    return True


def contains_chinese(text: str) -> bool:
    """Checks if a string contains Chinese characters"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def is_single_word_or_char(text: str) -> bool:
    """Checks if it's a single word (English) or a single character (Chinese)
    
    Args:
        text: The text to check
    
    Returns:
        True: If it's a single word (English) or a single character (Chinese)
        False: Otherwise
    """
    text = text.strip()
    if not text:
        return True
    
    # Check if contains Chinese
    if contains_chinese(text):
        # Chinese: count characters directly, filter if only 1 character
        return len(text) == 1
    else:
        # English: split by spaces, filter if only 1 word with length<=1
        words = text.split()
        return len(words) == 1 and len(words[0]) <= 1


def calculate_string_similarity(s1: str, s2: str) -> float:
    """Calculates the similarity percentage between two strings (based on LCS)"""
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
    max_length = max(m, n)
    
    if max_length == 0:
        return 0.0
    
    return (lcs_length / max_length) * 100.0


# ============================================================================
# Subgraph Extractor
# ============================================================================

class SubgraphExtractor:
    """Extracts subgraphs from complete knowledge graphs"""
    
    def __init__(
        self,
        total_num_targets: int = 10000000,  # Total number of subgraphs to generate
        max_num_targets_per_entity: int = 1,  # Max times an entity can be a target
        depth_range: List[int] = [1, 2, 3, 4, 5],
        min_entities_per_subgraph: int = 4,
        max_entities_per_subgraph: int = 8,
        min_entities_per_depth: int = 1,  # Min entities per depth layer
        max_entities_per_depth: int = 5,  # Max entities per depth layer
        min_cycle: int = 0,  # Min number of cycles (parent + two children + relation between children)
        min_entities_with_used_llm: int = 2,  # Min number of entities with used_llm=True
        min_degree_for_entities: int = 1,  # Min degree for entities in the graph
        use_factuality_filter: bool = True,  # Whether to enable factuality filtering
        llm_model: str = "deepseek-v3.2",  # LLM model
        llm_base_url: Optional[str] = None,  # LLM API base URL
        llm_api_key: Optional[str] = None,  # LLM API key
        batch_size_for_factuality_scoring: int = 20,  # Number of entities to score per LLM call
        working_dir: Optional[str] = None,  # LightRAG working directory (knowledge_graphs)
        targets_per_quality_node: float = 1.0,  # Number of subgraphs per high-quality node
        min_quality_score_threshold: float = 0.5,  # Quality score threshold for high-quality nodes
        parallel_subgraphs: int = 20,  # Concurrency for subgraph extraction
        parallel_factuality_batches: int = 5,  # Concurrency for factuality scoring batches
    ):
        """
        Args:
            total_num_targets: Total number of subgraphs to generate (across all graphs)
            max_num_targets_per_entity: Max times an entity can appear as a target in subgraphs
            depth_range: List of subgraph depth ranges, e.g. [1, 2, 3] for depths 1, 2, 3
            min_entities_per_subgraph: Minimum number of entities in a subgraph
            max_entities_per_subgraph: Maximum number of entities in a subgraph
            min_entities_per_depth: Minimum number of entities per depth layer (default: 1)
            max_entities_per_depth: Maximum number of entities per depth layer (default: 5)
            min_cycle: Minimum number of cycles (parent + two children + relation between children)
            min_entities_with_used_llm: Minimum number of entities with used_llm=True
            min_degree_for_entities: Minimum degree for entities in the graph (default: 1)
            use_factuality_filter: Whether to enable factuality filtering (LLM selects high-factuality nodes)
            llm_model: LLM model name
            llm_base_url: LLM API base URL
            llm_api_key: LLM API key
            batch_size_for_factuality_scoring: Number of entities to score per LLM call (default: 20)
            working_dir: LightRAG working directory (knowledge_graphs), used to check graph completeness
            targets_per_quality_node: Number of subgraphs per high-quality node (default: 2.0)
            min_quality_score_threshold: Quality score threshold for high-quality nodes (default: 0.5)
            parallel_subgraphs: Concurrency for subgraph extraction (default: 1, sequential)
            parallel_factuality_batches: Concurrency for factuality scoring batches (default: 1, sequential)
        """
        self.total_num_targets = total_num_targets
        self.max_num_targets_per_entity = max_num_targets_per_entity
        self.depth_range = sorted(depth_range)  # Sort to ensure ascending order
        self.max_depth = max(self.depth_range) if self.depth_range else 3  # Max depth for path finding
        self.min_entities_per_subgraph = min_entities_per_subgraph
        self.max_entities_per_subgraph = max_entities_per_subgraph
        self.min_entities_per_depth = min_entities_per_depth
        self.max_entities_per_depth = max_entities_per_depth
        self.min_cycle = min_cycle
        self.min_entities_with_used_llm = min_entities_with_used_llm
        self.min_degree_for_entities = min_degree_for_entities
        
        # Working directory (used to check graph completeness)
        self.working_dir = Path(working_dir) if working_dir else None
        
        # Factuality filtering
        self.use_factuality_filter = use_factuality_filter
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url or os.getenv("OPENAI_BASE_URL")
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY", "")
        self.batch_size_for_factuality_scoring = batch_size_for_factuality_scoring
        
        # Initialize LLM client (if factuality filtering is enabled)
        self.llm_client = None
        if self.use_factuality_filter:
            self.llm_client = AsyncOpenAI(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
            )
        
        # Track entity selection frequency (ensure diversity, limit per-entity usage)
        self.entity_selection_count = defaultdict(int)  # entity_name -> count (as target)
        self.entity_usage_in_subgraphs = defaultdict(int)  # entity_name -> count (in subgraphs)
        self.total_subgraphs_generated = 0  # Total subgraphs generated so far
        
        # Factuality scoring cache (per-graph, avoid redundant LLM calls)
        # entity_name -> {"score": int, "level": str, "key_info": str}
        self.factuality_cache: Dict[str, Dict[str, Any]] = {}
        
        # Adaptive subgraph extraction parameters
        self.targets_per_quality_node = targets_per_quality_node
        self.min_quality_score_threshold = min_quality_score_threshold
        
        # Concurrency control parameters
        self.parallel_subgraphs = parallel_subgraphs
        self.parallel_factuality_batches = parallel_factuality_batches
    
    async def _score_entities_batch(
        self,
        entities_batch: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Scores a batch of entities for factuality (single LLM call)
        
        Args:
            entities_batch: List of entities (a batch)
        
        Returns:
            Dict[entity_name, {"score": int, "level": str, "key_info": str}]
        """
        if not entities_batch or not self.llm_client:
            return {}
        
        # Build candidate entity info
        candidate_info_lines = []
        
        for idx, entity in enumerate(entities_batch):
            entity_name = entity.get("canonical_name", "")
            entity_type = entity.get("entity_type", "Unknown")
            description = entity.get("description", "No description")
            candidate_info_lines.append(
                f"{idx+1}. Entity Name: {entity_name}\n"
                f"   Type: {entity_type}\n"
                f"   Description: {description}"
            )
        
        candidate_info = "\n\n".join(candidate_info_lines)
        
        # Build prompt
        prompt = f"""Please score the **factuality** of the descriptions for the following candidate entities and extract key factual information.

**Factuality Assessment Criteria and Scoring:**
- **High Factuality (80-100 points)**: Description contains specific numbers, times, locations, amounts, measurements, dates, percentages, and other objective data.
  Example keywords: Founded in 2010, 5000 employees, Headquartered in Beijing, Market value of 10 billion USD, Covers an area of 500 square meters, November 2023
  
- **Medium Factuality (50-79 points)**: Contains some specific information, but mainly qualitative descriptions.
  Example keywords: Large company, Located in China, Diverse services, Rapid development, Industry participant
  
- **Low Factuality (0-49 points)**: Mainly abstract, subjective, or generic descriptions, lacking specific data.
  Example keywords: Important organization, Quality services, Influential, Widely concerned, Has potential

**Candidate Entity List (total {len(entities_batch)}):**

{candidate_info}

For **each entity**, please:
1. Score factuality (0-100 points)
2. Extract key factual information (key_info): Directly quote the most factual phrases or words from the original description, separated by commas.

**Important**: key_info must be **entirely from the original description**, not paraphrased or invented; only copy-paste key snippets.

Return JSON format:
{{
  "entity_scores": [
    {{
      "entity_name": "Entity Name",
      "score": 85,
      "level": "High",
      "key_info": "Founded in 2010, 5000 employees, Headquartered in Beijing"
    }},
    {{
      "entity_name": "Entity Name 2",
      "score": 45,
      "level": "Low",
      "key_info": "Important organization, Provides services"
    }},
    ...
  ]
}}

Note:
1. entity_scores must contain scores for **all {len(entities_batch)} candidate entities**.
2. Return in original order, no sorting needed.
3. Level can be: "High" (80-100 points), "Medium" (50-79 points), "Low" (0-49 points).
4. Entity names must exactly match those in the candidate list.
5. key_info must be direct quotes from the original description, not paraphrased or explained.
6. Scoring must be objective and fair, without bias towards certain entities."""

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to repair JSON
                result = json.loads(repair_json(result_text))
            
            entity_scores = result.get("entity_scores", [])
            
            # Build cache dict
            cache = {}
            entity_names = {entity.get("canonical_name", "") for entity in entities_batch}
            
            for score_info in entity_scores:
                entity_name = score_info.get("entity_name", "")
                score = score_info.get("score", 0)
                level = score_info.get("level", "Low")
                key_info = score_info.get("key_info", "")
                
                if entity_name in entity_names:
                    cache[entity_name] = {
                        "score": score,
                        "level": level,
                        "key_info": key_info,
                    }
            
            return cache
            
        except Exception as e:
            print(f"        ⚠️ Batch factuality scoring failed: {e}")
            return {}
    
    def _collect_candidate_entities_for_subgraphs(
        self,
        target_entities: List[Dict[str, Any]],
        G: nx.DiGraph,
        entity_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Collects all candidate entities that might be used in subgraph expansion
        
        From all target entities, find all entities reachable within max_depth.
        
        Args:
            target_entities: List of target entities
            G: NetworkX graph
            entity_map: Mapping from entity name to entity data
        
        Returns:
            List[Dict[str, Any]]: List of candidate entities (deduplicated)
        """
        candidate_entity_names = set()
        
        # Use reversed graph (from target outward)
        G_reversed = G.reverse()
        
        # For each target entity, find all entities reachable within max_depth
        for target_entity in target_entities:
            target_name = target_entity["canonical_name"]
            
            if target_name not in G.nodes():
                continue
            
            # BFS to find all reachable nodes within max_depth
            queue = deque([(target_name, 0)])
            visited = {target_name}
            
            while queue:
                node, depth = queue.popleft()
                
                # Add to candidate set
                if node in entity_map:
                    candidate_entity_names.add(node)
                
                if depth >= self.max_depth:
                    continue
                
                # Find neighbors in reversed graph (potential parent nodes)
                for neighbor in G_reversed.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        
        # Convert to entity data list
        candidate_entities = []
        for entity_name in candidate_entity_names:
            if entity_name in entity_map:
                candidate_entities.append(entity_map[entity_name])
        
        return candidate_entities
    
    async def _batch_score_entities(
        self,
        entities: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Batches entities for factuality scoring (processed in batches, supports concurrency)
        
        Args:
            entities: List of entities
        
        Returns:
            Dict[entity_name, {"score": int, "level": str, "key_info": str}]
        """
        if not entities or not self.llm_client:
            return {}
        
        # Batch processing
        total_batches = (len(entities) + self.batch_size_for_factuality_scoring - 1) // self.batch_size_for_factuality_scoring
        
        print(f"  📦 Batch scoring: {len(entities)} entities in {total_batches} batches (max {self.batch_size_for_factuality_scoring} per batch)")
        if self.parallel_factuality_batches > 1:
            print(f"  ⚡ Batch concurrency: {self.parallel_factuality_batches} batches in parallel")
        
        # Prepare all batches
        batch_tasks = []
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size_for_factuality_scoring
            end_idx = min(start_idx + self.batch_size_for_factuality_scoring, len(entities))
            entities_batch = entities[start_idx:end_idx]
            batch_tasks.append((batch_idx, entities_batch))
        
        # Use Semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.parallel_factuality_batches)
        
        async def process_batch_with_semaphore(batch_idx: int, entities_batch: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Dict[str, Any]]]:
            """Batch processing with semaphore control"""
            async with semaphore:
                print(f"     Batch {batch_idx + 1}/{total_batches}: processing {len(entities_batch)} entities...")
                batch_cache = await self._score_entities_batch(entities_batch)
                return (batch_idx, batch_cache)
        
        # Process all batches concurrently
        if self.parallel_factuality_batches > 1:
            batch_results = await asyncio.gather(*[
                process_batch_with_semaphore(batch_idx, entities_batch)
                for batch_idx, entities_batch in batch_tasks
            ])
        else:
            # Sequential processing
            batch_results = []
            for batch_idx, entities_batch in batch_tasks:
                result = await process_batch_with_semaphore(batch_idx, entities_batch)
                batch_results.append(result)
        
        # Merge all batch results
        all_cache = {}
        for batch_idx, batch_cache in batch_results:
            all_cache.update(batch_cache)
        
        print(f"  📊 Batch factuality scoring complete: {len(all_cache)}/{len(entities)} entities")
        if all_cache:
            print(f"     High factuality (>=80): {sum(1 for e in all_cache.values() if e.get('score', 0) >= 80)}")
            print(f"     Medium factuality (50-79): {sum(1 for e in all_cache.values() if 50 <= e.get('score', 0) < 80)}")
            print(f"     Low factuality (<50): {sum(1 for e in all_cache.values() if e.get('score', 0) < 50)}")
        
        return all_cache
    
    async def _select_entities_by_factuality(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
        max_select: int,
    ) -> List[str]:
        """Uses LLM to select high-factuality entities from candidates and filters each candidate by score
        
        Args:
            candidates: List[(entity_name, entity_data)], list of candidate entities
            max_select: Maximum number to select (for sorting, actual retained count may be less)
        
        Returns:
            List[entity_name], list of selected entity names (only medium-to-high factuality)
        """
        if not candidates:
            return []
        
        # Check cache for existing scores
        cached_scores = {}
        missing_candidates = []
        
        for entity_name, entity_data in candidates:
            if entity_name in self.factuality_cache:
                cached_scores[entity_name] = self.factuality_cache[entity_name]
            else:
                missing_candidates.append((entity_name, entity_data))
        
        # Score missing candidates
        if missing_candidates and self.llm_client:
            print(f"        ⚡ Cache hit: {len(cached_scores)}/{len(candidates)}, scoring {len(missing_candidates)} missing entities")
            
            # Build missing candidate info
            candidate_info_lines = []
            
            for idx, (entity_name, entity_data) in enumerate(missing_candidates):
                entity_type = entity_data.get("entity_type", "Unknown")
                description = entity_data.get("description", "No description")
                candidate_info_lines.append(
                    f"{idx+1}. Entity Name: {entity_name}\n"
                    f"   Type: {entity_type}\n"
                    f"   Description: {description}"
                )
            
            candidate_info = "\n\n".join(candidate_info_lines)
        
            # LLM scoring for missing candidates
            prompt = f"""Please score the **factuality** of the descriptions for the following candidate entities and extract key factual information.

**Factuality Assessment Criteria and Scoring:**
- **High Factuality (80-100 points)**: Description contains specific numbers, times, locations, amounts, measurements, dates, percentages, and other objective data.
  Example keywords: Founded in 2010, 5000 employees, Headquartered in Beijing, Market value of 10 billion USD, Covers an area of 500 square meters, November 2023
  
- **Medium Factuality (50-79 points)**: Contains some specific information, but mainly qualitative descriptions.
  Example keywords: Large company, Located in China, Diverse services, Rapid development, Industry participant
  
- **Low Factuality (0-49 points)**: Mainly abstract, subjective, or generic descriptions, lacking specific data.
  Example keywords: Important organization, Quality services, Influential, Widely concerned, Has potential

**Candidate Entity List (total {len(missing_candidates)}):**

{candidate_info}

For **each entity**, please:
1. Score factuality (0-100 points)
2. Extract key factual information (key_info): Directly quote the most factual phrases or words from the original description, separated by commas.

**Important**: key_info must be **entirely from the original description**, not paraphrased or invented; only copy-paste key snippets.

Return JSON format:
{{
  "entity_scores": [
    {{
      "entity_name": "Entity Name",
      "score": 85,
      "level": "High",
      "key_info": "Founded in 2010, 5000 employees, Headquartered in Beijing"
    }},
    {{
      "entity_name": "Entity Name 2",
      "score": 45,
      "level": "Low",
      "key_info": "Important organization, Provides services"
    }},
    ...
  ]
}}

Note:
1. entity_scores must contain scores for **all {len(missing_candidates)} candidate entities**.
2. Return in original order, no sorting needed.
3. Level can be: "High" (80-100 points), "Medium" (50-79 points), "Low" (0-49 points).
4. Entity names must exactly match those in the candidate list.
5. key_info must be direct quotes from the original description, not paraphrased or explained.
6. Scoring must be objective and fair, without bias towards certain entities."""

            try:
                response = await self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Try to parse JSON
                try:
                    result = json.loads(result_text)
                except json.JSONDecodeError:
                    # Try to repair JSON
                    result = json.loads(repair_json(result_text))
                
                entity_scores = result.get("entity_scores", [])
                missing_names = {name for name, _ in missing_candidates}
                
                # Add new scores to cache
                for score_info in entity_scores:
                    entity_name = score_info.get("entity_name", "")
                    if entity_name in missing_names:
                        self.factuality_cache[entity_name] = {
                            "score": score_info.get("score", 0),
                            "level": score_info.get("level", "Low"),
                            "key_info": score_info.get("key_info", ""),
                        }
                        cached_scores[entity_name] = self.factuality_cache[entity_name]
                
            except Exception as e:
                print(f"        ⚠️ Failed to score missing entities: {e}")
        else:
            # All candidates already in cache
            print(f"        ✅ Cache hit: {len(cached_scores)}/{len(candidates)}, using cached results directly")
        
        # Merge cached and new scores
        candidate_names = {name for name, _ in candidates}
        entity_to_score = {}
        entity_to_level = {}
        entity_to_key_info = {}
        
        for entity_name in candidate_names:
            if entity_name in cached_scores:
                cache_entry = cached_scores[entity_name]
                entity_to_score[entity_name] = cache_entry.get("score", 0)
                entity_to_level[entity_name] = cache_entry.get("level", "Low")
                entity_to_key_info[entity_name] = cache_entry.get("key_info", "")
            else:
                # If not in cache (scoring may have failed), default to 0
                entity_to_score[entity_name] = 0
                entity_to_level[entity_name] = "Low"
                entity_to_key_info[entity_name] = ""
            
        # Post-processing: sort and filter by score
        # 1. Create scored list (entity_name, score)
        scored_entities = []
        for entity_name in candidate_names:
            score = entity_to_score.get(entity_name, 0)  # Default 0 for unscored
            scored_entities.append((entity_name, score))
        
        # 2. Sort by score descending
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Filter: keep only entities with score >= 50
        valid_selected = []
        filtered_count = 0
        
        for entity_name, score in scored_entities:
            if score >= 50:
                # Medium-to-high factuality, keep
                valid_selected.append(entity_name)
            else:
                # Low factuality, filter out
                filtered_count += 1
        
        # 4. Limit count: take top max_select
        valid_selected = valid_selected[:max_select]
        
        if not valid_selected:
            # No entities passed factuality filter (score >= 50)
            print(f"        ⚠️ All {len(candidates)} candidate entities scored <50, no qualified entities")
            return []
        
        # Print detailed info
        print(f"        📊 Factuality scoring + post-processing:")
        print(f"           - Total candidates: {len(candidates)}")
        print(f"           - Cache hit: {len(cached_scores)}/{len(candidates)}")
        print(f"           - Scored: {len(entity_to_score)}/{len(candidates)}")
        print(f"           - High factuality (>=80): {sum(1 for s in entity_to_score.values() if s >= 80)}")
        print(f"           - Medium factuality (50-79): {sum(1 for s in entity_to_score.values() if 50 <= s < 80)}")
        print(f"           - Low factuality (<50): {sum(1 for s in entity_to_score.values() if s < 50)}")
        print(f"           - Filtered out: {filtered_count} low-factuality entities")
        print(f"           - Retained: {len(valid_selected)} (top {max_select} by score)")
        
        # Print top selected entities
        if valid_selected and entity_to_score:
            print(f"           - Final selection (score descending):")
            for i, entity_name in enumerate(valid_selected[:3]):
                score = entity_to_score.get(entity_name, 0)
                level = entity_to_level.get(entity_name, "Unknown")
                key_info = entity_to_key_info.get(entity_name, "")
                # Truncate long names and key_info
                display_name = entity_name[:35] + "..." if len(entity_name) > 35 else entity_name
                display_key_info = key_info[:60] + "..." if len(key_info) > 60 else key_info
                print(f"             {i+1}. [{score}/{level}] {display_name}")
                print(f"                Key info: {display_key_info}")
        
        return valid_selected
    
    def _build_networkx_graph(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Tuple[nx.DiGraph, Dict[str, Dict[str, Any]]]:
        """Builds a NetworkX directed graph"""
        G = nx.DiGraph()
        
        # Map from entity name to entity data
        entity_map = {}
        for entity in entities:
            entity_name = entity["canonical_name"]
            entity_map[entity_name] = entity
            G.add_node(entity_name, **entity)
        
        # Add edges (filter out edges with relationship_type == unknown)
        for relation in relations:
            source = relation.get("source_entity", "")
            target = relation.get("target_entity", "")
            relationship_type = relation.get("relationship_type", "")
            
            # Filter out edges with relationship_type unknown/UNKNOWN (case-insensitive)
            if relationship_type.lower() == "unknown":
                continue
            
            if source in G.nodes() and target in G.nodes():
                G.add_edge(source, target, **relation)
        
        return G, entity_map
    
    def _is_valid_entity(
        self,
        entity_name: str,
        entity_data: Dict[str, Any],
        selected_entities: Set[str],
    ) -> bool:
        """Checks if an entity meets the filtering criteria
        
        Filtering criteria:
        1. Skip if it's a single word/character.
        2. Skip if similarity to already selected entities exceeds 50%.
        3. Filter if source_urls count >= 5.
        4. Filter entities with entity_type "unknown".
        """
        # Condition 1: single character/word filter
        if is_single_word_or_char(entity_name):
            return False
        
        # Condition 2: similarity filter
        for selected_name in selected_entities:
            similarity = calculate_string_similarity(entity_name, selected_name)
            if similarity > 66.7:
                return False
        
        # Condition 3: source_urls count filter
        source_urls = entity_data.get("source_urls", [])
        if len(source_urls) >= 5:
            return False
        
        # Condition 4: filter entities with entity_type unknown/UNKNOWN (case-insensitive)
        entity_type = entity_data.get("entity_type", "")
        if entity_type.lower() == "unknown":
            return False
        
        return True
    
    def _precheck_target_entity(
        self,
        entity_name: str,
        G: nx.DiGraph,
        entity_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Coarse filter: Pre-checks if a target entity can form a valid subgraph
        
        Returns:
            Dict containing:
            - is_valid: bool, whether it passed the pre-check
            - reachable_nodes: int, number of reachable nodes within max_depth
            - potential_used_llm_count: int, number of reachable nodes with used_llm=True
            - potential_cycle_count: int, estimated number of potential cycles (rough estimate)
            - neighbor_degrees: List[int], list of neighbor node degrees
        """
        result = {
            "is_valid": False,
            "reachable_nodes": 0,
            "potential_used_llm_count": 0,
            "potential_cycle_count": 0,
            "neighbor_degrees": [],
        }
        
        # Check 1: target degree
        target_degree = G.degree(entity_name)
        if target_degree < self.min_degree_for_entities:
            return result
        
        # Check 2: BFS to find all reachable nodes within max_depth
        G_undirected = G.to_undirected()
        reachable = set()
        queue = deque([(entity_name, 0)])
        visited = {entity_name}
        depth_map = {entity_name: 0}
        
        while queue:
            node, dist = queue.popleft()
            reachable.add(node)
            
            if dist >= self.max_depth:
                continue
            
            for neighbor in G_undirected.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    depth_map[neighbor] = dist + 1
        
        # Exclude target itself
        reachable.discard(entity_name)
        
        # Filter out nodes with entity_type unknown/UNKNOWN (case-insensitive)
        reachable_filtered = {
            node for node in reachable
            if entity_map.get(node, {}).get("entity_type", "").lower() != "unknown"
        }
        
        result["reachable_nodes"] = len(reachable_filtered)
        
        # Check if reachable node count is sufficient (at least min_entities_per_subgraph - 1)
        if len(reachable_filtered) < self.min_entities_per_subgraph - 1:
            return result
        
        # Check 3: count used_llm=True nodes (filtered nodes only)
        used_llm_count = sum(
            1 for node in reachable_filtered
            if entity_map.get(node, {}).get("used_llm", False)
        )
        result["potential_used_llm_count"] = used_llm_count
        
        # Need at least min_entities_with_used_llm - 1 (target may also be used_llm)
        target_used_llm = entity_map.get(entity_name, {}).get("used_llm", False)
        total_used_llm = used_llm_count + (1 if target_used_llm else 0)
        
        if total_used_llm < self.min_entities_with_used_llm:
            return result
        
        # Check 4: estimate potential cycles (rough estimate: count edges among reachable nodes)
        # Cycle definition: parent + two children + relation between children
        # Rough estimate: count nodes with degree >= 2 (more likely to form cycles)
        potential_cycle_nodes = []
        neighbor_degrees = []
        
        for node in reachable_filtered:
            node_degree = G.degree(node)
            neighbor_degrees.append(node_degree)
            
            # Nodes with degree >= 2 may participate in cycle formation
            if node_degree >= 2:
                potential_cycle_nodes.append(node)
        
        result["neighbor_degrees"] = neighbor_degrees
        
        # Rough cycle estimate: count internal edges among reachable nodes
        internal_edges = 0
        for node1 in reachable_filtered:
            for node2 in reachable_filtered:
                if node1 != node2 and G.has_edge(node1, node2):
                    internal_edges += 1
        
        # Roughly 1 cycle per 3 edges (very rough estimate)
        result["potential_cycle_count"] = internal_edges // 3
        
        # Check 5: minimum cycle count requirement
        if result["potential_cycle_count"] < self.min_cycle:
            return result
        
        # Passed all coarse filter checks
        result["is_valid"] = True
        return result
    
    def _evaluate_node_quality(
        self,
        entity: Dict[str, Any],
        G: nx.DiGraph,
        entity_map: Dict[str, Dict[str, Any]],
    ) -> Tuple[float, bool]:
        """Evaluates the quality score of a node
        
        Args:
            entity: Entity data
            G: NetworkX graph
            entity_map: Mapping from entity name to entity data
            
        Returns:
            Tuple[quality_score (0-1), is_high_quality_node (bool)]
        """
        entity_name = entity.get("canonical_name", "")
        if not entity_name:
            return 0.0, False
        
        # ========== 1. Uniqueness/completeness check (required) ==========
        # Required: desc cannot be empty
        description = entity.get("description", "").strip()
        if not description or description == "No description":
            return 0.0, False
        
        # Required: entity_type cannot be unknown
        entity_type = entity.get("entity_type", "").strip()
        if not entity_type or entity_type.lower() == "unknown":
            return 0.0, False
        
        # Required: key_attributes cannot be empty
        key_attributes = entity.get("key_attributes", {})
        if not key_attributes or (isinstance(key_attributes, dict) and len(key_attributes) == 0):
            return 0.0, False
        
        # ========== 2. Bonus items (optional but important) ==========
        uniqueness_score = 0.0
        
        # surface_forms is a plus
        surface_forms = entity.get("surface_forms", [])
        if surface_forms and len(surface_forms) > 0:
            uniqueness_score += 0.15
        
        # aliases is a plus
        aliases = entity.get("aliases", [])
        if aliases and len(aliases) > 0:
            uniqueness_score += 0.15
        
        # ========== 3. Importance score ==========
        importance_score = entity.get("importance_score", 0.0)
        if not isinstance(importance_score, (int, float)):
            importance_score = 0.0
        
        # Normalize importance score (assume range 0-1)
        normalized_importance = min(max(importance_score, 0.0), 1.0)
        
        # ========== 4. Connectivity check ==========
        connectivity_score = 0.0
        
        if entity_name in G:
            # Get node degree
            node_degree = G.degree(entity_name)
            
            # BFS to find reachable nodes (up to 3 hops)
            G_undirected = G.to_undirected()
            reachable = set()
            queue = deque([(entity_name, 0)])
            visited = {entity_name}
            
            while queue:
                node, dist = queue.popleft()
                reachable.add(node)
                
                if dist >= 3:  # Max 3 hops
                    continue
                
                for neighbor in G_undirected.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            # Exclude self
            reachable.discard(entity_name)
            reachable_count = len(reachable)
            
            # Connectivity score: based on degree and reachable count
            # Very important nodes (importance_score > 0.7) can have fewer connections
            if normalized_importance > 0.7:
                # Very important node: even 1 reachable is acceptable
                if reachable_count >= 1:
                    connectivity_score = 0.3  # Base connectivity score
                else:
                    connectivity_score = 0.0
            else:
                # Regular nodes need more connections
                if reachable_count >= 5:
                    connectivity_score = 0.3
                elif reachable_count >= 3:
                    connectivity_score = 0.2
                elif reachable_count >= 1:
                    connectivity_score = 0.1
                else:
                    connectivity_score = 0.0
            
            # Degree bonus
            if node_degree >= 5:
                connectivity_score += 0.1
            elif node_degree >= 3:
                connectivity_score += 0.05
        
        # ========== 5. Overall quality score ==========
        # Weight allocation:
        # - Uniqueness (required passed + bonus): 30%
        # - Importance: 40%
        # - Connectivity: 30%
        quality_score = (
            (0.7 + uniqueness_score) * 0.3 +  # Base 0.7 (required passed) + bonus
            normalized_importance * 0.4 +
            connectivity_score * 0.3
        )
        
        # Determine if high-quality node (threshold adjustable)
        is_high_quality = quality_score >= self.min_quality_score_threshold
        
        return quality_score, is_high_quality
    
    def _count_high_quality_nodes(
        self,
        entities: List[Dict[str, Any]],
        G: nx.DiGraph,
        entity_map: Dict[str, Dict[str, Any]],
    ) -> Tuple[int, List[float]]:
        """Counts the number of high-quality nodes in the graph
        
        Returns:
            Tuple[number of high-quality nodes, list of quality scores for all nodes]
        """
        high_quality_count = 0
        quality_scores = []
        
        for entity in entities:
            quality_score, is_high_quality = self._evaluate_node_quality(
                entity, G, entity_map
            )
            quality_scores.append(quality_score)
            
            if is_high_quality:
                high_quality_count += 1
        
        return high_quality_count, quality_scores
    
    def _calculate_adaptive_targets(
        self,
        high_quality_count: int,
        total_entities: int,
        remaining_targets: int,
    ) -> int:
        """Adaptively calculates the number of subgraphs to extract from this graph based on high-quality node count
        
        Args:
            high_quality_count: Number of high-quality nodes
            total_entities: Total number of entities
            remaining_targets: Number of remaining targets
            
        Returns:
            Number of subgraphs to extract from this graph
        """
        # If no high-quality nodes, try at least 1
        if high_quality_count == 0:
            return min(1, remaining_targets)
        
        # Base strategy: each high-quality node corresponds to a certain number of subgraphs
        base_targets = int(high_quality_count * self.targets_per_quality_node)
        
        # Limit range:
        # - Min: 1 (even with few high-quality nodes)
        # - Max: no more than remaining targets, total entities, or 3x high-quality count
        min_targets = 1
        max_targets = min(
            remaining_targets,
            total_entities,
            high_quality_count * 3  # At most 3x the high-quality node count
        )
        
        adaptive_targets = max(min_targets, min(max_targets, base_targets))
        
        return adaptive_targets
    
    def _select_target_entities(
        self,
        entities: List[Dict[str, Any]],
        G: nx.DiGraph,
        entity_map: Dict[str, Dict[str, Any]],
        parent_graph_id: str,
        max_targets_needed: int,
    ) -> List[Dict[str, Any]]:
        """Selects target_entities (two-stage strategy: coarse filter + fine filter)
        
        Stage 1 (Coarse Filter): Pre-checks if a target entity can form a valid subgraph
        - Checks degree, number of reachable nodes, used_llm count, potential cycle count
        
        Stage 2 (Fine Filter): Applies specific hyperparameter constraints
        - Filtering criteria, similarity, selection frequency, importance score
        """
        print(f"  🔍 Stage 1 (coarse filter): pre-checking target entities...")
        
        # Stage 1: coarse filter - pre-check each entity
        coarse_candidates = []  # [(entity, precheck_result, importance)]
        
        for entity in entities:
            entity_name = entity["canonical_name"]
            
            # Check per-entity selection limit
            if self.entity_selection_count.get(entity_name, 0) >= self.max_num_targets_per_entity:
                continue
            
            # Basic filter: single char/word, too many source_urls
            if is_single_word_or_char(entity_name):
                continue
            
            source_urls = entity.get("source_urls", [])
            if len(source_urls) >= 5:
                continue
            
            # Filter entities with entity_type unknown/UNKNOWN (case-insensitive)
            entity_type = entity.get("entity_type", "")
            if entity_type.lower() == "unknown":
                continue
            
            # Pre-check: can it form a valid subgraph?
            precheck = self._precheck_target_entity(entity_name, G, entity_map)
            
            if not precheck["is_valid"]:
                continue
            
            importance = entity.get("importance_score", 0.0)
            coarse_candidates.append((entity, precheck, importance))
        
        print(f"     Coarse filter passed: {len(coarse_candidates)}/{len(entities)} candidates")
        
        if not coarse_candidates:
            print(f"  ⚠️ No qualified candidates after coarse filter")
            return []
        
        # Stage 2: fine filter - apply specific hyperparameters and similarity check
        print(f"  🎯 Stage 2 (fine filter): applying hyperparameter constraints...")
        
        # Sort by importance and precheck result
        # Composite score = importance * (1 + potential_cycle_count * 0.5 + potential_used_llm_count * 0.3)
        scored_candidates = []
        
        for entity, precheck, importance in coarse_candidates:
            entity_name = entity["canonical_name"]
            selection_count = self.entity_selection_count.get(entity_name, 0)
            # Also consider total appearances in subgraphs (not just as target)
            usage_in_subgraphs = self.entity_usage_in_subgraphs.get(entity_name, 0)
            
            # Composite score: importance, potential cycles, potential used_llm, usage frequency
            cycle_bonus = precheck["potential_cycle_count"] * 0.5
            llm_bonus = precheck["potential_used_llm_count"] * 0.3
            
            # Usage frequency penalty with exponential decay
            # penalty = 1.0 / (1.0 + selection_count^1.5 + usage_in_subgraphs^0.8)
            selection_penalty = pow(selection_count, 1.5) if selection_count > 0 else 0
            usage_penalty_factor = pow(usage_in_subgraphs, 0.8) if usage_in_subgraphs > 0 else 0
            usage_penalty = 1.0 / (1.0 + selection_penalty + usage_penalty_factor * 0.3)
            
            composite_score = importance * (1.0 + cycle_bonus + llm_bonus) * usage_penalty
            
            scored_candidates.append((entity, precheck, composite_score))
        
        # Sort by composite score descending
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Select targets (apply similarity check)
        selected = []
        selected_names = set()
        
        for entity, precheck, score in scored_candidates:
            entity_name = entity["canonical_name"]
            
            # Similarity check: compare with already selected entities
            skip_due_to_similarity = False
            for selected_name in selected_names:
                similarity = calculate_string_similarity(entity_name, selected_name)
                if similarity > 66.7:
                    skip_due_to_similarity = True
                    break
            
            if skip_due_to_similarity:
                continue
            
            selected.append(entity)
            selected_names.add(entity_name)
            
            # Update selection count
            self.entity_selection_count[entity_name] += 1
            
            if len(selected) >= max_targets_needed:
                break
        
        print(f"     Fine filter passed: {len(selected)}/{len(scored_candidates)} candidates")
        
        return selected
    
    def _find_all_paths_to_target(
        self,
        G: nx.DiGraph,
        target: str,
        max_depth: int,
    ) -> Dict[str, List[List[str]]]:
        """Finds all paths from other nodes to the target (max depth of max_depth)
        
        Uses an undirected graph version to find paths, so connected nodes can be found
        even if the target is a leaf node.
        
        Returns:
            Dict[entity_name, List[path]]
            path: List[entity_name], path from source to target
        """
        # Convert to undirected graph to find all connected nodes regardless of direction
        G_undirected = G.to_undirected()
        
        # BFS to find all nodes reachable to target within max_depth (undirected)
        reachable_nodes = {}  # node -> min_distance
        queue = deque([(target, 0)])
        visited = {target}
        
        while queue:
            node, dist = queue.popleft()
            reachable_nodes[node] = dist
            
            if dist >= max_depth:
                continue
            
            for neighbor in G_undirected.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        # For each reachable node, find all paths to target (undirected)
        node_paths = {}
        
        for source in reachable_nodes:
            if source == target:
                continue
            
            # DFS to find all paths (in undirected graph)
            paths = []
            
            def dfs(current, path, depth):
                if depth > max_depth:
                    return
                
                if current == target:
                    paths.append(path[:])
                    return
                
                # Use undirected graph neighbors
                for neighbor in G_undirected.neighbors(current):
                    if neighbor not in path:  # Avoid cycles
                        path.append(neighbor)
                        dfs(neighbor, path, depth + 1)
                        path.pop()
            
            dfs(source, [source], 0)
            
            if paths:
                node_paths[source] = paths
        
        return node_paths
    
    def _format_path_as_ere(
        self,
        path: List[str],
        G: nx.DiGraph,
    ) -> str:
        """Formats a path as ERE (Entity-Relation-Entity) format"""
        ere_parts = []
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            # Get edge relation type
            edge_data = G.get_edge_data(source, target, {})
            relation = edge_data.get("relationship_type", "related_to")
            
            ere_parts.append(f"{source} --[{relation}]--> {target}")
        
        return " | ".join(ere_parts)
    
    def _analyze_subgraph_topology(
        self,
        subgraph_entities: Set[str],
        target: str,
        G: nx.DiGraph,
        node_paths: Dict[str, List[List[str]]],
    ) -> Dict[str, Any]:
        """Analyzes the topological structure of the subgraph
        
        Returns:
            Dict containing:
            - depth_distribution: Number of nodes at each depth
            - layer_entities: List of entities at each layer
            - intra_layer_edges: List of intra-layer edges
        """
        # Calculate each node's depth (shortest distance from target)
        node_depths = {target: 0}
        
        for node in subgraph_entities:
            if node == target:
                continue
            
            # Find shortest path to target
            if node in node_paths:
                min_hops = min(len(path) - 1 for path in node_paths[node])
                node_depths[node] = min_hops
        
        # Group by depth
        depth_distribution = defaultdict(int)
        layer_entities = defaultdict(list)
        
        for node, depth in node_depths.items():
            depth_distribution[depth] += 1
            layer_entities[depth].append(node)
        
        # Find intra-layer edges (edges between nodes in the same layer)
        intra_layer_edges = []
        
        for depth, entities in layer_entities.items():
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    # Check bidirectional edges
                    if G.has_edge(e1, e2):
                        edge_data = G.get_edge_data(e1, e2, {})
                        intra_layer_edges.append({
                            "depth": depth,
                            "source": e1,
                            "target": e2,
                            "relation": edge_data.get("relationship_type", "related_to"),
                        })
                    if G.has_edge(e2, e1):
                        edge_data = G.get_edge_data(e2, e1, {})
                        intra_layer_edges.append({
                            "depth": depth,
                            "source": e2,
                            "target": e1,
                            "relation": edge_data.get("relationship_type", "related_to"),
                        })
        
        return {
            "depth_distribution": dict(depth_distribution),
            "layer_entities": {k: v for k, v in layer_entities.items()},
            "intra_layer_edges": intra_layer_edges,
            "max_depth": max(node_depths.values()) if node_depths else 0,
        }
    
    def _calculate_topology_hash(self, topology: Dict[str, Any]) -> str:
        """Calculates the hash value of the topological structure (for diversity check)"""
        # Use depth distribution, parent-child edge count, and intra-layer edge count as topology features
        depth_dist = topology.get("depth_distribution", {})
        parent_child_edges_count = len(topology.get("parent_child_edges", []))
        intra_edges_count = len(topology.get("intra_layer_edges", []))
        cycle_count = topology.get("cycle_count", 0)
        
        # Create topology signature
        signature = f"{sorted(depth_dist.items())}_{parent_child_edges_count}_{intra_edges_count}_{cycle_count}"
        return hashlib.md5(signature.encode()).hexdigest()[:8]
    
    def _count_cycles_in_subgraph(
        self,
        G: nx.DiGraph,
        entity_depths: Dict[str, int],
        subgraph_entities: Set[str],
    ) -> int:
        """Counts cycles in the subgraph (triangles formed by parent + two children + relation between children)
        
        Cycle definition: parent node at depth d, two child nodes at depth d+1, and a relation between the child nodes.
        """
        cycle_count = 0
        
        # Group entities by depth
        depth_to_entities = defaultdict(set)
        for entity, depth in entity_depths.items():
            if entity in subgraph_entities:
                depth_to_entities[depth].add(entity)
        
        # Iterate over each depth layer
        for depth in sorted(depth_to_entities.keys()):
            if depth + 1 not in depth_to_entities:
                continue
            
            parent_entities = depth_to_entities[depth]
            child_entities = depth_to_entities[depth + 1]
            
            # For each parent node, find its children
            for parent in parent_entities:
                # Find all children of parent in the graph (at depth+1)
                children_of_parent = []
                for child in child_entities:
                    if G.has_edge(parent, child):
                        children_of_parent.append(child)
                
                # Check if children have relations between them (forming a triangle)
                for i, child1 in enumerate(children_of_parent):
                    for child2 in children_of_parent[i+1:]:
                        # Check bidirectional relation
                        if G.has_edge(child1, child2) or G.has_edge(child2, child1):
                            cycle_count += 1
        
        return cycle_count
    
    async def _build_subgraph_by_depth(
        self,
        target_name: str,
        G: nx.DiGraph,
        entity_map: Dict[str, Dict[str, Any]],
        max_depth: int,
    ) -> Tuple[Set[str], Dict[str, int], int, int]:
        """Builds a subgraph by depth, prioritizing entities with more intra-layer relations
        
        Returns:
            Tuple[
                subgraph_entities: Set of subgraph entities,
                entity_depths: Mapping of entity to depth,
                cycle_count: Number of cycles,
                used_llm_count: Number of entities with used_llm=True
            ]
        """
        # Check target entity degree
        target_degree = G.degree(target_name)
        if target_degree < self.min_degree_for_entities:
            # Target doesn't meet degree requirement, return empty result
            return set(), {}, 0, 0
        
        # BFS from target, expand by depth
        subgraph_entities = {target_name}
        entity_depths = {target_name: 0}
        
        # Layer-by-layer selection: each layer only expands from the previous layer's selected entities
        # This naturally guarantees topology connectivity
        
        # Initialize: target entity at depth 0
        current_layer_entities = {target_name}  # Currently selected entities in this layer
        entity_depths[target_name] = 0
        subgraph_entities.add(target_name)
        
        # Use reversed graph (from target outward)
        G_reversed = G.reverse()
        
        # Layer-by-layer expansion
        for current_depth in range(1, max_depth + 1):
            # Check max entities limit
            if len(subgraph_entities) >= self.max_entities_per_subgraph:
                print(f"        Reached max subgraph entity limit {self.max_entities_per_subgraph}, stopping expansion")
                break
            
            # Find candidate nodes from all selected entities in the previous layer
            candidates = []  # [(node, importance, cross_layer_edges, potential_children, usage_count, parent_entity)]
            seen_candidates = set()
            
            for parent_entity in current_layer_entities:
                # Find neighbors in reversed graph (parent nodes)
                for candidate_node in G_reversed.neighbors(parent_entity):
                    # Skip nodes already in subgraph
                    if candidate_node in subgraph_entities:
                        continue
                    
                    # Skip already-processed candidates
                    if candidate_node in seen_candidates:
                        continue
                    seen_candidates.add(candidate_node)
                    
                    # Get node data
                    node_data = entity_map.get(candidate_node, {})
                    
                    # Check basic filter conditions
                    if not self._is_valid_entity(candidate_node, node_data, subgraph_entities):
                        continue
                    
                    # Check degree requirement
                    node_degree = G.degree(candidate_node)
                    if node_degree < self.min_degree_for_entities:
                        continue
                    
                    # Get importance score
                    importance = node_data.get("importance_score", 0.0)
                    
                    # Cross-layer edges will be updated dynamically during selection
                    cross_layer_edges = 0
                    
                    # Calculate potential children (neighbors in reversed graph = potential next-layer children)
                    potential_children = len(list(G_reversed.neighbors(candidate_node)))
                    
                    # Get usage count of this entity in previous subgraphs
                    usage_count = self.entity_usage_in_subgraphs.get(candidate_node, 0)
                    
                    candidates.append((candidate_node, importance, cross_layer_edges, potential_children, usage_count, parent_entity))
            
            if not candidates:
                print(f"        Depth {current_depth}: no more candidate nodes, stopping expansion")
                break
            
            print(f"        Depth {current_depth}: found {len(candidates)} candidate nodes")
            
            # Select entities for the current layer
            next_layer_entities = set()
            
            # If factuality filter is enabled, use LLM to select high-factuality nodes
            if self.use_factuality_filter and self.llm_client:
                print(f"        Depth {current_depth}: using LLM factuality scoring to filter nodes...")
                
                # Prepare candidate entity info (only entity name and data)
                entities_for_selection = [
                    (node, entity_map.get(node, {}))
                    for node, _, _, _, _, _ in candidates
                ]
                
                # LLM factuality scoring: max_entities_per_depth sets the upper limit,
                # but actual retained count may be less (low-factuality entities filtered out)
                max_to_select = min(
                    self.max_entities_per_depth,
                    self.max_entities_per_subgraph - len(subgraph_entities)
                )
                
                selected_entity_names = await self._select_entities_by_factuality(
                    entities_for_selection,
                    max_to_select
                )
                
                # Add selected entities to subgraph
                for node_name in selected_entity_names:
                    subgraph_entities.add(node_name)
                    entity_depths[node_name] = current_depth
                    next_layer_entities.add(node_name)
                
                # Note: actual selected count may be less than max_entities_per_depth
                # because low-factuality entities are filtered out
                actual_selected = len(next_layer_entities)
                print(f"        Depth {current_depth}: selected {actual_selected} medium-to-high factuality entities")
                
                # If 0 entities selected, no qualified candidates
                if actual_selected == 0:
                    print(f"        Depth {current_depth}: ⚠️ all candidates have low factuality, stopping expansion")
                    break
            else:
                # No factuality filter: use composite score selection strategy
                selected_count = 0
                remaining_candidates = candidates.copy()
                
                while remaining_candidates and selected_count < self.max_entities_per_depth:
                    # Check max entities limit
                    if len(subgraph_entities) >= self.max_entities_per_subgraph:
                        break
                    
                    # Recompute cross_layer_edges and composite score for each candidate
                    scored_candidates = []
                    for node, importance, _, potential_children, usage_count, parent in remaining_candidates:
                        cross_edges = 0
                        # Count edges with already-selected nodes in this layer
                        for selected_node in next_layer_entities:
                            if G.has_edge(node, selected_node) or G.has_edge(selected_node, node):
                                cross_edges += 1
                        
                        # Composite score components:
                        # 1. cross_edges: intra-layer edge count (highest weight)
                        # 2. potential_children: potential child count (second highest)
                        # 3. importance: importance score (normalized 0-1)
                        # 4. usage_penalty: frequency penalty (prefer less-used nodes)
                        
                        norm_importance = importance
                        
                        # Exponential decay usage penalty:
                        # penalty = 1.0 / (1.0 + usage_count^1.3)
                        # used 1x -> 1/2.3=0.43, 2x -> 1/3.5=0.29, 3x -> 1/4.9=0.20
                        usage_penalty = 1.0 / (1.0 + pow(usage_count, 1.3)) if usage_count > 0 else 1.0
                        
                        # Weighted composite:
                        # cross_edges weight: 10.0 (most important)
                        # potential_children weight: 5.0 (second)
                        # importance weight: 1.0 (base)
                        # usage_penalty: multiplicative factor
                        composite_score = (
                            cross_edges * 10.0 +
                            potential_children * 5.0 +
                            norm_importance * 1.0
                        ) * usage_penalty
                        
                        scored_candidates.append((node, importance, cross_edges, potential_children, usage_count, parent, composite_score))
                    
                    # Sort by composite score descending
                    scored_candidates.sort(key=lambda x: x[6], reverse=True)
                    
                    # Select highest-scoring node
                    selected_node, importance, cross_edges, potential_children, usage_count, parent, score = scored_candidates[0]
                    
                    # Remove from candidate list
                    remaining_candidates = [
                        (node, imp, _, pot_child, usage, par)
                        for node, imp, _, pot_child, usage, par, _ in scored_candidates[1:]
                    ]
                    
                    # Add to subgraph
                    subgraph_entities.add(selected_node)
                    entity_depths[selected_node] = current_depth
                    next_layer_entities.add(selected_node)
                    selected_count += 1
                
                print(f"        Depth {current_depth}: selected {len(next_layer_entities)} entities")
            
            # Update current layer for next iteration
            current_layer_entities = next_layer_entities
            
            # If no entities selected in this layer, stop expansion
            if not current_layer_entities:
                break
        
        # Count cycles
        cycle_count = self._count_cycles_in_subgraph(G, entity_depths, subgraph_entities)
        
        # Count entities with used_llm=True
        used_llm_count = sum(
            1 for entity_name in subgraph_entities
            if entity_map.get(entity_name, {}).get("used_llm", False)
        )
        
        return subgraph_entities, entity_depths, cycle_count, used_llm_count
    
    async def extract_subgraph_for_depth(
        self,
        target_entity: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        parent_graph_data: Dict[str, Any],
        subgraph_index: int,
        depth: int,
        all_node_paths: Dict[str, List[List[str]]],
        G: nx.DiGraph,
        entity_map: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extracts a subgraph for a target_entity at a specified depth (new topological structure construction method)"""
        target_name = target_entity["canonical_name"]
        
        print(f"      Depth {depth}: ", end="")
        
        # Build subgraph using depth-based method
        subgraph_entities, entity_depths, cycle_count, used_llm_count = await self._build_subgraph_by_depth(
            target_name, G, entity_map, depth
        )
        
        # Check if build succeeded (target degree may be insufficient)
        if not subgraph_entities:
            print(f"insufficient target degree")
            return None
        
        # Check per-layer entity count requirement
        depth_to_entities = defaultdict(list)
        for entity_name, entity_depth in entity_depths.items():
            if entity_name in subgraph_entities:
                depth_to_entities[entity_depth].append(entity_name)
        
        # Check each layer meets min_entities_per_depth (depth 0 is target, always at least 1)
        # If factuality filter is enabled, relax to at least 1 entity per layer
        for depth, entities_in_depth in depth_to_entities.items():
            if depth == 0:
                continue  # depth 0 is target, always at least 1
            
            # Relax to 1 if factuality filter enabled, otherwise strict min_entities_per_depth
            if self.use_factuality_filter:
                min_required = 1
            else:
                min_required = self.min_entities_per_depth
            
            if len(entities_in_depth) < min_required:
                if self.use_factuality_filter:
                    print(f"depth {depth} has no entities after factuality filtering")
                else:
                    print(f"depth {depth} has insufficient entities ({len(entities_in_depth)} < {self.min_entities_per_depth})")
                return None
        
        # Check minimum entity count
        if len(subgraph_entities) < self.min_entities_per_subgraph:
            print(f"insufficient entities ({len(subgraph_entities)} < {self.min_entities_per_subgraph})")
            return None
        
        # Check cycle count requirement
        if cycle_count < self.min_cycle:
            print(f"insufficient cycles ({cycle_count} < {self.min_cycle})")
            return None
        
        # Check used_llm entity count requirement
        if used_llm_count < self.min_entities_with_used_llm:
            print(f"insufficient used_llm entities ({used_llm_count} < {self.min_entities_with_used_llm})")
            return None
        
        print(f"✓ entities={len(subgraph_entities)}, cycles={cycle_count}, used_llm={used_llm_count}")
        
        # Filter relations: keep parent-child and intra-layer relations
        subgraph_relations_data = []
        for rel in relations:
            source = rel.get("source_entity")
            target = rel.get("target_entity")
            
            if source not in subgraph_entities or target not in subgraph_entities:
                continue
            
            source_depth = entity_depths.get(source, -1)
            target_depth = entity_depths.get(target, -1)
            
            # Keep: 1. parent-child relations (depth diff = 1) 2. intra-layer relations (same depth)
            if abs(source_depth - target_depth) <= 1:
                subgraph_relations_data.append(rel)
        
        # Add depth info to entities
        subgraph_entities_data = []
        for name in subgraph_entities:
            if name in entity_map:
                entity_data = entity_map[name].copy()
                entity_data["depth_in_subgraph"] = entity_depths.get(name, -1)
                subgraph_entities_data.append(entity_data)
        
        # Analyze topology
        topology = {
            "depth_distribution": defaultdict(int),
            "layer_entities": defaultdict(list),
            "intra_layer_edges": [],
            "max_depth": depth,
            "cycle_count": cycle_count,
            "used_llm_count": used_llm_count,
        }
        
        # Collect depth distribution and layer entities
        for entity_name, entity_depth in entity_depths.items():
            if entity_name in subgraph_entities:
                topology["depth_distribution"][entity_depth] += 1
                topology["layer_entities"][entity_depth].append(entity_name)
        
        # Collect intra-layer and parent-child edges (original relations only, excluding augmented)
        parent_child_edges = []  # parent-child relations (depth diff = 1)
        intra_layer_edges = []  # intra-layer relations (same depth)
        
        for rel in subgraph_relations_data:
            # Keep only original relations (relation_aug == 0)
            if rel.get("relation_aug", 0) != 0:
                continue

            source = rel.get("source_entity")
            target = rel.get("target_entity")
            source_depth = entity_depths.get(source, -1)
            target_depth = entity_depths.get(target, -1)
            
            edge_info = {
                "source": source,
                "target": target,
                "source_depth": source_depth,
                "target_depth": target_depth,
                "relation": rel.get("relationship_type", "related_to"),
            }
            
            if source_depth == target_depth:
                # Intra-layer relation
                edge_info["depth"] = source_depth
                intra_layer_edges.append(edge_info)
            elif abs(source_depth - target_depth) == 1:
                # Parent-child relation (depth diff = 1)
                parent_child_edges.append(edge_info)
        
        topology["parent_child_edges"] = parent_child_edges
        topology["intra_layer_edges"] = intra_layer_edges
        
        # Convert defaultdict to regular dict
        topology["depth_distribution"] = dict(topology["depth_distribution"])
        topology["layer_entities"] = {k: v for k, v in topology["layer_entities"].items()}
        
        # Assemble subgraph data
        subgraph = {
            "subgraph_id": f"{parent_graph_data.get('question_hash', 'unknown')}_{subgraph_index}_d{depth}",
            "target_entity": target_name,
            "target_entity_data": target_entity,
            "depth": depth,  # record subgraph depth
            "num_entities": len(subgraph_entities),
            "num_relations": len(subgraph_relations_data),
            "cycle_count": cycle_count,
            "used_llm_count": used_llm_count,
            "entities": subgraph_entities_data,  # includes depth_in_subgraph field
            "relations": subgraph_relations_data,  # only parent-child + intra-layer relations
            "topology": topology,
            "topology_hash": self._calculate_topology_hash(topology),
            "parent_graph": {
                "question": parent_graph_data.get("question", ""),
                # answer may be in "answer" or "golden_answer" field
                "golden_answer": parent_graph_data.get("golden_answer", "") or parent_graph_data.get("answer", ""),
                "question_hash": parent_graph_data.get("question_hash", ""),
            },
        }
        
        return subgraph
    
    async def extract_subgraph(
        self,
        target_entity: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        parent_graph_data: Dict[str, Any],
        subgraph_index: int,
        G: nx.DiGraph,
        entity_map: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extracts subgraphs of multiple depths for a target_entity, merging them into a single data entry
        
        Args:
            target_entity: Target entity
            entities: List of all entities
            relations: List of all relations
            parent_graph_data: Parent graph data
            subgraph_index: Subgraph index
            G: Constructed NetworkX graph
            entity_map: Mapping from entity name to entity data
        
        Returns:
            Merged subgraph (containing information from all depths), returns None if no depth is successful
        """
        target_name = target_entity["canonical_name"]
        
        print(f"    Extracting subgraph #{subgraph_index} (target: {target_name})")
        
        if target_name not in G.nodes():
            print(f"      ⚠️ Target entity not in graph")
            return None
        
        # Find all paths to target (using max depth)
        all_node_paths = self._find_all_paths_to_target(G, target_name, self.max_depth)
        
        if not all_node_paths:
            print(f"      ⚠️ No paths found to target")
            return None
        
        # Only try to expand to max depth; expand as far as possible
        # depth_range=[1,2,3,4] means depth range: min=1, max=4
        # Not required to generate subgraphs at all these depths
        max_target_depth = max(self.depth_range)
        min_target_depth = min(self.depth_range)
        
        subgraph = await self.extract_subgraph_for_depth(
            target_entity,
            entities,
            relations,
            parent_graph_data,
            subgraph_index,
            max_target_depth,  # try expanding to max depth
            all_node_paths,
            G,
            entity_map,
        )
        
        if not subgraph:
            print(f"      ⚠️ Failed to expand to depth {max_target_depth}")
            return None
        
        # Check if actual depth reached is within allowed range
        actual_max_depth = subgraph.get("topology", {}).get("max_depth", 0)
        
        if actual_max_depth < min_target_depth:
            print(f"      ⚠️ Actual depth {actual_max_depth} below minimum {min_target_depth}")
            return None
        
        print(f"      ✅ Actual depth: {actual_max_depth} (allowed: {min_target_depth}-{max_target_depth})")
        
        # No merge needed, use single subgraph directly
        depth_subgraphs = [subgraph]
        successful_depths = [actual_max_depth]
        
        # Merge entities (deduplicate, keep larger depth_in_subgraph)
        merged_entities_map = {}  # entity_name -> entity_data
        for subgraph in depth_subgraphs:
            for entity in subgraph.get("entities", []):
                entity_name = entity.get("canonical_name", "")
                if entity_name:
                    # If already exists, keep the one with larger depth (more complete)
                    existing_depth = merged_entities_map.get(entity_name, {}).get("depth_in_subgraph", -1)
                    current_depth = entity.get("depth_in_subgraph", -1)
                    if current_depth > existing_depth or entity_name not in merged_entities_map:
                        merged_entities_map[entity_name] = entity
        
        merged_entities = list(merged_entities_map.values())
        
        # Merge relations (deduplicate)
        merged_relations_map = {}  # (source, target) -> relation
        for subgraph in depth_subgraphs:
            for rel in subgraph.get("relations", []):
                source = rel.get("source_entity", "")
                target = rel.get("target_entity", "")
                key = (source, target)
                if key not in merged_relations_map:
                    merged_relations_map[key] = rel
        
        merged_relations = list(merged_relations_map.values())
        
        # Merge topology info
        merged_topology = {
            "depth_range": successful_depths,
            "depth_distribution": defaultdict(int),
            "layer_entities": defaultdict(list),
            "parent_child_edges": [],
            "intra_layer_edges": [],
            "max_depth": max(successful_depths) if successful_depths else 0,
            "cycle_count": max(sg.get("cycle_count", 0) for sg in depth_subgraphs),
            "used_llm_count": max(sg.get("used_llm_count", 0) for sg in depth_subgraphs),
        }
        
        # Merge topology info from all depths
        for subgraph in depth_subgraphs:
            sub_topology = subgraph.get("topology", {})
            
            # Merge depth_distribution and layer_entities
            for depth, count in sub_topology.get("depth_distribution", {}).items():
                merged_topology["depth_distribution"][depth] = max(
                    merged_topology["depth_distribution"][depth], count
                )
            
            for depth, entities_list in sub_topology.get("layer_entities", {}).items():
                existing = set(merged_topology["layer_entities"][depth])
                merged_topology["layer_entities"][depth].extend(
                    [e for e in entities_list if e not in existing]
                )
            
            # Merge parent_child_edges and intra_layer_edges (deduplicate)
            existing_parent_child_keys = {
                (e.get("source"), e.get("target")) 
                for e in merged_topology["parent_child_edges"]
            }
            for edge in sub_topology.get("parent_child_edges", []):
                key = (edge.get("source"), edge.get("target"))
                if key not in existing_parent_child_keys:
                    merged_topology["parent_child_edges"].append(edge)
                    existing_parent_child_keys.add(key)
            
            existing_intra_layer_keys = {
                (e.get("source"), e.get("target")) 
                for e in merged_topology["intra_layer_edges"]
            }
            for edge in sub_topology.get("intra_layer_edges", []):
                key = (edge.get("source"), edge.get("target"))
                if key not in existing_intra_layer_keys:
                    merged_topology["intra_layer_edges"].append(edge)
                    existing_intra_layer_keys.add(key)
        
        # Convert defaultdict to regular dict
        merged_topology["depth_distribution"] = dict(merged_topology["depth_distribution"])
        merged_topology["layer_entities"] = {k: v for k, v in merged_topology["layer_entities"].items()}
        
        # Assemble merged subgraph data
        merged_subgraph = {
            "subgraph_id": f"{parent_graph_data.get('question_hash', 'unknown')}_{subgraph_index}",
            "target_entity": target_name,
            "target_entity_data": target_entity,
            "depth_range": successful_depths,  # record successful depths
            "num_entities": len(merged_entities),
            "num_relations": len(merged_relations),
            "cycle_count": merged_topology["cycle_count"],
            "used_llm_count": merged_topology["used_llm_count"],
            "entities": merged_entities,  # merged entities (with depth_in_subgraph field)
            "relations": merged_relations,  # merged relations (parent-child + intra-layer only)
            "topology": merged_topology,  # includes parent_child_edges and intra_layer_edges
            "topology_hash": self._calculate_topology_hash(merged_topology),
            "parent_graph": {
                "question": parent_graph_data.get("question", ""),
                # answer may be in "answer" or "golden_answer" field
                "golden_answer": parent_graph_data.get("golden_answer", "") or parent_graph_data.get("answer", ""),
                "question_hash": parent_graph_data.get("question_hash", ""),
            },
        }
        
        print(f"      ✅ Merged: {len(successful_depths)} depths, {len(merged_entities)} entities, {len(merged_relations)} relations")
        
        # Update entity usage count in subgraphs (for diversity control)
        for entity in merged_entities:
            entity_name = entity.get("canonical_name", "")
            if entity_name:
                self.entity_usage_in_subgraphs[entity_name] += 1
        
        return merged_subgraph
    
    async def process_graph(
        self,
        graph_record: Dict[str, Any],
        graph_index: int,
        existing_subgraphs: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> List[Dict[str, Any]]:
        """Processes a single graph, extracting multiple subgraphs"""
        # Check if total limit already reached
        if self.total_subgraphs_generated >= self.total_num_targets:
            print(f"\n[{graph_index}] Total limit reached ({self.total_num_targets}), skipping")
            return []
        
        question = graph_record.get("question", "")
        print(f"\n[{graph_index}] Processing graph: {question[:60]}...")
        print(f"  Progress: {self.total_subgraphs_generated}/{self.total_num_targets} subgraphs generated")
        
        # Generate question hash
        question_hash = hashlib.md5(question.encode()).hexdigest()[:16]
        graph_record["question_hash"] = question_hash
        
        # Check graph completeness (must contain graph_chunk_entity_relation.graphml)
        if self.working_dir:
            if not is_graph_complete(self.working_dir, question_hash):
                print(f"  ⚠️ Graph not complete (missing graph_chunk_entity_relation.graphml), skipping")
                return []
            print(f"  ✅ Graph completeness check passed")
        
        entities = graph_record.get("entities", [])
        relations = graph_record.get("relations", [])
        
        print(f"  Graph info: {len(entities)} entities, {len(relations)} relations")
        
        # Build NetworkX graph (for coarse filter pre-check)
        G, entity_map = self._build_networkx_graph(entities, relations)
        
        # Adaptively compute number of subgraphs to extract from this graph based on node quality
        remaining_targets = self.total_num_targets - self.total_subgraphs_generated
        
        # Count high-quality nodes
        print(f"  📊 Evaluating node quality...")
        high_quality_count, quality_scores = self._count_high_quality_nodes(
            entities, G, entity_map
        )
        print(f"  ✅ High-quality nodes: {high_quality_count}/{len(entities)}")
        
        # Adaptively calculate target count based on high-quality node count
        max_targets_for_this_graph = self._calculate_adaptive_targets(
            high_quality_count, len(entities), remaining_targets
        )
        print(f"  🎯 Adaptive allocation: will extract up to {max_targets_for_this_graph} subgraphs from this graph")

        # Select target entities (two-stage: coarse + fine filter)
        target_entities = self._select_target_entities(
            entities, G, entity_map, question_hash, max_targets_for_this_graph
        )
        
        if not target_entities:
            print(f"  ⚠️ No suitable target entities")
            return []
        
        print(f"  ✅ Final selection: {len(target_entities)} target entities")
        
        # After fine filter, collect all candidate entities for subgraph expansion
        if self.use_factuality_filter and self.llm_client:
            print(f"  🔄 Collecting all subgraph candidate entities...")
            candidate_entities = self._collect_candidate_entities_for_subgraphs(
                target_entities, G, entity_map
            )
            
            if candidate_entities:
                print(f"  📦 Collected {len(candidate_entities)} candidates, performing batch factuality scoring...")
                # Clear cache for new graph
                self.factuality_cache.clear()
                # Batch scoring
                batch_scores = await self._batch_score_entities(candidate_entities)
                # Update cache
                self.factuality_cache.update(batch_scores)
                # Write key_info to entity data for later use
                for entity in entities:
                    entity_name = entity.get("canonical_name", "")
                    if entity_name in self.factuality_cache:
                        entity["key_info"] = self.factuality_cache[entity_name].get("key_info", "")
                print(f"  ✅ Batch scoring done, cached factuality for {len(self.factuality_cache)} entities")
        
        # Extract subgraphs for each target (merge all depths into one entry)
        tasks_to_process = []  # [(index, target_entity)]
        skipped_count = 0
        
        for i, target_entity in enumerate(target_entities, 1):
            # Check total limit
            if self.total_subgraphs_generated >= self.total_num_targets:
                print(f"  ℹ️  Total limit reached, stopping")
                break
            
            # Check for duplicates (considering max_num_targets_per_entity)
            target_name = target_entity["canonical_name"]
            key = (target_name, question_hash)
            
            if existing_subgraphs:
                existing_count = existing_subgraphs.get(key, 0)
                # Check if this entity has already been used as target max times in this graph
                if existing_count >= self.max_num_targets_per_entity:
                    print(f"    Skipping subgraph #{i} (target: {target_name}) - max times reached ({existing_count}/{self.max_num_targets_per_entity})")
                    skipped_count += 1
                    continue
            
            tasks_to_process.append((i, target_entity))
        
        if skipped_count > 0:
            print(f"  ⏭️  Skipped {skipped_count} existing subgraphs")
        
        if not tasks_to_process:
            print(f"  ⚠️  No target entities to process")
            return []
        
        print(f"  🚀 Preparing {len(tasks_to_process)} subgraph extraction tasks")
        if self.parallel_subgraphs > 1:
            print(f"  ⚡ Subgraph concurrency: {self.parallel_subgraphs} simultaneous")
        
        # Use Semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.parallel_subgraphs)
        
        async def extract_subgraph_with_semaphore(
            index: int,
            target_entity: Dict[str, Any],
        ) -> Optional[Dict[str, Any]]:
            """Subgraph extraction with semaphore control"""
            async with semaphore:
                # Check total limit again in concurrent context
                if self.total_subgraphs_generated >= self.total_num_targets:
                    return None
                
                merged_subgraph = await self.extract_subgraph(
                    target_entity,
                    entities,
                    relations,
                    graph_record,
                    index,
                    G,
                    entity_map,
                )
                
                # Update count if extraction succeeded
                if merged_subgraph:
                    self.total_subgraphs_generated += 1
                
                return merged_subgraph
        
        # Process all subgraph extraction tasks (concurrent or sequential)
        if self.parallel_subgraphs > 1:
            subgraph_results = await asyncio.gather(*[
                extract_subgraph_with_semaphore(index, target_entity)
                for index, target_entity in tasks_to_process
            ], return_exceptions=True)
        else:
            # Sequential processing (original behavior)
            subgraph_results = []
            for index, target_entity in tasks_to_process:
                try:
                    result = await extract_subgraph_with_semaphore(index, target_entity)
                    subgraph_results.append(result)
                except Exception as e:
                    print(f"  ❌ Subgraph extraction error: {e}")
                    import traceback
                    traceback.print_exc()
                    subgraph_results.append(None)
        
        # Collect successful results
        subgraphs = []
        for result in subgraph_results:
            if result is None or isinstance(result, Exception):
                continue
            subgraphs.append(result)
        
        print(f"  ✅ Successfully extracted {len(subgraphs)} subgraphs (all depths merged per target)")
        print(f"  Total progress: {self.total_subgraphs_generated}/{self.total_num_targets}")
        
        return subgraphs


# ============================================================================
# Main
# ============================================================================

def _default_output_path(input_path: Path) -> Path:
    """Generates the default output path (outputs to cache_5 directory)"""
    return Path("./data_synthesis/cache/cache_5") / f"{input_path.stem}_subgraphs.jsonl"


async def main():
    parser = argparse.ArgumentParser(description="v2 Step 5: Extract Subgraphs")
    parser.add_argument(
        "--input",
        default=None,
        help="Input file path (JSONL format, from 3_construct_graph.py)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data_synthesis/cache/cache_5/<input_stem>_subgraphs.jsonl)",
    )
    parser.add_argument(
        "--total-num-targets",
        type=int,
        default=10000000,
        help="Total number of subgraphs to generate across all graphs (default: 10000000)",
    )
    parser.add_argument(
        "--max-num-targets-per-entity",
        type=int,
        default=1,
        help="Max times an entity can appear as target across subgraphs (default: 1)",
    )
    parser.add_argument(
        "--depth-range",
        type=str,
        default="1,2,3,4,5",
        help="Subgraph depth range, comma-separated (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--min-entities-per-subgraph",
        type=int,
        default=4,
        help="Minimum entities per subgraph (default: 4)",
    )
    parser.add_argument(
        "--max-entities-per-subgraph",
        type=int,
        default=8,
        help="Maximum entities per subgraph (default: 8)",
    )
    parser.add_argument(
        "--min-entities-per-depth",
        type=int,
        default=1,
        help="Minimum entities per depth layer (default: 1)",
    )
    parser.add_argument(
        "--max-entities-per-depth",
        type=int,
        default=5,
        help="Maximum entities per depth layer (default: 5)",
    )
    parser.add_argument(
        "--min-cycle",
        type=int,
        default=0,
        help="Minimum cycle count (parent + two children + child-to-child relation, default: 0)",
    )
    parser.add_argument(
        "--min-entities-with-used-llm",
        type=int,
        default=2,
        help="Minimum number of entities with used_llm=True (default: 2)",
    )
    parser.add_argument(
        "--min-degree-for-entities",
        type=int,
        default=1,
        help="Minimum degree for entities to be included in subgraph (default: 1)",
    )
    parser.add_argument(
        "--use-factuality-filter",
        default=True,
        help="Whether to enable factuality filtering via LLM (default: True)",
    )
    parser.add_argument(
        "--llm-model",
        default="deepseek-v3.2",
        help="LLM model name (default: deepseek-v3.2)",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="LLM API base URL (default: read from OPENAI_BASE_URL env variable)",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="LLM API key (default: read from OPENAI_API_KEY env variable)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start processing from this record index (default: 0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000000,
        help="Maximum number of records to process (default: 10000000)",
    )
    parser.add_argument(
        "--batch-size-for-factuality-scoring",
        type=int,
        default=20,
        help="Number of entities per LLM factuality scoring batch (default: 20)",
    )
    parser.add_argument(
        "--working-dir",
        default="./knowledge_graphs",
        help="LightRAG working directory for graph completeness check (default: ./knowledge_graphs)",
    )
    parser.add_argument(
        "--targets-per-quality-node",
        type=float,
        default=1.0,
        help="Number of subgraphs per high-quality node (default: 1.0)",
    )
    parser.add_argument(
        "--min-quality-score-threshold",
        type=float,
        default=0.5,
        help="Quality score threshold for high-quality nodes (default: 0.5)",
    )
    parser.add_argument(
        "--parallel-subgraphs",
        type=int,
        default=20,
        help="Concurrency for subgraph extraction (default: 20)",
    )
    parser.add_argument(
        "--parallel-factuality-batches",
        type=int,
        default=5,
        help="Concurrency for factuality scoring batches (default: 5)",
    )
    
    args = parser.parse_args()
    
    # Convert use_factuality_filter to bool if string was passed
    if isinstance(args.use_factuality_filter, str):
        args.use_factuality_filter = args.use_factuality_filter.lower() in ('true', '1', 'yes', 'on')
    
    # Path setup
    if not args.input:
        raise ValueError("--input is required")
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _default_output_path(input_path)
    
    # Load existing subgraphs (for resume and deduplication)
    existing_subgraphs = load_existing_subgraphs(output_path)
    if not isinstance(existing_subgraphs, dict):
        existing_subgraphs = {}
    total_existing_count = sum(existing_subgraphs.values())
    
    if existing_subgraphs:
        print(f"📋 Existing output detected: {total_existing_count} subgraphs already processed")
        print(f"   Covering {len(existing_subgraphs)} unique (target_entity, question_hash) pairs")
        print(f"   Existing entries will be skipped (resume support)")
    elif output_path.exists():
        print(f"📋 Output file exists but has no valid records, will continue appending")
    
    # Parse depth range
    try:
        depth_range = [int(d.strip()) for d in args.depth_range.split(",") if d.strip()]
        if not depth_range:
            raise ValueError("Depth range cannot be empty")
        depth_range = sorted(set(depth_range))
    except Exception as e:
        raise ValueError(f"Invalid depth range format: {args.depth_range}, error: {e}")
    
    print(f"✅ Input: {input_path}")
    print(f"✅ Output: {output_path}")
    print(f"✅ Total subgraph target: {args.total_num_targets}")
    print(f"✅ Max target times per entity: {args.max_num_targets_per_entity}")
    print(f"✅ Depth range: {depth_range}")
    print(f"✅ Entities per subgraph: [{args.min_entities_per_subgraph}, {args.max_entities_per_subgraph}]")
    print(f"✅ Entities per depth: [{args.min_entities_per_depth}, {args.max_entities_per_depth}]")
    print(f"✅ Min cycles: {args.min_cycle}")
    print(f"✅ Min used_llm entities: {args.min_entities_with_used_llm}")
    print(f"✅ Min entity degree: {args.min_degree_for_entities}")
    print(f"✅ Factuality filter: {'enabled (LLM direct selection)' if args.use_factuality_filter else 'disabled'}")
    print(f"✅ Working dir: {args.working_dir} (for graph completeness check)")
    print(f"✅ Adaptive subgraph extraction:")
    print(f"   - Subgraphs per quality node: {args.targets_per_quality_node}")
    print(f"   - Quality score threshold: {args.min_quality_score_threshold}")
    if args.use_factuality_filter:
        print(f"   - LLM model: {args.llm_model}")
        print(f"   - API Base URL: {args.llm_base_url}")
        print(f"   - Batch size for scoring: {args.batch_size_for_factuality_scoring} entities/batch")
    print(f"✅ Concurrency config:")
    print(f"   - Subgraph concurrency: {args.parallel_subgraphs}")
    print(f"   - Factuality batch concurrency: {args.parallel_factuality_batches}")
    if args.parallel_subgraphs > 1 or args.parallel_factuality_batches > 1:
        total_concurrent_llm = args.parallel_subgraphs * args.parallel_factuality_batches
        print(f"   - Total LLM concurrency: {total_concurrent_llm} (subgraph × batch)")
    
    # Load data
    print(f"\n📖 Loading input file...")
    records = load_jsonl(input_path)
    print(f"✅ Loaded {len(records)} graph records")
    
    # Determine processing range
    start = args.start
    end = len(records) if args.max_samples is None else min(len(records), start + args.max_samples)
    
    # Create extractor
    extractor = SubgraphExtractor(
        total_num_targets=args.total_num_targets,
        max_num_targets_per_entity=args.max_num_targets_per_entity,
        depth_range=depth_range,
        min_entities_per_subgraph=args.min_entities_per_subgraph,
        max_entities_per_subgraph=args.max_entities_per_subgraph,
        min_entities_per_depth=args.min_entities_per_depth,
        max_entities_per_depth=args.max_entities_per_depth,
        min_cycle=args.min_cycle,
        min_entities_with_used_llm=args.min_entities_with_used_llm,
        min_degree_for_entities=args.min_degree_for_entities,
        use_factuality_filter=args.use_factuality_filter,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        batch_size_for_factuality_scoring=args.batch_size_for_factuality_scoring,
        working_dir=args.working_dir,
        targets_per_quality_node=args.targets_per_quality_node,
        min_quality_score_threshold=args.min_quality_score_threshold,
        parallel_subgraphs=args.parallel_subgraphs,
        parallel_factuality_batches=args.parallel_factuality_batches,
    )
    
    # Update generated count (considering existing)
    extractor.total_subgraphs_generated = total_existing_count
    
    # Restore entity_selection_count from existing records
    entity_selection_count_restored = defaultdict(int)
    for (target_entity, question_hash), count in existing_subgraphs.items():
        entity_selection_count_restored[target_entity] += count
    
    extractor.entity_selection_count.update(entity_selection_count_restored)
    
    if existing_subgraphs:
        print(f"📊 Resuming from {total_existing_count}/{args.total_num_targets} subgraphs")
        if entity_selection_count_restored:
            print(f"   Restored selection counts for {len(entity_selection_count_restored)} entities")

    # Process each graph
    processed_graphs = 0
    
    for i in range(start, end):
        # Check if target count reached
        if extractor.total_subgraphs_generated >= args.total_num_targets:
            print(f"\n✅ Target reached ({args.total_num_targets} subgraphs), stopping")
            break
        
        graph_record = records[i]
        
        subgraphs = await extractor.process_graph(graph_record, i + 1, existing_subgraphs)
        
        # Save subgraphs
        for subgraph in subgraphs:
            append_jsonl(subgraph, output_path)
            # Update processed dict (for deduplication; supports same entity as target multiple times)
            target_entity = subgraph.get("target_entity", "")
            question_hash = subgraph.get("parent_graph", {}).get("question_hash", "")
            if target_entity and question_hash:
                key = (target_entity, question_hash)
                existing_subgraphs[key] = existing_subgraphs.get(key, 0) + 1
        
        processed_graphs += 1
    
    print(f"\n" + "="*60)
    print(f"✅ Processing complete")
    print(f"  - Graphs processed: {processed_graphs}")
    print(f"  - Subgraphs extracted: {extractor.total_subgraphs_generated}")
    print(f"  - Target count: {args.total_num_targets}")
    if processed_graphs > 0:
        print(f"  - Average per graph: {extractor.total_subgraphs_generated / processed_graphs:.2f} subgraphs")
    print(f"  - Output file: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
