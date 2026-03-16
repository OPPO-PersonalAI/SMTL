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
Layer-wise Description Generator V2 (based on graph topology)

Starts from leaf nodes and generates descriptions layer by layer toward the root node.

V2 changes:
- Leaf nodes use entity name + description; no LLM generation or obfuscation.
- Non-leaf nodes undergo normal LLM description generation and obfuscation.
- Concurrent processing: each graph uses an independent GraphProcessingContext to avoid state conflicts.
"""

import os
import asyncio
import random
import copy
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from openai import AsyncOpenAI
from dotenv import load_dotenv
from json_repair import repair_json

# Load .env file
load_dotenv()

# Disable tokenizers parallelism to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import read_jsonl, write_jsonl, safe_json_loads


class GraphProcessingContext:
    """
    Processing context for a single graph; encapsulates all graph-related state.
    Used to support concurrent processing and avoid state conflicts between graphs.
    """
    def __init__(self):
        # Graph data attributes
        self.graph_data: Optional[Dict[str, Any]] = None
        self.target_entity: Optional[str] = None
        
        # Topology (obtained strictly from the topology field)
        self.layer_entities: Dict[int, List[str]] = {}  # depth -> [entity_names]
        self.entity_map: Dict[str, Dict[str, Any]] = {}  # entity_name -> entity_info
        
        # Relation data
        self.parent_child_edges: List[Dict[str, Any]] = []  # inter-layer relations
        self.intra_layer_edges: List[Dict[str, Any]] = []   # intra-layer relations
        self.relations: List[Dict[str, Any]] = []           # raw relation data
        
        # Child node mapping: parent_entity -> [(child_entity, relation_info)]
        self.children_map: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        
        # Peer relation mapping: entity -> [(peer_entity, relation_info)]
        self.peer_map: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        
        # Processing status flag: entity_name -> bool (marks whether a node has been processed)
        self.description_cache: Dict[str, bool] = {}
        
        # Question and facts trace: records question, facts, etc. generated for each processed node
        self.description_trace: List[Dict[str, Any]] = []
        
        # Global code mapping: entity_name -> code (assigned uniformly for the whole subgraph)
        self.entity_codes: Dict[str, str] = {}


def append_jsonl(record: Dict[str, Any], output_path: Path) -> None:
    """Append a single record to a JSONL file (thread-safe)."""
    import fcntl
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # file lock to protect concurrent writes
        try:
            import json
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_processed_targets(output_path: Path) -> Set[str]:
    """
    Load the set of already-processed target_entity values (for resume support).

    Processed subgraphs (by target_entity) will be skipped in subsequent runs.

    Args:
        output_path (Path): Path to the output file.

    Returns:
        Set[str]: Set of already-processed target_entity values.
    """
    if not output_path.exists():
        return set()
    
    processed_targets = set()
    try:
        records = read_jsonl(str(output_path))
        for record in records:
            target_entity = record.get('target_entity', '')
            if target_entity:
                processed_targets.add(target_entity)
        print(f"📁 Resume: loaded {len(processed_targets)} already-processed subgraphs (target_entity) from output file.")
    except Exception as e:
        print(f"⚠️ Failed to load processed results: {e}")
        print(f"   Will process all subgraphs from the beginning.")
    
    return processed_targets


class LayerwiseDescriptionBuilder:
    """Layer-wise description generator based on graph topology (supports concurrent processing)."""
    
    def __init__(self,
                 output_file: str,
                 generate_model: str = "deepseek-v3.2",
                 verify_model: str = "gpt-5-mini",
                 max_obfuscation_iterations: int = 5):
        """
        Initialize the layer-wise description builder.

        Args:
            output_file (str): Output file path.
            generate_model (str): LLM model for description generation (default: deepseek-v3.2).
            verify_model (str): LLM model for verification and obfuscation (default: gpt-5-mini).
            max_obfuscation_iterations (int): Maximum obfuscation iterations (default: 5).
        """
        self.output_file = Path(output_file)
        self.generate_model = generate_model
        self.verify_model = verify_model
        self.max_obfuscation_iterations = max_obfuscation_iterations
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        
        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Note: graph-related state is NOT stored on the instance; instead, an independent
        # GraphProcessingContext is created for each graph to support concurrent processing.
    
    async def _call_llm_with_retry(self,
                                    model: str,
                                    messages: List[Dict[str, Any]],
                                    max_tokens: int = 2048,
                                    response_format: Optional[Dict[str, Any]] = None,
                                    operation_name: str = "LLM call",
                                    max_retries: int = 3) -> Optional[Any]:
        """
        LLM call wrapper with retry logic.

        Args:
            model (str): Model name.
            messages (List[Dict[str, Any]]): Message list.
            max_tokens (int): Maximum tokens.
            response_format (Optional[Dict[str, Any]]): Response format.
            operation_name (str): Operation name (for logging).
            max_retries (int): Maximum retry attempts (default: 3).

        Returns:
            Optional[Any]: LLM response object, or None on failure.
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                kwargs = {
                    'model': model,
                    'messages': messages,
                    'max_tokens': max_tokens
                }
                if response_format:
                    kwargs['response_format'] = response_format
                
                response = await self.client.chat.completions.create(**kwargs)
                return response
                
            except ValueError as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"    ⚠️ {operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"    🔄 Retrying...")
                    await asyncio.sleep(1)  # wait 1 second before retrying
                else:
                    print(f"    ❌ {operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"    💥 Max retries reached, giving up.")
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"    ⚠️ {operation_name} exception (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"    🔄 Retrying...")
                    await asyncio.sleep(1)
                else:
                    print(f"    ❌ {operation_name} exception (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"    💥 Max retries reached, giving up.")
        
        # All retries failed
        print(f"    ❌ {operation_name} ultimately failed, returning None.")
        if last_error:
            print(f"    Last error: {last_error}")
        return None
    
    def load_graph_with_topology(self, ctx: GraphProcessingContext, graph_data: Dict[str, Any]) -> bool:
        """
        Load graph data and topology (strict mode, no error tolerance).

        Args:
            ctx (GraphProcessingContext): Graph processing context.
            graph_data (Dict[str, Any]): Graph data dictionary.

        Returns:
            bool: True if loaded successfully.
        """
        try:
            ctx.graph_data = graph_data
            
            # target_entity must exist
            ctx.target_entity = ctx.graph_data.get('target_entity')
            if not ctx.target_entity:
                raise ValueError("'target_entity' field not found.")
            
            # topology must exist
            topology = ctx.graph_data.get('topology')
            if not topology:
                raise ValueError("'topology' field not found.")
            
            # layer_entities must exist
            layer_entities_raw = topology.get('layer_entities')
            if not layer_entities_raw:
                raise ValueError("'layer_entities' not found in topology.")
            
            # Parse layer_entities, strict format check
            for depth_str, entity_names in layer_entities_raw.items():
                try:
                    depth = int(depth_str)
                except ValueError:
                    raise ValueError(f"depth in layer_entities is not a valid integer: {depth_str}")
                
                if not isinstance(entity_names, list):
                    raise ValueError(f"layer_entities[{depth}] is not a list.")
                
                if not entity_names:
                    raise ValueError(f"layer_entities[{depth}] is an empty list.")
                
                ctx.layer_entities[depth] = entity_names
            
            # Check target_entity is at depth 0
            if 0 not in ctx.layer_entities:
                raise ValueError("layer_entities is missing depth 0.")
            
            if ctx.target_entity not in ctx.layer_entities[0]:
                raise ValueError(f"target_entity '{ctx.target_entity}' not in layer_entities[0].")
            
            # Build entity mapping
            entities = ctx.graph_data.get('entities', [])
            if not entities:
                raise ValueError("'entities' field not found or empty.")
            
            for entity in entities:
                canonical_name = entity.get('canonical_name', '')
                if not canonical_name:
                    raise ValueError("Entity is missing 'canonical_name' field.")
                ctx.entity_map[canonical_name] = entity
            
            # Check all entities in layer_entities exist in entity_map
            for depth, entity_names in ctx.layer_entities.items():
                for entity_name in entity_names:
                    if entity_name not in ctx.entity_map:
                        raise ValueError(f"Entity '{entity_name}' in layer_entities[{depth}] not found in entities.")
            
            # Edge info must exist
            ctx.parent_child_edges = topology.get('parent_child_edges')
            if ctx.parent_child_edges is None:
                raise ValueError("'parent_child_edges' not found in topology.")
            
            ctx.intra_layer_edges = topology.get('intra_layer_edges')
            if ctx.intra_layer_edges is None:
                raise ValueError("'intra_layer_edges' not found in topology.")
            
            # Get raw relation data
            ctx.relations = ctx.graph_data.get('relations', [])
            
            print(f"✅ Graph structure loaded successfully (strict mode).")
            print(f"  Target entity: {ctx.target_entity}")
            print(f"  Entity count: {len(ctx.entity_map)}")
            print(f"  Layers: {len(ctx.layer_entities)}")
            print(f"  Depth range: {min(ctx.layer_entities.keys())} - {max(ctx.layer_entities.keys())}")
            print(f"  Inter-layer edges: {len(ctx.parent_child_edges)}")
            print(f"  Intra-layer edges: {len(ctx.intra_layer_edges)}")
            
            return True

        except Exception as e:
            print(f"❌ Failed to load graph structure (strict mode): {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_relation_info_from_relations(self, ctx: GraphProcessingContext, source_entity: str, target_entity: str) -> Dict[str, Any]:
        """
        Retrieve relation info from raw relations data by source_entity and target_entity.

        Args:
            ctx (GraphProcessingContext): Graph processing context.
            source_entity (str): Source entity name.
            target_entity (str): Target entity name.

        Returns:
            Dict[str, Any]: Relation info, including description, etc.
        """
        for relation in ctx.relations:
            if (relation.get('source_entity') == source_entity and 
                relation.get('target_entity') == target_entity):
                return {
                    'relation_id': relation.get('relation_id', ''),
                    'relationship_type': relation.get('relationship_type', ''),
                    'description': relation.get('description', ''),
                    'source_urls': relation.get('source_urls', []),
                    'evidence_spans': relation.get('evidence_spans', []),
                    'importance_score': relation.get('importance_score', 0.0),
                    'relation_aug': relation.get('relation_aug', 0)
                }
        return {}
    
    def _get_entity_depth(self, ctx: GraphProcessingContext, entity_name: str) -> Optional[int]:
        """
        Get the depth of an entity from layer_entities.

        Args:
            ctx (GraphProcessingContext): Graph processing context.
            entity_name (str): Entity name.

        Returns:
            Optional[int]: Entity depth, or None if not found.
        """
        for depth, entities in ctx.layer_entities.items():
            if entity_name in entities:
                return depth
        return None
    
    def _build_parent_child_mapping(self, ctx: GraphProcessingContext) -> bool:
        """
        Build parent-child relation mapping.

        Args:
            ctx (GraphProcessingContext): Graph processing context.

        Rules:
        - Parent-child relationships are determined by depth in layer_entities, independent of edge direction.
        - Nodes with smaller depth are parents; nodes with larger depth are children.

        Returns:
            bool: True if built successfully.
        """
        if not ctx.target_entity:
            print(f"❌ target_entity not set.")
            return False
        
        # Clear mappings
        ctx.children_map.clear()
        ctx.peer_map.clear()
        
        # Build parent-child mapping
        for edge in ctx.parent_child_edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            
            if not source or not target:
                print(f"⚠️ Skipping invalid parent_child_edge: {edge}")
                continue
            
            # Get depth from layer_entities (do not rely on edge's depth field)
            source_depth = self._get_entity_depth(ctx, source)
            target_depth = self._get_entity_depth(ctx, target)
            
            if source_depth is None:
                raise ValueError(f"Entity '{source}' not found in layer_entities.")
            if target_depth is None:
                raise ValueError(f"Entity '{target}' not found in layer_entities.")
            
            # Determine parent-child: smaller depth = parent, larger depth = child
            if source_depth < target_depth:
                parent = source
                child = target
                parent_depth = source_depth
                child_depth = target_depth
            elif source_depth > target_depth:
                parent = target
                child = source
                parent_depth = target_depth
                child_depth = source_depth
            else:
                # Same depth should not appear in parent_child_edges
                print(f"⚠️ Same-depth edge in parent_child_edges: {edge} (depth={source_depth})")
                continue
            
            # Get relation info from raw relations (try both directions)
            relation_info = self._get_relation_info_from_relations(ctx, parent, child)
            if not relation_info:
                relation_info = self._get_relation_info_from_relations(ctx, child, parent)
                if not relation_info:
                    # Fall back to edge info
                    relation_info = {
                        'relationship_type': edge.get('relation', ''),
                        'description': edge.get('description', ''),
                    }
            
            # Add depth info
            relation_info['parent_depth'] = parent_depth
            relation_info['child_depth'] = child_depth
            
            # Add to children_map
            if (child, relation_info) not in ctx.children_map[parent]:
                ctx.children_map[parent].append((child, relation_info))
        
        # Build intra-layer relation mapping
        for edge in ctx.intra_layer_edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            
            if not source or not target:
                print(f"⚠️ Skipping invalid intra_layer_edge: {edge}")
                continue
            
            # Get depth from layer_entities (do not rely on edge's depth field)
            source_depth = self._get_entity_depth(ctx, source)
            target_depth = self._get_entity_depth(ctx, target)
            
            if source_depth is None:
                raise ValueError(f"Entity '{source}' not found in layer_entities.")
            if target_depth is None:
                raise ValueError(f"Entity '{target}' not found in layer_entities.")
            
            # Validate same-layer relationship
            if source_depth != target_depth:
                print(f"⚠️ Nodes in intra_layer_edge are not in the same layer: {edge} (source_depth={source_depth}, target_depth={target_depth})")
                continue
            
            depth = source_depth  # use depth from layer_entities
            
            # Get source->target relation info from raw relations
            relation_info_source_to_target = self._get_relation_info_from_relations(ctx, source, target)
            if not relation_info_source_to_target:
                relation_info_source_to_target = {
                    'relationship_type': edge.get('relation', ''),
                    'description': edge.get('description', ''),
                    'depth': depth
                }
            else:
                relation_info_source_to_target['depth'] = depth
            
            # Get target->source relation info (reverse direction)
            relation_info_target_to_source = self._get_relation_info_from_relations(ctx, target, source)
            if not relation_info_target_to_source:
                relation_info_target_to_source = relation_info_source_to_target.copy()
            else:
                relation_info_target_to_source['depth'] = depth
            
            # source->target direction
            if target not in [peer for peer, _ in ctx.peer_map[source]]:
                ctx.peer_map[source].append((target, relation_info_source_to_target))
            
            # target->source direction (bidirectional)
            if source not in [peer for peer, _ in ctx.peer_map[target]]:
                ctx.peer_map[target].append((source, relation_info_target_to_source))
        
        print(f"✅ Parent-child mapping built successfully.")
        print(f"  Nodes with children: {len(ctx.children_map)}")
        print(f"  Nodes with peer relations: {len(ctx.peer_map)}")
        
        return True
    
    def _assign_all_entity_codes(self, ctx: GraphProcessingContext):
        """
        Uniformly assign codes (entity_type + index) to all entities in the subgraph.
        This ensures each entity has a unique, consistent code throughout processing.

        Args:
            ctx (GraphProcessingContext): Graph processing context.
        """
        ctx.entity_codes = {}
        type_counter = {}  # entity_type -> counter
        
        # Collect all entities (from deepest to shallowest)
        all_entities = []
        for depth in sorted(ctx.layer_entities.keys(), reverse=True):  # leaf to root
            for entity_name in ctx.layer_entities[depth]:
                if entity_name not in all_entities:
                    all_entities.append(entity_name)
        
        # Assign code to each entity
        for entity_name in all_entities:
            entity_info = ctx.entity_map.get(entity_name, {})
            entity_type = entity_info.get('entity_type', 'Entity')
            
            # Assign index per type
            if entity_type not in type_counter:
                type_counter[entity_type] = 0
            
            # Generate code: letters A, B, C...
            if type_counter[entity_type] < 26:
                index = chr(ord('A') + type_counter[entity_type])
            else:
                # For more than 26, use AA, AB, AC...
                first = chr(ord('A') + (type_counter[entity_type] // 26) - 1)
                second = chr(ord('A') + (type_counter[entity_type] % 26))
                index = f"{first}{second}"
            
            code = f"{entity_type} {index}"
            ctx.entity_codes[entity_name] = code
            type_counter[entity_type] += 1
        
        print(f"Assigned {len(ctx.entity_codes)} entity codes for the subgraph.")

    async def _verify_description(self, description: str, entity_name: str, entity_type: str) -> Dict[str, Any]:
        """
        Verify whether a description allows the model to directly identify the entity.

        Args:
            description (str): Entity description.
            entity_name (str): Entity name (ground truth answer).
            entity_type (str): Entity type.

        Returns:
            Dict[str, Any]: Verification result containing:
                - can_answer (bool): Whether the model can directly answer.
                - predicted_answer (str): Model's predicted answer.
                - reasoning (str): Model's reasoning basis.
                - is_correct (bool): Whether the prediction is correct.
                - verification_details (str): Verification details.
        """
        # First call: ask the model to answer and provide its reasoning
        question = f"{description}. What is this {entity_type}?"
        
        answer_prompt = f"""Please answer the question based on the following description.

Description: {description}
Question: What is this {entity_type}?

Please output your answer in JSON format:
{{
  "answer": "Your answer (entity name)",
  "reasoning": "Brief reasoning (explain which conditions you used to infer the answer, list key conditions)"
}}

Note: You must return a specific answer, not a concept or category. It must be a concrete entity name. If you cannot determine the answer, return "Cannot determine" in the answer field.
Output only JSON, do not add any other text."""
        
        try:
            answer_response = await self._call_llm_with_retry(
                model=self.verify_model,
                messages=[
                    {"role": "system", "content": "You are a question-answering expert who needs to answer questions based on descriptions and provide reasoning."},
                    {"role": "user", "content": answer_prompt}
                ],
                max_tokens=4096,
                response_format={"type": "json_object"},
                operation_name=f"verify-answer ({entity_name})"
            )
            
            if answer_response is None:
                raise ValueError(f"verify-answer failed: LLM call failed.")
            
            answer_output = answer_response.choices[0].message.content.strip()
            
            # Check if output is empty
            if not answer_output:
                raise ValueError(f"verify-answer returned empty output.")
            
            # Parse JSON
            try:
                repaired_json = repair_json(answer_output)
                answer_data = json.loads(repaired_json)
            except Exception as json_error:
                raise ValueError(f"verify-answer JSON parse failed: {json_error}\nRaw output: {answer_output[:200]}") from json_error
            
            predicted_answer = answer_data.get('answer', '').strip()
            reasoning = answer_data.get('reasoning', '').strip()
            
            # Second call: verify whether the predicted answer matches the ground truth
            verify_prompt = f"""Please determine whether the following two answers refer to the same entity.

Standard Answer: {entity_name}
Predicted Answer: {predicted_answer}

Please output your judgment in JSON format:
{{
  "is_correct": true/false,
  "explanation": "Brief explanation (why you think they match or do not match)"
}}

Judgment Criteria:
- Return true if both answers refer to the same entity (even if expressed differently)
- Return false if the predicted answer is "Cannot determine" or clearly incorrect
- The predicted answer must be very precise and explicitly point to the standard answer. It cannot be a concept or category; it must be a specific entity name. Otherwise, return false
- Consider aliases, abbreviations, and similar cases

Output only JSON, do not add any other text."""
            
            verify_response = await self._call_llm_with_retry(
                model=self.verify_model,
                messages=[
                    {"role": "system", "content": "You are an answer verification expert."},
                    {"role": "user", "content": verify_prompt}
                ],
                max_tokens=4096,
                response_format={"type": "json_object"},
                operation_name=f"verify-consistency ({entity_name})"
            )
            
            if verify_response is None:
                raise ValueError(f"verify-consistency failed: LLM call failed.")
            
            verify_output = verify_response.choices[0].message.content.strip()
            
            # Check if output is empty
            if not verify_output:
                raise ValueError(f"verify-consistency returned empty output.")
            
            # Parse JSON
            try:
                repaired_json = repair_json(verify_output)
                verify_data = json.loads(repaired_json)
            except Exception as json_error:
                raise ValueError(f"verify-consistency JSON parse failed: {json_error}\nRaw output: {verify_output[:200]}") from json_error
            
            is_correct = verify_data.get('is_correct', False)
            verification_details = verify_data.get('explanation', '')
            
            return {
                'can_answer': predicted_answer.lower() != "cannot determine",
                'predicted_answer': predicted_answer,
                'reasoning': reasoning,
                'is_correct': is_correct,
                'verification_details': verification_details,
                'question': question,
                'answer_raw_output': answer_output,
                'verify_raw_output': verify_output
            }
            
        except Exception as e:
            print(f"⚠️ Verification error: {e}")
            return {
                'can_answer': False,
                'predicted_answer': '',
                'reasoning': '',
                'is_correct': False,
                'verification_details': f'Verification error: {str(e)}',
                'question': question,
                'error': str(e)
            }
    
    async def _obfuscate_description(self, 
                                     description: str, 
                                     facts: List[Dict[str, Any]], 
                                     entity_name: str,
                                     entity_type: str,
                                     entity_info: Dict[str, Any],
                                     reasoning: str) -> Dict[str, Any]:
        """
        Obfuscate a description based on verification results so the model cannot answer directly.

        Args:
            description (str): Original description.
            facts (List[Dict[str, Any]]): Facts list (with source info).
            entity_name (str): Entity name.
            entity_type (str): Entity type.
            entity_info (Dict[str, Any]): Entity details (includes key_attributes, etc.).
            reasoning (str): Reasoning that allows the model to answer directly.

        Returns:
            Dict[str, Any]: Obfuscation result containing:
                - obfuscated_description (str): Obfuscated description.
                - obfuscated_facts (List[Dict]): Obfuscated facts list.
                - obfuscation_strategy (str): Obfuscation strategy used.
                - target_fact (Dict): The fact that was obfuscated.
        """
        # Prepare entity details
        key_attributes = entity_info.get('key_attributes', {})
        entity_description = entity_info.get('description', '')
        surface_forms = entity_info.get('surface_forms', [])
        entity_name_field = entity_info.get('name', entity_name)
        
        # Build facts detail info
        facts_detail_lines = []
        for idx, fact in enumerate(facts):
            fact_text = fact.get('fact', '')
            fact_source = fact.get('source', '')
            facts_detail_lines.append(f"Condition {idx+1}: {fact_text}")
            facts_detail_lines.append(f"  Source: {fact_source}")
        
        facts_detail_text = "\n".join(facts_detail_lines)
        
        # Prepare key_attributes info
        key_attrs_text = ""
        if key_attributes:
            attrs_lines = []
            for key, value in key_attributes.items():
                attrs_lines.append(f"- {key}: {value}")
            key_attrs_text = "\nKey Attributes:\n" + "\n".join(attrs_lines)
        
        # Prepare surface_forms info
        surface_forms_text = ""
        if surface_forms:
            surface_forms_text = f"\nAliases: {', '.join(surface_forms[:10])}"  # show at most 10
        
        obfuscate_prompt = f"""You are an expert responsible for obfuscating entity descriptions so that answers can only be determined through search.

Current Situation:
- Entity Type: {entity_type}
- Entity Name: {entity_name} (This is the standard answer and must never appear in the obfuscated description)
- Original Description: {description}

Conditions in the Description:
{facts_detail_text}

Reasoning Basis That Allows Direct Answering:
{reasoning}

Entity Detailed Information:
- Entity Name: {entity_name_field}
- Complete Description: {entity_description}
{key_attrs_text}{surface_forms_text}

🚫 **Strictly Prohibited (The obfuscated question must not contain)**:
1. ❌ Do not use prompt-like language such as "Related Information 0", "Related Information 1", etc.
2. ❌ Do not directly mention the entity name "{entity_name}" or any other specific entity names
3. ❌ Do not use expressions that may hint at the answer, such as "this {entity_type}", "the aforementioned {entity_type}", etc.

✅ **Must Use Natural Language Expressions**:
- Use generic terms: "an organization", "a company", "a country", "a writer"
- Use characteristic descriptions: "a media company founded in XX year", "a country located in East Africa"
- Use relational descriptions: "the company where one worked", "the country where one was born"

Task: Select the most critical condition to obfuscate so that the model cannot directly infer the answer and must verify through search.

Obfuscation Strategies (Please select based on condition type):

1. **Temporal Obfuscation**:
   - Convert precise time to vague time periods to increase ambiguity; or add descriptions requiring factual search, where the relevant fact appears in the entity's related information
   - Example: "Founded in 1949" → "Founded in the late 1940s"
   - Example: "Born on January 6, 1967" → "Born in a year when the Cultural Revolution was ongoing, with a specific event occurring on that day"

2. **Entity Obfuscation**:
   - Replace entity names with searchable characteristics; or add descriptions requiring search, where the relevant fact appears in the entity's related information
   - Example: "Founded by Steve Jobs" → "Founded by a Silicon Valley entrepreneur whose name starts with 'S'"
   - Example: "Harvard University professor" → "A professor at a U.S. university that was ranked first in world university rankings in year XX"

3. **Quantitative Information Obfuscation**:
   - Convert precise numbers to qualitative or range descriptions; or add descriptions requiring search, where the relevant fact appears in the entity's related information
   - Example: "Production of 2500 tons" → "Production between 2000-3000 tons"
   - Example: "Won two Nobel Prizes" → "Belongs to scientists who have won the Nobel Prize more than once"

4. **Information Deletion**:
   - If conditions are redundant or excessive, making it very easy to infer the answer, prioritize deleting such conditions
   - Example: "An Asian country founded in 1949" → "An Asian country" (because many Asian countries were founded in 1949, making it easy to infer the answer)

5. **Concept Obfuscation**:
   - Convert concepts to more abstract, higher-level concepts, referencing related searchable facts
   - Example: "Beihang University" → "A university with at least XX subjects rated A+ in discipline evaluation" (the entity has appeared with related, searchable factual descriptions)

6. **Combined Obfuscation**:
   - Apply mild obfuscation to multiple conditions simultaneously
   - Maintain overall comprehensibility

Selection Criteria:
- Prioritize obfuscating key conditions mentioned in the model's reasoning basis
- Choose conditions that most reduce certainty
- Maintain description comprehensibility and logicality
- Ensure the obfuscated description can still be found through search
- Ensure obfuscation results are objective and fair, without subjectivity. Descriptions like "usually", "generally" must not appear

Please output in JSON format:
{{
  "target_fact_index": 0,  // Index of the condition selected for obfuscation (starting from 0)
  "obfuscation_strategy": "Temporal Obfuscation/Entity Obfuscation/Quantitative Information Obfuscation/Information Deletion/Concept Obfuscation/Combined Obfuscation",
  "obfuscated_fact": "Obfuscated condition text",
  "obfuscated_description": "Complete obfuscated description",
  "reasoning": "Why this condition was selected for obfuscation and how it was obfuscated"
}}

Output only JSON, do not add any other text."""
        
        try:
            obfuscate_response = await self._call_llm_with_retry(
                model=self.verify_model,
                messages=[
                    {"role": "system", "content": "You are a description obfuscation expert skilled at converting deterministic descriptions into descriptions that require search verification."},
                    {"role": "user", "content": obfuscate_prompt}
                ],
                max_tokens=4096,
                response_format={"type": "json_object"},
                operation_name=f"obfuscate-description ({entity_name})"
            )
            
            if obfuscate_response is None:
                raise ValueError(f"obfuscate-description failed: LLM call failed.")
            
            obfuscate_output = obfuscate_response.choices[0].message.content.strip()
            
            # Check if output is empty
            if not obfuscate_output:
                raise ValueError(f"obfuscate-description returned empty output.")
            
            # Parse JSON
            try:
                repaired_json = repair_json(obfuscate_output)
                obfuscate_data = json.loads(repaired_json)
            except Exception as json_error:
                raise ValueError(f"obfuscate-description JSON parse failed: {json_error}\nRaw output: {obfuscate_output[:200]}") from json_error
            
            target_fact_index = obfuscate_data.get('target_fact_index', 0)
            obfuscation_strategy = obfuscate_data.get('obfuscation_strategy', '')
            obfuscated_fact = obfuscate_data.get('obfuscated_fact', '')
            obfuscated_description = obfuscate_data.get('obfuscated_description', '')
            obfuscation_reasoning = obfuscate_data.get('reasoning', '')
            
            # Update facts list
            obfuscated_facts = facts.copy()
            if 0 <= target_fact_index < len(obfuscated_facts):
                obfuscated_facts[target_fact_index] = {
                    **obfuscated_facts[target_fact_index],
                    'fact': obfuscated_fact,
                    'obfuscated': True
                }
            
            return {
                'obfuscated_description': obfuscated_description,
                'obfuscated_facts': obfuscated_facts,
                'obfuscation_strategy': obfuscation_strategy,
                'target_fact_index': target_fact_index,
                'target_fact': facts[target_fact_index] if 0 <= target_fact_index < len(facts) else {},
                'obfuscation_reasoning': obfuscation_reasoning,
                'raw_output': obfuscate_output
            }
            
        except Exception as e:
            print(f"⚠️ Obfuscation error: {e}")
            # If obfuscation fails, return the original description
            return {
                'obfuscated_description': description,
                'obfuscated_facts': facts,
                'obfuscation_strategy': 'none',
                'target_fact_index': -1,
                'target_fact': {},
                'obfuscation_reasoning': f'Obfuscation failed: {str(e)}',
                'error': str(e)
            }
    
    async def _iterative_verify_and_obfuscate(self,
                                              question: str,
                                              facts: List[Dict[str, Any]],
                                              entity_name: str,
                                              entity_type: str,
                                              entity_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Iteratively verify and obfuscate until the model can no longer answer directly.

        Args:
            question (str): Initial question.
            facts (List[Dict[str, Any]]): Facts list.
            entity_name (str): Entity name.
            entity_type (str): Entity type.
            entity_info (Dict[str, Any]): Entity details.

        Returns:
            Dict[str, Any]: Contains final question and all iteration records.
        """
        current_question = question
        current_facts = facts
        iterations = []
        max_iterations = self.max_obfuscation_iterations
        
        for iteration in range(max_iterations):
            print(f"       🔍 Iteration {iteration + 1}/{max_iterations}: verifying...")
            
            # Verify current question (check whether the answer can be directly inferred)
            verify_result = await self._verify_description(
                current_question,
                entity_name,
                entity_type
            )
            
            iteration_record = {
                'iteration': iteration + 1,
                'question': current_question,
                'facts': current_facts.copy(),
                'verification': verify_result
            }
            
            # If the model cannot answer or answers incorrectly, obfuscation succeeded
            if not verify_result['is_correct']:
                print(f"       ✅ Verification passed: model cannot answer directly.")
                iteration_record['obfuscation'] = None
                iteration_record['success'] = True
                iterations.append(iteration_record)
                break
            
            # If the model answers correctly, continue obfuscation
            print(f"       ⚠️  Model can answer directly; continuing obfuscation...")
            
            obfuscate_result = await self._obfuscate_description(
                current_question,
                current_facts,
                entity_name,
                entity_type,
                entity_info,
                verify_result['reasoning']
            )
            
            iteration_record['obfuscation'] = obfuscate_result
            iteration_record['success'] = False
            iterations.append(iteration_record)
            
            # Update current question and facts
            current_question = obfuscate_result['obfuscated_description']
            current_facts = obfuscate_result['obfuscated_facts']
            
            print(f"    📝 Obfuscation strategy: {obfuscate_result['obfuscation_strategy']}")
        
        # If max iterations reached without success, use the last obfuscated result
        if iterations and not iterations[-1].get('success', False):
            print(f"    ⚠️ Max iterations reached ({max_iterations}); using last obfuscation result.")
        
        return {
            'final_question': current_question,
            'final_facts': current_facts,
            'iterations': iterations,
            'total_iterations': len(iterations),
            'obfuscation_successful': iterations[-1].get('success', False) if iterations else False
        }
    
    def _build_topology_output(self, ctx: GraphProcessingContext) -> Dict[str, Any]:
        """
        Organize output info for all nodes according to topology structure.

        Args:
            ctx (GraphProcessingContext): Graph processing context.

        Returns:
            Dict[str, Any]: Output organized by topology.
        """
        # Get all depths, sorted ascending (root to leaf)
        depths = sorted(ctx.layer_entities.keys())
        
        topology_layers = []
        
        for depth in depths:
            entities_at_depth = ctx.layer_entities[depth]
            
            layer_info = {
                'depth': depth,
                'is_root': (depth == min(depths)),
                'is_leaf': (depth == max(depths)),
                'entity_count': len(entities_at_depth),
                'entities': []
            }
            
            for entity_name in entities_at_depth:
                # Get full info for this node from trace
                entity_output = None
                for trace_entry in ctx.description_trace:
                    if trace_entry.get('node') == entity_name:
                        entity_info = ctx.entity_map.get(entity_name, {})
                        
                        # Build node output (description field removed; only core question and facts kept)
                        entity_output = {
                            'entity_name': entity_name,
                            'entity_type': entity_info.get('entity_type', 'Unknown'),
                            'node_type': trace_entry.get('node_type', 'unknown'),
                            'question': trace_entry.get('final_question') or trace_entry.get('question', ''),
                            'facts': trace_entry.get('final_facts') or trace_entry.get('facts', []),
                            'obfuscation_info': {
                                'skipped': trace_entry.get('obfuscation_skipped', False),
                                'count': trace_entry.get('obfuscation_count', 0),    # total obfuscation count
                                'iterations': trace_entry.get('obfuscation_iterations', []),  # detailed iteration records
                                'successful': trace_entry.get('obfuscation_successful', None)
                            }
                        }
                        
                        # For non-leaf nodes, add children info
                        if trace_entry.get('node_type') == 'non-leaf':
                            entity_output['children_nodes'] = trace_entry.get('children_nodes', [])
                            entity_output['llm_summary'] = trace_entry.get('llm_summary', '')
                        
                        break
                
                if entity_output:
                    layer_info['entities'].append(entity_output)
            
            topology_layers.append(layer_info)
        
        return {
            'total_layers': len(topology_layers),
            'depth_range': {
                'min': min(depths),
                'max': max(depths)
            },
            'layers': topology_layers
        }
    
    def _get_non_leaf_node_description_prompt(self, ctx: GraphProcessingContext, node: str, node_info: Dict[str, Any],
                                              children_descriptions: List[Tuple[str, Dict[str, Any]]],
                                              children_peers_info: List[Dict[str, Any]]) -> str:
        """
        Generate the prompt for a non-leaf node description (summary part only).

        Args:
            node (str): Node name.
            node_info (Dict[str, Any]): Node info.
            children_descriptions: List of (child_name, relation_info) tuples.
            children_peers_info: Peer relation info for child nodes.

        Returns:
            str: Prompt text.
        """
        entity_type = node_info.get('entity_type', 'Entity')
        canonical_name = node_info.get('canonical_name', node)
        node_description = node_info.get('description', '')
        node_key_attributes = node_info.get('key_attributes', {})
        node_surface_forms = node_info.get('surface_forms', [])
        
        # Format children info (structured, each entry includes its facts)
        children_info_lines = []
        children_facts_map = {}  # child_name -> facts list
        
        for idx, (child_name, relation_info) in enumerate(children_descriptions):
            child_entity = ctx.entity_map.get(child_name, {})
            child_type = child_entity.get('entity_type', 'Entity')
            child_description = child_entity.get('description', '')
            child_key_attributes = child_entity.get('key_attributes', {})
            child_surface_forms = child_entity.get('surface_forms', [])
            child_name_field = child_entity.get('name', child_name)  # use name field; fall back to canonical_name
            
            relation = relation_info.get('relationship_type', '')
            rel_desc = relation_info.get('description', '')
            
            # Get child node's facts from trace
            child_facts = []
            for trace_entry in reversed(ctx.description_trace):
                if trace_entry.get('node') == child_name:
                    child_facts = trace_entry.get('facts', [])
                    break
            
            children_facts_map[child_name] = child_facts
            
            # Use abstract description; do not expose entity name directly
            line = f"  - Related Information {idx}: A {child_type}"
            
            # Add entity details (helps LLM understand the entity better)
            entity_details = []
            if child_description:
                entity_details.append(f"Entity Description: {child_description}")
            if child_key_attributes:
                attrs_text = ", ".join([f"{k}: {v}" for k, v in child_key_attributes.items()])
                if attrs_text:
                    entity_details.append(f"Key Attributes: {attrs_text}")
            if child_surface_forms:
                surface_text = ", ".join(child_surface_forms[:5])  # show at most 5 surface_forms
                if surface_text:
                    entity_details.append(f"Aliases: {surface_text}")
            
            if entity_details:
                line += f"\n    - Entity Information: {'; '.join(entity_details)}"
            
            # Add relation info
            if relation:
                line += f"\n    - Relationship Type: {relation}"
                if rel_desc:
                    line += f"\n    - Relationship Description: {rel_desc}"
            
            # Add facts list (propagated iteratively)
            if child_facts:
                line += f"\n    - facts: ["
                fact_texts = []
                for fact_item in child_facts:
                    fact_text = fact_item.get('fact', '')
                    fact_texts.append(f'"{fact_text}"')
                line += ", ".join(fact_texts)
                line += "]"
            else:
                line += f"\n    - facts: []"
            
            children_info_lines.append(line)
        
        children_info_text = "\n".join(children_info_lines)
        
        # Create child-name to index mapping
        child_to_idx = {child_name: idx for idx, (child_name, _) in enumerate(children_descriptions)}
        
        # Format peer relations of child nodes (structured, deduplicate reverse pairs)
        if children_peers_info:
            peers_relations = []
            seen_pairs = set()  # deduplicate: track already-added relation pairs
            
            for peer_info in children_peers_info:
                peer_node = peer_info.get('node', '')
                related_node = peer_info.get('related_node', '')
                peer_relation = peer_info.get('relationship_type', '')
                peer_desc = peer_info.get('description', '')
                
                # Use frozenset so (A, B) and (B, A) are treated as the same pair
                pair = frozenset([peer_node, related_node])
                
                # Skip if this pair has already been added
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                # Get indices for the related info entries
                peer_idx = child_to_idx.get(peer_node, '?')
                related_idx = child_to_idx.get(related_node, '?')
                
                # Format: Related Information X --[relation]--> Related Information Y, description
                relation_str = f"Related Information {peer_idx} --[{peer_relation}]--> Related Information {related_idx}"
                if peer_desc:
                    relation_str += f", {peer_desc}"
                peers_relations.append(relation_str)
            
            peers_info_text = "; ".join(peers_relations) if peers_relations else "None"
        else:
            peers_info_text = "None"
        
        prompt = f"""You are an expert responsible for generating **structured descriptions** about the current entity based on related information.

🎯 **Core Principle: Use only objective, verifiable factual information**

**✅ Required Information Types to Extract**:
- 📊 Precise Numbers: Years, quantities, amounts, rankings, counts (e.g., "Founded in 1959", "Over 5,000 employees", "Annual revenue of $1 billion")
- 📅 Specific Times: Dates, event times (e.g., "Independence on December 12, 1963", "Participated in the 2000 Olympics")
- 📍 Specific Locations: Countries, cities, positions, distances, elevations (e.g., "Located in East Africa", "5 km from city center", "Elevation 1,600 meters")
- 🎯 Specific Events: Historical events, awards, works, activities (e.g., "Won the Nobel Prize", "Published 'XXX'")
- 📏 Measurable Attributes: Area, population, scale (e.g., "Area of 580,000 square kilometers", "Population of 53 million", "Has 12 colleges")

⚠️ **Key Requirements**:
- All facts must be objective facts directly extracted from entity descriptions and relations
- Precise information such as numbers, times, and locations must appear unchanged in the output
- If a piece of related information has no objective facts (only subjective descriptions), do not extract facts from it
- Questions must be constructed based on these objective facts, ensuring verification requires search
- If the given input information is redundant or verbose, simplify it appropriately
- If information is insufficient, return {{"summary": "", "question": "", "facts": []}}

🔍 **Special Emphasis: Hierarchical Obfuscation Strategy**
In the related information list below, you need to hierarchically present all information. Do not omit, discard, or freely invent information.

**Input Content**:
1. Current Entity Information:
   - Entity Type: {entity_type}
   - Entity Name: {canonical_name} (⚠️ DO NOT use this name or its aliases in the output question or facts)
{f"   - Entity Description: {node_description}" if node_description else ""}
{f"   - Key Attributes: {', '.join([f'{k}: {v}' for k, v in node_key_attributes.items()])}" if node_key_attributes else ""}
{f"   - Aliases: {', '.join(node_surface_forms[:5])}" if node_surface_forms else ""}
2. Related Information List ({len(children_descriptions)} items):
{children_info_text}
3. Relationships Between Related Information:
{peers_info_text}

**Output Requirements**:
1. **facts**: Extract objective facts from related information. Each fact must contain at least one verifiable information point
   - Format: {{"fact": "[factual description, without entity names]", "source": "child:0" or "child_peer:0_1"}}
   - ⚠️ DO NOT include "{canonical_name}" or its aliases in the facts
2. **summary**: A one-sentence summary of the relationship types between the current entity and related information
3. **question**: A search question constructed based on facts, formatted as "Which [entity_type] satisfies the following conditions: [objective factual conditions]?"
   - ⚠️ DO NOT include "{canonical_name}" or its aliases in the question
   - ⚠️ If Entity Type is "Other", infer the actual entity type (e.g., country, organization, event, person) from the description and use it instead of "Other"

**JSON Output Format**:
{{
  "facts": [
    {{"fact": "[factual description]", "source": "child:0"}},
    {{"fact": "[factual description]", "source": "child_peer:0_1"}}
  ],
  "summary": "Relationship type description",
  "question": "Which {entity_type} satisfies the following conditions: [objective factual conditions]?"
}}

**Example 1**:
Input:
1. Current Entity Information:
  - Entity Type: Person
  - Number of Related Information: 3
2. Related Information List: 3 items
  - Related Information 0: An organization
    - Relationship Type: worked_at
    - facts: ["Founded in 1959", "Over 5,000 employees", "Annual revenue of $1 billion", "Media industry", "Provides news services", "Is an important organization"]
  - Related Information 1: A region
    - Relationship Type: born_in
    - facts: ["Independence on December 12, 1963", "Located in East Africa", "Population of approximately 53 million", "Area of 580,000 square kilometers", "Is an important country", "Rapid economic development"]
  - Related Information 2: An organization
    - Relationship Type: educated_at
    - facts: ["Founded in 1850", "Located in the capital", "Has 12 colleges", "Student population over 30,000", "Is a famous university", "Provides quality education"]

**Analysis of Input Facts**:
- Related Information 0: 4 objective facts (founding year, employee count, revenue, industry), 2 subjective descriptions (important, provides services)
- Related Information 1: 4 objective facts (independence date, location, population, area), 2 subjective descriptions (important, development)
- Related Information 2: 4 objective facts (founding year, location, college count, student count), 2 subjective descriptions (famous, quality)

Output:
{{
  "summary": "Three entities related to work, birthplace, and education, containing precise temporal, quantitative, and geographical information",
  "question": "Which person worked at a media company founded in 1959 with over 5,000 employees and annual revenue of $1 billion; was born in a country that gained independence on December 12, 1963, located in East Africa, with a population of approximately 53 million and an area of 580,000 square kilometers; and attended a university founded in 1850, located in the capital, with 12 colleges and a student population over 30,000?",
  "facts": [
    {{"fact": "Worked at a media company founded in 1959 with over 5,000 employees and annual revenue of $1 billion", "source": "child:0"}},
    {{"fact": "Born in a country that gained independence on December 12, 1963, located in East Africa, with a population of approximately 53 million and an area of 580,000 square kilometers", "source": "child:1"}},
    {{"fact": "Attended a university founded in 1850, located in the capital, with 12 colleges and a student population over 30,000", "source": "child:2"}}
  ]
}}

**Note**: The output facts and question contain only objective facts (numbers, times, locations), completely ignoring subjective descriptions ("important", "famous", "provides services")

【Example 2: Multi-hop Relationship Natural Language Expression】
Input:
1. Current Entity Information:
  - Entity Type: Person
  - Related Information List: 2 items
2. Related Information List: 2 items
  - Related Information 0: An organization
    - facts: ["Founded in 1959", "Headquarters 5 km from city center", "Covers an area of 20,000 square meters", "Media industry", "Once had an employee named xxx who later participated in the 2000 Olympics women's 100-meter sprint (this is a multi-hop fact)"]
  - Related Information 1: A region
    - facts: ["Independence in December 1963", "Located in East Africa", "Elevation 1,600 meters", "Average annual temperature 22 degrees", "Produced 10 Nobel Prize winners in Economics (this is a multi-hop fact)"]
3. Relationships Between Related Information: Information 0 (BBC London) --[located_in]--> Information 1 (London). The organization's headquarters is located in the capital of this region.
Output:
{{
  "summary": "Two entities related to work and geographical location, with geographical relationships between them, containing multiple precise location and measurement information",
  "question": "Which person worked at a media company founded in 1959 with headquarters 5 km from the city center covering an area of 20,000 square meters; and this company's headquarters is located in the capital of an East African region that gained independence in December 1963, with an elevation of 1,600 meters and an average annual temperature of 22 degrees?",
  "facts": [
    {{"fact": "Worked at a media company founded in 1959 with headquarters 5 km from the city center covering an area of 20,000 square meters", "source": "child:0"}},
    {{"fact": "Work location in an East African region that gained independence in December 1963, with an elevation of 1,600 meters and an average annual temperature of 22 degrees", "source": "child:1"}},
    {{"fact": "Company headquarters located in the capital of this region", "source": "child_peer:0_1"}}
  ]
}}

Output only JSON, do not add any other text or explanation."""
        
        return prompt
    
    async def _describe_non_leaf_node(self, ctx: GraphProcessingContext, node: str, current_depth: int) -> str:
        """
        Describe a non-leaf node.

        Args:
            node (str): Node name.
            current_depth (int): Depth of the current node.

        Returns:
            str: Non-leaf node description.
        """
        node_info = ctx.entity_map.get(node, {}).copy()
        
        # Get children list
        children = ctx.children_map.get(node, [])
        
        if not children:
            # No children: treat as a leaf node at a different depth; use name + description
            canonical_name = node_info.get('canonical_name', node)
            entity_desc = node_info.get('description', '')
            
            # Build description: name + description
            if entity_desc:
                description = f"{canonical_name}: {entity_desc}"
            else:
                description = canonical_name
            
            print(f"     ✓ Special case: no children; treating as leaf node.")
            print(f"     ✓ Description: {description[:200]}...")
            
            # Generate question for leaf node: remove entity name, keep only objective facts
            entity_type = node_info.get('entity_type', 'Entity')
            canonical_name = node_info.get('canonical_name', node)
            key_attributes = node_info.get('key_attributes', {})
            surface_forms = node_info.get('surface_forms', [])
            
            # Prepare key_attributes info
            key_attrs_text = ""
            if key_attributes:
                attrs_lines = []
                for key, value in key_attributes.items():
                    attrs_lines.append(f"- {key}: {value}")
                key_attrs_text = "\nKey Attributes:\n" + "\n".join(attrs_lines)
            
            # Prepare surface_forms info
            surface_forms_text = ""
            if surface_forms:
                surface_forms_text = f"\nAliases: {', '.join(surface_forms[:10])}"
            
            if entity_desc:
                # Use LLM to extract objective facts from the description, removing the entity name
                leaf_prompt = f"""You are an expert responsible for extracting objective facts from entity descriptions and generating search questions without revealing the entity name.

Entity Information:
- Entity Type: {entity_type}
- Entity Name: {canonical_name} (⚠️ DO NOT use this name or its aliases in the output question or facts)
- Entity Description: {entity_desc}
{key_attrs_text}{surface_forms_text}

Task: Extract objective, verifiable facts from the description and generate a search question that does NOT reveal the entity name.

Requirements:
1. Extract only objective facts: dates, numbers, locations, events, positions, etc.
2. Remove the entity name "{canonical_name}" and any references to it (including aliases)
3. Use generic terms instead of specific names (e.g., "a religious figure" instead of the actual name)
4. Generate a question that requires search to answer
5. The question must be in English

Output JSON format:
{{
  "facts": [
    {{"fact": "[objective fact without entity name]", "source": "self"}},
    {{"fact": "[another objective fact]", "source": "self"}}
  ],
  "question": "Which {entity_type} [objective facts extracted from description]?"
}}

Example:
Input:
- Entity Type: Person
- Entity Name: Sister Carmela Cecilia Carpio
- Entity Description: Sister Carmela Cecilia Carpio was born in 1933 in Naga City, Philippines. Served as principal of High School Department of University Of La Salette from 1968 to 1974.

Output:
{{
  "facts": [
    {{"fact": "Born in 1933 in Naga City, Philippines", "source": "self"}},
    {{"fact": "Served as principal of High School Department of University Of La Salette from 1968 to 1974", "source": "self"}}
  ],
  "question": "Which Person was born in 1933 in Naga City, Philippines, and served as principal of High School Department of University Of La Salette from 1968 to 1974?"
}}

Output only JSON, do not add any other text."""
                
                try:
                    leaf_response = await self._call_llm_with_retry(
                        model=self.generate_model,
                        messages=[
                            {"role": "system", "content": "You are an expert at extracting objective facts from descriptions and generating search questions without revealing entity names."},
                            {"role": "user", "content": leaf_prompt}
                        ],
                        max_tokens=2048,
                        response_format={"type": "json_object"},
                        operation_name=f"Generate leaf node question ({canonical_name})"
                    )
                    
                    if leaf_response and leaf_response.choices[0].message.content.strip():
                        raw_output = leaf_response.choices[0].message.content.strip()
                        try:
                            repaired_json = repair_json(raw_output)
                            leaf_data = json.loads(repaired_json)
                            leaf_question = leaf_data.get('question', '').strip()
                            leaf_facts = leaf_data.get('facts', [])
                            
                            if not leaf_question:
                                raise ValueError("Empty question in LLM response")
                        except Exception as json_error:
                            raise ValueError(f"JSON parsing failed: {json_error}")
                    else:
                        raise ValueError("LLM call failed or returned empty")
                        
                except Exception as e:
                    print(f"     ⚠️  LLM failed to generate leaf question: {e}; using fallback.")
                    # Fallback: simply remove the entity name
                    desc_without_name = entity_desc
                    # Try removing entity name and aliases
                    for alias in [canonical_name] + surface_forms:
                        desc_without_name = desc_without_name.replace(alias, f"a {entity_type.lower()}")
                    leaf_question = f"Which {entity_type} matches the following description: {desc_without_name}?"
                    leaf_facts = [{'fact': desc_without_name, 'source': 'self'}]
            else:
                # No description: generate generic question
                leaf_question = f"What is this {entity_type}?"
                leaf_facts = [{'fact': canonical_name, 'source': 'self'}]
            
            # Record in trace
            trace_entry = {
                'node': node,
                'node_type': 'leaf',      # marked as leaf node
                'depth': current_depth,
                'node_info': node_info,
                'description': description,   # leaf node description (stored only, not used downstream)
                'question': leaf_question,    # leaf question (entity name removed)
                'facts': leaf_facts,
                'is_leaf_simple': True,       # simplified leaf node flag
                'obfuscation_skipped': True,  # obfuscation skipped
                'obfuscation_count': 0,       # obfuscation count: 0 for leaf nodes
                'obfuscation_iterations': [], # obfuscation iterations: empty for leaf nodes
                'reason': 'no_children'       # reason: no children
            }
            ctx.description_trace.append(trace_entry)
            
            # Return question (consistent with non-leaf nodes)
            return leaf_question
        
        # Collect children info
        children_descriptions = []
        children_peers_info = []
        seen_peer_pairs = set()  # for deduplicating peer relations
        
        # Build children set for fast lookup
        children_set = set(child_node for child_node, _ in children)
        
        for child_node, relation_info in children:
            # Keep only child name and relation info
            children_descriptions.append((child_node, relation_info))

            # Check if child has peer relations
            child_peers = ctx.peer_map.get(child_node, [])
            if child_peers:
                for peer_name, peer_relation_info in child_peers:
                    # **Key check**: only add intra-child peer relations
                    # peer_name must also be in the children list
                    if peer_name not in children_set:
                        continue
                    
                    # Use frozenset to deduplicate; keep only one direction per pair
                    pair = frozenset([child_node, peer_name])
                    if pair in seen_peer_pairs:
                        continue
                    seen_peer_pairs.add(pair)
                    
                    children_peers_info.append({
                        'node': child_node,
                        'related_node': peer_name,
                        'relationship_type': peer_relation_info.get('relationship_type', ''),
                        'description': peer_relation_info.get('description', ''),  # add relation description
                        'relation_info': peer_relation_info
                    })
        
        try:
            prompt = self._get_non_leaf_node_description_prompt(ctx, node, node_info, children_descriptions, children_peers_info)
            canonical_name = node_info.get('canonical_name', node)
            
            response = await self._call_llm_with_retry(
                model=self.generate_model,
                messages=[
                    {"role": "system", "content": "You are an expert responsible for generating concise descriptions based on entity and its child node information. Please output results in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                response_format={"type": "json_object"},
                operation_name=f"non-leaf description ({canonical_name})"
            )
            
            if response is None:
                raise ValueError(f"Non-leaf node {canonical_name} description generation failed: LLM call failed.")
            
            raw_output = response.choices[0].message.content.strip()
            
            # If LLM returns empty string, raise error
            if not raw_output:
                raise ValueError(f"LLM returned empty output; non-leaf node {canonical_name} description generation failed.")
            
            # Parse JSON using json_repair
            try:
                repaired_json = repair_json(raw_output)
                result_data = json.loads(repaired_json)
            except Exception as json_error:
                raise ValueError(f"JSON parse failed for non-leaf node {canonical_name}: {json_error}\nRaw output: {raw_output}") from json_error
            
            # Extract summary, question, and facts
            summary = result_data.get('summary', '').strip()
            question = result_data.get('question', '').strip()
            facts = result_data.get('facts', [])
            
            # If summary is empty, raise error
            if not summary:
                raise ValueError(f"LLM returned empty summary; non-leaf node {canonical_name} description generation failed.")
            
            # If question is empty, raise error
            if not question:
                raise ValueError(f"LLM returned empty question; non-leaf node {canonical_name} description generation failed.")
            
            # description field is no longer generated; only question is used as the key info
            # description is not used downstream; focus is on question and facts
            
            # Build nested facts structure (including child node facts)
            nested_facts = []
            for fact_item in facts:
                fact_source = fact_item.get('source', '')
                nested_fact = fact_item.copy()
                
                # For 'child:'-type sources, attach child node's facts
                if fact_source.startswith('child:'):
                    try:
                        child_idx = int(fact_source.split(':')[1])
                        child_name = children_descriptions[child_idx][0]
                        
                        # Get child node's facts from trace
                        child_facts = []
                        for trace_entry_temp in reversed(ctx.description_trace):
                            if trace_entry_temp.get('node') == child_name:
                                child_facts = trace_entry_temp.get('facts', [])
                                break
                        
                        # Attach child facts to nested_fact
                        if child_facts:
                            nested_fact['child_facts'] = child_facts
                            nested_fact['child_node'] = child_name
                    except (IndexError, ValueError):
                        pass
                
                nested_facts.append(nested_fact)
            
            # Record in trace
            trace_entry = {
                'node': node,
                'node_type': 'non-leaf',
                'depth': current_depth,
                'node_info': node_info,
                'children_nodes': [c[0] for c in children],
                'children_relations': [
                    {
                        'node': n,
                        'relation': r.copy()
                    }
                    for n, r in children_descriptions
                ],
                'children_peers_info': [info.copy() for info in children_peers_info],
                'prompt': prompt,
                'llm_summary': summary,       # LLM-generated summary
                'question': question,         # LLM-generated question (core field)
                'facts': nested_facts,        # nested facts structure (includes child facts)
                'raw_llm_output': raw_output, # raw LLM output for debugging
                'obfuscation_count': 0,       # obfuscation count (updated later)
                'obfuscation_iterations': []  # obfuscation iteration records (updated later)
            }
            ctx.description_trace.append(trace_entry)
            
            # Return question as the key info for this node
            return question
        except Exception as e:
            print(f"❌ LLM non-leaf node description failed: {e}")
            # Raise exception directly; no fallback
            canonical_name = node_info.get('canonical_name', node)
            raise ValueError(f"Non-leaf node {canonical_name} description generation failed: {e}") from e
    
    async def _build_layerwise_descriptions(self, ctx: GraphProcessingContext) -> str:
        """
        Build descriptions layer by layer (from leaf nodes to root).

        Args:
            ctx (GraphProcessingContext): Graph processing context.

        V2 changes:
        - Leaf nodes use entity name + description; no LLM generation or obfuscation.
        - Non-leaf nodes undergo normal LLM generation and iterative obfuscation.

        Returns:
            str: Description of the root node (target_entity).
        """
        # Get all depths, sorted descending (leaf to root)
        depths = sorted(ctx.layer_entities.keys(), reverse=True)
        max_depth = depths[0]
        min_depth = depths[-1]
        
        print(f"\n{'='*80}")
        print(f"📝 Starting layer-wise description generation")
        print(f"{'='*80}")
        print(f"Topology info:")
        print(f"  • Depth range: {min_depth} (root) ━━> {max_depth} (leaf)")
        print(f"  • Total layers: {len(depths)}")
        print(f"  • Total nodes: {sum(len(ctx.layer_entities[d]) for d in depths)}")
        print(f"Strategy:")
        print(f"  • Leaf nodes: use name + description, skip obfuscation.")
        print(f"  • Non-leaf nodes: LLM generation + iterative obfuscation.")
        print(f"{'='*80}\n")
        
        # Start from the maximum depth (leaf nodes)
        for depth in depths:
            entities_at_depth = ctx.layer_entities[depth]
            depth_label = "root" if depth == min_depth else "leaf" if depth == max_depth else "middle"
            print(f"\n{'─'*80}")
            print(f"📊 Depth {depth} ({depth_label}): {len(entities_at_depth)} entities")
            print(f"{'─'*80}")
            
            for entity_name in entities_at_depth:
                if entity_name in ctx.description_cache:
                    # Already processed; skip
                    print(f"  ⏭️  {entity_name} (cached)")
                    continue
                
                print(f"\n  🔄 Processing: {entity_name}")
                
                # Check whether this is a leaf node
                is_leaf = (depth == max_depth)
                
                if is_leaf:
                    # V2: leaf nodes use entity name + description
                    entity_info = ctx.entity_map.get(entity_name, {})
                    entity_desc = entity_info.get('description', '')
                    
                    # Build description: name + description
                    if entity_desc:
                        description = f"{entity_name}: {entity_desc}"
                    else:
                        description = entity_name
                    
                    print(f"     ✓ Type: leaf node")
                    print(f"     ✓ Description: {description[:200]}...")
                    
                    # Generate question for leaf node: remove entity name, keep only objective facts
                    entity_type = entity_info.get('entity_type', 'Entity')
                    key_attributes = entity_info.get('key_attributes', {})
                    surface_forms = entity_info.get('surface_forms', [])
                    
                    # Prepare key_attributes info
                    key_attrs_text = ""
                    if key_attributes:
                        attrs_lines = []
                        for key, value in key_attributes.items():
                            attrs_lines.append(f"- {key}: {value}")
                        key_attrs_text = "\nKey Attributes:\n" + "\n".join(attrs_lines)
                    
                    # Prepare surface_forms info
                    surface_forms_text = ""
                    if surface_forms:
                        surface_forms_text = f"\nAliases: {', '.join(surface_forms[:10])}"
                    
                    if entity_desc:
                        # Use LLM to extract objective facts, removing entity name
                        leaf_prompt = f"""You are an expert responsible for extracting objective facts from entity descriptions and generating search questions without revealing the entity name.

Entity Information:
- Entity Type: {entity_type}
- Entity Name: {entity_name} (⚠️ DO NOT use this name or its aliases in the output question or facts)
- Entity Description: {entity_desc}
{key_attrs_text}{surface_forms_text}

Task: Extract objective, verifiable facts from the description and generate a search question that does NOT reveal the entity name.

Requirements:
1. Extract only objective facts: dates, numbers, locations, events, positions, etc.
2. Remove the entity name "{entity_name}" and any references to it (including aliases)
3. Use generic terms instead of specific names (e.g., "a religious figure" instead of the actual name)
4. Generate a question that requires search to answer
5. The question must be in English
6. If Entity Type is "Other", infer the actual entity type (e.g., country, organization, event, person) from the description and use it in the question instead of "Other"

Output JSON format:
{{
  "facts": [
    {{"fact": "[objective fact without entity name]", "source": "self"}},
    {{"fact": "[another objective fact]", "source": "self"}}
  ],
  "question": "Which {entity_type} [objective facts extracted from description]?"
}}

Example:
Input:
- Entity Type: Person
- Entity Name: Sister Carmela Cecilia Carpio
- Entity Description: Sister Carmela Cecilia Carpio was born in 1933 in Naga City, Philippines. Served as principal of High School Department of University Of La Salette from 1968 to 1974.

Output:
{{
  "facts": [
    {{"fact": "Born in 1933 in Naga City, Philippines", "source": "self"}},
    {{"fact": "Served as principal of High School Department of University Of La Salette from 1968 to 1974", "source": "self"}}
  ],
  "question": "Which Person was born in 1933 in Naga City, Philippines, and served as principal of High School Department of University Of La Salette from 1968 to 1974?"
}}

Output only JSON, do not add any other text."""
                        
                        try:
                            leaf_response = await self._call_llm_with_retry(
                                model=self.generate_model,
                                messages=[
                                    {"role": "system", "content": "You are an expert at extracting objective facts from descriptions and generating search questions without revealing entity names."},
                                    {"role": "user", "content": leaf_prompt}
                                ],
                                max_tokens=2048,
                                response_format={"type": "json_object"},
                                operation_name=f"Generate leaf node question ({entity_name})"
                            )
                            
                            if leaf_response and leaf_response.choices[0].message.content.strip():
                                raw_output = leaf_response.choices[0].message.content.strip()
                                try:
                                    repaired_json = repair_json(raw_output)
                                    leaf_data = json.loads(repaired_json)
                                    leaf_question = leaf_data.get('question', '').strip()
                                    leaf_facts = leaf_data.get('facts', [])
                                    
                                    if not leaf_question:
                                        raise ValueError("Empty question in LLM response")
                                except Exception as json_error:
                                    raise ValueError(f"JSON parsing failed: {json_error}")
                            else:
                                raise ValueError("LLM call failed or returned empty")
                                
                        except Exception as e:
                            print(f"     ⚠️  LLM failed to generate leaf question: {e}; using fallback.")
                            # Fallback: simply remove the entity name
                            desc_without_name = entity_desc
                            # Try removing entity name and aliases
                            for alias in [entity_name] + surface_forms:
                                desc_without_name = desc_without_name.replace(alias, f"a {entity_type.lower()}")
                            leaf_question = f"Which {entity_type} matches the following description: {desc_without_name}?"
                            leaf_facts = [{'fact': desc_without_name, 'source': 'self'}]
                    else:
                        # No description: generate generic question
                        leaf_question = f"What is this {entity_type}?"
                        leaf_facts = [{'fact': entity_name, 'source': 'self'}]
                    
                    # Record in trace
                    trace_entry = {
                        'node': entity_name,
                        'node_type': 'leaf',
                        'depth': depth,
                        'node_info': entity_info,
                        'description': description,
                        'question': leaf_question,    # leaf question (entity name removed)
                        'facts': leaf_facts,
                        'is_leaf_simple': True,       # simplified leaf flag
                        'obfuscation_skipped': True,  # obfuscation skipped
                        'obfuscation_count': 0,       # obfuscation count: 0 for leaf nodes
                        'obfuscation_iterations': []  # obfuscation iterations: empty for leaf nodes
                    }
                    ctx.description_trace.append(trace_entry)
                    
                    # Mark as processed
                    ctx.description_cache[entity_name] = True
                    
                else:
                    # Non-leaf node: normal description generation
                    _ = await self._describe_non_leaf_node(ctx, entity_name, depth)
                    
                    # Check if this node was treated as a leaf because it has no children
                    should_skip_obfuscation = False
                    for trace_entry in reversed(ctx.description_trace):
                        if trace_entry.get('node') == entity_name:
                            if trace_entry.get('obfuscation_skipped', False):
                                should_skip_obfuscation = True
                                print(f"    Node has no children; skipping obfuscation.")
                            break
                    
                    if should_skip_obfuscation:
                        # Mark as processed
                        ctx.description_cache[entity_name] = True
                    else:
                        # Normal non-leaf node: verify and obfuscate
                        print(f"     ✓ Type: non-leaf node")
                        
                        # Run iterative verification and obfuscation
                        # Get question and facts from the most recent trace entry
                        question = ""
                        facts = []
                        entity_info = ctx.entity_map.get(entity_name, {})
                        entity_type = entity_info.get('entity_type', 'Entity')
                        
                        # Get question and facts from trace
                        for trace_entry in reversed(ctx.description_trace):
                            if trace_entry.get('node') == entity_name:
                                question = trace_entry.get('question', '')
                                facts = trace_entry.get('facts', [])
                                break
                        
                        print(f"     ✓ Initial question: {question[:200]}...")
                        print(f"     🔄 Starting verification and obfuscation...")
                        
                        # Iteratively verify and obfuscate question
                        obfuscation_result = await self._iterative_verify_and_obfuscate(
                            question,
                            facts,
                            entity_name,
                            entity_type,
                            entity_info
                        )
                        
                        # Use the obfuscated question
                        final_question = obfuscation_result['final_question']
                        
                        # Update obfuscation info in trace
                        for trace_entry in reversed(ctx.description_trace):
                            if trace_entry.get('node') == entity_name:
                                trace_entry['obfuscation_iterations'] = obfuscation_result['iterations']
                                trace_entry['obfuscation_count'] = obfuscation_result['total_iterations']  # total obfuscation count
                                trace_entry['obfuscation_successful'] = obfuscation_result['obfuscation_successful']
                                trace_entry['final_question'] = final_question
                                trace_entry['final_facts'] = obfuscation_result['final_facts']
                                break
                        
                        success_icon = "✅" if obfuscation_result['obfuscation_successful'] else "⚠️"
                        print(f"     {success_icon} Obfuscation done: {obfuscation_result['total_iterations']} iterations.")
                        print(f"     ✓ Final question: {final_question[:200]}...")
                        
                        # Mark as processed
                        ctx.description_cache[entity_name] = True
        
        # Get root node's final question from trace
        root_question = ""
        for trace_entry in reversed(ctx.description_trace):
            if trace_entry.get('node') == ctx.target_entity:
                root_question = trace_entry.get('final_question') or trace_entry.get('question', '')
                break
        
        print(f"\n✅ All layer descriptions generated (V2).")
        print(f"  Root question length: {len(root_question)} characters.")
        
        return root_question
    async def build_description(self, ctx: GraphProcessingContext, graph_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build description (main entry method).

        Args:
            ctx (GraphProcessingContext): Graph processing context.
            graph_data (Dict[str, Any]): Graph data dictionary.

        Returns:
            Optional[Dict[str, Any]]: Description result dictionary.
        """
        # Load graph data (ctx is newly created; no need to reset)
        if not self.load_graph_with_topology(ctx, graph_data):
            return None
        
        # Save original graph data (deep copy to avoid side effects from later modifications)
        original_graph_data = copy.deepcopy(ctx.graph_data)
        
        # Build parent-child mapping
        if not self._build_parent_child_mapping(ctx):
            return None
        
        # V3: code system removed; entity names are used directly
        # _assign_all_entity_codes(ctx) is no longer called
        
        # Build layerwise descriptions (includes iterative verification and obfuscation)
        description = await self._build_layerwise_descriptions(ctx)
        
        # Update depth info in trace
        for trace_entry in ctx.description_trace:
            node = trace_entry['node']
            # Look up depth from layer_entities
            for depth, entities in ctx.layer_entities.items():
                if node in entities:
                    trace_entry['depth'] = depth
                    break
        
        # Organize all node outputs by topology
        topology_output = self._build_topology_output(ctx)
        
        # Build result (generated_description and all_node_descriptions removed; only core trace info kept)
        result = {
            'original_graph_data': original_graph_data,  # original graph data (no generated_description added)
            'target_entity': ctx.target_entity,
            'final_question': description,                # final generated question (root node)
            'description_trace': copy.deepcopy(ctx.description_trace),  # full trace (all questions and facts)
            'topology_structured_output': topology_output,               # topology-organized node outputs
            'generation_metadata': {
                'version': 'v3_optimized',             # V3 optimized version tag
                'generate_model': self.generate_model,
                'verify_model': self.verify_model,
                'max_obfuscation_iterations': self.max_obfuscation_iterations,
                'timestamp': datetime.now().isoformat(),
                'method': 'layerwise',                 # method: layer-wise
                'leaf_node_strategy': 'use_entity_name_and_description',  # V2: leaf nodes use name + description
                'total_nodes_processed': len(ctx.description_cache)
            }
        }
        
        # Compute obfuscation iteration distribution (reflects data difficulty)
        obfuscation_stats = {
            'skipped': 0,         # nodes that skipped obfuscation (leaf nodes)
            '0_iterations': 0,    # 0 iterations (first verification already failed)
            '1_iterations': 0,    # 1 iteration
            '2_iterations': 0,    # 2 iterations
            '3_iterations': 0,    # 3 iterations
            '4_iterations': 0,    # 4 iterations
            '5+_iterations': 0,   # 5+ iterations
            'max_iterations': 0,  # maximum iterations
            'total_iterations': 0 # total iterations
        }
        
        for trace_entry in ctx.description_trace:
            if trace_entry.get('obfuscation_skipped', False):
                obfuscation_stats['skipped'] += 1
            else:
                count = trace_entry.get('obfuscation_count', 0)
                obfuscation_stats['total_iterations'] += count
                if count > obfuscation_stats['max_iterations']:
                    obfuscation_stats['max_iterations'] = count
                
                if count == 0:
                    obfuscation_stats['0_iterations'] += 1
                elif count == 1:
                    obfuscation_stats['1_iterations'] += 1
                elif count == 2:
                    obfuscation_stats['2_iterations'] += 1
                elif count == 3:
                    obfuscation_stats['3_iterations'] += 1
                elif count == 4:
                    obfuscation_stats['4_iterations'] += 1
                else:
                    obfuscation_stats['5+_iterations'] += 1
        
        # Attach obfuscation statistics to metadata (reflects data difficulty)
        result['generation_metadata']['obfuscation_statistics'] = {
            'skipped_count': obfuscation_stats['skipped'],
            'nodes_with_obfuscation': len(ctx.description_trace) - obfuscation_stats['skipped'],
            'total_iterations': obfuscation_stats['total_iterations'],
            'max_iterations': obfuscation_stats['max_iterations'],
            'average_iterations': obfuscation_stats['total_iterations'] / max(1, len(ctx.description_trace) - obfuscation_stats['skipped']),
            'iteration_distribution': {
                '0': obfuscation_stats['0_iterations'],
                '1': obfuscation_stats['1_iterations'],
                '2': obfuscation_stats['2_iterations'],
                '3': obfuscation_stats['3_iterations'],
                '4': obfuscation_stats['4_iterations'],
                '5+': obfuscation_stats['5+_iterations']
            }
        }
        
        print(f"\n{'─'*80}")
        print(f"✅ Question generation complete.")
        print(f"{'─'*80}")
        print(f"Target entity: {ctx.target_entity}")
        print(f"Root question length: {len(description)} characters.")
        print(f"Total nodes processed: {len(ctx.description_cache)}")
        print(f"Trace records: {len(ctx.description_trace)}")
        print(f"\n📊 Obfuscation iteration statistics (reflects data difficulty):")
        print(f"  • Skipped (leaf nodes): {obfuscation_stats['skipped']}")
        print(f"  • 0 iterations (first verification failed): {obfuscation_stats['0_iterations']}")
        print(f"  • 1 iteration: {obfuscation_stats['1_iterations']}")
        print(f"  • 2 iterations: {obfuscation_stats['2_iterations']}")
        print(f"  • 3 iterations: {obfuscation_stats['3_iterations']}")
        print(f"  • 4 iterations: {obfuscation_stats['4_iterations']}")
        print(f"  • 5+ iterations: {obfuscation_stats['5+_iterations']}")
        print(f"  • Max iterations: {obfuscation_stats['max_iterations']}")
        print(f"  • Avg iterations: {obfuscation_stats['total_iterations'] / max(1, len(ctx.description_trace) - obfuscation_stats['skipped']):.2f}")
        print(f"{'─'*80}")

        return result
    
    async def process_single_graph(self, graph_data: Dict[str, Any], index: int, total: int) -> Optional[Dict[str, Any]]:
        """
        Process a single subgraph and generate its description.

        Args:
            graph_data (Dict[str, Any]): Graph data dictionary.
            index (int): Current index in the to-process list.
            total (int): Total number of items to process.

        Returns:
            Optional[Dict[str, Any]]: Description result dict on success, or None on failure.

        Note:
            - On success, the result is immediately appended to the output file.
            - Saved subgraphs are automatically skipped on the next run (resume support).
        """
        target_entity = graph_data.get('target_entity', '')
        if not target_entity:
            print(f"⚠️  [{index+1}/{total}] Skipped: graph data missing 'target_entity'.")
            return None
        
        print(f"\n{'='*100}")
        print(f"🎯 [{index+1}/{total}] Processing subgraph: {target_entity}")
        print(f"{'='*100}")
        
        try:
            # Create an independent context for each graph to avoid concurrency conflicts
            ctx = GraphProcessingContext()
            result = await self.build_description(ctx, graph_data)
            
            if result:
                # Immediately save (append mode) to ensure resume support
                # Saved subgraphs will be automatically skipped on the next run
                # Use asyncio.to_thread to avoid blocking the event loop
                await asyncio.to_thread(append_jsonl, result, self.output_file)
                print(f"\n{'='*100}")
                print(f"✅ [{index+1}/{total}] Done and saved: {target_entity}")
                print(f"   💾 Saved to output file; will be skipped on next run.")
                print(f"{'='*100}\n")
                return result
            else:
                print(f"\n{'='*100}")
                print(f"⚠️  [{index+1}/{total}] Processing returned empty result: {target_entity}")
                print(f"   ⚠️  Not saved; will be reprocessed on next run.")
                print(f"{'='*100}\n")
                return None
                
        except Exception as e:
            print(f"\n{'='*100}")
            print(f"❌ [{index+1}/{total}] Processing failed: {target_entity}")
            print(f"   Error: {e}")
            print(f"   ⚠️  Not saved; will be reprocessed on next run.")
            print(f"{'='*100}\n")
            import traceback
            traceback.print_exc()
            return None
    
    async def process_all(self, 
                         graph_file: str,
                         enable_resume: bool = True,
                         parallel: int = 1,
                         start: int = 0,
                         max_samples: Optional[int] = None) -> int:
        """
        Process all graphs and generate descriptions (supports batch, resume, and concurrency).

        Args:
            graph_file (str): Graph structure file path (JSONL format).
            enable_resume (bool): Whether to enable resume support (default: True).
            parallel (int): Concurrency level (default: 1).
            start (int): Index to start from (default: 0).
            max_samples (Optional[int]): Max number of samples to process (default: None = all).

        Returns:
            int: Number of successfully processed subgraphs.
        """
        graph_file_path = Path(graph_file)
        if not graph_file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_file_path}")
        
        print(f"\n📖 Loading graph file: {graph_file_path}")
        all_graphs = read_jsonl(str(graph_file_path))
        print(f"✅ Loaded {len(all_graphs)} graph records (subgraphs).")
        
        # Load already-processed target_entity values (resume support)
        processed_targets = set()
        if enable_resume:
            processed_targets = load_processed_targets(self.output_file)
            if processed_targets:
                print(f"📋 Resume status:")
                print(f"   - Already processed subgraphs: {len(processed_targets)}")
                print(f"   - Remaining subgraphs: {len(all_graphs) - len(processed_targets)}")
        else:
            print(f"📋 Resume: disabled; will process all subgraphs.")
        
        # Filter already-processed graphs (per subgraph)
        graphs_to_process = []
        skipped_count = 0
        for graph_data in all_graphs:
            target_entity = graph_data.get('target_entity', '')
            if not target_entity:
                print(f"⚠️  Warning: found graph data without 'target_entity'; skipping.")
                skipped_count += 1
                continue
            
            if enable_resume and target_entity in processed_targets:
                skipped_count += 1
                continue  # silent skip to avoid excessive logging
            
            graphs_to_process.append(graph_data)
        
        # Display statistics
        print(f"\n📊 Processing statistics:")
        print(f"   - Total subgraphs: {len(all_graphs)}")
        if enable_resume and processed_targets:
            print(f"   - Already processed (skipped): {skipped_count}")
        print(f"   - To process: {len(graphs_to_process)}")
        
        if skipped_count > 0 and enable_resume:
            print(f"   💡 Skipped subgraphs will not be reprocessed (resume mode).")
        
        # Determine processing range
        if start < 0:
            start = 0
        end = len(graphs_to_process) if max_samples is None else min(len(graphs_to_process), start + max_samples)
        graphs_to_process = graphs_to_process[start:end]
        
        if not graphs_to_process:
            print(f"ℹ️  No graphs to process.")
            return 0
        
        print(f"🚀 Starting to process {len(graphs_to_process)} graphs (concurrency: {parallel}).")
        
        # Use Semaphore to control concurrency
        semaphore = asyncio.Semaphore(parallel)
        
        async def process_with_semaphore(graph_data: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.process_single_graph(graph_data, idx, len(graphs_to_process))
        
        # Concurrent processing
        tasks = [process_with_semaphore(graph_data, start + i) 
                 for i, graph_data in enumerate(graphs_to_process)]
        
        if parallel > 1:
            print(f"⚡ Concurrent processing: {parallel} graphs at a time.")
            print(f"   Submitted {len(tasks)} tasks; waiting for all to complete...")
            # Use gather to collect results, ensuring all tasks complete
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                print(f"   ✅ All tasks completed; {len(results)} results.")
            except Exception as e:
                print(f"⚠️ Concurrent processing exception: {e}")
                import traceback
                traceback.print_exc()
                # If gather fails, wait for tasks individually
                results = []
                for i, task in enumerate(tasks):
                    try:
                        print(f"   Waiting for task {i+1}/{len(tasks)}...")
                        result = await task
                        results.append(result)
                    except Exception as task_error:
                        print(f"⚠️ Task {i+1} exception: {task_error}")
                        results.append(task_error)
        else:
            # Sequential processing
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    print(f"❌ Task execution exception: {e}")
                    results.append(None)
        
        # Compute results
        success_count = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        failed_count = sum(1 for r in results if r is None or isinstance(r, Exception))
        
        # Reload processed target_entity values for final stats
        final_processed_count = len(load_processed_targets(self.output_file))
        
        print(f"\n" + "="*80)
        print(f"✅ Processing complete.")
        print(f"  - Successfully processed this run: {success_count} subgraphs")
        print(f"  - Failed this run: {failed_count} subgraphs")
        print(f"  - Total this run: {len(results)} subgraphs")
        print(f"  - Output file: {self.output_file}")
        print(f"  - Cumulative processed: {final_processed_count} subgraphs (including previous runs).")
        print(f"="*80)
        
        if failed_count > 0:
            print(f"\n💡 Note:")
            print(f"   - Failed subgraphs are not saved and will be retried on the next run.")
            print(f"   - Successful subgraphs are saved and will be skipped on the next run (resume).")
        
        return success_count


def _default_output_path(input_path: Path) -> Path:
    """Generate default output path (outputs to cache_7 directory)."""
    return Path("./data_synthesis/cache/cache_7") / f"{input_path.stem}_descriptions.jsonl"


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Layer-wise description generation based on graph topology.')
    parser.add_argument('--graph_file', type=str,
                        required=True,
                        help='Graph structure file path (JSONL format).')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: data_synthesis/cache/cache_7/<input_stem>_descriptions.jsonl).')
    parser.add_argument('--generate-model', type=str, default='deepseek-v3.2',
                        help='LLM model for description generation (default: deepseek-v3.2).')
    parser.add_argument('--verify-model', type=str, default='gpt-5-mini',
                        help='LLM model for verification and obfuscation (default: gpt-5-mini).')
    parser.add_argument('--max-obfuscation-iterations', type=int, default=5,
                        help='Max obfuscation iterations (default: 5).')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Concurrency level (default: 1 = sequential).')
    parser.add_argument('--no-resume', action='store_true', default=False,
                        help='Disable resume support (default: enabled).')
    parser.add_argument('--start', type=int, default=0,
                        help='Index to start from (default: 0).')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to process (default: all).')
    
    args = parser.parse_args()
    
    # Handle paths
    input_path = Path(args.graph_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = _default_output_path(input_path)
    
    print(f"✅ Input:       {input_path}")
    print(f"✅ Output:      {output_path}")
    print(f"✅ Concurrency: {args.parallel}")
    print(f"✅ Resume:      {'disabled' if args.no_resume else 'enabled'}")
    
    # Create builder
    builder = LayerwiseDescriptionBuilder(
        output_file=str(output_path),
        generate_model=args.generate_model,
        verify_model=args.verify_model,
        max_obfuscation_iterations=args.max_obfuscation_iterations,
    )
    
    # Process and generate descriptions (using asyncio.run)
    asyncio.run(builder.process_all(
        graph_file=str(input_path),
        enable_resume=not args.no_resume,
        parallel=args.parallel,
        start=args.start,
        max_samples=args.max_samples
    ))


if __name__ == '__main__':
    main()

