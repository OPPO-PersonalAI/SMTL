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
v2 Step-8: Extract QA pairs from the output of 7_layerwise_generate_root_desc.py

Goal:
- Input : output of 7_layerwise_generate_root_desc.py (contains layer-wise generated questions)
- Process: extract QA pairs (question + golden_answer) for each node
- Output : QA pair file + statistics file

Each QA pair contains:
- question      : generated question
- golden_answer : entity name (answer)
- graph info    : original QA, graph ID, etc.
- subgraph info : subgraph ID, root node, etc.
- QA info       : topology, obfuscation count, etc.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

from utils import read_jsonl, write_jsonl


def extract_qa_pairs(input_file: Path, output_file: Path, stats_file: Path):
    """
    Extract QA pairs from the input file.

    Args:
        input_file  (Path): Input file path (output of 7_layerwise_generate_root_desc.py).
        output_file (Path): Output file path (QA pairs).
        stats_file  (Path): Statistics file path.
    """
    print(f"\n{'='*80}")
    print(f"📖 Extracting QA pairs")
    print(f"{'='*80}")
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Stats file:  {stats_file}")
    print(f"{'='*80}\n")
    
    # Read input data
    print(f"📚 Reading input data...")
    input_records = read_jsonl(str(input_file))
    print(f"✅ Read {len(input_records)} subgraph records.\n")
    
    # Statistics
    stats = {
        'total_graphs': 0,                              # unique graph count
        'total_subgraphs': len(input_records),          # total subgraph count
        'total_qa_pairs': 0,                            # total QA pairs
        'graphs': {},                                   # graph_id -> graph info
        'obfuscation_distribution': defaultdict(int),   # obfuscation count distribution
        'node_type_distribution': defaultdict(int),     # node type distribution
        'depth_distribution': defaultdict(int),         # depth distribution
    }
    
    # Extract QA pairs
    qa_pairs = []
    graph_ids_seen = set()
    
    for subgraph_idx, subgraph_record in enumerate(input_records):
        # Get original graph data
        original_graph_data = subgraph_record.get('original_graph_data', {})
        target_entity = subgraph_record.get('target_entity', '')
        topology_output = subgraph_record.get('topology_structured_output', {})
        generation_metadata = subgraph_record.get('generation_metadata', {})
        
        # Extract graph info
        original_question = original_graph_data.get('question', '')
        original_answer = original_graph_data.get('golden_answer', '') or original_graph_data.get('answer', '')
        question_hash = original_graph_data.get('question_hash', '')
        subgraph_id = original_graph_data.get('subgraph_id', f'unknown_{subgraph_idx}')
        
        # If question_hash is missing, try to get it from parent_graph
        if not question_hash:
            parent_graph = original_graph_data.get('parent_graph', {})
            question_hash = parent_graph.get('question_hash', '')

        # Generate graph ID (use question_hash or subgraph_id)
        graph_id = question_hash if question_hash else subgraph_id.split('_')[0] if '_' in subgraph_id else f'graph_{subgraph_idx}'

        # Record graph info (once per graph)
        if graph_id not in graph_ids_seen:
            graph_ids_seen.add(graph_id)
            stats['total_graphs'] += 1
            stats['graphs'][graph_id] = {
                'graph_id': graph_id,
                'question': original_question,
                'golden_answer': original_answer,
                'question_hash': question_hash,
                'subgraph_count': 0  # updated below
            }
        
        # Update subgraph count for this graph
        stats['graphs'][graph_id]['subgraph_count'] += 1
        
        # Extract subgraph statistics
        subgraph_stats = {
            'subgraph_id': subgraph_id,
            'target_entity': target_entity,
            'num_entities': original_graph_data.get('num_entities', 0),
            'num_relations': original_graph_data.get('num_relations', 0),
            'depth': original_graph_data.get('depth', 0),
            'cycle_count': original_graph_data.get('cycle_count', 0),
        }
        
        # Extract QA pairs for each node from topology_output
        layers = topology_output.get('layers', [])
        
        for layer in layers:
            depth = layer.get('depth', 0)
            entities = layer.get('entities', [])
            
            for entity in entities:
                entity_name = entity.get('entity_name', '')
                question = entity.get('question', '')
                entity_type = entity.get('entity_type', 'Unknown')
                node_type = entity.get('node_type', 'unknown')
                facts = entity.get('facts', [])
                obfuscation_info = entity.get('obfuscation_info', {})
                
                # Skip nodes without a question
                if not question or not entity_name:
                    continue
                
                # Extract obfuscation info
                obfuscation_count = obfuscation_info.get('count', 0)
                obfuscation_skipped = obfuscation_info.get('skipped', False)
                obfuscation_successful = obfuscation_info.get('successful', None)
                
                # Build QA record
                qa_record = {
                    # Core QA fields
                    'question': question,
                    'golden_answer': entity_name,
                    
                    # Graph info
                    'graph_info': {
                        'graph_id': graph_id,
                        'question_hash': question_hash,
                        'original_question': original_question,
                        'original_golden_answer': original_answer,
                    },
                    
                    # Subgraph info
                    'subgraph_info': {
                        'subgraph_id': subgraph_id,
                        'target_entity': target_entity,  # subgraph root node
                        'subgraph_statistics': subgraph_stats,
                    },
                    
                    # QA info
                    'qa_info': {
                        'entity_name': entity_name,
                        'entity_type': entity_type,
                        'node_type': node_type,
                        'depth': depth,
                        'is_root': layer.get('is_root', False),
                        'is_leaf': layer.get('is_leaf', False),
                        'facts': facts,
                        'obfuscation_info': {
                            'skipped': obfuscation_skipped,
                            'count': obfuscation_count,
                            'successful': obfuscation_successful,
                        },
                        # Store full topology info for downstream analysis
                        'topology': {
                            'total_layers': topology_output.get('total_layers', 0),
                            'depth_range': topology_output.get('depth_range', {}),
                            'layer_index': depth,
                        },
                    },
                    
                    # Generation metadata
                    'generation_metadata': {
                        'generate_model': generation_metadata.get('generate_model', ''),
                        'verify_model': generation_metadata.get('verify_model', ''),
                        'max_obfuscation_iterations': generation_metadata.get('max_obfuscation_iterations', 0),
                        'version': generation_metadata.get('version', ''),
                    },
                }
                
                # For non-leaf nodes, add children info
                if node_type == 'non-leaf':
                    qa_record['qa_info']['children_nodes'] = entity.get('children_nodes', [])
                    qa_record['qa_info']['llm_summary'] = entity.get('llm_summary', '')
                
                qa_pairs.append(qa_record)
                stats['total_qa_pairs'] += 1
                
                # Update statistics
                if obfuscation_skipped:
                    stats['obfuscation_distribution']['skipped'] += 1
                else:
                    stats['obfuscation_distribution'][str(obfuscation_count)] += 1
                
                stats['node_type_distribution'][node_type] += 1
                stats['depth_distribution'][str(depth)] += 1
        
        # Show progress
        if (subgraph_idx + 1) % 100 == 0:
            print(f"  Progress: {subgraph_idx + 1}/{len(input_records)} subgraphs processed; {stats['total_qa_pairs']} QA pairs extracted.")
    
    # Save QA pairs
    print(f"\n💾 Saving QA pairs...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(qa_pairs, str(output_file))
    print(f"✅ Saved {len(qa_pairs)} QA pairs to: {output_file}")
    
    # Build statistics
    print(f"\n📊 Generating statistics...")
    
    # Compute obfuscation statistics
    obfuscation_stats = {
        'skipped': stats['obfuscation_distribution'].get('skipped', 0),
        'with_obfuscation': stats['total_qa_pairs'] - stats['obfuscation_distribution'].get('skipped', 0),
        'distribution': dict(stats['obfuscation_distribution']),
        'average_iterations': 0.0,
        'max_iterations': 0,
    }
    
    # Compute average iterations (excluding skipped)
    total_iterations = 0
    count_with_obfuscation = 0
    for count_str, num_nodes in stats['obfuscation_distribution'].items():
        if count_str != 'skipped':
            count = int(count_str)
            total_iterations += count * num_nodes
            count_with_obfuscation += num_nodes
            if count > obfuscation_stats['max_iterations']:
                obfuscation_stats['max_iterations'] = count
    
    if count_with_obfuscation > 0:
        obfuscation_stats['average_iterations'] = total_iterations / count_with_obfuscation
    
    # Build complete statistics
    final_stats = {
        'summary': {
            'total_graphs': stats['total_graphs'],
            'total_subgraphs': stats['total_subgraphs'],
            'total_qa_pairs': stats['total_qa_pairs'],
            'average_qa_per_subgraph': stats['total_qa_pairs'] / max(1, stats['total_subgraphs']),
            'average_subgraphs_per_graph': stats['total_subgraphs'] / max(1, stats['total_graphs']),
        },
        'obfuscation_statistics': obfuscation_stats,
        'node_type_distribution': dict(stats['node_type_distribution']),
        'depth_distribution': dict(stats['depth_distribution']),
        'graph_details': stats['graphs'],
    }
    
    # Save statistics file
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with stats_file.open('w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved statistics to: {stats_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 Extraction complete")
    print(f"{'='*80}")
    print(f"Total graphs:                {stats['total_graphs']}")
    print(f"Total subgraphs:             {stats['total_subgraphs']}")
    print(f"Total QA pairs:              {stats['total_qa_pairs']}")
    print(f"Avg QA pairs per subgraph:   {stats['total_qa_pairs'] / max(1, stats['total_subgraphs']):.2f}")
    print(f"Avg subgraphs per graph:     {stats['total_subgraphs'] / max(1, stats['total_graphs']):.2f}")
    print(f"\nObfuscation statistics:")
    print(f"  Skipped:           {obfuscation_stats['skipped']}")
    print(f"  With obfuscation:  {obfuscation_stats['with_obfuscation']}")
    print(f"  Avg iterations:    {obfuscation_stats['average_iterations']:.2f}")
    print(f"  Max iterations:    {obfuscation_stats['max_iterations']}")
    print(f"  Iteration distribution:")
    for count_str in sorted([k for k in stats['obfuscation_distribution'].keys() if k != 'skipped'], key=lambda x: int(x) if x.isdigit() else 999):
        num = stats['obfuscation_distribution'][count_str]
        percentage = (num / stats['total_qa_pairs']) * 100
        print(f"    {count_str} iter(s): {num} ({percentage:.1f}%)")
    print(f"\nNode type distribution:")
    for node_type, count in sorted(stats['node_type_distribution'].items()):
        percentage = (count / stats['total_qa_pairs']) * 100
        print(f"  {node_type}: {count} ({percentage:.1f}%)")
    print(f"\nDepth distribution:")
    for depth_str in sorted(stats['depth_distribution'].keys(), key=lambda x: int(x) if x.isdigit() else 999):
        count = stats['depth_distribution'][depth_str]
        percentage = (count / stats['total_qa_pairs']) * 100
        print(f"  Depth {depth_str}: {count} ({percentage:.1f}%)")
    print(f"{'='*80}\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Extract QA pairs from the output of 7_layerwise_generate_root_desc.py.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input file path (output of 7_layerwise_generate_root_desc.py).')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (QA pairs; default: data_synthesis/result/<stem>_qa.jsonl).')
    parser.add_argument('--stats_file', type=str, default=None,
                        help='Statistics file path (default: data_synthesis/result/<stem>_qa_stats.json).')
    
    args = parser.parse_args()
    
    # Handle paths
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Generate default output paths
    result_dir = Path("./data_synthesis/result")
    
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_filename = f"{input_path.stem}_qa.jsonl"
        output_path = result_dir / output_filename
    
    if args.stats_file:
        stats_path = Path(args.stats_file)
    else:
        stats_filename = f"{input_path.stem}_qa_stats.json"
        stats_path = result_dir / stats_filename
    
    # Run extraction
    extract_qa_pairs(input_path, output_path, stats_path)


if __name__ == '__main__':
    main()
