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
v2 Step-4: Knowledge Graph Visualization Tool

Reads graph data from data_synthesis/cache/cache_3b/*.jsonl files and visualizes them.
Supports 2D graph visualization, statistics display, and interactive exploration.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import numpy as np


# ============================================================================
# Utility Functions
# ============================================================================

def load_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return a list of records."""
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


# ============================================================================
# Knowledge Graph Visualizer
# ============================================================================

class V2GraphVisualizer:
    """v2 Knowledge Graph Visualizer (reads from JSONL output)."""
    
    def __init__(self, jsonl_path: Path):
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.jsonl_path}")
        
        # Load data
        self.records = load_jsonl(self.jsonl_path)
        print(f"✅ Loaded {len(self.records)} records")
        
        # Index of the currently processed record
        self.current_index = 0
    
    def get_record(self, index: int = 0) -> Optional[Dict[str, Any]]:
        """Get the record at the specified index."""
        if 0 <= index < len(self.records):
            return self.records[index]
        return None
    
    def build_networkx_graph(self, record: Dict[str, Any]) -> nx.DiGraph:
        """Build a NetworkX graph from a record."""
        G = nx.DiGraph()
        
        entities = record.get("entities", [])
        relations = record.get("relations", [])
        node_metrics = record.get("node_metrics", {})
        
        # Map entity IDs to entity names
        entity_id_to_name = {}
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            entity_name = entity.get("canonical_name", "")
            if entity_id and entity_name:
                entity_id_to_name[entity_id] = entity_name
        
        # Add nodes
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            entity_name = entity.get("canonical_name", "")
            
            if not entity_id or not entity_name:
                continue
            
            # Get node metrics
            metrics = node_metrics.get(entity_id, {})
            
            # Add node attributes
            node_attrs = {
                "entity_id": entity_id,
                "canonical_name": entity_name,
                "entity_type": entity.get("entity_type", "Other"),
                "description": entity.get("description", ""),
                "degree": metrics.get("degree", 0),
                "pagerank": metrics.get("pagerank", 0.0),
                "betweenness": metrics.get("betweenness", 0.0),
                "is_seed": metrics.get("is_seed", False),
                "depth_from_seed": metrics.get("depth_from_seed", -1),
            }
            
            G.add_node(entity_id, **node_attrs)
        
        # Add edges
        for relation in relations:
            source_entity = relation.get("source_entity", "")
            target_entity = relation.get("target_entity", "")
            relationship_type = relation.get("relationship_type", "related_to")
            description = relation.get("description", "")
            
            # Look up entity IDs
            source_id = None
            target_id = None
            
            for entity in entities:
                if entity.get("canonical_name") == source_entity:
                    source_id = entity.get("entity_id")
                if entity.get("canonical_name") == target_entity:
                    target_id = entity.get("entity_id")
            
            if source_id and target_id:
                edge_attrs = {
                    "relationship_type": relationship_type,
                    "description": description,
                    "source_urls": relation.get("source_urls", []),
                    "evidence_spans": relation.get("evidence_spans", []),
                }
                G.add_edge(source_id, target_id, **edge_attrs)
        
        return G
    
    def print_summary(self, record: Dict[str, Any]):
        """Print graph summary information"""
        print("\n" + "="*60)
        print("📊 Knowledge Graph Summary")
        print("="*60)
        
        question = record.get("question", "")
        # answer may be in either 'answer' or 'golden_answer' field
        golden_answer = record.get("golden_answer", "") or record.get("answer", "")
        seed_entities = record.get("seed_entities", [])
        graph_stats = record.get("graph_statistics", {})
        
        print(f"\nQuestion: {question[:100]}...")
        print(f"Answer: {golden_answer}")
        print(f"Seed Entities: {', '.join(seed_entities) if seed_entities else 'N/A'}")
        
        print(f"\n📈 Graph Statistics:")
        print(f"  - Entities: {graph_stats.get('num_entities', 0)}")
        print(f"  - Relations: {graph_stats.get('num_relations', 0)}")
        print(f"  - Connected Components: {graph_stats.get('num_components', 0)}")
        print(f"  - Has Cycles: {graph_stats.get('has_cycles', False)}")
        print(f"  - Cycle Count (Undirected): {graph_stats.get('undirected_cycle_count', graph_stats.get('cycle_count', 0))}")
        print(f"  - Cycle Count (Directed): {graph_stats.get('directed_cycle_count', 0)}")
        print(f"  - Average Degree: {graph_stats.get('avg_degree', 0.0):.2f}")
        print(f"  - Graph Density: {graph_stats.get('graph_density', 0.0):.4f}")
        print(f"  - Max Depth: {graph_stats.get('max_depth', 0)}")
        
        # Entity type distribution
        entities = record.get("entities", [])
        entity_types = Counter([e.get("entity_type", "Unknown") for e in entities])
        if entity_types:
            print(f"\n📋 Entity Type Distribution:")
            for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {entity_type}: {count}")
        
        # Relation type distribution
        relations = record.get("relations", [])
        relation_types = Counter([r.get("relationship_type", "unknown") for r in relations])
        
        # Count original vs. reverse relations
        original_relations = [r for r in relations if r.get("relation_aug", 0) == 0]
        reverse_relations = [r for r in relations if r.get("relation_aug", 0) == 1]
        
        if relation_types:
            print(f"\n🔗 Relation Statistics:")
            print(f"  - Total Relations: {len(relations)}")
            print(f"  - Original Relations: {len(original_relations)}")
            print(f"  - Reverse Relations: {len(reverse_relations)}")
            
            print(f"\n🔗 Relation Type Distribution (Top 10):")
            for rel_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {rel_type}: {count}")
        
        # Top entities (by degree)
        node_metrics = record.get("node_metrics", {})
        entities_with_degree = []
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            metrics = node_metrics.get(entity_id, {})
            entities_with_degree.append({
                "name": entity.get("canonical_name", ""),
                "type": entity.get("entity_type", ""),
                "degree": metrics.get("degree", 0),
                "pagerank": metrics.get("pagerank", 0.0),
                "is_seed": metrics.get("is_seed", False),
            })
        
        entities_with_degree.sort(key=lambda x: x["degree"], reverse=True)
        if entities_with_degree:
            print(f"\n⭐ Top 10 Entities (by Degree):")
            for i, entity in enumerate(entities_with_degree[:10], 1):
                seed_mark = "🌱" if entity["is_seed"] else "  "
                print(f"  {i}. {seed_mark} {entity['name']} ({entity['type']}) - Degree: {entity['degree']}, PageRank: {entity['pagerank']:.4f}")
    
    def plot_2d_graph(
        self,
        record: Dict[str, Any],
        save_path: Optional[str] = None,
        max_nodes: int = 100,
        show_all_labels: bool = False,
        layout: str = "spring",
    ):
        """Plot a 2D graph visualization."""
        # Build NetworkX graph
        G = self.build_networkx_graph(record)
        
        if G.number_of_nodes() == 0:
            print("❌ No nodes in graph, cannot visualize")
            return
        
        print(f"\n📊 Building graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # If too many nodes, only show top nodes
        if G.number_of_nodes() > max_nodes:
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G = G.subgraph([node for node, _ in top_nodes])
            print(f"⚠️  Too many nodes, showing only top {max_nodes} nodes by degree")
        
        # Create figure
        plt.figure(figsize=(20, 16))
        
        # Calculate layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        
        # Get node attributes
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        min_degree = min(degrees.values()) if degrees else 1
        
        # Node sizes and colors
        node_sizes = []
        node_colors = []
        node_labels = {}
        
        # Entity type color mapping
        entity_types = set()
        for node in G.nodes():
            entity_type = G.nodes[node].get("entity_type", "Unknown")
            entity_types.add(entity_type)
        
        # Create color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(entity_types)))
        type_color_map = {entity_type: colors[i] for i, entity_type in enumerate(sorted(entity_types))}
        
        # Seed entities use special color
        seed_color = '#FFD700'  # Gold
        
        for node in G.nodes():
            node_data = G.nodes[node]
            degree = degrees[node]
            
            # Node size (based on degree)
            normalized_degree = (degree - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 0.5
            size = 300 + normalized_degree * 1000
            node_sizes.append(size)
            
            # Node color (seed entities in gold, others by type)
            is_seed = node_data.get("is_seed", False)
            if is_seed:
                node_colors.append(seed_color)
            else:
                entity_type = node_data.get("entity_type", "Unknown")
                node_colors.append(type_color_map[entity_type])
            
            # Node labels
            entity_name = node_data.get("canonical_name", node)
            entity_type = node_data.get("entity_type", "Unknown")
            description = node_data.get("description", "")
            
            # Truncate description
            if len(description) > 40:
                description = description[:37] + "..."
            
            # Create label
            label_parts = [f"{entity_name}"]
            if entity_type != "Unknown":
                label_parts.append(f"Type: {entity_type}")
            label_parts.append(f"Degree: {degree}")
            if is_seed:
                label_parts.append("🌱 SEED")
            
            node_labels[node] = "\n".join(label_parts)
        
        # Edge colors and widths (based on relation type)
        edge_colors = []
        edge_widths = []
        edge_labels = {}
        
        # Extended relation type color mapping (covering all core relations)
        relation_type_colors = {
            # Person relations
            "worked_at": "blue",
            "born_in": "green",
            "died_in": "red",
            "educated_at": "cyan",
            "created": "orange",
            "awarded": "gold",
            "member_of": "magenta",
            # Organization relations
            "located_in": "purple",
            "founded_by": "brown",
            "owns": "darkblue",
            "partner_with": "teal",
            # Event relations
            "occurred_in": "coral",
            "participated_by": "lime",
            "caused_by": "maroon",
            # Product relations
            "created_by": "navy",
            "published_by": "olive",
            "released_in": "indigo",
            # General relations
            "related_to": "gray",
            "part_of": "silver",
            "mentioned_in": "lightgray",
        }
        
        for edge in G.edges():
            edge_data = G.get_edge_data(edge[0], edge[1], {})
            rel_type = edge_data.get("relationship_type", "related_to")
            
            # Edge color
            edge_color = relation_type_colors.get(rel_type, "gray")
            edge_colors.append(edge_color)
            
            # Edge width (based on relation importance)
            edge_widths.append(1.5)
            
            # Edge labels (only show relation type)
            if show_all_labels:
                edge_labels[(edge[0], edge[1])] = rel_type
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            edgecolors='black',
            linewidths=1.5,
        )
        
        # Draw node labels
        if show_all_labels:
            nx.draw_networkx_labels(
                G, pos,
                labels=node_labels,
                font_size=7,
                font_weight='bold',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='gray'
                )
            )
        else:
            # Only show high-degree nodes and seed nodes
            important_nodes = {
                node: node_labels[node]
                for node in G.nodes()
                if degrees[node] > 2 or G.nodes[node].get("is_seed", False)
            }
            nx.draw_networkx_labels(
                G, pos,
                labels=important_nodes,
                font_size=8,
                font_weight='bold',
            )
        
        # Draw edge labels
        if show_all_labels and edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.5)
            )
        
        # Create legends
        legend_elements = []
        
        # Entity type legend
        for entity_type, color in sorted(type_color_map.items()):
            legend_elements.append(mpatches.Patch(color=color, label=entity_type))
        
        # Seed entity legend
        legend_elements.append(mpatches.Patch(color=seed_color, label="🌱 Seed Entity"))
        
        plt.legend(
            handles=legend_elements,
            title="Entity Types",
            loc='upper left',
            bbox_to_anchor=(0, 1),
            fontsize=9
        )
        
        # Relation type legend - show all defined relations
        relation_legend = [
            mpatches.Patch(color=color, label=rel_type)
            for rel_type, color in sorted(relation_type_colors.items())
        ]
        # Only add "Other Relations" if there are relations not in our mapping
        all_relation_types = set()
        for edge in G.edges():
            edge_data = G.get_edge_data(edge[0], edge[1], {})
            rel_type = edge_data.get("relationship_type", "related_to")
            all_relation_types.add(rel_type)
        
        if all_relation_types - set(relation_type_colors.keys()):
            relation_legend.append(mpatches.Patch(color="gray", label="Other Relations"))
        
        plt.legend(
            handles=relation_legend,
            title="Relation Types",
            loc='upper right',
            bbox_to_anchor=(1, 1),
            fontsize=8,
            ncol=2  # Two columns for better layout
        )
        
        # Title
        graph_stats = record.get("graph_statistics", {})
        title = f"Knowledge Graph Visualization\n"
        title += f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | "
        title += f"Entity Types: {len(entity_types)} | Max Degree: {max_degree}"
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add statistics text box
        stats_text = f"Graph Statistics:\n"
        stats_text += f"• Avg Degree: {sum(degrees.values())/len(degrees):.2f}\n"
        stats_text += f"• Components: {graph_stats.get('num_components', 0)}\n"
        stats_text += f"• Density: {graph_stats.get('graph_density', 0.0):.4f}\n"
        stats_text += f"• Has Cycles: {graph_stats.get('has_cycles', False)}\n"
        stats_text += f"• Max Depth: {graph_stats.get('max_depth', 0)}"
        
        plt.text(
            0.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
            fontsize=10
        )
        
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Graph saved to: {save_path}")
        
        plt.show()
    
    def export_graphml(self, record: Dict[str, Any], output_path: Path):
        """Export to GraphML format (can be used by other tools)"""
        G = self.build_networkx_graph(record)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(G, output_path)
        print(f"✅ GraphML exported to: {output_path}")


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="v2 Knowledge Graph Visualization Tool")
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=50,
        help="Record index to visualize (default: 0)",
    )
    parser.add_argument(
        "--mode",
        choices=["summary", "plot", "export", "all"],
        default="all",
        help="Visualization mode",
    )
    parser.add_argument(
        "--save-plot",
        help="Path to save 2D graph image",
    )
    parser.add_argument(
        "--export-graphml",
        help="Path to export GraphML format",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=100,
        help="Maximum number of nodes to display in 2D graph (default: 100)",
    )
    parser.add_argument(
        "--show-all-labels",
        action="store_true",
        help="Show all node and edge labels",
    )
    parser.add_argument(
        "--layout",
        choices=["spring", "kamada_kawai", "circular"],
        default="spring",
        help="Graph layout algorithm (default: spring)",
    )
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = V2GraphVisualizer(args.input)
        
        # Get record
        record = visualizer.get_record(args.index)
        if not record:
            print(f"❌ Record index {args.index} out of range (total: {len(visualizer.records)} records)")
            return 1
        
        print(f"\n📝 Processing record {args.index + 1}/{len(visualizer.records)}")
        
        # Execute different modes
        if args.mode in ["summary", "all"]:
            visualizer.print_summary(record)
        
        if args.mode in ["plot", "all"]:
            visualizer.plot_2d_graph(
                record,
                save_path=args.save_plot,
                max_nodes=args.max_nodes,
                show_all_labels=args.show_all_labels,
                layout=args.layout,
            )
        
        if args.mode in ["export", "all"]:
            if args.export_graphml:
                visualizer.export_graphml(record, Path(args.export_graphml))
            else:
                # Default export path
                default_path = Path(args.input).parent / f"graph_{args.index}.graphml"
                visualizer.export_graphml(record, default_path)
        
        print("\n✅ Visualization completed")
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
