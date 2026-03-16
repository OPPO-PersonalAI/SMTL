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
v2 Step-6: Subgraph Visualization Tool

Reads subgraph data from data_synthesis/cache/cache_5/*_subgraphs.jsonl files and visualizes them.
Supports 2D graph visualization, topology display, and statistics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# Subgraph Visualizer
# ============================================================================

class SubgraphVisualizer:
    """Subgraph visualizer (reads from JSONL output)."""
    
    def __init__(self, jsonl_path: Path):
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.jsonl_path}")
        
        # Load data
        self.records = load_jsonl(self.jsonl_path)
        print(f"✅ Loaded {len(self.records)} subgraph records")
        
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
        target_entity = record.get("target_entity", "")
        topology = record.get("topology", {})
        
        # Create entity name to entity data mapping
        entity_map = {}
        for entity in entities:
            entity_name = entity.get("canonical_name", "")
            if entity_name:
                entity_map[entity_name] = entity
        
        # Add nodes
        for entity in entities:
            entity_name = entity.get("canonical_name", "")
            if not entity_name:
                continue
            
            # Get node attributes
            depth = entity.get("depth_in_subgraph", -1)
            is_target = (entity_name == target_entity)
            
            node_attrs = {
                "canonical_name": entity_name,
                "entity_type": entity.get("entity_type", "Other"),
                "description": entity.get("description", ""),
                "key_info": entity.get("key_info", ""),  # key_info support
                "depth": depth,
                "is_target": is_target,
                "used_llm": entity.get("used_llm", False),
            }
            
            G.add_node(entity_name, **node_attrs)
        
        # Add edges
        for relation in relations:
            source = relation.get("source_entity", "")
            target = relation.get("target_entity", "")
            relationship_type = relation.get("relationship_type", "related_to")
            relation_aug = relation.get("relation_aug", 0)  # 0=original, 1=reversed
            
            if source in G.nodes() and target in G.nodes():
                edge_attrs = {
                    "relationship_type": relationship_type,
                    "relation_aug": relation_aug,
                    "description": relation.get("description", ""),
                }
                G.add_edge(source, target, **edge_attrs)
        
        return G
    
    def print_summary(self, record: Dict[str, Any]):
        """Print subgraph summary information."""
        print("\n" + "="*60)
        print("📊 Subgraph Summary")
        print("="*60)
        
        target_entity = record.get("target_entity", "")
        parent_graph = record.get("parent_graph", {})
        question = parent_graph.get("question", "")
        golden_answer = parent_graph.get("golden_answer", "")
        topology = record.get("topology", {})
        
        print(f"\nQuestion: {question[:100]}...")
        print(f"Answer: {golden_answer}")
        print(f"Target Entity: {target_entity}")
        
        print(f"\n📈 Subgraph Statistics:")
        print(f"  - Entities: {record.get('num_entities', 0)}")
        print(f"  - Relations: {record.get('num_relations', 0)}")
        print(f"  - Cycle Count: {record.get('cycle_count', 0)}")
        print(f"  - Used LLM Entities: {record.get('used_llm_count', 0)}")
        print(f"  - Depth Range: {record.get('depth_range', [])}")
        print(f"  - Max Depth: {topology.get('max_depth', 0)}")
        
        # Depth distribution
        depth_dist = topology.get("depth_distribution", {})
        if depth_dist:
            print(f"\n📋 Depth Distribution:")
            for depth in sorted([int(d) for d in depth_dist.keys()]):
                count = depth_dist[str(depth)]
                print(f"  - Depth {depth}: {count} entities")
        
        # Topology
        layer_entities = topology.get("layer_entities", {})
        parent_child_edges = topology.get("parent_child_edges", [])
        intra_layer_edges = topology.get("intra_layer_edges", [])
        
        print(f"\n🔗 Topology Structure:")
        print(f"  - Parent-Child Edges: {len(parent_child_edges)}")
        print(f"  - Intra-Layer Edges: {len(intra_layer_edges)}")
        
        if layer_entities:
            print(f"\n📊 Layer Entities:")
            for depth in sorted([int(d) for d in layer_entities.keys()]):
                entities_in_layer = layer_entities[str(depth)]
                print(f"  - Depth {depth}: {len(entities_in_layer)} entities")
                for entity in entities_in_layer[:5]:  # show first 5 only
                    print(f"      • {entity}")
                if len(entities_in_layer) > 5:
                    print(f"      ... and {len(entities_in_layer) - 5} more")
        
        # Entity type distribution
        entities = record.get("entities", [])
        entity_types = defaultdict(int)
        for entity in entities:
            entity_type = entity.get("entity_type", "Unknown")
            entity_types[entity_type] += 1
        
        if entity_types:
            print(f"\n📋 Entity Type Distribution:")
            for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {entity_type}: {count}")
        
        # Relation type distribution
        relations = record.get("relations", [])
        relation_types = defaultdict(int)
        for rel in relations:
            rel_type = rel.get("relationship_type", "unknown")
            relation_types[rel_type] += 1
        
        if relation_types:
            print(f"\n🔗 Relation Type Distribution (Top 10):")
            for rel_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {rel_type}: {count}")
    
    def plot_2d_graph(
        self,
        record: Dict[str, Any],
        save_path: Optional[str] = None,
        show_all_labels: bool = False,
        layout: str = "hierarchical",
    ):
        """Plot a 2D subgraph visualization."""
        # Build NetworkX graph
        G = self.build_networkx_graph(record)
        
        if G.number_of_nodes() == 0:
            print("❌ No nodes in graph, cannot visualize")
            return
        
        print(f"\n📊 Building subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        target_entity = record.get("target_entity", "")
        topology = record.get("topology", {})
        layer_entities = topology.get("layer_entities", {})
        
        # Create figure
        plt.figure(figsize=(20, 16))
        
        # Calculate layout
        if layout == "hierarchical":
            # Hierarchical layout: arrange by depth
            pos = self._hierarchical_layout(G, layer_entities, target_entity)
        elif layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
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
        
        # Target entity uses a special color
        target_color = '#FFD700'  # Gold
        
        for node in G.nodes():
            node_data = G.nodes[node]
            degree = degrees[node]
            is_target = node_data.get("is_target", False)
            depth = node_data.get("depth", -1)
            
            # Node size (based on degree; target entity is larger)
            normalized_degree = (degree - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 0.5
            base_size = 300 + normalized_degree * 1000
            if is_target:
                base_size *= 1.5  # Target entity is larger
            node_sizes.append(base_size)
            
            # Node color (gold for target, otherwise by type)
            if is_target:
                node_colors.append(target_color)
            else:
                entity_type = node_data.get("entity_type", "Unknown")
                node_colors.append(type_color_map[entity_type])
            
            # Node label
            entity_name = node_data.get("canonical_name", node)
            entity_type = node_data.get("entity_type", "Unknown")
            description = node_data.get("description", "")
            key_info = node_data.get("key_info", "")  # prefer key_info if available
            
            # Build label
            label_parts = [f"{entity_name}"]
            if entity_type != "Unknown":
                label_parts.append(f"Type: {entity_type}")
            if depth >= 0:
                label_parts.append(f"Depth: {depth}")
            if is_target:
                label_parts.append("🎯 TARGET")
            
            # Add description (prefer key_info, fallback to description)
            desc_text = key_info if key_info else description
            if desc_text:
                # Truncate long descriptions (max 80 chars)
                if len(desc_text) > 80:
                    desc_text = desc_text[:77] + "..."
                label_parts.append(f"Desc: {desc_text}")
            
            node_labels[node] = "\n".join(label_parts)
        
        # Edge colors and widths (distinguish parent-child vs intra-layer)
        edge_colors = []
        edge_widths = []
        edge_labels = {}
        
        # Get topology edge info
        parent_child_edges_set = set()
        intra_layer_edges_set = set()
        
        for edge in topology.get("parent_child_edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            parent_child_edges_set.add((source, target))
        
        for edge in topology.get("intra_layer_edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            intra_layer_edges_set.add((source, target))
        
        # Relation type color mapping
        relation_type_colors = {
            "worked_at": "blue",
            "born_in": "green",
            "died_in": "red",
            "educated_at": "cyan",
            "created": "orange",
            "awarded": "gold",
            "member_of": "magenta",
            "located_in": "purple",
            "founded_by": "brown",
            "owns": "darkblue",
            "partner_with": "teal",
            "occurred_in": "coral",
            "participated_by": "lime",
            "caused_by": "maroon",
            "created_by": "navy",
            "published_by": "olive",
            "released_in": "indigo",
            "related_to": "gray",
            "part_of": "silver",
            "mentioned_in": "lightgray",
        }
        
        for edge in G.edges():
            edge_data = G.get_edge_data(edge[0], edge[1], {})
            rel_type = edge_data.get("relationship_type", "related_to")
            rel_desc = edge_data.get("description", "")
            
            # Determine edge type
            edge_key = (edge[0], edge[1])
            if edge_key in parent_child_edges_set:
                # Parent-child edge: thicker solid line
                edge_widths.append(2.5)
                edge_colors.append(relation_type_colors.get(rel_type, "blue"))
            elif edge_key in intra_layer_edges_set:
                # Intra-layer edge: thinner dashed line
                edge_widths.append(1.5)
                edge_colors.append(relation_type_colors.get(rel_type, "green"))
            else:
                # Other edges
                edge_widths.append(1.0)
                edge_colors.append(relation_type_colors.get(rel_type, "gray"))
            
            # Edge label (description only, not relation type)
            if rel_desc:
                # Truncate long descriptions (max 50 chars)
                desc_short = rel_desc[:47] + "..." if len(rel_desc) > 50 else rel_desc
                edge_labels[(edge[0], edge[1])] = desc_short
            else:
                edge_labels[(edge[0], edge[1])] = ""
        
        # Draw edges in two passes: parent-child then intra-layer
        # Draw parent-child edges first
        parent_child_edges_list = [
            (u, v) for u, v in G.edges()
            if (u, v) in parent_child_edges_set
        ]
        if parent_child_edges_list:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=parent_child_edges_list,
                edge_color=[relation_type_colors.get(G[u][v].get("relationship_type", "blue"), "blue") for u, v in parent_child_edges_list],
                width=2.5,
                alpha=0.7,
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                style='solid',
            )
        
        # Then draw intra-layer edges
        intra_layer_edges_list = [
            (u, v) for u, v in G.edges()
            if (u, v) in intra_layer_edges_set
        ]
        if intra_layer_edges_list:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=intra_layer_edges_list,
                edge_color=[relation_type_colors.get(G[u][v].get("relationship_type", "green"), "green") for u, v in intra_layer_edges_list],
                width=1.5,
                alpha=0.5,
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                style='dashed',
            )
        
        # Draw other edges
        other_edges_list = [
            (u, v) for u, v in G.edges()
            if (u, v) not in parent_child_edges_set and (u, v) not in intra_layer_edges_set
        ]
        if other_edges_list:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=other_edges_list,
                edge_color="gray",
                width=1.0,
                alpha=0.3,
                arrows=True,
                arrowsize=10,
                arrowstyle='->',
                style='dotted',
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
            # Show all node labels (including description)
            nx.draw_networkx_labels(
                G, pos,
                labels=node_labels,
                font_size=6,
                font_weight='bold',
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor='white',
                    alpha=0.85,
                    edgecolor='gray',
                    linewidth=1
                )
            )
        else:
            # Show only important and target nodes (also with description)
            important_nodes = {
                node: node_labels[node]
                for node in G.nodes()
                if G.nodes[node].get("is_target", False) or degrees[node] > 1
            }
            nx.draw_networkx_labels(
                G, pos,
                labels=important_nodes,
                font_size=7,
                font_weight='bold',
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor='white',
                    alpha=0.85,
                    edgecolor='gray',
                    linewidth=1
                )
            )
        
        # Draw edge labels (description only, no relation type)
        # Filter out empty labels
        if edge_labels:
            display_edge_labels = {
                (u, v): label
                for (u, v), label in edge_labels.items()
                if label and label.strip()  # filter empty labels
            }
            
            if display_edge_labels:
                nx.draw_networkx_edge_labels(
                    G, pos,
                    edge_labels=display_edge_labels,
                    font_size=6,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor='yellow',
                        alpha=0.7,
                        edgecolor='orange',
                        linewidth=0.5
                    )
                )
        
        # Create legends
        legend_elements = []
        
        # Entity type legend
        for entity_type, color in sorted(type_color_map.items()):
            legend_elements.append(mpatches.Patch(color=color, label=entity_type))
        
        # Target entity legend
        legend_elements.append(mpatches.Patch(color=target_color, label="🎯 Target Entity"))
        
        plt.legend(
            handles=legend_elements,
            title="Entity Types",
            loc='upper left',
            bbox_to_anchor=(0, 1),
            fontsize=9
        )
        
        # Edge type legend
        edge_legend = [
            mpatches.Patch(color="blue", label="Parent-Child Edge (Solid)"),
            mpatches.Patch(color="green", label="Intra-Layer Edge (Dashed)"),
            mpatches.Patch(color="gray", label="Other Edge (Dotted)"),
        ]
        plt.legend(
            handles=edge_legend,
            title="Edge Types",
            loc='upper right',
            bbox_to_anchor=(1, 1),
            fontsize=8,
        )
        
        # Title
        title = f"Subgraph Visualization\n"
        title += f"Target: {target_entity}\n"
        title += f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | "
        title += f"Cycles: {record.get('cycle_count', 0)} | Max Depth: {topology.get('max_depth', 0)}"
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add statistics text box
        stats_text = f"Subgraph Statistics:\n"
        stats_text += f"• Entities: {record.get('num_entities', 0)}\n"
        stats_text += f"• Relations: {record.get('num_relations', 0)}\n"
        stats_text += f"• Cycles: {record.get('cycle_count', 0)}\n"
        stats_text += f"• Used LLM: {record.get('used_llm_count', 0)}\n"
        stats_text += f"• Parent-Child Edges: {len(topology.get('parent_child_edges', []))}\n"
        stats_text += f"• Intra-Layer Edges: {len(topology.get('intra_layer_edges', []))}"
        
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
    
    def _hierarchical_layout(
        self,
        G: nx.DiGraph,
        layer_entities: Dict[str, List[str]],
        target_entity: str,
    ) -> Dict[str, tuple]:
        """Create a hierarchical layout (arranged by depth from top to bottom)."""
        pos = {}
        
        # Get all depths
        depths = sorted([int(d) for d in layer_entities.keys()])
        max_depth = max(depths) if depths else 0
        
        # Calculate y-coordinate for each layer (top to bottom)
        y_spacing = 1.0 / (max_depth + 1) if max_depth > 0 else 1.0
        
        for depth in depths:
            entities_in_layer = layer_entities[str(depth)]
            y = 1.0 - depth * y_spacing
            
            # Distribute evenly along x-axis
            num_entities = len(entities_in_layer)
            if num_entities == 1:
                x_positions = [0.5]
            else:
                x_positions = np.linspace(0.1, 0.9, num_entities)
            
            for i, entity in enumerate(entities_in_layer):
                if entity in G.nodes():
                    pos[entity] = (x_positions[i], y)
        
        # Fallback position for nodes not in layer_entities
        for node in G.nodes():
            if node not in pos:
                pos[node] = (0.5, 0.5)
        
        return pos
    
    def export_graphml(self, record: Dict[str, Any], output_path: Path):
        """Export to GraphML format (can be used by other tools)"""
        G = self.build_networkx_graph(record)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(G, output_path)
        print(f"✅ GraphML exported to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="v2 Subgraph Visualization Tool")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file path (from 5_extract_subgraph.py)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
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
        "--show-all-labels",
        action="store_true",
        help="Show all node and edge labels",
    )
    parser.add_argument(
        "--layout",
        choices=["hierarchical", "spring", "kamada_kawai"],
        default="hierarchical",
        help="Graph layout algorithm (default: hierarchical)",
    )
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = SubgraphVisualizer(args.input)
        
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
                show_all_labels=args.show_all_labels,
                layout=args.layout,
            )
        
        if args.mode in ["export", "all"]:
            if args.export_graphml:
                visualizer.export_graphml(record, Path(args.export_graphml))
            else:
                # Default export path
                default_path = Path(args.input).parent / f"subgraph_{args.index}.graphml"
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
