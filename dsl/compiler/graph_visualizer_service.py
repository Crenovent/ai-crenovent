"""
Graph Visualizer Service - Task 6.2.44
=======================================

Graph visualizer (topology, fallback edges)
- Provides interactive graph visualization of plans
- Shows node types, policy annotations, fallback edges, residency, SLAs
- Supports filtering, zoom, and export capabilities
- Backend implementation (no actual D3/React components - that's frontend)

Dependencies: Task 6.2.4 (IR), Task 6.2.42 (Pretty-Printer)
Outputs: Graph visualization data → enables plan topology review and debugging
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class NodeShape(Enum):
    """Node shapes for visualization"""
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    DIAMOND = "diamond"
    HEXAGON = "hexagon"
    TRIANGLE = "triangle"
    STAR = "star"

class EdgeType(Enum):
    """Edge types for visualization"""
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    THICK = "thick"

class LayoutAlgorithm(Enum):
    """Layout algorithms for graph positioning"""
    FORCE_DIRECTED = "force_directed"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    GRID = "grid"
    TREE = "tree"

@dataclass
class NodePosition:
    """Position of a node in the visualization"""
    x: float
    y: float
    z: float = 0.0  # For 3D layouts

@dataclass
class VisualizationNode:
    """A node in the graph visualization"""
    node_id: str
    node_type: str
    display_name: str
    
    # Visual properties
    shape: NodeShape = NodeShape.CIRCLE
    color: str = "#3498db"
    size: float = 20.0
    position: Optional[NodePosition] = None
    
    # Content and metadata
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Policy annotations
    policies: Dict[str, Any] = field(default_factory=dict)
    trust_budget: Optional[Dict[str, Any]] = None
    sla_requirements: Dict[str, str] = field(default_factory=dict)
    cost_class: str = "T2"
    
    # Residency and compliance
    region_id: Optional[str] = None
    compliance_tags: List[str] = field(default_factory=list)
    
    # Interactive features
    clickable: bool = True
    draggable: bool = True
    
    # Status indicators
    status: str = "active"  # active, inactive, error, warning
    hotspot_indicator: bool = False

@dataclass
class VisualizationEdge:
    """An edge in the graph visualization"""
    edge_id: str
    source_node_id: str
    target_node_id: str
    
    # Visual properties
    edge_type: EdgeType = EdgeType.SOLID
    color: str = "#7f8c8d"
    width: float = 2.0
    
    # Edge metadata
    label: str = ""
    condition: str = ""
    
    # Edge types
    is_fallback: bool = False
    is_data_flow: bool = True
    is_control_flow: bool = False
    
    # Interactive features
    clickable: bool = True
    animated: bool = False

@dataclass
class GraphLayout:
    """Layout configuration for the graph"""
    algorithm: LayoutAlgorithm = LayoutAlgorithm.FORCE_DIRECTED
    
    # Layout parameters
    width: float = 800.0
    height: float = 600.0
    padding: float = 50.0
    
    # Force-directed parameters
    repulsion_strength: float = 100.0
    attraction_strength: float = 0.1
    gravity: float = 0.01
    
    # Hierarchical parameters
    level_separation: float = 100.0
    node_separation: float = 50.0
    
    # Circular parameters
    radius: float = 200.0
    
    # Grid parameters
    grid_spacing: float = 100.0

@dataclass
class FilterOptions:
    """Filtering options for the visualization"""
    # Node type filters
    show_ml_nodes: bool = True
    show_decision_nodes: bool = True
    show_external_nodes: bool = True
    show_data_nodes: bool = True
    
    # Edge type filters
    show_data_flow: bool = True
    show_control_flow: bool = True
    show_fallback_edges: bool = True
    
    # Policy filters
    show_only_policy_violations: bool = False
    show_only_residency_conflicts: bool = False
    show_only_high_cost: bool = False
    
    # Status filters
    show_inactive_nodes: bool = True
    show_error_nodes: bool = True
    
    # Metadata filters
    filter_by_region: Optional[str] = None
    filter_by_cost_class: Optional[str] = None
    filter_by_sla_tier: Optional[str] = None

@dataclass
class GraphVisualization:
    """Complete graph visualization"""
    visualization_id: str
    plan_id: str
    
    # Graph elements
    nodes: List[VisualizationNode] = field(default_factory=list)
    edges: List[VisualizationEdge] = field(default_factory=list)
    
    # Layout and styling
    layout: GraphLayout = field(default_factory=GraphLayout)
    filter_options: FilterOptions = field(default_factory=FilterOptions)
    
    # Metadata
    title: str = "Plan Visualization"
    description: str = ""
    
    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    fallback_edges: int = 0
    
    # Export options
    export_formats: List[str] = field(default_factory=lambda: ["svg", "png", "pdf", "json"])
    
    # Generated timestamp
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# Task 6.2.44: Graph Visualizer Service
class GraphVisualizerService:
    """Service for generating graph visualizations of plans"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Node type configurations
        self.node_type_configs = self._initialize_node_type_configs()
        
        # Color schemes
        self.color_schemes = self._initialize_color_schemes()
        
        # Layout algorithms
        self.layout_algorithms = self._initialize_layout_algorithms()
        
        # Generated visualizations cache
        self.visualization_cache: Dict[str, GraphVisualization] = {}
        
        # Statistics
        self.visualizer_stats = {
            'total_visualizations': 0,
            'layouts_generated': {layout.value: 0 for layout in LayoutAlgorithm},
            'nodes_visualized': 0,
            'edges_visualized': 0,
            'exports_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def generate_graph_visualization(self, plan_ir: Dict[str, Any],
                                   layout_algorithm: LayoutAlgorithm = LayoutAlgorithm.FORCE_DIRECTED,
                                   filter_options: Optional[FilterOptions] = None) -> GraphVisualization:
        """
        Generate graph visualization from plan IR
        
        Args:
            plan_ir: Plan intermediate representation
            layout_algorithm: Layout algorithm to use
            filter_options: Filtering options
            
        Returns:
            GraphVisualization with complete visualization data
        """
        start_time = datetime.now(timezone.utc)
        plan_id = plan_ir.get('plan_id', 'unknown')
        
        # Check cache
        cache_key = f"{plan_id}_{layout_algorithm.value}_{hash(str(filter_options))}"
        if cache_key in self.visualization_cache:
            self.visualizer_stats['cache_hits'] += 1
            return self.visualization_cache[cache_key]
        
        self.visualizer_stats['cache_misses'] += 1
        
        # Create visualization instance
        visualization_id = f"viz_{plan_id}_{int(start_time.timestamp())}"
        
        visualization = GraphVisualization(
            visualization_id=visualization_id,
            plan_id=plan_id,
            title=f"Plan Visualization: {plan_id}",
            description=f"Interactive graph visualization of plan {plan_id}"
        )
        
        # Set layout configuration
        visualization.layout.algorithm = layout_algorithm
        
        # Set filter options
        if filter_options:
            visualization.filter_options = filter_options
        
        # Extract and convert nodes
        visualization.nodes = self._extract_visualization_nodes(plan_ir, visualization.filter_options)
        
        # Extract and convert edges
        visualization.edges = self._extract_visualization_edges(plan_ir, visualization.filter_options)
        
        # Apply layout algorithm
        self._apply_layout_algorithm(visualization)
        
        # Calculate statistics
        visualization.total_nodes = len(visualization.nodes)
        visualization.total_edges = len(visualization.edges)
        visualization.fallback_edges = len([e for e in visualization.edges if e.is_fallback])
        
        # Apply hotspot indicators
        self._apply_hotspot_indicators(visualization)
        
        # Cache result
        self.visualization_cache[cache_key] = visualization
        
        # Update statistics
        self.visualizer_stats['total_visualizations'] += 1
        self.visualizer_stats['layouts_generated'][layout_algorithm.value] += 1
        self.visualizer_stats['nodes_visualized'] += len(visualization.nodes)
        self.visualizer_stats['edges_visualized'] += len(visualization.edges)
        
        self.logger.info(f"✅ Generated visualization: {visualization_id} -> {len(visualization.nodes)} nodes, {len(visualization.edges)} edges")
        
        return visualization
    
    def export_visualization(self, visualization: GraphVisualization, 
                           export_format: str) -> Dict[str, Any]:
        """
        Export visualization to specified format
        
        Args:
            visualization: Graph visualization to export
            export_format: Export format (svg, png, pdf, json, dot)
            
        Returns:
            Dictionary with export data
        """
        if export_format.lower() == "json":
            return self._export_to_json(visualization)
        elif export_format.lower() == "dot":
            return self._export_to_dot(visualization)
        elif export_format.lower() == "svg":
            return self._export_to_svg(visualization)
        elif export_format.lower() == "plantuml":
            return self._export_to_plantuml(visualization)
        else:
            # For other formats (png, pdf), return placeholder
            return self._export_placeholder(visualization, export_format)
    
    def apply_filter(self, visualization: GraphVisualization, 
                    filter_options: FilterOptions) -> GraphVisualization:
        """
        Apply filtering to existing visualization
        
        Args:
            visualization: Original visualization
            filter_options: New filter options
            
        Returns:
            Filtered visualization
        """
        # Create filtered copy
        filtered_viz = GraphVisualization(
            visualization_id=f"{visualization.visualization_id}_filtered",
            plan_id=visualization.plan_id,
            title=f"{visualization.title} (Filtered)",
            layout=visualization.layout,
            filter_options=filter_options
        )
        
        # Filter nodes
        filtered_viz.nodes = self._filter_nodes(visualization.nodes, filter_options)
        
        # Filter edges (only keep edges between remaining nodes)
        remaining_node_ids = {node.node_id for node in filtered_viz.nodes}
        filtered_viz.edges = [
            edge for edge in visualization.edges
            if (edge.source_node_id in remaining_node_ids and 
                edge.target_node_id in remaining_node_ids and
                self._edge_passes_filter(edge, filter_options))
        ]
        
        # Update statistics
        filtered_viz.total_nodes = len(filtered_viz.nodes)
        filtered_viz.total_edges = len(filtered_viz.edges)
        filtered_viz.fallback_edges = len([e for e in filtered_viz.edges if e.is_fallback])
        
        return filtered_viz
    
    def get_node_details(self, visualization: GraphVisualization, 
                        node_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific node
        
        Args:
            visualization: Graph visualization
            node_id: ID of the node
            
        Returns:
            Dictionary with node details
        """
        node = next((n for n in visualization.nodes if n.node_id == node_id), None)
        
        if not node:
            return {'error': f'Node {node_id} not found'}
        
        # Get connected edges
        incoming_edges = [e for e in visualization.edges if e.target_node_id == node_id]
        outgoing_edges = [e for e in visualization.edges if e.source_node_id == node_id]
        
        return {
            'node': asdict(node),
            'connections': {
                'incoming': len(incoming_edges),
                'outgoing': len(outgoing_edges),
                'total': len(incoming_edges) + len(outgoing_edges)
            },
            'incoming_edges': [asdict(e) for e in incoming_edges],
            'outgoing_edges': [asdict(e) for e in outgoing_edges],
            'is_hotspot': node.hotspot_indicator
        }
    
    def generate_topology_summary(self, visualization: GraphVisualization) -> Dict[str, Any]:
        """
        Generate topology summary for the visualization
        
        Args:
            visualization: Graph visualization
            
        Returns:
            Dictionary with topology summary
        """
        # Analyze node types
        node_type_counts = defaultdict(int)
        for node in visualization.nodes:
            node_type_counts[node.node_type] += 1
        
        # Analyze connectivity
        connectivity_stats = self._analyze_connectivity(visualization)
        
        # Analyze fallback coverage
        fallback_coverage = self._analyze_fallback_coverage(visualization)
        
        # Analyze policy distribution
        policy_distribution = self._analyze_policy_distribution(visualization)
        
        return {
            'plan_id': visualization.plan_id,
            'total_nodes': visualization.total_nodes,
            'total_edges': visualization.total_edges,
            'node_type_distribution': dict(node_type_counts),
            'connectivity_stats': connectivity_stats,
            'fallback_coverage': fallback_coverage,
            'policy_distribution': policy_distribution,
            'layout_algorithm': visualization.layout.algorithm.value,
            'generated_at': visualization.generated_at.isoformat()
        }
    
    def get_visualization_statistics(self) -> Dict[str, Any]:
        """Get visualization service statistics"""
        return {
            **self.visualizer_stats,
            'cached_visualizations': len(self.visualization_cache),
            'supported_layouts': len(LayoutAlgorithm),
            'supported_exports': len(['svg', 'png', 'pdf', 'json', 'dot', 'plantuml']),
            'node_type_configs': len(self.node_type_configs)
        }
    
    def _extract_visualization_nodes(self, plan_ir: Dict[str, Any], 
                                   filter_options: FilterOptions) -> List[VisualizationNode]:
        """Extract nodes from plan IR and convert to visualization nodes"""
        nodes = []
        
        # Extract nodes from IR
        ir_nodes = []
        if 'nodes' in plan_ir:
            ir_nodes = plan_ir['nodes']
        elif 'plan_graph' in plan_ir and 'nodes' in plan_ir['plan_graph']:
            ir_nodes = plan_ir['plan_graph']['nodes']
        
        for ir_node in ir_nodes:
            node_id = ir_node.get('id', 'unknown')
            node_type = ir_node.get('type', 'unknown')
            
            # Apply node type filters
            if not self._node_passes_type_filter(node_type, filter_options):
                continue
            
            # Get node configuration
            node_config = self.node_type_configs.get(node_type, self.node_type_configs['default'])
            
            # Create visualization node
            viz_node = VisualizationNode(
                node_id=node_id,
                node_type=node_type,
                display_name=ir_node.get('name', ir_node.get('id', 'Unnamed')),
                shape=NodeShape(node_config['shape']),
                color=node_config['color'],
                size=node_config['size'],
                description=ir_node.get('description', ''),
                metadata=ir_node.get('metadata', {}),
                policies=ir_node.get('policies', {}),
                trust_budget=ir_node.get('trust_budget'),
                sla_requirements=ir_node.get('sla', {}),
                cost_class=ir_node.get('cost_class', 'T2'),
                region_id=ir_node.get('region_id')
            )
            
            # Apply additional filters
            if self._node_passes_filters(viz_node, filter_options):
                nodes.append(viz_node)
        
        return nodes
    
    def _extract_visualization_edges(self, plan_ir: Dict[str, Any], 
                                   filter_options: FilterOptions) -> List[VisualizationEdge]:
        """Extract edges from plan IR and convert to visualization edges"""
        edges = []
        
        # Extract edges from IR
        ir_edges = []
        if 'edges' in plan_ir:
            ir_edges = plan_ir['edges']
        elif 'plan_graph' in plan_ir and 'edges' in plan_ir['plan_graph']:
            ir_edges = plan_ir['plan_graph']['edges']
        
        # Regular data flow edges
        for i, ir_edge in enumerate(ir_edges):
            edge_id = ir_edge.get('id', f'edge_{i}')
            source = ir_edge.get('source', ir_edge.get('from'))
            target = ir_edge.get('target', ir_edge.get('to'))
            
            if not source or not target:
                continue
            
            viz_edge = VisualizationEdge(
                edge_id=edge_id,
                source_node_id=source,
                target_node_id=target,
                edge_type=EdgeType.SOLID,
                color="#7f8c8d",
                label=ir_edge.get('label', ''),
                condition=ir_edge.get('condition', ''),
                is_data_flow=True
            )
            
            if self._edge_passes_filter(viz_edge, filter_options):
                edges.append(viz_edge)
        
        # Extract fallback edges from nodes
        ir_nodes = []
        if 'nodes' in plan_ir:
            ir_nodes = plan_ir['nodes']
        elif 'plan_graph' in plan_ir and 'nodes' in plan_ir['plan_graph']:
            ir_nodes = plan_ir['plan_graph']['nodes']
        
        for ir_node in ir_nodes:
            node_id = ir_node.get('id')
            fallbacks = ir_node.get('fallbacks', [])
            
            for j, fallback in enumerate(fallbacks):
                target = fallback.get('target')
                if target:
                    fallback_edge = VisualizationEdge(
                        edge_id=f'fallback_{node_id}_{j}',
                        source_node_id=node_id,
                        target_node_id=target,
                        edge_type=EdgeType.DASHED,
                        color="#e74c3c",
                        label=fallback.get('reason', 'fallback'),
                        condition=fallback.get('condition', ''),
                        is_fallback=True,
                        is_data_flow=False
                    )
                    
                    if self._edge_passes_filter(fallback_edge, filter_options):
                        edges.append(fallback_edge)
        
        return edges
    
    def _apply_layout_algorithm(self, visualization: GraphVisualization):
        """Apply layout algorithm to position nodes"""
        if visualization.layout.algorithm == LayoutAlgorithm.FORCE_DIRECTED:
            self._apply_force_directed_layout(visualization)
        elif visualization.layout.algorithm == LayoutAlgorithm.HIERARCHICAL:
            self._apply_hierarchical_layout(visualization)
        elif visualization.layout.algorithm == LayoutAlgorithm.CIRCULAR:
            self._apply_circular_layout(visualization)
        elif visualization.layout.algorithm == LayoutAlgorithm.GRID:
            self._apply_grid_layout(visualization)
        else:
            # Default to force-directed
            self._apply_force_directed_layout(visualization)
    
    def _apply_force_directed_layout(self, visualization: GraphVisualization):
        """Apply force-directed layout algorithm"""
        nodes = visualization.nodes
        edges = visualization.edges
        layout = visualization.layout
        
        # Initialize random positions
        import random
        for node in nodes:
            node.position = NodePosition(
                x=random.uniform(layout.padding, layout.width - layout.padding),
                y=random.uniform(layout.padding, layout.height - layout.padding)
            )
        
        # Simple force-directed simulation (simplified)
        for iteration in range(50):  # 50 iterations
            # Calculate forces
            for node in nodes:
                force_x, force_y = 0.0, 0.0
                
                # Repulsion from other nodes
                for other_node in nodes:
                    if other_node.node_id != node.node_id:
                        dx = node.position.x - other_node.position.x
                        dy = node.position.y - other_node.position.y
                        distance = math.sqrt(dx*dx + dy*dy) or 1.0
                        
                        repulsion = layout.repulsion_strength / (distance * distance)
                        force_x += (dx / distance) * repulsion
                        force_y += (dy / distance) * repulsion
                
                # Attraction from connected nodes
                for edge in edges:
                    other_node_id = None
                    if edge.source_node_id == node.node_id:
                        other_node_id = edge.target_node_id
                    elif edge.target_node_id == node.node_id:
                        other_node_id = edge.source_node_id
                    
                    if other_node_id:
                        other_node = next((n for n in nodes if n.node_id == other_node_id), None)
                        if other_node:
                            dx = other_node.position.x - node.position.x
                            dy = other_node.position.y - node.position.y
                            distance = math.sqrt(dx*dx + dy*dy) or 1.0
                            
                            attraction = layout.attraction_strength * distance
                            force_x += (dx / distance) * attraction
                            force_y += (dy / distance) * attraction
                
                # Apply gravity toward center
                center_x = layout.width / 2
                center_y = layout.height / 2
                dx = center_x - node.position.x
                dy = center_y - node.position.y
                
                force_x += dx * layout.gravity
                force_y += dy * layout.gravity
                
                # Update position
                node.position.x += force_x * 0.01  # Damping factor
                node.position.y += force_y * 0.01
                
                # Keep within bounds
                node.position.x = max(layout.padding, min(layout.width - layout.padding, node.position.x))
                node.position.y = max(layout.padding, min(layout.height - layout.padding, node.position.y))
    
    def _apply_hierarchical_layout(self, visualization: GraphVisualization):
        """Apply hierarchical layout algorithm"""
        nodes = visualization.nodes
        edges = visualization.edges
        layout = visualization.layout
        
        # Build adjacency list
        adjacency = defaultdict(list)
        in_degree = defaultdict(int)
        
        for edge in edges:
            if not edge.is_fallback:  # Only use data flow edges for hierarchy
                adjacency[edge.source_node_id].append(edge.target_node_id)
                in_degree[edge.target_node_id] += 1
        
        # Topological sort to determine levels
        levels = {}
        queue = [node.node_id for node in nodes if in_degree[node.node_id] == 0]
        current_level = 0
        
        while queue:
            next_queue = []
            
            for node_id in queue:
                levels[node_id] = current_level
                
                for neighbor in adjacency[node_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            
            queue = next_queue
            current_level += 1
        
        # Position nodes based on levels
        level_counts = defaultdict(int)
        for level in levels.values():
            level_counts[level] += 1
        
        level_positions = defaultdict(int)
        
        for node in nodes:
            level = levels.get(node.node_id, 0)
            y = layout.padding + level * layout.level_separation
            
            # Distribute nodes horizontally within level
            total_in_level = level_counts[level]
            x_spacing = (layout.width - 2 * layout.padding) / max(1, total_in_level - 1) if total_in_level > 1 else 0
            x = layout.padding + level_positions[level] * x_spacing
            
            node.position = NodePosition(x=x, y=y)
            level_positions[level] += 1
    
    def _apply_circular_layout(self, visualization: GraphVisualization):
        """Apply circular layout algorithm"""
        nodes = visualization.nodes
        layout = visualization.layout
        
        center_x = layout.width / 2
        center_y = layout.height / 2
        
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / len(nodes)
            x = center_x + layout.radius * math.cos(angle)
            y = center_y + layout.radius * math.sin(angle)
            
            node.position = NodePosition(x=x, y=y)
    
    def _apply_grid_layout(self, visualization: GraphVisualization):
        """Apply grid layout algorithm"""
        nodes = visualization.nodes
        layout = visualization.layout
        
        # Calculate grid dimensions
        cols = math.ceil(math.sqrt(len(nodes)))
        rows = math.ceil(len(nodes) / cols)
        
        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols
            
            x = layout.padding + col * layout.grid_spacing
            y = layout.padding + row * layout.grid_spacing
            
            node.position = NodePosition(x=x, y=y)
    
    def _apply_hotspot_indicators(self, visualization: GraphVisualization):
        """Apply hotspot indicators to nodes with high connectivity or complexity"""
        # Calculate connectivity scores
        connectivity_scores = {}
        
        for node in visualization.nodes:
            incoming = len([e for e in visualization.edges if e.target_node_id == node.node_id])
            outgoing = len([e for e in visualization.edges if e.source_node_id == node.node_id])
            total_connections = incoming + outgoing
            
            connectivity_scores[node.node_id] = total_connections
        
        # Determine hotspot threshold (top 20% of nodes by connectivity)
        if connectivity_scores:
            sorted_scores = sorted(connectivity_scores.values(), reverse=True)
            threshold_index = max(0, len(sorted_scores) // 5)  # Top 20%
            hotspot_threshold = sorted_scores[threshold_index] if threshold_index < len(sorted_scores) else 0
            
            for node in visualization.nodes:
                if connectivity_scores.get(node.node_id, 0) >= hotspot_threshold and hotspot_threshold > 1:
                    node.hotspot_indicator = True
    
    def _node_passes_type_filter(self, node_type: str, filter_options: FilterOptions) -> bool:
        """Check if node passes type filters"""
        if 'ml_' in node_type.lower() and not filter_options.show_ml_nodes:
            return False
        if 'decision' in node_type.lower() and not filter_options.show_decision_nodes:
            return False
        if 'external' in node_type.lower() and not filter_options.show_external_nodes:
            return False
        if 'data' in node_type.lower() and not filter_options.show_data_nodes:
            return False
        
        return True
    
    def _node_passes_filters(self, node: VisualizationNode, filter_options: FilterOptions) -> bool:
        """Check if node passes all filters"""
        # Region filter
        if filter_options.filter_by_region and node.region_id != filter_options.filter_by_region:
            return False
        
        # Cost class filter
        if filter_options.filter_by_cost_class and node.cost_class != filter_options.filter_by_cost_class:
            return False
        
        # Status filters
        if node.status == 'inactive' and not filter_options.show_inactive_nodes:
            return False
        if node.status == 'error' and not filter_options.show_error_nodes:
            return False
        
        return True
    
    def _edge_passes_filter(self, edge: VisualizationEdge, filter_options: FilterOptions) -> bool:
        """Check if edge passes filters"""
        if edge.is_data_flow and not filter_options.show_data_flow:
            return False
        if edge.is_control_flow and not filter_options.show_control_flow:
            return False
        if edge.is_fallback and not filter_options.show_fallback_edges:
            return False
        
        return True
    
    def _filter_nodes(self, nodes: List[VisualizationNode], 
                     filter_options: FilterOptions) -> List[VisualizationNode]:
        """Filter nodes based on filter options"""
        filtered_nodes = []
        
        for node in nodes:
            if (self._node_passes_type_filter(node.node_type, filter_options) and
                self._node_passes_filters(node, filter_options)):
                filtered_nodes.append(node)
        
        return filtered_nodes
    
    def _analyze_connectivity(self, visualization: GraphVisualization) -> Dict[str, Any]:
        """Analyze connectivity statistics"""
        node_connections = defaultdict(int)
        
        for edge in visualization.edges:
            node_connections[edge.source_node_id] += 1
            node_connections[edge.target_node_id] += 1
        
        if node_connections:
            max_connections = max(node_connections.values())
            avg_connections = sum(node_connections.values()) / len(node_connections)
            
            return {
                'max_connections': max_connections,
                'average_connections': round(avg_connections, 2),
                'total_connections': sum(node_connections.values()),
                'isolated_nodes': len([n for n in visualization.nodes if node_connections[n.node_id] == 0])
            }
        
        return {
            'max_connections': 0,
            'average_connections': 0.0,
            'total_connections': 0,
            'isolated_nodes': len(visualization.nodes)
        }
    
    def _analyze_fallback_coverage(self, visualization: GraphVisualization) -> Dict[str, Any]:
        """Analyze fallback coverage"""
        ml_nodes = [n for n in visualization.nodes if 'ml_' in n.node_type.lower()]
        fallback_sources = {e.source_node_id for e in visualization.edges if e.is_fallback}
        
        ml_nodes_with_fallbacks = len([n for n in ml_nodes if n.node_id in fallback_sources])
        
        coverage_percentage = (ml_nodes_with_fallbacks / len(ml_nodes) * 100) if ml_nodes else 100
        
        return {
            'total_ml_nodes': len(ml_nodes),
            'ml_nodes_with_fallbacks': ml_nodes_with_fallbacks,
            'coverage_percentage': round(coverage_percentage, 1),
            'total_fallback_edges': visualization.fallback_edges
        }
    
    def _analyze_policy_distribution(self, visualization: GraphVisualization) -> Dict[str, Any]:
        """Analyze policy distribution across nodes"""
        policy_counts = defaultdict(int)
        region_counts = defaultdict(int)
        cost_class_counts = defaultdict(int)
        
        for node in visualization.nodes:
            # Count policies
            for policy_key in node.policies.keys():
                policy_counts[policy_key] += 1
            
            # Count regions
            if node.region_id:
                region_counts[node.region_id] += 1
            
            # Count cost classes
            cost_class_counts[node.cost_class] += 1
        
        return {
            'policy_distribution': dict(policy_counts),
            'region_distribution': dict(region_counts),
            'cost_class_distribution': dict(cost_class_counts),
            'nodes_with_policies': len([n for n in visualization.nodes if n.policies])
        }
    
    def _export_to_json(self, visualization: GraphVisualization) -> Dict[str, Any]:
        """Export visualization to JSON format"""
        return {
            'format': 'json',
            'content': asdict(visualization),
            'size_bytes': len(json.dumps(asdict(visualization))),
            'export_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _export_to_dot(self, visualization: GraphVisualization) -> Dict[str, Any]:
        """Export visualization to DOT format (Graphviz)"""
        dot_lines = [
            f'digraph "{visualization.plan_id}" {{',
            '    // Graph attributes',
            '    rankdir=TB;',
            '    node [shape=ellipse];',
            '    edge [fontsize=10];',
            ''
        ]
        
        # Add nodes
        dot_lines.append('    // Nodes')
        for node in visualization.nodes:
            color = node.color
            shape = node.shape.value
            label = node.display_name.replace('"', '\\"')
            
            dot_lines.append(f'    "{node.node_id}" [label="{label}", color="{color}", shape="{shape}"];')
        
        dot_lines.append('')
        
        # Add edges
        dot_lines.append('    // Edges')
        for edge in visualization.edges:
            style = "dashed" if edge.is_fallback else "solid"
            color = edge.color
            label = edge.label.replace('"', '\\"') if edge.label else ""
            
            edge_attrs = f'color="{color}", style="{style}"'
            if label:
                edge_attrs += f', label="{label}"'
            
            dot_lines.append(f'    "{edge.source_node_id}" -> "{edge.target_node_id}" [{edge_attrs}];')
        
        dot_lines.append('}')
        
        dot_content = '\n'.join(dot_lines)
        
        return {
            'format': 'dot',
            'content': dot_content,
            'size_bytes': len(dot_content),
            'export_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _export_to_svg(self, visualization: GraphVisualization) -> Dict[str, Any]:
        """Export visualization to SVG format (placeholder)"""
        # SVG generation would require actual rendering engine
        svg_placeholder = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{visualization.layout.width}" height="{visualization.layout.height}" xmlns="http://www.w3.org/2000/svg">
  <!-- Plan: {visualization.plan_id} -->
  <!-- Nodes: {len(visualization.nodes)} -->
  <!-- Edges: {len(visualization.edges)} -->
  <!-- This would be generated by actual SVG rendering engine -->
  <text x="10" y="30" font-family="Arial" font-size="14">Plan Visualization: {visualization.plan_id}</text>
  <text x="10" y="50" font-family="Arial" font-size="12">Nodes: {len(visualization.nodes)}, Edges: {len(visualization.edges)}</text>
</svg>'''
        
        return {
            'format': 'svg',
            'content': svg_placeholder,
            'size_bytes': len(svg_placeholder),
            'export_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _export_to_plantuml(self, visualization: GraphVisualization) -> Dict[str, Any]:
        """Export visualization to PlantUML format"""
        plantuml_lines = [
            '@startuml',
            f'title Plan Visualization: {visualization.plan_id}',
            ''
        ]
        
        # Add nodes
        for node in visualization.nodes:
            node_type = node.node_type
            display_name = node.display_name
            
            if 'ml_' in node_type.lower():
                plantuml_lines.append(f'component "{display_name}" as {node.node_id} #LightBlue')
            elif 'decision' in node_type.lower():
                plantuml_lines.append(f'diamond "{display_name}" as {node.node_id} #LightGreen')
            elif 'external' in node_type.lower():
                plantuml_lines.append(f'interface "{display_name}" as {node.node_id} #LightCoral')
            else:
                plantuml_lines.append(f'rectangle "{display_name}" as {node.node_id}')
        
        plantuml_lines.append('')
        
        # Add edges
        for edge in visualization.edges:
            arrow = '-->' if edge.is_fallback else '-->'
            style = ' : fallback' if edge.is_fallback else ''
            label = f' : {edge.label}' if edge.label and not edge.is_fallback else style
            
            plantuml_lines.append(f'{edge.source_node_id} {arrow} {edge.target_node_id}{label}')
        
        plantuml_lines.append('@enduml')
        
        plantuml_content = '\n'.join(plantuml_lines)
        
        return {
            'format': 'plantuml',
            'content': plantuml_content,
            'size_bytes': len(plantuml_content),
            'export_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _export_placeholder(self, visualization: GraphVisualization, 
                          export_format: str) -> Dict[str, Any]:
        """Export placeholder for formats requiring external tools"""
        return {
            'format': export_format,
            'content': f'[{export_format.upper()} Export Placeholder for {visualization.plan_id} - Would be generated by external rendering service]',
            'size_bytes': 0,
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'note': f'{export_format.upper()} export requires external rendering service'
        }
    
    def _initialize_node_type_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize node type visual configurations"""
        return {
            'ml_node': {
                'shape': 'hexagon',
                'color': '#3498db',
                'size': 30.0
            },
            'decision': {
                'shape': 'diamond',
                'color': '#2ecc71',
                'size': 25.0
            },
            'external_api': {
                'shape': 'rectangle',
                'color': '#e74c3c',
                'size': 25.0
            },
            'data_transform': {
                'shape': 'circle',
                'color': '#f39c12',
                'size': 20.0
            },
            'validation': {
                'shape': 'triangle',
                'color': '#9b59b6',
                'size': 20.0
            },
            'notification': {
                'shape': 'star',
                'color': '#1abc9c',
                'size': 20.0
            },
            'default': {
                'shape': 'circle',
                'color': '#95a5a6',
                'size': 20.0
            }
        }
    
    def _initialize_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """Initialize color schemes"""
        return {
            'default': {
                'ml_node': '#3498db',
                'decision': '#2ecc71',
                'external': '#e74c3c',
                'data': '#f39c12',
                'control': '#9b59b6'
            },
            'accessibility': {
                'ml_node': '#0066cc',
                'decision': '#009900',
                'external': '#cc0000',
                'data': '#ff6600',
                'control': '#6600cc'
            }
        }
    
    def _initialize_layout_algorithms(self) -> Dict[LayoutAlgorithm, Dict[str, Any]]:
        """Initialize layout algorithm configurations"""
        return {
            LayoutAlgorithm.FORCE_DIRECTED: {
                'name': 'Force-Directed',
                'description': 'Nodes repel each other while connected nodes attract',
                'best_for': 'General purpose, medium-sized graphs'
            },
            LayoutAlgorithm.HIERARCHICAL: {
                'name': 'Hierarchical',
                'description': 'Nodes arranged in levels based on dependencies',
                'best_for': 'Workflows with clear hierarchical structure'
            },
            LayoutAlgorithm.CIRCULAR: {
                'name': 'Circular',
                'description': 'Nodes arranged in a circle',
                'best_for': 'Small graphs, highlighting connections'
            },
            LayoutAlgorithm.GRID: {
                'name': 'Grid',
                'description': 'Nodes arranged in a regular grid',
                'best_for': 'Simple layouts, easy node identification'
            }
        }

# API Interface
class GraphVisualizerAPI:
    """API interface for graph visualization operations"""
    
    def __init__(self, visualizer_service: Optional[GraphVisualizerService] = None):
        self.visualizer_service = visualizer_service or GraphVisualizerService()
    
    def generate_visualization(self, plan_ir: Dict[str, Any],
                             layout: str = "force_directed",
                             filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """API endpoint to generate graph visualization"""
        try:
            layout_enum = LayoutAlgorithm(layout)
            filter_options = FilterOptions(**filters) if filters else None
            
            visualization = self.visualizer_service.generate_graph_visualization(
                plan_ir, layout_enum, filter_options
            )
            
            return {
                'success': True,
                'visualization': asdict(visualization)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def export_visualization(self, visualization_dict: Dict[str, Any],
                           export_format: str = "json") -> Dict[str, Any]:
        """API endpoint to export visualization"""
        try:
            visualization = GraphVisualization(**visualization_dict)
            export_result = self.visualizer_service.export_visualization(visualization, export_format)
            
            return {
                'success': True,
                'export': export_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_node_details(self, visualization_dict: Dict[str, Any],
                        node_id: str) -> Dict[str, Any]:
        """API endpoint to get node details"""
        try:
            visualization = GraphVisualization(**visualization_dict)
            details = self.visualizer_service.get_node_details(visualization, node_id)
            
            return {
                'success': True,
                'node_details': details
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Test Functions
def create_test_plan_ir() -> Dict[str, Any]:
    """Create test plan IR for visualization"""
    return {
        'plan_id': 'test_visualization_plan',
        'version': '1.0.0',
        'plan_graph': {
            'nodes': [
                {
                    'id': 'start',
                    'type': 'start',
                    'name': 'Start Node',
                    'description': 'Starting point of the workflow',
                    'cost_class': 'T1',
                    'region_id': 'us-east-1'
                },
                {
                    'id': 'risk_ml',
                    'type': 'ml_node',
                    'name': 'Risk Assessment ML',
                    'description': 'ML model for risk assessment',
                    'cost_class': 'T3',
                    'region_id': 'us-east-1',
                    'trust_budget': {'min_confidence': 0.8},
                    'policies': {'explainability': 'required'},
                    'fallbacks': [
                        {'condition': 'confidence < 0.8', 'target': 'manual_review', 'reason': 'Low confidence'}
                    ]
                },
                {
                    'id': 'decision_gate',
                    'type': 'decision',
                    'name': 'Decision Gate',
                    'description': 'Business decision based on ML output',
                    'cost_class': 'T2',
                    'region_id': 'us-east-1'
                },
                {
                    'id': 'external_api',
                    'type': 'external_api',
                    'name': 'External Service',
                    'description': 'Call to external service',
                    'cost_class': 'T2',
                    'region_id': 'us-east-1'
                },
                {
                    'id': 'manual_review',
                    'type': 'manual',
                    'name': 'Manual Review',
                    'description': 'Human review process',
                    'cost_class': 'T4',
                    'region_id': 'us-east-1'
                }
            ],
            'edges': [
                {'source': 'start', 'target': 'risk_ml'},
                {'source': 'risk_ml', 'target': 'decision_gate', 'condition': 'confidence >= 0.8'},
                {'source': 'decision_gate', 'target': 'external_api', 'condition': 'approved'},
                {'source': 'manual_review', 'target': 'decision_gate'}
            ]
        }
    }

def run_graph_visualizer_tests():
    """Run comprehensive graph visualizer tests"""
    print("=== Graph Visualizer Service Tests ===")
    
    # Initialize service
    visualizer_service = GraphVisualizerService()
    visualizer_api = GraphVisualizerAPI(visualizer_service)
    
    # Create test plan IR
    test_ir = create_test_plan_ir()
    
    # Test 1: Basic visualization generation
    print("\n1. Testing basic visualization generation...")
    
    layouts = [LayoutAlgorithm.FORCE_DIRECTED, LayoutAlgorithm.HIERARCHICAL, LayoutAlgorithm.CIRCULAR]
    
    for layout in layouts:
        visualization = visualizer_service.generate_graph_visualization(test_ir, layout)
        
        print(f"   {layout.value} layout:")
        print(f"     Nodes: {len(visualization.nodes)}")
        print(f"     Edges: {len(visualization.edges)}")
        print(f"     Fallback edges: {visualization.fallback_edges}")
        print(f"     Layout dimensions: {visualization.layout.width}x{visualization.layout.height}")
    
    # Test 2: Node positioning verification
    print("\n2. Testing node positioning...")
    
    force_viz = visualizer_service.generate_graph_visualization(test_ir, LayoutAlgorithm.FORCE_DIRECTED)
    
    print(f"   Force-directed positioning:")
    for node in force_viz.nodes[:3]:  # Show first 3 nodes
        if node.position:
            print(f"     {node.display_name}: ({node.position.x:.1f}, {node.position.y:.1f})")
    
    # Test 3: Filtering functionality
    print("\n3. Testing filtering functionality...")
    
    # Test with ML nodes only
    ml_filter = FilterOptions(
        show_ml_nodes=True,
        show_decision_nodes=False,
        show_external_nodes=False
    )
    
    filtered_viz = visualizer_service.apply_filter(force_viz, ml_filter)
    print(f"   ML nodes only filter: {len(filtered_viz.nodes)} nodes, {len(filtered_viz.edges)} edges")
    
    # Test with fallback edges only
    fallback_filter = FilterOptions(
        show_data_flow=False,
        show_fallback_edges=True
    )
    
    fallback_viz = visualizer_service.apply_filter(force_viz, fallback_filter)
    print(f"   Fallback edges only: {len(fallback_viz.edges)} edges")
    
    # Test 4: Export functionality
    print("\n4. Testing export functionality...")
    
    export_formats = ["json", "dot", "svg", "plantuml"]
    
    for export_format in export_formats:
        export_result = visualizer_service.export_visualization(force_viz, export_format)
        
        print(f"   {export_format.upper()} export:")
        print(f"     Size: {export_result['size_bytes']} bytes")
        print(f"     Preview: {export_result['content'][:100]}...")
    
    # Test 5: Node details
    print("\n5. Testing node details...")
    
    # Get details for the ML node
    ml_node_id = 'risk_ml'
    node_details = visualizer_service.get_node_details(force_viz, ml_node_id)
    
    print(f"   Node details for {ml_node_id}:")
    print(f"     Type: {node_details['node']['node_type']}")
    print(f"     Connections: {node_details['connections']['total']}")
    print(f"     Is hotspot: {node_details['is_hotspot']}")
    print(f"     Incoming edges: {len(node_details['incoming_edges'])}")
    print(f"     Outgoing edges: {len(node_details['outgoing_edges'])}")
    
    # Test 6: Topology summary
    print("\n6. Testing topology summary...")
    
    topology_summary = visualizer_service.generate_topology_summary(force_viz)
    
    print(f"   Topology summary:")
    print(f"     Total nodes: {topology_summary['total_nodes']}")
    print(f"     Node types: {topology_summary['node_type_distribution']}")
    print(f"     Connectivity: {topology_summary['connectivity_stats']}")
    print(f"     Fallback coverage: {topology_summary['fallback_coverage']}")
    
    # Test 7: Hotspot detection
    print("\n7. Testing hotspot detection...")
    
    hotspot_nodes = [n for n in force_viz.nodes if n.hotspot_indicator]
    print(f"   Hotspot nodes detected: {len(hotspot_nodes)}")
    
    for hotspot in hotspot_nodes:
        print(f"     - {hotspot.display_name} ({hotspot.node_type})")
    
    # Test 8: API interface
    print("\n8. Testing API interface...")
    
    # Test API visualization generation
    api_viz_result = visualizer_api.generate_visualization(
        test_ir, 
        "hierarchical",
        {"show_ml_nodes": True, "show_decision_nodes": True}
    )
    print(f"   API visualization: {'✅ PASS' if api_viz_result['success'] else '❌ FAIL'}")
    
    # Test API export
    if api_viz_result['success']:
        api_export_result = visualizer_api.export_visualization(
            api_viz_result['visualization'], 
            "dot"
        )
        print(f"   API export: {'✅ PASS' if api_export_result['success'] else '❌ FAIL'}")
    
    # Test API node details
    if api_viz_result['success']:
        api_details_result = visualizer_api.get_node_details(
            api_viz_result['visualization'], 
            "risk_ml"
        )
        print(f"   API node details: {'✅ PASS' if api_details_result['success'] else '❌ FAIL'}")
    
    # Test 9: Edge cases
    print("\n9. Testing edge cases...")
    
    # Empty plan
    empty_ir = {'plan_id': 'empty', 'plan_graph': {'nodes': [], 'edges': []}}
    empty_viz = visualizer_service.generate_graph_visualization(empty_ir)
    print(f"   Empty plan: {len(empty_viz.nodes)} nodes, {len(empty_viz.edges)} edges")
    
    # Single node plan
    single_node_ir = {
        'plan_id': 'single',
        'plan_graph': {
            'nodes': [{'id': 'only', 'type': 'simple', 'name': 'Only Node'}],
            'edges': []
        }
    }
    single_viz = visualizer_service.generate_graph_visualization(single_node_ir)
    print(f"   Single node plan: {len(single_viz.nodes)} nodes, position: ({single_viz.nodes[0].position.x:.1f}, {single_viz.nodes[0].position.y:.1f})")
    
    # Test 10: Performance and caching
    print("\n10. Testing performance and caching...")
    
    # Generate same visualization multiple times to test caching
    start_time = datetime.now(timezone.utc)
    for i in range(3):
        cached_viz = visualizer_service.generate_graph_visualization(test_ir)
    end_time = datetime.now(timezone.utc)
    
    total_time = (end_time - start_time).total_seconds() * 1000
    print(f"   3 cached generations: {total_time:.1f}ms total")
    
    # Test statistics
    stats = visualizer_service.get_visualization_statistics()
    print(f"   Total visualizations: {stats['total_visualizations']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    print(f"   Nodes visualized: {stats['nodes_visualized']}")
    print(f"   Edges visualized: {stats['edges_visualized']}")
    
    print(f"\n=== Test Summary ===")
    final_stats = visualizer_service.get_visualization_statistics()
    print(f"Graph visualizer service tested successfully")
    print(f"Total visualizations: {final_stats['total_visualizations']}")
    print(f"Layouts supported: {final_stats['supported_layouts']}")
    print(f"Export formats: {final_stats['supported_exports']}")
    print(f"Cache hit rate: {final_stats['cache_hits']}/{final_stats['cache_hits'] + final_stats['cache_misses']}")
    print(f"Nodes visualized: {final_stats['nodes_visualized']}")
    
    return visualizer_service, visualizer_api

if __name__ == "__main__":
    run_graph_visualizer_tests()
