"""
Fallback DAG Resolver - Task 6.2.8
===================================

Encode fallback DAG (RBIA→RBA and node-level) in IR
- Extend IR node model with fallback[] field
- Fallback DAG resolver with validation
- Ensure fallback graph is acyclic
- Compute fallback coverage metrics per ML node

Dependencies: Task 6.2.4 (IR), Task 6.2.1 (DSL Grammar)
Outputs: Validated fallback DAG → input for runtime (safe degradation)
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Task 6.2.8: Fallback DAG Resolver
class FallbackDAGResolver:
    """Resolves and validates fallback DAG for safe degradation"""
    
    def __init__(self):
        self.validation_errors: List[str] = []
        self.coverage_metrics: Dict[str, Dict[str, Any]] = {}
    
    def resolve_fallback_dag(self, ir_graph) -> bool:
        """Resolve and validate complete fallback DAG"""
        from .ir import IRGraph
        
        self.validation_errors.clear()
        self.coverage_metrics.clear()
        
        # Validate fallback references
        if not self._validate_fallback_references(ir_graph):
            return False
        
        # Check for cycles in fallback graph
        if not self._validate_fallback_acyclic(ir_graph):
            return False
        
        # Compute coverage metrics
        self._compute_fallback_coverage_metrics(ir_graph)
        
        # Validate residency compatibility
        if not self._validate_residency_compatibility(ir_graph):
            return False
        
        return len(self.validation_errors) == 0
    
    def _validate_fallback_references(self, ir_graph) -> bool:
        """Validate that all fallback references exist"""
        node_ids = {node.id for node in ir_graph.nodes}
        
        for node in ir_graph.nodes:
            if not node.fallbacks or not node.fallbacks.enabled:
                continue
            
            for fallback_item in node.fallbacks.fallback_items:
                target_id = fallback_item.target_node_id
                
                # Check if target exists
                if target_id not in node_ids:
                    self.validation_errors.append(
                        f"Node {node.id}: fallback target '{target_id}' does not exist"
                    )
                    return False
                
                # Check if fallback type is valid
                valid_types = ['rba_rule', 'default_action', 'human_escalation', 'previous_step']
                if fallback_item.type not in valid_types:
                    self.validation_errors.append(
                        f"Node {node.id}: invalid fallback type '{fallback_item.type}'"
                    )
                    return False
        
        return True
    
    def _validate_fallback_acyclic(self, ir_graph) -> bool:
        """Validate that fallback graph is acyclic"""
        # Build fallback graph
        fallback_graph = self._build_fallback_graph(ir_graph)
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        for node_id in fallback_graph:
            if node_id not in visited:
                if self._has_cycle_dfs(fallback_graph, node_id, visited, rec_stack):
                    self.validation_errors.append(
                        f"Cycle detected in fallback graph involving node: {node_id}"
                    )
                    return False
        
        return True
    
    def _build_fallback_graph(self, ir_graph) -> Dict[str, List[str]]:
        """Build fallback dependency graph"""
        fallback_graph = {}
        
        for node in ir_graph.nodes:
            fallback_graph[node.id] = []
            
            if node.fallbacks and node.fallbacks.enabled:
                for fallback_item in node.fallbacks.fallback_items:
                    fallback_graph[node.id].append(fallback_item.target_node_id)
        
        return fallback_graph
    
    def _has_cycle_dfs(self, graph: Dict[str, List[str]], node: str, 
                      visited: Set[str], rec_stack: Set[str]) -> bool:
        """DFS-based cycle detection"""
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if self._has_cycle_dfs(graph, neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    def _compute_fallback_coverage_metrics(self, ir_graph):
        """Compute fallback coverage metrics per ML node"""
        ml_nodes = ir_graph.get_ml_nodes()
        
        for node in ml_nodes:
            metrics = {
                'node_id': node.id,
                'node_type': node.type.value,
                'has_fallback': bool(node.fallbacks and node.fallbacks.enabled),
                'fallback_count': 0,
                'fallback_types': [],
                'coverage_score': 0.0,
                'rba_fallback_available': False,
                'human_escalation_available': False,
                'max_fallback_depth': 0
            }
            
            if node.fallbacks and node.fallbacks.enabled:
                metrics['fallback_count'] = len(node.fallbacks.fallback_items)
                metrics['fallback_types'] = [item.type for item in node.fallbacks.fallback_items]
                metrics['max_fallback_depth'] = node.fallbacks.max_fallback_depth
                
                # Check for specific fallback types
                metrics['rba_fallback_available'] = any(
                    item.type == 'rba_rule' for item in node.fallbacks.fallback_items
                )
                metrics['human_escalation_available'] = any(
                    item.type == 'human_escalation' for item in node.fallbacks.fallback_items
                )
                
                # Compute coverage score (0.0 - 1.0)
                coverage_score = 0.0
                if metrics['rba_fallback_available']:
                    coverage_score += 0.4  # RBA fallback is important
                if metrics['human_escalation_available']:
                    coverage_score += 0.3  # Human escalation is important
                if metrics['fallback_count'] > 1:
                    coverage_score += 0.2  # Multiple fallbacks
                if any(item.condition for item in node.fallbacks.fallback_items):
                    coverage_score += 0.1  # Conditional fallbacks
                
                metrics['coverage_score'] = min(coverage_score, 1.0)
            
            self.coverage_metrics[node.id] = metrics
    
    def _validate_residency_compatibility(self, ir_graph) -> bool:
        """Validate residency compatibility for fallback paths"""
        for node in ir_graph.nodes:
            if not node.fallbacks or not node.fallbacks.enabled:
                continue
            
            node_region = node.region_id
            if not node_region:
                continue
            
            for fallback_item in node.fallbacks.fallback_items:
                target_node = ir_graph.get_node(fallback_item.target_node_id)
                if not target_node:
                    continue
                
                target_region = target_node.region_id
                if target_region and target_region != node_region:
                    # Check if cross-region fallback is allowed
                    if not self._is_cross_region_fallback_allowed(node, target_node):
                        self.validation_errors.append(
                            f"Node {node.id}: fallback to {target_node.id} violates residency constraints "
                            f"({node_region} -> {target_region})"
                        )
                        return False
        
        return True
    
    def _is_cross_region_fallback_allowed(self, source_node, target_node) -> bool:
        """Check if cross-region fallback is allowed by policies"""
        # Check residency policies
        for policy in source_node.policies:
            if policy.policy_type == "residency":
                params = policy.parameters
                if not params.get("cross_region_allowed", False):
                    return False
        
        return True
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """Get fallback coverage report"""
        if not self.coverage_metrics:
            return {"error": "No coverage metrics computed"}
        
        total_ml_nodes = len(self.coverage_metrics)
        nodes_with_fallback = sum(1 for m in self.coverage_metrics.values() if m['has_fallback'])
        nodes_with_rba_fallback = sum(1 for m in self.coverage_metrics.values() if m['rba_fallback_available'])
        nodes_with_human_escalation = sum(1 for m in self.coverage_metrics.values() if m['human_escalation_available'])
        
        avg_coverage_score = sum(m['coverage_score'] for m in self.coverage_metrics.values()) / total_ml_nodes if total_ml_nodes > 0 else 0
        
        return {
            "summary": {
                "total_ml_nodes": total_ml_nodes,
                "nodes_with_fallback": nodes_with_fallback,
                "fallback_coverage_percentage": (nodes_with_fallback / total_ml_nodes * 100) if total_ml_nodes > 0 else 0,
                "nodes_with_rba_fallback": nodes_with_rba_fallback,
                "nodes_with_human_escalation": nodes_with_human_escalation,
                "average_coverage_score": avg_coverage_score
            },
            "node_details": self.coverage_metrics,
            "recommendations": self._generate_coverage_recommendations()
        }
    
    def _generate_coverage_recommendations(self) -> List[str]:
        """Generate recommendations for improving fallback coverage"""
        recommendations = []
        
        for node_id, metrics in self.coverage_metrics.items():
            if not metrics['has_fallback']:
                recommendations.append(f"Node {node_id}: Add fallback configuration")
            elif not metrics['rba_fallback_available']:
                recommendations.append(f"Node {node_id}: Consider adding RBA fallback for deterministic degradation")
            elif not metrics['human_escalation_available']:
                recommendations.append(f"Node {node_id}: Consider adding human escalation fallback")
            elif metrics['coverage_score'] < 0.7:
                recommendations.append(f"Node {node_id}: Improve fallback coverage (current: {metrics['coverage_score']:.2f})")
        
        return recommendations

# Task 6.2.8: Fallback DAG Integration
def validate_and_resolve_fallback_dag(ir_graph) -> Tuple[bool, Dict[str, Any]]:
    """Validate and resolve fallback DAG for IR graph"""
    resolver = FallbackDAGResolver()
    is_valid = resolver.resolve_fallback_dag(ir_graph)
    
    result = {
        "is_valid": is_valid,
        "errors": resolver.validation_errors,
        "coverage_report": resolver.get_coverage_report()
    }
    
    return is_valid, result
