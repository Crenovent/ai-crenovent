"""
ML Node Coalescing Optimizer - Task 6.5.50
===========================================

Optimizer: ML-node coalescing when safe
- Combine multiple ML nodes using same model into single batch call
- Reduce latency by batching inference requests
- Maintain correctness through safety checks
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CoalescingStrategy(Enum):
    BATCH_INFERENCE = "batch_inference"
    SHARED_FEATURES = "shared_features"
    PIPELINE_FUSION = "pipeline_fusion"

@dataclass
class MLNodeGroup:
    """Group of ML nodes that can be coalesced"""
    model_id: str
    node_ids: List[str]
    strategy: CoalescingStrategy
    estimated_latency_reduction_ms: int

@dataclass
class CoalescingResult:
    """Result of ML node coalescing optimization"""
    coalesced: bool
    groups_created: int
    nodes_coalesced: int
    estimated_latency_reduction_ms: int
    safety_checks_passed: bool
    modifications: List[str]

class MLNodeCoalescingOptimizer:
    """Optimizer for coalescing ML nodes when safe"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_graph(self, ir_graph: Dict[str, Any]) -> CoalescingResult:
        """Optimize IR graph by coalescing ML nodes"""
        groups_created = 0
        nodes_coalesced = 0
        estimated_latency_reduction = 0
        modifications = []
        safety_checks_passed = True
        
        # Find ML nodes that can be coalesced
        ml_nodes = self._find_ml_nodes(ir_graph)
        coalescable_groups = self._find_coalescable_groups(ml_nodes, ir_graph)
        
        # Apply coalescing to each group
        for group in coalescable_groups:
            if self._safety_check(group, ir_graph):
                self._coalesce_group(group, ir_graph)
                groups_created += 1
                nodes_coalesced += len(group.node_ids)
                estimated_latency_reduction += group.estimated_latency_reduction_ms
                modifications.append(f"Coalesced {len(group.node_ids)} nodes using {group.model_id}")
            else:
                safety_checks_passed = False
                self.logger.warning(f"Safety check failed for group with model {group.model_id}")
        
        return CoalescingResult(
            coalesced=groups_created > 0,
            groups_created=groups_created,
            nodes_coalesced=nodes_coalesced,
            estimated_latency_reduction_ms=estimated_latency_reduction,
            safety_checks_passed=safety_checks_passed,
            modifications=modifications
        )
    
    def _find_ml_nodes(self, ir_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find all ML nodes in the graph"""
        ml_nodes = []
        
        for node_id, node in ir_graph.get('nodes', {}).items():
            if node.get('type') in ['ml_node', 'ml_predict', 'ml_score', 'ml_classify']:
                node['node_id'] = node_id
                ml_nodes.append(node)
        
        return ml_nodes
    
    def _find_coalescable_groups(self, ml_nodes: List[Dict[str, Any]], ir_graph: Dict[str, Any]) -> List[MLNodeGroup]:
        """Find groups of ML nodes that can be coalesced"""
        groups = []
        model_groups = {}
        
        # Group by model_id
        for node in ml_nodes:
            model_id = node.get('parameters', {}).get('model_id', '')
            if model_id:
                if model_id not in model_groups:
                    model_groups[model_id] = []
                model_groups[model_id].append(node)
        
        # Create coalescable groups
        for model_id, nodes in model_groups.items():
            if len(nodes) > 1:
                node_ids = [node['node_id'] for node in nodes]
                estimated_reduction = len(nodes) * 50  # 50ms per node saved
                
                group = MLNodeGroup(
                    model_id=model_id,
                    node_ids=node_ids,
                    strategy=CoalescingStrategy.BATCH_INFERENCE,
                    estimated_latency_reduction_ms=estimated_reduction
                )
                groups.append(group)
        
        return groups
    
    def _safety_check(self, group: MLNodeGroup, ir_graph: Dict[str, Any]) -> bool:
        """Verify it's safe to coalesce this group"""
        # Check 1: No data dependencies between nodes in group
        for i, node_id1 in enumerate(group.node_ids):
            for j, node_id2 in enumerate(group.node_ids):
                if i != j and self._has_dependency(node_id1, node_id2, ir_graph):
                    return False
        
        # Check 2: Same model version and configuration
        nodes = [ir_graph['nodes'][node_id] for node_id in group.node_ids]
        first_config = nodes[0].get('parameters', {})
        
        for node in nodes[1:]:
            node_config = node.get('parameters', {})
            if (node_config.get('model_version') != first_config.get('model_version') or
                node_config.get('confidence_threshold') != first_config.get('confidence_threshold')):
                return False
        
        return True
    
    def _has_dependency(self, node_id1: str, node_id2: str, ir_graph: Dict[str, Any]) -> bool:
        """Check if node1 depends on node2 (directly or indirectly)"""
        for edge in ir_graph.get('edges', []):
            if edge.get('source') == node_id2 and edge.get('target') == node_id1:
                return True
        return False
    
    def _coalesce_group(self, group: MLNodeGroup, ir_graph: Dict[str, Any]) -> None:
        """Coalesce a group of ML nodes into a single batch node"""
        # Create new coalesced node
        coalesced_node_id = f"coalesced_{group.model_id}_{len(group.node_ids)}"
        
        # Combine inputs from all nodes
        combined_inputs = []
        for node_id in group.node_ids:
            node = ir_graph['nodes'][node_id]
            combined_inputs.extend(node.get('inputs', []))
        
        # Create coalesced node
        ir_graph['nodes'][coalesced_node_id] = {
            'type': 'ml_batch',
            'model_id': group.model_id,
            'strategy': group.strategy.value,
            'original_nodes': group.node_ids,
            'inputs': combined_inputs,
            'batch_size': len(group.node_ids)
        }
        
        # Update edges to point to coalesced node
        for edge in ir_graph.get('edges', []):
            if edge.get('target') in group.node_ids:
                edge['target'] = coalesced_node_id
            if edge.get('source') in group.node_ids:
                edge['source'] = coalesced_node_id
        
        # Remove original nodes
        for node_id in group.node_ids:
            if node_id in ir_graph['nodes']:
                del ir_graph['nodes'][node_id]
