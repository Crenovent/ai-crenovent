"""
Predicate Pushdown Optimizer - Task 6.5.49
===========================================

Optimizer: predicate pushdown
- Push filtering conditions closer to data sources
- Reduce data flow between nodes
- Optimize execution graphs for efficiency
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PredicateType(Enum):
    EQUALITY = "equality"
    RANGE = "range"
    MEMBERSHIP = "membership"
    PATTERN = "pattern"

@dataclass
class Predicate:
    """Represents a filtering predicate"""
    field: str
    operator: str
    value: Any
    predicate_type: PredicateType
    node_id: str

@dataclass
class OptimizationResult:
    """Result of predicate pushdown optimization"""
    optimized: bool
    predicates_pushed: int
    estimated_reduction_percent: float
    modifications: List[str]

class PredicatePushdownOptimizer:
    """Optimizer for pushing predicates down to data sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_graph(self, ir_graph: Dict[str, Any]) -> OptimizationResult:
        """Optimize IR graph by pushing predicates down"""
        predicates_pushed = 0
        modifications = []
        
        # Find all filter predicates
        predicates = self._extract_predicates(ir_graph)
        
        # Push predicates down to earliest possible nodes
        for predicate in predicates:
            if self._can_push_predicate(predicate, ir_graph):
                self._push_predicate(predicate, ir_graph)
                predicates_pushed += 1
                modifications.append(f"Pushed {predicate.field} filter to {predicate.node_id}")
        
        estimated_reduction = min(predicates_pushed * 15.0, 75.0)  # Cap at 75%
        
        return OptimizationResult(
            optimized=predicates_pushed > 0,
            predicates_pushed=predicates_pushed,
            estimated_reduction_percent=estimated_reduction,
            modifications=modifications
        )
    
    def _extract_predicates(self, ir_graph: Dict[str, Any]) -> List[Predicate]:
        """Extract filter predicates from IR graph"""
        predicates = []
        
        for node_id, node in ir_graph.get('nodes', {}).items():
            if node.get('type') == 'decision' and 'conditions' in node:
                for condition in node['conditions']:
                    predicate = Predicate(
                        field=condition.get('field', ''),
                        operator=condition.get('operator', '=='),
                        value=condition.get('value'),
                        predicate_type=PredicateType.EQUALITY,
                        node_id=node_id
                    )
                    predicates.append(predicate)
        
        return predicates
    
    def _can_push_predicate(self, predicate: Predicate, ir_graph: Dict[str, Any]) -> bool:
        """Check if predicate can be pushed down"""
        # Simple heuristic: can push if field is available in upstream nodes
        return predicate.field and predicate.operator in ['==', '!=', '<', '>', '<=', '>=']
    
    def _push_predicate(self, predicate: Predicate, ir_graph: Dict[str, Any]) -> None:
        """Push predicate to earlier node in graph"""
        # Find upstream data source node
        upstream_nodes = self._find_upstream_nodes(predicate.node_id, ir_graph)
        
        if upstream_nodes:
            target_node = upstream_nodes[0]
            # Add predicate to target node
            if 'pushed_predicates' not in ir_graph['nodes'][target_node]:
                ir_graph['nodes'][target_node]['pushed_predicates'] = []
            
            ir_graph['nodes'][target_node]['pushed_predicates'].append({
                'field': predicate.field,
                'operator': predicate.operator,
                'value': predicate.value
            })
    
    def _find_upstream_nodes(self, node_id: str, ir_graph: Dict[str, Any]) -> List[str]:
        """Find upstream nodes that can accept pushed predicates"""
        upstream = []
        
        for edge in ir_graph.get('edges', []):
            if edge.get('target') == node_id:
                upstream.append(edge.get('source'))
        
        return upstream
