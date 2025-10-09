"""
Dead Code Elimination - Task 6.2.16
====================================

Dead code elimination (prune unreachable nodes)
- Optimizer pass that uses results from reachability analysis and constant folding
- Marks and removes unreachable nodes to shrink plan size
- Emits a "pruned nodes" report listing removed nodes, reason for removal, and source lines
- Ensures removal doesn't break source maps or diagnostics

Dependencies: Task 6.2.4 (IR), Task 6.2.13 (Static Rule Validation), Task 6.2.15 (Constant Folding)
Outputs: Optimized IR with dead code removed → smaller plans, reduced surface area
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PrunedNodeInfo:
    """Information about a pruned node"""
    node_id: str
    node_type: str
    removal_reason: str
    source_location: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    original_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeadCodeEliminationReport:
    """Report of dead code elimination optimization"""
    total_nodes_before: int
    total_nodes_after: int
    nodes_removed: int
    pruned_nodes: List[PrunedNodeInfo]
    optimization_timestamp: str
    size_reduction_percent: float
    warnings: List[str] = field(default_factory=list)

class RemovalReason(Enum):
    """Reasons for node removal"""
    UNREACHABLE = "unreachable"
    CONSTANT_CONDITION_FALSE = "constant_condition_false"
    REDUNDANT_AFTER_FOLDING = "redundant_after_folding"
    UNUSED_OUTPUT = "unused_output"
    DEAD_BRANCH = "dead_branch"
    CIRCULAR_DEPENDENCY = "circular_dependency"

# Task 6.2.16: Dead Code Elimination Optimizer
class DeadCodeEliminator:
    """Optimizer pass for removing unreachable and dead code"""
    
    def __init__(self):
        self.pruned_nodes: List[PrunedNodeInfo] = []
        self.reachability_cache: Dict[str, bool] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependency_graph: Dict[str, Set[str]] = {}
        
    def eliminate_dead_code(self, ir_graph) -> Tuple[bool, DeadCodeEliminationReport]:
        """Eliminate dead code from IR graph"""
        from .ir import IRGraph
        
        original_node_count = len(ir_graph.nodes)
        self.pruned_nodes.clear()
        self.reachability_cache.clear()
        
        # Build dependency graphs
        self._build_dependency_graphs(ir_graph)
        
        # Phase 1: Mark unreachable nodes
        unreachable_nodes = self._find_unreachable_nodes(ir_graph)
        
        # Phase 2: Mark nodes with constant false conditions
        constant_false_nodes = self._find_constant_false_nodes(ir_graph)
        
        # Phase 3: Mark redundant nodes after constant folding
        redundant_nodes = self._find_redundant_nodes(ir_graph)
        
        # Phase 4: Mark nodes with unused outputs
        unused_output_nodes = self._find_unused_output_nodes(ir_graph)
        
        # Combine all nodes to remove
        nodes_to_remove = (unreachable_nodes | constant_false_nodes | 
                          redundant_nodes | unused_output_nodes)
        
        # Remove nodes while preserving graph integrity
        removed_count = self._remove_nodes_safely(ir_graph, nodes_to_remove)
        
        # Generate report
        final_node_count = len(ir_graph.nodes)
        size_reduction = ((original_node_count - final_node_count) / original_node_count * 100 
                         if original_node_count > 0 else 0)
        
        report = DeadCodeEliminationReport(
            total_nodes_before=original_node_count,
            total_nodes_after=final_node_count,
            nodes_removed=removed_count,
            pruned_nodes=self.pruned_nodes.copy(),
            optimization_timestamp=datetime.utcnow().isoformat(),
            size_reduction_percent=size_reduction
        )
        
        return removed_count > 0, report
    
    def _build_dependency_graphs(self, ir_graph):
        """Build forward and reverse dependency graphs"""
        self.dependency_graph.clear()
        self.reverse_dependency_graph.clear()
        
        # Initialize graphs
        for node in ir_graph.nodes:
            self.dependency_graph[node.id] = set()
            self.reverse_dependency_graph[node.id] = set()
        
        # Build from explicit dependencies
        for node in ir_graph.nodes:
            for dep_id in node.dependencies:
                if dep_id in self.dependency_graph:
                    self.dependency_graph[node.id].add(dep_id)
                    self.reverse_dependency_graph[dep_id].add(node.id)
        
        # Build from IR edges
        for edge in ir_graph.edges:
            from_id = edge.get("from")
            to_id = edge.get("to")
            if from_id and to_id:
                if from_id in self.dependency_graph and to_id in self.dependency_graph:
                    self.dependency_graph[to_id].add(from_id)
                    self.reverse_dependency_graph[from_id].add(to_id)
    
    def _find_unreachable_nodes(self, ir_graph) -> Set[str]:
        """Find nodes that are unreachable from entry points"""
        # Find entry points (nodes with no dependencies)
        entry_points = []
        for node in ir_graph.nodes:
            if not node.dependencies and not self.dependency_graph.get(node.id):
                entry_points.append(node.id)
        
        # If no clear entry points, assume first node is entry
        if not entry_points and ir_graph.nodes:
            entry_points = [ir_graph.nodes[0].id]
        
        # Mark reachable nodes using DFS
        reachable = set()
        for entry in entry_points:
            self._mark_reachable_dfs(entry, reachable)
        
        # Find unreachable nodes
        unreachable = set()
        for node in ir_graph.nodes:
            if node.id not in reachable:
                unreachable.add(node.id)
                self.pruned_nodes.append(PrunedNodeInfo(
                    node_id=node.id,
                    node_type=str(node.type),
                    removal_reason=RemovalReason.UNREACHABLE.value,
                    source_location={"node_id": node.id},
                    dependencies=list(node.dependencies),
                    dependents=list(node.dependents),
                    original_parameters=node.parameters.copy()
                ))
        
        return unreachable
    
    def _mark_reachable_dfs(self, node_id: str, reachable: Set[str]):
        """Mark node and its dependents as reachable using DFS"""
        if node_id in reachable:
            return
        
        reachable.add(node_id)
        
        # Follow reverse dependencies (nodes that depend on this one)
        for dependent in self.reverse_dependency_graph.get(node_id, set()):
            self._mark_reachable_dfs(dependent, reachable)
    
    def _find_constant_false_nodes(self, ir_graph) -> Set[str]:
        """Find nodes with conditions that are always false"""
        constant_false = set()
        
        for node in ir_graph.nodes:
            if self._has_constant_false_condition(node):
                constant_false.add(node.id)
                self.pruned_nodes.append(PrunedNodeInfo(
                    node_id=node.id,
                    node_type=str(node.type),
                    removal_reason=RemovalReason.CONSTANT_CONDITION_FALSE.value,
                    source_location={"node_id": node.id},
                    dependencies=list(node.dependencies),
                    dependents=list(node.dependents),
                    original_parameters=node.parameters.copy()
                ))
        
        return constant_false
    
    def _has_constant_false_condition(self, node) -> bool:
        """Check if node has a condition that's always false"""
        # Check rule conditions
        if 'rules' in node.parameters:
            for rule in node.parameters['rules']:
                condition = rule.get('condition', '')
                if condition.strip().lower() == 'false':
                    return True
                
                # Check for impossible conditions like "1 > 2"
                if self._is_impossible_condition(condition):
                    return True
        
        # Check decision tree conditions
        if 'decision_tree' in node.parameters:
            return self._has_impossible_tree_condition(node.parameters['decision_tree'])
        
        return False
    
    def _is_impossible_condition(self, condition: str) -> bool:
        """Check if condition is logically impossible"""
        condition = condition.strip().lower()
        
        # Simple impossible conditions
        impossible_patterns = [
            'false', '1 > 2', '0 == 1', 'true == false',
            '1 < 0', '2 <= 1', '0 >= 1'
        ]
        
        return condition in impossible_patterns
    
    def _has_impossible_tree_condition(self, tree: Dict[str, Any]) -> bool:
        """Check if decision tree has impossible conditions"""
        if 'condition' in tree:
            if self._is_impossible_condition(tree['condition']):
                return True
        
        if 'branches' in tree:
            return any(self._has_impossible_tree_condition(branch) 
                      for branch in tree['branches'])
        
        return False
    
    def _find_redundant_nodes(self, ir_graph) -> Set[str]:
        """Find nodes that are redundant after constant folding"""
        redundant = set()
        
        for node in ir_graph.nodes:
            if self._is_redundant_after_folding(node):
                redundant.add(node.id)
                self.pruned_nodes.append(PrunedNodeInfo(
                    node_id=node.id,
                    node_type=str(node.type),
                    removal_reason=RemovalReason.REDUNDANT_AFTER_FOLDING.value,
                    source_location={"node_id": node.id},
                    dependencies=list(node.dependencies),
                    dependents=list(node.dependents),
                    original_parameters=node.parameters.copy()
                ))
        
        return redundant
    
    def _is_redundant_after_folding(self, node) -> bool:
        """Check if node is redundant after constant folding"""
        # Node with only constant outputs
        if self._has_only_constant_outputs(node):
            return True
        
        # Node that always produces the same result
        if self._always_produces_same_result(node):
            return True
        
        # Identity operations (like x = x)
        if self._is_identity_operation(node):
            return True
        
        return False
    
    def _has_only_constant_outputs(self, node) -> bool:
        """Check if node only produces constant outputs"""
        if 'rules' in node.parameters:
            for rule in node.parameters['rules']:
                action = rule.get('action', {})
                if isinstance(action, dict):
                    # If all action values are constants
                    for value in action.values():
                        if not self._is_constant_value(str(value)):
                            return False
                    return True
        
        return False
    
    def _always_produces_same_result(self, node) -> bool:
        """Check if node always produces the same result regardless of input"""
        if 'rules' in node.parameters:
            rules = node.parameters['rules']
            if len(rules) == 1:
                rule = rules[0]
                condition = rule.get('condition', '').strip().lower()
                # If condition is always true, node always produces same result
                return condition in ['true', '1', 'always']
        
        return False
    
    def _is_identity_operation(self, node) -> bool:
        """Check if node performs identity operation (output = input)"""
        # Simple heuristic - check if any rule just copies input to output
        if 'rules' in node.parameters:
            for rule in node.parameters['rules']:
                action = rule.get('action', {})
                if isinstance(action, dict):
                    for key, value in action.items():
                        # Check if output variable equals input variable
                        if isinstance(value, str) and key == value:
                            return True
        
        return False
    
    def _find_unused_output_nodes(self, ir_graph) -> Set[str]:
        """Find nodes whose outputs are never used"""
        unused_output = set()
        
        for node in ir_graph.nodes:
            if not self.reverse_dependency_graph.get(node.id) and not node.dependents:
                # Node has no dependents - check if it's a terminal action
                if not self._is_terminal_action_node(node):
                    unused_output.add(node.id)
                    self.pruned_nodes.append(PrunedNodeInfo(
                        node_id=node.id,
                        node_type=str(node.type),
                        removal_reason=RemovalReason.UNUSED_OUTPUT.value,
                        source_location={"node_id": node.id},
                        dependencies=list(node.dependencies),
                        dependents=list(node.dependents),
                        original_parameters=node.parameters.copy()
                    ))
        
        return unused_output
    
    def _is_terminal_action_node(self, node) -> bool:
        """Check if node is a terminal action (like notify, action)"""
        from .ir import IRNodeType
        
        terminal_types = {
            IRNodeType.ACTION,
            IRNodeType.NOTIFY,
            IRNodeType.AGENT_CALL
        }
        
        return node.type in terminal_types
    
    def _remove_nodes_safely(self, ir_graph, nodes_to_remove: Set[str]) -> int:
        """Remove nodes while preserving graph integrity"""
        removed_count = 0
        
        # Sort nodes by dependency order (remove dependents first)
        removal_order = self._get_safe_removal_order(nodes_to_remove)
        
        for node_id in removal_order:
            if self._can_remove_node_safely(ir_graph, node_id):
                node = ir_graph.get_node(node_id)
                if node:
                    # Update dependencies before removal
                    self._update_dependencies_for_removal(ir_graph, node)
                    
                    # Remove from nodes list
                    ir_graph.nodes = [n for n in ir_graph.nodes if n.id != node_id]
                    
                    # Remove from edges
                    ir_graph.edges = [e for e in ir_graph.edges 
                                    if e.get("from") != node_id and e.get("to") != node_id]
                    
                    removed_count += 1
                    logger.info(f"Removed dead code node: {node_id}")
        
        return removed_count
    
    def _get_safe_removal_order(self, nodes_to_remove: Set[str]) -> List[str]:
        """Get safe order for node removal (dependents before dependencies)"""
        # Simple topological sort for removal
        removal_order = []
        remaining = nodes_to_remove.copy()
        
        while remaining:
            # Find nodes with no dependents in remaining set
            candidates = []
            for node_id in remaining:
                dependents_in_remaining = (self.reverse_dependency_graph.get(node_id, set()) 
                                         & remaining)
                if not dependents_in_remaining:
                    candidates.append(node_id)
            
            if not candidates:
                # Break cycles by picking any remaining node
                candidates = [next(iter(remaining))]
            
            for candidate in candidates:
                removal_order.append(candidate)
                remaining.remove(candidate)
        
        return removal_order
    
    def _can_remove_node_safely(self, ir_graph, node_id: str) -> bool:
        """Check if node can be removed without breaking critical paths"""
        node = ir_graph.get_node(node_id)
        if not node:
            return False
        
        # Don't remove entry points unless they're truly unreachable
        if not node.dependencies and not self.dependency_graph.get(node_id):
            # Check if this is the only entry point
            entry_points = [n for n in ir_graph.nodes 
                          if not n.dependencies and not self.dependency_graph.get(n.id)]
            if len(entry_points) <= 1:
                return False
        
        # Don't remove nodes with critical side effects
        if self._has_critical_side_effects(node):
            return False
        
        return True
    
    def _has_critical_side_effects(self, node) -> bool:
        """Check if node has critical side effects that shouldn't be optimized away"""
        from .ir import IRNodeType
        
        # Nodes with external effects should not be removed
        critical_types = {
            IRNodeType.ACTION,
            IRNodeType.NOTIFY,
            IRNodeType.AGENT_CALL
        }
        
        if node.type in critical_types:
            return True
        
        # Check for external API calls or database operations
        if 'external_calls' in node.parameters:
            return True
        
        if 'side_effects' in node.parameters:
            return True
        
        return False
    
    def _update_dependencies_for_removal(self, ir_graph, node_to_remove):
        """Update dependencies when removing a node"""
        node_id = node_to_remove.id
        
        # Connect dependencies directly to dependents
        dependencies = self.dependency_graph.get(node_id, set())
        dependents = self.reverse_dependency_graph.get(node_id, set())
        
        for dependent_id in dependents:
            dependent_node = ir_graph.get_node(dependent_id)
            if dependent_node:
                # Remove reference to removed node
                if node_id in dependent_node.dependencies:
                    dependent_node.dependencies.remove(node_id)
                
                # Add direct dependencies to removed node's dependencies
                for dep_id in dependencies:
                    if dep_id not in dependent_node.dependencies:
                        dependent_node.dependencies.append(dep_id)
        
        # Update reverse dependencies
        for dep_id in dependencies:
            dep_node = ir_graph.get_node(dep_id)
            if dep_node:
                if node_id in dep_node.dependents:
                    dep_node.dependents.remove(node_id)
                
                # Add dependents of removed node
                for dependent_id in dependents:
                    if dependent_id not in dep_node.dependents:
                        dep_node.dependents.append(dependent_id)
    
    def _is_constant_value(self, value: str) -> bool:
        """Check if value is a constant"""
        value = value.strip()
        
        # Numeric constants
        try:
            float(value)
            return True
        except ValueError:
            pass
        
        # Boolean constants
        if value.lower() in ['true', 'false']:
            return True
        
        # String constants
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return True
        
        return False

# Test helper functions
def create_test_dead_code_graph():
    """Create a test graph with dead code for testing"""
    from .ir import IRGraph, IRNode, IRNodeType
    
    nodes = [
        IRNode(id="entry", type=IRNodeType.QUERY, dependents=["reachable"]),
        IRNode(id="reachable", type=IRNodeType.DECISION, 
               dependencies=["entry"], dependents=["terminal"]),
        IRNode(id="unreachable", type=IRNodeType.DECISION,  # Dead code
               parameters={'rules': [{'condition': 'false', 'action': 'never'}]}),
        IRNode(id="terminal", type=IRNodeType.ACTION, dependencies=["reachable"]),
        IRNode(id="unused_computation", type=IRNodeType.QUERY,  # Unused output
               parameters={'rules': [{'condition': 'true', 'action': {'result': 'constant'}}]})
    ]
    
    return IRGraph(
        workflow_id="test_dead_code",
        name="Test Dead Code Workflow",
        version="1.0.0",
        automation_type="rbia",
        nodes=nodes,
        edges=[
            {"from": "entry", "to": "reachable"},
            {"from": "reachable", "to": "terminal"}
        ]
    )

def run_dead_code_elimination_tests():
    """Run basic dead code elimination tests"""
    eliminator = DeadCodeEliminator()
    
    # Test with sample graph
    test_graph = create_test_dead_code_graph()
    
    print(f"Before optimization: {len(test_graph.nodes)} nodes")
    optimized, report = eliminator.eliminate_dead_code(test_graph)
    print(f"After optimization: {len(test_graph.nodes)} nodes")
    
    print(f"Optimization performed: {optimized}")
    print(f"Nodes removed: {report.nodes_removed}")
    print(f"Size reduction: {report.size_reduction_percent:.1f}%")
    
    for pruned in report.pruned_nodes:
        print(f"  Removed {pruned.node_id}: {pruned.removal_reason}")
    
    return report

if __name__ == "__main__":
    run_dead_code_elimination_tests()
