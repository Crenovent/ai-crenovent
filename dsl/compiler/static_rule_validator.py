"""
Static Rule Validator - Task 6.2.13
====================================

Static rule validation (unreachable branches, missing else)
- Analyzer pass over IR that performs control-flow analysis on rule nodes
- Detects unreachable branches (conditions that are always false)
- Detects missing else or default handlers
- Produces human-friendly diagnostics with location, reasoning, suggested fix
- Tests with representative rule graphs and counterexamples

Dependencies: Task 6.2.4 (IR)
Outputs: Validation diagnostics → prevents runtime errors
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import ast
import re

logger = logging.getLogger(__name__)

@dataclass
class ControlFlowDiagnostic:
    """Diagnostic for control flow analysis issues"""
    error_code: str
    node_id: str
    severity: str  # "error", "warning", "info"
    message: str
    reasoning: str
    suggested_fix: Optional[str] = None
    source_location: Optional[Dict[str, Any]] = None

class ConditionType(Enum):
    """Types of conditions in rule nodes"""
    BOOLEAN_LITERAL = "boolean_literal"
    COMPARISON = "comparison"
    LOGICAL_AND = "logical_and" 
    LOGICAL_OR = "logical_or"
    FIELD_CHECK = "field_check"
    COMPLEX_EXPRESSION = "complex_expression"

@dataclass
class ConditionAnalysis:
    """Analysis result for a condition"""
    type: ConditionType
    is_always_true: bool = False
    is_always_false: bool = False
    depends_on_fields: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)

# Task 6.2.13: Static Rule Validator
class StaticRuleValidator:
    """Analyzer pass for static rule validation"""
    
    def __init__(self):
        self.diagnostics: List[ControlFlowDiagnostic] = []
        self.field_constraints: Dict[str, Dict[str, Any]] = {}
        
    def validate_ir_graph(self, ir_graph) -> List[ControlFlowDiagnostic]:
        """Validate static rules in IR graph"""
        from .ir import IRGraph, IRNodeType
        
        self.diagnostics.clear()
        self.field_constraints.clear()
        
        # Analyze each rule node
        for node in ir_graph.nodes:
            if self._is_rule_node(node):
                self._analyze_rule_node(node, ir_graph)
        
        # Perform global control flow analysis
        self._analyze_global_control_flow(ir_graph)
        
        return self.diagnostics
    
    def _is_rule_node(self, node) -> bool:
        """Check if node contains rule logic"""
        from .ir import IRNodeType
        
        rule_types = {
            IRNodeType.DECISION, 
            IRNodeType.QUERY,
            IRNodeType.GOVERNANCE
        }
        return node.type in rule_types
    
    def _analyze_rule_node(self, node, ir_graph):
        """Analyze a single rule node for control flow issues"""
        # Check for unreachable branches
        self._check_unreachable_branches(node)
        
        # Check for missing else/default handlers
        self._check_missing_default_handlers(node)
        
        # Check for contradictory conditions
        self._check_contradictory_conditions(node)
        
        # Check for redundant conditions
        self._check_redundant_conditions(node)
    
    def _check_unreachable_branches(self, node):
        """Detect unreachable branches in rule conditions"""
        conditions = self._extract_conditions(node)
        
        for i, condition in enumerate(conditions):
            analysis = self._analyze_condition(condition)
            
            if analysis.is_always_false:
                self.diagnostics.append(ControlFlowDiagnostic(
                    error_code="E013",
                    node_id=node.id,
                    severity="warning",
                    message=f"Unreachable branch detected in condition {i+1}",
                    reasoning=f"Condition '{condition}' is always false",
                    suggested_fix="Remove this branch or fix the condition logic",
                    source_location={"node_id": node.id, "condition_index": i}
                ))
            
            # Check if condition is made unreachable by previous conditions
            if self._is_condition_unreachable(condition, conditions[:i]):
                self.diagnostics.append(ControlFlowDiagnostic(
                    error_code="E014",
                    node_id=node.id,
                    severity="warning", 
                    message=f"Branch {i+1} is unreachable due to previous conditions",
                    reasoning=f"Previous conditions make '{condition}' impossible to reach",
                    suggested_fix="Reorder conditions or remove redundant logic",
                    source_location={"node_id": node.id, "condition_index": i}
                ))
    
    def _check_missing_default_handlers(self, node):
        """Check for missing else/default branches"""
        conditions = self._extract_conditions(node)
        has_default = self._has_default_handler(node)
        
        if not has_default and not self._conditions_cover_all_cases(conditions):
            self.diagnostics.append(ControlFlowDiagnostic(
                error_code="E015",
                node_id=node.id,
                severity="error",
                message="Missing default/else handler",
                reasoning="Conditions do not cover all possible input cases",
                suggested_fix="Add an 'else' clause or ensure conditions are exhaustive",
                source_location={"node_id": node.id}
            ))
    
    def _check_contradictory_conditions(self, node):
        """Check for contradictory conditions that can never be true together"""
        conditions = self._extract_conditions(node)
        
        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions[i+1:], i+1):
                if self._are_conditions_contradictory(cond1, cond2):
                    self.diagnostics.append(ControlFlowDiagnostic(
                        error_code="E016",
                        node_id=node.id,
                        severity="error",
                        message=f"Contradictory conditions found",
                        reasoning=f"Conditions '{cond1}' and '{cond2}' cannot both be true",
                        suggested_fix="Review and fix the logical contradiction",
                        source_location={"node_id": node.id, "conditions": [i, j]}
                    ))
    
    def _check_redundant_conditions(self, node):
        """Check for redundant conditions"""
        conditions = self._extract_conditions(node)
        
        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions[i+1:], i+1):
                if self._are_conditions_equivalent(cond1, cond2):
                    self.diagnostics.append(ControlFlowDiagnostic(
                        error_code="E017",
                        node_id=node.id,
                        severity="warning",
                        message=f"Redundant condition detected",
                        reasoning=f"Conditions '{cond1}' and '{cond2}' are equivalent",
                        suggested_fix="Remove the duplicate condition",
                        source_location={"node_id": node.id, "conditions": [i, j]}
                    ))
    
    def _extract_conditions(self, node) -> List[str]:
        """Extract condition expressions from rule node"""
        conditions = []
        
        # Check parameters for condition logic
        if 'conditions' in node.parameters:
            conditions.extend(node.parameters['conditions'])
        
        # Check for if-else structures
        if 'rules' in node.parameters:
            for rule in node.parameters['rules']:
                if 'condition' in rule:
                    conditions.append(rule['condition'])
        
        # Check for decision tree structures
        if 'decision_tree' in node.parameters:
            conditions.extend(self._extract_tree_conditions(node.parameters['decision_tree']))
        
        return conditions
    
    def _extract_tree_conditions(self, tree: Dict[str, Any]) -> List[str]:
        """Extract conditions from decision tree structure"""
        conditions = []
        
        if 'condition' in tree:
            conditions.append(tree['condition'])
        
        if 'branches' in tree:
            for branch in tree['branches']:
                conditions.extend(self._extract_tree_conditions(branch))
        
        return conditions
    
    def _analyze_condition(self, condition: str) -> ConditionAnalysis:
        """Analyze a single condition for static properties"""
        condition = condition.strip()
        
        # Boolean literals
        if condition.lower() in ['true', 'false']:
            return ConditionAnalysis(
                type=ConditionType.BOOLEAN_LITERAL,
                is_always_true=(condition.lower() == 'true'),
                is_always_false=(condition.lower() == 'false')
            )
        
        # Simple comparisons
        if any(op in condition for op in ['==', '!=', '<', '>', '<=', '>=']):
            return self._analyze_comparison(condition)
        
        # Logical operators
        if ' and ' in condition.lower():
            return self._analyze_logical_and(condition)
        
        if ' or ' in condition.lower():
            return self._analyze_logical_or(condition)
        
        # Field checks
        if re.match(r'^\w+\s*[<>=!]+\s*[\w\d.]+$', condition):
            return self._analyze_field_check(condition)
        
        # Complex expressions
        return ConditionAnalysis(type=ConditionType.COMPLEX_EXPRESSION)
    
    def _analyze_comparison(self, condition: str) -> ConditionAnalysis:
        """Analyze comparison expressions"""
        # Extract field dependencies
        fields = set(re.findall(r'\b[a-zA-Z_]\w*\b', condition))
        
        # Remove operators and constants
        fields = {f for f in fields if f not in ['true', 'false', 'and', 'or', 'not']}
        
        return ConditionAnalysis(
            type=ConditionType.COMPARISON,
            depends_on_fields=fields
        )
    
    def _analyze_logical_and(self, condition: str) -> ConditionAnalysis:
        """Analyze logical AND expressions"""
        parts = [p.strip() for p in condition.lower().split(' and ')]
        fields = set()
        
        for part in parts:
            part_analysis = self._analyze_condition(part)
            fields.update(part_analysis.depends_on_fields)
        
        return ConditionAnalysis(
            type=ConditionType.LOGICAL_AND,
            depends_on_fields=fields
        )
    
    def _analyze_logical_or(self, condition: str) -> ConditionAnalysis:
        """Analyze logical OR expressions"""
        parts = [p.strip() for p in condition.lower().split(' or ')]
        fields = set()
        
        for part in parts:
            part_analysis = self._analyze_condition(part)
            fields.update(part_analysis.depends_on_fields)
        
        return ConditionAnalysis(
            type=ConditionType.LOGICAL_OR,
            depends_on_fields=fields
        )
    
    def _analyze_field_check(self, condition: str) -> ConditionAnalysis:
        """Analyze field check expressions"""
        # Extract field name
        field_match = re.match(r'^(\w+)\s*([<>=!]+)\s*([\w\d.]+)$', condition)
        if field_match:
            field_name = field_match.group(1)
            operator = field_match.group(2)
            value = field_match.group(3)
            
            return ConditionAnalysis(
                type=ConditionType.FIELD_CHECK,
                depends_on_fields={field_name},
                constraints={field_name: {'operator': operator, 'value': value}}
            )
        
        return ConditionAnalysis(type=ConditionType.FIELD_CHECK)
    
    def _is_condition_unreachable(self, condition: str, previous_conditions: List[str]) -> bool:
        """Check if condition is unreachable due to previous conditions"""
        current_analysis = self._analyze_condition(condition)
        
        for prev_condition in previous_conditions:
            prev_analysis = self._analyze_condition(prev_condition)
            
            # If previous condition is always true and covers same fields,
            # current condition might be unreachable
            if (prev_analysis.is_always_true and 
                current_analysis.depends_on_fields.issubset(prev_analysis.depends_on_fields)):
                return True
            
            # Check for logical contradictions
            if self._are_conditions_contradictory(condition, prev_condition):
                return True
        
        return False
    
    def _has_default_handler(self, node) -> bool:
        """Check if node has a default/else handler"""
        # Check for explicit else clause
        if 'else' in node.parameters:
            return True
        
        # Check for default action in rules
        if 'rules' in node.parameters:
            for rule in node.parameters['rules']:
                if rule.get('is_default', False) or rule.get('condition', '').lower() == 'else':
                    return True
        
        # Check for catch-all condition
        conditions = self._extract_conditions(node)
        return any(cond.lower() in ['true', 'default', 'else'] for cond in conditions)
    
    def _conditions_cover_all_cases(self, conditions: List[str]) -> bool:
        """Check if conditions exhaustively cover all possible cases"""
        # Simple heuristic: if we have boolean literals or catch-all conditions
        for condition in conditions:
            if condition.lower() in ['true', 'default', 'else']:
                return True
        
        # More sophisticated analysis would require constraint solving
        # For now, return False to be conservative
        return False
    
    def _are_conditions_contradictory(self, cond1: str, cond2: str) -> bool:
        """Check if two conditions are contradictory"""
        analysis1 = self._analyze_condition(cond1)
        analysis2 = self._analyze_condition(cond2)
        
        # Simple contradiction detection
        if (analysis1.is_always_true and analysis2.is_always_false) or \
           (analysis1.is_always_false and analysis2.is_always_true):
            return True
        
        # Check for field constraint contradictions
        common_fields = analysis1.depends_on_fields & analysis2.depends_on_fields
        for field in common_fields:
            if (field in analysis1.constraints and 
                field in analysis2.constraints):
                # Simple contradiction check for same field
                const1 = analysis1.constraints[field]
                const2 = analysis2.constraints[field]
                if self._are_constraints_contradictory(const1, const2):
                    return True
        
        return False
    
    def _are_conditions_equivalent(self, cond1: str, cond2: str) -> bool:
        """Check if two conditions are equivalent"""
        # Normalize whitespace and case
        norm1 = ' '.join(cond1.lower().split())
        norm2 = ' '.join(cond2.lower().split())
        
        return norm1 == norm2
    
    def _are_constraints_contradictory(self, const1: Dict[str, Any], const2: Dict[str, Any]) -> bool:
        """Check if two field constraints are contradictory"""
        op1, val1 = const1.get('operator'), const1.get('value')
        op2, val2 = const2.get('operator'), const2.get('value')
        
        if not (op1 and val1 and op2 and val2):
            return False
        
        try:
            # Convert values to numbers for comparison
            num1, num2 = float(val1), float(val2)
            
            # Check for obvious contradictions
            if op1 == '==' and op2 == '==' and num1 != num2:
                return True
            if op1 == '>' and op2 == '<' and num1 >= num2:
                return True
            if op1 == '<' and op2 == '>' and num1 <= num2:
                return True
                
        except ValueError:
            # String comparison
            if op1 == '==' and op2 == '==' and val1 != val2:
                return True
        
        return False
    
    def _analyze_global_control_flow(self, ir_graph):
        """Analyze global control flow patterns"""
        # Check for unreachable nodes
        self._check_unreachable_nodes(ir_graph)
        
        # Check for infinite loops
        self._check_infinite_loops(ir_graph)
    
    def _check_unreachable_nodes(self, ir_graph):
        """Check for nodes that are never reached"""
        # Build reachability graph
        reachable = set()
        
        # Find entry points (nodes with no dependencies)
        entry_points = [node for node in ir_graph.nodes if not node.dependencies]
        
        # DFS from entry points
        for entry in entry_points:
            self._mark_reachable(entry.id, ir_graph, reachable)
        
        # Report unreachable nodes
        for node in ir_graph.nodes:
            if node.id not in reachable:
                self.diagnostics.append(ControlFlowDiagnostic(
                    error_code="E018",
                    node_id=node.id,
                    severity="warning",
                    message="Unreachable node detected",
                    reasoning=f"Node '{node.id}' is never reached in any execution path",
                    suggested_fix="Remove unreachable node or fix graph connectivity",
                    source_location={"node_id": node.id}
                ))
    
    def _mark_reachable(self, node_id: str, ir_graph, reachable: Set[str]):
        """Mark node and its dependents as reachable"""
        if node_id in reachable:
            return
        
        reachable.add(node_id)
        node = ir_graph.get_node(node_id)
        
        if node:
            for dependent_id in node.dependents:
                self._mark_reachable(dependent_id, ir_graph, reachable)
    
    def _check_infinite_loops(self, ir_graph):
        """Check for potential infinite loops in rule logic"""
        # This is a simplified check - more sophisticated analysis would be needed
        # for complex loop detection
        
        visited = set()
        rec_stack = set()
        
        for node in ir_graph.nodes:
            if node.id not in visited:
                if self._has_cycle_dfs(node.id, ir_graph, visited, rec_stack):
                    self.diagnostics.append(ControlFlowDiagnostic(
                        error_code="E019",
                        node_id=node.id,
                        severity="error",
                        message="Potential infinite loop detected",
                        reasoning=f"Cycle detected involving node '{node.id}'",
                        suggested_fix="Review node dependencies to break the cycle",
                        source_location={"node_id": node.id}
                    ))
    
    def _has_cycle_dfs(self, node_id: str, ir_graph, visited: Set[str], rec_stack: Set[str]) -> bool:
        """DFS-based cycle detection"""
        visited.add(node_id)
        rec_stack.add(node_id)
        
        node = ir_graph.get_node(node_id)
        if node:
            for dependent_id in node.dependents:
                if dependent_id not in visited:
                    if self._has_cycle_dfs(dependent_id, ir_graph, visited, rec_stack):
                        return True
                elif dependent_id in rec_stack:
                    return True
        
        rec_stack.remove(node_id)
        return False

# Test helper functions
def create_test_rule_node():
    """Create a test rule node for validation testing"""
    from .ir import IRNode, IRNodeType
    
    return IRNode(
        id="test_rule_1",
        type=IRNodeType.DECISION,
        parameters={
            'rules': [
                {'condition': 'amount > 1000', 'action': 'approve'},
                {'condition': 'amount < 500', 'action': 'auto_approve'},
                {'condition': 'false', 'action': 'never_reached'},  # Unreachable
                {'condition': 'amount > 1000', 'action': 'duplicate'}  # Redundant
            ]
        }
    )

def run_validation_tests():
    """Run basic validation tests"""
    validator = StaticRuleValidator()
    
    # Test with sample rule node
    test_node = create_test_rule_node()
    from .ir import IRGraph
    
    test_graph = IRGraph(
        workflow_id="test_validation",
        name="Test Validation Workflow", 
        version="1.0.0",
        automation_type="rbia",
        nodes=[test_node]
    )
    
    diagnostics = validator.validate_ir_graph(test_graph)
    
    print(f"Found {len(diagnostics)} validation issues:")
    for diag in diagnostics:
        print(f"  {diag.error_code}: {diag.message}")
    
    return diagnostics

if __name__ == "__main__":
    run_validation_tests()
