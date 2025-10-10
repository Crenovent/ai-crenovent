"""
Constant Folding & Partial Evaluation - Task 6.2.15
===================================================

Constant folding & partial evaluation for rules
- Optimizer pass that identifies constant expressions in IR nodes
- Evaluates them at compile-time and replaces with computed constants
- Performs partial evaluation when only some inputs are constant  
- Preserves provenance (keeps mapping to original source for diagnostics)
- Tests to ensure semantics identical to interpreted runtime

Dependencies: Task 6.2.4 (IR)
Outputs: Optimized IR → reduced runtime cost, earlier error detection
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import ast
import operator
import re
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of constant folding optimization"""
    optimized: bool
    original_expression: str
    optimized_expression: str
    computed_value: Any
    provenance: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

@dataclass
class PartialEvaluationResult:
    """Result of partial evaluation"""
    simplified: bool
    original_node: Dict[str, Any]
    simplified_node: Dict[str, Any]
    constant_inputs: Dict[str, Any]
    remaining_variables: Set[str]
    provenance: Dict[str, Any]

class ExpressionType(Enum):
    """Types of expressions that can be optimized"""
    ARITHMETIC = "arithmetic"
    COMPARISON = "comparison"
    LOGICAL = "logical"
    STRING_OPERATION = "string_operation"
    FUNCTION_CALL = "function_call"
    FIELD_ACCESS = "field_access"
    CONSTANT = "constant"

# Task 6.2.15: Constant Folding & Partial Evaluation Optimizer
class ConstantFoldingOptimizer:
    """Optimizer pass for constant folding and partial evaluation"""
    
    def __init__(self):
        self.optimizations: List[OptimizationResult] = []
        self.partial_evaluations: List[PartialEvaluationResult] = []
        self.constant_values: Dict[str, Any] = {}
        self.safe_functions = {
            'abs', 'min', 'max', 'len', 'str', 'int', 'float', 'bool',
            'round', 'sum', 'any', 'all'
        }
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow,
            '==': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '<=': operator.le,
            '>': operator.gt,
            '>=': operator.ge,
            'and': operator.and_,
            'or': operator.or_,
            'not': operator.not_
        }
    
    def optimize_ir_graph(self, ir_graph) -> Tuple[bool, List[OptimizationResult]]:
        """Optimize IR graph with constant folding and partial evaluation"""
        from .ir import IRGraph
        
        self.optimizations.clear()
        self.partial_evaluations.clear()
        self.constant_values.clear()
        
        optimized = False
        
        # First pass: collect known constants
        self._collect_constants(ir_graph)
        
        # Second pass: optimize each node
        for node in ir_graph.nodes:
            if self._optimize_node(node):
                optimized = True
        
        # Third pass: partial evaluation for nodes with some constant inputs
        for node in ir_graph.nodes:
            if self._partial_evaluate_node(node):
                optimized = True
        
        return optimized, self.optimizations
    
    def _collect_constants(self, ir_graph):
        """Collect constant values from the IR graph"""
        for node in ir_graph.nodes:
            # Look for constant assignments in parameters
            if 'constants' in node.parameters:
                self.constant_values.update(node.parameters['constants'])
            
            # Look for literal values in rules
            if 'rules' in node.parameters:
                for rule in node.parameters['rules']:
                    self._extract_constants_from_rule(rule)
            
            # Look for constant inputs
            for input_spec in node.inputs:
                if input_spec.source and self._is_constant_value(input_spec.source):
                    self.constant_values[input_spec.name] = self._parse_constant(input_spec.source)
    
    def _extract_constants_from_rule(self, rule: Dict[str, Any]):
        """Extract constant values from rule definitions"""
        if 'condition' in rule:
            constants = self._find_constants_in_expression(rule['condition'])
            self.constant_values.update(constants)
        
        if 'action' in rule and isinstance(rule['action'], dict):
            for key, value in rule['action'].items():
                if self._is_constant_value(str(value)):
                    self.constant_values[f"action.{key}"] = value
    
    def _find_constants_in_expression(self, expression: str) -> Dict[str, Any]:
        """Find constant assignments in expressions"""
        constants = {}
        
        # Look for patterns like "variable = constant"
        assignments = re.findall(r'(\w+)\s*=\s*([^,\s]+)', expression)
        for var, val in assignments:
            if self._is_constant_value(val):
                constants[var] = self._parse_constant(val)
        
        return constants
    
    def _optimize_node(self, node) -> bool:
        """Optimize a single node with constant folding"""
        optimized = False
        
        # Optimize rule conditions
        if 'rules' in node.parameters:
            for i, rule in enumerate(node.parameters['rules']):
                if 'condition' in rule:
                    result = self._fold_constants_in_expression(
                        rule['condition'], 
                        f"{node.id}.rules[{i}].condition"
                    )
                    if result.optimized:
                        rule['condition'] = result.optimized_expression
                        self.optimizations.append(result)
                        optimized = True
                
                # Optimize action expressions
                if 'action' in rule and isinstance(rule['action'], dict):
                    for key, value in rule['action'].items():
                        if isinstance(value, str):
                            result = self._fold_constants_in_expression(
                                value,
                                f"{node.id}.rules[{i}].action.{key}"
                            )
                            if result.optimized:
                                rule['action'][key] = result.computed_value
                                self.optimizations.append(result)
                                optimized = True
        
        # Optimize decision trees
        if 'decision_tree' in node.parameters:
            if self._optimize_decision_tree(node.parameters['decision_tree'], node.id):
                optimized = True
        
        # Optimize parameter expressions
        for key, value in node.parameters.items():
            if isinstance(value, str) and self._is_expression(value):
                result = self._fold_constants_in_expression(value, f"{node.id}.{key}")
                if result.optimized:
                    node.parameters[key] = result.computed_value
                    self.optimizations.append(result)
                    optimized = True
        
        return optimized
    
    def _fold_constants_in_expression(self, expression: str, location: str) -> OptimizationResult:
        """Fold constants in a single expression"""
        original_expr = expression
        
        try:
            # Parse expression into AST
            parsed = ast.parse(expression, mode='eval')
            
            # Evaluate if it's a constant expression
            if self._is_constant_expression(parsed.body):
                computed_value = self._evaluate_constant_expression(parsed.body)
                
                return OptimizationResult(
                    optimized=True,
                    original_expression=original_expr,
                    optimized_expression=str(computed_value),
                    computed_value=computed_value,
                    provenance={
                        'location': location,
                        'optimization_type': 'constant_folding',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
            
            # Try partial constant folding
            optimized_ast = self._fold_constants_in_ast(parsed.body)
            if optimized_ast != parsed.body:
                optimized_expr = ast.unparse(optimized_ast)
                
                return OptimizationResult(
                    optimized=True,
                    original_expression=original_expr,
                    optimized_expression=optimized_expr,
                    computed_value=optimized_expr,
                    provenance={
                        'location': location,
                        'optimization_type': 'partial_constant_folding',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
        
        except Exception as e:
            logger.warning(f"Failed to optimize expression '{expression}': {e}")
            return OptimizationResult(
                optimized=False,
                original_expression=original_expr,
                optimized_expression=original_expr,
                computed_value=None,
                provenance={'location': location, 'error': str(e)},
                warnings=[f"Optimization failed: {e}"]
            )
        
        return OptimizationResult(
            optimized=False,
            original_expression=original_expr,
            optimized_expression=original_expr,
            computed_value=None,
            provenance={'location': location}
        )
    
    def _is_constant_expression(self, node: ast.AST) -> bool:
        """Check if AST node represents a constant expression"""
        if isinstance(node, ast.Constant):
            return True
        
        if isinstance(node, ast.Name):
            return node.id in self.constant_values
        
        if isinstance(node, ast.BinOp):
            return (self._is_constant_expression(node.left) and 
                   self._is_constant_expression(node.right))
        
        if isinstance(node, ast.UnaryOp):
            return self._is_constant_expression(node.operand)
        
        if isinstance(node, ast.Compare):
            return (self._is_constant_expression(node.left) and
                   all(self._is_constant_expression(comp) for comp in node.comparators))
        
        if isinstance(node, ast.BoolOp):
            return all(self._is_constant_expression(val) for val in node.values)
        
        if isinstance(node, ast.Call):
            return (isinstance(node.func, ast.Name) and 
                   node.func.id in self.safe_functions and
                   all(self._is_constant_expression(arg) for arg in node.args))
        
        return False
    
    def _evaluate_constant_expression(self, node: ast.AST) -> Any:
        """Evaluate a constant AST expression"""
        if isinstance(node, ast.Constant):
            return node.value
        
        if isinstance(node, ast.Name):
            return self.constant_values.get(node.id)
        
        if isinstance(node, ast.BinOp):
            left = self._evaluate_constant_expression(node.left)
            right = self._evaluate_constant_expression(node.right)
            op_func = self._get_operator_function(node.op)
            return op_func(left, right)
        
        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate_constant_expression(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.Not):
                return not operand
        
        if isinstance(node, ast.Compare):
            left = self._evaluate_constant_expression(node.left)
            result = True
            
            for op, comp in zip(node.ops, node.comparators):
                right = self._evaluate_constant_expression(comp)
                op_func = self._get_comparison_function(op)
                result = result and op_func(left, right)
                left = right
            
            return result
        
        if isinstance(node, ast.BoolOp):
            values = [self._evaluate_constant_expression(val) for val in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
        
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.safe_functions:
                args = [self._evaluate_constant_expression(arg) for arg in node.args]
                return getattr(__builtins__, func_name)(*args)
        
        raise ValueError(f"Cannot evaluate expression: {ast.dump(node)}")
    
    def _fold_constants_in_ast(self, node: ast.AST) -> ast.AST:
        """Fold constants in AST, returning optimized AST"""
        if self._is_constant_expression(node):
            try:
                value = self._evaluate_constant_expression(node)
                return ast.Constant(value=value)
            except:
                pass
        
        # Recursively optimize sub-expressions
        if isinstance(node, ast.BinOp):
            left = self._fold_constants_in_ast(node.left)
            right = self._fold_constants_in_ast(node.right)
            
            new_node = ast.BinOp(left=left, op=node.op, right=right)
            if self._is_constant_expression(new_node):
                try:
                    value = self._evaluate_constant_expression(new_node)
                    return ast.Constant(value=value)
                except:
                    pass
            return new_node
        
        if isinstance(node, ast.Compare):
            left = self._fold_constants_in_ast(node.left)
            comparators = [self._fold_constants_in_ast(comp) for comp in node.comparators]
            
            new_node = ast.Compare(left=left, ops=node.ops, comparators=comparators)
            if self._is_constant_expression(new_node):
                try:
                    value = self._evaluate_constant_expression(new_node)
                    return ast.Constant(value=value)
                except:
                    pass
            return new_node
        
        return node
    
    def _get_operator_function(self, op: ast.AST):
        """Get operator function for AST operator"""
        op_map = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow
        }
        return op_map.get(type(op), lambda x, y: None)
    
    def _get_comparison_function(self, op: ast.AST):
        """Get comparison function for AST comparison operator"""
        op_map = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Is: operator.is_,
            ast.IsNot: operator.is_not,
            ast.In: lambda x, y: x in y,
            ast.NotIn: lambda x, y: x not in y
        }
        return op_map.get(type(op), lambda x, y: None)
    
    def _optimize_decision_tree(self, tree: Dict[str, Any], node_id: str) -> bool:
        """Optimize decision tree structures"""
        optimized = False
        
        if 'condition' in tree:
            result = self._fold_constants_in_expression(
                tree['condition'], 
                f"{node_id}.decision_tree.condition"
            )
            if result.optimized:
                tree['condition'] = result.optimized_expression
                self.optimizations.append(result)
                optimized = True
        
        if 'branches' in tree:
            for i, branch in enumerate(tree['branches']):
                if self._optimize_decision_tree(branch, f"{node_id}.branch[{i}]"):
                    optimized = True
        
        return optimized
    
    def _partial_evaluate_node(self, node) -> bool:
        """Perform partial evaluation when some inputs are constant"""
        # This is a simplified implementation
        # Full partial evaluation would require more sophisticated analysis
        
        if not hasattr(node, 'inputs') or not node.inputs:
            return False
        
        constant_inputs = {}
        variable_inputs = set()
        
        # Identify which inputs are constant
        for input_spec in node.inputs:
            if input_spec.name in self.constant_values:
                constant_inputs[input_spec.name] = self.constant_values[input_spec.name]
            else:
                variable_inputs.add(input_spec.name)
        
        if not constant_inputs:
            return False
        
        # Try to simplify node logic with known constants
        original_params = node.parameters.copy()
        simplified = False
        
        if 'rules' in node.parameters:
            for rule in node.parameters['rules']:
                if 'condition' in rule:
                    simplified_condition = self._substitute_constants(
                        rule['condition'], 
                        constant_inputs
                    )
                    if simplified_condition != rule['condition']:
                        rule['condition'] = simplified_condition
                        simplified = True
        
        if simplified:
            self.partial_evaluations.append(PartialEvaluationResult(
                simplified=True,
                original_node=original_params,
                simplified_node=node.parameters,
                constant_inputs=constant_inputs,
                remaining_variables=variable_inputs,
                provenance={
                    'node_id': node.id,
                    'optimization_type': 'partial_evaluation',
                    'timestamp': datetime.utcnow().isoformat()
                }
            ))
        
        return simplified
    
    def _substitute_constants(self, expression: str, constants: Dict[str, Any]) -> str:
        """Substitute constant values into expression"""
        result = expression
        
        for var, value in constants.items():
            # Simple substitution - more sophisticated parsing would be better
            pattern = r'\b' + re.escape(var) + r'\b'
            if isinstance(value, str):
                replacement = f"'{value}'"
            else:
                replacement = str(value)
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _is_constant_value(self, value: str) -> bool:
        """Check if a string represents a constant value"""
        value = value.strip()
        
        # Check for numeric constants
        try:
            float(value)
            return True
        except ValueError:
            pass
        
        # Check for boolean constants
        if value.lower() in ['true', 'false']:
            return True
        
        # Check for string constants (quoted)
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return True
        
        return False
    
    def _parse_constant(self, value: str) -> Any:
        """Parse a constant value from string"""
        value = value.strip()
        
        # Try numeric
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Try boolean
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        
        # Try string (remove quotes)
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        return value
    
    def _is_expression(self, value: str) -> bool:
        """Check if string contains an expression that can be optimized"""
        # Simple heuristic - look for operators
        operators = ['+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>=', 'and', 'or']
        return any(op in value for op in operators)

# Test helper functions
def create_test_optimization_node():
    """Create a test node for optimization testing"""
    from .ir import IRNode, IRNodeType
    
    return IRNode(
        id="test_optimization_1",
        type=IRNodeType.DECISION,
        parameters={
            'constants': {'max_amount': 1000, 'min_amount': 100},
            'rules': [
                {'condition': '5 + 3 > 7', 'action': 'always_true'},
                {'condition': 'amount > max_amount', 'action': 'high_amount'},
                {'condition': '2 * 3 + 4', 'action': 'computed_to_10'},
                {'condition': 'len("hello") == 5', 'action': 'string_length'}
            ]
        }
    )

def run_optimization_tests():
    """Run basic optimization tests"""
    optimizer = ConstantFoldingOptimizer()
    
    # Test with sample node
    test_node = create_test_optimization_node()
    from .ir import IRGraph
    
    test_graph = IRGraph(
        workflow_id="test_optimization",
        name="Test Optimization Workflow",
        version="1.0.0", 
        automation_type="rbia",
        nodes=[test_node]
    )
    
    optimized, results = optimizer.optimize_ir_graph(test_graph)
    
    print(f"Optimization completed: {optimized}")
    print(f"Found {len(results)} optimizations:")
    for result in results:
        print(f"  {result.original_expression} → {result.optimized_expression}")
    
    return results

if __name__ == "__main__":
    run_optimization_tests()
