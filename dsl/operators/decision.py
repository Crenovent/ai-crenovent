"""
Decision Operator - Handles conditional logic and routing in workflows
"""

import re
from typing import Dict, List, Any, Optional
from .base import BaseOperator, OperatorContext, OperatorResult
import logging

logger = logging.getLogger(__name__)

class DecisionOperator(BaseOperator):
    """
    Decision operator for conditional workflow routing
    Supports:
    - Simple boolean expressions
    - Data field comparisons
    - Complex conditional logic
    - Multi-path routing (if/elseif/else)
    """
    
    def __init__(self, config=None):
        super().__init__("decision_operator")
        self.config = config or {}
    
    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate decision operator configuration"""
        errors = []
        
        # Check required fields
        if 'expression' not in config:
            errors.append("'expression' is required")
        
        # At least one routing path should be defined
        if not any(key in config for key in ['on_true', 'on_false', 'on_else']):
            errors.append("At least one routing path (on_true, on_false, on_else) must be defined")
        
        # Validate expression syntax (basic check)
        if 'expression' in config:
            try:
                self._validate_expression_syntax(config['expression'])
            except ValueError as e:
                errors.append(f"Invalid expression: {e}")
        
        return errors
    
    async def execute_async(self, context: OperatorContext, config: Dict[str, Any]) -> OperatorResult:
        """Execute the decision logic"""
        try:
            expression = config['expression']
            
            # Evaluate the expression
            result, explanation = await self._evaluate_expression(expression, context)
            
            # Determine next step based on result
            next_step = self._determine_next_step(result, config)
            
            return OperatorResult(
                success=True,
                output_data={
                    'decision_result': result,
                    'explanation': explanation,
                    'expression_evaluated': expression,
                    'next_step': next_step
                },
                next_step_id=next_step,
                confidence_score=1.0
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                error_message=f"Decision evaluation failed: {e}"
            )
    
    def _validate_expression_syntax(self, expression: str):
        """Basic validation of expression syntax"""
        # Check for dangerous operations
        dangerous_patterns = [
            r'import\s+',
            r'__\w+__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                raise ValueError(f"Dangerous operation detected in expression: {pattern}")
        
        # Basic syntax check (more comprehensive validation would go here)
        if not expression.strip():
            raise ValueError("Expression cannot be empty")
    
    async def _evaluate_expression(self, expression: str, context: OperatorContext) -> tuple[bool, str]:
        """
        Safely evaluate the decision expression
        
        Returns:
            tuple: (result: bool, explanation: str)
        """
        try:
            # Create safe evaluation context
            eval_context = self._create_evaluation_context(context)
            
            # Replace template variables
            processed_expression = self._process_template_variables(expression, eval_context)
            
            # Evaluate the expression safely
            result = self._safe_eval(processed_expression, eval_context)
            
            explanation = f"Expression '{processed_expression}' evaluated to {result}"
            
            return bool(result), explanation
            
        except Exception as e:
            logger.error(f"Expression evaluation error: {e}")
            raise ValueError(f"Failed to evaluate expression: {e}")
    
    def _create_evaluation_context(self, context: OperatorContext) -> Dict[str, Any]:
        """Create safe context for expression evaluation"""
        eval_context = {
            # Previous step outputs
            **context.previous_outputs,
            
            # User context
            'user_id': context.user_id,
            'tenant_id': context.tenant_id,
            
            # Common functions
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            
            # Custom helper functions
            'has_required_fields': self._has_required_fields,
            'count_non_empty': self._count_non_empty,
            'any_match': self._any_match,
            'all_match': self._all_match,
        }
        
        return eval_context
    
    def _process_template_variables(self, expression: str, eval_context: Dict[str, Any]) -> str:
        """Process template variables like {{ variable.field }}"""
        
        # Find all template variables
        template_pattern = r'\{\{\s*([^}]+)\s*\}\}'
        
        def replace_template(match):
            var_path = match.group(1).strip()
            try:
                # Navigate through nested objects
                value = eval_context
                for part in var_path.split('.'):
                    if isinstance(value, dict):
                        value = value.get(part)
                    elif isinstance(value, list) and part.isdigit():
                        value = value[int(part)]
                    else:
                        value = getattr(value, part, None)
                
                # Convert to string representation for expression
                if isinstance(value, str):
                    return f"'{value}'"
                elif value is None:
                    return 'None'
                else:
                    return str(value)
            except:
                return 'None'
        
        return re.sub(template_pattern, replace_template, expression)
    
    def _safe_eval(self, expression: str, eval_context: Dict[str, Any]) -> Any:
        """Safely evaluate expression with restricted context"""
        
        # Define allowed names for security
        allowed_names = {
            '__builtins__': {},
            **eval_context
        }
        
        try:
            # Use eval with restricted globals
            result = eval(expression, allowed_names)
            return result
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}")
    
    def _determine_next_step(self, result: bool, config: Dict[str, Any]) -> Optional[str]:
        """Determine next step based on decision result"""
        if result and 'on_true' in config:
            return config['on_true']
        elif not result and 'on_false' in config:
            return config['on_false']
        elif 'on_else' in config:
            return config['on_else']
        else:
            return None
    
    # Helper functions for expressions
    def _has_required_fields(self, data: Dict[str, Any], fields: List[str]) -> bool:
        """Check if all required fields are present and non-empty"""
        if not isinstance(data, dict):
            return False
        
        for field in fields:
            if field not in data or not data[field] or str(data[field]).strip() == '':
                return False
        return True
    
    def _count_non_empty(self, data: Dict[str, Any]) -> int:
        """Count non-empty fields in data"""
        if not isinstance(data, dict):
            return 0
        
        count = 0
        for value in data.values():
            if value and str(value).strip():
                count += 1
        return count
    
    def _any_match(self, data_list: List[Any], condition_func) -> bool:
        """Check if any item in list matches condition"""
        if not isinstance(data_list, list):
            return False
        
        return any(condition_func(item) for item in data_list)
    
    def _all_match(self, data_list: List[Any], condition_func) -> bool:
        """Check if all items in list match condition"""
        if not isinstance(data_list, list):
            return False
        
        return all(condition_func(item) for item in data_list)
