"""
Type System for RBIA Compiler
=============================

Task 6.2.5: Add Type System
- Define primitive types: scalar, enum, money, percent, date, vector
- Implement type checker on IR graph edges
- Add model IO validation for ML nodes

Dependencies: Task 6.2.4 (IR)
Outputs: Type-safe IR → input for analyzers (6.2.13+)
"""

from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re
from decimal import Decimal

# Task 6.2.5: Primitive Types
class PrimitiveType(Enum):
    """Primitive types for RBIA type system"""
    SCALAR = "scalar"        # Numbers (int, float)
    ENUM = "enum"           # Enumerated values
    MONEY = "money"         # Monetary values with currency
    PERCENT = "percent"     # Percentage values (0-100 or 0.0-1.0)
    DATE = "date"           # Date/datetime values
    VECTOR = "vector"       # Array/list of values
    STRING = "string"       # Text values
    BOOLEAN = "boolean"     # True/false values
    OBJECT = "object"       # Complex objects/dictionaries
    NULL = "null"           # Null/undefined values

@dataclass
class TypeConstraints:
    """Type constraints and validation rules"""
    min_value: Optional[Union[int, float, Decimal]] = None
    max_value: Optional[Union[int, float, Decimal]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    currency: Optional[str] = None  # Currency code for money types
    precision: Optional[int] = None  # Decimal precision
    vector_element_type: Optional['TypeDefinition'] = None
    vector_min_length: Optional[int] = None
    vector_max_length: Optional[int] = None
    required_fields: Optional[List[str]] = None  # For object types
    nullable: bool = False

@dataclass
class TypeDefinition:
    """Complete type definition with constraints"""
    primitive_type: PrimitiveType
    constraints: Optional[TypeConstraints] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = TypeConstraints()

class TypeChecker:
    """Type checker for RBIA compiler"""
    
    def __init__(self):
        self.type_errors: List[str] = []
        
    def validate_value(self, value: Any, type_def: TypeDefinition) -> bool:
        """Validate a value against a type definition"""
        self.type_errors.clear()
        
        if value is None:
            if type_def.constraints and type_def.constraints.nullable:
                return True
            else:
                self.type_errors.append("Value cannot be null")
                return False
        
        return self._validate_primitive_type(value, type_def)
    
    def _validate_primitive_type(self, value: Any, type_def: TypeDefinition) -> bool:
        """Validate against primitive type"""
        ptype = type_def.primitive_type
        constraints = type_def.constraints
        
        if ptype == PrimitiveType.SCALAR:
            return self._validate_scalar(value, constraints)
        elif ptype == PrimitiveType.ENUM:
            return self._validate_enum(value, constraints)
        elif ptype == PrimitiveType.MONEY:
            return self._validate_money(value, constraints)
        elif ptype == PrimitiveType.PERCENT:
            return self._validate_percent(value, constraints)
        elif ptype == PrimitiveType.DATE:
            return self._validate_date(value, constraints)
        elif ptype == PrimitiveType.VECTOR:
            return self._validate_vector(value, constraints)
        elif ptype == PrimitiveType.STRING:
            return self._validate_string(value, constraints)
        elif ptype == PrimitiveType.BOOLEAN:
            return self._validate_boolean(value, constraints)
        elif ptype == PrimitiveType.OBJECT:
            return self._validate_object(value, constraints)
        else:
            self.type_errors.append(f"Unknown primitive type: {ptype}")
            return False
    
    def _validate_scalar(self, value: Any, constraints: Optional[TypeConstraints]) -> bool:
        """Validate scalar (numeric) values"""
        if not isinstance(value, (int, float, Decimal)):
            self.type_errors.append(f"Expected numeric value, got {type(value).__name__}")
            return False
        
        if constraints:
            if constraints.min_value is not None and value < constraints.min_value:
                self.type_errors.append(f"Value {value} below minimum {constraints.min_value}")
                return False
            if constraints.max_value is not None and value > constraints.max_value:
                self.type_errors.append(f"Value {value} above maximum {constraints.max_value}")
                return False
        
        return True
    
    def _validate_enum(self, value: Any, constraints: Optional[TypeConstraints]) -> bool:
        """Validate enumerated values"""
        if not constraints or not constraints.allowed_values:
            self.type_errors.append("Enum type requires allowed_values constraint")
            return False
        
        if value not in constraints.allowed_values:
            self.type_errors.append(f"Value '{value}' not in allowed values: {constraints.allowed_values}")
            return False
        
        return True
    
    def _validate_money(self, value: Any, constraints: Optional[TypeConstraints]) -> bool:
        """Validate monetary values"""
        # Money can be represented as number or object with amount/currency
        if isinstance(value, (int, float, Decimal)):
            return self._validate_scalar(value, constraints)
        elif isinstance(value, dict):
            if 'amount' not in value:
                self.type_errors.append("Money object requires 'amount' field")
                return False
            if not self._validate_scalar(value['amount'], constraints):
                return False
            if constraints and constraints.currency:
                if value.get('currency') != constraints.currency:
                    self.type_errors.append(f"Expected currency {constraints.currency}, got {value.get('currency')}")
                    return False
            return True
        else:
            self.type_errors.append(f"Money value must be number or object, got {type(value).__name__}")
            return False
    
    def _validate_percent(self, value: Any, constraints: Optional[TypeConstraints]) -> bool:
        """Validate percentage values"""
        if not isinstance(value, (int, float, Decimal)):
            self.type_errors.append(f"Percent value must be numeric, got {type(value).__name__}")
            return False
        
        # Default percent range is 0-100, but can be 0.0-1.0
        min_val = constraints.min_value if constraints and constraints.min_value is not None else 0
        max_val = constraints.max_value if constraints and constraints.max_value is not None else 100
        
        if value < min_val or value > max_val:
            self.type_errors.append(f"Percent value {value} outside range [{min_val}, {max_val}]")
            return False
        
        return True
    
    def _validate_date(self, value: Any, constraints: Optional[TypeConstraints]) -> bool:
        """Validate date/datetime values"""
        if isinstance(value, datetime):
            return True
        elif isinstance(value, str):
            # Try to parse ISO format
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return True
            except ValueError:
                self.type_errors.append(f"Invalid date format: {value}")
                return False
        else:
            self.type_errors.append(f"Date value must be datetime or ISO string, got {type(value).__name__}")
            return False
    
    def _validate_vector(self, value: Any, constraints: Optional[TypeConstraints]) -> bool:
        """Validate vector (array) values"""
        if not isinstance(value, (list, tuple)):
            self.type_errors.append(f"Vector value must be list or tuple, got {type(value).__name__}")
            return False
        
        if constraints:
            if constraints.vector_min_length is not None and len(value) < constraints.vector_min_length:
                self.type_errors.append(f"Vector length {len(value)} below minimum {constraints.vector_min_length}")
                return False
            if constraints.vector_max_length is not None and len(value) > constraints.vector_max_length:
                self.type_errors.append(f"Vector length {len(value)} above maximum {constraints.vector_max_length}")
                return False
            
            # Validate element types
            if constraints.vector_element_type:
                for i, element in enumerate(value):
                    if not self.validate_value(element, constraints.vector_element_type):
                        self.type_errors.append(f"Vector element {i} validation failed")
                        return False
        
        return True
    
    def _validate_string(self, value: Any, constraints: Optional[TypeConstraints]) -> bool:
        """Validate string values"""
        if not isinstance(value, str):
            self.type_errors.append(f"Expected string value, got {type(value).__name__}")
            return False
        
        if constraints and constraints.pattern:
            if not re.match(constraints.pattern, value):
                self.type_errors.append(f"String '{value}' does not match pattern '{constraints.pattern}'")
                return False
        
        return True
    
    def _validate_boolean(self, value: Any, constraints: Optional[TypeConstraints]) -> bool:
        """Validate boolean values"""
        if not isinstance(value, bool):
            self.type_errors.append(f"Expected boolean value, got {type(value).__name__}")
            return False
        return True
    
    def _validate_object(self, value: Any, constraints: Optional[TypeConstraints]) -> bool:
        """Validate object (dictionary) values"""
        if not isinstance(value, dict):
            self.type_errors.append(f"Expected object value, got {type(value).__name__}")
            return False
        
        if constraints and constraints.required_fields:
            for field in constraints.required_fields:
                if field not in value:
                    self.type_errors.append(f"Required field '{field}' missing from object")
                    return False
        
        return True

# Task 6.2.5: IR Graph Type Checker
class IRGraphTypeChecker:
    """Type checker for IR graph edges and nodes"""
    
    def __init__(self):
        self.type_checker = TypeChecker()
        self.graph_errors: List[str] = []
        
    def check_ir_graph(self, ir_graph) -> bool:
        """Check types across entire IR graph"""
        from .ir import IRGraph  # Import here to avoid circular imports
        
        self.graph_errors.clear()
        
        # Check each node's input/output types
        for node in ir_graph.nodes:
            if not self._check_node_types(node):
                return False
        
        # Check edge type compatibility
        for edge in ir_graph.edges:
            if not self._check_edge_types(ir_graph, edge):
                return False
        
        return len(self.graph_errors) == 0
    
    def _check_node_types(self, node) -> bool:
        """Check types for a single IR node"""
        # Validate input types
        for input_spec in node.inputs:
            type_def = self._parse_type_string(input_spec.type)
            if not type_def:
                self.graph_errors.append(f"Invalid input type '{input_spec.type}' in node {node.id}")
                return False
        
        # Validate output types
        for output_spec in node.outputs:
            type_def = self._parse_type_string(output_spec.type)
            if not type_def:
                self.graph_errors.append(f"Invalid output type '{output_spec.type}' in node {node.id}")
                return False
        
        # Special validation for ML nodes
        if self._is_ml_node(node):
            return self._check_ml_node_types(node)
        
        return True
    
    def _check_edge_types(self, ir_graph, edge) -> bool:
        """Check type compatibility across an edge"""
        from_node = ir_graph.get_node(edge["from"])
        to_node = ir_graph.get_node(edge["to"])
        
        if not from_node or not to_node:
            self.graph_errors.append(f"Edge references non-existent nodes: {edge}")
            return False
        
        # Find matching output -> input type pairs
        for output in from_node.outputs:
            for input_spec in to_node.inputs:
                if input_spec.source == from_node.id:
                    # Check type compatibility
                    output_type = self._parse_type_string(output.type)
                    input_type = self._parse_type_string(input_spec.type)
                    
                    if not self._types_compatible(output_type, input_type):
                        self.graph_errors.append(
                            f"Type mismatch: {from_node.id}.{output.name} ({output.type}) "
                            f"-> {to_node.id}.{input_spec.name} ({input_spec.type})"
                        )
                        return False
        
        return True
    
    def _check_ml_node_types(self, node) -> bool:
        """Special type checking for ML nodes"""
        # ML nodes should have specific input/output patterns
        required_inputs = ["input_data"]
        required_outputs = ["prediction", "confidence"]
        
        input_names = [inp.name for inp in node.inputs]
        output_names = [out.name for out in node.outputs]
        
        for req_input in required_inputs:
            if req_input not in input_names:
                self.graph_errors.append(f"ML node {node.id} missing required input: {req_input}")
                return False
        
        for req_output in required_outputs:
            if req_output not in output_names:
                self.graph_errors.append(f"ML node {node.id} missing required output: {req_output}")
                return False
        
        # Validate confidence output is percent type
        confidence_output = next((out for out in node.outputs if out.name == "confidence"), None)
        if confidence_output:
            conf_type = self._parse_type_string(confidence_output.type)
            if conf_type and conf_type.primitive_type != PrimitiveType.PERCENT:
                self.graph_errors.append(f"ML node {node.id} confidence output must be percent type")
                return False
        
        return True
    
    def _is_ml_node(self, node) -> bool:
        """Check if node is an ML node"""
        from .ir import IRNodeType
        ml_types = {IRNodeType.ML_NODE, IRNodeType.ML_PREDICT, 
                   IRNodeType.ML_SCORE, IRNodeType.ML_CLASSIFY, IRNodeType.ML_EXPLAIN}
        return node.type in ml_types
    
    def _parse_type_string(self, type_str: str) -> Optional[TypeDefinition]:
        """Parse type string into TypeDefinition"""
        # Handle basic types
        if type_str in ["scalar", "number", "float", "int"]:
            return TypeDefinition(PrimitiveType.SCALAR)
        elif type_str == "enum":
            return TypeDefinition(PrimitiveType.ENUM)
        elif type_str == "money":
            return TypeDefinition(PrimitiveType.MONEY)
        elif type_str == "percent":
            return TypeDefinition(PrimitiveType.PERCENT)
        elif type_str in ["date", "datetime"]:
            return TypeDefinition(PrimitiveType.DATE)
        elif type_str in ["vector", "array", "list"]:
            return TypeDefinition(PrimitiveType.VECTOR)
        elif type_str == "string":
            return TypeDefinition(PrimitiveType.STRING)
        elif type_str == "boolean":
            return TypeDefinition(PrimitiveType.BOOLEAN)
        elif type_str == "object":
            return TypeDefinition(PrimitiveType.OBJECT)
        else:
            return None
    
    def _types_compatible(self, from_type: Optional[TypeDefinition], 
                         to_type: Optional[TypeDefinition]) -> bool:
        """Check if two types are compatible"""
        if not from_type or not to_type:
            return False
        
        # Exact match
        if from_type.primitive_type == to_type.primitive_type:
            return True
        
        # Compatible conversions
        compatible_conversions = {
            PrimitiveType.SCALAR: [PrimitiveType.MONEY, PrimitiveType.PERCENT],
            PrimitiveType.STRING: [PrimitiveType.ENUM],
            PrimitiveType.OBJECT: [PrimitiveType.VECTOR]  # Object can be converted to vector
        }
        
        return to_type.primitive_type in compatible_conversions.get(from_type.primitive_type, [])

# Task 6.2.5: Common Type Definitions
class CommonTypes:
    """Common type definitions for RBIA workflows"""
    
    # Scalar types
    AMOUNT = TypeDefinition(
        PrimitiveType.SCALAR,
        TypeConstraints(min_value=0, precision=2),
        "Monetary amount (non-negative)"
    )
    
    PROBABILITY = TypeDefinition(
        PrimitiveType.PERCENT,
        TypeConstraints(min_value=0.0, max_value=1.0),
        "Probability value (0.0-1.0)"
    )
    
    CONFIDENCE_SCORE = TypeDefinition(
        PrimitiveType.PERCENT,
        TypeConstraints(min_value=0.0, max_value=1.0),
        "ML confidence score (0.0-1.0)"
    )
    
    # Enum types
    OPPORTUNITY_STAGE = TypeDefinition(
        PrimitiveType.ENUM,
        TypeConstraints(allowed_values=["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]),
        "Sales opportunity stage"
    )
    
    AUTOMATION_TYPE = TypeDefinition(
        PrimitiveType.ENUM,
        TypeConstraints(allowed_values=["RBA", "RBIA", "AALA"]),
        "Workflow automation type"
    )
    
    # Money types
    USD_AMOUNT = TypeDefinition(
        PrimitiveType.MONEY,
        TypeConstraints(min_value=0, currency="USD", precision=2),
        "USD monetary amount"
    )
    
    # Date types
    CLOSE_DATE = TypeDefinition(
        PrimitiveType.DATE,
        description="Opportunity close date"
    )
    
    # Vector types
    FEATURE_VECTOR = TypeDefinition(
        PrimitiveType.VECTOR,
        TypeConstraints(
            vector_element_type=TypeDefinition(PrimitiveType.SCALAR),
            vector_min_length=1
        ),
        "ML feature vector"
    )
    
    # Object types
    OPPORTUNITY_DATA = TypeDefinition(
        PrimitiveType.OBJECT,
        TypeConstraints(required_fields=["id", "amount", "stage", "close_date"]),
        "Opportunity data object"
    )

# Task 6.2.5: Model IO Validation
class ModelIOValidator:
    """Validator for ML model input/output schemas"""
    
    def __init__(self):
        self.type_checker = TypeChecker()
        
    def validate_model_inputs(self, inputs: Dict[str, Any], 
                            input_schema: Dict[str, TypeDefinition]) -> bool:
        """Validate ML model inputs against schema"""
        for field_name, type_def in input_schema.items():
            if field_name not in inputs:
                if not type_def.constraints or not type_def.constraints.nullable:
                    return False
            else:
                if not self.type_checker.validate_value(inputs[field_name], type_def):
                    return False
        return True
    
    def validate_model_outputs(self, outputs: Dict[str, Any],
                             output_schema: Dict[str, TypeDefinition]) -> bool:
        """Validate ML model outputs against schema"""
        for field_name, type_def in output_schema.items():
            if field_name not in outputs:
                return False
            if not self.type_checker.validate_value(outputs[field_name], type_def):
                return False
        return True
    
    def get_model_input_schema(self, model_id: str) -> Dict[str, TypeDefinition]:
        """Get input schema for a model (would integrate with model registry)"""
        # Mock implementation - would query model registry
        schemas = {
            "saas_churn_predictor_v2": {
                "days_since_last_login": CommonTypes.AMOUNT,
                "mrr": CommonTypes.USD_AMOUNT,
                "support_tickets": CommonTypes.AMOUNT,
                "feature_usage_score": CommonTypes.PROBABILITY
            },
            "opportunity_score_model": {
                "amount": CommonTypes.USD_AMOUNT,
                "stage": CommonTypes.OPPORTUNITY_STAGE,
                "probability": CommonTypes.PROBABILITY,
                "days_to_close": CommonTypes.AMOUNT
            }
        }
        return schemas.get(model_id, {})
    
    def get_model_output_schema(self, model_id: str) -> Dict[str, TypeDefinition]:
        """Get output schema for a model"""
        # Mock implementation - would query model registry
        schemas = {
            "saas_churn_predictor_v2": {
                "churn_probability": CommonTypes.PROBABILITY,
                "confidence": CommonTypes.CONFIDENCE_SCORE,
                "risk_factors": TypeDefinition(PrimitiveType.VECTOR)
            },
            "opportunity_score_model": {
                "score": TypeDefinition(PrimitiveType.SCALAR, TypeConstraints(min_value=0, max_value=100)),
                "confidence": CommonTypes.CONFIDENCE_SCORE,
                "stage_recommendation": CommonTypes.OPPORTUNITY_STAGE
            }
        }
        return schemas.get(model_id, {})
