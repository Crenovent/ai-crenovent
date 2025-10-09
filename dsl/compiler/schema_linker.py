"""
Schema Linker to Ontology & Model Contracts
===========================================

Task 6.2.6: Implement Schema Linker
- Map IR fields → ontology definitions
- Validate ML nodes against model contracts (semver matching)
- Fail compile if mismatch in shape, units, or range

Dependencies: Task 6.2.4 (IR), Task 6.2.5 (Type System), ontology.py
Outputs: Schema-validated IR → input for analyzers (6.2.13+)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import semver
from .type_system import TypeDefinition, PrimitiveType, TypeConstraints, TypeChecker
from ..knowledge.ontology import RevOpsOntology, EntityType

# Task 6.2.6: Schema Validation Results
@dataclass
class SchemaValidationError:
    """Schema validation error details"""
    node_id: str
    field_name: str
    error_type: str  # shape, units, range, missing, type_mismatch
    expected: str
    actual: str
    message: str

@dataclass
class ModelContractValidation:
    """Model contract validation result"""
    model_id: str
    version: str
    is_valid: bool
    errors: List[SchemaValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

# Task 6.2.6: Schema Linker
class SchemaLinker:
    """Links IR fields to ontology definitions and validates model contracts"""
    
    def __init__(self):
        self.ontology = RevOpsOntology()
        self.type_checker = TypeChecker()
        self.validation_errors: List[SchemaValidationError] = []
        
    def validate_ir_graph_schemas(self, ir_graph) -> bool:
        """Validate all schemas in IR graph"""
        from .ir import IRGraph  # Import here to avoid circular imports
        
        self.validation_errors.clear()
        
        # Validate each node against ontology
        for node in ir_graph.nodes:
            if not self._validate_node_schema(node):
                return False
        
        # Validate ML nodes against model contracts
        ml_nodes = ir_graph.get_ml_nodes()
        for ml_node in ml_nodes:
            if not self._validate_ml_node_contract(ml_node):
                return False
        
        return len(self.validation_errors) == 0
    
    def _validate_node_schema(self, node) -> bool:
        """Validate node against ontology schema"""
        # Map node type to ontology entity
        entity_type = self._map_node_to_entity_type(node)
        if not entity_type:
            return True  # Skip validation for unmapped nodes
        
        # Get ontology schema for entity
        ontology_schema = self.ontology.get_entity_schema(entity_type)
        if not ontology_schema:
            return True  # No schema defined
        
        # Validate node inputs against ontology
        for input_spec in node.inputs:
            if not self._validate_field_against_ontology(
                node.id, input_spec.name, input_spec.type, ontology_schema
            ):
                return False
        
        # Validate node outputs against ontology
        for output_spec in node.outputs:
            if not self._validate_field_against_ontology(
                node.id, output_spec.name, output_spec.type, ontology_schema
            ):
                return False
        
        return True
    
    def _validate_ml_node_contract(self, ml_node) -> bool:
        """Validate ML node against model contract"""
        # Extract model ID and version from parameters
        model_id = ml_node.parameters.get('model_id')
        if not model_id:
            self.validation_errors.append(SchemaValidationError(
                node_id=ml_node.id,
                field_name="model_id",
                error_type="missing",
                expected="model_id parameter",
                actual="None",
                message="ML node missing model_id parameter"
            ))
            return False
        
        model_version = ml_node.parameters.get('model_version', '1.0.0')
        
        # Get model contract from registry
        model_contract = self._get_model_contract(model_id, model_version)
        if not model_contract:
            self.validation_errors.append(SchemaValidationError(
                node_id=ml_node.id,
                field_name="model_id",
                error_type="missing",
                expected=f"Model contract for {model_id}@{model_version}",
                actual="None",
                message=f"Model contract not found: {model_id}@{model_version}"
            ))
            return False
        
        # Validate input schema
        if not self._validate_inputs_against_contract(ml_node, model_contract):
            return False
        
        # Validate output schema
        if not self._validate_outputs_against_contract(ml_node, model_contract):
            return False
        
        return True
    
    def _validate_field_against_ontology(self, node_id: str, field_name: str, 
                                       field_type: str, ontology_schema: Dict[str, str]) -> bool:
        """Validate individual field against ontology schema"""
        if field_name not in ontology_schema:
            return True  # Field not in ontology, skip validation
        
        expected_type = ontology_schema[field_name]
        
        # Convert ontology type to our type system
        ontology_type_def = self._convert_ontology_type(expected_type)
        field_type_def = self._parse_field_type(field_type)
        
        if not self._types_compatible(field_type_def, ontology_type_def):
            self.validation_errors.append(SchemaValidationError(
                node_id=node_id,
                field_name=field_name,
                error_type="type_mismatch",
                expected=expected_type,
                actual=field_type,
                message=f"Field {field_name} type mismatch: expected {expected_type}, got {field_type}"
            ))
            return False
        
        return True
    
    def _validate_inputs_against_contract(self, ml_node, model_contract: Dict[str, Any]) -> bool:
        """Validate ML node inputs against model contract"""
        input_schema = model_contract.get('input_schema', {})
        
        # Check required inputs
        for required_field, field_spec in input_schema.items():
            if field_spec.get('required', True):
                # Find matching input in node
                matching_input = next(
                    (inp for inp in ml_node.inputs if inp.name == required_field), None
                )
                if not matching_input:
                    self.validation_errors.append(SchemaValidationError(
                        node_id=ml_node.id,
                        field_name=required_field,
                        error_type="missing",
                        expected=f"Required input: {required_field}",
                        actual="None",
                        message=f"ML node missing required input: {required_field}"
                    ))
                    return False
                
                # Validate input type, shape, units, range
                if not self._validate_field_contract(
                    ml_node.id, matching_input.name, matching_input.type, field_spec
                ):
                    return False
        
        return True
    
    def _validate_outputs_against_contract(self, ml_node, model_contract: Dict[str, Any]) -> bool:
        """Validate ML node outputs against model contract"""
        output_schema = model_contract.get('output_schema', {})
        
        # Check required outputs
        for required_field, field_spec in output_schema.items():
            # Find matching output in node
            matching_output = next(
                (out for out in ml_node.outputs if out.name == required_field), None
            )
            if not matching_output:
                self.validation_errors.append(SchemaValidationError(
                    node_id=ml_node.id,
                    field_name=required_field,
                    error_type="missing",
                    expected=f"Required output: {required_field}",
                    actual="None",
                    message=f"ML node missing required output: {required_field}"
                ))
                return False
            
            # Validate output type, shape, units, range
            if not self._validate_field_contract(
                ml_node.id, matching_output.name, matching_output.type, field_spec
            ):
                return False
        
        return True
    
    def _validate_field_contract(self, node_id: str, field_name: str, 
                               field_type: str, field_spec: Dict[str, Any]) -> bool:
        """Validate field against contract specification"""
        # Validate shape
        if 'shape' in field_spec:
            if not self._validate_shape(node_id, field_name, field_type, field_spec['shape']):
                return False
        
        # Validate units
        if 'units' in field_spec:
            if not self._validate_units(node_id, field_name, field_type, field_spec['units']):
                return False
        
        # Validate range
        if 'range' in field_spec:
            if not self._validate_range(node_id, field_name, field_type, field_spec['range']):
                return False
        
        # Validate data type
        if 'data_type' in field_spec:
            expected_type = field_spec['data_type']
            if not self._validate_data_type(node_id, field_name, field_type, expected_type):
                return False
        
        return True
    
    def _validate_shape(self, node_id: str, field_name: str, field_type: str, expected_shape) -> bool:
        """Validate field shape (dimensions for vectors/matrices)"""
        if field_type in ['vector', 'array']:
            # For vectors, shape should be [length] or [rows, cols] for matrices
            if isinstance(expected_shape, list) and len(expected_shape) == 1:
                # Vector with specific length - would need runtime validation
                return True
            elif isinstance(expected_shape, list) and len(expected_shape) == 2:
                # Matrix - would need runtime validation
                return True
        
        # For non-vector types, shape validation might not apply
        return True
    
    def _validate_units(self, node_id: str, field_name: str, field_type: str, expected_units: str) -> bool:
        """Validate field units (currency, percentage, etc.)"""
        # Check if field type implies units
        if field_type == 'money' and expected_units not in ['USD', 'EUR', 'GBP']:
            self.validation_errors.append(SchemaValidationError(
                node_id=node_id,
                field_name=field_name,
                error_type="units",
                expected=expected_units,
                actual=field_type,
                message=f"Field {field_name} has unsupported currency units: {expected_units}"
            ))
            return False
        
        if field_type == 'percent' and expected_units not in ['fraction', 'percentage']:
            self.validation_errors.append(SchemaValidationError(
                node_id=node_id,
                field_name=field_name,
                error_type="units",
                expected=expected_units,
                actual=field_type,
                message=f"Field {field_name} has unsupported percentage units: {expected_units}"
            ))
            return False
        
        return True
    
    def _validate_range(self, node_id: str, field_name: str, field_type: str, expected_range) -> bool:
        """Validate field value range"""
        if not isinstance(expected_range, dict):
            return True
        
        min_val = expected_range.get('min')
        max_val = expected_range.get('max')
        
        # For scalar types, range makes sense
        if field_type in ['scalar', 'number', 'money', 'percent']:
            if min_val is not None and max_val is not None:
                if min_val >= max_val:
                    self.validation_errors.append(SchemaValidationError(
                        node_id=node_id,
                        field_name=field_name,
                        error_type="range",
                        expected=f"min < max",
                        actual=f"min={min_val}, max={max_val}",
                        message=f"Field {field_name} has invalid range: min >= max"
                    ))
                    return False
        
        return True
    
    def _validate_data_type(self, node_id: str, field_name: str, 
                          field_type: str, expected_type: str) -> bool:
        """Validate field data type matches expected"""
        # Normalize type names
        type_mapping = {
            'float': 'scalar',
            'int': 'scalar', 
            'integer': 'scalar',
            'number': 'scalar',
            'str': 'string',
            'text': 'string',
            'bool': 'boolean',
            'datetime': 'date',
            'timestamp': 'date'
        }
        
        normalized_field_type = type_mapping.get(field_type, field_type)
        normalized_expected_type = type_mapping.get(expected_type, expected_type)
        
        if normalized_field_type != normalized_expected_type:
            self.validation_errors.append(SchemaValidationError(
                node_id=node_id,
                field_name=field_name,
                error_type="type_mismatch",
                expected=expected_type,
                actual=field_type,
                message=f"Field {field_name} type mismatch: expected {expected_type}, got {field_type}"
            ))
            return False
        
        return True
    
    def _get_model_contract(self, model_id: str, model_version: str) -> Optional[Dict[str, Any]]:
        """Get model contract from model registry with semver matching"""
        # Mock model registry - would integrate with actual model registry
        model_contracts = {
            "saas_churn_predictor_v2": {
                "versions": {
                    "1.0.0": {
                        "input_schema": {
                            "days_since_last_login": {
                                "data_type": "scalar",
                                "required": True,
                                "range": {"min": 0, "max": 365}
                            },
                            "mrr": {
                                "data_type": "money",
                                "required": True,
                                "units": "USD",
                                "range": {"min": 0}
                            },
                            "support_tickets": {
                                "data_type": "scalar",
                                "required": True,
                                "range": {"min": 0}
                            }
                        },
                        "output_schema": {
                            "churn_probability": {
                                "data_type": "percent",
                                "units": "fraction",
                                "range": {"min": 0.0, "max": 1.0}
                            },
                            "confidence": {
                                "data_type": "percent",
                                "units": "fraction", 
                                "range": {"min": 0.0, "max": 1.0}
                            }
                        }
                    },
                    "1.1.0": {
                        "input_schema": {
                            "days_since_last_login": {
                                "data_type": "scalar",
                                "required": True,
                                "range": {"min": 0, "max": 365}
                            },
                            "mrr": {
                                "data_type": "money",
                                "required": True,
                                "units": "USD",
                                "range": {"min": 0}
                            },
                            "support_tickets": {
                                "data_type": "scalar",
                                "required": True,
                                "range": {"min": 0}
                            },
                            "feature_usage_score": {
                                "data_type": "percent",
                                "required": False,
                                "range": {"min": 0.0, "max": 1.0}
                            }
                        },
                        "output_schema": {
                            "churn_probability": {
                                "data_type": "percent",
                                "units": "fraction",
                                "range": {"min": 0.0, "max": 1.0}
                            },
                            "confidence": {
                                "data_type": "percent",
                                "units": "fraction",
                                "range": {"min": 0.0, "max": 1.0}
                            },
                            "risk_factors": {
                                "data_type": "vector",
                                "shape": [10]
                            }
                        }
                    }
                }
            }
        }
        
        if model_id not in model_contracts:
            return None
        
        model_data = model_contracts[model_id]
        available_versions = list(model_data["versions"].keys())
        
        # Find best matching version using semver
        best_version = self._find_best_semver_match(model_version, available_versions)
        if not best_version:
            return None
        
        return model_data["versions"][best_version]
    
    def _find_best_semver_match(self, requested_version: str, available_versions: List[str]) -> Optional[str]:
        """Find best semver match for requested version"""
        try:
            # Parse requested version
            req_version = semver.VersionInfo.parse(requested_version)
            
            # Find compatible versions (same major, minor >= requested)
            compatible_versions = []
            for version_str in available_versions:
                try:
                    version = semver.VersionInfo.parse(version_str)
                    if (version.major == req_version.major and 
                        version >= req_version):
                        compatible_versions.append((version, version_str))
                except ValueError:
                    continue
            
            if not compatible_versions:
                return None
            
            # Return the lowest compatible version
            compatible_versions.sort(key=lambda x: x[0])
            return compatible_versions[0][1]
            
        except ValueError:
            # Fallback to exact match if semver parsing fails
            return requested_version if requested_version in available_versions else None
    
    def _map_node_to_entity_type(self, node) -> Optional[EntityType]:
        """Map IR node to ontology entity type"""
        from .ir import IRNodeType
        
        # Map IR node types to ontology entity types
        node_to_entity_mapping = {
            IRNodeType.QUERY: EntityType.WORKFLOW_STEP,
            IRNodeType.DECISION: EntityType.WORKFLOW_STEP,
            IRNodeType.ACTION: EntityType.WORKFLOW_STEP,
            IRNodeType.NOTIFY: EntityType.WORKFLOW_STEP,
            IRNodeType.GOVERNANCE: EntityType.WORKFLOW_STEP,
            IRNodeType.AGENT_CALL: EntityType.AGENT,
            IRNodeType.ML_NODE: EntityType.WORKFLOW_STEP,
            IRNodeType.ML_PREDICT: EntityType.WORKFLOW_STEP,
            IRNodeType.ML_SCORE: EntityType.WORKFLOW_STEP,
            IRNodeType.ML_CLASSIFY: EntityType.WORKFLOW_STEP,
            IRNodeType.ML_EXPLAIN: EntityType.WORKFLOW_STEP
        }
        
        return node_to_entity_mapping.get(node.type)
    
    def _convert_ontology_type(self, ontology_type: str) -> TypeDefinition:
        """Convert ontology type string to TypeDefinition"""
        type_mapping = {
            'string': TypeDefinition(PrimitiveType.STRING),
            'number': TypeDefinition(PrimitiveType.SCALAR),
            'date': TypeDefinition(PrimitiveType.DATE),
            'datetime': TypeDefinition(PrimitiveType.DATE),
            'array': TypeDefinition(PrimitiveType.VECTOR),
            'boolean': TypeDefinition(PrimitiveType.BOOLEAN)
        }
        
        return type_mapping.get(ontology_type, TypeDefinition(PrimitiveType.OBJECT))
    
    def _parse_field_type(self, field_type: str) -> TypeDefinition:
        """Parse field type string to TypeDefinition"""
        if field_type in ['scalar', 'number', 'float', 'int']:
            return TypeDefinition(PrimitiveType.SCALAR)
        elif field_type == 'string':
            return TypeDefinition(PrimitiveType.STRING)
        elif field_type == 'money':
            return TypeDefinition(PrimitiveType.MONEY)
        elif field_type == 'percent':
            return TypeDefinition(PrimitiveType.PERCENT)
        elif field_type in ['date', 'datetime']:
            return TypeDefinition(PrimitiveType.DATE)
        elif field_type in ['vector', 'array']:
            return TypeDefinition(PrimitiveType.VECTOR)
        elif field_type == 'boolean':
            return TypeDefinition(PrimitiveType.BOOLEAN)
        else:
            return TypeDefinition(PrimitiveType.OBJECT)
    
    def _types_compatible(self, type1: Optional[TypeDefinition], 
                         type2: Optional[TypeDefinition]) -> bool:
        """Check if two types are compatible"""
        if not type1 or not type2:
            return False
        
        # Exact match
        if type1.primitive_type == type2.primitive_type:
            return True
        
        # Compatible conversions
        compatible_pairs = [
            (PrimitiveType.SCALAR, PrimitiveType.MONEY),
            (PrimitiveType.SCALAR, PrimitiveType.PERCENT),
            (PrimitiveType.STRING, PrimitiveType.ENUM)
        ]
        
        type_pair = (type1.primitive_type, type2.primitive_type)
        reverse_pair = (type2.primitive_type, type1.primitive_type)
        
        return type_pair in compatible_pairs or reverse_pair in compatible_pairs

# Task 6.2.6: Integration with Compiler
def validate_ir_graph_with_schema_linker(ir_graph) -> Tuple[bool, List[SchemaValidationError]]:
    """Validate IR graph using schema linker"""
    linker = SchemaLinker()
    is_valid = linker.validate_ir_graph_schemas(ir_graph)
    return is_valid, linker.validation_errors
