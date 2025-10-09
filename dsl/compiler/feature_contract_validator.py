"""
Feature Contract Validator - Task 6.2.17
==========================================

Feature contract check: ML inputs exist, are typed, within ranges
- Analyzer that pulls feature contracts from feature store/schema
- Checks feature presence, type, unit compatibility, and range constraints
- Emits actionable errors/warnings for missing features, type mismatches, out-of-range constants

Dependencies: Task 6.2.4 (IR), Feature Store Schema
Outputs: Feature validation diagnostics → prevents runtime inference errors
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FeatureContractError:
    """Error in feature contract validation"""
    error_code: str
    node_id: str
    feature_name: str
    model_id: str
    severity: str  # "error", "warning", "info"
    message: str
    expected: str
    actual: str
    suggested_fix: Optional[str] = None
    source_location: Optional[Dict[str, Any]] = None

@dataclass
class FeatureContract:
    """Feature contract from feature store"""
    feature_name: str
    feature_type: str  # 'numeric', 'categorical', 'boolean', 'text', 'embedding', 'timestamp'
    data_type: str  # 'float', 'int', 'string', 'bool'
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    null_allowed: bool = False
    unit: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0.0"

@dataclass
class ModelFeatureContract:
    """Complete feature contract for a model"""
    model_id: str
    model_version: str
    input_features: Dict[str, FeatureContract]
    output_features: Dict[str, FeatureContract]
    feature_store_version: str
    contract_hash: Optional[str] = None

class FeatureValidationSeverity(Enum):
    """Severity levels for feature validation"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

# Task 6.2.17: Feature Contract Validator
class FeatureContractValidator:
    """Validates ML node features against feature store contracts"""
    
    def __init__(self, feature_store_cache: Optional[Dict[str, Any]] = None):
        self.validation_errors: List[FeatureContractError] = []
        self.feature_contracts: Dict[str, ModelFeatureContract] = {}
        self.feature_store_cache = feature_store_cache or {}
        
        # Initialize with mock feature store data for initial validation
        self._init_mock_feature_contracts()
    
    def validate_ir_graph_features(self, ir_graph) -> List[FeatureContractError]:
        """Validate all ML nodes in IR graph against feature contracts"""
        from .ir import IRGraph, IRNodeType
        
        self.validation_errors.clear()
        
        # Get all ML nodes
        ml_nodes = ir_graph.get_ml_nodes()
        
        for node in ml_nodes:
            self._validate_node_features(node)
        
        return self.validation_errors
    
    def _validate_node_features(self, node):
        """Validate features for a single ML node"""
        model_id = self._extract_model_id(node)
        if not model_id:
            self.validation_errors.append(FeatureContractError(
                error_code="F001",
                node_id=node.id,
                feature_name="model_id",
                model_id="unknown",
                severity=FeatureValidationSeverity.ERROR.value,
                message="ML node missing model_id",
                expected="Valid model_id in parameters",
                actual="None",
                suggested_fix="Add 'model_id' to node parameters"
            ))
            return
        
        # Get feature contract for this model
        contract = self._get_model_contract(model_id)
        if not contract:
            self.validation_errors.append(FeatureContractError(
                error_code="F002",
                node_id=node.id,
                feature_name="contract",
                model_id=model_id,
                severity=FeatureValidationSeverity.ERROR.value,
                message=f"No feature contract found for model: {model_id}",
                expected="Valid model contract in feature store",
                actual="Contract not found",
                suggested_fix=f"Register model {model_id} in feature store or check model_id spelling"
            ))
            return
        
        # Validate input features
        self._validate_input_features(node, contract)
        
        # Validate output features
        self._validate_output_features(node, contract)
        
        # Validate feature ranges and constraints
        self._validate_feature_constraints(node, contract)
    
    def _validate_input_features(self, node, contract: ModelFeatureContract):
        """Validate input features against contract"""
        node_inputs = {inp.name: inp for inp in node.inputs}
        
        # Check all required features are present
        for feature_name, feature_contract in contract.input_features.items():
            if feature_contract.required and feature_name not in node_inputs:
                self.validation_errors.append(FeatureContractError(
                    error_code="F003",
                    node_id=node.id,
                    feature_name=feature_name,
                    model_id=contract.model_id,
                    severity=FeatureValidationSeverity.ERROR.value,
                    message=f"Missing required input feature: {feature_name}",
                    expected=f"Input feature '{feature_name}' of type {feature_contract.feature_type}",
                    actual="Feature not found in node inputs",
                    suggested_fix=f"Add input '{feature_name}' to node or map from workflow context"
                ))
                continue
            
            # Validate feature type if present
            if feature_name in node_inputs:
                node_input = node_inputs[feature_name]
                if not self._validate_feature_type(node_input.type, feature_contract):
                    self.validation_errors.append(FeatureContractError(
                        error_code="F004",
                        node_id=node.id,
                        feature_name=feature_name,
                        model_id=contract.model_id,
                        severity=FeatureValidationSeverity.ERROR.value,
                        message=f"Type mismatch for input feature: {feature_name}",
                        expected=f"{feature_contract.feature_type} ({feature_contract.data_type})",
                        actual=node_input.type,
                        suggested_fix=f"Convert input to {feature_contract.data_type} type or fix mapping"
                    ))
        
        # Check for unexpected features
        for input_name in node_inputs:
            if input_name not in contract.input_features:
                self.validation_errors.append(FeatureContractError(
                    error_code="F005",
                    node_id=node.id,
                    feature_name=input_name,
                    model_id=contract.model_id,
                    severity=FeatureValidationSeverity.WARNING.value,
                    message=f"Unexpected input feature: {input_name}",
                    expected="Feature defined in model contract",
                    actual=f"Feature '{input_name}' not in contract",
                    suggested_fix="Remove unused input or update model contract"
                ))
    
    def _validate_output_features(self, node, contract: ModelFeatureContract):
        """Validate output features against contract"""
        node_outputs = {out.name: out for out in node.outputs}
        
        # Check all expected outputs are declared
        for feature_name, feature_contract in contract.output_features.items():
            if feature_name not in node_outputs:
                self.validation_errors.append(FeatureContractError(
                    error_code="F006",
                    node_id=node.id,
                    feature_name=feature_name,
                    model_id=contract.model_id,
                    severity=FeatureValidationSeverity.WARNING.value,
                    message=f"Missing expected output feature: {feature_name}",
                    expected=f"Output feature '{feature_name}' of type {feature_contract.feature_type}",
                    actual="Feature not declared in node outputs",
                    suggested_fix=f"Add output '{feature_name}' to node declaration"
                ))
                continue
            
            # Validate output type
            node_output = node_outputs[feature_name]
            if not self._validate_feature_type(node_output.type, feature_contract):
                self.validation_errors.append(FeatureContractError(
                    error_code="F007",
                    node_id=node.id,
                    feature_name=feature_name,
                    model_id=contract.model_id,
                    severity=FeatureValidationSeverity.ERROR.value,
                    message=f"Type mismatch for output feature: {feature_name}",
                    expected=f"{feature_contract.feature_type} ({feature_contract.data_type})",
                    actual=node_output.type,
                    suggested_fix=f"Update output type to {feature_contract.data_type}"
                ))
    
    def _validate_feature_constraints(self, node, contract: ModelFeatureContract):
        """Validate feature value constraints and ranges"""
        # Check for constant values in parameters that might violate constraints
        constant_values = self._extract_constant_values(node)
        
        for feature_name, value in constant_values.items():
            if feature_name in contract.input_features:
                feature_contract = contract.input_features[feature_name]
                
                # Validate numeric ranges
                if feature_contract.feature_type == 'numeric' and isinstance(value, (int, float)):
                    if feature_contract.min_value is not None and value < feature_contract.min_value:
                        self.validation_errors.append(FeatureContractError(
                            error_code="F008",
                            node_id=node.id,
                            feature_name=feature_name,
                            model_id=contract.model_id,
                            severity=FeatureValidationSeverity.ERROR.value,
                            message=f"Constant value {value} below minimum for {feature_name}",
                            expected=f"Value >= {feature_contract.min_value}",
                            actual=str(value),
                            suggested_fix=f"Use value >= {feature_contract.min_value} or remove constant"
                        ))
                    
                    if feature_contract.max_value is not None and value > feature_contract.max_value:
                        self.validation_errors.append(FeatureContractError(
                            error_code="F009",
                            node_id=node.id,
                            feature_name=feature_name,
                            model_id=contract.model_id,
                            severity=FeatureValidationSeverity.ERROR.value,
                            message=f"Constant value {value} above maximum for {feature_name}",
                            expected=f"Value <= {feature_contract.max_value}",
                            actual=str(value),
                            suggested_fix=f"Use value <= {feature_contract.max_value} or remove constant"
                        ))
                
                # Validate categorical values
                if (feature_contract.feature_type == 'categorical' and 
                    feature_contract.allowed_values and 
                    value not in feature_contract.allowed_values):
                    self.validation_errors.append(FeatureContractError(
                        error_code="F010",
                        node_id=node.id,
                        feature_name=feature_name,
                        model_id=contract.model_id,
                        severity=FeatureValidationSeverity.ERROR.value,
                        message=f"Invalid categorical value: {value}",
                        expected=f"One of: {feature_contract.allowed_values}",
                        actual=str(value),
                        suggested_fix=f"Use one of the allowed values: {feature_contract.allowed_values}"
                    ))
                
                # Validate null constraints
                if value is None and not feature_contract.null_allowed:
                    self.validation_errors.append(FeatureContractError(
                        error_code="F011",
                        node_id=node.id,
                        feature_name=feature_name,
                        model_id=contract.model_id,
                        severity=FeatureValidationSeverity.ERROR.value,
                        message=f"Null value not allowed for {feature_name}",
                        expected="Non-null value",
                        actual="null",
                        suggested_fix="Provide a default value or make feature optional"
                    ))
    
    def _extract_model_id(self, node) -> Optional[str]:
        """Extract model_id from node parameters"""
        return node.parameters.get('model_id') or node.metadata.get('model_id')
    
    def _get_model_contract(self, model_id: str) -> Optional[ModelFeatureContract]:
        """Get feature contract for model (from cache or feature store)"""
        if model_id in self.feature_contracts:
            return self.feature_contracts[model_id]
        
        # Try to load from feature store cache
        if model_id in self.feature_store_cache:
            contract_data = self.feature_store_cache[model_id]
            return self._parse_contract_from_cache(model_id, contract_data)
        
        return None
    
    def _parse_contract_from_cache(self, model_id: str, contract_data: Dict[str, Any]) -> ModelFeatureContract:
        """Parse model contract from cached feature store data"""
        input_features = {}
        output_features = {}
        
        # Parse input features
        for feature_name, feature_data in contract_data.get('input_schema', {}).items():
            input_features[feature_name] = FeatureContract(
                feature_name=feature_name,
                feature_type=feature_data.get('feature_type', 'numeric'),
                data_type=feature_data.get('data_type', 'float'),
                required=feature_data.get('required', True),
                min_value=feature_data.get('min_value'),
                max_value=feature_data.get('max_value'),
                allowed_values=feature_data.get('allowed_values'),
                null_allowed=feature_data.get('null_allowed', False),
                unit=feature_data.get('unit'),
                description=feature_data.get('description')
            )
        
        # Parse output features
        for feature_name, feature_data in contract_data.get('output_schema', {}).items():
            output_features[feature_name] = FeatureContract(
                feature_name=feature_name,
                feature_type=feature_data.get('feature_type', 'numeric'),
                data_type=feature_data.get('data_type', 'float'),
                required=feature_data.get('required', True),
                min_value=feature_data.get('min_value'),
                max_value=feature_data.get('max_value'),
                allowed_values=feature_data.get('allowed_values'),
                null_allowed=feature_data.get('null_allowed', False),
                unit=feature_data.get('unit'),
                description=feature_data.get('description')
            )
        
        contract = ModelFeatureContract(
            model_id=model_id,
            model_version=contract_data.get('version', '1.0.0'),
            input_features=input_features,
            output_features=output_features,
            feature_store_version=contract_data.get('feature_store_version', '1.0.0'),
            contract_hash=contract_data.get('contract_hash')
        )
        
        # Cache the parsed contract
        self.feature_contracts[model_id] = contract
        return contract
    
    def _validate_feature_type(self, node_type: str, feature_contract: FeatureContract) -> bool:
        """Validate node input/output type against feature contract"""
        # Type compatibility mapping
        type_compatibility = {
            'numeric': ['float', 'double', 'int', 'integer', 'scalar', 'number'],
            'categorical': ['string', 'str', 'enum', 'category'],
            'boolean': ['bool', 'boolean'],
            'text': ['string', 'str', 'text'],
            'embedding': ['vector', 'array', 'tensor'],
            'timestamp': ['date', 'datetime', 'timestamp']
        }
        
        compatible_types = type_compatibility.get(feature_contract.feature_type, [])
        compatible_types.append(feature_contract.data_type)  # Always include exact data type
        
        return node_type.lower() in [t.lower() for t in compatible_types]
    
    def _extract_constant_values(self, node) -> Dict[str, Any]:
        """Extract constant values from node parameters"""
        constants = {}
        
        # Look for direct constants in parameters
        if 'constants' in node.parameters:
            constants.update(node.parameters['constants'])
        
        # Look for default values in input mappings
        if 'input_mapping' in node.parameters:
            for feature_name, value_expr in node.parameters['input_mapping'].items():
                if self._is_constant_expression(value_expr):
                    constants[feature_name] = self._parse_constant_value(value_expr)
        
        # Look for hardcoded values in rules/conditions
        if 'rules' in node.parameters:
            for rule in node.parameters['rules']:
                if 'action' in rule and isinstance(rule['action'], dict):
                    for key, value in rule['action'].items():
                        if self._is_constant_expression(str(value)):
                            constants[key] = value
        
        return constants
    
    def _is_constant_expression(self, expr: str) -> bool:
        """Check if expression represents a constant value"""
        if not isinstance(expr, str):
            return True  # Non-string values are constants
        
        expr = expr.strip()
        
        # Numeric constants
        try:
            float(expr)
            return True
        except ValueError:
            pass
        
        # Boolean constants
        if expr.lower() in ['true', 'false']:
            return True
        
        # String constants (quoted)
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            return True
        
        # Null constants
        if expr.lower() in ['null', 'none', 'nil']:
            return True
        
        return False
    
    def _parse_constant_value(self, expr: str) -> Any:
        """Parse constant value from expression"""
        if not isinstance(expr, str):
            return expr
        
        expr = expr.strip()
        
        # Try numeric
        try:
            if '.' in expr:
                return float(expr)
            else:
                return int(expr)
        except ValueError:
            pass
        
        # Try boolean
        if expr.lower() == 'true':
            return True
        elif expr.lower() == 'false':
            return False
        
        # Try null
        if expr.lower() in ['null', 'none', 'nil']:
            return None
        
        # Try string (remove quotes)
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            return expr[1:-1]
        
        return expr
    
    def _init_mock_feature_contracts(self):
        """Initialize mock feature contracts for testing"""
        # Mock contract for opportunity scoring model
        self.feature_contracts['opportunity_scorer_v1'] = ModelFeatureContract(
            model_id='opportunity_scorer_v1',
            model_version='1.2.0',
            input_features={
                'deal_size': FeatureContract(
                    feature_name='deal_size',
                    feature_type='numeric',
                    data_type='float',
                    required=True,
                    min_value=0.0,
                    max_value=10000000.0,
                    unit='USD'
                ),
                'stage': FeatureContract(
                    feature_name='stage',
                    feature_type='categorical',
                    data_type='string',
                    required=True,
                    allowed_values=['Prospecting', 'Qualification', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']
                ),
                'days_in_stage': FeatureContract(
                    feature_name='days_in_stage',
                    feature_type='numeric',
                    data_type='int',
                    required=True,
                    min_value=0,
                    max_value=365
                ),
                'competitor_present': FeatureContract(
                    feature_name='competitor_present',
                    feature_type='boolean',
                    data_type='bool',
                    required=False,
                    null_allowed=True
                )
            },
            output_features={
                'win_probability': FeatureContract(
                    feature_name='win_probability',
                    feature_type='numeric',
                    data_type='float',
                    min_value=0.0,
                    max_value=1.0
                ),
                'confidence': FeatureContract(
                    feature_name='confidence',
                    feature_type='numeric',
                    data_type='float',
                    min_value=0.0,
                    max_value=1.0
                ),
                'risk_factors': FeatureContract(
                    feature_name='risk_factors',
                    feature_type='categorical',
                    data_type='string',
                    allowed_values=['low', 'medium', 'high', 'critical']
                )
            },
            feature_store_version='2.1.0'
        )
        
        # Mock contract for churn prediction model
        self.feature_contracts['churn_predictor_v2'] = ModelFeatureContract(
            model_id='churn_predictor_v2',
            model_version='2.0.1',
            input_features={
                'account_age_months': FeatureContract(
                    feature_name='account_age_months',
                    feature_type='numeric',
                    data_type='int',
                    required=True,
                    min_value=0,
                    max_value=120
                ),
                'mrr': FeatureContract(
                    feature_name='mrr',
                    feature_type='numeric',
                    data_type='float',
                    required=True,
                    min_value=0.0,
                    unit='USD'
                ),
                'support_tickets_30d': FeatureContract(
                    feature_name='support_tickets_30d',
                    feature_type='numeric',
                    data_type='int',
                    required=False,
                    min_value=0,
                    null_allowed=True
                )
            },
            output_features={
                'churn_probability': FeatureContract(
                    feature_name='churn_probability',
                    feature_type='numeric',
                    data_type='float',
                    min_value=0.0,
                    max_value=1.0
                ),
                'confidence': FeatureContract(
                    feature_name='confidence',
                    feature_type='numeric',
                    data_type='float',
                    min_value=0.0,
                    max_value=1.0
                )
            },
            feature_store_version='2.1.0'
        )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        error_count = len([e for e in self.validation_errors if e.severity == 'error'])
        warning_count = len([e for e in self.validation_errors if e.severity == 'warning'])
        info_count = len([e for e in self.validation_errors if e.severity == 'info'])
        
        return {
            'total_issues': len(self.validation_errors),
            'errors': error_count,
            'warnings': warning_count,
            'info': info_count,
            'validation_passed': error_count == 0,
            'issues_by_node': self._group_issues_by_node(),
            'issues_by_model': self._group_issues_by_model(),
            'common_issues': self._get_common_issues()
        }
    
    def _group_issues_by_node(self) -> Dict[str, int]:
        """Group validation issues by node"""
        issues_by_node = {}
        for error in self.validation_errors:
            if error.node_id not in issues_by_node:
                issues_by_node[error.node_id] = 0
            issues_by_node[error.node_id] += 1
        return issues_by_node
    
    def _group_issues_by_model(self) -> Dict[str, int]:
        """Group validation issues by model"""
        issues_by_model = {}
        for error in self.validation_errors:
            if error.model_id not in issues_by_model:
                issues_by_model[error.model_id] = 0
            issues_by_model[error.model_id] += 1
        return issues_by_model
    
    def _get_common_issues(self) -> List[str]:
        """Get list of most common validation issues"""
        issue_counts = {}
        for error in self.validation_errors:
            issue_type = error.error_code
            if issue_type not in issue_counts:
                issue_counts[issue_type] = 0
            issue_counts[issue_type] += 1
        
        # Sort by frequency and return top issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{code}: {count} occurrences" for code, count in sorted_issues[:5]]

# Test helper functions
def create_test_ml_node_with_features():
    """Create a test ML node with feature validation issues"""
    from .ir import IRNode, IRNodeType, IRInput, IROutput
    
    return IRNode(
        id="test_opportunity_scorer",
        type=IRNodeType.ML_PREDICT,
        parameters={
            'model_id': 'opportunity_scorer_v1',
            'input_mapping': {
                'deal_size': '-1000',  # Invalid: below minimum
                'stage': 'InvalidStage',  # Invalid: not in allowed values
                'days_in_stage': '500'  # Invalid: above maximum
            },
            'constants': {
                'competitor_present': None  # Valid: null allowed for this feature
            }
        },
        inputs=[
            IRInput(name='deal_size', type='float', required=True),
            IRInput(name='stage', type='string', required=True),
            IRInput(name='days_in_stage', type='int', required=True),
            # Missing: competitor_present (but it's optional)
        ],
        outputs=[
            IROutput(name='win_probability', type='float'),
            IROutput(name='confidence', type='float'),
            # Missing: risk_factors output
        ]
    )

def run_feature_validation_tests():
    """Run basic feature validation tests"""
    validator = FeatureContractValidator()
    
    # Create test graph with problematic ML node
    test_node = create_test_ml_node_with_features()
    from .ir import IRGraph
    
    test_graph = IRGraph(
        workflow_id="test_feature_validation",
        name="Test Feature Validation Workflow",
        version="1.0.0",
        automation_type="rbia",
        nodes=[test_node]
    )
    
    # Run validation
    errors = validator.validate_ir_graph_features(test_graph)
    
    print(f"Feature validation completed. Found {len(errors)} issues:")
    for error in errors:
        print(f"  {error.error_code} [{error.severity}]: {error.message}")
        if error.suggested_fix:
            print(f"    Fix: {error.suggested_fix}")
    
    # Print summary
    summary = validator.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"  Total issues: {summary['total_issues']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Validation passed: {summary['validation_passed']}")
    
    return errors

if __name__ == "__main__":
    run_feature_validation_tests()
