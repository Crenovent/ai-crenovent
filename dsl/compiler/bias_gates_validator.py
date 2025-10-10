"""
Bias/Drift Gates Validator - Task 6.2.20
=========================================

Bias/drift gates declared? else fail
- Lint/analyzer rule that checks for bias monitoring config and drift detection hooks
- Fails compile in non-dev tenants; soft-fail in sandbox with warning
- Integrates with model registry metadata to know which models require specific gate types

Dependencies: Task 6.2.4 (IR), drift_bias_monitor.py, model registry
Outputs: Bias/drift gate validation → ensures ML governance compliance
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class BiasGateError:
    """Bias gate validation error"""
    error_code: str
    node_id: str
    model_id: Optional[str]
    gate_type: str  # "bias", "drift", "fairness"
    severity: str  # "error", "warning", "info"
    message: str
    expected: str
    actual: str
    tenant_context: str  # "production", "staging", "sandbox"
    autofix_available: bool = False
    suggested_config: Optional[Dict[str, Any]] = None
    source_location: Optional[Dict[str, Any]] = None

@dataclass
class BiasGateRequirement:
    """Bias/drift gate requirement for a model"""
    model_id: str
    required_bias_gates: List[str]  # ["demographic_parity", "equalized_odds", etc.]
    required_drift_gates: List[str]  # ["data_drift", "concept_drift", "prediction_drift"]
    fairness_metrics: List[str]  # Required fairness metrics
    monitoring_frequency: str  # "real_time", "daily", "weekly"
    compliance_level: str  # "strict", "moderate", "basic"

class GateValidationSeverity(Enum):
    """Severity levels for gate validation"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class TenantContext(Enum):
    """Tenant deployment context"""
    PRODUCTION = "production"
    STAGING = "staging"
    SANDBOX = "sandbox"
    DEVELOPMENT = "development"

# Task 6.2.20: Bias/Drift Gates Validator
class BiasGatesValidator:
    """Validates bias and drift monitoring gates for ML nodes"""
    
    def __init__(self, tenant_context: TenantContext = TenantContext.PRODUCTION):
        self.gate_errors: List[BiasGateError] = []
        self.tenant_context = tenant_context
        self.model_requirements: Dict[str, BiasGateRequirement] = {}
        
        # Initialize model requirements (would come from model registry)
        self._init_model_requirements()
        
        # Default gate configurations
        self.default_bias_gates = {
            'demographic_parity': {'threshold': 0.8, 'protected_attributes': ['gender', 'race', 'age']},
            'equalized_odds': {'threshold': 0.1, 'metrics': ['tpr_diff', 'fpr_diff']},
            'disparate_impact': {'threshold': 0.8, 'reference_group': 'majority'}
        }
        
        self.default_drift_gates = {
            'data_drift': {'threshold': 0.1, 'method': 'ks_test', 'window_size': 1000},
            'concept_drift': {'threshold': 0.05, 'method': 'performance_degradation'},
            'prediction_drift': {'threshold': 0.15, 'method': 'distribution_shift'}
        }
    
    def validate_bias_drift_gates(self, ir_graph) -> List[BiasGateError]:
        """Validate bias and drift gates for all ML nodes"""
        from .ir import IRGraph, IRNodeType
        
        self.gate_errors.clear()
        
        # Get all ML nodes
        ml_nodes = ir_graph.get_ml_nodes()
        
        for node in ml_nodes:
            self._validate_node_gates(node)
        
        return self.gate_errors
    
    def _validate_node_gates(self, node):
        """Validate gates for a single ML node"""
        model_id = self._extract_model_id(node)
        
        # Get requirements for this model
        requirements = self._get_model_requirements(model_id)
        
        # Validate bias gates
        self._validate_bias_gates(node, model_id, requirements)
        
        # Validate drift gates
        self._validate_drift_gates(node, model_id, requirements)
        
        # Validate fairness metrics
        self._validate_fairness_metrics(node, model_id, requirements)
    
    def _validate_bias_gates(self, node, model_id: Optional[str], requirements: Optional[BiasGateRequirement]):
        """Validate bias monitoring gates"""
        # Check for bias monitoring configuration
        bias_config = self._extract_bias_config(node)
        
        if not bias_config:
            # No bias config found - check if required
            if self._is_bias_monitoring_required(node, model_id, requirements):
                severity = self._get_severity_for_missing_gate("bias")
                
                self.gate_errors.append(BiasGateError(
                    error_code="B001",
                    node_id=node.id,
                    model_id=model_id,
                    gate_type="bias",
                    severity=severity.value,
                    message="Missing bias monitoring configuration",
                    expected="Bias monitoring gates with fairness metrics",
                    actual="No bias configuration found",
                    tenant_context=self.tenant_context.value,
                    autofix_available=True,
                    suggested_config=self._generate_default_bias_config(node, requirements)
                ))
            return
        
        # Validate bias configuration completeness
        self._validate_bias_config_completeness(node, model_id, bias_config, requirements)
        
        # Validate bias thresholds
        self._validate_bias_thresholds(node, model_id, bias_config)
    
    def _validate_drift_gates(self, node, model_id: Optional[str], requirements: Optional[BiasGateRequirement]):
        """Validate drift monitoring gates"""
        # Check for drift monitoring configuration
        drift_config = self._extract_drift_config(node)
        
        if not drift_config:
            # No drift config found - check if required
            if self._is_drift_monitoring_required(node, model_id, requirements):
                severity = self._get_severity_for_missing_gate("drift")
                
                self.gate_errors.append(BiasGateError(
                    error_code="D001",
                    node_id=node.id,
                    model_id=model_id,
                    gate_type="drift",
                    severity=severity.value,
                    message="Missing drift monitoring configuration",
                    expected="Drift monitoring gates with detection methods",
                    actual="No drift configuration found",
                    tenant_context=self.tenant_context.value,
                    autofix_available=True,
                    suggested_config=self._generate_default_drift_config(node, requirements)
                ))
            return
        
        # Validate drift configuration completeness
        self._validate_drift_config_completeness(node, model_id, drift_config, requirements)
        
        # Validate drift thresholds
        self._validate_drift_thresholds(node, model_id, drift_config)
    
    def _validate_fairness_metrics(self, node, model_id: Optional[str], requirements: Optional[BiasGateRequirement]):
        """Validate fairness metrics configuration"""
        fairness_config = self._extract_fairness_config(node)
        
        if not fairness_config and requirements and requirements.fairness_metrics:
            severity = self._get_severity_for_missing_gate("fairness")
            
            self.gate_errors.append(BiasGateError(
                error_code="F001",
                node_id=node.id,
                model_id=model_id,
                gate_type="fairness",
                severity=severity.value,
                message="Missing fairness metrics configuration",
                expected=f"Fairness metrics: {requirements.fairness_metrics}",
                actual="No fairness configuration found",
                tenant_context=self.tenant_context.value,
                autofix_available=True,
                suggested_config=self._generate_default_fairness_config(requirements.fairness_metrics)
            ))
    
    def _extract_bias_config(self, node) -> Optional[Dict[str, Any]]:
        """Extract bias monitoring configuration from node"""
        # Check direct bias configuration
        if 'bias_monitoring' in node.parameters:
            return node.parameters['bias_monitoring']
        
        # Check governance policies
        for policy in node.policies:
            if policy.policy_type == 'bias':
                return policy.parameters
        
        # Check metadata
        if 'bias_config' in node.metadata:
            return node.metadata['bias_config']
        
        return None
    
    def _extract_drift_config(self, node) -> Optional[Dict[str, Any]]:
        """Extract drift monitoring configuration from node"""
        # Check direct drift configuration
        if 'drift_monitoring' in node.parameters:
            return node.parameters['drift_monitoring']
        
        # Check governance policies
        for policy in node.policies:
            if policy.policy_type == 'drift':
                return policy.parameters
        
        # Check metadata
        if 'drift_config' in node.metadata:
            return node.metadata['drift_config']
        
        return None
    
    def _extract_fairness_config(self, node) -> Optional[Dict[str, Any]]:
        """Extract fairness metrics configuration from node"""
        # Check direct fairness configuration
        if 'fairness_metrics' in node.parameters:
            return node.parameters['fairness_metrics']
        
        # Check bias config for fairness metrics
        bias_config = self._extract_bias_config(node)
        if bias_config and 'fairness_metrics' in bias_config:
            return bias_config['fairness_metrics']
        
        return None
    
    def _is_bias_monitoring_required(self, node, model_id: Optional[str], requirements: Optional[BiasGateRequirement]) -> bool:
        """Check if bias monitoring is required for this node/model"""
        # Always required for high-risk models
        if self._is_high_risk_model(node, model_id):
            return True
        
        # Required if model has specific requirements
        if requirements and requirements.required_bias_gates:
            return True
        
        # Required for classification models in production
        from .ir import IRNodeType
        if (node.type == IRNodeType.ML_CLASSIFY and 
            self.tenant_context in [TenantContext.PRODUCTION, TenantContext.STAGING]):
            return True
        
        return False
    
    def _is_drift_monitoring_required(self, node, model_id: Optional[str], requirements: Optional[BiasGateRequirement]) -> bool:
        """Check if drift monitoring is required for this node/model"""
        # Always required for production ML nodes
        if self.tenant_context == TenantContext.PRODUCTION:
            return True
        
        # Required if model has specific requirements
        if requirements and requirements.required_drift_gates:
            return True
        
        # Required for high-risk models in any environment
        if self._is_high_risk_model(node, model_id):
            return True
        
        return False
    
    def _is_high_risk_model(self, node, model_id: Optional[str]) -> bool:
        """Check if model is high-risk and requires strict monitoring"""
        # Check explicit risk annotations
        if node.parameters.get('risk_level') == 'high_risk':
            return True
        
        if node.metadata.get('risk_tier') == 'high_risk':
            return True
        
        # Infer from model ID patterns
        if model_id:
            high_risk_patterns = [
                'fraud', 'credit', 'loan', 'legal', 'compliance', 
                'financial', 'hiring', 'criminal', 'medical'
            ]
            return any(pattern in model_id.lower() for pattern in high_risk_patterns)
        
        return False
    
    def _get_severity_for_missing_gate(self, gate_type: str) -> GateValidationSeverity:
        """Get severity level for missing gate based on tenant context"""
        if self.tenant_context == TenantContext.PRODUCTION:
            return GateValidationSeverity.ERROR  # Fail compilation
        elif self.tenant_context == TenantContext.STAGING:
            return GateValidationSeverity.ERROR  # Fail compilation
        elif self.tenant_context == TenantContext.SANDBOX:
            return GateValidationSeverity.WARNING  # Soft fail
        else:  # DEVELOPMENT
            return GateValidationSeverity.INFO  # Just inform
    
    def _validate_bias_config_completeness(self, node, model_id: Optional[str], 
                                         bias_config: Dict[str, Any], 
                                         requirements: Optional[BiasGateRequirement]):
        """Validate completeness of bias configuration"""
        # Check for required bias gates
        if requirements and requirements.required_bias_gates:
            for required_gate in requirements.required_bias_gates:
                if required_gate not in bias_config:
                    self.gate_errors.append(BiasGateError(
                        error_code="B002",
                        node_id=node.id,
                        model_id=model_id,
                        gate_type="bias",
                        severity=GateValidationSeverity.ERROR.value,
                        message=f"Missing required bias gate: {required_gate}",
                        expected=f"Bias gate configuration for {required_gate}",
                        actual="Gate not configured",
                        tenant_context=self.tenant_context.value,
                        autofix_available=True,
                        suggested_config={required_gate: self.default_bias_gates.get(required_gate, {})}
                    ))
        
        # Check for protected attributes
        if 'protected_attributes' not in bias_config:
            self.gate_errors.append(BiasGateError(
                error_code="B003",
                node_id=node.id,
                model_id=model_id,
                gate_type="bias",
                severity=GateValidationSeverity.WARNING.value,
                message="No protected attributes specified for bias monitoring",
                expected="List of protected attributes (e.g., gender, race, age)",
                actual="No protected attributes configured",
                tenant_context=self.tenant_context.value,
                autofix_available=True,
                suggested_config={'protected_attributes': ['gender', 'race', 'age']}
            ))
    
    def _validate_drift_config_completeness(self, node, model_id: Optional[str], 
                                          drift_config: Dict[str, Any], 
                                          requirements: Optional[BiasGateRequirement]):
        """Validate completeness of drift configuration"""
        # Check for required drift gates
        if requirements and requirements.required_drift_gates:
            for required_gate in requirements.required_drift_gates:
                if required_gate not in drift_config:
                    self.gate_errors.append(BiasGateError(
                        error_code="D002",
                        node_id=node.id,
                        model_id=model_id,
                        gate_type="drift",
                        severity=GateValidationSeverity.ERROR.value,
                        message=f"Missing required drift gate: {required_gate}",
                        expected=f"Drift gate configuration for {required_gate}",
                        actual="Gate not configured",
                        tenant_context=self.tenant_context.value,
                        autofix_available=True,
                        suggested_config={required_gate: self.default_drift_gates.get(required_gate, {})}
                    ))
        
        # Check for monitoring frequency
        if 'monitoring_frequency' not in drift_config:
            self.gate_errors.append(BiasGateError(
                error_code="D003",
                node_id=node.id,
                model_id=model_id,
                gate_type="drift",
                severity=GateValidationSeverity.WARNING.value,
                message="No monitoring frequency specified for drift detection",
                expected="Monitoring frequency (real_time, daily, weekly)",
                actual="No frequency configured",
                tenant_context=self.tenant_context.value,
                autofix_available=True,
                suggested_config={'monitoring_frequency': 'daily'}
            ))
    
    def _validate_bias_thresholds(self, node, model_id: Optional[str], bias_config: Dict[str, Any]):
        """Validate bias threshold values"""
        for gate_name, gate_config in bias_config.items():
            if isinstance(gate_config, dict) and 'threshold' in gate_config:
                threshold = gate_config['threshold']
                
                # Validate threshold range
                if not (0.0 <= threshold <= 1.0):
                    self.gate_errors.append(BiasGateError(
                        error_code="B004",
                        node_id=node.id,
                        model_id=model_id,
                        gate_type="bias",
                        severity=GateValidationSeverity.ERROR.value,
                        message=f"Invalid bias threshold for {gate_name}: {threshold}",
                        expected="Threshold between 0.0 and 1.0",
                        actual=str(threshold),
                        tenant_context=self.tenant_context.value,
                        autofix_available=True,
                        suggested_config={gate_name: {'threshold': max(0.0, min(1.0, threshold))}}
                    ))
    
    def _validate_drift_thresholds(self, node, model_id: Optional[str], drift_config: Dict[str, Any]):
        """Validate drift threshold values"""
        for gate_name, gate_config in drift_config.items():
            if isinstance(gate_config, dict) and 'threshold' in gate_config:
                threshold = gate_config['threshold']
                
                # Validate threshold range (typically 0.0 to 1.0 for most drift metrics)
                if not (0.0 <= threshold <= 1.0):
                    self.gate_errors.append(BiasGateError(
                        error_code="D004",
                        node_id=node.id,
                        model_id=model_id,
                        gate_type="drift",
                        severity=GateValidationSeverity.ERROR.value,
                        message=f"Invalid drift threshold for {gate_name}: {threshold}",
                        expected="Threshold between 0.0 and 1.0",
                        actual=str(threshold),
                        tenant_context=self.tenant_context.value,
                        autofix_available=True,
                        suggested_config={gate_name: {'threshold': max(0.0, min(1.0, threshold))}}
                    ))
    
    def _generate_default_bias_config(self, node, requirements: Optional[BiasGateRequirement]) -> Dict[str, Any]:
        """Generate default bias monitoring configuration"""
        config = {
            'enabled': True,
            'protected_attributes': ['gender', 'race', 'age'],
            'fairness_metrics': ['demographic_parity', 'equalized_odds']
        }
        
        if requirements and requirements.required_bias_gates:
            for gate in requirements.required_bias_gates:
                config[gate] = self.default_bias_gates.get(gate, {'threshold': 0.8})
        else:
            # Add default gates
            config.update(self.default_bias_gates)
        
        return config
    
    def _generate_default_drift_config(self, node, requirements: Optional[BiasGateRequirement]) -> Dict[str, Any]:
        """Generate default drift monitoring configuration"""
        config = {
            'enabled': True,
            'monitoring_frequency': 'daily',
            'alert_on_drift': True
        }
        
        if requirements and requirements.required_drift_gates:
            for gate in requirements.required_drift_gates:
                config[gate] = self.default_drift_gates.get(gate, {'threshold': 0.1})
        else:
            # Add default gates
            config.update(self.default_drift_gates)
        
        return config
    
    def _generate_default_fairness_config(self, required_metrics: List[str]) -> Dict[str, Any]:
        """Generate default fairness metrics configuration"""
        return {
            'enabled': True,
            'metrics': required_metrics,
            'reporting_frequency': 'weekly',
            'alert_thresholds': {metric: 0.8 for metric in required_metrics}
        }
    
    def _extract_model_id(self, node) -> Optional[str]:
        """Extract model_id from node"""
        return node.parameters.get('model_id') or node.metadata.get('model_id')
    
    def _get_model_requirements(self, model_id: Optional[str]) -> Optional[BiasGateRequirement]:
        """Get bias/drift requirements for model from registry"""
        if model_id and model_id in self.model_requirements:
            return self.model_requirements[model_id]
        return None
    
    def _init_model_requirements(self):
        """Initialize model requirements (would come from model registry)"""
        # High-risk fraud detection model
        self.model_requirements['fraud_detector_v1'] = BiasGateRequirement(
            model_id='fraud_detector_v1',
            required_bias_gates=['demographic_parity', 'equalized_odds', 'disparate_impact'],
            required_drift_gates=['data_drift', 'concept_drift', 'prediction_drift'],
            fairness_metrics=['demographic_parity', 'equalized_odds'],
            monitoring_frequency='real_time',
            compliance_level='strict'
        )
        
        # Credit scoring model
        self.model_requirements['credit_scorer_v2'] = BiasGateRequirement(
            model_id='credit_scorer_v2',
            required_bias_gates=['demographic_parity', 'equalized_odds'],
            required_drift_gates=['data_drift', 'concept_drift'],
            fairness_metrics=['demographic_parity'],
            monitoring_frequency='daily',
            compliance_level='strict'
        )
        
        # Opportunity scoring (medium risk)
        self.model_requirements['opportunity_scorer_v1'] = BiasGateRequirement(
            model_id='opportunity_scorer_v1',
            required_bias_gates=[],  # Not required but recommended
            required_drift_gates=['data_drift'],
            fairness_metrics=[],
            monitoring_frequency='weekly',
            compliance_level='moderate'
        )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of bias/drift gate validation"""
        error_count = len([e for e in self.gate_errors if e.severity == 'error'])
        warning_count = len([e for e in self.gate_errors if e.severity == 'warning'])
        info_count = len([e for e in self.gate_errors if e.severity == 'info'])
        
        return {
            'total_issues': len(self.gate_errors),
            'errors': error_count,
            'warnings': warning_count,
            'info': info_count,
            'compliance_passed': error_count == 0,
            'tenant_context': self.tenant_context.value,
            'issues_by_gate_type': self._group_issues_by_gate_type(),
            'issues_by_node': self._group_issues_by_node(),
            'autofix_available': len([e for e in self.gate_errors if e.autofix_available]),
            'high_risk_violations': self._get_high_risk_violations()
        }
    
    def _group_issues_by_gate_type(self) -> Dict[str, int]:
        """Group issues by gate type"""
        gate_counts = {}
        for error in self.gate_errors:
            gate_type = error.gate_type
            if gate_type not in gate_counts:
                gate_counts[gate_type] = 0
            gate_counts[gate_type] += 1
        return gate_counts
    
    def _group_issues_by_node(self) -> Dict[str, int]:
        """Group issues by node"""
        node_counts = {}
        for error in self.gate_errors:
            node_id = error.node_id
            if node_id not in node_counts:
                node_counts[node_id] = 0
            node_counts[node_id] += 1
        return node_counts
    
    def _get_high_risk_violations(self) -> List[str]:
        """Get high-risk violations that require immediate attention"""
        high_risk = []
        for error in self.gate_errors:
            if (error.severity == 'error' and 
                error.model_id and 
                self._is_high_risk_model_id(error.model_id)):
                high_risk.append(f"{error.node_id}: {error.message}")
        return high_risk
    
    def _is_high_risk_model_id(self, model_id: str) -> bool:
        """Check if model ID indicates high risk"""
        high_risk_patterns = ['fraud', 'credit', 'loan', 'legal', 'compliance', 'financial']
        return any(pattern in model_id.lower() for pattern in high_risk_patterns)

# Test helper functions
def create_test_ml_nodes_for_gate_validation():
    """Create test ML nodes with various gate validation issues"""
    from .ir import IRNode, IRNodeType, IRPolicy
    
    nodes = [
        # Node with proper gates
        IRNode(
            id="proper_gates",
            type=IRNodeType.ML_CLASSIFY,
            parameters={
                'model_id': 'opportunity_scorer_v1',
                'bias_monitoring': {
                    'enabled': True,
                    'protected_attributes': ['gender', 'age'],
                    'demographic_parity': {'threshold': 0.8}
                },
                'drift_monitoring': {
                    'enabled': True,
                    'data_drift': {'threshold': 0.1, 'method': 'ks_test'}
                }
            }
        ),
        
        # High-risk node missing gates
        IRNode(
            id="missing_gates_high_risk",
            type=IRNodeType.ML_PREDICT,
            parameters={
                'model_id': 'fraud_detector_v1',
                'risk_level': 'high_risk'
            }
        ),
        
        # Node with invalid thresholds
        IRNode(
            id="invalid_thresholds",
            type=IRNodeType.ML_SCORE,
            parameters={
                'model_id': 'credit_scorer_v2',
                'bias_monitoring': {
                    'demographic_parity': {'threshold': 1.5}  # Invalid
                },
                'drift_monitoring': {
                    'data_drift': {'threshold': -0.1}  # Invalid
                }
            }
        ),
        
        # Node with partial configuration
        IRNode(
            id="partial_config",
            type=IRNodeType.ML_CLASSIFY,
            parameters={
                'model_id': 'churn_predictor_v2',
                'bias_monitoring': {'enabled': True}  # Missing specifics
            }
        )
    ]
    
    return nodes

def run_bias_gate_validation_tests():
    """Run bias/drift gate validation tests"""
    # Test in production context (strict)
    validator_prod = BiasGatesValidator(TenantContext.PRODUCTION)
    
    # Create test graph
    test_nodes = create_test_ml_nodes_for_gate_validation()
    from .ir import IRGraph
    
    test_graph = IRGraph(
        workflow_id="test_gate_validation",
        name="Test Gate Validation Workflow",
        version="1.0.0",
        automation_type="rbia",
        nodes=test_nodes
    )
    
    # Run validation
    errors = validator_prod.validate_bias_drift_gates(test_graph)
    
    print(f"Bias/Drift gate validation completed. Found {len(errors)} issues:")
    for error in errors:
        print(f"  {error.error_code} [{error.severity}] {error.node_id}: {error.message}")
        if error.suggested_config:
            print(f"    Suggested config: {error.suggested_config}")
    
    # Print summary
    summary = validator_prod.get_validation_summary()
    print(f"\nValidation Summary (Production):")
    print(f"  Compliance passed: {summary['compliance_passed']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  High-risk violations: {len(summary['high_risk_violations'])}")
    
    # Test in sandbox context (permissive)
    print(f"\n--- Testing in Sandbox Context ---")
    validator_sandbox = BiasGatesValidator(TenantContext.SANDBOX)
    errors_sandbox = validator_sandbox.validate_bias_drift_gates(test_graph)
    
    print(f"Sandbox validation found {len(errors_sandbox)} issues:")
    summary_sandbox = validator_sandbox.get_validation_summary()
    print(f"  Compliance passed: {summary_sandbox['compliance_passed']}")
    print(f"  Errors: {summary_sandbox['errors']}")
    print(f"  Warnings: {summary_sandbox['warnings']}")
    
    return errors, summary

if __name__ == "__main__":
    run_bias_gate_validation_tests()
