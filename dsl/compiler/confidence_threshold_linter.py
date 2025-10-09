"""
Policy Lint: Confidence Threshold Validator - Task 6.2.19
==========================================================

Policy lint: confidence threshold present for all ML nodes
- Linter rule that walks IR and verifies confidence_threshold exists for ML nodes
- Checks policy pack defaults if node-level threshold is absent
- Fails compile or produces autofix suggestions if still absent
- Autofix CLI mode that can inject conservative default threshold with metadata

Dependencies: Task 6.2.4 (IR), Task 6.2.7 (Policy Binder)
Outputs: Confidence threshold validation → governance enforcement
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceLintError:
    """Confidence threshold lint error"""
    error_code: str
    node_id: str
    model_id: Optional[str]
    severity: str  # "error", "warning", "info"
    message: str
    expected: str
    actual: str
    autofix_available: bool = False
    suggested_threshold: Optional[float] = None
    justification_required: bool = False
    source_location: Optional[Dict[str, Any]] = None

@dataclass
class ConfidenceAutofix:
    """Autofix suggestion for confidence threshold"""
    node_id: str
    suggested_threshold: float
    confidence_level: str  # "conservative", "moderate", "aggressive"
    justification: str
    metadata: Dict[str, Any]

class ConfidenceLintSeverity(Enum):
    """Severity levels for confidence lint"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ConfidenceLintMode(Enum):
    """Lint modes for confidence validation"""
    STRICT = "strict"  # Fail compilation if missing
    WARN = "warn"      # Warn but continue
    AUTOFIX = "autofix"  # Suggest fixes

# Task 6.2.19: Confidence Threshold Policy Linter
class ConfidenceThresholdLinter:
    """Linter rule for ML node confidence threshold governance"""
    
    def __init__(self, lint_mode: ConfidenceLintMode = ConfidenceLintMode.STRICT):
        self.lint_errors: List[ConfidenceLintError] = []
        self.autofix_suggestions: List[ConfidenceAutofix] = []
        self.lint_mode = lint_mode
        
        # Default confidence thresholds by model type/risk level
        self.default_thresholds = {
            'high_risk': 0.85,      # Financial, legal decisions
            'medium_risk': 0.75,    # Business decisions
            'low_risk': 0.65,       # Informational, recommendations
            'conservative': 0.80,   # Safe default
            'moderate': 0.70,       # Balanced default
            'aggressive': 0.60      # Permissive default
        }
    
    def lint_confidence_thresholds(self, ir_graph) -> Tuple[List[ConfidenceLintError], List[ConfidenceAutofix]]:
        """Lint all ML nodes for confidence threshold compliance"""
        from .ir import IRGraph, IRNodeType
        
        self.lint_errors.clear()
        self.autofix_suggestions.clear()
        
        # Get all ML nodes
        ml_nodes = ir_graph.get_ml_nodes()
        
        # Get policy pack for default thresholds
        policy_pack_defaults = self._extract_policy_pack_defaults(ir_graph)
        
        for node in ml_nodes:
            self._lint_node_confidence_threshold(node, policy_pack_defaults)
        
        return self.lint_errors, self.autofix_suggestions
    
    def _lint_node_confidence_threshold(self, node, policy_defaults: Dict[str, Any]):
        """Lint confidence threshold for a single ML node"""
        model_id = self._extract_model_id(node)
        
        # Check for node-level confidence threshold
        node_threshold = self._extract_node_confidence_threshold(node)
        
        if node_threshold is not None:
            # Validate threshold value
            self._validate_threshold_value(node, model_id, node_threshold)
            return
        
        # Check for policy pack default
        policy_threshold = self._get_policy_pack_threshold(node, policy_defaults)
        
        if policy_threshold is not None:
            # Policy provides threshold - validate it
            self._validate_threshold_value(node, model_id, policy_threshold, from_policy=True)
            return
        
        # No threshold found - this is a governance violation
        self._report_missing_threshold(node, model_id, policy_defaults)
    
    def _extract_node_confidence_threshold(self, node) -> Optional[float]:
        """Extract confidence threshold from node configuration"""
        # Check direct threshold parameter
        if 'confidence_threshold' in node.parameters:
            return float(node.parameters['confidence_threshold'])
        
        # Check confidence block
        if 'confidence' in node.parameters:
            confidence_config = node.parameters['confidence']
            if isinstance(confidence_config, dict):
                if 'min_confidence' in confidence_config:
                    return float(confidence_config['min_confidence'])
                if 'threshold' in confidence_config:
                    return float(confidence_config['threshold'])
        
        # Check trust budget
        if node.trust_budget:
            return node.trust_budget.min_trust
        
        # Check ML-specific threshold fields
        ml_threshold_fields = ['score_threshold', 'prediction_threshold', 'classification_threshold']
        for field in ml_threshold_fields:
            if field in node.parameters:
                return float(node.parameters[field])
        
        return None
    
    def _get_policy_pack_threshold(self, node, policy_defaults: Dict[str, Any]) -> Optional[float]:
        """Get confidence threshold from policy pack"""
        # Check node-specific policies
        for policy in node.policies:
            if policy.policy_type == 'threshold':
                params = policy.parameters
                if 'confidence_min' in params:
                    return float(params['confidence_min'])
                if 'confidence_threshold' in params:
                    return float(params['confidence_threshold'])
        
        # Check global policy defaults
        if 'confidence_threshold' in policy_defaults:
            return float(policy_defaults['confidence_threshold'])
        
        if 'thresholds' in policy_defaults:
            threshold_config = policy_defaults['thresholds']
            if isinstance(threshold_config, dict):
                if 'confidence_min' in threshold_config:
                    return float(threshold_config['confidence_min'])
                if 'default_confidence' in threshold_config:
                    return float(threshold_config['default_confidence'])
        
        return None
    
    def _validate_threshold_value(self, node, model_id: Optional[str], threshold: float, from_policy: bool = False):
        """Validate that threshold value is reasonable"""
        source = "policy pack" if from_policy else "node configuration"
        
        # Check threshold is in valid range
        if threshold < 0.0 or threshold > 1.0:
            self.lint_errors.append(ConfidenceLintError(
                error_code="C001",
                node_id=node.id,
                model_id=model_id,
                severity=ConfidenceLintSeverity.ERROR.value,
                message=f"Invalid confidence threshold: {threshold}",
                expected="Value between 0.0 and 1.0",
                actual=str(threshold),
                autofix_available=True,
                suggested_threshold=max(0.0, min(1.0, threshold))
            ))
        
        # Check threshold is not too permissive for high-risk models
        risk_level = self._assess_model_risk_level(node, model_id)
        min_threshold_for_risk = {
            'high_risk': 0.8,
            'medium_risk': 0.65,
            'low_risk': 0.5
        }
        
        min_required = min_threshold_for_risk.get(risk_level, 0.5)
        if threshold < min_required:
            severity = ConfidenceLintSeverity.ERROR if risk_level == 'high_risk' else ConfidenceLintSeverity.WARNING
            self.lint_errors.append(ConfidenceLintError(
                error_code="C002",
                node_id=node.id,
                model_id=model_id,
                severity=severity.value,
                message=f"Confidence threshold {threshold} too low for {risk_level} model",
                expected=f"Threshold >= {min_required} for {risk_level} models",
                actual=str(threshold),
                autofix_available=True,
                suggested_threshold=min_required,
                justification_required=True
            ))
        
        # Check threshold is not too restrictive (might block legitimate predictions)
        if threshold > 0.95:
            self.lint_errors.append(ConfidenceLintError(
                error_code="C003",
                node_id=node.id,
                model_id=model_id,
                severity=ConfidenceLintSeverity.WARNING.value,
                message=f"Confidence threshold {threshold} very restrictive",
                expected="Threshold <= 0.95 for practical use",
                actual=str(threshold),
                autofix_available=True,
                suggested_threshold=0.85
            ))
    
    def _report_missing_threshold(self, node, model_id: Optional[str], policy_defaults: Dict[str, Any]):
        """Report missing confidence threshold as governance violation"""
        risk_level = self._assess_model_risk_level(node, model_id)
        suggested_threshold = self.default_thresholds.get(risk_level, self.default_thresholds['conservative'])
        
        # Determine severity based on lint mode
        if self.lint_mode == ConfidenceLintMode.STRICT:
            severity = ConfidenceLintSeverity.ERROR
        elif self.lint_mode == ConfidenceLintMode.WARN:
            severity = ConfidenceLintSeverity.WARNING
        else:  # AUTOFIX
            severity = ConfidenceLintSeverity.INFO
        
        self.lint_errors.append(ConfidenceLintError(
            error_code="C004",
            node_id=node.id,
            model_id=model_id,
            severity=severity.value,
            message="ML node missing confidence threshold",
            expected="confidence_threshold parameter or policy pack default",
            actual="No threshold found",
            autofix_available=True,
            suggested_threshold=suggested_threshold,
            justification_required=True
        ))
        
        # Generate autofix suggestion
        if self.lint_mode == ConfidenceLintMode.AUTOFIX:
            self.autofix_suggestions.append(ConfidenceAutofix(
                node_id=node.id,
                suggested_threshold=suggested_threshold,
                confidence_level=risk_level,
                justification=f"Conservative default for {risk_level} model",
                metadata={
                    'model_id': model_id,
                    'risk_assessment': risk_level,
                    'generated_at': datetime.utcnow().isoformat(),
                    'requires_review': True,
                    'policy_pack_checked': bool(policy_defaults)
                }
            ))
    
    def _assess_model_risk_level(self, node, model_id: Optional[str]) -> str:
        """Assess risk level of ML model for threshold recommendations"""
        # Check explicit risk annotations
        if 'risk_level' in node.parameters:
            return node.parameters['risk_level']
        
        if 'risk_tier' in node.metadata:
            return node.metadata['risk_tier']
        
        # Infer from model type or domain
        if model_id:
            # High-risk patterns
            high_risk_patterns = ['fraud', 'credit', 'loan', 'legal', 'compliance', 'financial']
            if any(pattern in model_id.lower() for pattern in high_risk_patterns):
                return 'high_risk'
            
            # Medium-risk patterns
            medium_risk_patterns = ['pricing', 'forecast', 'churn', 'conversion', 'opportunity']
            if any(pattern in model_id.lower() for pattern in medium_risk_patterns):
                return 'medium_risk'
        
        # Check node type for risk assessment
        from .ir import IRNodeType
        if node.type == IRNodeType.ML_CLASSIFY:
            return 'medium_risk'  # Classification often has business impact
        elif node.type == IRNodeType.ML_SCORE:
            return 'medium_risk'  # Scoring affects decisions
        elif node.type == IRNodeType.ML_PREDICT:
            return 'medium_risk'  # Predictions drive actions
        
        return 'low_risk'  # Default to low risk
    
    def _extract_model_id(self, node) -> Optional[str]:
        """Extract model_id from node"""
        return node.parameters.get('model_id') or node.metadata.get('model_id')
    
    def _extract_policy_pack_defaults(self, ir_graph) -> Dict[str, Any]:
        """Extract policy pack defaults from IR graph"""
        defaults = {}
        
        # Check workflow-level policy pack metadata
        if 'policy_pack_metadata' in ir_graph.metadata:
            policy_metadata = ir_graph.metadata['policy_pack_metadata']
            defaults.update(policy_metadata)
        
        # Check governance configuration
        if ir_graph.governance:
            defaults.update(ir_graph.governance)
        
        return defaults
    
    def generate_autofix_cli_commands(self) -> List[str]:
        """Generate CLI commands for applying autofixes"""
        commands = []
        
        for autofix in self.autofix_suggestions:
            cmd = (f"rbia lint --autofix --node {autofix.node_id} "
                  f"--set-confidence-threshold {autofix.suggested_threshold} "
                  f"--justification \"{autofix.justification}\" "
                  f"--risk-level {autofix.confidence_level}")
            
            if autofix.metadata.get('requires_review'):
                cmd += " --require-approval"
            
            commands.append(cmd)
        
        return commands
    
    def get_lint_summary(self) -> Dict[str, Any]:
        """Get summary of confidence threshold lint results"""
        error_count = len([e for e in self.lint_errors if e.severity == 'error'])
        warning_count = len([e for e in self.lint_errors if e.severity == 'warning'])
        info_count = len([e for e in self.lint_errors if e.severity == 'info'])
        
        autofix_count = len(self.autofix_suggestions)
        
        return {
            'total_issues': len(self.lint_errors),
            'errors': error_count,
            'warnings': warning_count,
            'info': info_count,
            'autofix_available': autofix_count,
            'compliance_passed': error_count == 0,
            'issues_by_node': self._group_issues_by_node(),
            'risk_distribution': self._get_risk_distribution(),
            'common_violations': self._get_common_violations(),
            'autofix_commands': self.generate_autofix_cli_commands()
        }
    
    def _group_issues_by_node(self) -> Dict[str, int]:
        """Group lint issues by node"""
        issues_by_node = {}
        for error in self.lint_errors:
            if error.node_id not in issues_by_node:
                issues_by_node[error.node_id] = 0
            issues_by_node[error.node_id] += 1
        return issues_by_node
    
    def _get_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of issues by risk level"""
        risk_counts = {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0}
        
        for autofix in self.autofix_suggestions:
            risk_level = autofix.confidence_level
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1
        
        return risk_counts
    
    def _get_common_violations(self) -> List[str]:
        """Get most common confidence threshold violations"""
        violation_counts = {}
        
        for error in self.lint_errors:
            error_type = error.error_code
            if error_type not in violation_counts:
                violation_counts[error_type] = 0
            violation_counts[error_type] += 1
        
        # Sort by frequency
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        
        violation_descriptions = {
            'C001': 'Invalid threshold range',
            'C002': 'Threshold too low for risk level',
            'C003': 'Threshold too restrictive',
            'C004': 'Missing confidence threshold'
        }
        
        return [f"{violation_descriptions.get(code, code)}: {count} occurrences" 
                for code, count in sorted_violations[:5]]

# CLI Integration Helper
class ConfidenceThresholdAutoFixer:
    """Helper for applying confidence threshold autofixes"""
    
    def __init__(self):
        self.applied_fixes: List[Dict[str, Any]] = []
    
    def apply_autofix(self, ir_graph, autofix: ConfidenceAutofix, 
                     author_justification: str = "") -> bool:
        """Apply autofix to IR graph"""
        node = ir_graph.get_node(autofix.node_id)
        if not node:
            logger.error(f"Node {autofix.node_id} not found for autofix")
            return False
        
        # Apply the threshold
        if 'confidence' not in node.parameters:
            node.parameters['confidence'] = {}
        
        node.parameters['confidence']['min_confidence'] = autofix.suggested_threshold
        
        # Add metadata about the autofix
        if 'autofix_metadata' not in node.metadata:
            node.metadata['autofix_metadata'] = []
        
        fix_metadata = {
            'type': 'confidence_threshold',
            'applied_at': datetime.utcnow().isoformat(),
            'suggested_threshold': autofix.suggested_threshold,
            'risk_level': autofix.confidence_level,
            'system_justification': autofix.justification,
            'author_justification': author_justification,
            'requires_approval': autofix.metadata.get('requires_review', False)
        }
        
        node.metadata['autofix_metadata'].append(fix_metadata)
        self.applied_fixes.append(fix_metadata)
        
        logger.info(f"Applied confidence threshold autofix to node {autofix.node_id}: {autofix.suggested_threshold}")
        return True
    
    def get_applied_fixes_report(self) -> Dict[str, Any]:
        """Get report of all applied fixes"""
        return {
            'total_fixes_applied': len(self.applied_fixes),
            'fixes_requiring_approval': len([f for f in self.applied_fixes if f.get('requires_approval')]),
            'fixes_by_risk_level': self._group_fixes_by_risk(),
            'applied_fixes': self.applied_fixes
        }
    
    def _group_fixes_by_risk(self) -> Dict[str, int]:
        """Group applied fixes by risk level"""
        risk_counts = {}
        for fix in self.applied_fixes:
            risk_level = fix.get('risk_level', 'unknown')
            if risk_level not in risk_counts:
                risk_counts[risk_level] = 0
            risk_counts[risk_level] += 1
        return risk_counts

# Test helper functions
def create_test_ml_nodes_for_confidence_lint():
    """Create test ML nodes with various confidence threshold issues"""
    from .ir import IRNode, IRNodeType
    
    nodes = [
        # Node with valid threshold
        IRNode(
            id="valid_threshold",
            type=IRNodeType.ML_PREDICT,
            parameters={
                'model_id': 'opportunity_scorer_v1',
                'confidence': {'min_confidence': 0.75}
            }
        ),
        
        # Node missing threshold
        IRNode(
            id="missing_threshold",
            type=IRNodeType.ML_SCORE,
            parameters={'model_id': 'churn_predictor_v2'}
        ),
        
        # Node with invalid threshold
        IRNode(
            id="invalid_threshold",
            type=IRNodeType.ML_CLASSIFY,
            parameters={
                'model_id': 'fraud_detector_v1',
                'confidence_threshold': 1.5  # Invalid: > 1.0
            }
        ),
        
        # High-risk node with low threshold
        IRNode(
            id="risky_low_threshold",
            type=IRNodeType.ML_PREDICT,
            parameters={
                'model_id': 'credit_approval_v1',
                'confidence_threshold': 0.5,  # Too low for high-risk
                'risk_level': 'high_risk'
            }
        )
    ]
    
    return nodes

def run_confidence_threshold_lint_tests():
    """Run confidence threshold lint tests"""
    linter = ConfidenceThresholdLinter(ConfidenceLintMode.AUTOFIX)
    
    # Create test graph
    test_nodes = create_test_ml_nodes_for_confidence_lint()
    from .ir import IRGraph
    
    test_graph = IRGraph(
        workflow_id="test_confidence_lint",
        name="Test Confidence Lint Workflow",
        version="1.0.0",
        automation_type="rbia",
        nodes=test_nodes
    )
    
    # Run lint
    errors, autofixes = linter.lint_confidence_thresholds(test_graph)
    
    print(f"Confidence threshold lint completed. Found {len(errors)} issues:")
    for error in errors:
        print(f"  {error.error_code} [{error.severity}] {error.node_id}: {error.message}")
        if error.suggested_threshold:
            print(f"    Suggested fix: {error.suggested_threshold}")
    
    print(f"\nAutofix suggestions: {len(autofixes)}")
    for autofix in autofixes:
        print(f"  {autofix.node_id}: {autofix.suggested_threshold} ({autofix.confidence_level})")
    
    # Print summary
    summary = linter.get_lint_summary()
    print(f"\nLint Summary:")
    print(f"  Compliance passed: {summary['compliance_passed']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Autofix available: {summary['autofix_available']}")
    
    return errors, autofixes

if __name__ == "__main__":
    run_confidence_threshold_lint_tests()
