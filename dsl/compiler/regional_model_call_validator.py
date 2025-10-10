"""
Regional Model Call Validator - Task 6.2.24
============================================

Residency check: nodes only call regional models
- Analyzer that checks region_id consistency between caller and target for every external call
- Uses policy binder and regional model registry metadata to determine allowed endpoints
- Fails compile on mismatch and emits remediation steps (use regional model digest or proxy)
- Produces residency compliance report

Dependencies: Task 6.2.4 (IR), Task 6.2.7 (Policy Binder), Task 6.2.10 (Residency Propagation)
Outputs: Regional model call validation → ensures geo-compliance at compile time
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RegionalCallViolation:
    """Regional model call violation"""
    violation_code: str
    node_id: str
    model_id: Optional[str]
    severity: str  # "error", "warning", "info"
    violation_type: str  # "region_mismatch", "model_not_regional", "connector_cross_region"
    message: str
    caller_region: Optional[str]
    target_region: Optional[str]
    expected: str
    actual: str
    remediation_steps: List[str]
    autofix_available: bool = False
    suggested_endpoint: Optional[str] = None

@dataclass
class ModelRegistryEntry:
    """Model registry entry with regional deployment info"""
    model_id: str
    model_version: str
    deployed_regions: List[str]
    primary_region: str
    regional_endpoints: Dict[str, str]  # region -> endpoint
    cross_region_allowed: bool
    proxy_available: bool
    proxy_endpoints: Dict[str, str]  # region -> proxy_endpoint
    compliance_tags: List[str]

@dataclass
class ConnectorRegistryEntry:
    """Connector registry entry with regional info"""
    connector_id: str
    connector_type: str  # "database", "api", "service"
    deployed_regions: List[str]
    regional_endpoints: Dict[str, str]
    data_residency_required: bool
    cross_region_proxy: Optional[str] = None

@dataclass
class ResidencyComplianceReport:
    """Comprehensive residency compliance report"""
    workflow_id: str
    total_external_calls: int
    compliant_calls: int
    violation_count: int
    violations_by_type: Dict[str, int]
    violations_by_region: Dict[str, int]
    remediation_summary: Dict[str, int]
    compliance_score: float
    high_risk_violations: List[str]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)

class RegionalViolationType(Enum):
    """Types of regional violations"""
    REGION_MISMATCH = "region_mismatch"
    MODEL_NOT_REGIONAL = "model_not_regional"
    CONNECTOR_CROSS_REGION = "connector_cross_region"
    PROXY_REQUIRED = "proxy_required"
    ENDPOINT_UNAVAILABLE = "endpoint_unavailable"

class ComplianceLevel(Enum):
    """Compliance enforcement levels"""
    STRICT = "strict"      # Fail on any violation
    MODERATE = "moderate"  # Allow with proxy
    PERMISSIVE = "permissive"  # Warn only

# Task 6.2.24: Regional Model Call Validator
class RegionalModelCallValidator:
    """Validates that nodes only call region-allowed models and connectors"""
    
    def __init__(self, compliance_level: ComplianceLevel = ComplianceLevel.STRICT):
        self.violations: List[RegionalCallViolation] = []
        self.model_registry: Dict[str, ModelRegistryEntry] = {}
        self.connector_registry: Dict[str, ConnectorRegistryEntry] = {}
        self.compliance_level = compliance_level
        
        # Initialize mock registries for testing
        self._init_mock_registries()
    
    def validate_regional_calls(self, ir_graph) -> Tuple[List[RegionalCallViolation], ResidencyComplianceReport]:
        """Validate all external calls for regional compliance"""
        from .ir import IRGraph, IRNodeType
        
        self.violations.clear()
        
        # Validate each node's external calls
        for node in ir_graph.nodes:
            self._validate_node_regional_calls(node, ir_graph)
        
        # Generate compliance report
        report = self._generate_compliance_report(ir_graph)
        
        return self.violations, report
    
    def _validate_node_regional_calls(self, node, ir_graph):
        """Validate regional calls for a single node"""
        node_region = self._get_node_region(node, ir_graph)
        
        # Validate ML model calls
        if self._is_ml_node(node):
            self._validate_ml_model_call(node, node_region)
        
        # Validate connector calls
        self._validate_connector_calls(node, node_region)
        
        # Validate external service calls
        self._validate_external_service_calls(node, node_region)
    
    def _validate_ml_model_call(self, node, node_region: Optional[str]):
        """Validate ML model call for regional compliance"""
        model_id = self._extract_model_id(node)
        if not model_id:
            return
        
        # Get model registry entry
        model_entry = self._get_model_registry_entry(model_id)
        if not model_entry:
            self._report_model_not_found(node, model_id, node_region)
            return
        
        # Check if model is deployed in node's region
        if node_region and node_region not in model_entry.deployed_regions:
            self._report_model_region_mismatch(node, model_id, node_region, model_entry)
        
        # Check cross-region policy compliance
        if node_region and model_entry.primary_region != node_region:
            self._validate_cross_region_model_access(node, model_id, node_region, model_entry)
    
    def _validate_connector_calls(self, node, node_region: Optional[str]):
        """Validate connector calls for regional compliance"""
        connector_calls = self._extract_connector_calls(node)
        
        for connector_id, call_config in connector_calls.items():
            connector_entry = self._get_connector_registry_entry(connector_id)
            
            if not connector_entry:
                self._report_connector_not_found(node, connector_id, node_region)
                continue
            
            # Check if connector is available in node's region
            if node_region and node_region not in connector_entry.deployed_regions:
                self._report_connector_region_mismatch(node, connector_id, node_region, connector_entry)
            
            # Check data residency requirements
            if connector_entry.data_residency_required and node_region:
                self._validate_connector_data_residency(node, connector_id, node_region, connector_entry, call_config)
    
    def _validate_external_service_calls(self, node, node_region: Optional[str]):
        """Validate external service calls for regional compliance"""
        external_calls = node.external_call_regions
        
        for call_id, target_region in external_calls.items():
            if node_region and target_region and node_region != target_region:
                # Check if cross-region call is allowed
                if not self._is_cross_region_call_allowed(node, call_id, node_region, target_region):
                    self.violations.append(RegionalCallViolation(
                        violation_code="RC001",
                        node_id=node.id,
                        model_id=self._extract_model_id(node),
                        severity="error" if self.compliance_level == ComplianceLevel.STRICT else "warning",
                        violation_type=RegionalViolationType.REGION_MISMATCH.value,
                        message=f"Cross-region external call not allowed: {call_id}",
                        caller_region=node_region,
                        target_region=target_region,
                        expected=f"Call target in same region ({node_region})",
                        actual=f"Call target in different region ({target_region})",
                        remediation_steps=[
                            f"Use regional endpoint in {node_region}",
                            "Configure regional proxy if available",
                            "Update policy to allow cross-region calls"
                        ]
                    ))
    
    def _report_model_not_found(self, node, model_id: str, node_region: Optional[str]):
        """Report model not found in registry"""
        self.violations.append(RegionalCallViolation(
            violation_code="RC002",
            node_id=node.id,
            model_id=model_id,
            severity="error",
            violation_type=RegionalViolationType.MODEL_NOT_REGIONAL.value,
            message=f"Model not found in regional registry: {model_id}",
            caller_region=node_region,
            target_region=None,
            expected="Model registered with regional deployment info",
            actual="Model not found in registry",
            remediation_steps=[
                f"Register model {model_id} in model registry",
                "Specify regional deployment configuration",
                "Verify model ID spelling and version"
            ]
        ))
    
    def _report_model_region_mismatch(self, node, model_id: str, node_region: str, model_entry: ModelRegistryEntry):
        """Report model not available in required region"""
        available_regions = ", ".join(model_entry.deployed_regions)
        remediation_steps = [
            f"Deploy model {model_id} to region {node_region}",
            f"Use model in available regions: {available_regions}",
            "Move node to region with model deployment"
        ]
        
        # Add proxy option if available
        if model_entry.proxy_available and node_region in model_entry.proxy_endpoints:
            remediation_steps.insert(1, f"Use regional proxy: {model_entry.proxy_endpoints[node_region]}")
        
        self.violations.append(RegionalCallViolation(
            violation_code="RC003",
            node_id=node.id,
            model_id=model_id,
            severity="error" if self.compliance_level == ComplianceLevel.STRICT else "warning",
            violation_type=RegionalViolationType.REGION_MISMATCH.value,
            message=f"Model {model_id} not deployed in required region {node_region}",
            caller_region=node_region,
            target_region=model_entry.primary_region,
            expected=f"Model deployed in {node_region}",
            actual=f"Model only available in: {available_regions}",
            remediation_steps=remediation_steps,
            autofix_available=model_entry.proxy_available,
            suggested_endpoint=model_entry.proxy_endpoints.get(node_region)
        ))
    
    def _validate_cross_region_model_access(self, node, model_id: str, node_region: str, model_entry: ModelRegistryEntry):
        """Validate cross-region model access policies"""
        if not model_entry.cross_region_allowed:
            # Check if node policy allows cross-region access
            if not self._node_allows_cross_region_access(node):
                self.violations.append(RegionalCallViolation(
                    violation_code="RC004",
                    node_id=node.id,
                    model_id=model_id,
                    severity="error",
                    violation_type=RegionalViolationType.REGION_MISMATCH.value,
                    message=f"Cross-region model access not allowed for {model_id}",
                    caller_region=node_region,
                    target_region=model_entry.primary_region,
                    expected="Model access within same region or cross-region policy",
                    actual="Cross-region access denied by model policy",
                    remediation_steps=[
                        f"Use regional model deployment in {node_region}",
                        "Update model policy to allow cross-region access",
                        "Use regional proxy if available"
                    ]
                ))
    
    def _report_connector_not_found(self, node, connector_id: str, node_region: Optional[str]):
        """Report connector not found in registry"""
        self.violations.append(RegionalCallViolation(
            violation_code="RC005",
            node_id=node.id,
            model_id=self._extract_model_id(node),
            severity="error",
            violation_type=RegionalViolationType.CONNECTOR_CROSS_REGION.value,
            message=f"Connector not found in registry: {connector_id}",
            caller_region=node_region,
            target_region=None,
            expected="Connector registered with regional info",
            actual="Connector not found in registry",
            remediation_steps=[
                f"Register connector {connector_id} in connector registry",
                "Specify regional deployment configuration",
                "Verify connector ID and configuration"
            ]
        ))
    
    def _report_connector_region_mismatch(self, node, connector_id: str, node_region: str, connector_entry: ConnectorRegistryEntry):
        """Report connector not available in required region"""
        available_regions = ", ".join(connector_entry.deployed_regions)
        remediation_steps = [
            f"Deploy connector {connector_id} to region {node_region}",
            f"Use connector in available regions: {available_regions}"
        ]
        
        # Add proxy option if available
        if connector_entry.cross_region_proxy:
            remediation_steps.insert(1, f"Use cross-region proxy: {connector_entry.cross_region_proxy}")
        
        self.violations.append(RegionalCallViolation(
            violation_code="RC006",
            node_id=node.id,
            model_id=self._extract_model_id(node),
            severity="error" if connector_entry.data_residency_required else "warning",
            violation_type=RegionalViolationType.CONNECTOR_CROSS_REGION.value,
            message=f"Connector {connector_id} not available in region {node_region}",
            caller_region=node_region,
            target_region=connector_entry.deployed_regions[0] if connector_entry.deployed_regions else None,
            expected=f"Connector available in {node_region}",
            actual=f"Connector only available in: {available_regions}",
            remediation_steps=remediation_steps,
            autofix_available=bool(connector_entry.cross_region_proxy),
            suggested_endpoint=connector_entry.cross_region_proxy
        ))
    
    def _validate_connector_data_residency(self, node, connector_id: str, node_region: str, 
                                         connector_entry: ConnectorRegistryEntry, call_config: Dict[str, Any]):
        """Validate connector data residency requirements"""
        # Check if data being accessed requires residency compliance
        data_types = call_config.get('data_types', [])
        sensitive_data_types = ['customer_data', 'financial_data', 'pii', 'payment_info']
        
        has_sensitive_data = any(dt in sensitive_data_types for dt in data_types)
        
        if has_sensitive_data and connector_entry.data_residency_required:
            # Ensure connector endpoint is in same region
            endpoint = call_config.get('endpoint', '')
            if not self._is_endpoint_in_region(endpoint, node_region):
                self.violations.append(RegionalCallViolation(
                    violation_code="RC007",
                    node_id=node.id,
                    model_id=self._extract_model_id(node),
                    severity="error",
                    violation_type=RegionalViolationType.CONNECTOR_CROSS_REGION.value,
                    message=f"Data residency violation: sensitive data access across regions",
                    caller_region=node_region,
                    target_region=self._extract_region_from_endpoint(endpoint),
                    expected="Sensitive data access within same region",
                    actual=f"Cross-region data access detected",
                    remediation_steps=[
                        f"Use regional connector endpoint in {node_region}",
                        "Configure data replication to local region",
                        "Use data residency compliant proxy"
                    ]
                ))
    
    def _get_node_region(self, node, ir_graph) -> Optional[str]:
        """Get region for node"""
        # Check node-specific region
        if node.region_id:
            return node.region_id
        
        # Check workflow-level region
        if hasattr(ir_graph, 'region_id') and ir_graph.region_id:
            return ir_graph.region_id
        
        # Check governance metadata
        if ir_graph.governance and 'region_id' in ir_graph.governance:
            return ir_graph.governance['region_id']
        
        return None
    
    def _is_ml_node(self, node) -> bool:
        """Check if node is an ML node"""
        from .ir import IRNodeType
        ml_types = {IRNodeType.ML_NODE, IRNodeType.ML_PREDICT, 
                   IRNodeType.ML_SCORE, IRNodeType.ML_CLASSIFY, IRNodeType.ML_EXPLAIN}
        return node.type in ml_types
    
    def _extract_model_id(self, node) -> Optional[str]:
        """Extract model_id from node"""
        return node.parameters.get('model_id') or node.metadata.get('model_id')
    
    def _extract_connector_calls(self, node) -> Dict[str, Dict[str, Any]]:
        """Extract connector calls from node configuration"""
        connector_calls = {}
        
        # Check direct connector configuration
        if 'connectors' in node.parameters:
            for connector_id, config in node.parameters['connectors'].items():
                connector_calls[connector_id] = config
        
        # Check data source calls
        if 'data_sources' in node.parameters:
            for source_id, config in node.parameters['data_sources'].items():
                if 'connector_id' in config:
                    connector_calls[config['connector_id']] = config
        
        # Check external calls metadata
        for call_id, region in node.external_call_regions.items():
            if call_id.startswith('connector_'):
                connector_id = call_id.replace('connector_', '')
                connector_calls[connector_id] = {'target_region': region}
        
        return connector_calls
    
    def _get_model_registry_entry(self, model_id: str) -> Optional[ModelRegistryEntry]:
        """Get model registry entry"""
        return self.model_registry.get(model_id)
    
    def _get_connector_registry_entry(self, connector_id: str) -> Optional[ConnectorRegistryEntry]:
        """Get connector registry entry"""
        return self.connector_registry.get(connector_id)
    
    def _is_cross_region_call_allowed(self, node, call_id: str, source_region: str, target_region: str) -> bool:
        """Check if cross-region call is allowed by policies"""
        # Check node-level policies
        for policy in node.policies:
            if policy.policy_type == 'residency':
                params = policy.parameters
                if not params.get('cross_region_allowed', False):
                    return False
        
        # Check if specific call is whitelisted
        if 'cross_region_whitelist' in node.parameters:
            whitelist = node.parameters['cross_region_whitelist']
            if call_id in whitelist:
                return True
        
        return self.compliance_level == ComplianceLevel.PERMISSIVE
    
    def _node_allows_cross_region_access(self, node) -> bool:
        """Check if node policy allows cross-region access"""
        for policy in node.policies:
            if policy.policy_type == 'residency':
                return policy.parameters.get('cross_region_allowed', False)
        return False
    
    def _is_endpoint_in_region(self, endpoint: str, region: str) -> bool:
        """Check if endpoint is in specified region"""
        # Simple heuristic based on endpoint URL
        region_patterns = {
            'us-east-1': ['us-east-1', 'use1', 'virginia'],
            'us-west-2': ['us-west-2', 'usw2', 'oregon'],
            'eu-west-1': ['eu-west-1', 'euw1', 'ireland'],
            'ap-southeast-1': ['ap-southeast-1', 'apse1', 'singapore']
        }
        
        patterns = region_patterns.get(region, [region])
        return any(pattern in endpoint.lower() for pattern in patterns)
    
    def _extract_region_from_endpoint(self, endpoint: str) -> Optional[str]:
        """Extract region from endpoint URL"""
        # Simple pattern matching
        region_patterns = {
            'us-east-1': ['us-east-1', 'use1', 'virginia'],
            'us-west-2': ['us-west-2', 'usw2', 'oregon'],
            'eu-west-1': ['eu-west-1', 'euw1', 'ireland'],
            'ap-southeast-1': ['ap-southeast-1', 'apse1', 'singapore']
        }
        
        for region, patterns in region_patterns.items():
            if any(pattern in endpoint.lower() for pattern in patterns):
                return region
        
        return None
    
    def _generate_compliance_report(self, ir_graph) -> ResidencyComplianceReport:
        """Generate comprehensive residency compliance report"""
        total_calls = self._count_total_external_calls(ir_graph)
        violation_count = len(self.violations)
        compliant_calls = total_calls - violation_count
        
        # Group violations by type and region
        violations_by_type = {}
        violations_by_region = {}
        remediation_summary = {}
        
        for violation in self.violations:
            # By type
            vtype = violation.violation_type
            violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1
            
            # By region
            region = violation.caller_region or 'unknown'
            violations_by_region[region] = violations_by_region.get(region, 0) + 1
            
            # Remediation steps
            for step in violation.remediation_steps:
                step_key = step.split(':')[0] if ':' in step else step[:30]
                remediation_summary[step_key] = remediation_summary.get(step_key, 0) + 1
        
        # Calculate compliance score
        compliance_score = (compliant_calls / total_calls * 100) if total_calls > 0 else 100.0
        
        # Identify high-risk violations
        high_risk_violations = [
            f"{v.node_id}: {v.message}" 
            for v in self.violations 
            if v.severity == 'error' and 'sensitive data' in v.message.lower()
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return ResidencyComplianceReport(
            workflow_id=ir_graph.workflow_id,
            total_external_calls=total_calls,
            compliant_calls=compliant_calls,
            violation_count=violation_count,
            violations_by_type=violations_by_type,
            violations_by_region=violations_by_region,
            remediation_summary=remediation_summary,
            compliance_score=compliance_score,
            high_risk_violations=high_risk_violations,
            recommendations=recommendations
        )
    
    def _count_total_external_calls(self, ir_graph) -> int:
        """Count total external calls in workflow"""
        total_calls = 0
        
        for node in ir_graph.nodes:
            # Count ML model calls
            if self._is_ml_node(node) and self._extract_model_id(node):
                total_calls += 1
            
            # Count connector calls
            connector_calls = self._extract_connector_calls(node)
            total_calls += len(connector_calls)
            
            # Count external service calls
            total_calls += len(node.external_call_regions)
        
        return total_calls
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []
        
        # Analyze violation patterns
        violation_types = [v.violation_type for v in self.violations]
        
        if RegionalViolationType.REGION_MISMATCH.value in violation_types:
            recommendations.append("Deploy models to all required regions for better compliance")
        
        if RegionalViolationType.MODEL_NOT_REGIONAL.value in violation_types:
            recommendations.append("Register all models in regional model registry")
        
        if RegionalViolationType.CONNECTOR_CROSS_REGION.value in violation_types:
            recommendations.append("Configure regional connectors and data sources")
        
        # Add proxy recommendations if applicable
        proxy_available_count = len([v for v in self.violations if v.autofix_available])
        if proxy_available_count > 0:
            recommendations.append(f"Consider using regional proxies for {proxy_available_count} violations")
        
        # Add policy recommendations
        error_count = len([v for v in self.violations if v.severity == 'error'])
        if error_count > 0:
            recommendations.append("Review and update residency policies for compliance")
        
        return recommendations
    
    def _init_mock_registries(self):
        """Initialize mock model and connector registries for testing"""
        # Mock model registry
        self.model_registry = {
            'financial_scorer_v1': ModelRegistryEntry(
                model_id='financial_scorer_v1',
                model_version='1.2.0',
                deployed_regions=['us-east-1', 'us-west-2'],
                primary_region='us-east-1',
                regional_endpoints={
                    'us-east-1': 'https://ml-api.us-east-1.example.com/financial_scorer_v1',
                    'us-west-2': 'https://ml-api.us-west-2.example.com/financial_scorer_v1'
                },
                cross_region_allowed=False,
                proxy_available=True,
                proxy_endpoints={
                    'eu-west-1': 'https://ml-proxy.eu-west-1.example.com/financial_scorer_v1'
                },
                compliance_tags=['gdpr', 'sox']
            ),
            'fraud_detector_v1': ModelRegistryEntry(
                model_id='fraud_detector_v1',
                model_version='2.1.0',
                deployed_regions=['us-east-1', 'eu-west-1', 'ap-southeast-1'],
                primary_region='us-east-1',
                regional_endpoints={
                    'us-east-1': 'https://ml-api.us-east-1.example.com/fraud_detector_v1',
                    'eu-west-1': 'https://ml-api.eu-west-1.example.com/fraud_detector_v1',
                    'ap-southeast-1': 'https://ml-api.ap-southeast-1.example.com/fraud_detector_v1'
                },
                cross_region_allowed=True,
                proxy_available=False,
                proxy_endpoints={},
                compliance_tags=['pci', 'gdpr']
            ),
            'churn_predictor_v2': ModelRegistryEntry(
                model_id='churn_predictor_v2',
                model_version='2.0.1',
                deployed_regions=['us-west-2'],  # Limited deployment
                primary_region='us-west-2',
                regional_endpoints={
                    'us-west-2': 'https://ml-api.us-west-2.example.com/churn_predictor_v2'
                },
                cross_region_allowed=False,
                proxy_available=True,
                proxy_endpoints={
                    'us-east-1': 'https://ml-proxy.us-east-1.example.com/churn_predictor_v2'
                },
                compliance_tags=[]
            )
        }
        
        # Mock connector registry
        self.connector_registry = {
            'salesforce_connector': ConnectorRegistryEntry(
                connector_id='salesforce_connector',
                connector_type='api',
                deployed_regions=['us-east-1', 'us-west-2', 'eu-west-1'],
                regional_endpoints={
                    'us-east-1': 'https://api.salesforce.com/us-east-1',
                    'us-west-2': 'https://api.salesforce.com/us-west-2',
                    'eu-west-1': 'https://api.salesforce.com/eu-west-1'
                },
                data_residency_required=True,
                cross_region_proxy='https://proxy.salesforce.com'
            ),
            'postgres_connector': ConnectorRegistryEntry(
                connector_id='postgres_connector',
                connector_type='database',
                deployed_regions=['us-east-1'],  # Single region
                regional_endpoints={
                    'us-east-1': 'postgres://db.us-east-1.example.com:5432/prod'
                },
                data_residency_required=True,
                cross_region_proxy=None  # No cross-region access
            )
        }
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of regional validation results"""
        error_count = len([v for v in self.violations if v.severity == 'error'])
        warning_count = len([v for v in self.violations if v.severity == 'warning'])
        info_count = len([v for v in self.violations if v.severity == 'info'])
        
        autofix_count = len([v for v in self.violations if v.autofix_available])
        
        return {
            'total_violations': len(self.violations),
            'errors': error_count,
            'warnings': warning_count,
            'info': info_count,
            'regional_compliance_passed': error_count == 0,
            'autofix_available': autofix_count,
            'compliance_level': self.compliance_level.value,
            'violations_by_type': self._group_violations_by_type(),
            'violations_by_region': self._group_violations_by_region(),
            'common_violations': self._get_common_violations(),
            'remediation_actions': self._get_remediation_actions()
        }
    
    def _group_violations_by_type(self) -> Dict[str, int]:
        """Group violations by type"""
        type_counts = {}
        for violation in self.violations:
            vtype = violation.violation_type
            if vtype not in type_counts:
                type_counts[vtype] = 0
            type_counts[vtype] += 1
        return type_counts
    
    def _group_violations_by_region(self) -> Dict[str, int]:
        """Group violations by caller region"""
        region_counts = {}
        for violation in self.violations:
            region = violation.caller_region or 'unknown'
            if region not in region_counts:
                region_counts[region] = 0
            region_counts[region] += 1
        return region_counts
    
    def _get_common_violations(self) -> List[str]:
        """Get most common regional violations"""
        violation_descriptions = {
            'RC001': 'Cross-region external call not allowed',
            'RC002': 'Model not found in registry',
            'RC003': 'Model not deployed in required region',
            'RC004': 'Cross-region model access denied',
            'RC005': 'Connector not found in registry',
            'RC006': 'Connector not available in region',
            'RC007': 'Data residency violation'
        }
        
        violation_counts = {}
        for violation in self.violations:
            code = violation.violation_code
            if code not in violation_counts:
                violation_counts[code] = 0
            violation_counts[code] += 1
        
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [f"{violation_descriptions.get(code, code)}: {count} occurrences" 
                for code, count in sorted_violations[:5]]
    
    def _get_remediation_actions(self) -> List[str]:
        """Get top remediation actions"""
        action_counts = {}
        
        for violation in self.violations:
            for step in violation.remediation_steps:
                action_key = step.split(':')[0] if ':' in step else step[:50]
                if action_key not in action_counts:
                    action_counts[action_key] = 0
                action_counts[action_key] += 1
        
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [f"{action}: {count} occurrences" for action, count in sorted_actions[:5]]

# Test helper functions
def create_test_nodes_for_regional_validation():
    """Create test nodes with regional compliance issues"""
    from .ir import IRNode, IRNodeType, IRPolicy
    
    nodes = [
        # Node in us-east-1 calling model available in us-west-2 only
        IRNode(
            id="region_mismatch_node",
            type=IRNodeType.ML_PREDICT,
            region_id="us-east-1",
            parameters={
                'model_id': 'churn_predictor_v2',  # Only in us-west-2
                'connectors': {
                    'postgres_connector': {
                        'endpoint': 'postgres://db.us-east-1.example.com:5432/prod',
                        'data_types': ['customer_data']
                    }
                }
            },
            external_call_regions={
                'model_churn_predictor_v2': 'us-west-2'
            }
        ),
        
        # Node with proper regional setup
        IRNode(
            id="compliant_node",
            type=IRNodeType.ML_SCORE,
            region_id="us-east-1",
            parameters={
                'model_id': 'financial_scorer_v1',  # Available in us-east-1
                'connectors': {
                    'salesforce_connector': {
                        'endpoint': 'https://api.salesforce.com/us-east-1',
                        'data_types': ['opportunity_data']
                    }
                }
            },
            policies=[
                IRPolicy(
                    policy_id="residency_policy",
                    policy_type="residency",
                    parameters={
                        'cross_region_allowed': False,
                        'data_residency_required': True
                    }
                )
            ]
        ),
        
        # Node calling unregistered model
        IRNode(
            id="unregistered_model_node",
            type=IRNodeType.ML_CLASSIFY,
            region_id="eu-west-1",
            parameters={
                'model_id': 'unknown_model_v1'  # Not in registry
            }
        ),
        
        # Node with cross-region data access violation
        IRNode(
            id="data_residency_violation",
            type=IRNodeType.ML_PREDICT,
            region_id="eu-west-1",
            parameters={
                'model_id': 'fraud_detector_v1',
                'connectors': {
                    'postgres_connector': {
                        'endpoint': 'postgres://db.us-east-1.example.com:5432/prod',  # Cross-region
                        'data_types': ['customer_data', 'pii']  # Sensitive data
                    }
                }
            }
        )
    ]
    
    return nodes

def run_regional_validation_tests():
    """Run regional model call validation tests"""
    validator = RegionalModelCallValidator(ComplianceLevel.STRICT)
    
    # Create test graph
    test_nodes = create_test_nodes_for_regional_validation()
    from .ir import IRGraph
    
    test_graph = IRGraph(
        workflow_id="test_regional_validation",
        name="Test Regional Validation Workflow",
        version="1.0.0",
        automation_type="rbia",
        nodes=test_nodes
    )
    
    # Run validation
    violations, report = validator.validate_regional_calls(test_graph)
    
    print(f"Regional validation completed. Found {len(violations)} violations:")
    for violation in violations:
        print(f"  {violation.violation_code} [{violation.severity}] {violation.node_id}: {violation.message}")
        if violation.autofix_available:
            print(f"    → Autofix available: {violation.suggested_endpoint}")
        print(f"    → Remediation: {violation.remediation_steps[0]}")
    
    # Print compliance report
    print(f"\nCompliance Report:")
    print(f"  Total external calls: {report.total_external_calls}")
    print(f"  Compliant calls: {report.compliant_calls}")
    print(f"  Compliance score: {report.compliance_score:.1f}%")
    print(f"  High-risk violations: {len(report.high_risk_violations)}")
    
    # Print summary
    summary = validator.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"  Regional compliance passed: {summary['regional_compliance_passed']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Autofix available: {summary['autofix_available']}")
    
    return violations, report

if __name__ == "__main__":
    run_regional_validation_tests()
