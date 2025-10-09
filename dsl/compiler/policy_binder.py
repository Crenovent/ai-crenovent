"""
Policy Binder - Attach Policy Packs to IR
==========================================

Task 6.2.7: Build policy binder
- Load policy packs (YAML) from policy/ directory or policy registry
- Validate policy pack schema (versions, signatures)
- Attach constraints and metadata to IR nodes/edges
- Produce combined policy view for analyzer passes
- CLI/endpoint to preview effective policies

Dependencies: Task 6.2.4 (IR)
Outputs: Policy-annotated IR → input for analyzers (6.2.13+)
"""

import yaml
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)

# Task 6.2.7: Policy Pack Schema
@dataclass
class PolicyPackMetadata:
    """Policy pack metadata and versioning"""
    pack_id: str
    name: str
    version: str
    description: str
    created_at: datetime
    created_by: str
    signature: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class ResidencyPolicy:
    """Residency constraint policy"""
    allowed_regions: List[str]
    data_sovereignty: Dict[str, str]  # data_type -> required_region
    cross_region_allowed: bool = False
    fallback_region: Optional[str] = None

@dataclass
class ThresholdPolicy:
    """Threshold constraint policy"""
    confidence_min: float
    confidence_max: float
    score_thresholds: Dict[str, float]  # metric -> threshold
    auto_escalate_below: float
    block_below: float

@dataclass
class SoDPolicy:
    """Separation of Duties policy"""
    author_approver_separation: bool = True
    required_approvers: int = 1
    approver_roles: List[str]
    excluded_combinations: List[Tuple[str, str]]  # (role1, role2) cannot both approve

@dataclass
class BiasPolicy:
    """Bias and fairness policy"""
    fairness_metrics: List[str]  # demographic_parity, equalized_odds, etc.
    protected_attributes: List[str]
    bias_thresholds: Dict[str, float]
    mitigation_required: bool = True

@dataclass
class DriftPolicy:
    """Model drift monitoring policy"""
    data_drift_threshold: float
    prediction_drift_threshold: float
    monitoring_window_days: int
    retrain_trigger_threshold: float

@dataclass
class PolicyPack:
    """Complete policy pack definition"""
    metadata: PolicyPackMetadata
    residency: Optional[ResidencyPolicy] = None
    thresholds: Optional[ThresholdPolicy] = None
    sod: Optional[SoDPolicy] = None
    bias: Optional[BiasPolicy] = None
    drift: Optional[DriftPolicy] = None
    custom_policies: Dict[str, Any] = field(default_factory=dict)

# Task 6.2.7: Policy Binder
class PolicyBinder:
    """Binds policy packs to IR nodes and produces combined policy view"""
    
    def __init__(self, policy_directory: str = "policy/"):
        self.policy_directory = Path(policy_directory)
        self.loaded_packs: Dict[str, PolicyPack] = {}
        self.policy_registry: Dict[str, str] = {}  # pack_id -> file_path
        self._discover_policy_packs()
    
    def _discover_policy_packs(self):
        """Discover policy packs in directory"""
        if not self.policy_directory.exists():
            logger.warning(f"Policy directory {self.policy_directory} does not exist")
            return
        
        for yaml_file in self.policy_directory.glob("*.yaml"):
            try:
                pack_id = yaml_file.stem
                self.policy_registry[pack_id] = str(yaml_file)
                logger.info(f"Discovered policy pack: {pack_id}")
            except Exception as e:
                logger.error(f"Error discovering policy pack {yaml_file}: {e}")
    
    def load_policy_pack(self, pack_id: str) -> Optional[PolicyPack]:
        """Load and validate policy pack from YAML"""
        if pack_id in self.loaded_packs:
            return self.loaded_packs[pack_id]
        
        if pack_id not in self.policy_registry:
            logger.error(f"Policy pack not found: {pack_id}")
            return None
        
        try:
            file_path = self.policy_registry[pack_id]
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
                pack_data = yaml.safe_load(yaml_content)
            
            # Validate schema
            if not self._validate_policy_pack_schema(pack_data):
                logger.error(f"Invalid schema for policy pack: {pack_id}")
                return None
            
            # Validate signature if present
            if not self._validate_policy_pack_signature(pack_data, yaml_content):
                logger.error(f"Invalid signature for policy pack: {pack_id}")
                return None
            
            # Convert to PolicyPack object
            policy_pack = self._convert_to_policy_pack(pack_data)
            self.loaded_packs[pack_id] = policy_pack
            
            logger.info(f"Loaded policy pack: {pack_id} v{policy_pack.metadata.version}")
            return policy_pack
            
        except Exception as e:
            logger.error(f"Error loading policy pack {pack_id}: {e}")
            return None
    
    def bind_policies_to_ir(self, ir_graph) -> bool:
        """Bind policy pack constraints to IR graph nodes"""
        from .ir import IRGraph, IRPolicy  # Import here to avoid circular imports
        
        # Load workflow-level policy pack
        pack_id = ir_graph.policy_pack
        if not pack_id:
            logger.warning("No policy pack specified for workflow")
            return True
        
        policy_pack = self.load_policy_pack(pack_id)
        if not policy_pack:
            logger.error(f"Failed to load policy pack: {pack_id}")
            return False
        
        # Bind policies to each node
        for node in ir_graph.nodes:
            self._bind_node_policies(node, policy_pack)
        
        # Bind policies to edges
        for edge in ir_graph.edges:
            self._bind_edge_policies(ir_graph, edge, policy_pack)
        
        # Store policy pack metadata in IR
        ir_graph.metadata['policy_pack_metadata'] = {
            'pack_id': policy_pack.metadata.pack_id,
            'version': policy_pack.metadata.version,
            'signature': policy_pack.metadata.signature,
            'bound_at': datetime.utcnow().isoformat()
        }
        
        return True
    
    def _bind_node_policies(self, node, policy_pack: PolicyPack):
        """Bind policy constraints to individual IR node"""
        from .ir import IRPolicy
        
        # Bind residency policies
        if policy_pack.residency:
            residency_policy = IRPolicy(
                policy_id=f"{policy_pack.metadata.pack_id}_residency",
                policy_type="residency",
                parameters={
                    "allowed_regions": policy_pack.residency.allowed_regions,
                    "data_sovereignty": policy_pack.residency.data_sovereignty,
                    "cross_region_allowed": policy_pack.residency.cross_region_allowed,
                    "fallback_region": policy_pack.residency.fallback_region
                },
                enforcement_action="fail_closed"
            )
            node.policies.append(residency_policy)
            
            # Set node region if not already set
            if not node.region_id and policy_pack.residency.allowed_regions:
                node.region_id = policy_pack.residency.allowed_regions[0]
        
        # Bind threshold policies for ML nodes
        if self._is_ml_node(node) and policy_pack.thresholds:
            threshold_policy = IRPolicy(
                policy_id=f"{policy_pack.metadata.pack_id}_thresholds",
                policy_type="threshold",
                parameters={
                    "confidence_min": policy_pack.thresholds.confidence_min,
                    "confidence_max": policy_pack.thresholds.confidence_max,
                    "score_thresholds": policy_pack.thresholds.score_thresholds,
                    "auto_escalate_below": policy_pack.thresholds.auto_escalate_below,
                    "block_below": policy_pack.thresholds.block_below
                },
                enforcement_action="fail_closed"
            )
            node.policies.append(threshold_policy)
            
            # Update trust budget based on thresholds
            if node.trust_budget:
                node.trust_budget.min_trust = max(
                    node.trust_budget.min_trust,
                    policy_pack.thresholds.confidence_min
                )
                node.trust_budget.auto_execute_above = max(
                    node.trust_budget.auto_execute_above,
                    policy_pack.thresholds.auto_escalate_below
                )
        
        # Bind SoD policies
        if policy_pack.sod:
            sod_policy = IRPolicy(
                policy_id=f"{policy_pack.metadata.pack_id}_sod",
                policy_type="sod",
                parameters={
                    "author_approver_separation": policy_pack.sod.author_approver_separation,
                    "required_approvers": policy_pack.sod.required_approvers,
                    "approver_roles": policy_pack.sod.approver_roles,
                    "excluded_combinations": policy_pack.sod.excluded_combinations
                },
                enforcement_action="fail_closed"
            )
            node.policies.append(sod_policy)
        
        # Bind bias policies for ML nodes
        if self._is_ml_node(node) and policy_pack.bias:
            bias_policy = IRPolicy(
                policy_id=f"{policy_pack.metadata.pack_id}_bias",
                policy_type="bias",
                parameters={
                    "fairness_metrics": policy_pack.bias.fairness_metrics,
                    "protected_attributes": policy_pack.bias.protected_attributes,
                    "bias_thresholds": policy_pack.bias.bias_thresholds,
                    "mitigation_required": policy_pack.bias.mitigation_required
                },
                enforcement_action="warn"  # Bias policies typically warn rather than fail
            )
            node.policies.append(bias_policy)
        
        # Bind drift policies for ML nodes
        if self._is_ml_node(node) and policy_pack.drift:
            drift_policy = IRPolicy(
                policy_id=f"{policy_pack.metadata.pack_id}_drift",
                policy_type="drift",
                parameters={
                    "data_drift_threshold": policy_pack.drift.data_drift_threshold,
                    "prediction_drift_threshold": policy_pack.drift.prediction_drift_threshold,
                    "monitoring_window_days": policy_pack.drift.monitoring_window_days,
                    "retrain_trigger_threshold": policy_pack.drift.retrain_trigger_threshold
                },
                enforcement_action="warn"
            )
            node.policies.append(drift_policy)
    
    def _bind_edge_policies(self, ir_graph, edge: Dict[str, str], policy_pack: PolicyPack):
        """Bind policy constraints to IR graph edges"""
        # Edge-level policies (e.g., data flow residency constraints)
        if policy_pack.residency and not policy_pack.residency.cross_region_allowed:
            from_node = ir_graph.get_node(edge["from"])
            to_node = ir_graph.get_node(edge["to"])
            
            if from_node and to_node:
                if from_node.region_id != to_node.region_id:
                    # Add cross-region warning to edge metadata
                    if 'edge_policies' not in edge:
                        edge['edge_policies'] = []
                    
                    edge['edge_policies'].append({
                        'policy_type': 'residency',
                        'violation': 'cross_region_data_flow',
                        'from_region': from_node.region_id,
                        'to_region': to_node.region_id,
                        'enforcement': 'warn'
                    })
    
    def produce_combined_policy_view(self, ir_graph) -> Dict[str, Any]:
        """Produce combined policy view for analyzer passes"""
        policy_view = {
            "workflow_id": ir_graph.workflow_id,
            "policy_pack": ir_graph.policy_pack,
            "node_policies": {},
            "edge_policies": {},
            "global_constraints": {},
            "policy_summary": {
                "total_policies": 0,
                "policy_types": set(),
                "enforcement_actions": {}
            }
        }
        
        # Collect node-level policies
        for node in ir_graph.nodes:
            node_policies = []
            for policy in node.policies:
                policy_dict = {
                    "policy_id": policy.policy_id,
                    "policy_type": policy.policy_type,
                    "parameters": policy.parameters,
                    "enforcement_action": policy.enforcement_action
                }
                node_policies.append(policy_dict)
                
                # Update summary
                policy_view["policy_summary"]["total_policies"] += 1
                policy_view["policy_summary"]["policy_types"].add(policy.policy_type)
                
                enforcement = policy.enforcement_action
                if enforcement not in policy_view["policy_summary"]["enforcement_actions"]:
                    policy_view["policy_summary"]["enforcement_actions"][enforcement] = 0
                policy_view["policy_summary"]["enforcement_actions"][enforcement] += 1
            
            policy_view["node_policies"][node.id] = node_policies
        
        # Collect edge-level policies
        for edge in ir_graph.edges:
            edge_key = f"{edge['from']}->{edge['to']}"
            if 'edge_policies' in edge:
                policy_view["edge_policies"][edge_key] = edge['edge_policies']
        
        # Convert set to list for JSON serialization
        policy_view["policy_summary"]["policy_types"] = list(policy_view["policy_summary"]["policy_types"])
        
        return policy_view
    
    def preview_effective_policies(self, workflow_id: str, pack_id: str) -> Dict[str, Any]:
        """Preview effective policies for a given workflow and policy pack"""
        policy_pack = self.load_policy_pack(pack_id)
        if not policy_pack:
            return {"error": f"Policy pack not found: {pack_id}"}
        
        preview = {
            "workflow_id": workflow_id,
            "policy_pack": {
                "pack_id": policy_pack.metadata.pack_id,
                "version": policy_pack.metadata.version,
                "description": policy_pack.metadata.description
            },
            "effective_policies": {}
        }
        
        # Preview residency policies
        if policy_pack.residency:
            preview["effective_policies"]["residency"] = {
                "allowed_regions": policy_pack.residency.allowed_regions,
                "cross_region_allowed": policy_pack.residency.cross_region_allowed,
                "data_sovereignty_rules": len(policy_pack.residency.data_sovereignty)
            }
        
        # Preview threshold policies
        if policy_pack.thresholds:
            preview["effective_policies"]["thresholds"] = {
                "confidence_range": [policy_pack.thresholds.confidence_min, policy_pack.thresholds.confidence_max],
                "auto_escalate_below": policy_pack.thresholds.auto_escalate_below,
                "block_below": policy_pack.thresholds.block_below,
                "score_thresholds": len(policy_pack.thresholds.score_thresholds)
            }
        
        # Preview SoD policies
        if policy_pack.sod:
            preview["effective_policies"]["sod"] = {
                "author_approver_separation": policy_pack.sod.author_approver_separation,
                "required_approvers": policy_pack.sod.required_approvers,
                "approver_roles": policy_pack.sod.approver_roles
            }
        
        # Preview bias policies
        if policy_pack.bias:
            preview["effective_policies"]["bias"] = {
                "fairness_metrics": policy_pack.bias.fairness_metrics,
                "protected_attributes": policy_pack.bias.protected_attributes,
                "mitigation_required": policy_pack.bias.mitigation_required
            }
        
        # Preview drift policies
        if policy_pack.drift:
            preview["effective_policies"]["drift"] = {
                "data_drift_threshold": policy_pack.drift.data_drift_threshold,
                "prediction_drift_threshold": policy_pack.drift.prediction_drift_threshold,
                "monitoring_window_days": policy_pack.drift.monitoring_window_days
            }
        
        return preview
    
    def _validate_policy_pack_schema(self, pack_data: Dict[str, Any]) -> bool:
        """Validate policy pack YAML schema"""
        required_fields = ['metadata']
        
        for field in required_fields:
            if field not in pack_data:
                logger.error(f"Missing required field in policy pack: {field}")
                return False
        
        # Validate metadata
        metadata = pack_data['metadata']
        required_metadata = ['pack_id', 'name', 'version', 'created_by']
        
        for field in required_metadata:
            if field not in metadata:
                logger.error(f"Missing required metadata field: {field}")
                return False
        
        return True
    
    def _validate_policy_pack_signature(self, pack_data: Dict[str, Any], yaml_content: str) -> bool:
        """Validate policy pack signature"""
        metadata = pack_data.get('metadata', {})
        signature = metadata.get('signature')
        
        if not signature:
            # Signature not required, but generate checksum
            checksum = hashlib.sha256(yaml_content.encode()).hexdigest()
            metadata['checksum'] = checksum
            return True
        
        # TODO: Implement actual signature validation with KMS/HSM
        # For now, just validate checksum
        expected_checksum = metadata.get('checksum')
        if expected_checksum:
            actual_checksum = hashlib.sha256(yaml_content.encode()).hexdigest()
            return actual_checksum == expected_checksum
        
        return True
    
    def _convert_to_policy_pack(self, pack_data: Dict[str, Any]) -> PolicyPack:
        """Convert YAML data to PolicyPack object"""
        metadata_data = pack_data['metadata']
        
        metadata = PolicyPackMetadata(
            pack_id=metadata_data['pack_id'],
            name=metadata_data['name'],
            version=metadata_data['version'],
            description=metadata_data.get('description', ''),
            created_at=datetime.fromisoformat(metadata_data.get('created_at', datetime.utcnow().isoformat())),
            created_by=metadata_data['created_by'],
            signature=metadata_data.get('signature'),
            checksum=metadata_data.get('checksum')
        )
        
        # Convert policy sections
        residency = None
        if 'residency' in pack_data:
            res_data = pack_data['residency']
            residency = ResidencyPolicy(
                allowed_regions=res_data.get('allowed_regions', []),
                data_sovereignty=res_data.get('data_sovereignty', {}),
                cross_region_allowed=res_data.get('cross_region_allowed', False),
                fallback_region=res_data.get('fallback_region')
            )
        
        thresholds = None
        if 'thresholds' in pack_data:
            thresh_data = pack_data['thresholds']
            thresholds = ThresholdPolicy(
                confidence_min=thresh_data.get('confidence_min', 0.7),
                confidence_max=thresh_data.get('confidence_max', 1.0),
                score_thresholds=thresh_data.get('score_thresholds', {}),
                auto_escalate_below=thresh_data.get('auto_escalate_below', 0.8),
                block_below=thresh_data.get('block_below', 0.5)
            )
        
        sod = None
        if 'sod' in pack_data:
            sod_data = pack_data['sod']
            sod = SoDPolicy(
                author_approver_separation=sod_data.get('author_approver_separation', True),
                required_approvers=sod_data.get('required_approvers', 1),
                approver_roles=sod_data.get('approver_roles', []),
                excluded_combinations=sod_data.get('excluded_combinations', [])
            )
        
        bias = None
        if 'bias' in pack_data:
            bias_data = pack_data['bias']
            bias = BiasPolicy(
                fairness_metrics=bias_data.get('fairness_metrics', []),
                protected_attributes=bias_data.get('protected_attributes', []),
                bias_thresholds=bias_data.get('bias_thresholds', {}),
                mitigation_required=bias_data.get('mitigation_required', True)
            )
        
        drift = None
        if 'drift' in pack_data:
            drift_data = pack_data['drift']
            drift = DriftPolicy(
                data_drift_threshold=drift_data.get('data_drift_threshold', 0.1),
                prediction_drift_threshold=drift_data.get('prediction_drift_threshold', 0.1),
                monitoring_window_days=drift_data.get('monitoring_window_days', 30),
                retrain_trigger_threshold=drift_data.get('retrain_trigger_threshold', 0.2)
            )
        
        return PolicyPack(
            metadata=metadata,
            residency=residency,
            thresholds=thresholds,
            sod=sod,
            bias=bias,
            drift=drift,
            custom_policies=pack_data.get('custom_policies', {})
        )
    
    def _is_ml_node(self, node) -> bool:
        """Check if node is an ML node"""
        from .ir import IRNodeType
        ml_types = {IRNodeType.ML_NODE, IRNodeType.ML_PREDICT, 
                   IRNodeType.ML_SCORE, IRNodeType.ML_CLASSIFY, IRNodeType.ML_EXPLAIN}
        return node.type in ml_types

# Task 6.2.7: CLI/API Endpoint
class PolicyPreviewAPI:
    """API endpoint for policy preview"""
    
    def __init__(self):
        self.policy_binder = PolicyBinder()
    
    def preview_policies(self, workflow_id: str, pack_id: str) -> Dict[str, Any]:
        """CLI/API endpoint to preview effective policies"""
        return self.policy_binder.preview_effective_policies(workflow_id, pack_id)
    
    def list_policy_packs(self) -> List[Dict[str, Any]]:
        """List available policy packs"""
        packs = []
        for pack_id in self.policy_binder.policy_registry.keys():
            policy_pack = self.policy_binder.load_policy_pack(pack_id)
            if policy_pack:
                packs.append({
                    "pack_id": policy_pack.metadata.pack_id,
                    "name": policy_pack.metadata.name,
                    "version": policy_pack.metadata.version,
                    "description": policy_pack.metadata.description,
                    "created_by": policy_pack.metadata.created_by
                })
        return packs

# Example policy pack YAML structure
EXAMPLE_POLICY_PACK_YAML = """
metadata:
  pack_id: "saas_ml_governance_v1"
  name: "SaaS ML Governance Policy Pack"
  version: "1.2.0"
  description: "Governance policies for SaaS ML workflows"
  created_at: "2024-01-15T10:30:00Z"
  created_by: "governance-team"
  signature: "sha256:abc123..."
  checksum: "def456..."

residency:
  allowed_regions: ["us-east-1", "us-west-2"]
  data_sovereignty:
    customer_data: "us-east-1"
    financial_data: "us-east-1"
  cross_region_allowed: false
  fallback_region: "us-east-1"

thresholds:
  confidence_min: 0.7
  confidence_max: 1.0
  score_thresholds:
    churn_risk: 0.8
    fraud_score: 0.9
  auto_escalate_below: 0.8
  block_below: 0.5

sod:
  author_approver_separation: true
  required_approvers: 2
  approver_roles: ["ml_engineer", "data_scientist", "compliance_officer"]
  excluded_combinations:
    - ["author", "primary_approver"]

bias:
  fairness_metrics: ["demographic_parity", "equalized_odds"]
  protected_attributes: ["gender", "race", "age"]
  bias_thresholds:
    demographic_parity: 0.1
    equalized_odds: 0.1
  mitigation_required: true

drift:
  data_drift_threshold: 0.1
  prediction_drift_threshold: 0.15
  monitoring_window_days: 30
  retrain_trigger_threshold: 0.2

custom_policies:
  sox_compliance:
    evidence_retention_days: 2555
    audit_trail_required: true
  gdpr_compliance:
    data_minimization: true
    right_to_explanation: true
"""
