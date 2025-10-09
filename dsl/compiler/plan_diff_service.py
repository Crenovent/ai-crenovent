"""
Plan Diff Tool - Task 6.2.74
=============================

Compare two manifests with semantic diff
- Shows risk deltas between plan versions
- Safe reviews for plan changes
- Diff engine for plan comparison
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from dataclasses import dataclass
from enum import Enum
import json
import difflib

logger = logging.getLogger(__name__)

class DiffType(Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PlanDiff:
    diff_id: str
    source_plan_id: str
    target_plan_id: str
    diff_type: DiffType
    path: str
    old_value: Any
    new_value: Any
    risk_level: RiskLevel
    description: str

@dataclass
class SemanticDiffResult:
    diff_id: str
    source_plan_id: str
    target_plan_id: str
    
    # Diff details
    node_diffs: List[PlanDiff]
    edge_diffs: List[PlanDiff]
    policy_diffs: List[PlanDiff]
    metadata_diffs: List[PlanDiff]
    
    # Risk assessment
    overall_risk_level: RiskLevel
    risk_summary: Dict[str, int]
    
    # Statistics
    total_changes: int
    breaking_changes: int
    safe_changes: int

class PlanDiffService:
    """Service for comparing plan manifests"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_rules = self._initialize_risk_rules()
    
    def _initialize_risk_rules(self) -> Dict[str, RiskLevel]:
        """Initialize risk assessment rules"""
        return {
            # High-risk changes
            'nodes.*.type': RiskLevel.HIGH,
            'nodes.*.trust_budget': RiskLevel.HIGH,
            'nodes.*.fallbacks': RiskLevel.MEDIUM,
            'policies.approval_required': RiskLevel.HIGH,
            'policies.security_level': RiskLevel.CRITICAL,
            
            # Medium-risk changes
            'nodes.*.config': RiskLevel.MEDIUM,
            'nodes.*.policies': RiskLevel.MEDIUM,
            'edges.*.condition': RiskLevel.MEDIUM,
            'metadata.cost_summary': RiskLevel.LOW,
            
            # Low-risk changes
            'nodes.*.description': RiskLevel.LOW,
            'nodes.*.name': RiskLevel.LOW,
            'metadata.labels': RiskLevel.LOW,
            'metadata.documentation': RiskLevel.LOW
        }
    
    def compare_plans(self, source_plan: Dict[str, Any], 
                     target_plan: Dict[str, Any]) -> SemanticDiffResult:
        """Compare two plan manifests and generate semantic diff"""
        
        source_plan_id = source_plan.get('plan_id', 'source')
        target_plan_id = target_plan.get('plan_id', 'target')
        
        diff_id = f"diff_{source_plan_id}_{target_plan_id}"
        
        # Initialize result
        result = SemanticDiffResult(
            diff_id=diff_id,
            source_plan_id=source_plan_id,
            target_plan_id=target_plan_id,
            node_diffs=[],
            edge_diffs=[],
            policy_diffs=[],
            metadata_diffs=[],
            overall_risk_level=RiskLevel.LOW,
            risk_summary={level.value: 0 for level in RiskLevel},
            total_changes=0,
            breaking_changes=0,
            safe_changes=0
        )
        
        # Compare different sections
        result.node_diffs = self._compare_nodes(source_plan, target_plan)
        result.edge_diffs = self._compare_edges(source_plan, target_plan)
        result.policy_diffs = self._compare_policies(source_plan, target_plan)
        result.metadata_diffs = self._compare_metadata(source_plan, target_plan)
        
        # Calculate overall risk and statistics
        self._calculate_risk_summary(result)
        
        return result
    
    def _compare_nodes(self, source_plan: Dict[str, Any], 
                      target_plan: Dict[str, Any]) -> List[PlanDiff]:
        """Compare nodes between plans"""
        diffs = []
        
        source_nodes = {n.get('id'): n for n in source_plan.get('nodes', [])}
        target_nodes = {n.get('id'): n for n in target_plan.get('nodes', [])}
        
        # Find added nodes
        for node_id in target_nodes:
            if node_id not in source_nodes:
                diffs.append(PlanDiff(
                    diff_id=f"node_added_{node_id}",
                    source_plan_id=source_plan.get('plan_id', 'source'),
                    target_plan_id=target_plan.get('plan_id', 'target'),
                    diff_type=DiffType.ADDED,
                    path=f"nodes.{node_id}",
                    old_value=None,
                    new_value=target_nodes[node_id],
                    risk_level=self._assess_node_addition_risk(target_nodes[node_id]),
                    description=f"Added node: {node_id} ({target_nodes[node_id].get('type', 'unknown')})"
                ))
        
        # Find removed nodes
        for node_id in source_nodes:
            if node_id not in target_nodes:
                diffs.append(PlanDiff(
                    diff_id=f"node_removed_{node_id}",
                    source_plan_id=source_plan.get('plan_id', 'source'),
                    target_plan_id=target_plan.get('plan_id', 'target'),
                    diff_type=DiffType.REMOVED,
                    path=f"nodes.{node_id}",
                    old_value=source_nodes[node_id],
                    new_value=None,
                    risk_level=RiskLevel.HIGH,  # Removing nodes is high risk
                    description=f"Removed node: {node_id} ({source_nodes[node_id].get('type', 'unknown')})"
                ))
        
        # Find modified nodes
        for node_id in source_nodes:
            if node_id in target_nodes:
                node_diffs = self._compare_node_details(
                    source_nodes[node_id], target_nodes[node_id], node_id
                )
                diffs.extend(node_diffs)
        
        return diffs
    
    def _compare_node_details(self, source_node: Dict[str, Any], 
                            target_node: Dict[str, Any], node_id: str) -> List[PlanDiff]:
        """Compare details of a single node"""
        diffs = []
        
        # Compare key fields
        key_fields = ['type', 'config', 'policies', 'trust_budget', 'fallbacks']
        
        for field in key_fields:
            source_value = source_node.get(field)
            target_value = target_node.get(field)
            
            if source_value != target_value:
                risk_level = self._assess_field_change_risk(f"nodes.*.{field}", source_value, target_value)
                
                diffs.append(PlanDiff(
                    diff_id=f"node_modified_{node_id}_{field}",
                    source_plan_id="source",
                    target_plan_id="target",
                    diff_type=DiffType.MODIFIED,
                    path=f"nodes.{node_id}.{field}",
                    old_value=source_value,
                    new_value=target_value,
                    risk_level=risk_level,
                    description=f"Modified {field} in node {node_id}"
                ))
        
        return diffs
    
    def _compare_edges(self, source_plan: Dict[str, Any], 
                      target_plan: Dict[str, Any]) -> List[PlanDiff]:
        """Compare edges between plans"""
        diffs = []
        
        source_edges = source_plan.get('edges', [])
        target_edges = target_plan.get('edges', [])
        
        # Create edge signatures for comparison
        source_edge_sigs = {self._create_edge_signature(e): e for e in source_edges}
        target_edge_sigs = {self._create_edge_signature(e): e for e in target_edges}
        
        # Find added edges
        for sig in target_edge_sigs:
            if sig not in source_edge_sigs:
                edge = target_edge_sigs[sig]
                diffs.append(PlanDiff(
                    diff_id=f"edge_added_{sig}",
                    source_plan_id=source_plan.get('plan_id', 'source'),
                    target_plan_id=target_plan.get('plan_id', 'target'),
                    diff_type=DiffType.ADDED,
                    path=f"edges.{sig}",
                    old_value=None,
                    new_value=edge,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"Added edge: {edge.get('source')} → {edge.get('target')}"
                ))
        
        # Find removed edges
        for sig in source_edge_sigs:
            if sig not in target_edge_sigs:
                edge = source_edge_sigs[sig]
                diffs.append(PlanDiff(
                    diff_id=f"edge_removed_{sig}",
                    source_plan_id=source_plan.get('plan_id', 'source'),
                    target_plan_id=target_plan.get('plan_id', 'target'),
                    diff_type=DiffType.REMOVED,
                    path=f"edges.{sig}",
                    old_value=edge,
                    new_value=None,
                    risk_level=RiskLevel.HIGH,
                    description=f"Removed edge: {edge.get('source')} → {edge.get('target')}"
                ))
        
        return diffs
    
    def _compare_policies(self, source_plan: Dict[str, Any], 
                         target_plan: Dict[str, Any]) -> List[PlanDiff]:
        """Compare policies between plans"""
        diffs = []
        
        source_policies = source_plan.get('policies', {})
        target_policies = target_plan.get('policies', {})
        
        all_policy_keys = set(source_policies.keys()) | set(target_policies.keys())
        
        for key in all_policy_keys:
            source_value = source_policies.get(key)
            target_value = target_policies.get(key)
            
            if source_value != target_value:
                if source_value is None:
                    diff_type = DiffType.ADDED
                    description = f"Added policy: {key}"
                elif target_value is None:
                    diff_type = DiffType.REMOVED
                    description = f"Removed policy: {key}"
                else:
                    diff_type = DiffType.MODIFIED
                    description = f"Modified policy: {key}"
                
                risk_level = self._assess_field_change_risk(f"policies.{key}", source_value, target_value)
                
                diffs.append(PlanDiff(
                    diff_id=f"policy_{diff_type.value}_{key}",
                    source_plan_id=source_plan.get('plan_id', 'source'),
                    target_plan_id=target_plan.get('plan_id', 'target'),
                    diff_type=diff_type,
                    path=f"policies.{key}",
                    old_value=source_value,
                    new_value=target_value,
                    risk_level=risk_level,
                    description=description
                ))
        
        return diffs
    
    def _compare_metadata(self, source_plan: Dict[str, Any], 
                         target_plan: Dict[str, Any]) -> List[PlanDiff]:
        """Compare metadata between plans"""
        diffs = []
        
        source_metadata = source_plan.get('metadata', {})
        target_metadata = target_plan.get('metadata', {})
        
        # Compare important metadata fields
        important_fields = ['cost_summary', 'plan_labels', 'approval_gates', 'sandbox_mode']
        
        for field in important_fields:
            source_value = source_metadata.get(field)
            target_value = target_metadata.get(field)
            
            if source_value != target_value:
                if source_value is None:
                    diff_type = DiffType.ADDED
                elif target_value is None:
                    diff_type = DiffType.REMOVED
                else:
                    diff_type = DiffType.MODIFIED
                
                risk_level = self._assess_field_change_risk(f"metadata.{field}", source_value, target_value)
                
                diffs.append(PlanDiff(
                    diff_id=f"metadata_{diff_type.value}_{field}",
                    source_plan_id=source_plan.get('plan_id', 'source'),
                    target_plan_id=target_plan.get('plan_id', 'target'),
                    diff_type=diff_type,
                    path=f"metadata.{field}",
                    old_value=source_value,
                    new_value=target_value,
                    risk_level=risk_level,
                    description=f"Changed metadata: {field}"
                ))
        
        return diffs
    
    def _create_edge_signature(self, edge: Dict[str, Any]) -> str:
        """Create unique signature for edge comparison"""
        source = edge.get('source', edge.get('from', ''))
        target = edge.get('target', edge.get('to', ''))
        condition = edge.get('condition', '')
        return f"{source}→{target}:{condition}"
    
    def _assess_node_addition_risk(self, node: Dict[str, Any]) -> RiskLevel:
        """Assess risk level of adding a node"""
        node_type = node.get('type', '')
        
        if 'ml_' in node_type or node_type == 'ml_node':
            return RiskLevel.HIGH
        elif 'external' in node_type:
            return RiskLevel.MEDIUM
        elif 'critical' in node.get('description', '').lower():
            return RiskLevel.HIGH
        else:
            return RiskLevel.LOW
    
    def _assess_field_change_risk(self, field_path: str, old_value: Any, new_value: Any) -> RiskLevel:
        """Assess risk level of field change"""
        # Check specific risk rules
        for pattern, risk_level in self.risk_rules.items():
            if self._matches_pattern(field_path, pattern):
                return risk_level
        
        # Default risk assessment
        if old_value is None or new_value is None:
            return RiskLevel.MEDIUM  # Adding/removing fields
        
        return RiskLevel.LOW  # Modifying existing fields
    
    def _matches_pattern(self, field_path: str, pattern: str) -> bool:
        """Check if field path matches pattern"""
        # Simple pattern matching (could be enhanced with regex)
        pattern_parts = pattern.split('.')
        path_parts = field_path.split('.')
        
        if len(pattern_parts) != len(path_parts):
            return False
        
        for pattern_part, path_part in zip(pattern_parts, path_parts):
            if pattern_part != '*' and pattern_part != path_part:
                return False
        
        return True
    
    def _calculate_risk_summary(self, result: SemanticDiffResult):
        """Calculate overall risk and statistics"""
        all_diffs = (result.node_diffs + result.edge_diffs + 
                    result.policy_diffs + result.metadata_diffs)
        
        result.total_changes = len(all_diffs)
        
        # Count by risk level
        for diff in all_diffs:
            result.risk_summary[diff.risk_level.value] += 1
        
        # Determine overall risk
        if result.risk_summary['critical'] > 0:
            result.overall_risk_level = RiskLevel.CRITICAL
        elif result.risk_summary['high'] > 0:
            result.overall_risk_level = RiskLevel.HIGH
        elif result.risk_summary['medium'] > 0:
            result.overall_risk_level = RiskLevel.MEDIUM
        else:
            result.overall_risk_level = RiskLevel.LOW
        
        # Count breaking vs safe changes
        result.breaking_changes = (result.risk_summary['critical'] + 
                                 result.risk_summary['high'])
        result.safe_changes = result.total_changes - result.breaking_changes
    
    def generate_diff_report(self, diff_result: SemanticDiffResult) -> str:
        """Generate human-readable diff report"""
        lines = []
        
        # Header
        lines.append(f"Plan Diff Report: {diff_result.source_plan_id} → {diff_result.target_plan_id}")
        lines.append("=" * 60)
        lines.append("")
        
        # Summary
        lines.append("Summary:")
        lines.append(f"  Total changes: {diff_result.total_changes}")
        lines.append(f"  Overall risk: {diff_result.overall_risk_level.value.upper()}")
        lines.append(f"  Breaking changes: {diff_result.breaking_changes}")
        lines.append(f"  Safe changes: {diff_result.safe_changes}")
        lines.append("")
        
        # Risk breakdown
        lines.append("Risk Breakdown:")
        for level, count in diff_result.risk_summary.items():
            if count > 0:
                lines.append(f"  {level.title()}: {count}")
        lines.append("")
        
        # Detailed changes
        if diff_result.node_diffs:
            lines.append("Node Changes:")
            for diff in diff_result.node_diffs:
                lines.append(f"  [{diff.risk_level.value.upper()}] {diff.description}")
            lines.append("")
        
        if diff_result.edge_diffs:
            lines.append("Edge Changes:")
            for diff in diff_result.edge_diffs:
                lines.append(f"  [{diff.risk_level.value.upper()}] {diff.description}")
            lines.append("")
        
        if diff_result.policy_diffs:
            lines.append("Policy Changes:")
            for diff in diff_result.policy_diffs:
                lines.append(f"  [{diff.risk_level.value.upper()}] {diff.description}")
            lines.append("")
        
        if diff_result.metadata_diffs:
            lines.append("Metadata Changes:")
            for diff in diff_result.metadata_diffs:
                lines.append(f"  [{diff.risk_level.value.upper()}] {diff.description}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_json_diff(self, diff_result: SemanticDiffResult) -> Dict[str, Any]:
        """Generate JSON representation of diff"""
        return {
            'diff_id': diff_result.diff_id,
            'source_plan_id': diff_result.source_plan_id,
            'target_plan_id': diff_result.target_plan_id,
            'summary': {
                'total_changes': diff_result.total_changes,
                'overall_risk_level': diff_result.overall_risk_level.value,
                'breaking_changes': diff_result.breaking_changes,
                'safe_changes': diff_result.safe_changes,
                'risk_breakdown': diff_result.risk_summary
            },
            'changes': {
                'nodes': [
                    {
                        'diff_id': diff.diff_id,
                        'type': diff.diff_type.value,
                        'path': diff.path,
                        'risk_level': diff.risk_level.value,
                        'description': diff.description,
                        'old_value': diff.old_value,
                        'new_value': diff.new_value
                    }
                    for diff in diff_result.node_diffs
                ],
                'edges': [
                    {
                        'diff_id': diff.diff_id,
                        'type': diff.diff_type.value,
                        'path': diff.path,
                        'risk_level': diff.risk_level.value,
                        'description': diff.description
                    }
                    for diff in diff_result.edge_diffs
                ],
                'policies': [
                    {
                        'diff_id': diff.diff_id,
                        'type': diff.diff_type.value,
                        'path': diff.path,
                        'risk_level': diff.risk_level.value,
                        'description': diff.description,
                        'old_value': diff.old_value,
                        'new_value': diff.new_value
                    }
                    for diff in diff_result.policy_diffs
                ],
                'metadata': [
                    {
                        'diff_id': diff.diff_id,
                        'type': diff.diff_type.value,
                        'path': diff.path,
                        'risk_level': diff.risk_level.value,
                        'description': diff.description
                    }
                    for diff in diff_result.metadata_diffs
                ]
            }
        }
    
    def should_block_deployment(self, diff_result: SemanticDiffResult) -> Tuple[bool, List[str]]:
        """Determine if deployment should be blocked based on diff"""
        blocking_reasons = []
        
        # Block on critical changes
        if diff_result.risk_summary['critical'] > 0:
            blocking_reasons.append(f"Contains {diff_result.risk_summary['critical']} critical risk changes")
        
        # Block on too many high-risk changes
        if diff_result.risk_summary['high'] > 5:
            blocking_reasons.append(f"Contains {diff_result.risk_summary['high']} high-risk changes (limit: 5)")
        
        # Block on node removals
        node_removals = len([d for d in diff_result.node_diffs if d.diff_type == DiffType.REMOVED])
        if node_removals > 0:
            blocking_reasons.append(f"Contains {node_removals} node removals")
        
        should_block = len(blocking_reasons) > 0
        return should_block, blocking_reasons
