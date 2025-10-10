"""
Fallback Coverage Validator - Task 6.2.21
==========================================

Fallback coverage ≥ 1 per ML node
- Analyzer that inspects ML nodes for fallback[] field and ensures non-empty coverage
- Coverage report showing for each ML node: fallback target(s), residency compatibility, risk tier
- Option to auto-generate simple RBA stub fallback for author review (linted as needing approval)

Dependencies: Task 6.2.4 (IR), Task 6.2.8 (Fallback DAG Resolver)
Outputs: Fallback coverage validation → ensures safe degradation paths
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class BuilderWarning:
    """Builder warning for missing fallback coverage"""
    warning_id: str
    node_id: str
    warning_type: str  # "missing_fallback", "incomplete_coverage", "high_risk_no_fallback"
    severity: str  # "error", "warning", "info"
    title: str
    message: str
    suggested_action: str
    blocks_publish: bool = False
    auto_fix_available: bool = False
    
@dataclass
class FallbackCoverageError:
    """Fallback coverage validation error"""
    error_code: str
    node_id: str
    model_id: Optional[str]
    severity: str  # "error", "warning", "info"
    message: str
    expected: str
    actual: str
    risk_tier: str  # "high", "medium", "low"
    autofix_available: bool = False
    suggested_fallback: Optional[Dict[str, Any]] = None
    requires_approval: bool = False
    source_location: Optional[Dict[str, Any]] = None

@dataclass
class FallbackCoverageReport:
    """Comprehensive fallback coverage report"""
    node_id: str
    model_id: Optional[str]
    node_type: str
    has_fallback: bool
    fallback_count: int
    fallback_targets: List[str]
    fallback_types: List[str]
    residency_compatible: bool
    risk_tier: str
    coverage_score: float  # 0.0 - 1.0
    coverage_gaps: List[str]
    recommendations: List[str]
    rba_fallback_available: bool
    human_escalation_available: bool
    max_fallback_depth: int

@dataclass
class FallbackAutofix:
    """Auto-generated fallback suggestion"""
    node_id: str
    fallback_type: str  # "rba_stub", "human_escalation", "safe_default"
    fallback_config: Dict[str, Any]
    justification: str
    requires_review: bool
    confidence_level: str  # "conservative", "moderate", "basic"

class FallbackCoverageSeverity(Enum):
    """Severity levels for fallback coverage"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class RiskTier(Enum):
    """Risk tiers for ML nodes"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Task 6.2.21: Fallback Coverage Validator
class FallbackCoverageValidator:
    """Validates fallback coverage for ML nodes"""
    
    def __init__(self):
        self.coverage_errors: List[FallbackCoverageError] = []
        self.coverage_reports: List[FallbackCoverageReport] = []
        self.autofix_suggestions: List[FallbackAutofix] = []
        
        # Fallback templates for auto-generation
        self.fallback_templates = {
            'rba_stub': {
                'type': 'rba_rule',
                'condition': 'true',  # Always trigger
                'action': {
                    'prediction': 'fallback_default',
                    'confidence': 0.3,
                    'reason': 'ML model fallback - requires review'
                },
                'metadata': {
                    'generated': True,
                    'requires_approval': True
                }
            },
            'human_escalation': {
                'type': 'human_escalation',
                'escalation_type': 'manual_review',
                'routing': 'default_queue',
                'sla_minutes': 60,
                'metadata': {
                    'generated': True,
                    'requires_approval': False
                }
            },
            'safe_default': {
                'type': 'safe_default',
                'action': {
                    'prediction': 'unknown',
                    'confidence': 0.1,
                    'reason': 'Safe default - no prediction available'
                },
                'metadata': {
                    'generated': True,
                    'requires_approval': False
                }
            }
        }
    
    def validate_fallback_coverage(self, ir_graph) -> Tuple[List[FallbackCoverageError], List[FallbackCoverageReport]]:
        """Validate fallback coverage for all ML nodes"""
        from .ir import IRGraph, IRNodeType
        
        self.coverage_errors.clear()
        self.coverage_reports.clear()
        self.autofix_suggestions.clear()
        
        # Get all ML nodes
        ml_nodes = ir_graph.get_ml_nodes()
        
        for node in ml_nodes:
            self._validate_node_fallback_coverage(node, ir_graph)
        
        return self.coverage_errors, self.coverage_reports
    
    def _validate_node_fallback_coverage(self, node, ir_graph):
        """Validate fallback coverage for a single ML node"""
        model_id = self._extract_model_id(node)
        risk_tier = self._assess_risk_tier(node, model_id)
        
        # Generate coverage report
        coverage_report = self._generate_coverage_report(node, model_id, risk_tier, ir_graph)
        self.coverage_reports.append(coverage_report)
        
        # Check for minimum coverage requirement
        if not coverage_report.has_fallback:
            self._report_missing_fallback(node, model_id, risk_tier)
        elif coverage_report.fallback_count == 0:
            self._report_empty_fallback(node, model_id, risk_tier)
        else:
            # Validate quality of existing fallbacks
            self._validate_fallback_quality(node, model_id, coverage_report)
    
    def _generate_coverage_report(self, node, model_id: Optional[str], risk_tier: str, ir_graph) -> FallbackCoverageReport:
        """Generate comprehensive coverage report for a node"""
        fallback_config = node.fallbacks
        
        # Basic coverage metrics
        has_fallback = fallback_config is not None and fallback_config.enabled
        fallback_count = len(fallback_config.fallback_items) if has_fallback else 0
        
        # Extract fallback details
        fallback_targets = []
        fallback_types = []
        rba_fallback_available = False
        human_escalation_available = False
        max_depth = 0
        
        if has_fallback and fallback_config.fallback_items:
            for item in fallback_config.fallback_items:
                fallback_targets.append(item.target_node_id or "unknown")
                fallback_types.append(item.type)
                
                if item.type == 'rba_rule':
                    rba_fallback_available = True
                elif item.type == 'human_escalation':
                    human_escalation_available = True
            
            max_depth = fallback_config.max_fallback_depth
        
        # Assess coverage quality
        coverage_score = self._calculate_coverage_score(
            fallback_count, fallback_types, risk_tier, rba_fallback_available, human_escalation_available
        )
        
        # Identify coverage gaps
        coverage_gaps = self._identify_coverage_gaps(
            fallback_types, risk_tier, rba_fallback_available, human_escalation_available
        )
        
        # Generate recommendations
        recommendations = self._generate_coverage_recommendations(
            node, coverage_gaps, risk_tier, fallback_count
        )
        
        # Check residency compatibility
        residency_compatible = self._check_residency_compatibility(node, fallback_targets, ir_graph)
        
        return FallbackCoverageReport(
            node_id=node.id,
            model_id=model_id,
            node_type=str(node.type),
            has_fallback=has_fallback,
            fallback_count=fallback_count,
            fallback_targets=fallback_targets,
            fallback_types=fallback_types,
            residency_compatible=residency_compatible,
            risk_tier=risk_tier,
            coverage_score=coverage_score,
            coverage_gaps=coverage_gaps,
            recommendations=recommendations,
            rba_fallback_available=rba_fallback_available,
            human_escalation_available=human_escalation_available,
            max_fallback_depth=max_depth
        )
    
    def _calculate_coverage_score(self, fallback_count: int, fallback_types: List[str], 
                                risk_tier: str, rba_available: bool, human_available: bool) -> float:
        """Calculate coverage quality score (0.0 - 1.0)"""
        score = 0.0
        
        # Base score for having any fallback
        if fallback_count > 0:
            score += 0.3
        
        # Score for RBA fallback (deterministic degradation)
        if rba_available:
            score += 0.3
        
        # Score for human escalation
        if human_available:
            score += 0.2
        
        # Score for multiple fallbacks
        if fallback_count > 1:
            score += 0.1
        
        # Score for diversity of fallback types
        unique_types = len(set(fallback_types))
        if unique_types > 1:
            score += 0.1
        
        # Adjust for risk tier requirements
        if risk_tier == 'high':
            # High-risk nodes need both RBA and human escalation
            if not (rba_available and human_available):
                score *= 0.6  # Penalty for missing critical fallbacks
        elif risk_tier == 'medium':
            # Medium-risk nodes need at least RBA or human escalation
            if not (rba_available or human_available):
                score *= 0.7
        
        return min(1.0, score)
    
    def _identify_coverage_gaps(self, fallback_types: List[str], risk_tier: str, 
                              rba_available: bool, human_available: bool) -> List[str]:
        """Identify gaps in fallback coverage"""
        gaps = []
        
        if not fallback_types:
            gaps.append("No fallback strategies defined")
            return gaps
        
        # Check for missing RBA fallback
        if not rba_available:
            if risk_tier in ['high', 'medium']:
                gaps.append("Missing RBA deterministic fallback")
            else:
                gaps.append("Consider adding RBA fallback for predictable degradation")
        
        # Check for missing human escalation
        if not human_available:
            if risk_tier == 'high':
                gaps.append("Missing human escalation for high-risk decisions")
            elif risk_tier == 'medium':
                gaps.append("Consider adding human escalation option")
        
        # Check for missing safe defaults
        if 'safe_default' not in fallback_types and risk_tier == 'high':
            gaps.append("Consider adding safe default for critical failures")
        
        # Check for single point of failure
        if len(set(fallback_types)) == 1:
            gaps.append("Limited fallback diversity - consider multiple strategies")
        
        return gaps
    
    def _generate_coverage_recommendations(self, node, gaps: List[str], 
                                         risk_tier: str, fallback_count: int) -> List[str]:
        """Generate specific recommendations for improving coverage"""
        recommendations = []
        
        if fallback_count == 0:
            recommendations.append(f"Add at least one fallback strategy for {risk_tier}-risk node")
            
            if risk_tier == 'high':
                recommendations.append("Implement both RBA fallback and human escalation")
            elif risk_tier == 'medium':
                recommendations.append("Implement RBA fallback or human escalation")
            else:
                recommendations.append("Implement basic safe default fallback")
        
        # Specific gap-based recommendations
        for gap in gaps:
            if "RBA" in gap:
                recommendations.append("Add deterministic RBA rule as fallback")
            elif "human escalation" in gap:
                recommendations.append("Add human review escalation path")
            elif "safe default" in gap:
                recommendations.append("Add safe default action for critical failures")
            elif "diversity" in gap:
                recommendations.append("Add secondary fallback strategy for redundancy")
        
        # Risk-specific recommendations
        if risk_tier == 'high' and fallback_count < 2:
            recommendations.append("High-risk nodes should have multiple fallback layers")
        
        return recommendations
    
    def _check_residency_compatibility(self, node, fallback_targets: List[str], ir_graph) -> bool:
        """Check if fallback targets are residency-compatible"""
        node_region = node.region_id
        if not node_region:
            return True  # No residency constraints
        
        for target_id in fallback_targets:
            if target_id == "unknown":
                continue
                
            target_node = ir_graph.get_node(target_id)
            if target_node and target_node.region_id:
                if target_node.region_id != node_region:
                    return False
        
        return True
    
    def _report_missing_fallback(self, node, model_id: Optional[str], risk_tier: str):
        """Report missing fallback configuration"""
        severity = self._get_severity_for_risk_tier(risk_tier)
        
        self.coverage_errors.append(FallbackCoverageError(
            error_code="FB001",
            node_id=node.id,
            model_id=model_id,
            severity=severity.value,
            message="ML node missing fallback configuration",
            expected="At least one fallback strategy",
            actual="No fallback configuration found",
            risk_tier=risk_tier,
            autofix_available=True,
            suggested_fallback=self._suggest_fallback_for_risk_tier(risk_tier),
            requires_approval=True
        ))
        
        # Generate autofix suggestion
        self._generate_autofix_suggestion(node, risk_tier, "missing_fallback")
    
    def _report_empty_fallback(self, node, model_id: Optional[str], risk_tier: str):
        """Report empty fallback configuration"""
        severity = self._get_severity_for_risk_tier(risk_tier)
        
        self.coverage_errors.append(FallbackCoverageError(
            error_code="FB002",
            node_id=node.id,
            model_id=model_id,
            severity=severity.value,
            message="ML node has empty fallback configuration",
            expected="At least one fallback item in fallback array",
            actual="Empty fallback configuration",
            risk_tier=risk_tier,
            autofix_available=True,
            suggested_fallback=self._suggest_fallback_for_risk_tier(risk_tier),
            requires_approval=True
        ))
        
        # Generate autofix suggestion
        self._generate_autofix_suggestion(node, risk_tier, "empty_fallback")
    
    def _validate_fallback_quality(self, node, model_id: Optional[str], coverage_report: FallbackCoverageReport):
        """Validate quality of existing fallback configuration"""
        # Check coverage score
        if coverage_report.coverage_score < 0.5:
            self.coverage_errors.append(FallbackCoverageError(
                error_code="FB003",
                node_id=node.id,
                model_id=model_id,
                severity=FallbackCoverageSeverity.WARNING.value,
                message=f"Low fallback coverage score: {coverage_report.coverage_score:.2f}",
                expected="Coverage score >= 0.5",
                actual=f"Score: {coverage_report.coverage_score:.2f}",
                risk_tier=coverage_report.risk_tier,
                autofix_available=True,
                requires_approval=False
            ))
        
        # Check residency compatibility
        if not coverage_report.residency_compatible:
            self.coverage_errors.append(FallbackCoverageError(
                error_code="FB004",
                node_id=node.id,
                model_id=model_id,
                severity=FallbackCoverageSeverity.ERROR.value,
                message="Fallback targets not residency-compatible",
                expected="Fallback targets in same region as node",
                actual="Cross-region fallback references found",
                risk_tier=coverage_report.risk_tier,
                autofix_available=False
            ))
        
        # Check for high-risk specific requirements
        if coverage_report.risk_tier == 'high':
            if not coverage_report.rba_fallback_available:
                self.coverage_errors.append(FallbackCoverageError(
                    error_code="FB005",
                    node_id=node.id,
                    model_id=model_id,
                    severity=FallbackCoverageSeverity.ERROR.value,
                    message="High-risk node missing RBA fallback",
                    expected="Deterministic RBA fallback for high-risk nodes",
                    actual="No RBA fallback found",
                    risk_tier=coverage_report.risk_tier,
                    autofix_available=True,
                    suggested_fallback=self.fallback_templates['rba_stub'],
                    requires_approval=True
                ))
            
            if not coverage_report.human_escalation_available:
                self.coverage_errors.append(FallbackCoverageError(
                    error_code="FB006",
                    node_id=node.id,
                    model_id=model_id,
                    severity=FallbackCoverageSeverity.WARNING.value,
                    message="High-risk node missing human escalation",
                    expected="Human escalation path for high-risk nodes",
                    actual="No human escalation found",
                    risk_tier=coverage_report.risk_tier,
                    autofix_available=True,
                    suggested_fallback=self.fallback_templates['human_escalation'],
                    requires_approval=False
                ))
    
    def _get_severity_for_risk_tier(self, risk_tier: str) -> FallbackCoverageSeverity:
        """Get validation severity based on risk tier"""
        if risk_tier == 'high':
            return FallbackCoverageSeverity.ERROR
        elif risk_tier == 'medium':
            return FallbackCoverageSeverity.WARNING
        else:
            return FallbackCoverageSeverity.INFO
    
    def _suggest_fallback_for_risk_tier(self, risk_tier: str) -> Dict[str, Any]:
        """Suggest appropriate fallback configuration for risk tier"""
        if risk_tier == 'high':
            return {
                'enabled': True,
                'fallback_items': [
                    self.fallback_templates['rba_stub'],
                    self.fallback_templates['human_escalation']
                ],
                'max_fallback_depth': 2
            }
        elif risk_tier == 'medium':
            return {
                'enabled': True,
                'fallback_items': [
                    self.fallback_templates['rba_stub']
                ],
                'max_fallback_depth': 1
            }
        else:
            return {
                'enabled': True,
                'fallback_items': [
                    self.fallback_templates['safe_default']
                ],
                'max_fallback_depth': 1
            }
    
    def _generate_autofix_suggestion(self, node, risk_tier: str, issue_type: str):
        """Generate autofix suggestion for fallback issue"""
        if risk_tier == 'high':
            # High-risk: RBA stub + human escalation
            self.autofix_suggestions.append(FallbackAutofix(
                node_id=node.id,
                fallback_type='rba_stub',
                fallback_config=self.fallback_templates['rba_stub'],
                justification=f"Generated RBA stub for high-risk node - {issue_type}",
                requires_review=True,
                confidence_level='conservative'
            ))
            
            self.autofix_suggestions.append(FallbackAutofix(
                node_id=node.id,
                fallback_type='human_escalation',
                fallback_config=self.fallback_templates['human_escalation'],
                justification=f"Added human escalation for high-risk node - {issue_type}",
                requires_review=False,
                confidence_level='conservative'
            ))
        
        elif risk_tier == 'medium':
            # Medium-risk: RBA stub
            self.autofix_suggestions.append(FallbackAutofix(
                node_id=node.id,
                fallback_type='rba_stub',
                fallback_config=self.fallback_templates['rba_stub'],
                justification=f"Generated RBA stub for medium-risk node - {issue_type}",
                requires_review=True,
                confidence_level='moderate'
            ))
        
        else:
            # Low-risk: Safe default
            self.autofix_suggestions.append(FallbackAutofix(
                node_id=node.id,
                fallback_type='safe_default',
                fallback_config=self.fallback_templates['safe_default'],
                justification=f"Added safe default for low-risk node - {issue_type}",
                requires_review=False,
                confidence_level='basic'
            ))
    
    def _assess_risk_tier(self, node, model_id: Optional[str]) -> str:
        """Assess risk tier of ML node"""
        # Check explicit risk annotations
        if 'risk_tier' in node.parameters:
            return node.parameters['risk_tier']
        
        if 'risk_level' in node.metadata:
            return node.metadata['risk_level']
        
        # Infer from model ID patterns
        if model_id:
            high_risk_patterns = ['fraud', 'credit', 'loan', 'legal', 'compliance', 'financial']
            if any(pattern in model_id.lower() for pattern in high_risk_patterns):
                return 'high'
            
            medium_risk_patterns = ['pricing', 'forecast', 'churn', 'conversion', 'opportunity']
            if any(pattern in model_id.lower() for pattern in medium_risk_patterns):
                return 'medium'
        
        # Default to medium for ML nodes
        return 'medium'
    
    def _extract_model_id(self, node) -> Optional[str]:
        """Extract model_id from node"""
        return node.parameters.get('model_id') or node.metadata.get('model_id')
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of fallback coverage validation"""
        error_count = len([e for e in self.coverage_errors if e.severity == 'error'])
        warning_count = len([e for e in self.coverage_errors if e.severity == 'warning'])
        info_count = len([e for e in self.coverage_errors if e.severity == 'info'])
        
        # Calculate overall coverage metrics
        total_nodes = len(self.coverage_reports)
        nodes_with_fallback = len([r for r in self.coverage_reports if r.has_fallback])
        avg_coverage_score = (sum(r.coverage_score for r in self.coverage_reports) / total_nodes 
                             if total_nodes > 0 else 0.0)
        
        return {
            'total_issues': len(self.coverage_errors),
            'errors': error_count,
            'warnings': warning_count,
            'info': info_count,
            'coverage_passed': error_count == 0,
            'total_ml_nodes': total_nodes,
            'nodes_with_fallback': nodes_with_fallback,
            'coverage_percentage': (nodes_with_fallback / total_nodes * 100) if total_nodes > 0 else 0,
            'average_coverage_score': avg_coverage_score,
            'autofix_available': len(self.autofix_suggestions),
            'issues_by_risk_tier': self._group_issues_by_risk_tier(),
            'coverage_by_risk_tier': self._group_coverage_by_risk_tier(),
            'recommendations_summary': self._get_top_recommendations()
        }
    
    def generate_builder_warnings(self, ir_graph) -> List[BuilderWarning]:
        """Generate Builder warnings for missing fallback coverage - Task 6.4.41"""
        
        warnings = []
        
        # Validate fallback coverage and get errors
        errors, reports = self.validate_fallback_coverage(ir_graph)
        
        for error in errors:
            warning_type = "missing_fallback"
            severity = "error" if error.risk_tier == "high" else "warning"
            blocks_publish = error.risk_tier == "high"
            
            if error.error_code == "MISSING_FALLBACK":
                title = f"Missing Fallback: {error.node_id}"
                message = f"ML node '{error.node_id}' has no fallback configuration. This creates a risk of system failure if the ML model becomes unavailable."
                suggested_action = "Add at least one fallback option (RBA rule, human escalation, or safe default)"
            
            elif error.error_code == "INCOMPLETE_COVERAGE":
                warning_type = "incomplete_coverage"
                title = f"Incomplete Fallback Coverage: {error.node_id}"
                message = f"ML node '{error.node_id}' has partial fallback coverage but is missing critical fallback types for {error.risk_tier} risk scenarios."
                suggested_action = "Add recommended fallback types: " + ", ".join(error.expected.split(","))
            
            elif error.error_code == "HIGH_RISK_NO_RBA":
                warning_type = "high_risk_no_fallback"
                title = f"High-Risk Node Without RBA Fallback: {error.node_id}"
                message = f"High-risk ML node '{error.node_id}' lacks deterministic RBA fallback. This violates safety requirements."
                suggested_action = "Add RBA rule fallback for deterministic behavior"
                blocks_publish = True
            
            else:
                title = f"Fallback Issue: {error.node_id}"
                message = error.message
                suggested_action = "Review fallback configuration"
            
            warning = BuilderWarning(
                warning_id=f"fallback_{error.error_code.lower()}_{error.node_id}",
                node_id=error.node_id,
                warning_type=warning_type,
                severity=severity,
                title=title,
                message=message,
                suggested_action=suggested_action,
                blocks_publish=blocks_publish,
                auto_fix_available=error.autofix_available
            )
            
            warnings.append(warning)
        
        # Check for nodes with low coverage scores
        for report in reports:
            if report.coverage_score < 0.7 and not any(w.node_id == report.node_id for w in warnings):
                warning = BuilderWarning(
                    warning_id=f"fallback_low_coverage_{report.node_id}",
                    node_id=report.node_id,
                    warning_type="incomplete_coverage",
                    severity="warning",
                    title=f"Low Fallback Coverage: {report.node_id}",
                    message=f"Node has low fallback coverage score ({report.coverage_score:.1f}). Consider adding more fallback options.",
                    suggested_action=f"Add missing fallback types: {', '.join(report.coverage_gaps)}",
                    blocks_publish=False,
                    auto_fix_available=True
                )
                warnings.append(warning)
        
        logger.info(f"Generated {len(warnings)} Builder warnings for fallback coverage")
        return warnings
    
    def get_detailed_coverage_report(self) -> Dict[str, Any]:
        """Get detailed coverage report for all nodes"""
        return {
            'summary': self.get_validation_summary(),
            'node_reports': [
                {
                    'node_id': report.node_id,
                    'model_id': report.model_id,
                    'risk_tier': report.risk_tier,
                    'coverage_score': report.coverage_score,
                    'has_fallback': report.has_fallback,
                    'fallback_count': report.fallback_count,
                    'fallback_types': report.fallback_types,
                    'coverage_gaps': report.coverage_gaps,
                    'recommendations': report.recommendations,
                    'residency_compatible': report.residency_compatible
                }
                for report in self.coverage_reports
            ],
            'autofix_suggestions': [
                {
                    'node_id': autofix.node_id,
                    'fallback_type': autofix.fallback_type,
                    'justification': autofix.justification,
                    'requires_review': autofix.requires_review,
                    'confidence_level': autofix.confidence_level
                }
                for autofix in self.autofix_suggestions
            ]
        }
    
    def _group_issues_by_risk_tier(self) -> Dict[str, int]:
        """Group issues by risk tier"""
        tier_counts = {'high': 0, 'medium': 0, 'low': 0}
        for error in self.coverage_errors:
            if error.risk_tier in tier_counts:
                tier_counts[error.risk_tier] += 1
        return tier_counts
    
    def _group_coverage_by_risk_tier(self) -> Dict[str, Dict[str, Any]]:
        """Group coverage metrics by risk tier"""
        tier_metrics = {}
        
        for tier in ['high', 'medium', 'low']:
            tier_reports = [r for r in self.coverage_reports if r.risk_tier == tier]
            
            if tier_reports:
                tier_metrics[tier] = {
                    'total_nodes': len(tier_reports),
                    'nodes_with_fallback': len([r for r in tier_reports if r.has_fallback]),
                    'average_coverage_score': sum(r.coverage_score for r in tier_reports) / len(tier_reports),
                    'coverage_percentage': len([r for r in tier_reports if r.has_fallback]) / len(tier_reports) * 100
                }
            else:
                tier_metrics[tier] = {
                    'total_nodes': 0,
                    'nodes_with_fallback': 0,
                    'average_coverage_score': 0.0,
                    'coverage_percentage': 0.0
                }
        
        return tier_metrics
    
    def _get_top_recommendations(self) -> List[str]:
        """Get most common recommendations across all nodes"""
        recommendation_counts = {}
        
        for report in self.coverage_reports:
            for rec in report.recommendations:
                if rec not in recommendation_counts:
                    recommendation_counts[rec] = 0
                recommendation_counts[rec] += 1
        
        # Sort by frequency and return top 5
        sorted_recs = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{rec}: {count} nodes" for rec, count in sorted_recs[:5]]

# Test helper functions
def create_test_ml_nodes_for_fallback_coverage():
    """Create test ML nodes with various fallback coverage scenarios"""
    from .ir import IRNode, IRNodeType, IRFallback, IRFallbackItem
    
    nodes = [
        # Node with good fallback coverage
        IRNode(
            id="good_coverage",
            type=IRNodeType.ML_PREDICT,
            parameters={'model_id': 'opportunity_scorer_v1', 'risk_tier': 'medium'},
            fallbacks=IRFallback(
                enabled=True,
                fallback_items=[
                    IRFallbackItem(type='rba_rule', target_node_id='rba_fallback_1', condition='confidence < 0.7'),
                    IRFallbackItem(type='human_escalation', target_node_id='human_review', condition='confidence < 0.5')
                ],
                max_fallback_depth=2
            )
        ),
        
        # High-risk node missing fallback
        IRNode(
            id="missing_fallback_high_risk",
            type=IRNodeType.ML_CLASSIFY,
            parameters={'model_id': 'fraud_detector_v1', 'risk_tier': 'high'}
            # No fallback configuration
        ),
        
        # Node with empty fallback
        IRNode(
            id="empty_fallback",
            type=IRNodeType.ML_SCORE,
            parameters={'model_id': 'churn_predictor_v2'},
            fallbacks=IRFallback(
                enabled=True,
                fallback_items=[],  # Empty
                max_fallback_depth=0
            )
        ),
        
        # Node with poor coverage
        IRNode(
            id="poor_coverage",
            type=IRNodeType.ML_PREDICT,
            parameters={'model_id': 'credit_scorer_v2', 'risk_tier': 'high'},
            fallbacks=IRFallback(
                enabled=True,
                fallback_items=[
                    IRFallbackItem(type='safe_default', target_node_id='default_action')  # Only one, not ideal for high-risk
                ],
                max_fallback_depth=1
            )
        )
    ]
    
    return nodes

def run_fallback_coverage_tests():
    """Run fallback coverage validation tests"""
    validator = FallbackCoverageValidator()
    
    # Create test graph
    test_nodes = create_test_ml_nodes_for_fallback_coverage()
    from .ir import IRGraph
    
    test_graph = IRGraph(
        workflow_id="test_fallback_coverage",
        name="Test Fallback Coverage Workflow",
        version="1.0.0",
        automation_type="rbia",
        nodes=test_nodes
    )
    
    # Run validation
    errors, reports = validator.validate_fallback_coverage(test_graph)
    
    print(f"Fallback coverage validation completed. Found {len(errors)} issues:")
    for error in errors:
        print(f"  {error.error_code} [{error.severity}] {error.node_id}: {error.message}")
        if error.suggested_fallback:
            print(f"    Autofix available: {error.autofix_available}")
    
    print(f"\nCoverage Reports:")
    for report in reports:
        print(f"  {report.node_id} ({report.risk_tier} risk):")
        print(f"    Coverage Score: {report.coverage_score:.2f}")
        print(f"    Fallback Count: {report.fallback_count}")
        print(f"    Gaps: {', '.join(report.coverage_gaps) if report.coverage_gaps else 'None'}")
    
    # Print summary
    summary = validator.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"  Coverage passed: {summary['coverage_passed']}")
    print(f"  Coverage percentage: {summary['coverage_percentage']:.1f}%")
    print(f"  Average coverage score: {summary['average_coverage_score']:.2f}")
    print(f"  Autofix suggestions: {summary['autofix_available']}")
    
    # Print detailed report
    detailed_report = validator.get_detailed_coverage_report()
    print(f"\nAutofix Suggestions: {len(detailed_report['autofix_suggestions'])}")
    for autofix in detailed_report['autofix_suggestions']:
        print(f"  {autofix['node_id']}: {autofix['fallback_type']} ({'needs review' if autofix['requires_review'] else 'auto-apply'})")
    
    return errors, reports

if __name__ == "__main__":
    run_fallback_coverage_tests()
