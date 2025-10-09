"""
Trust Budget API and Model Specification - Task 6.2.9
======================================================

Add trust budget fields and API for trust budget summary
- Enhanced trust budget model with trust_min_auto, trust_min_assisted, trust_max_block
- Trust computation with weighted signals
- API to export trust budget summary for dashboarding/approval gates
- Validation for ML nodes

Dependencies: Task 6.2.4 (IR - enhanced IRTrustBudget)
Outputs: Trust budget summary API → dashboards and approval gates
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

# Task 6.2.9: Trust Budget Model Specification
@dataclass
class TrustSignalConfig:
    """Configuration for trust signal computation"""
    signal_name: str
    weight: float
    min_value: float = 0.0
    max_value: float = 1.0
    normalization: str = "linear"  # linear, log, sigmoid
    description: str = ""

@dataclass
class TrustBudgetModelSpec:
    """Trust budget model specification"""
    model_id: str
    version: str
    description: str
    
    # Input signals configuration
    signals: List[TrustSignalConfig] = field(default_factory=list)
    
    # Aggregation rules
    aggregation_method: str = "weighted_average"  # weighted_average, min, max, product
    normalization_method: str = "linear"
    
    # Thresholds
    auto_execute_threshold: float = 0.9
    assisted_threshold: float = 0.7
    block_threshold: float = 0.5
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    
    def validate(self) -> List[str]:
        """Validate trust budget model specification"""
        errors = []
        
        # Check weights sum to 1.0
        total_weight = sum(signal.weight for signal in self.signals)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Signal weights sum to {total_weight}, should sum to 1.0")
        
        # Check threshold ordering
        if not (self.block_threshold <= self.assisted_threshold <= self.auto_execute_threshold):
            errors.append("Thresholds must be ordered: block <= assisted <= auto_execute")
        
        # Check signal configurations
        for signal in self.signals:
            if signal.weight < 0 or signal.weight > 1:
                errors.append(f"Signal {signal.signal_name} weight must be between 0 and 1")
            if signal.min_value >= signal.max_value:
                errors.append(f"Signal {signal.signal_name} min_value must be < max_value")
        
        return errors

# Task 6.2.9: Trust Budget Validator
class TrustBudgetValidator:
    """Validates trust budget configurations for ML nodes"""
    
    def __init__(self):
        self.validation_errors: List[str] = []
    
    def validate_ir_graph_trust_budgets(self, ir_graph) -> bool:
        """Validate trust budgets for all ML nodes in IR graph"""
        self.validation_errors.clear()
        
        ml_nodes = ir_graph.get_ml_nodes()
        
        for node in ml_nodes:
            if not self._validate_node_trust_budget(node):
                return False
        
        return len(self.validation_errors) == 0
    
    def _validate_node_trust_budget(self, node) -> bool:
        """Validate trust budget for individual ML node"""
        if not node.trust_budget:
            self.validation_errors.append(f"ML node {node.id} missing trust budget")
            return False
        
        trust_budget = node.trust_budget
        
        # Validate threshold ordering
        if not (trust_budget.trust_max_block <= trust_budget.trust_min_assisted <= trust_budget.trust_min_auto):
            self.validation_errors.append(
                f"Node {node.id}: trust thresholds must be ordered: "
                f"trust_max_block ({trust_budget.trust_max_block}) <= "
                f"trust_min_assisted ({trust_budget.trust_min_assisted}) <= "
                f"trust_min_auto ({trust_budget.trust_min_auto})"
            )
            return False
        
        # Validate signal weights
        if trust_budget.signal_weights:
            total_weight = sum(trust_budget.signal_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                self.validation_errors.append(
                    f"Node {node.id}: signal weights sum to {total_weight}, should sum to 1.0"
                )
                return False
        
        # Validate signal names
        if trust_budget.trust_signals and trust_budget.signal_weights:
            for signal_name in trust_budget.trust_signals:
                if signal_name not in trust_budget.signal_weights:
                    self.validation_errors.append(
                        f"Node {node.id}: signal '{signal_name}' missing weight"
                    )
                    return False
        
        return True

# Task 6.2.9: Trust Budget Summary API
class TrustBudgetSummaryAPI:
    """API for trust budget summary and dashboarding"""
    
    def __init__(self):
        self.validator = TrustBudgetValidator()
    
    def get_workflow_trust_summary(self, ir_graph) -> Dict[str, Any]:
        """Get trust budget summary for entire workflow"""
        ml_nodes = ir_graph.get_ml_nodes()
        
        if not ml_nodes:
            return {
                "workflow_id": ir_graph.workflow_id,
                "ml_nodes_count": 0,
                "trust_summary": "No ML nodes in workflow"
            }
        
        # Validate trust budgets
        is_valid = self.validator.validate_ir_graph_trust_budgets(ir_graph)
        
        # Collect trust budget data
        node_summaries = []
        trust_scores = []
        execution_modes = {"AUTO_EXECUTE": 0, "ASSISTED": 0, "MANUAL_REVIEW": 0, "BLOCKED": 0}
        
        for node in ml_nodes:
            node_summary = self._get_node_trust_summary(node)
            node_summaries.append(node_summary)
            
            if node_summary["trust_budget"]["current_trust_score"]:
                trust_scores.append(node_summary["trust_budget"]["current_trust_score"])
            
            execution_mode = node_summary["trust_budget"]["execution_mode"]
            if execution_mode in execution_modes:
                execution_modes[execution_mode] += 1
        
        # Calculate aggregate metrics
        avg_trust_score = sum(trust_scores) / len(trust_scores) if trust_scores else 0
        min_trust_score = min(trust_scores) if trust_scores else 0
        max_trust_score = max(trust_scores) if trust_scores else 0
        
        return {
            "workflow_id": ir_graph.workflow_id,
            "workflow_name": ir_graph.name,
            "automation_type": ir_graph.automation_type,
            "ml_nodes_count": len(ml_nodes),
            "trust_validation": {
                "is_valid": is_valid,
                "errors": self.validator.validation_errors
            },
            "aggregate_metrics": {
                "average_trust_score": avg_trust_score,
                "min_trust_score": min_trust_score,
                "max_trust_score": max_trust_score,
                "execution_mode_distribution": execution_modes
            },
            "node_summaries": node_summaries,
            "recommendations": self._generate_trust_recommendations(node_summaries)
        }
    
    def _get_node_trust_summary(self, node) -> Dict[str, Any]:
        """Get trust budget summary for individual node"""
        if not node.trust_budget:
            return {
                "node_id": node.id,
                "node_type": node.type.value,
                "trust_budget": None,
                "status": "missing_trust_budget"
            }
        
        trust_budget = node.trust_budget
        
        # Simulate current trust score (in real implementation, would be computed from signals)
        current_signals = {
            "model_confidence": 0.85,
            "data_quality": 0.9,
            "override_rate": 0.05
        }
        current_trust_score = trust_budget.compute_trust_score(current_signals)
        current_execution_mode = trust_budget.get_execution_mode_for_trust(current_trust_score)
        
        return {
            "node_id": node.id,
            "node_type": node.type.value,
            "trust_budget": {
                "trust_min_auto": trust_budget.trust_min_auto,
                "trust_min_assisted": trust_budget.trust_min_assisted,
                "trust_max_block": trust_budget.trust_max_block,
                "trust_signals": trust_budget.trust_signals,
                "signal_weights": trust_budget.signal_weights,
                "normalization_method": trust_budget.normalization_method,
                "current_trust_score": current_trust_score,
                "execution_mode": current_execution_mode.value
            },
            "status": "configured"
        }
    
    def _generate_trust_recommendations(self, node_summaries: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for trust budget optimization"""
        recommendations = []
        
        for summary in node_summaries:
            node_id = summary["node_id"]
            trust_budget = summary.get("trust_budget")
            
            if not trust_budget:
                recommendations.append(f"Node {node_id}: Configure trust budget")
                continue
            
            current_score = trust_budget.get("current_trust_score", 0)
            execution_mode = trust_budget.get("execution_mode", "UNKNOWN")
            
            if execution_mode == "BLOCKED":
                recommendations.append(f"Node {node_id}: Trust score too low ({current_score:.2f}) - consider model retraining")
            elif execution_mode == "MANUAL_REVIEW":
                recommendations.append(f"Node {node_id}: Low trust score ({current_score:.2f}) - review data quality")
            elif execution_mode == "ASSISTED":
                recommendations.append(f"Node {node_id}: Moderate trust score ({current_score:.2f}) - monitor closely")
            
            # Check for imbalanced signal weights
            weights = trust_budget.get("signal_weights", {})
            if weights:
                max_weight = max(weights.values())
                if max_weight > 0.8:
                    recommendations.append(f"Node {node_id}: Consider rebalancing signal weights (max: {max_weight:.2f})")
        
        return recommendations
    
    def get_approval_gate_summary(self, ir_graph) -> Dict[str, Any]:
        """Get trust budget summary for approval gates"""
        workflow_summary = self.get_workflow_trust_summary(ir_graph)
        
        # Extract key metrics for approval gates
        ml_nodes_count = workflow_summary["ml_nodes_count"]
        avg_trust = workflow_summary["aggregate_metrics"]["average_trust_score"]
        min_trust = workflow_summary["aggregate_metrics"]["min_trust_score"]
        execution_modes = workflow_summary["aggregate_metrics"]["execution_mode_distribution"]
        
        # Determine approval requirements
        approval_required = False
        approval_reasons = []
        
        if min_trust < 0.7:
            approval_required = True
            approval_reasons.append(f"Low minimum trust score: {min_trust:.2f}")
        
        if execution_modes.get("BLOCKED", 0) > 0:
            approval_required = True
            approval_reasons.append(f"{execution_modes['BLOCKED']} nodes blocked due to low trust")
        
        if execution_modes.get("MANUAL_REVIEW", 0) > ml_nodes_count * 0.5:
            approval_required = True
            approval_reasons.append("More than 50% of ML nodes require manual review")
        
        risk_level = "LOW"
        if min_trust < 0.5:
            risk_level = "HIGH"
        elif min_trust < 0.7 or avg_trust < 0.8:
            risk_level = "MEDIUM"
        
        return {
            "workflow_id": ir_graph.workflow_id,
            "approval_gate": {
                "approval_required": approval_required,
                "approval_reasons": approval_reasons,
                "risk_level": risk_level,
                "recommended_approvers": self._get_recommended_approvers(risk_level)
            },
            "trust_metrics": {
                "ml_nodes_count": ml_nodes_count,
                "average_trust_score": avg_trust,
                "minimum_trust_score": min_trust,
                "nodes_requiring_review": execution_modes.get("MANUAL_REVIEW", 0) + execution_modes.get("BLOCKED", 0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_recommended_approvers(self, risk_level: str) -> List[str]:
        """Get recommended approvers based on risk level"""
        if risk_level == "HIGH":
            return ["ml_engineer", "data_scientist", "compliance_officer", "director"]
        elif risk_level == "MEDIUM":
            return ["ml_engineer", "data_scientist"]
        else:
            return ["ml_engineer"]

# Task 6.2.9: Default Trust Budget Model Specifications
DEFAULT_TRUST_MODELS = {
    "standard_ml_model": TrustBudgetModelSpec(
        model_id="standard_ml_model",
        version="1.0.0",
        description="Standard trust budget model for ML nodes",
        signals=[
            TrustSignalConfig("model_confidence", 0.6, 0.0, 1.0, "linear", "ML model confidence score"),
            TrustSignalConfig("data_quality", 0.3, 0.0, 1.0, "linear", "Input data quality score"),
            TrustSignalConfig("override_rate", 0.1, 0.0, 1.0, "inverse", "Historical override rate (inverted)")
        ],
        aggregation_method="weighted_average",
        auto_execute_threshold=0.9,
        assisted_threshold=0.7,
        block_threshold=0.5
    ),
    "high_risk_ml_model": TrustBudgetModelSpec(
        model_id="high_risk_ml_model", 
        version="1.0.0",
        description="Conservative trust budget model for high-risk ML nodes",
        signals=[
            TrustSignalConfig("model_confidence", 0.5, 0.0, 1.0, "linear", "ML model confidence score"),
            TrustSignalConfig("data_quality", 0.3, 0.0, 1.0, "linear", "Input data quality score"),
            TrustSignalConfig("override_rate", 0.1, 0.0, 1.0, "inverse", "Historical override rate (inverted)"),
            TrustSignalConfig("bias_score", 0.1, 0.0, 1.0, "inverse", "Bias detection score (inverted)")
        ],
        aggregation_method="weighted_average",
        auto_execute_threshold=0.95,
        assisted_threshold=0.8,
        block_threshold=0.6
    )
}
