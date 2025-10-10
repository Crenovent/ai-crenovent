"""
Trust Ladder Encoding - Task 6.2.60
====================================

Safety policy encoding trust levels
- RBA=1.0; RBIA 0.75–0.95
- IR attribute for threshold gating
- Trust ladder for execution modes
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TrustLevel(Enum):
    FULL_TRUST = "full_trust"          # 1.0 - RBA deterministic
    HIGH_TRUST = "high_trust"          # 0.95-0.99 - RBIA high confidence
    MEDIUM_TRUST = "medium_trust"      # 0.85-0.94 - RBIA medium confidence
    LOW_TRUST = "low_trust"            # 0.75-0.84 - RBIA low confidence
    NO_TRUST = "no_trust"              # < 0.75 - Block execution

@dataclass
class TrustLadderEntry:
    trust_level: TrustLevel
    trust_score_min: float
    trust_score_max: float
    execution_mode: str
    human_oversight_required: bool
    description: str

class TrustLadderEncoder:
    """Encodes trust levels for execution gating"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trust_ladder = self._initialize_trust_ladder()
    
    def _initialize_trust_ladder(self) -> List[TrustLadderEntry]:
        """Initialize trust ladder with standard levels"""
        return [
            TrustLadderEntry(
                trust_level=TrustLevel.FULL_TRUST,
                trust_score_min=1.0,
                trust_score_max=1.0,
                execution_mode="auto_execute",
                human_oversight_required=False,
                description="Deterministic RBA rules - full trust"
            ),
            TrustLadderEntry(
                trust_level=TrustLevel.HIGH_TRUST,
                trust_score_min=0.95,
                trust_score_max=0.99,
                execution_mode="auto_execute",
                human_oversight_required=False,
                description="High confidence RBIA - auto execute"
            ),
            TrustLadderEntry(
                trust_level=TrustLevel.MEDIUM_TRUST,
                trust_score_min=0.85,
                trust_score_max=0.94,
                execution_mode="assisted",
                human_oversight_required=True,
                description="Medium confidence RBIA - human assisted"
            ),
            TrustLadderEntry(
                trust_level=TrustLevel.LOW_TRUST,
                trust_score_min=0.75,
                trust_score_max=0.84,
                execution_mode="manual_review",
                human_oversight_required=True,
                description="Low confidence RBIA - manual review required"
            ),
            TrustLadderEntry(
                trust_level=TrustLevel.NO_TRUST,
                trust_score_min=0.0,
                trust_score_max=0.74,
                execution_mode="blocked",
                human_oversight_required=True,
                description="Below threshold - execution blocked"
            )
        ]
    
    def calculate_trust_score(self, node: Dict[str, Any]) -> float:
        """Calculate trust score for a node"""
        node_type = node.get('type', '')
        
        # RBA nodes get full trust
        if node_type in ['rule', 'decision', 'validation']:
            return 1.0
        
        # ML nodes use confidence scores
        if 'ml_' in node_type or node_type == 'ml_node':
            trust_budget = node.get('trust_budget', {})
            confidence = trust_budget.get('confidence_score', 0.5)
            
            # Apply trust decay factors
            model_age_factor = trust_budget.get('model_age_factor', 1.0)
            data_quality_factor = trust_budget.get('data_quality_factor', 1.0)
            
            return confidence * model_age_factor * data_quality_factor
        
        # Default trust for other node types
        return 0.8
    
    def get_trust_level(self, trust_score: float) -> TrustLadderEntry:
        """Get trust level for a given trust score"""
        for entry in self.trust_ladder:
            if entry.trust_score_min <= trust_score <= entry.trust_score_max:
                return entry
        
        # Default to no trust if score doesn't match any range
        return self.trust_ladder[-1]  # NO_TRUST
    
    def encode_trust_attributes(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encode trust ladder attributes into IR nodes"""
        nodes = ir_data.get('nodes', [])
        
        for node in nodes:
            # Calculate trust score
            trust_score = self.calculate_trust_score(node)
            
            # Get trust level
            trust_level_entry = self.get_trust_level(trust_score)
            
            # Add trust attributes to node
            node['trust_ladder'] = {
                'trust_score': trust_score,
                'trust_level': trust_level_entry.trust_level.value,
                'execution_mode': trust_level_entry.execution_mode,
                'human_oversight_required': trust_level_entry.human_oversight_required,
                'description': trust_level_entry.description
            }
            
            # Add threshold gating attributes
            node['threshold_gating'] = {
                'auto_execute_threshold': 0.95,
                'assisted_threshold': 0.85,
                'manual_review_threshold': 0.75,
                'block_threshold': 0.75
            }
        
        return ir_data
    
    def validate_trust_thresholds(self, ir_data: Dict[str, Any]) -> List[str]:
        """Validate trust thresholds across the plan"""
        issues = []
        nodes = ir_data.get('nodes', [])
        
        for node in nodes:
            node_id = node.get('id', 'unknown')
            trust_ladder = node.get('trust_ladder', {})
            trust_score = trust_ladder.get('trust_score', 0.0)
            execution_mode = trust_ladder.get('execution_mode', 'blocked')
            
            # Check for risky auto-execution
            if execution_mode == 'auto_execute' and trust_score < 0.95:
                issues.append(f"Node {node_id}: Auto-execution with trust score {trust_score:.2f} < 0.95")
            
            # Check for ML nodes without proper trust encoding
            if 'ml_' in node.get('type', '') and trust_score == 0.0:
                issues.append(f"ML node {node_id}: Missing trust score")
            
            # Check for blocked nodes in critical paths
            if execution_mode == 'blocked':
                issues.append(f"Node {node_id}: Execution blocked due to low trust ({trust_score:.2f})")
        
        return issues
    
    def get_trust_summary(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get trust summary for the entire plan"""
        nodes = ir_data.get('nodes', [])
        trust_distribution = {level.value: 0 for level in TrustLevel}
        execution_modes = {}
        
        for node in nodes:
            trust_ladder = node.get('trust_ladder', {})
            trust_level = trust_ladder.get('trust_level', 'no_trust')
            execution_mode = trust_ladder.get('execution_mode', 'blocked')
            
            trust_distribution[trust_level] += 1
            execution_modes[execution_mode] = execution_modes.get(execution_mode, 0) + 1
        
        total_nodes = len(nodes)
        auto_executable = execution_modes.get('auto_execute', 0)
        
        return {
            'total_nodes': total_nodes,
            'trust_distribution': trust_distribution,
            'execution_modes': execution_modes,
            'auto_executable_percentage': (auto_executable / total_nodes * 100) if total_nodes > 0 else 0,
            'human_oversight_required': sum([
                execution_modes.get('assisted', 0),
                execution_modes.get('manual_review', 0),
                execution_modes.get('blocked', 0)
            ])
        }

