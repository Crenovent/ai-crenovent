"""
Assisted-Mode Flagging - Task 6.2.58
=====================================

Human-in-loop flagging for low confidence paths
- Flags nodes/paths requiring human assistance
- Drives UI decisions for manual intervention
- IR attribute marking for runtime handling
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AssistedModeReason(Enum):
    LOW_CONFIDENCE = "low_confidence"
    HIGH_RISK = "high_risk"
    POLICY_REQUIRED = "policy_required"
    COMPLEX_DECISION = "complex_decision"
    FALLBACK_TRIGGERED = "fallback_triggered"
    AUTO_EXEC_DENIED = "auto_exec_denied"

@dataclass
class AssistedModeFlag:
    node_id: str
    reason: AssistedModeReason
    confidence_score: Optional[float]
    threshold: float
    message: str
    ui_guidance: str

class AssistedModeFlaggingService:
    """Service for flagging nodes requiring human assistance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.confidence_thresholds = {
            'ml_node': 0.8,
            'decision': 0.9,
            'risk_assessment': 0.85
        }
    
    def analyze_ir_for_assisted_mode(self, ir_data: Dict[str, Any]) -> List[AssistedModeFlag]:
        """Analyze IR and flag nodes needing assistance"""
        flags = []
        nodes = ir_data.get('nodes', [])
        
        for node in nodes:
            node_flags = self._analyze_node(node)
            flags.extend(node_flags)
        
        return flags
    
    def _analyze_node(self, node: Dict[str, Any]) -> List[AssistedModeFlag]:
        """Analyze individual node for assisted mode requirements"""
        flags = []
        node_id = node.get('id', 'unknown')
        node_type = node.get('type', 'unknown')
        
        # Check confidence scores
        trust_budget = node.get('trust_budget', {})
        confidence = trust_budget.get('confidence_score')
        threshold = self.confidence_thresholds.get(node_type, 0.8)
        
        if confidence is not None and confidence < threshold:
            flags.append(AssistedModeFlag(
                node_id=node_id,
                reason=AssistedModeReason.LOW_CONFIDENCE,
                confidence_score=confidence,
                threshold=threshold,
                message=f"Confidence {confidence:.2f} below threshold {threshold}",
                ui_guidance="Show confidence warning and require manual review"
            ))
        
        # Check for high-risk indicators
        policies = node.get('policies', {})
        if policies.get('risk_level') == 'high':
            flags.append(AssistedModeFlag(
                node_id=node_id,
                reason=AssistedModeReason.HIGH_RISK,
                confidence_score=confidence,
                threshold=1.0,
                message="Node marked as high-risk",
                ui_guidance="Display risk warning and require approval"
            ))
        
        # Check for mandatory human review policies
        if policies.get('human_review_required'):
            flags.append(AssistedModeFlag(
                node_id=node_id,
                reason=AssistedModeReason.POLICY_REQUIRED,
                confidence_score=confidence,
                threshold=1.0,
                message="Policy requires human review",
                ui_guidance="Block auto-execution, show review interface"
            ))
        
        return flags
    
    def add_assisted_mode_attributes(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add assisted mode attributes to IR nodes"""
        flags = self.analyze_ir_for_assisted_mode(ir_data)
        
        # Create flag lookup
        flag_lookup = {}
        for flag in flags:
            if flag.node_id not in flag_lookup:
                flag_lookup[flag.node_id] = []
            flag_lookup[flag.node_id].append(flag)
        
        # Add attributes to nodes
        nodes = ir_data.get('nodes', [])
        for node in nodes:
            node_id = node.get('id')
            if node_id in flag_lookup:
                node['assisted_mode_required'] = True
                node['assisted_mode_flags'] = [
                    {
                        'reason': flag.reason.value,
                        'message': flag.message,
                        'ui_guidance': flag.ui_guidance,
                        'confidence_score': flag.confidence_score,
                        'threshold': flag.threshold
                    }
                    for flag in flag_lookup[node_id]
                ]
            else:
                node['assisted_mode_required'] = False
        
        return ir_data

    def flag_auto_exec_denied(
        self,
        node_id: str,
        denial_reason: str,
        policy_name: str = None,
        justification_required: bool = True
    ) -> AssistedModeFlag:
        """Flag node for assisted mode when auto-execution is denied - Task 6.4.39"""
        
        message = f"Automatic execution denied: {denial_reason}"
        if policy_name:
            message += f" (Policy: {policy_name})"
        
        ui_guidance = "Manual approval required. Please provide justification and review the decision before proceeding."
        if justification_required:
            ui_guidance += " A business justification is mandatory for this action."
        
        flag = AssistedModeFlag(
            node_id=node_id,
            reason=AssistedModeReason.AUTO_EXEC_DENIED,
            confidence_score=None,
            threshold=1.0,  # Always requires assistance when denied
            message=message,
            ui_guidance=ui_guidance
        )
        
        self.logger.info(f"Auto-exec denied flag created for node {node_id}: {denial_reason}")
        return flag
