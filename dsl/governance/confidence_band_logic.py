"""
Task 6.4.10: Add confidence band logic (gray zone prompts for Assisted)
======================================================================

Safe human-in-loop assisted UX hook configurable
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConfidenceBand(str, Enum):
    """Confidence bands for decision making"""
    HIGH_CONFIDENCE = "high_confidence"      # Auto-execute zone
    GRAY_ZONE = "gray_zone"                  # Assisted mode zone
    LOW_CONFIDENCE = "low_confidence"        # Block or escalate zone

class ConfidenceAction(str, Enum):
    """Actions based on confidence band"""
    AUTO_EXECUTE = "auto_execute"
    PROMPT_ASSISTED = "prompt_assisted"
    REQUIRE_REVIEW = "require_review"
    BLOCK_EXECUTION = "block_execution"

@dataclass
class ConfidenceBandConfig:
    """Confidence band configuration"""
    high_confidence_threshold: float = 0.9
    gray_zone_lower_threshold: float = 0.7
    gray_zone_upper_threshold: float = 0.9
    low_confidence_threshold: float = 0.7

@dataclass
class ConfidenceBandRequest:
    """Confidence band evaluation request"""
    confidence_score: float
    tenant_id: str
    workflow_id: str
    model_id: str
    operation_context: Dict[str, Any]

@dataclass
class ConfidenceBandResponse:
    """Confidence band evaluation response"""
    confidence_band: ConfidenceBand
    confidence_score: float
    action: ConfidenceAction
    message: str
    ui_prompt: Optional[str] = None
    requires_justification: bool = False
    assisted_mode_config: Optional[Dict[str, Any]] = None

class ConfidenceBandLogic:
    """Confidence band logic service - Task 6.4.10"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default configurations by workflow type
        self.band_configs = {
            "default": ConfidenceBandConfig(
                high_confidence_threshold=0.9,
                gray_zone_lower_threshold=0.7,
                gray_zone_upper_threshold=0.9,
                low_confidence_threshold=0.7
            ),
            "high_risk": ConfidenceBandConfig(
                high_confidence_threshold=0.95,
                gray_zone_lower_threshold=0.8,
                gray_zone_upper_threshold=0.95,
                low_confidence_threshold=0.8
            ),
            "financial": ConfidenceBandConfig(
                high_confidence_threshold=0.92,
                gray_zone_lower_threshold=0.75,
                gray_zone_upper_threshold=0.92,
                low_confidence_threshold=0.75
            ),
            "compliance": ConfidenceBandConfig(
                high_confidence_threshold=0.95,
                gray_zone_lower_threshold=0.85,
                gray_zone_upper_threshold=0.95,
                low_confidence_threshold=0.85
            )
        }
        
        # UI prompts for gray zone
        self.gray_zone_prompts = {
            "default": "The AI confidence is moderate. Would you like to review the decision before proceeding?",
            "high_risk": "This is a high-risk operation with moderate AI confidence. Manual review is recommended.",
            "financial": "Financial impact detected with moderate confidence. Please review the recommendation.",
            "compliance": "Compliance implications with moderate confidence. Regulatory review may be required."
        }
    
    def evaluate_confidence_band(self, request: ConfidenceBandRequest) -> ConfidenceBandResponse:
        """Evaluate confidence band and determine action - Task 6.4.10"""
        
        # Determine workflow type from context
        workflow_type = request.operation_context.get("workflow_type", "default")
        config = self.band_configs.get(workflow_type, self.band_configs["default"])
        
        confidence_score = request.confidence_score
        
        # Determine confidence band
        if confidence_score >= config.high_confidence_threshold:
            confidence_band = ConfidenceBand.HIGH_CONFIDENCE
            action = ConfidenceAction.AUTO_EXECUTE
            message = f"High confidence ({confidence_score:.3f}) - auto-executing"
            ui_prompt = None
            requires_justification = False
            assisted_mode_config = None
            
        elif config.gray_zone_lower_threshold <= confidence_score < config.gray_zone_upper_threshold:
            confidence_band = ConfidenceBand.GRAY_ZONE
            action = ConfidenceAction.PROMPT_ASSISTED
            message = f"Gray zone confidence ({confidence_score:.3f}) - prompting for assistance"
            ui_prompt = self.gray_zone_prompts.get(workflow_type, self.gray_zone_prompts["default"])
            requires_justification = True
            assisted_mode_config = self._get_assisted_mode_config(workflow_type, confidence_score)
            
        else:
            confidence_band = ConfidenceBand.LOW_CONFIDENCE
            action = ConfidenceAction.REQUIRE_REVIEW
            message = f"Low confidence ({confidence_score:.3f}) - requiring manual review"
            ui_prompt = "AI confidence is too low for automatic execution. Manual review required."
            requires_justification = True
            assisted_mode_config = self._get_assisted_mode_config(workflow_type, confidence_score)
        
        response = ConfidenceBandResponse(
            confidence_band=confidence_band,
            confidence_score=confidence_score,
            action=action,
            message=message,
            ui_prompt=ui_prompt,
            requires_justification=requires_justification,
            assisted_mode_config=assisted_mode_config
        )
        
        self.logger.info(f"Confidence band evaluation: {request.workflow_id} -> {confidence_band.value} ({confidence_score:.3f})")
        return response
    
    def _get_assisted_mode_config(self, workflow_type: str, confidence_score: float) -> Dict[str, Any]:
        """Get assisted mode configuration for gray zone"""
        return {
            "show_confidence_score": True,
            "show_explanation": True,
            "require_justification": True,
            "allow_override": confidence_score > 0.5,
            "escalation_required": confidence_score < 0.6,
            "review_timeout_minutes": 30,
            "workflow_type": workflow_type
        }
    
    def is_in_gray_zone(self, confidence_score: float, workflow_type: str = "default") -> bool:
        """Check if confidence score is in gray zone"""
        config = self.band_configs.get(workflow_type, self.band_configs["default"])
        return config.gray_zone_lower_threshold <= confidence_score < config.gray_zone_upper_threshold
    
    def get_gray_zone_prompt(self, workflow_type: str = "default") -> str:
        """Get gray zone UI prompt for workflow type"""
        return self.gray_zone_prompts.get(workflow_type, self.gray_zone_prompts["default"])
    
    def update_band_config(self, workflow_type: str, config: ConfidenceBandConfig):
        """Update confidence band configuration for workflow type"""
        self.band_configs[workflow_type] = config
        self.logger.info(f"Updated confidence band config for {workflow_type}: {config}")

class ConfidenceBandService:
    """Confidence band UX service - Task 6.4.10"""
    
    def __init__(self):
        self.logic = ConfidenceBandLogic()
        self.logger = logging.getLogger(__name__)
    
    def check_confidence_band(
        self,
        confidence_score: float,
        tenant_id: str,
        workflow_id: str,
        model_id: str,
        operation_context: Dict[str, Any] = None
    ) -> ConfidenceBandResponse:
        """Check confidence band and get UX guidance"""
        
        request = ConfidenceBandRequest(
            confidence_score=confidence_score,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            model_id=model_id,
            operation_context=operation_context or {}
        )
        
        return self.logic.evaluate_confidence_band(request)
    
    def should_prompt_for_assistance(
        self,
        confidence_score: float,
        workflow_type: str = "default"
    ) -> bool:
        """Check if should prompt for assisted mode"""
        return self.logic.is_in_gray_zone(confidence_score, workflow_type)
    
    def get_assisted_mode_ui_config(
        self,
        confidence_score: float,
        workflow_type: str = "default"
    ) -> Dict[str, Any]:
        """Get UI configuration for assisted mode"""
        if self.logic.is_in_gray_zone(confidence_score, workflow_type):
            return self.logic._get_assisted_mode_config(workflow_type, confidence_score)
        return {}

