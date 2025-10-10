"""
Task 6.4.9: Implement trust threshold guard (min_trust to auto-execute)
======================================================================

Risk gating governance service - Else → Assisted
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ExecutionMode(str, Enum):
    """Execution modes based on trust score"""
    AUTO_EXECUTE = "auto_execute"
    ASSISTED = "assisted"
    BLOCKED = "blocked"

class TrustLevel(str, Enum):
    """Trust levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"

@dataclass
class TrustThresholds:
    """Trust threshold configuration"""
    min_trust_auto_execute: float = 0.9
    min_trust_assisted: float = 0.7
    min_trust_block: float = 0.5
    
@dataclass
class TrustGuardRequest:
    """Trust guard evaluation request"""
    trust_score: float
    tenant_id: str
    workflow_id: str
    operation_type: str
    risk_level: str = "medium"

@dataclass
class TrustGuardResponse:
    """Trust guard evaluation response"""
    execution_mode: ExecutionMode
    trust_score: float
    trust_level: TrustLevel
    allowed: bool
    message: str
    requires_justification: bool = False

class TrustThresholdGuard:
    """Trust threshold guard service - Task 6.4.9"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default thresholds by risk level
        self.risk_thresholds = {
            "low": TrustThresholds(
                min_trust_auto_execute=0.8,
                min_trust_assisted=0.6,
                min_trust_block=0.4
            ),
            "medium": TrustThresholds(
                min_trust_auto_execute=0.9,
                min_trust_assisted=0.7,
                min_trust_block=0.5
            ),
            "high": TrustThresholds(
                min_trust_auto_execute=0.95,
                min_trust_assisted=0.8,
                min_trust_block=0.6
            ),
            "critical": TrustThresholds(
                min_trust_auto_execute=0.98,
                min_trust_assisted=0.9,
                min_trust_block=0.7
            )
        }
    
    def evaluate_trust_guard(self, request: TrustGuardRequest) -> TrustGuardResponse:
        """Evaluate trust threshold guard - Task 6.4.9"""
        
        thresholds = self.risk_thresholds.get(request.risk_level, self.risk_thresholds["medium"])
        trust_score = request.trust_score
        
        # Determine execution mode based on trust score
        if trust_score >= thresholds.min_trust_auto_execute:
            execution_mode = ExecutionMode.AUTO_EXECUTE
            trust_level = TrustLevel.HIGH
            allowed = True
            message = f"Trust score {trust_score:.3f} allows auto-execution"
            requires_justification = False
            
        elif trust_score >= thresholds.min_trust_assisted:
            execution_mode = ExecutionMode.ASSISTED
            trust_level = TrustLevel.MEDIUM
            allowed = True
            message = f"Trust score {trust_score:.3f} requires assisted mode"
            requires_justification = True
            
        elif trust_score >= thresholds.min_trust_block:
            execution_mode = ExecutionMode.ASSISTED
            trust_level = TrustLevel.LOW
            allowed = True
            message = f"Trust score {trust_score:.3f} requires assisted mode with justification"
            requires_justification = True
            
        else:
            execution_mode = ExecutionMode.BLOCKED
            trust_level = TrustLevel.CRITICAL
            allowed = False
            message = f"Trust score {trust_score:.3f} too low - operation blocked"
            requires_justification = True
        
        response = TrustGuardResponse(
            execution_mode=execution_mode,
            trust_score=trust_score,
            trust_level=trust_level,
            allowed=allowed,
            message=message,
            requires_justification=requires_justification
        )
        
        self.logger.info(f"Trust guard evaluation: {request.workflow_id} -> {execution_mode.value} (trust: {trust_score:.3f})")
        return response
    
    def get_trust_thresholds(self, risk_level: str) -> TrustThresholds:
        """Get trust thresholds for risk level"""
        return self.risk_thresholds.get(risk_level, self.risk_thresholds["medium"])
    
    def update_trust_thresholds(self, risk_level: str, thresholds: TrustThresholds):
        """Update trust thresholds for risk level"""
        self.risk_thresholds[risk_level] = thresholds
        self.logger.info(f"Updated trust thresholds for {risk_level}: {thresholds}")

class TrustThresholdGuardService:
    """Trust threshold guard governance service - Task 6.4.9"""
    
    def __init__(self):
        self.guard = TrustThresholdGuard()
        self.logger = logging.getLogger(__name__)
    
    def check_trust_gate(
        self,
        trust_score: float,
        tenant_id: str,
        workflow_id: str,
        operation_type: str,
        risk_level: str = "medium"
    ) -> TrustGuardResponse:
        """Check trust gate for operation"""
        
        request = TrustGuardRequest(
            trust_score=trust_score,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            operation_type=operation_type,
            risk_level=risk_level
        )
        
        return self.guard.evaluate_trust_guard(request)
    
    def is_auto_execution_allowed(
        self,
        trust_score: float,
        risk_level: str = "medium"
    ) -> bool:
        """Check if auto-execution is allowed based on trust score"""
        thresholds = self.guard.get_trust_thresholds(risk_level)
        return trust_score >= thresholds.min_trust_auto_execute
    
    def requires_assisted_mode(
        self,
        trust_score: float,
        risk_level: str = "medium"
    ) -> bool:
        """Check if assisted mode is required"""
        thresholds = self.guard.get_trust_thresholds(risk_level)
        return (thresholds.min_trust_assisted <= trust_score < thresholds.min_trust_auto_execute)
    
    def is_operation_blocked(
        self,
        trust_score: float,
        risk_level: str = "medium"
    ) -> bool:
        """Check if operation should be blocked"""
        thresholds = self.guard.get_trust_thresholds(risk_level)
        return trust_score < thresholds.min_trust_block

