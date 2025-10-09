"""
Task 6.4.7: Add error classes (4xx contract, 5xx infra, policy-deny, timeout, drift/bias)
=========================================================================================

Actionable handling error map that drives messaging
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

class ErrorCategory(str, Enum):
    """Error categories for classification"""
    CONTRACT_4XX = "4xx_contract"
    INFRA_5XX = "5xx_infra"
    POLICY_DENY = "policy_deny"
    TIMEOUT = "timeout"
    DRIFT_BIAS = "drift_bias"

class ErrorClass(str, Enum):
    """Error classes - Task 6.4.7"""
    # 4xx Contract errors
    CONTRACT_VIOLATION = "contract_violation"
    SCHEMA_MISMATCH = "schema_mismatch"
    VALIDATION_FAILED = "validation_failed"
    INVALID_INPUT = "invalid_input"
    
    # 5xx Infrastructure errors
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"
    NETWORK_ERROR = "network_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    
    # Policy denial errors
    ACCESS_DENIED = "access_denied"
    PERMISSION_DENIED = "permission_denied"
    COMPLIANCE_VIOLATION = "compliance_violation"
    GOVERNANCE_BLOCK = "governance_block"
    
    # Timeout errors
    REQUEST_TIMEOUT = "request_timeout"
    PROCESSING_TIMEOUT = "processing_timeout"
    CONNECTION_TIMEOUT = "connection_timeout"
    CIRCUIT_BREAKER_TIMEOUT = "circuit_breaker_timeout"
    
    # Drift/bias errors
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    BIAS_DETECTED = "bias_detected"
    FAIRNESS_VIOLATION = "fairness_violation"

@dataclass
class ErrorDefinition:
    """Error definition with handling information"""
    error_class: ErrorClass
    category: ErrorCategory
    http_code: int
    message_template: str
    actionable_message: str
    retry_allowed: bool = False
    fallback_recommended: bool = True

class ErrorClassificationMap:
    """Error classification map - Task 6.4.7"""
    
    def __init__(self):
        self.error_definitions = {
            # 4xx Contract errors
            ErrorClass.CONTRACT_VIOLATION: ErrorDefinition(
                error_class=ErrorClass.CONTRACT_VIOLATION,
                category=ErrorCategory.CONTRACT_4XX,
                http_code=400,
                message_template="Contract violation: {details}",
                actionable_message="Check input format and contract specification",
                retry_allowed=False
            ),
            ErrorClass.SCHEMA_MISMATCH: ErrorDefinition(
                error_class=ErrorClass.SCHEMA_MISMATCH,
                category=ErrorCategory.CONTRACT_4XX,
                http_code=400,
                message_template="Schema mismatch: {details}",
                actionable_message="Verify input schema matches expected format",
                retry_allowed=False
            ),
            ErrorClass.VALIDATION_FAILED: ErrorDefinition(
                error_class=ErrorClass.VALIDATION_FAILED,
                category=ErrorCategory.CONTRACT_4XX,
                http_code=422,
                message_template="Validation failed: {details}",
                actionable_message="Fix validation errors and retry",
                retry_allowed=False
            ),
            ErrorClass.INVALID_INPUT: ErrorDefinition(
                error_class=ErrorClass.INVALID_INPUT,
                category=ErrorCategory.CONTRACT_4XX,
                http_code=400,
                message_template="Invalid input: {details}",
                actionable_message="Correct input parameters",
                retry_allowed=False
            ),
            
            # 5xx Infrastructure errors
            ErrorClass.SERVICE_UNAVAILABLE: ErrorDefinition(
                error_class=ErrorClass.SERVICE_UNAVAILABLE,
                category=ErrorCategory.INFRA_5XX,
                http_code=503,
                message_template="Service unavailable: {details}",
                actionable_message="Service is temporarily unavailable, fallback activated",
                retry_allowed=True
            ),
            ErrorClass.INTERNAL_ERROR: ErrorDefinition(
                error_class=ErrorClass.INTERNAL_ERROR,
                category=ErrorCategory.INFRA_5XX,
                http_code=500,
                message_template="Internal error: {details}",
                actionable_message="Internal system error, fallback activated",
                retry_allowed=True
            ),
            ErrorClass.NETWORK_ERROR: ErrorDefinition(
                error_class=ErrorClass.NETWORK_ERROR,
                category=ErrorCategory.INFRA_5XX,
                http_code=502,
                message_template="Network error: {details}",
                actionable_message="Network connectivity issue, fallback activated",
                retry_allowed=True
            ),
            ErrorClass.RESOURCE_EXHAUSTED: ErrorDefinition(
                error_class=ErrorClass.RESOURCE_EXHAUSTED,
                category=ErrorCategory.INFRA_5XX,
                http_code=503,
                message_template="Resource exhausted: {details}",
                actionable_message="System resources exhausted, fallback activated",
                retry_allowed=True
            ),
            
            # Policy denial errors
            ErrorClass.ACCESS_DENIED: ErrorDefinition(
                error_class=ErrorClass.ACCESS_DENIED,
                category=ErrorCategory.POLICY_DENY,
                http_code=403,
                message_template="Access denied: {details}",
                actionable_message="Access denied by policy, manual review required",
                retry_allowed=False
            ),
            ErrorClass.PERMISSION_DENIED: ErrorDefinition(
                error_class=ErrorClass.PERMISSION_DENIED,
                category=ErrorCategory.POLICY_DENY,
                http_code=403,
                message_template="Permission denied: {details}",
                actionable_message="Insufficient permissions, contact administrator",
                retry_allowed=False
            ),
            ErrorClass.COMPLIANCE_VIOLATION: ErrorDefinition(
                error_class=ErrorClass.COMPLIANCE_VIOLATION,
                category=ErrorCategory.POLICY_DENY,
                http_code=403,
                message_template="Compliance violation: {details}",
                actionable_message="Operation violates compliance policy",
                retry_allowed=False
            ),
            ErrorClass.GOVERNANCE_BLOCK: ErrorDefinition(
                error_class=ErrorClass.GOVERNANCE_BLOCK,
                category=ErrorCategory.POLICY_DENY,
                http_code=403,
                message_template="Governance block: {details}",
                actionable_message="Blocked by governance policy, approval required",
                retry_allowed=False
            ),
            
            # Timeout errors
            ErrorClass.REQUEST_TIMEOUT: ErrorDefinition(
                error_class=ErrorClass.REQUEST_TIMEOUT,
                category=ErrorCategory.TIMEOUT,
                http_code=408,
                message_template="Request timeout: {details}",
                actionable_message="Request timed out, fallback activated",
                retry_allowed=True
            ),
            ErrorClass.PROCESSING_TIMEOUT: ErrorDefinition(
                error_class=ErrorClass.PROCESSING_TIMEOUT,
                category=ErrorCategory.TIMEOUT,
                http_code=504,
                message_template="Processing timeout: {details}",
                actionable_message="Processing took too long, fallback activated",
                retry_allowed=True
            ),
            ErrorClass.CONNECTION_TIMEOUT: ErrorDefinition(
                error_class=ErrorClass.CONNECTION_TIMEOUT,
                category=ErrorCategory.TIMEOUT,
                http_code=504,
                message_template="Connection timeout: {details}",
                actionable_message="Connection timed out, fallback activated",
                retry_allowed=True
            ),
            ErrorClass.CIRCUIT_BREAKER_TIMEOUT: ErrorDefinition(
                error_class=ErrorClass.CIRCUIT_BREAKER_TIMEOUT,
                category=ErrorCategory.TIMEOUT,
                http_code=503,
                message_template="Circuit breaker timeout: {details}",
                actionable_message="Circuit breaker open, fallback activated",
                retry_allowed=True
            ),
            
            # Drift/bias errors
            ErrorClass.DATA_DRIFT: ErrorDefinition(
                error_class=ErrorClass.DATA_DRIFT,
                category=ErrorCategory.DRIFT_BIAS,
                http_code=422,
                message_template="Data drift detected: {details}",
                actionable_message="Data drift detected, fallback to baseline model",
                retry_allowed=False
            ),
            ErrorClass.CONCEPT_DRIFT: ErrorDefinition(
                error_class=ErrorClass.CONCEPT_DRIFT,
                category=ErrorCategory.DRIFT_BIAS,
                http_code=422,
                message_template="Concept drift detected: {details}",
                actionable_message="Concept drift detected, model retraining required",
                retry_allowed=False
            ),
            ErrorClass.BIAS_DETECTED: ErrorDefinition(
                error_class=ErrorClass.BIAS_DETECTED,
                category=ErrorCategory.DRIFT_BIAS,
                http_code=422,
                message_template="Bias detected: {details}",
                actionable_message="Model bias detected, fallback to unbiased alternative",
                retry_allowed=False
            ),
            ErrorClass.FAIRNESS_VIOLATION: ErrorDefinition(
                error_class=ErrorClass.FAIRNESS_VIOLATION,
                category=ErrorCategory.DRIFT_BIAS,
                http_code=422,
                message_template="Fairness violation: {details}",
                actionable_message="Fairness criteria violated, manual review required",
                retry_allowed=False
            )
        }
    
    def classify_error(self, error_message: str, error_code: Optional[str] = None) -> ErrorClass:
        """Classify error based on message and code"""
        error_message_lower = error_message.lower()
        
        # Simple classification based on keywords
        if "timeout" in error_message_lower:
            return ErrorClass.REQUEST_TIMEOUT
        elif "validation" in error_message_lower or "schema" in error_message_lower:
            return ErrorClass.VALIDATION_FAILED
        elif "access denied" in error_message_lower or "permission" in error_message_lower:
            return ErrorClass.ACCESS_DENIED
        elif "drift" in error_message_lower:
            return ErrorClass.DATA_DRIFT
        elif "bias" in error_message_lower:
            return ErrorClass.BIAS_DETECTED
        elif "service unavailable" in error_message_lower:
            return ErrorClass.SERVICE_UNAVAILABLE
        else:
            return ErrorClass.INTERNAL_ERROR
    
    def get_error_definition(self, error_class: ErrorClass) -> ErrorDefinition:
        """Get error definition for error class"""
        return self.error_definitions.get(error_class)
    
    def get_actionable_message(self, error_class: ErrorClass, details: str = "") -> str:
        """Get actionable message for error class"""
        definition = self.get_error_definition(error_class)
        if definition:
            return definition.message_template.format(details=details)
        return f"Unknown error: {details}"
