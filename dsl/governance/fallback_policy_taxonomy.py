"""
Task 6.4.1: Define fallback policy taxonomy
===========================================

Uniform trigger vocabulary for fallback policy taxonomy covering:
- SLA breach
- Infrastructure error  
- Contract mismatch
- Trust below threshold
- Drift/bias breach
- Residency/ABAC denial
"""

from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass

class FallbackPolicyTrigger(str, Enum):
    """Fallback policy taxonomy - Task 6.4.1"""
    # SLA breach triggers
    SLA_BREACH = "sla_breach"
    SLA_LATENCY_BREACH = "sla_latency_breach"
    SLA_AVAILABILITY_BREACH = "sla_availability_breach"
    SLA_ERROR_BUDGET_BREACH = "sla_error_budget_breach"
    
    # Infrastructure error triggers
    INFRA_ERROR = "infra_error"
    INFRA_SERVICE_DOWN = "infra_service_down"
    INFRA_NETWORK_ERROR = "infra_network_error"
    INFRA_RESOURCE_EXHAUSTION = "infra_resource_exhaustion"
    
    # Contract mismatch triggers
    CONTRACT_MISMATCH = "contract_mismatch"
    CONTRACT_SCHEMA_VIOLATION = "contract_schema_violation"
    CONTRACT_VERSION_MISMATCH = "contract_version_mismatch"
    CONTRACT_VALIDATION_FAILED = "contract_validation_failed"
    
    # Trust below threshold triggers
    TRUST_BELOW_THRESHOLD = "trust_below_threshold"
    TRUST_SCORE_LOW = "trust_score_low"
    TRUST_DEGRADATION = "trust_degradation"
    TRUST_INSUFFICIENT = "trust_insufficient"
    
    # Drift/bias breach triggers
    DRIFT_BIAS_BREACH = "drift_bias_breach"
    DATA_DRIFT_DETECTED = "data_drift_detected"
    CONCEPT_DRIFT_DETECTED = "concept_drift_detected"
    BIAS_VIOLATION = "bias_violation"
    FAIRNESS_BREACH = "fairness_breach"
    
    # Residency/ABAC denial triggers
    RESIDENCY_ABAC_DENIAL = "residency_abac_denial"
    RESIDENCY_VIOLATION = "residency_violation"
    ABAC_ACCESS_DENIED = "abac_access_denied"
    RBAC_PERMISSION_DENIED = "rbac_permission_denied"
    GEO_COMPLIANCE_VIOLATION = "geo_compliance_violation"

@dataclass
class FallbackPolicyRule:
    """Fallback policy rule definition"""
    trigger: FallbackPolicyTrigger
    condition: str
    action: str
    priority: int = 1
    enabled: bool = True

class FallbackPolicyTaxonomy:
    """Fallback policy taxonomy manager - Task 6.4.1"""
    
    def __init__(self):
        self.trigger_categories = {
            "sla": [
                FallbackPolicyTrigger.SLA_BREACH,
                FallbackPolicyTrigger.SLA_LATENCY_BREACH,
                FallbackPolicyTrigger.SLA_AVAILABILITY_BREACH,
                FallbackPolicyTrigger.SLA_ERROR_BUDGET_BREACH
            ],
            "infrastructure": [
                FallbackPolicyTrigger.INFRA_ERROR,
                FallbackPolicyTrigger.INFRA_SERVICE_DOWN,
                FallbackPolicyTrigger.INFRA_NETWORK_ERROR,
                FallbackPolicyTrigger.INFRA_RESOURCE_EXHAUSTION
            ],
            "contract": [
                FallbackPolicyTrigger.CONTRACT_MISMATCH,
                FallbackPolicyTrigger.CONTRACT_SCHEMA_VIOLATION,
                FallbackPolicyTrigger.CONTRACT_VERSION_MISMATCH,
                FallbackPolicyTrigger.CONTRACT_VALIDATION_FAILED
            ],
            "trust": [
                FallbackPolicyTrigger.TRUST_BELOW_THRESHOLD,
                FallbackPolicyTrigger.TRUST_SCORE_LOW,
                FallbackPolicyTrigger.TRUST_DEGRADATION,
                FallbackPolicyTrigger.TRUST_INSUFFICIENT
            ],
            "drift_bias": [
                FallbackPolicyTrigger.DRIFT_BIAS_BREACH,
                FallbackPolicyTrigger.DATA_DRIFT_DETECTED,
                FallbackPolicyTrigger.CONCEPT_DRIFT_DETECTED,
                FallbackPolicyTrigger.BIAS_VIOLATION,
                FallbackPolicyTrigger.FAIRNESS_BREACH
            ],
            "residency_access": [
                FallbackPolicyTrigger.RESIDENCY_ABAC_DENIAL,
                FallbackPolicyTrigger.RESIDENCY_VIOLATION,
                FallbackPolicyTrigger.ABAC_ACCESS_DENIED,
                FallbackPolicyTrigger.RBAC_PERMISSION_DENIED,
                FallbackPolicyTrigger.GEO_COMPLIANCE_VIOLATION
            ]
        }
    
    def get_triggers_by_category(self, category: str) -> List[FallbackPolicyTrigger]:
        """Get triggers by category"""
        return self.trigger_categories.get(category, [])
    
    def get_all_triggers(self) -> List[FallbackPolicyTrigger]:
        """Get all triggers"""
        return list(FallbackPolicyTrigger)
    
    def is_valid_trigger(self, trigger: str) -> bool:
        """Check if trigger is valid"""
        try:
            FallbackPolicyTrigger(trigger)
            return True
        except ValueError:
            return False

