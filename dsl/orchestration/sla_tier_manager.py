"""
SLA Tier Manager - Task 6.1.23
================================
Connect SLA tiers (T0, T1, T2) to ML node execution with QoS guarantees
"""

import logging
from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class SLATier(str, Enum):
    """SLA tier levels"""
    T0_REGULATED = "T0"  # Highest priority - regulated industries
    T1_ENTERPRISE = "T1"  # Enterprise customers
    T2_MIDMARKET = "T2"  # Mid-market customers


@dataclass
class SLASpec:
    """SLA specifications per tier"""
    tier: SLATier
    max_latency_ms: int
    availability_sla: float  # 0.0-1.0
    max_concurrency: int
    priority_weight: int
    cost_multiplier: float


class SLATierManager:
    """Manage SLA tiers for ML node execution"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sla_specs = {
            SLATier.T0_REGULATED: SLASpec(
                tier=SLATier.T0_REGULATED,
                max_latency_ms=500,
                availability_sla=0.999,  # 99.9%
                max_concurrency=1000,
                priority_weight=10,
                cost_multiplier=3.0
            ),
            SLATier.T1_ENTERPRISE: SLASpec(
                tier=SLATier.T1_ENTERPRISE,
                max_latency_ms=1000,
                availability_sla=0.99,  # 99%
                max_concurrency=500,
                priority_weight=5,
                cost_multiplier=1.5
            ),
            SLATier.T2_MIDMARKET: SLASpec(
                tier=SLATier.T2_MIDMARKET,
                max_latency_ms=2000,
                availability_sla=0.95,  # 95%
                max_concurrency=100,
                priority_weight=1,
                cost_multiplier=1.0
            )
        }
        # Tenant -> Tier mapping
        self.tenant_tiers: Dict[str, SLATier] = {}
    
    def assign_tier(self, tenant_id: str, tier: SLATier):
        """Assign SLA tier to tenant"""
        self.tenant_tiers[tenant_id] = tier
        self.logger.info(f"Assigned tier {tier.value} to tenant {tenant_id}")
    
    def get_tier(self, tenant_id: str) -> SLATier:
        """Get SLA tier for tenant"""
        return self.tenant_tiers.get(tenant_id, SLATier.T2_MIDMARKET)
    
    def get_sla_spec(self, tenant_id: str) -> SLASpec:
        """Get SLA specifications for tenant"""
        tier = self.get_tier(tenant_id)
        return self.sla_specs[tier]
    
    def should_throttle(self, tenant_id: str, current_load: int) -> bool:
        """Check if tenant should be throttled"""
        spec = self.get_sla_spec(tenant_id)
        return current_load >= spec.max_concurrency


# Singleton
_sla_manager: Optional[SLATierManager] = None

def get_sla_tier_manager() -> SLATierManager:
    global _sla_manager
    if _sla_manager is None:
        _sla_manager = SLATierManager()
    return _sla_manager

