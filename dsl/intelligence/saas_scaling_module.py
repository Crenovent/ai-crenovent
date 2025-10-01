"""
SaaS Scaling Intelligence Module
===============================
Small module for SaaS-specific scaling policies and decisions.
Integrates with existing intelligence/sla_monitoring_system.py

Tasks: 7.3.2, 7.3.6, 7.3.7 - SaaS scaling strategies and SLA-aware policies
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SaaSTenantTier(Enum):
    T0_REGULATED = "T0"  # Regulated SaaS customers
    T1_ENTERPRISE = "T1"  # Enterprise SaaS customers  
    T2_MIDMARKET = "T2"   # Mid-market SaaS customers

@dataclass
class SaaSScalingDecision:
    """SaaS scaling decision with business context"""
    action: str  # scale_up, scale_down, no_action
    reason: str
    tenant_tier: SaaSTenantTier
    saas_metrics: Dict[str, float]
    recommended_instances: int
    cost_impact: float
    business_impact: str
    urgency: str

class SaaSScalingIntelligence:
    """
    SaaS-focused scaling intelligence module
    Integrates with existing SLA monitoring system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # SaaS-specific scaling thresholds by tenant tier
        self.saas_scaling_thresholds = {
            SaaSTenantTier.T0_REGULATED: {
                'cpu_scale_up': 0.6, 'memory_scale_up': 0.7, 'response_time_max': 100,
                'min_instances': 3, 'max_instances': 50, 'cost_priority': False
            },
            SaaSTenantTier.T1_ENTERPRISE: {
                'cpu_scale_up': 0.75, 'memory_scale_up': 0.8, 'response_time_max': 200,
                'min_instances': 2, 'max_instances': 30, 'cost_priority': True
            },
            SaaSTenantTier.T2_MIDMARKET: {
                'cpu_scale_up': 0.85, 'memory_scale_up': 0.85, 'response_time_max': 500,
                'min_instances': 1, 'max_instances': 15, 'cost_priority': True
            }
        }
    
    def evaluate_saas_scaling_decision(self, tenant_id: int, tenant_tier: SaaSTenantTier,
                                     performance_metrics: Dict[str, Any],
                                     saas_business_metrics: Dict[str, Any]) -> SaaSScalingDecision:
        """
        Evaluate scaling decision with SaaS business context
        
        Args:
            tenant_id: Multi-tenant ID
            tenant_tier: T0/T1/T2 tier
            performance_metrics: CPU, memory, response time, etc.
            saas_business_metrics: ARR, customer count, churn rate, etc.
        """
        try:
            thresholds = self.saas_scaling_thresholds[tenant_tier]
            
            # Check performance triggers
            cpu_util = performance_metrics.get('cpu_utilization', 0)
            memory_util = performance_metrics.get('memory_utilization', 0)
            response_time = performance_metrics.get('response_time_p95', 0)
            
            # Check SaaS business triggers
            customer_count = saas_business_metrics.get('customer_count', 0)
            arr_growth = saas_business_metrics.get('arr_growth_rate', 0)
            churn_rate = saas_business_metrics.get('churn_rate', 0)
            
            # Determine scaling action
            if (cpu_util > thresholds['cpu_scale_up'] or 
                memory_util > thresholds['memory_scale_up'] or
                response_time > thresholds['response_time_max']):
                
                action = "scale_up"
                reason = f"Performance thresholds exceeded (CPU: {cpu_util:.2f}, Memory: {memory_util:.2f}, RT: {response_time}ms)"
                urgency = "high" if tenant_tier == SaaSTenantTier.T0_REGULATED else "medium"
                recommended_instances = min(self._calculate_scale_up_instances(performance_metrics, saas_business_metrics), 
                                          thresholds['max_instances'])
                
            elif (cpu_util < 0.3 and memory_util < 0.4 and 
                  thresholds['cost_priority'] and customer_count < 100):
                
                action = "scale_down"
                reason = f"Low utilization with cost optimization enabled (CPU: {cpu_util:.2f}, Memory: {memory_util:.2f})"
                urgency = "low"
                recommended_instances = max(self._calculate_scale_down_instances(), thresholds['min_instances'])
                
            else:
                action = "no_action"
                reason = "Performance within acceptable thresholds"
                urgency = "none"
                recommended_instances = 0
            
            # Calculate business impact
            business_impact = self._assess_business_impact(action, saas_business_metrics, tenant_tier)
            cost_impact = self._estimate_cost_impact(action, recommended_instances, tenant_tier)
            
            return SaaSScalingDecision(
                action=action,
                reason=reason,
                tenant_tier=tenant_tier,
                saas_metrics={
                    'customer_count': customer_count,
                    'arr_growth': arr_growth,
                    'churn_rate': churn_rate,
                    'cpu_utilization': cpu_util,
                    'memory_utilization': memory_util
                },
                recommended_instances=recommended_instances,
                cost_impact=cost_impact,
                business_impact=business_impact,
                urgency=urgency
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to evaluate SaaS scaling decision: {e}")
            return SaaSScalingDecision(
                action="no_action",
                reason=f"Evaluation error: {str(e)}",
                tenant_tier=tenant_tier,
                saas_metrics={},
                recommended_instances=0,
                cost_impact=0.0,
                business_impact="unknown",
                urgency="none"
            )
    
    def _calculate_scale_up_instances(self, performance_metrics: Dict[str, Any], 
                                    saas_metrics: Dict[str, Any]) -> int:
        """Calculate recommended scale-up instances based on SaaS context"""
        base_scale = 2
        
        # Scale based on customer count
        customer_factor = max(1.0, saas_metrics.get('customer_count', 0) / 1000)
        
        # Scale based on ARR growth
        arr_growth = saas_metrics.get('arr_growth_rate', 0)
        growth_factor = 1.0 + (arr_growth * 0.5)  # 50% scaling factor for ARR growth
        
        return int(base_scale * customer_factor * growth_factor)
    
    def _calculate_scale_down_instances(self) -> int:
        """Calculate scale-down instances (conservative approach)"""
        return 1  # Scale down by 1 instance at a time
    
    def _assess_business_impact(self, action: str, saas_metrics: Dict[str, Any], 
                              tenant_tier: SaaSTenantTier) -> str:
        """Assess business impact of scaling decision"""
        if action == "scale_up":
            if tenant_tier == SaaSTenantTier.T0_REGULATED:
                return "critical_performance_protection"
            elif saas_metrics.get('churn_rate', 0) > 0.1:
                return "churn_prevention_scaling"
            else:
                return "growth_accommodation"
        elif action == "scale_down":
            return "cost_optimization"
        else:
            return "stable_operations"
    
    def _estimate_cost_impact(self, action: str, instances: int, 
                            tenant_tier: SaaSTenantTier) -> float:
        """Estimate cost impact of scaling decision"""
        # Mock cost calculation (would integrate with actual cloud pricing)
        base_cost_per_instance = {
            SaaSTenantTier.T0_REGULATED: 100.0,  # Premium instances
            SaaSTenantTier.T1_ENTERPRISE: 50.0,   # Standard instances
            SaaSTenantTier.T2_MIDMARKET: 25.0     # Basic instances
        }
        
        if action == "scale_up":
            return instances * base_cost_per_instance[tenant_tier]
        elif action == "scale_down":
            return -instances * base_cost_per_instance[tenant_tier]  # Cost savings
        else:
            return 0.0

# Global instance for integration
saas_scaling_intelligence = SaaSScalingIntelligence()

def get_saas_scaling_intelligence():
    """Get SaaS scaling intelligence instance"""
    return saas_scaling_intelligence
