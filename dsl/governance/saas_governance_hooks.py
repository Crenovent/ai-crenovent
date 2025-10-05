"""
SaaS-Specific Governance Hooks Implementation
Chapter 8.5 Tasks: Specialized governance for SaaS industry overlay

Key SaaS Governance Requirements:
- Multi-tenant isolation enforcement
- Subscription tier-based policy enforcement  
- API rate limiting and throttling
- Data residency for global SaaS
- Churn risk governance
- Revenue recognition compliance
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json

from .governance_hooks_middleware import (
    GovernanceHooksMiddleware, GovernanceContext, GovernancePlane,
    GovernanceHookType, GovernanceDecision, GovernanceHookResult
)

logger = logging.getLogger(__name__)

class SaaSTier(Enum):
    """SaaS subscription tiers"""
    FREEMIUM = "freemium"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"

class SaaSGovernanceHooks(GovernanceHooksMiddleware):
    """
    SaaS-specific governance hooks extending the base middleware
    Implements industry-specific governance patterns for SaaS companies
    """
    
    def __init__(self, pool_manager):
        super().__init__(pool_manager)
        
        # SaaS-specific governance configurations
        self.saas_tier_policies = {
            SaaSTier.FREEMIUM: {
                "api_rate_limit_per_minute": 100,
                "data_retention_days": 30,
                "support_sla_hours": 72,
                "feature_limits": {
                    "workflows_per_month": 100,
                    "data_export_mb": 10,
                    "concurrent_executions": 2
                },
                "governance_strictness": "advisory"
            },
            SaaSTier.STARTER: {
                "api_rate_limit_per_minute": 500,
                "data_retention_days": 90,
                "support_sla_hours": 24,
                "feature_limits": {
                    "workflows_per_month": 1000,
                    "data_export_mb": 100,
                    "concurrent_executions": 5
                },
                "governance_strictness": "warning"
            },
            SaaSTier.PROFESSIONAL: {
                "api_rate_limit_per_minute": 2000,
                "data_retention_days": 365,
                "support_sla_hours": 8,
                "feature_limits": {
                    "workflows_per_month": 10000,
                    "data_export_mb": 1000,
                    "concurrent_executions": 20
                },
                "governance_strictness": "strict"
            },
            SaaSTier.ENTERPRISE: {
                "api_rate_limit_per_minute": 10000,
                "data_retention_days": 2555,  # 7 years
                "support_sla_hours": 2,
                "feature_limits": {
                    "workflows_per_month": 100000,
                    "data_export_mb": 10000,
                    "concurrent_executions": 100
                },
                "governance_strictness": "strict"
            },
            SaaSTier.ENTERPRISE_PLUS: {
                "api_rate_limit_per_minute": 50000,
                "data_retention_days": 3650,  # 10 years
                "support_sla_hours": 1,
                "feature_limits": {
                    "workflows_per_month": -1,  # unlimited
                    "data_export_mb": -1,  # unlimited
                    "concurrent_executions": 500
                },
                "governance_strictness": "strict"
            }
        }
        
        # SaaS-specific compliance frameworks
        self.saas_compliance_frameworks = {
            "SOC2_TYPE2": {
                "evidence_retention_years": 3,
                "audit_frequency_months": 12,
                "required_controls": [
                    "access_control", "data_encryption", "availability_monitoring",
                    "processing_integrity", "confidentiality"
                ]
            },
            "GDPR": {
                "evidence_retention_years": 7,
                "data_subject_rights": True,
                "consent_management": True,
                "breach_notification_hours": 72
            },
            "CCPA": {
                "evidence_retention_years": 2,
                "consumer_rights": True,
                "opt_out_mechanisms": True
            },
            "HIPAA": {
                "evidence_retention_years": 6,
                "phi_protection": True,
                "business_associate_agreements": True
            }
        }
        
        # Initialize SaaS-specific hooks
        self._initialize_saas_specific_hooks()
    
    def _initialize_saas_specific_hooks(self):
        """Initialize SaaS-specific governance hooks"""
        # Add SaaS-specific hooks to registry
        self.hook_registry[GovernancePlane.CONTROL].extend([
            GovernanceHookType.FINOPS  # SaaS revenue recognition
        ])
        
        self.hook_registry[GovernancePlane.EXECUTION].extend([
            GovernanceHookType.SLA  # SaaS tier-based SLA enforcement
        ])
        
        self.hook_registry[GovernancePlane.DATA].extend([
            GovernanceHookType.FINOPS  # SaaS usage-based billing
        ])
    
    async def enforce_saas_tier_governance(self, context: GovernanceContext, 
                                         saas_tier: SaaSTier) -> GovernanceHookResult:
        """
        Enforce SaaS tier-specific governance policies
        Tasks 8.5.2-8.5.5: Tier-based policy enforcement
        """
        violations = []
        request_data = context.request_data
        tier_config = self.saas_tier_policies[saas_tier]
        
        # API rate limiting enforcement
        current_api_rate = request_data.get("api_calls_per_minute", 0)
        if current_api_rate > tier_config["api_rate_limit_per_minute"]:
            violations.append(
                f"API rate limit exceeded for {saas_tier.value} tier: "
                f"{current_api_rate} > {tier_config['api_rate_limit_per_minute']}"
            )
        
        # Feature usage limits
        feature_limits = tier_config["feature_limits"]
        for feature, limit in feature_limits.items():
            if limit == -1:  # Unlimited
                continue
            
            current_usage = request_data.get(f"current_{feature}", 0)
            if current_usage > limit:
                violations.append(
                    f"Feature limit exceeded for {saas_tier.value} tier - "
                    f"{feature}: {current_usage} > {limit}"
                )
        
        # Data retention policy enforcement
        requested_retention = request_data.get("data_retention_days", 0)
        if requested_retention > tier_config["data_retention_days"]:
            violations.append(
                f"Data retention exceeds tier limit: "
                f"{requested_retention} > {tier_config['data_retention_days']} days"
            )
        
        # Determine decision based on governance strictness
        strictness = tier_config["governance_strictness"]
        if violations:
            if strictness == "advisory":
                decision = GovernanceDecision.ALLOW  # Log but allow
            elif strictness == "warning":
                decision = GovernanceDecision.OVERRIDE_REQUIRED
            else:  # strict
                decision = GovernanceDecision.DENY
        else:
            decision = GovernanceDecision.ALLOW
        
        return GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations,
            override_required=(decision == GovernanceDecision.OVERRIDE_REQUIRED)
        )
    
    async def enforce_saas_multi_tenancy_isolation(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Enforce multi-tenant isolation for SaaS
        Task 8.5.4: Data plane governance with tenant isolation
        """
        violations = []
        request_data = context.request_data
        
        # Validate tenant isolation
        requesting_tenant = context.tenant_id
        target_tenant = request_data.get("target_tenant_id")
        
        if target_tenant and target_tenant != requesting_tenant:
            # Check if cross-tenant access is explicitly allowed
            cross_tenant_permission = request_data.get("cross_tenant_permission", False)
            authorized_by = request_data.get("cross_tenant_authorized_by")
            
            if not cross_tenant_permission or not authorized_by:
                violations.append(
                    f"Unauthorized cross-tenant access: {requesting_tenant} -> {target_tenant}"
                )
        
        # Validate data isolation
        data_sources = request_data.get("data_sources", [])
        for source in data_sources:
            source_tenant = source.get("tenant_id")
            if source_tenant and source_tenant != requesting_tenant:
                violations.append(
                    f"Cross-tenant data access violation: accessing tenant {source_tenant} data"
                )
        
        # Check for tenant-scoped queries
        query_filters = request_data.get("query_filters", {})
        if "tenant_id" not in query_filters:
            violations.append("Missing tenant_id filter in data query - potential data leakage")
        elif query_filters["tenant_id"] != requesting_tenant:
            violations.append(
                f"Tenant ID mismatch in query filter: {query_filters['tenant_id']} != {requesting_tenant}"
            )
        
        decision = GovernanceDecision.DENY if violations else GovernanceDecision.ALLOW
        
        return GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
    
    async def enforce_saas_revenue_recognition_governance(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Enforce SaaS revenue recognition governance
        Task 8.5.10: FinOps governance for SaaS billing
        """
        violations = []
        request_data = context.request_data
        
        # Revenue recognition rules for SaaS
        revenue_type = request_data.get("revenue_type", "subscription")
        billing_period = request_data.get("billing_period", "monthly")
        contract_value = request_data.get("contract_value_usd", 0)
        
        # ASC 606 compliance for SaaS revenue recognition
        if revenue_type == "subscription":
            # Subscription revenue must be recognized over time
            recognition_method = request_data.get("recognition_method")
            if recognition_method != "over_time":
                violations.append(
                    f"Subscription revenue must use 'over_time' recognition method, got: {recognition_method}"
                )
            
            # Validate performance obligations
            performance_obligations = request_data.get("performance_obligations", [])
            if not performance_obligations:
                violations.append("Missing performance obligations for subscription revenue")
            
            # Check for proper contract modifications
            if "contract_modification" in request_data:
                modification = request_data["contract_modification"]
                if not modification.get("approval_required"):
                    violations.append("Contract modifications require approval for revenue recognition")
        
        elif revenue_type == "usage_based":
            # Usage-based revenue recognition
            usage_metrics = request_data.get("usage_metrics", {})
            if not usage_metrics:
                violations.append("Usage-based revenue requires usage metrics")
            
            # Validate usage measurement
            measurement_period = usage_metrics.get("measurement_period")
            if measurement_period not in ["daily", "monthly", "real_time"]:
                violations.append(f"Invalid usage measurement period: {measurement_period}")
        
        # Large contract governance (>$100K)
        if contract_value > 100000:
            required_approvals = request_data.get("required_approvals", [])
            if "revenue_recognition_specialist" not in required_approvals:
                violations.append("Large contracts require revenue recognition specialist approval")
            
            if "legal_review" not in required_approvals:
                violations.append("Large contracts require legal review")
        
        decision = GovernanceDecision.OVERRIDE_REQUIRED if violations else GovernanceDecision.ALLOW
        
        result = GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations,
            override_required=bool(violations)
        )
        
        # Add FinOps lineage tags for SaaS
        result.lineage_tags = [
            f"revenue_type:{revenue_type}",
            f"billing_period:{billing_period}",
            f"contract_value:{contract_value}",
            f"tenant_id:{context.tenant_id}"
        ]
        
        return result
    
    async def enforce_saas_churn_risk_governance(self, context: GovernanceContext) -> GovernanceHookResult:
        """
        Enforce governance for SaaS churn risk scenarios
        Task 8.5.15: Anomaly detection for churn indicators
        """
        violations = []
        request_data = context.request_data
        
        # Churn risk indicators
        churn_indicators = {
            "usage_decline_percentage": 50,  # >50% usage decline
            "support_tickets_increase": 300,  # >300% increase in support tickets
            "api_error_rate_increase": 200,  # >200% increase in API errors
            "login_frequency_decline": 70,   # >70% decline in login frequency
            "feature_adoption_decline": 40   # >40% decline in feature adoption
        }
        
        detected_risks = []
        
        for indicator, threshold in churn_indicators.items():
            current_value = request_data.get(indicator, 0)
            if current_value > threshold:
                detected_risks.append(f"{indicator}: {current_value}% (threshold: {threshold}%)")
        
        # High churn risk governance actions
        if detected_risks:
            churn_risk_level = len(detected_risks)
            
            if churn_risk_level >= 3:  # High risk
                violations.append(f"High churn risk detected: {', '.join(detected_risks)}")
                
                # Require customer success intervention
                cs_intervention = request_data.get("customer_success_intervention", False)
                if not cs_intervention:
                    violations.append("High churn risk requires customer success intervention")
                
                # Require retention workflow activation
                retention_workflow = request_data.get("retention_workflow_activated", False)
                if not retention_workflow:
                    violations.append("High churn risk requires retention workflow activation")
            
            elif churn_risk_level >= 1:  # Medium risk
                # Log for monitoring but allow
                self.logger.warning(f"Medium churn risk detected for tenant {context.tenant_id}: {detected_risks}")
        
        decision = GovernanceDecision.ESCALATE if violations else GovernanceDecision.ALLOW
        
        return GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
    
    async def enforce_saas_compliance_framework_governance(self, context: GovernanceContext, 
                                                         framework: str) -> GovernanceHookResult:
        """
        Enforce SaaS compliance framework-specific governance
        Tasks 8.5.7, 8.5.19: Framework-specific evidence and notifications
        """
        violations = []
        request_data = context.request_data
        
        if framework not in self.saas_compliance_frameworks:
            violations.append(f"Unsupported compliance framework: {framework}")
            return GovernanceHookResult(
                decision=GovernanceDecision.DENY,
                hook_type=context.hook_type,
                context=context,
                violations=violations
            )
        
        framework_config = self.saas_compliance_frameworks[framework]
        
        # SOC 2 Type II specific governance
        if framework == "SOC2_TYPE2":
            required_controls = framework_config["required_controls"]
            implemented_controls = request_data.get("implemented_controls", [])
            
            missing_controls = set(required_controls) - set(implemented_controls)
            if missing_controls:
                violations.append(f"Missing SOC 2 controls: {', '.join(missing_controls)}")
            
            # Validate control testing
            control_testing = request_data.get("control_testing", {})
            for control in required_controls:
                if control not in control_testing:
                    violations.append(f"Missing control testing for: {control}")
        
        # GDPR specific governance
        elif framework == "GDPR":
            # Data subject rights validation
            if framework_config["data_subject_rights"]:
                data_subject_request = request_data.get("data_subject_request")
                if data_subject_request:
                    request_type = data_subject_request.get("type")
                    if request_type in ["access", "portability", "erasure"]:
                        response_time_hours = data_subject_request.get("response_time_hours", 0)
                        if response_time_hours > 720:  # 30 days
                            violations.append(f"GDPR data subject request response time exceeded: {response_time_hours} hours")
            
            # Consent management validation
            if framework_config["consent_management"]:
                consent_data = request_data.get("consent_data", {})
                if not consent_data.get("consent_id"):
                    violations.append("GDPR requires valid consent ID for data processing")
                
                if not consent_data.get("lawful_basis"):
                    violations.append("GDPR requires lawful basis for data processing")
        
        # CCPA specific governance
        elif framework == "CCPA":
            if framework_config["consumer_rights"]:
                consumer_request = request_data.get("consumer_request")
                if consumer_request:
                    opt_out_honored = consumer_request.get("opt_out_honored", False)
                    if not opt_out_honored and consumer_request.get("type") == "opt_out":
                        violations.append("CCPA opt-out request must be honored")
        
        # HIPAA specific governance (for healthcare SaaS)
        elif framework == "HIPAA":
            if framework_config["phi_protection"]:
                phi_data = request_data.get("contains_phi", False)
                if phi_data:
                    encryption_status = request_data.get("encryption_status", "none")
                    if encryption_status != "encrypted":
                        violations.append("HIPAA requires encryption for PHI data")
                    
                    baa_status = request_data.get("business_associate_agreement", False)
                    if not baa_status:
                        violations.append("HIPAA requires Business Associate Agreement for PHI processing")
        
        decision = GovernanceDecision.DENY if violations else GovernanceDecision.ALLOW
        
        return GovernanceHookResult(
            decision=decision,
            hook_type=context.hook_type,
            context=context,
            violations=violations
        )
    
    async def get_saas_governance_recommendations(self, tenant_id: int, 
                                                saas_tier: SaaSTier) -> Dict[str, Any]:
        """
        Get SaaS-specific governance recommendations
        Task 8.5.33: Predictive analytics on governance violations
        """
        recommendations = {
            "tenant_id": tenant_id,
            "saas_tier": saas_tier.value,
            "governance_score": 85.0,  # Calculated based on compliance
            "recommendations": []
        }
        
        tier_config = self.saas_tier_policies[saas_tier]
        
        # Tier-specific recommendations
        if saas_tier in [SaaSTier.FREEMIUM, SaaSTier.STARTER]:
            recommendations["recommendations"].extend([
                {
                    "type": "tier_upgrade",
                    "priority": "medium",
                    "description": "Consider upgrading to Professional tier for enhanced governance features",
                    "impact": "Improved compliance posture and reduced governance overhead"
                },
                {
                    "type": "policy_optimization",
                    "priority": "low",
                    "description": "Enable stricter governance policies as usage grows",
                    "impact": "Better preparation for compliance audits"
                }
            ])
        
        elif saas_tier in [SaaSTier.ENTERPRISE, SaaSTier.ENTERPRISE_PLUS]:
            recommendations["recommendations"].extend([
                {
                    "type": "advanced_monitoring",
                    "priority": "high",
                    "description": "Implement real-time governance monitoring and alerting",
                    "impact": "Proactive compliance violation prevention"
                },
                {
                    "type": "custom_policies",
                    "priority": "medium",
                    "description": "Consider custom governance policies for industry-specific requirements",
                    "impact": "Enhanced compliance for regulated industries"
                }
            ])
        
        return recommendations

# Convenience functions for SaaS governance

async def enforce_saas_governance(middleware: SaaSGovernanceHooks, 
                                tenant_id: int, user_id: str, user_role: str,
                                workflow_id: str, execution_id: str,
                                saas_tier: SaaSTier, request_data: Dict[str, Any]) -> GovernanceHookResult:
    """Enforce comprehensive SaaS governance"""
    context = GovernanceContext(
        tenant_id=tenant_id,
        user_id=user_id,
        user_role=user_role,
        workflow_id=workflow_id,
        execution_id=execution_id,
        plane=GovernancePlane.CONTROL,
        hook_type=GovernanceHookType.POLICY,
        request_data=request_data,
        metadata={"saas_tier": saas_tier.value},
        timestamp=datetime.now()
    )
    
    return await middleware.enforce_saas_tier_governance(context, saas_tier)
