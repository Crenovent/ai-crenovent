"""
CHAPTER 16.1: POLICY ORCHESTRATION SERVICE
Tasks 16.1.5-16.1.11, 16.1.18-16.1.22: Policy Registry and OPA Integration
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import os
from pathlib import Path

from src.database.connection_pool import ConnectionPoolManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyType(Enum):
    COMPLIANCE = "compliance"
    BUSINESS_RULE = "business_rule"
    SECURITY = "security"
    GOVERNANCE = "governance"

class ComplianceFramework(Enum):
    SOX = "SOX"
    GDPR = "GDPR"
    RBI = "RBI"
    DPDP = "DPDP"
    HIPAA = "HIPAA"
    NAIC = "NAIC"
    AML_KYC = "AML_KYC"
    PCI_DSS = "PCI_DSS"
    SAAS_BUSINESS_RULES = "SAAS_BUSINESS_RULES"

class EnforcementMode(Enum):
    ENFORCING = "enforcing"
    PERMISSIVE = "permissive"
    DISABLED = "disabled"

class LifecycleState(Enum):
    DRAFT = "draft"
    TESTING = "testing"
    VALIDATED = "validated"
    PUBLISHED = "published"
    PROMOTED = "promoted"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

@dataclass
class PolicyDefinition:
    """Task 16.1.5: SaaS policy pack definition"""
    policy_id: str
    policy_name: str
    policy_description: str
    policy_type: PolicyType
    industry_code: str
    compliance_framework: ComplianceFramework
    jurisdiction: str
    sla_tier: str
    policy_content: Dict[str, Any]
    enforcement_mode: EnforcementMode
    fail_closed: bool
    override_allowed: bool
    target_automation_types: List[str]
    created_by_user_id: Optional[int] = None
    tenant_id: Optional[int] = None

@dataclass
class PolicyVersion:
    """Task 16.1.6: Policy versioning"""
    version_id: str
    policy_id: str
    version_number: str
    version_hash: str
    policy_content: Dict[str, Any]
    lifecycle_state: LifecycleState
    validation_status: str
    breaking_changes: bool
    changelog: Optional[str] = None
    created_by_user_id: Optional[int] = None
    tenant_id: Optional[int] = None

@dataclass
class PolicyEnforcementResult:
    """Task 16.1.18: Policy enforcement result"""
    policy_id: str
    version_id: str
    enforcement_result: str  # 'allowed', 'denied', 'overridden'
    policy_decision: Dict[str, Any]
    execution_time_ms: int
    evidence_pack_id: Optional[str] = None
    override_id: Optional[str] = None
    override_reason: Optional[str] = None

class PolicyRegistryService:
    """
    Task 16.1.9: Policy Registry Service
    Manages policy lifecycle, OPA integration, and enforcement
    """
    
    def __init__(self, pool_manager: ConnectionPoolManager):
        self.pool_manager = pool_manager
        self.logger = logger
        self.config = self._load_config()
        
        # Task 16.1.7: OPA integration configuration
        self.opa_enabled = self.config.get("opa_enabled", True)
        self.opa_endpoint = self.config.get("opa_endpoint", "http://localhost:8181")
        
        # SaaS policy packs cache
        self.saas_policy_cache = {}
        self._initialize_saas_policies()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load dynamic configuration from environment"""
        return {
            "opa_enabled": os.getenv("POLICY_OPA_ENABLED", "true").lower() == "true",
            "opa_endpoint": os.getenv("POLICY_OPA_ENDPOINT", "http://localhost:8181"),
            "policy_cache_ttl_minutes": int(os.getenv("POLICY_CACHE_TTL_MINUTES", "30")),
            "evidence_retention_days": int(os.getenv("POLICY_EVIDENCE_RETENTION_DAYS", "2555")),  # 7 years
            "fail_closed_default": os.getenv("POLICY_FAIL_CLOSED_DEFAULT", "true").lower() == "true",
            "override_approval_required": os.getenv("POLICY_OVERRIDE_APPROVAL_REQUIRED", "true").lower() == "true"
        }
    
    def _initialize_saas_policies(self):
        """Task 16.1.5: Initialize SaaS policy packs"""
        self.saas_policy_cache = {
            "sox_revenue_recognition": self._create_sox_revenue_policy(),
            "gdpr_data_privacy": self._create_gdpr_privacy_policy(),
            "subscription_lifecycle": self._create_subscription_lifecycle_policy(),
            "churn_prevention": self._create_churn_prevention_policy(),
            "usage_monitoring": self._create_usage_monitoring_policy()
        }
    
    def _create_sox_revenue_policy(self) -> PolicyDefinition:
        """Task 16.1.5: SOX compliance policy for SaaS revenue recognition"""
        return PolicyDefinition(
            policy_id=str(uuid.uuid4()),
            policy_name="SaaS SOX Revenue Recognition Policy",
            policy_description="Ensures SOX compliance for SaaS revenue recognition workflows",
            policy_type=PolicyType.COMPLIANCE,
            industry_code="SaaS",
            compliance_framework=ComplianceFramework.SOX,
            jurisdiction="US",
            sla_tier="enterprise",
            policy_content={
                "package": "saas.sox.revenue",
                "rules": {
                    "dual_approval_required": {
                        "condition": "input.revenue_amount > 10000",
                        "action": "require_approval",
                        "approver_roles": ["finance_manager", "cfo"],
                        "segregation_of_duties": True
                    },
                    "audit_trail_mandatory": {
                        "condition": "true",
                        "action": "log_evidence",
                        "retention_years": 7,
                        "immutable_logging": True
                    },
                    "quarterly_reconciliation": {
                        "condition": "input.period_type == 'quarterly'",
                        "action": "trigger_reconciliation",
                        "deadline_days": 15
                    }
                },
                "metadata": {
                    "sox_section": "404",
                    "risk_level": "high",
                    "automation_types": ["RBA", "RBIA"]
                }
            },
            enforcement_mode=EnforcementMode.ENFORCING,
            fail_closed=True,
            override_allowed=True,
            target_automation_types=["RBA", "RBIA"]
        )
    
    def _create_gdpr_privacy_policy(self) -> PolicyDefinition:
        """Task 16.1.5: GDPR compliance policy for SaaS data privacy"""
        return PolicyDefinition(
            policy_id=str(uuid.uuid4()),
            policy_name="SaaS GDPR Data Privacy Policy",
            policy_description="Ensures GDPR compliance for SaaS customer data processing",
            policy_type=PolicyType.COMPLIANCE,
            industry_code="SaaS",
            compliance_framework=ComplianceFramework.GDPR,
            jurisdiction="EU",
            sla_tier="enterprise",
            policy_content={
                "package": "saas.gdpr.privacy",
                "rules": {
                    "consent_validation": {
                        "condition": "input.data_type == 'personal_data'",
                        "action": "validate_consent",
                        "consent_types": ["explicit", "legitimate_interest"]
                    },
                    "data_minimization": {
                        "condition": "true",
                        "action": "validate_necessity",
                        "max_retention_days": 1095,
                        "purpose_limitation": True
                    },
                    "right_to_erasure": {
                        "condition": "input.erasure_request == true",
                        "action": "delete_data",
                        "confirmation_required": True,
                        "cascade_deletion": True
                    },
                    "data_portability": {
                        "condition": "input.portability_request == true",
                        "action": "export_data",
                        "format": "machine_readable"
                    }
                },
                "metadata": {
                    "gdpr_articles": ["6", "7", "17", "20"],
                    "risk_level": "high",
                    "automation_types": ["RBA", "RBIA", "AALA"]
                }
            },
            enforcement_mode=EnforcementMode.ENFORCING,
            fail_closed=True,
            override_allowed=False,  # GDPR violations cannot be overridden
            target_automation_types=["RBA", "RBIA", "AALA"]
        )
    
    def _create_subscription_lifecycle_policy(self) -> PolicyDefinition:
        """Task 16.1.5: SaaS subscription lifecycle business rules"""
        return PolicyDefinition(
            policy_id=str(uuid.uuid4()),
            policy_name="SaaS Subscription Lifecycle Policy",
            policy_description="Business rules for SaaS subscription lifecycle management",
            policy_type=PolicyType.BUSINESS_RULE,
            industry_code="SaaS",
            compliance_framework=ComplianceFramework.SAAS_BUSINESS_RULES,
            jurisdiction="US",
            sla_tier="standard",
            policy_content={
                "package": "saas.business.subscription",
                "rules": {
                    "trial_conversion": {
                        "condition": "input.trial_days_remaining <= 3",
                        "action": "trigger_conversion_workflow",
                        "conversion_incentives": ["discount", "extended_trial"]
                    },
                    "usage_overage": {
                        "condition": "input.usage_percent > 90",
                        "action": "notify_upgrade",
                        "auto_upgrade_threshold": 110
                    },
                    "payment_failure": {
                        "condition": "input.payment_failed == true",
                        "action": "retry_payment",
                        "max_retries": 3,
                        "grace_period_days": 7
                    },
                    "contract_renewal": {
                        "condition": "input.days_to_renewal <= 30",
                        "action": "trigger_renewal_workflow",
                        "early_renewal_discount": 0.05
                    }
                },
                "metadata": {
                    "business_impact": "high",
                    "automation_types": ["RBA", "RBIA", "AALA"]
                }
            },
            enforcement_mode=EnforcementMode.ENFORCING,
            fail_closed=False,
            override_allowed=True,
            target_automation_types=["RBA", "RBIA", "AALA"]
        )
    
    def _create_churn_prevention_policy(self) -> PolicyDefinition:
        """Task 16.1.5: SaaS churn prevention policy"""
        return PolicyDefinition(
            policy_id=str(uuid.uuid4()),
            policy_name="SaaS Churn Prevention Policy",
            policy_description="ML-driven churn prevention and retention workflows",
            policy_type=PolicyType.BUSINESS_RULE,
            industry_code="SaaS",
            compliance_framework=ComplianceFramework.SAAS_BUSINESS_RULES,
            jurisdiction="US",
            sla_tier="premium",
            policy_content={
                "package": "saas.churn.prevention",
                "rules": {
                    "high_churn_risk": {
                        "condition": "input.churn_risk_score > 0.8",
                        "action": "trigger_retention_workflow",
                        "escalation_required": True,
                        "csm_assignment": True
                    },
                    "usage_decline": {
                        "condition": "input.usage_trend == 'declining' and input.decline_percent > 30",
                        "action": "proactive_outreach",
                        "outreach_channels": ["email", "phone", "in_app"]
                    },
                    "support_ticket_spike": {
                        "condition": "input.support_tickets_30d > input.support_tickets_avg * 2",
                        "action": "escalate_to_csm",
                        "priority": "high"
                    }
                },
                "metadata": {
                    "ml_models": ["churn_prediction", "usage_forecasting"],
                    "automation_types": ["RBIA", "AALA"]
                }
            },
            enforcement_mode=EnforcementMode.ENFORCING,
            fail_closed=False,
            override_allowed=True,
            target_automation_types=["RBIA", "AALA"]
        )
    
    def _create_usage_monitoring_policy(self) -> PolicyDefinition:
        """Task 16.1.5: SaaS usage monitoring policy"""
        return PolicyDefinition(
            policy_id=str(uuid.uuid4()),
            policy_name="SaaS Usage Monitoring Policy",
            policy_description="Real-time usage monitoring and billing automation",
            policy_type=PolicyType.BUSINESS_RULE,
            industry_code="SaaS",
            compliance_framework=ComplianceFramework.SAAS_BUSINESS_RULES,
            jurisdiction="US",
            sla_tier="standard",
            policy_content={
                "package": "saas.usage.monitoring",
                "rules": {
                    "quota_exceeded": {
                        "condition": "input.usage_percent >= 100",
                        "action": "block_usage",
                        "notification_required": True,
                        "upgrade_offer": True
                    },
                    "billing_anomaly": {
                        "condition": "input.billing_amount > input.expected_amount * 1.5",
                        "action": "flag_for_review",
                        "auto_adjustment": False
                    },
                    "feature_usage_tracking": {
                        "condition": "input.feature_used == true",
                        "action": "log_usage",
                        "billing_impact": True
                    }
                },
                "metadata": {
                    "real_time": True,
                    "automation_types": ["RBA", "RBIA"]
                }
            },
            enforcement_mode=EnforcementMode.ENFORCING,
            fail_closed=True,
            override_allowed=True,
            target_automation_types=["RBA", "RBIA"]
        )
    
    async def create_policy(self, policy_def: PolicyDefinition, tenant_id: int, user_id: int) -> str:
        """Task 16.1.9: Create new policy"""
        try:
            policy_id = str(uuid.uuid4())
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                # Insert policy
                await conn.execute("""
                    INSERT INTO policy_orch_policy (
                        policy_id, tenant_id, policy_name, policy_description, policy_type,
                        industry_code, compliance_framework, jurisdiction, sla_tier,
                        target_automation_types, policy_content, policy_metadata,
                        enforcement_mode, fail_closed, override_allowed, created_by_user_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """, policy_id, tenant_id, policy_def.policy_name, policy_def.policy_description,
                policy_def.policy_type.value, policy_def.industry_code, 
                policy_def.compliance_framework.value, policy_def.jurisdiction,
                policy_def.sla_tier, policy_def.target_automation_types,
                json.dumps(policy_def.policy_content), json.dumps({}),
                policy_def.enforcement_mode.value, policy_def.fail_closed,
                policy_def.override_allowed, user_id)
                
                # Create initial version
                version_id = await self._create_initial_version(conn, policy_id, policy_def, tenant_id, user_id)
                
                self.logger.info(f"✅ Created policy {policy_id} with initial version {version_id}")
                return policy_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create policy: {e}")
            raise
    
    async def _create_initial_version(self, conn, policy_id: str, policy_def: PolicyDefinition, 
                                    tenant_id: int, user_id: int) -> str:
        """Task 16.1.6: Create initial policy version"""
        version_id = str(uuid.uuid4())
        version_number = "1.0.0"
        version_hash = hashlib.sha256(json.dumps(policy_def.policy_content, sort_keys=True).encode()).hexdigest()
        
        await conn.execute("""
            INSERT INTO policy_orch_version (
                version_id, policy_id, tenant_id, version_number, version_hash,
                policy_content, policy_metadata, lifecycle_state, validation_status,
                breaking_changes, created_by_user_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """, version_id, policy_id, tenant_id, version_number, version_hash,
        json.dumps(policy_def.policy_content), json.dumps({}),
        LifecycleState.DRAFT.value, "pending", False, user_id)
        
        return version_id
    
    async def get_tenant_policies(self, tenant_id: int, industry_filter: str = None, 
                                compliance_filter: str = None) -> List[Dict[str, Any]]:
        """Task 16.1.10: Get tenant policies with filters"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                query = """
                    SELECT p.*, pv.version_number, pv.lifecycle_state, pv.validation_status
                    FROM policy_orch_policy p
                    LEFT JOIN policy_orch_version pv ON p.policy_id = pv.policy_id
                    WHERE p.tenant_id = $1
                """
                params = [tenant_id]
                
                if industry_filter:
                    query += " AND p.industry_code = $2"
                    params.append(industry_filter)
                
                if compliance_filter:
                    filter_idx = len(params) + 1
                    query += f" AND p.compliance_framework = ${filter_idx}"
                    params.append(compliance_filter)
                
                query += " ORDER BY p.created_at DESC"
                
                rows = await conn.fetch(query, *params)
                
                policies = []
                for row in rows:
                    policy_dict = dict(row)
                    policy_dict['policy_content'] = json.loads(policy_dict.get('policy_content', '{}'))
                    policy_dict['policy_metadata'] = json.loads(policy_dict.get('policy_metadata', '{}'))
                    policies.append(policy_dict)
                
                self.logger.info(f"✅ Retrieved {len(policies)} policies for tenant {tenant_id}")
                return policies
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get tenant policies: {e}")
            raise
    
    async def enforce_policy(self, tenant_id: int, policy_id: str, context: Dict[str, Any]) -> PolicyEnforcementResult:
        """Task 16.1.18: Policy enforcement with OPA integration"""
        start_time = datetime.now()
        
        try:
            # Get active policy version
            policy_version = await self._get_active_policy_version(tenant_id, policy_id)
            if not policy_version:
                raise ValueError(f"No active policy version found for policy {policy_id}")
            
            # Task 16.1.7: OPA integration (simulated for now)
            if self.opa_enabled:
                decision = await self._evaluate_with_opa(policy_version, context)
            else:
                decision = await self._evaluate_locally(policy_version, context)
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Determine enforcement result
            enforcement_result = "allowed" if decision.get("allow", False) else "denied"
            
            # Log enforcement
            evidence_pack_id = await self._log_enforcement(
                tenant_id, policy_id, policy_version["version_id"], 
                enforcement_result, decision, execution_time, context
            )
            
            result = PolicyEnforcementResult(
                policy_id=policy_id,
                version_id=policy_version["version_id"],
                enforcement_result=enforcement_result,
                policy_decision=decision,
                execution_time_ms=execution_time,
                evidence_pack_id=evidence_pack_id
            )
            
            self.logger.info(f"✅ Policy enforcement: {enforcement_result} in {execution_time}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Policy enforcement failed: {e}")
            raise
    
    async def _get_active_policy_version(self, tenant_id: int, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get the active (published/promoted) version of a policy"""
        async with self.pool_manager.get_connection() as conn:
            await conn.execute("SET app.current_tenant_id = $1", tenant_id)
            
            row = await conn.fetchrow("""
                SELECT * FROM policy_orch_version 
                WHERE policy_id = $1 AND tenant_id = $2 
                AND lifecycle_state IN ('published', 'promoted')
                ORDER BY created_at DESC LIMIT 1
            """, policy_id, tenant_id)
            
            if row:
                version_dict = dict(row)
                version_dict['policy_content'] = json.loads(version_dict.get('policy_content', '{}'))
                return version_dict
            return None
    
    async def _evaluate_with_opa(self, policy_version: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Task 16.1.7: OPA policy evaluation (simulated)"""
        # In real implementation, this would make HTTP request to OPA server
        # For now, simulate based on policy rules
        policy_content = policy_version.get("policy_content", {})
        rules = policy_content.get("rules", {})
        
        decision = {"allow": True, "violations": [], "warnings": []}
        
        for rule_name, rule_config in rules.items():
            condition = rule_config.get("condition", "true")
            action = rule_config.get("action", "allow")
            
            # Simple condition evaluation (in real OPA, this would be Rego)
            if self._evaluate_condition(condition, context):
                if action in ["require_approval", "deny", "block_usage"]:
                    decision["allow"] = False
                    decision["violations"].append({
                        "rule": rule_name,
                        "action": action,
                        "condition": condition
                    })
                elif action in ["log_evidence", "notify", "trigger_workflow"]:
                    decision["warnings"].append({
                        "rule": rule_name,
                        "action": action,
                        "condition": condition
                    })
        
        return decision
    
    async def _evaluate_locally(self, policy_version: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Local policy evaluation fallback"""
        return await self._evaluate_with_opa(policy_version, context)
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Simple condition evaluator (replace with proper expression engine)"""
        if condition == "true":
            return True
        
        # Simple string-based evaluation for demo
        if "revenue_amount" in condition and "revenue_amount" in context:
            if "> 10000" in condition:
                return context["revenue_amount"] > 10000
        
        if "churn_risk_score" in condition and "churn_risk_score" in context:
            if "> 0.8" in condition:
                return context["churn_risk_score"] > 0.8
        
        if "usage_percent" in condition and "usage_percent" in context:
            if "> 90" in condition:
                return context["usage_percent"] > 90
        
        return False
    
    async def _log_enforcement(self, tenant_id: int, policy_id: str, version_id: str,
                             enforcement_result: str, decision: Dict[str, Any], 
                             execution_time_ms: int, context: Dict[str, Any]) -> str:
        """Task 16.1.18: Log policy enforcement for evidence"""
        evidence_pack_id = str(uuid.uuid4())
        
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                await conn.execute("""
                    INSERT INTO policy_orch_enforcement_log (
                        tenant_id, policy_id, version_id, enforcement_result,
                        policy_decision, evidence_pack_id, execution_time_ms,
                        workflow_id, automation_type
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, tenant_id, policy_id, version_id, enforcement_result,
                json.dumps(decision), evidence_pack_id, execution_time_ms,
                context.get("workflow_id"), context.get("automation_type", "RBA"))
                
                return evidence_pack_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log enforcement: {e}")
            return evidence_pack_id
    
    async def get_saas_policy_templates(self) -> Dict[str, Any]:
        """Task 16.1.5: Get SaaS policy templates"""
        return {
            "templates": {
                name: asdict(policy) for name, policy in self.saas_policy_cache.items()
            },
            "compliance_frameworks": [framework.value for framework in ComplianceFramework],
            "policy_types": [policy_type.value for policy_type in PolicyType],
            "enforcement_modes": [mode.value for mode in EnforcementMode]
        }
    
    async def validate_policy_syntax(self, policy_content: Dict[str, Any]) -> Dict[str, Any]:
        """Task 16.1.11: Policy syntax validation"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Basic validation
        if "package" not in policy_content:
            validation_result["valid"] = False
            validation_result["errors"].append("Missing 'package' field")
        
        if "rules" not in policy_content:
            validation_result["valid"] = False
            validation_result["errors"].append("Missing 'rules' field")
        else:
            rules = policy_content["rules"]
            for rule_name, rule_config in rules.items():
                if "condition" not in rule_config:
                    validation_result["warnings"].append(f"Rule '{rule_name}' missing condition")
                if "action" not in rule_config:
                    validation_result["warnings"].append(f"Rule '{rule_name}' missing action")
        
        return validation_result
    
    async def get_policy_metrics(self, tenant_id: int, days: int = 30) -> Dict[str, Any]:
        """Task 16.1.22: Policy enforcement metrics"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                # Get enforcement metrics
                metrics_query = """
                    SELECT 
                        enforcement_result,
                        COUNT(*) as count,
                        AVG(execution_time_ms) as avg_execution_time,
                        MAX(execution_time_ms) as max_execution_time
                    FROM policy_orch_enforcement_log 
                    WHERE tenant_id = $1 
                    AND enforced_at >= NOW() - INTERVAL '%s days'
                    GROUP BY enforcement_result
                """ % days
                
                metrics_rows = await conn.fetch(metrics_query, tenant_id)
                
                metrics = {
                    "enforcement_summary": {
                        row["enforcement_result"]: {
                            "count": row["count"],
                            "avg_execution_time_ms": float(row["avg_execution_time"] or 0),
                            "max_execution_time_ms": row["max_execution_time"]
                        } for row in metrics_rows
                    },
                    "total_enforcements": sum(row["count"] for row in metrics_rows),
                    "period_days": days
                }
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get policy metrics: {e}")
            return {"error": str(e)}
