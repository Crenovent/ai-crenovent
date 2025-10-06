"""
CHAPTER 16.3: COMPLIANCE OVERLAY SERVICE
Tasks 16.3.6-16.3.11: Multi-industry compliance overlay management and enforcement
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import os

from src.database.connection_pool import ConnectionPoolManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    SOX = "SOX"
    GDPR = "GDPR"
    RBI = "RBI"
    DPDP = "DPDP"
    HIPAA = "HIPAA"
    NAIC = "NAIC"
    AML_KYC = "AML_KYC"
    PCI_DSS = "PCI_DSS"

class EnforcementMode(Enum):
    ENFORCING = "enforcing"
    MONITORING = "monitoring"
    DISABLED = "disabled"

class EnforcementResult(Enum):
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    WARNING = "warning"
    EXCEPTION = "exception"

class ViolationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComplianceOverlay:
    """Task 16.3.6: Compliance overlay definition"""
    overlay_id: str
    overlay_name: str
    compliance_framework: ComplianceFramework
    industry_code: str
    jurisdiction: str
    overlay_config: Dict[str, Any]
    enforcement_rules: Dict[str, Any]
    enforcement_mode: EnforcementMode
    applicable_workflows: List[str]
    tenant_id: int

@dataclass
class ComplianceRule:
    """Task 16.3.7: Individual compliance rule"""
    rule_id: str
    overlay_id: str
    rule_name: str
    rule_category: str
    regulatory_section: str
    rule_condition: Dict[str, Any]
    rule_action: Dict[str, Any]
    enforcement_level: str
    violation_severity: ViolationSeverity

@dataclass
class ComplianceEnforcementResult:
    """Task 16.3.8: Compliance enforcement result"""
    enforcement_id: str
    overlay_id: str
    enforcement_result: EnforcementResult
    compliance_score: float
    violation_count: int
    violation_details: Dict[str, Any]
    remediation_required: bool
    evidence_pack_id: str

class ComplianceOverlayService:
    """
    Task 16.3.6: Compliance Overlay Service
    Manages multi-industry compliance overlays with automated enforcement
    """
    
    def __init__(self, pool_manager: ConnectionPoolManager):
        self.pool_manager = pool_manager
        self.logger = logger
        self.config = self._load_config()
        
        # Initialize industry-specific compliance overlays
        self.saas_overlays = self._initialize_saas_overlays()
        self.banking_overlays = self._initialize_banking_overlays()
        self.insurance_overlays = self._initialize_insurance_overlays()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load dynamic configuration from environment"""
        return {
            "compliance_enforcement_enabled": os.getenv("COMPLIANCE_ENFORCEMENT_ENABLED", "true").lower() == "true",
            "sox_materiality_threshold": float(os.getenv("SOX_MATERIALITY_THRESHOLD", "100000")),
            "gdpr_breach_notification_hours": int(os.getenv("GDPR_BREACH_NOTIFICATION_HOURS", "72")),
            "compliance_score_threshold": float(os.getenv("COMPLIANCE_SCORE_THRESHOLD", "85.0")),
            "regulatory_notification_enabled": os.getenv("REGULATORY_NOTIFICATION_ENABLED", "true").lower() == "true",
            "audit_trail_retention_years": int(os.getenv("AUDIT_TRAIL_RETENTION_YEARS", "7"))
        }
    
    def _initialize_saas_overlays(self) -> Dict[str, Dict[str, Any]]:
        """Task 16.3.6: Initialize SaaS compliance overlays"""
        return {
            "sox_saas": {
                "name": "SOX SaaS Revenue Recognition",
                "framework": ComplianceFramework.SOX,
                "description": "SOX compliance for SaaS revenue recognition and financial reporting",
                "rules": {
                    "dual_approval_material": {
                        "condition": {"revenue_amount": {"gt": self.config["sox_materiality_threshold"]}},
                        "action": {"require_dual_approval": True, "approvers": ["finance_manager", "cfo"]},
                        "severity": ViolationSeverity.HIGH
                    },
                    "segregation_of_duties": {
                        "condition": {"always": True},
                        "action": {"enforce_sod": True, "incompatible_roles": [["revenue_calculator", "revenue_approver"]]},
                        "severity": ViolationSeverity.CRITICAL
                    },
                    "quarterly_certification": {
                        "condition": {"period_type": "quarterly"},
                        "action": {"require_certification": True, "certifier": "cfo"},
                        "severity": ViolationSeverity.HIGH
                    },
                    "audit_trail_retention": {
                        "condition": {"always": True},
                        "action": {"retain_audit_trail": True, "retention_years": 7},
                        "severity": ViolationSeverity.MEDIUM
                    }
                },
                "enforcement_points": ["workflow_start", "decision_points", "workflow_completion"],
                "regulatory_authority": "SEC",
                "jurisdiction": "US"
            },
            "gdpr_saas": {
                "name": "GDPR SaaS Data Privacy",
                "framework": ComplianceFramework.GDPR,
                "description": "GDPR compliance for SaaS customer data processing and privacy",
                "rules": {
                    "consent_validation": {
                        "condition": {"data_type": "personal_data"},
                        "action": {"validate_consent": True, "consent_types": ["explicit", "legitimate_interest"]},
                        "severity": ViolationSeverity.HIGH
                    },
                    "data_minimization": {
                        "condition": {"always": True},
                        "action": {"validate_necessity": True, "purpose_limitation": True},
                        "severity": ViolationSeverity.MEDIUM
                    },
                    "retention_limits": {
                        "condition": {"data_age_days": {"gt": "retention_limit"}},
                        "action": {"delete_data": True, "confirmation_required": True},
                        "severity": ViolationSeverity.HIGH
                    },
                    "right_to_erasure": {
                        "condition": {"erasure_request": True},
                        "action": {"delete_data": True, "cascade_deletion": True},
                        "severity": ViolationSeverity.CRITICAL
                    },
                    "breach_notification": {
                        "condition": {"data_breach": True},
                        "action": {"notify_dpa": True, "notification_hours": 72},
                        "severity": ViolationSeverity.CRITICAL
                    }
                },
                "enforcement_points": ["data_collection", "data_processing", "data_storage", "data_deletion"],
                "regulatory_authority": "GDPR_DPA",
                "jurisdiction": "EU"
            }
        }
    
    def _initialize_banking_overlays(self) -> Dict[str, Dict[str, Any]]:
        """Task 16.3.6: Initialize Banking compliance overlays"""
        return {
            "rbi_digital_lending": {
                "name": "RBI Digital Lending Guidelines",
                "framework": ComplianceFramework.RBI,
                "description": "RBI compliance for digital lending and fintech operations",
                "rules": {
                    "kyc_verification": {
                        "condition": {"loan_amount": {"gt": 50000}},
                        "action": {"require_kyc": True, "verification_level": "enhanced"},
                        "severity": ViolationSeverity.CRITICAL
                    },
                    "interest_rate_caps": {
                        "condition": {"interest_rate": {"gt": "rbi_cap"}},
                        "action": {"reject_loan": True, "reason": "interest_rate_violation"},
                        "severity": ViolationSeverity.HIGH
                    },
                    "lending_limit_check": {
                        "condition": {"total_exposure": {"gt": "prudential_limit"}},
                        "action": {"require_approval": True, "approver": "chief_risk_officer"},
                        "severity": ViolationSeverity.HIGH
                    }
                },
                "enforcement_points": ["loan_origination", "risk_assessment", "disbursement"],
                "regulatory_authority": "RBI",
                "jurisdiction": "IN"
            },
            "aml_kyc": {
                "name": "AML/KYC Compliance",
                "framework": ComplianceFramework.AML_KYC,
                "description": "Anti-Money Laundering and Know Your Customer compliance",
                "rules": {
                    "suspicious_transaction": {
                        "condition": {"transaction_amount": {"gt": 1000000}, "risk_score": {"gt": 0.8}},
                        "action": {"file_str": True, "freeze_account": True},
                        "severity": ViolationSeverity.CRITICAL
                    },
                    "customer_due_diligence": {
                        "condition": {"customer_risk": "high"},
                        "action": {"enhanced_due_diligence": True, "periodic_review": True},
                        "severity": ViolationSeverity.HIGH
                    }
                },
                "enforcement_points": ["transaction_processing", "account_opening", "periodic_review"],
                "regulatory_authority": "FATF",
                "jurisdiction": "Global"
            }
        }
    
    def _initialize_insurance_overlays(self) -> Dict[str, Dict[str, Any]]:
        """Task 16.3.6: Initialize Insurance compliance overlays"""
        return {
            "naic_solvency": {
                "name": "NAIC Solvency Requirements",
                "framework": ComplianceFramework.NAIC,
                "description": "NAIC solvency and capital adequacy requirements",
                "rules": {
                    "capital_adequacy": {
                        "condition": {"capital_ratio": {"lt": 1.5}},
                        "action": {"regulatory_action": True, "capital_injection_required": True},
                        "severity": ViolationSeverity.CRITICAL
                    },
                    "reserve_adequacy": {
                        "condition": {"reserve_ratio": {"lt": "regulatory_minimum"}},
                        "action": {"increase_reserves": True, "actuarial_review": True},
                        "severity": ViolationSeverity.HIGH
                    }
                },
                "enforcement_points": ["underwriting", "claims_processing", "financial_reporting"],
                "regulatory_authority": "NAIC",
                "jurisdiction": "US"
            }
        }
    
    async def create_compliance_overlay(self, overlay_config: Dict[str, Any], tenant_id: int, user_id: int) -> str:
        """Task 16.3.6: Create new compliance overlay"""
        try:
            overlay_id = str(uuid.uuid4())
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                # Insert compliance overlay
                await conn.execute("""
                    INSERT INTO compliance_overlay (
                        overlay_id, tenant_id, overlay_name, overlay_description,
                        compliance_framework, industry_code, jurisdiction, regulatory_authority,
                        overlay_version, overlay_config, enforcement_rules,
                        applicable_workflows, applicable_automation_types,
                        enforcement_mode, fail_closed, override_allowed,
                        created_by_user_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """, overlay_id, tenant_id, overlay_config["overlay_name"], 
                overlay_config.get("overlay_description", ""),
                overlay_config["compliance_framework"], overlay_config["industry_code"],
                overlay_config.get("jurisdiction", "US"), overlay_config.get("regulatory_authority", ""),
                overlay_config.get("overlay_version", "1.0"), json.dumps(overlay_config.get("overlay_config", {})),
                json.dumps(overlay_config.get("enforcement_rules", {})),
                overlay_config.get("applicable_workflows", []), 
                overlay_config.get("applicable_automation_types", ["RBA", "RBIA", "AALA"]),
                overlay_config.get("enforcement_mode", "enforcing"), 
                overlay_config.get("fail_closed", True), overlay_config.get("override_allowed", False),
                user_id)
                
                # Create associated rules
                rules = overlay_config.get("rules", {})
                for rule_name, rule_config in rules.items():
                    await self._create_compliance_rule(conn, overlay_id, rule_name, rule_config, tenant_id)
                
                self.logger.info(f"✅ Created compliance overlay {overlay_id} with {len(rules)} rules")
                return overlay_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create compliance overlay: {e}")
            raise
    
    async def _create_compliance_rule(self, conn, overlay_id: str, rule_name: str, 
                                    rule_config: Dict[str, Any], tenant_id: int):
        """Task 16.3.7: Create individual compliance rule"""
        rule_id = str(uuid.uuid4())
        
        await conn.execute("""
            INSERT INTO compliance_overlay_rule (
                rule_id, overlay_id, tenant_id, rule_name, rule_description,
                rule_category, regulatory_section, rule_condition, rule_action,
                enforcement_level, violation_severity, automation_hook, execution_order
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """, rule_id, overlay_id, tenant_id, rule_name, 
        rule_config.get("description", ""), rule_config.get("category", "general"),
        rule_config.get("regulatory_section", ""), json.dumps(rule_config.get("condition", {})),
        json.dumps(rule_config.get("action", {})), rule_config.get("enforcement_level", "mandatory"),
        rule_config.get("severity", ViolationSeverity.MEDIUM).value if isinstance(rule_config.get("severity"), ViolationSeverity) else rule_config.get("severity", "medium"),
        rule_config.get("automation_hook", ""), rule_config.get("execution_order", 100))
    
    async def enforce_compliance(self, tenant_id: int, workflow_context: Dict[str, Any]) -> ComplianceEnforcementResult:
        """Task 16.3.8: Enforce compliance overlays for workflow"""
        try:
            enforcement_id = str(uuid.uuid4())
            
            # Get applicable overlays
            overlays = await self._get_applicable_overlays(tenant_id, workflow_context)
            
            if not overlays:
                # No applicable overlays - compliant by default
                return ComplianceEnforcementResult(
                    enforcement_id=enforcement_id,
                    overlay_id="none",
                    enforcement_result=EnforcementResult.COMPLIANT,
                    compliance_score=100.0,
                    violation_count=0,
                    violation_details={},
                    remediation_required=False,
                    evidence_pack_id=str(uuid.uuid4())
                )
            
            # Evaluate all applicable overlays
            total_violations = 0
            all_violation_details = {}
            critical_violations = 0
            high_violations = 0
            medium_violations = 0
            low_violations = 0
            
            for overlay in overlays:
                overlay_violations = await self._evaluate_overlay_rules(overlay, workflow_context, tenant_id)
                
                if overlay_violations:
                    total_violations += len(overlay_violations)
                    all_violation_details[overlay["overlay_id"]] = overlay_violations
                    
                    # Count violations by severity
                    for violation in overlay_violations:
                        severity = violation.get("severity", "medium")
                        if severity == "critical":
                            critical_violations += 1
                        elif severity == "high":
                            high_violations += 1
                        elif severity == "medium":
                            medium_violations += 1
                        else:
                            low_violations += 1
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(
                total_violations, critical_violations, high_violations, medium_violations, low_violations
            )
            
            # Determine enforcement result
            if total_violations == 0:
                enforcement_result = EnforcementResult.COMPLIANT
            elif critical_violations > 0:
                enforcement_result = EnforcementResult.VIOLATION
            elif compliance_score < self.config["compliance_score_threshold"]:
                enforcement_result = EnforcementResult.VIOLATION
            else:
                enforcement_result = EnforcementResult.WARNING
            
            # Log enforcement result
            evidence_pack_id = await self._log_compliance_enforcement(
                tenant_id, enforcement_id, overlays[0]["overlay_id"] if overlays else "none",
                enforcement_result, compliance_score, total_violations, all_violation_details,
                critical_violations, high_violations, medium_violations, low_violations,
                workflow_context
            )
            
            result = ComplianceEnforcementResult(
                enforcement_id=enforcement_id,
                overlay_id=overlays[0]["overlay_id"] if overlays else "none",
                enforcement_result=enforcement_result,
                compliance_score=compliance_score,
                violation_count=total_violations,
                violation_details=all_violation_details,
                remediation_required=critical_violations > 0 or high_violations > 0,
                evidence_pack_id=evidence_pack_id
            )
            
            self.logger.info(f"✅ Compliance enforcement: {enforcement_result.value}, score: {compliance_score:.1f}%, violations: {total_violations}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Compliance enforcement failed: {e}")
            raise
    
    async def _get_applicable_overlays(self, tenant_id: int, workflow_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get compliance overlays applicable to the workflow context"""
        async with self.pool_manager.get_connection() as conn:
            await conn.execute("SET app.current_tenant_id = $1", tenant_id)
            
            # Get active overlays for tenant
            query = """
                SELECT co.*, array_agg(cor.rule_id) as rule_ids
                FROM compliance_overlay co
                LEFT JOIN compliance_overlay_rule cor ON co.overlay_id = cor.overlay_id
                WHERE co.tenant_id = $1 
                AND co.status = 'active' 
                AND co.enforcement_mode = 'enforcing'
                AND (co.effective_until IS NULL OR co.effective_until > NOW())
                GROUP BY co.overlay_id
                ORDER BY co.created_at
            """
            
            rows = await conn.fetch(query, tenant_id)
            
            overlays = []
            for row in rows:
                overlay_dict = dict(row)
                overlay_dict['overlay_config'] = json.loads(overlay_dict.get('overlay_config', '{}'))
                overlay_dict['enforcement_rules'] = json.loads(overlay_dict.get('enforcement_rules', '{}'))
                
                # Check if overlay applies to this workflow
                if self._overlay_applies_to_workflow(overlay_dict, workflow_context):
                    overlays.append(overlay_dict)
            
            return overlays
    
    def _overlay_applies_to_workflow(self, overlay: Dict[str, Any], workflow_context: Dict[str, Any]) -> bool:
        """Check if overlay applies to the current workflow context"""
        # Check workflow type
        applicable_workflows = overlay.get("applicable_workflows", [])
        if applicable_workflows and workflow_context.get("workflow_type") not in applicable_workflows:
            return False
        
        # Check automation type
        applicable_automation_types = overlay.get("applicable_automation_types", ["RBA", "RBIA", "AALA"])
        if workflow_context.get("automation_type") not in applicable_automation_types:
            return False
        
        # Check industry match
        if overlay.get("industry_code") != workflow_context.get("industry_code", "SaaS"):
            return False
        
        return True
    
    async def _evaluate_overlay_rules(self, overlay: Dict[str, Any], workflow_context: Dict[str, Any], 
                                    tenant_id: int) -> List[Dict[str, Any]]:
        """Evaluate all rules in a compliance overlay"""
        violations = []
        
        async with self.pool_manager.get_connection() as conn:
            await conn.execute("SET app.current_tenant_id = $1", tenant_id)
            
            # Get rules for this overlay
            rules_query = """
                SELECT * FROM compliance_overlay_rule 
                WHERE overlay_id = $1 AND rule_status = 'active'
                ORDER BY execution_order
            """
            
            rule_rows = await conn.fetch(rules_query, overlay["overlay_id"])
            
            for rule_row in rule_rows:
                rule_dict = dict(rule_row)
                rule_dict['rule_condition'] = json.loads(rule_dict.get('rule_condition', '{}'))
                rule_dict['rule_action'] = json.loads(rule_dict.get('rule_action', '{}'))
                
                # Evaluate rule condition
                if self._evaluate_rule_condition(rule_dict['rule_condition'], workflow_context):
                    # Rule condition met - check if it's a violation
                    if self._is_violation(rule_dict, workflow_context):
                        violations.append({
                            "rule_id": rule_dict["rule_id"],
                            "rule_name": rule_dict["rule_name"],
                            "rule_category": rule_dict["rule_category"],
                            "regulatory_section": rule_dict["regulatory_section"],
                            "severity": rule_dict["violation_severity"],
                            "violation_details": self._get_violation_details(rule_dict, workflow_context),
                            "remediation_action": rule_dict['rule_action']
                        })
        
        return violations
    
    def _evaluate_rule_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate rule condition against workflow context"""
        if condition.get("always"):
            return True
        
        for field, criteria in condition.items():
            if field not in context:
                continue
            
            context_value = context[field]
            
            if isinstance(criteria, dict):
                if "gt" in criteria:
                    threshold = criteria["gt"]
                    if isinstance(threshold, str) and threshold in context:
                        threshold = context[threshold]
                    if context_value <= threshold:
                        return False
                
                if "lt" in criteria:
                    threshold = criteria["lt"]
                    if isinstance(threshold, str) and threshold in context:
                        threshold = context[threshold]
                    if context_value >= threshold:
                        return False
                
                if "eq" in criteria:
                    if context_value != criteria["eq"]:
                        return False
            
            elif context_value != criteria:
                return False
        
        return True
    
    def _is_violation(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Determine if rule evaluation constitutes a violation"""
        rule_action = rule.get("rule_action", {})
        
        # Check for blocking actions that indicate violations
        blocking_actions = ["require_dual_approval", "require_approval", "block_execution", 
                          "reject_loan", "file_str", "freeze_account"]
        
        for action_key in rule_action.keys():
            if action_key in blocking_actions:
                # Check if the required action is satisfied in context
                if not context.get(f"{action_key}_satisfied", False):
                    return True
        
        return False
    
    def _get_violation_details(self, rule: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed violation information"""
        return {
            "rule_condition": rule.get("rule_condition", {}),
            "context_values": {k: v for k, v in context.items() if k in rule.get("rule_condition", {})},
            "required_action": rule.get("rule_action", {}),
            "enforcement_level": rule.get("enforcement_level", "mandatory")
        }
    
    def _calculate_compliance_score(self, total_violations: int, critical: int, high: int, 
                                  medium: int, low: int) -> float:
        """Calculate overall compliance score"""
        if total_violations == 0:
            return 100.0
        
        # Weighted scoring based on violation severity
        violation_score = (critical * 10) + (high * 5) + (medium * 2) + (low * 1)
        max_possible_score = total_violations * 10  # If all were critical
        
        # Calculate compliance as percentage
        compliance_percentage = max(0, 100 - (violation_score / max_possible_score * 100))
        return round(compliance_percentage, 2)
    
    async def _log_compliance_enforcement(self, tenant_id: int, enforcement_id: str, overlay_id: str,
                                        enforcement_result: EnforcementResult, compliance_score: float,
                                        violation_count: int, violation_details: Dict[str, Any],
                                        critical_violations: int, high_violations: int,
                                        medium_violations: int, low_violations: int,
                                        workflow_context: Dict[str, Any]) -> str:
        """Log compliance enforcement result"""
        evidence_pack_id = str(uuid.uuid4())
        
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                await conn.execute("""
                    INSERT INTO compliance_overlay_enforcement (
                        enforcement_id, overlay_id, tenant_id, workflow_id, execution_id,
                        automation_type, enforcement_result, violation_details, compliance_score,
                        violation_count, critical_violations, high_violations, medium_violations, low_violations,
                        remediation_required, evidence_pack_id, enforcement_time_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """, enforcement_id, overlay_id, tenant_id, 
                workflow_context.get("workflow_id"), workflow_context.get("execution_id"),
                workflow_context.get("automation_type", "RBA"), enforcement_result.value,
                json.dumps(violation_details), compliance_score, violation_count,
                critical_violations, high_violations, medium_violations, low_violations,
                critical_violations > 0 or high_violations > 0, evidence_pack_id, 
                workflow_context.get("enforcement_time_ms", 0))
                
                return evidence_pack_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to log compliance enforcement: {e}")
            return evidence_pack_id
    
    async def get_tenant_compliance_overlays(self, tenant_id: int, framework_filter: str = None) -> List[Dict[str, Any]]:
        """Task 16.3.9: Get tenant compliance overlays"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                query = """
                    SELECT co.*, COUNT(cor.rule_id) as rule_count
                    FROM compliance_overlay co
                    LEFT JOIN compliance_overlay_rule cor ON co.overlay_id = cor.overlay_id
                    WHERE co.tenant_id = $1
                """
                params = [tenant_id]
                
                if framework_filter:
                    query += " AND co.compliance_framework = $2"
                    params.append(framework_filter)
                
                query += " GROUP BY co.overlay_id ORDER BY co.created_at DESC"
                
                rows = await conn.fetch(query, *params)
                
                overlays = []
                for row in rows:
                    overlay_dict = dict(row)
                    overlay_dict['overlay_config'] = json.loads(overlay_dict.get('overlay_config', '{}'))
                    overlay_dict['enforcement_rules'] = json.loads(overlay_dict.get('enforcement_rules', '{}'))
                    overlays.append(overlay_dict)
                
                self.logger.info(f"✅ Retrieved {len(overlays)} compliance overlays for tenant {tenant_id}")
                return overlays
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get compliance overlays: {e}")
            raise
    
    async def get_compliance_analytics(self, tenant_id: int, period_days: int = 30) -> Dict[str, Any]:
        """Task 16.3.10: Get compliance analytics and metrics"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)
                
                # Get enforcement metrics
                metrics_query = """
                    SELECT 
                        COUNT(*) as total_enforcements,
                        COUNT(CASE WHEN enforcement_result = 'compliant' THEN 1 END) as compliant_count,
                        COUNT(CASE WHEN enforcement_result = 'violation' THEN 1 END) as violation_count,
                        COUNT(CASE WHEN enforcement_result = 'warning' THEN 1 END) as warning_count,
                        AVG(compliance_score) as avg_compliance_score,
                        SUM(violation_count) as total_violations,
                        SUM(critical_violations) as total_critical_violations,
                        SUM(high_violations) as total_high_violations
                    FROM compliance_overlay_enforcement 
                    WHERE tenant_id = $1 
                    AND enforced_at >= NOW() - INTERVAL '%s days'
                """ % period_days
                
                metrics_row = await conn.fetchrow(metrics_query, tenant_id)
                
                # Get framework breakdown
                framework_query = """
                    SELECT co.compliance_framework, COUNT(coe.*) as enforcement_count
                    FROM compliance_overlay_enforcement coe
                    JOIN compliance_overlay co ON coe.overlay_id = co.overlay_id
                    WHERE coe.tenant_id = $1 
                    AND coe.enforced_at >= NOW() - INTERVAL '%s days'
                    GROUP BY co.compliance_framework
                """ % period_days
                
                framework_rows = await conn.fetch(framework_query, tenant_id)
                
                analytics = {
                    "summary": dict(metrics_row) if metrics_row else {},
                    "framework_breakdown": {row["compliance_framework"]: row["enforcement_count"] for row in framework_rows},
                    "period_days": period_days,
                    "calculated_at": datetime.now().isoformat()
                }
                
                return analytics
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get compliance analytics: {e}")
            return {"error": str(e)}
    
    async def get_saas_compliance_templates(self) -> Dict[str, Any]:
        """Task 16.3.11: Get SaaS compliance overlay templates"""
        return {
            "saas_overlays": self.saas_overlays,
            "banking_overlays": self.banking_overlays,
            "insurance_overlays": self.insurance_overlays,
            "compliance_frameworks": [framework.value for framework in ComplianceFramework],
            "enforcement_modes": [mode.value for mode in EnforcementMode],
            "violation_severities": [severity.value for severity in ViolationSeverity]
        }
