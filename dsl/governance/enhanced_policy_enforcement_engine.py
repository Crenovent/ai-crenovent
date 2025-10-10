#!/usr/bin/env python3
"""
Enhanced Policy Pack Enforcement Engine - Chapter 14.1
======================================================
Tasks 14.1-T10, 14.1-T11: Policy pack enforcement and residency enforcement in approvals

Features:
- Real-time policy validation during approval workflows
- Residency-aware approval routing (GDPR, DPDP compliance)
- Multi-framework policy enforcement (SOX, RBI, IRDAI, HIPAA, PCI-DSS)
- Fail-closed enforcement with override ledger integration
- Dynamic policy rule evaluation with tenant isolation
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

from pydantic import BaseModel, Field
from src.database.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)

class PolicyEnforcementResult(str, Enum):
    APPROVED = "approved"
    DENIED = "denied" 
    REQUIRES_OVERRIDE = "requires_override"
    ESCALATED = "escalated"

class ResidencyRegion(str, Enum):
    EU = "EU"
    INDIA = "IN"
    US = "US"
    APAC = "APAC"
    GLOBAL = "GLOBAL"

class ComplianceFramework(str, Enum):
    SOX = "SOX"
    GDPR = "GDPR"
    DPDP = "DPDP"
    RBI = "RBI"
    IRDAI = "IRDAI"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI_DSS"

class PolicyEnforcementRequest(BaseModel):
    """Request for policy enforcement validation"""
    workflow_id: str
    tenant_id: int
    user_id: int
    approval_type: str  # 'workflow_approval', 'policy_exception', 'override_request'
    requested_action: str
    context_data: Dict[str, Any] = Field(default_factory=dict)
    compliance_frameworks: List[ComplianceFramework] = Field(default_factory=list)
    residency_region: ResidencyRegion = ResidencyRegion.GLOBAL
    risk_level: str = "medium"  # low, medium, high, critical

class PolicyEnforcementResponse(BaseModel):
    """Response from policy enforcement validation"""
    enforcement_id: str
    result: PolicyEnforcementResult
    policy_violations: List[Dict[str, Any]] = Field(default_factory=list)
    required_approvers: List[Dict[str, Any]] = Field(default_factory=list)
    residency_constraints: Dict[str, Any] = Field(default_factory=dict)
    override_options: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_requirements: List[str] = Field(default_factory=list)
    expiry_time: Optional[datetime] = None
    reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EnhancedPolicyEnforcementEngine:
    """
    Enhanced Policy Pack Enforcement Engine with residency awareness
    Tasks 14.1-T10, 14.1-T11: Policy enforcement and residency enforcement
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        
        # Policy rule definitions by framework
        self.policy_rules = {
            ComplianceFramework.SOX: {
                "segregation_of_duties": True,
                "financial_approval_threshold": 10000,  # USD
                "required_approvers": ["finance_manager", "cfo"],
                "evidence_required": ["financial_controls", "approval_chain"],
                "retention_years": 7
            },
            ComplianceFramework.GDPR: {
                "data_subject_consent": True,
                "processing_purpose_documented": True,
                "data_minimization": True,
                "residency_enforcement": True,
                "required_approvers": ["privacy_officer", "legal_director"],
                "retention_years": 2
            },
            ComplianceFramework.DPDP: {
                "data_fiduciary_consent": True,
                "cross_border_transfer_approval": True,
                "residency_enforcement": True,
                "required_approvers": ["privacy_officer", "compliance_manager"],
                "retention_years": 3
            },
            ComplianceFramework.RBI: {
                "regulatory_approval_required": True,
                "risk_assessment_mandatory": True,
                "audit_trail_complete": True,
                "required_approvers": ["compliance_manager", "risk_officer", "ceo"],
                "retention_years": 10
            },
            ComplianceFramework.IRDAI: {
                "solvency_check_required": True,
                "actuarial_approval": True,
                "regulatory_filing": True,
                "required_approvers": ["actuarial_head", "compliance_officer"],
                "retention_years": 5
            },
            ComplianceFramework.HIPAA: {
                "phi_access_justified": True,
                "minimum_necessary_standard": True,
                "audit_log_required": True,
                "required_approvers": ["privacy_officer", "security_officer"],
                "retention_years": 6
            },
            ComplianceFramework.PCI_DSS: {
                "cardholder_data_protection": True,
                "access_control_measures": True,
                "vulnerability_management": True,
                "required_approvers": ["security_manager", "compliance_officer"],
                "retention_years": 3
            }
        }
        
        # Residency-based approval routing
        self.residency_routing = {
            ResidencyRegion.EU: {
                "allowed_approvers": ["eu_compliance_officer", "eu_privacy_officer"],
                "data_residency_required": True,
                "cross_border_restrictions": ["US", "APAC"],
                "applicable_frameworks": [ComplianceFramework.GDPR]
            },
            ResidencyRegion.INDIA: {
                "allowed_approvers": ["in_compliance_officer", "in_risk_officer"],
                "data_residency_required": True,
                "cross_border_restrictions": [],
                "applicable_frameworks": [ComplianceFramework.DPDP, ComplianceFramework.RBI, ComplianceFramework.IRDAI]
            },
            ResidencyRegion.US: {
                "allowed_approvers": ["us_compliance_officer", "us_privacy_officer"],
                "data_residency_required": False,
                "cross_border_restrictions": [],
                "applicable_frameworks": [ComplianceFramework.SOX, ComplianceFramework.HIPAA, ComplianceFramework.PCI_DSS]
            }
        }
        
    async def enforce_policy(self, request: PolicyEnforcementRequest) -> PolicyEnforcementResponse:
        """
        Main policy enforcement entry point
        Task 14.1-T10: Policy pack enforcement in approvals
        """
        try:
            enforcement_id = str(uuid.uuid4())
            logger.info(f"🔒 Starting policy enforcement for workflow {request.workflow_id}")
            
            # Step 1: Validate tenant and user permissions
            tenant_valid = await self._validate_tenant_permissions(request.tenant_id, request.user_id)
            if not tenant_valid:
                return PolicyEnforcementResponse(
                    enforcement_id=enforcement_id,
                    result=PolicyEnforcementResult.DENIED,
                    reason="Tenant or user permissions invalid"
                )
            
            # Step 2: Enforce residency constraints
            residency_result = await self._enforce_residency_constraints(request)
            if residency_result["blocked"]:
                return PolicyEnforcementResponse(
                    enforcement_id=enforcement_id,
                    result=PolicyEnforcementResult.DENIED,
                    reason=f"Residency violation: {residency_result['reason']}",
                    residency_constraints=residency_result
                )
            
            # Step 3: Evaluate policy rules for each framework
            policy_violations = []
            required_approvers = []
            evidence_requirements = []
            
            for framework in request.compliance_frameworks:
                violations, approvers, evidence = await self._evaluate_framework_policies(
                    framework, request
                )
                policy_violations.extend(violations)
                required_approvers.extend(approvers)
                evidence_requirements.extend(evidence)
            
            # Step 4: Determine enforcement result
            if policy_violations:
                # Check if violations can be overridden
                override_options = await self._get_override_options(policy_violations, request)
                
                if override_options:
                    result = PolicyEnforcementResult.REQUIRES_OVERRIDE
                else:
                    result = PolicyEnforcementResult.DENIED
            else:
                result = PolicyEnforcementResult.APPROVED
            
            # Step 5: Set expiry time based on risk level
            expiry_time = self._calculate_expiry_time(request.risk_level)
            
            # Step 6: Log enforcement decision
            await self._log_enforcement_decision(enforcement_id, request, result, policy_violations)
            
            response = PolicyEnforcementResponse(
                enforcement_id=enforcement_id,
                result=result,
                policy_violations=policy_violations,
                required_approvers=required_approvers,
                residency_constraints=residency_result,
                override_options=override_options if 'override_options' in locals() else [],
                evidence_requirements=evidence_requirements,
                expiry_time=expiry_time,
                reason=f"Policy enforcement completed with {len(policy_violations)} violations",
                metadata={
                    "frameworks_evaluated": [f.value for f in request.compliance_frameworks],
                    "residency_region": request.residency_region.value,
                    "risk_level": request.risk_level
                }
            )
            
            logger.info(f"✅ Policy enforcement completed: {result.value} for workflow {request.workflow_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Policy enforcement failed: {e}")
            return PolicyEnforcementResponse(
                enforcement_id=str(uuid.uuid4()),
                result=PolicyEnforcementResult.DENIED,
                reason=f"Policy enforcement error: {str(e)}"
            )
    
    async def _validate_tenant_permissions(self, tenant_id: int, user_id: int) -> bool:
        """Validate tenant and user have required permissions"""
        try:
            async with self.pool_manager.get_connection() as conn:
                # Check tenant exists and is active
                tenant_row = await conn.fetchrow("""
                    SELECT tenant_id, status FROM tenants 
                    WHERE tenant_id = $1 AND status = 'active'
                """, tenant_id)
                
                if not tenant_row:
                    return False
                
                # Check user belongs to tenant and has approval permissions
                user_row = await conn.fetchrow("""
                    SELECT u.user_id, ur.permissions 
                    FROM users u
                    JOIN users_role ur ON u.user_id = ur.user_id
                    WHERE u.user_id = $1 AND u.tenant_id = $2
                    AND ur.permissions ? 'crux_approve'
                """, user_id, tenant_id)
                
                return user_row is not None
                
        except Exception as e:
            logger.error(f"❌ Tenant permission validation failed: {e}")
            return False
    
    async def _enforce_residency_constraints(self, request: PolicyEnforcementRequest) -> Dict[str, Any]:
        """
        Enforce residency-based approval routing
        Task 14.1-T11: Residency enforcement in approvals
        """
        try:
            residency_rules = self.residency_routing.get(request.residency_region, {})
            
            # Check if data residency is required
            if residency_rules.get("data_residency_required", False):
                # Validate that approvers are in the same region
                allowed_approvers = residency_rules.get("allowed_approvers", [])
                
                # Check cross-border restrictions
                restricted_regions = residency_rules.get("cross_border_restrictions", [])
                
                # Get user's actual region from database
                user_region = await self._get_user_region(request.user_id)
                
                if user_region != request.residency_region.value and user_region in restricted_regions:
                    return {
                        "blocked": True,
                        "reason": f"Cross-border approval not allowed: {user_region} -> {request.residency_region.value}",
                        "allowed_approvers": allowed_approvers,
                        "user_region": user_region,
                        "target_region": request.residency_region.value
                    }
            
            return {
                "blocked": False,
                "allowed_approvers": residency_rules.get("allowed_approvers", []),
                "applicable_frameworks": [f.value for f in residency_rules.get("applicable_frameworks", [])],
                "data_residency_required": residency_rules.get("data_residency_required", False)
            }
            
        except Exception as e:
            logger.error(f"❌ Residency enforcement failed: {e}")
            return {
                "blocked": True,
                "reason": f"Residency validation error: {str(e)}"
            }
    
    async def _get_user_region(self, user_id: int) -> str:
        """Get user's region from database"""
        try:
            async with self.pool_manager.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT ur.region FROM users u
                    JOIN users_role ur ON u.user_id = ur.user_id
                    WHERE u.user_id = $1
                """, user_id)
                
                return row['region'] if row else "GLOBAL"
                
        except Exception as e:
            logger.error(f"❌ Failed to get user region: {e}")
            return "GLOBAL"
    
    async def _evaluate_framework_policies(self, framework: ComplianceFramework, 
                                         request: PolicyEnforcementRequest) -> Tuple[List[Dict], List[Dict], List[str]]:
        """Evaluate policies for a specific compliance framework"""
        violations = []
        required_approvers = []
        evidence_requirements = []
        
        try:
            rules = self.policy_rules.get(framework, {})
            
            # Check framework-specific rules
            if framework == ComplianceFramework.SOX:
                violations.extend(await self._check_sox_compliance(request, rules))
            elif framework == ComplianceFramework.GDPR:
                violations.extend(await self._check_gdpr_compliance(request, rules))
            elif framework == ComplianceFramework.RBI:
                violations.extend(await self._check_rbi_compliance(request, rules))
            elif framework == ComplianceFramework.IRDAI:
                violations.extend(await self._check_irdai_compliance(request, rules))
            elif framework == ComplianceFramework.HIPAA:
                violations.extend(await self._check_hipaa_compliance(request, rules))
            elif framework == ComplianceFramework.PCI_DSS:
                violations.extend(await self._check_pci_compliance(request, rules))
            
            # Add required approvers from rules
            for approver_role in rules.get("required_approvers", []):
                required_approvers.append({
                    "role": approver_role,
                    "framework": framework.value,
                    "required": True
                })
            
            # Add evidence requirements
            evidence_requirements.extend(rules.get("evidence_required", []))
            
        except Exception as e:
            logger.error(f"❌ Framework policy evaluation failed for {framework.value}: {e}")
            violations.append({
                "framework": framework.value,
                "violation_type": "evaluation_error",
                "message": f"Policy evaluation failed: {str(e)}",
                "severity": "high"
            })
        
        return violations, required_approvers, evidence_requirements
    
    async def _check_sox_compliance(self, request: PolicyEnforcementRequest, rules: Dict) -> List[Dict]:
        """Check SOX compliance rules"""
        violations = []
        
        # Check segregation of duties
        if rules.get("segregation_of_duties", False):
            if await self._check_sod_violation(request.user_id, request.workflow_id):
                violations.append({
                    "framework": "SOX",
                    "violation_type": "segregation_of_duties",
                    "message": "User cannot approve their own workflow (SoD violation)",
                    "severity": "high"
                })
        
        # Check financial threshold
        financial_amount = request.context_data.get("financial_amount", 0)
        threshold = rules.get("financial_approval_threshold", 10000)
        
        if financial_amount > threshold:
            violations.append({
                "framework": "SOX",
                "violation_type": "financial_threshold",
                "message": f"Amount ${financial_amount} exceeds threshold ${threshold}",
                "severity": "medium",
                "requires_additional_approval": True
            })
        
        return violations
    
    async def _check_gdpr_compliance(self, request: PolicyEnforcementRequest, rules: Dict) -> List[Dict]:
        """Check GDPR compliance rules"""
        violations = []
        
        # Check data subject consent
        if rules.get("data_subject_consent", False):
            if not request.context_data.get("data_subject_consent", False):
                violations.append({
                    "framework": "GDPR",
                    "violation_type": "missing_consent",
                    "message": "Data subject consent not documented",
                    "severity": "high"
                })
        
        # Check processing purpose
        if rules.get("processing_purpose_documented", False):
            if not request.context_data.get("processing_purpose"):
                violations.append({
                    "framework": "GDPR",
                    "violation_type": "missing_purpose",
                    "message": "Data processing purpose not documented",
                    "severity": "medium"
                })
        
        return violations
    
    async def _check_rbi_compliance(self, request: PolicyEnforcementRequest, rules: Dict) -> List[Dict]:
        """Check RBI compliance rules"""
        violations = []
        
        # Check regulatory approval
        if rules.get("regulatory_approval_required", False):
            if not request.context_data.get("regulatory_approval"):
                violations.append({
                    "framework": "RBI",
                    "violation_type": "missing_regulatory_approval",
                    "message": "RBI regulatory approval not obtained",
                    "severity": "critical"
                })
        
        # Check risk assessment
        if rules.get("risk_assessment_mandatory", False):
            if not request.context_data.get("risk_assessment_completed"):
                violations.append({
                    "framework": "RBI",
                    "violation_type": "missing_risk_assessment",
                    "message": "Risk assessment not completed",
                    "severity": "high"
                })
        
        return violations
    
    async def _check_irdai_compliance(self, request: PolicyEnforcementRequest, rules: Dict) -> List[Dict]:
        """Check IRDAI compliance rules"""
        violations = []
        
        # Check solvency requirements
        if rules.get("solvency_check_required", False):
            solvency_ratio = request.context_data.get("solvency_ratio", 0)
            if solvency_ratio < 1.5:  # IRDAI minimum
                violations.append({
                    "framework": "IRDAI",
                    "violation_type": "solvency_breach",
                    "message": f"Solvency ratio {solvency_ratio} below IRDAI minimum 1.5",
                    "severity": "critical"
                })
        
        return violations
    
    async def _check_hipaa_compliance(self, request: PolicyEnforcementRequest, rules: Dict) -> List[Dict]:
        """Check HIPAA compliance rules"""
        violations = []
        
        # Check PHI access justification
        if rules.get("phi_access_justified", False):
            if not request.context_data.get("phi_access_justification"):
                violations.append({
                    "framework": "HIPAA",
                    "violation_type": "phi_access_unjustified",
                    "message": "PHI access not properly justified",
                    "severity": "high"
                })
        
        return violations
    
    async def _check_pci_compliance(self, request: PolicyEnforcementRequest, rules: Dict) -> List[Dict]:
        """Check PCI DSS compliance rules"""
        violations = []
        
        # Check cardholder data protection
        if rules.get("cardholder_data_protection", False):
            if not request.context_data.get("cardholder_data_encrypted", False):
                violations.append({
                    "framework": "PCI_DSS",
                    "violation_type": "cardholder_data_unprotected",
                    "message": "Cardholder data not properly encrypted",
                    "severity": "critical"
                })
        
        return violations
    
    async def _check_sod_violation(self, user_id: int, workflow_id: str) -> bool:
        """Check if user is trying to approve their own workflow"""
        try:
            async with self.pool_manager.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT created_by_user_id FROM dsl_workflows 
                    WHERE workflow_id = $1
                """, workflow_id)
                
                return row and row['created_by_user_id'] == user_id
                
        except Exception as e:
            logger.error(f"❌ SoD check failed: {e}")
            return True  # Fail closed
    
    async def _get_override_options(self, violations: List[Dict], request: PolicyEnforcementRequest) -> List[Dict]:
        """Get available override options for policy violations"""
        override_options = []
        
        for violation in violations:
            if violation.get("severity") in ["medium", "low"]:
                override_options.append({
                    "violation_id": violation.get("violation_type"),
                    "framework": violation.get("framework"),
                    "override_type": "policy_exception",
                    "required_approvers": ["compliance_manager", "legal_director"],
                    "reason_required": True,
                    "evidence_required": True
                })
            elif violation.get("severity") == "high" and request.risk_level == "low":
                override_options.append({
                    "violation_id": violation.get("violation_type"),
                    "framework": violation.get("framework"),
                    "override_type": "emergency_override",
                    "required_approvers": ["ceo", "compliance_director"],
                    "reason_required": True,
                    "evidence_required": True,
                    "time_limited": True,
                    "max_duration_hours": 24
                })
        
        return override_options
    
    def _calculate_expiry_time(self, risk_level: str) -> datetime:
        """Calculate when the enforcement decision expires"""
        hours_map = {
            "low": 72,      # 3 days
            "medium": 48,   # 2 days
            "high": 24,     # 1 day
            "critical": 4   # 4 hours
        }
        
        hours = hours_map.get(risk_level, 24)
        return datetime.utcnow() + timedelta(hours=hours)
    
    async def _log_enforcement_decision(self, enforcement_id: str, request: PolicyEnforcementRequest, 
                                      result: PolicyEnforcementResult, violations: List[Dict]):
        """Log the enforcement decision for audit trail"""
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO policy_enforcement_log (
                        enforcement_id, workflow_id, tenant_id, user_id,
                        approval_type, result, violations, residency_region,
                        compliance_frameworks, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, 
                enforcement_id, request.workflow_id, request.tenant_id, request.user_id,
                request.approval_type, result.value, json.dumps(violations), 
                request.residency_region.value, 
                [f.value for f in request.compliance_frameworks],
                datetime.utcnow())
                
        except Exception as e:
            logger.error(f"❌ Failed to log enforcement decision: {e}")

# Policy Enforcement Service Factory
class PolicyEnforcementService:
    """Service factory for policy enforcement"""
    
    def __init__(self, pool_manager):
        self.engine = EnhancedPolicyEnforcementEngine(pool_manager)
    
    async def enforce_approval_policy(self, request: PolicyEnforcementRequest) -> PolicyEnforcementResponse:
        """Enforce policy for approval requests"""
        return await self.engine.enforce_policy(request)
    
    async def validate_residency_compliance(self, tenant_id: int, user_id: int, 
                                          target_region: ResidencyRegion) -> Dict[str, Any]:
        """Validate residency compliance for cross-border operations"""
        request = PolicyEnforcementRequest(
            workflow_id="residency_check",
            tenant_id=tenant_id,
            user_id=user_id,
            approval_type="residency_validation",
            requested_action="cross_border_operation",
            residency_region=target_region
        )
        
        return await self.engine._enforce_residency_constraints(request)
    
    async def get_framework_requirements(self, frameworks: List[ComplianceFramework]) -> Dict[str, Any]:
        """Get requirements for specific compliance frameworks"""
        requirements = {}
        
        for framework in frameworks:
            rules = self.engine.policy_rules.get(framework, {})
            requirements[framework.value] = {
                "required_approvers": rules.get("required_approvers", []),
                "evidence_required": rules.get("evidence_required", []),
                "retention_years": rules.get("retention_years", 7)
            }
        
        return requirements
